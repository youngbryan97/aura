"""Verified Azure Digital Twins scalar bridge for Reality Reach.

Azure Digital Twins is a remote representation, not independent evidence that a
physical device changed.  Writable resources therefore require separate desired
and device-reported JSON Pointer paths.  A PATCH is transport completion only;
the shared scalar adapter must subsequently observe the reported path before it
can claim an effect.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from core.governance_context import (
    get_active_governance,
    governance_runtime_active,
    governed_scope,
    local_internal_decision,
    require_governance,
)
from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import LiveChannelAdapter
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarWriteResult,
)
from core.runtime.action_executor import ActionExecutor
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_API_VERSION = "2023-10-31"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_GUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
_MODEL_ID = re.compile(r"^dtmi:[A-Za-z0-9_:.-]+;[1-9][0-9]*$")
_AZURE_HOST = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.api\.[a-z0-9-]+\.digitaltwins\.azure\.net$"
)
_MAX_JSON_BYTES = 512 * 1024
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_IDEMPOTENCY_RECEIPTS = 4096
_CONTROL_DOMAINS = ("environment_action", "external_action", "tool_execution")


class AzureDigitalTwinsConnectorError(RuntimeError):
    """An Azure twin identity, payload, credential, or effect contract failed."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _strict_json(payload: bytes, *, role: str) -> Any:
    if len(payload) > _MAX_JSON_BYTES:
        raise AzureDigitalTwinsConnectorError(f"azure_dt_{role}_too_large")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AzureDigitalTwinsConnectorError(f"azure_dt_{role}_duplicate_json_key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )
    except AzureDigitalTwinsConnectorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AzureDigitalTwinsConnectorError(f"azure_dt_{role}_invalid_json") from exc


def _json_pointer(value: object, *, name: str) -> tuple[str, tuple[str, ...]]:
    pointer = str(value or "").strip()
    if (
        not pointer.startswith("/")
        or len(pointer.encode("utf-8")) > 1024
        or any(character in pointer for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{name} must be a bounded absolute JSON Pointer")
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded: list[str] = []
        while index < len(raw):
            if raw[index] != "~":
                decoded.append(raw[index])
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise ValueError(f"{name} contains an invalid JSON Pointer escape")
            decoded.append("~" if raw[index + 1] == "0" else "/")
            index += 2
        token = "".join(decoded)
        if not token or token.startswith("$"):
            raise ValueError(f"{name} must address a user property")
        tokens.append(token)
    return pointer, tuple(tokens)


def _pointer_value(document: Mapping[str, Any], tokens: tuple[str, ...]) -> Any:
    current: Any = document
    for token in tokens:
        if not isinstance(current, Mapping) or token not in current:
            raise AzureDigitalTwinsConnectorError("azure_dt_reported_property_missing")
        current = current[token]
    return current


def _header(headers: object, name: str) -> str:
    if not isinstance(headers, Mapping):
        return ""
    expected = name.lower()
    for key, value in headers.items():
        if str(key).lower() == expected:
            return str(value or "").strip()
    return ""


def _captured_at_ns(document: Mapping[str, Any], tokens: tuple[str, ...]) -> tuple[int, str]:
    metadata = document.get("$metadata")
    current: Any = metadata
    for token in tokens:
        if not isinstance(current, Mapping):
            current = None
            break
        current = current.get(token)
    timestamp = current.get("lastUpdateTime") if isinstance(current, Mapping) else None
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            captured = int(parsed.timestamp() * 1_000_000_000)
            if captured > 0:
                return captured, "azure_digital_twins.metadata.lastUpdateTime"
        # not a failure: a value that is not a number is not one this can read.
        except (OverflowError, ValueError):
            pass
    return max(1, time.time_ns()), "system.time_ns.response_receipt"


@dataclass(frozen=True, slots=True)
class AzureDigitalTwinResourceSpec:
    resource_id: str
    twin_id: str
    observable: str
    unit: str
    reported_path: str
    expected_model_id: str
    domain: NumericDomain
    resolution: float
    desired_path: str = ""
    patch_operation: str = "replace"
    safe_value: float | None = None
    tolerance: float | None = None
    uncertainty: float | None = None
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("resource_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        twin_id = str(self.twin_id or "").strip()
        if (
            not twin_id
            or len(twin_id.encode("utf-8")) > 128
            or any(character in twin_id for character in ("\x00", "\n", "\r", "/"))
        ):
            raise ValueError("twin_id must be a bounded Azure twin identity")
        object.__setattr__(self, "twin_id", twin_id)
        model_id = str(self.expected_model_id or "").strip()
        if not _MODEL_ID.fullmatch(model_id):
            raise ValueError("expected_model_id must be a versioned DTDL model id")
        object.__setattr__(self, "expected_model_id", model_id)
        reported, reported_tokens = _json_pointer(
            self.reported_path,
            name="reported_path",
        )
        object.__setattr__(self, "reported_path", reported)
        desired = str(self.desired_path or "").strip()
        if desired:
            desired, desired_tokens = _json_pointer(desired, name="desired_path")
            if desired_tokens == reported_tokens:
                raise ValueError("writable cloud twins require distinct desired and reported paths")
        object.__setattr__(self, "desired_path", desired)
        operation = str(self.patch_operation or "replace").strip().lower()
        if operation not in {"add", "replace"}:
            raise ValueError("patch_operation must be add or replace")
        object.__setattr__(self, "patch_operation", operation)
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be NumericDomain")
        resolution = _finite(self.resolution, name="resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        object.__setattr__(self, "resolution", resolution)
        tolerance = (
            resolution
            if self.tolerance is None
            else _finite(
                self.tolerance,
                name="tolerance",
            )
        )
        if tolerance < resolution:
            raise ValueError("tolerance must not be smaller than resolution")
        object.__setattr__(self, "tolerance", tolerance)
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, name="uncertainty")
            if uncertainty < 0.0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)
        if self.safe_value is not None:
            safe = _finite(self.safe_value, name="safe_value")
            if not desired or not self.domain.contains(safe):
                raise ValueError("safe_value requires a writable in-domain resource")
            object.__setattr__(self, "safe_value", safe)
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("Azure twin timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def writable(self) -> bool:
        return bool(self.desired_path)

    @property
    def reported_tokens(self) -> tuple[str, ...]:
        return _json_pointer(self.reported_path, name="reported_path")[1]

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "twin_id": self.twin_id,
            "observable": self.observable,
            "unit": self.unit,
            "reported_path": self.reported_path,
            "desired_path": self.desired_path,
            "patch_operation": self.patch_operation,
            "expected_model_id": self.expected_model_id,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "safe_value": self.safe_value,
            "tolerance": self.tolerance,
            "uncertainty": self.uncertainty,
            "max_commands_per_minute": self.max_commands_per_minute,
            "cooldown_s": self.cooldown_s,
            "stale_after_s": self.stale_after_s,
        }

    def decode(self, document: Mapping[str, Any]) -> float:
        value = _pointer_value(document, self.reported_tokens)
        number = (
            (1.0 if value else 0.0)
            if isinstance(value, bool)
            else _finite(
                value,
                name="Azure twin reported property",
            )
        )
        if not self.domain.contains(number):
            raise AzureDigitalTwinsConnectorError(
                "azure_dt_reported_property_outside_manifest_domain"
            )
        return number


def parse_azure_digital_twin_manifest(
    raw: object,
) -> tuple[AzureDigitalTwinResourceSpec, ...]:
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
            raise AzureDigitalTwinsConnectorError("azure_dt_manifest_too_large")
        raw = _strict_json(raw.encode("utf-8"), role="manifest")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AzureDigitalTwinsConnectorError("azure_dt_manifest_must_be_a_list")
    if not 1 <= len(raw) <= 512:
        raise AzureDigitalTwinsConnectorError("azure_dt_manifest_size_invalid")
    resources: list[AzureDigitalTwinResourceSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise AzureDigitalTwinsConnectorError("azure_dt_manifest_entry_invalid")
        resources.append(
            AzureDigitalTwinResourceSpec(
                resource_id=str(item.get("resource_id") or ""),
                twin_id=str(item.get("twin_id") or ""),
                observable=str(item.get("observable") or ""),
                unit=str(item.get("unit") or ""),
                reported_path=str(item.get("reported_path") or ""),
                desired_path=str(item.get("desired_path") or ""),
                patch_operation=str(item.get("patch_operation") or "replace"),
                expected_model_id=str(item.get("expected_model_id") or ""),
                domain=NumericDomain(
                    _finite(item.get("minimum"), name="minimum"),
                    _finite(item.get("maximum"), name="maximum"),
                ),
                resolution=_finite(item.get("resolution"), name="resolution"),
                safe_value=(
                    None
                    if item.get("safe_value") is None
                    else _finite(item.get("safe_value"), name="safe_value")
                ),
                tolerance=(
                    None
                    if item.get("tolerance") is None
                    else _finite(item.get("tolerance"), name="tolerance")
                ),
                uncertainty=(
                    None
                    if item.get("uncertainty") is None
                    else _finite(item.get("uncertainty"), name="uncertainty")
                ),
                max_commands_per_minute=int(item.get("max_commands_per_minute") or 12),
                cooldown_s=_finite(item.get("cooldown_s") or 0.0, name="cooldown_s"),
                stale_after_s=_finite(item.get("stale_after_s") or 60.0, name="stale_after_s"),
            )
        )
    keys = [(item.twin_id, item.reported_path) for item in resources]
    if len(set(keys)) != len(keys):
        raise AzureDigitalTwinsConnectorError("azure_dt_reported_property_duplicate")
    if len({item.resource_id for item in resources}) != len(resources):
        raise AzureDigitalTwinsConnectorError("azure_dt_resource_id_duplicate")
    return tuple(sorted(resources, key=lambda item: item.resource_id))


@runtime_checkable
class AzureAccessTokenProvider(Protocol):
    @property
    def identity_sha256(self) -> str: ...

    async def access_token(self) -> str: ...


class EntraClientCredentialsTokenProvider:
    """Refresh an Entra OAuth token without exposing credentials in receipts."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tenant_id = str(os.getenv("AURA_AZURE_DT_TENANT_ID") or "").strip()
        self._client_id = str(os.getenv("AURA_AZURE_DT_CLIENT_ID") or "").strip()
        if not _GUID.fullmatch(self._tenant_id) or not _GUID.fullmatch(self._client_id):
            raise AzureDigitalTwinsConnectorError("azure_dt_tenant_or_client_id_invalid")
        self._read_client_secret()
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._monotonic_clock = monotonic_clock
        self._token = ""
        self._expires_monotonic = 0.0
        self._lock = checked_async_lock("azure_dt.entra_token")

    @staticmethod
    def _read_client_secret() -> str:
        secret = str(os.getenv("AURA_AZURE_DT_CLIENT_SECRET") or "").strip()
        if not secret or len(secret.encode("utf-8")) > 4096:
            raise AzureDigitalTwinsConnectorError("azure_dt_client_secret_invalid")
        return secret

    @property
    def identity_sha256(self) -> str:
        return _digest({"tenant_id": self._tenant_id, "client_id": self._client_id})

    async def access_token(self) -> str:
        if self._token and self._monotonic_clock() < self._expires_monotonic:
            return self._token
        async with self._lock:
            if self._token and self._monotonic_clock() < self._expires_monotonic:
                return self._token
            client_secret = self._read_client_secret()
            body = urllib.parse.urlencode(
                {
                    "client_id": self._client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                    "scope": "https://digitaltwins.azure.net/.default",
                }
            )
            response = await ActionExecutor.request_network_transport(
                method="POST",
                url=(
                    "https://login.microsoftonline.com/"
                    f"{urllib.parse.quote(self._tenant_id, safe='')}/oauth2/v2.0/token"
                ),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=body,
                timeout_s=10.0,
                source="world_bridge:azure_digital_twins.oauth",
                read_only=False,
            )
            if not response.get("ok") or int(response.get("status_code") or 0) != 200:
                raise AzureDigitalTwinsConnectorError("azure_dt_token_exchange_failed")
            document = _strict_json(bytes(response.get("content") or b""), role="token")
            token = document.get("access_token") if isinstance(document, Mapping) else None
            expires = document.get("expires_in") if isinstance(document, Mapping) else None
            if not isinstance(token, str) or not token or len(token.encode("utf-8")) > 32_768:
                raise AzureDigitalTwinsConnectorError("azure_dt_access_token_invalid")
            lifetime = _finite(expires, name="expires_in")
            if not 120.0 <= lifetime <= 86_400.0:
                raise AzureDigitalTwinsConnectorError("azure_dt_access_token_lifetime_invalid")
            self._token = token
            self._expires_monotonic = self._monotonic_clock() + lifetime - 60.0
            return token


class AzureDigitalTwinsScalarTransport:
    """ETag-fenced Azure Digital Twins data-plane scalar transport."""

    transport_id = "azure.digital_twins.rest"

    def __init__(
        self,
        resources: tuple[AzureDigitalTwinResourceSpec, ...],
        token_provider: AzureAccessTokenProvider,
    ) -> None:
        if not resources or not isinstance(token_provider, AzureAccessTokenProvider):
            raise TypeError("resources and token_provider must satisfy Azure twin contracts")
        endpoint = str(os.getenv("AURA_AZURE_DT_ENDPOINT") or "").strip().rstrip("/")
        instance_id = _identifier(
            os.getenv("AURA_AZURE_DT_INSTANCE_ID"),
            name="AURA_AZURE_DT_INSTANCE_ID",
        )
        parsed = urllib.parse.urlparse(endpoint)
        host = str(parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not _AZURE_HOST.fullmatch(host)
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise AzureDigitalTwinsConnectorError("azure_dt_endpoint_invalid")
        self._resources = {item.resource_id: item for item in resources}
        self._endpoint = endpoint
        self._token_provider = token_provider
        self._identity = _digest(
            {
                "instance_id": instance_id,
                "endpoint_host": host,
                "credential_identity_sha256": token_provider.identity_sha256,
            }
        )
        self._etags: dict[str, str] = {}
        self._sequences: dict[str, int] = {}
        self._idempotency: dict[str, tuple[str, float, ScalarWriteResult]] = {}
        self._lock = checked_async_lock("azure_dt.transport")

    @property
    def instance_identity_sha256(self) -> str:
        return self._identity

    @property
    def identity_stable(self) -> bool:
        return True

    def _url(self, twin_id: str) -> str:
        encoded = urllib.parse.quote(twin_id, safe="")
        return f"{self._endpoint}/digitaltwins/{encoded}?api-version={_API_VERSION}"

    async def _request_governed(
        self,
        *,
        method: str,
        twin_id: str,
        headers: dict[str, str],
        data: bytes | None,
        read_only: bool,
    ) -> dict[str, Any]:
        token = await self._token_provider.access_token()
        response = await ActionExecutor.request_network_transport(
            method=method,
            url=self._url(twin_id),
            headers={"Authorization": f"Bearer {token}", **headers},
            data=data,
            timeout_s=10.0,
            source="world_bridge:azure_digital_twins.data_plane",
            read_only=read_only,
        )
        if not isinstance(response, Mapping):
            raise AzureDigitalTwinsConnectorError("azure_dt_network_gateway_returned_non_mapping")
        return dict(response)

    async def _read_governed(self, spec: AzureDigitalTwinResourceSpec) -> ScalarSample:
        response = await self._request_governed(
            method="GET",
            twin_id=spec.twin_id,
            headers={"Accept": "application/json"},
            data=None,
            read_only=True,
        )
        if not response.get("ok") or int(response.get("status_code") or 0) != 200:
            raise AzureDigitalTwinsConnectorError("azure_dt_twin_read_failed")
        document = _strict_json(bytes(response.get("content") or b""), role="twin")
        if not isinstance(document, Mapping):
            raise AzureDigitalTwinsConnectorError("azure_dt_twin_not_an_object")
        metadata = document.get("$metadata")
        model_id = metadata.get("$model") if isinstance(metadata, Mapping) else None
        if document.get("$dtId") != spec.twin_id or model_id != spec.expected_model_id:
            raise AzureDigitalTwinsConnectorError("azure_dt_twin_identity_mismatch")
        etag = _header(response.get("headers"), "ETag")
        if not etag or len(etag.encode("utf-8")) > 256:
            raise AzureDigitalTwinsConnectorError("azure_dt_etag_missing")
        value = spec.decode(document)
        captured_at_ns, clock_source = _captured_at_ns(document, spec.reported_tokens)
        sequence = self._sequences.get(spec.resource_id, 0) + 1
        self._sequences[spec.resource_id] = sequence
        self._etags[spec.twin_id] = etag
        return ScalarSample(
            value=value,
            captured_at_ns=captured_at_ns,
            source_event_id=_digest(
                {
                    "instance": self._identity,
                    "twin_id": spec.twin_id,
                    "model_id": model_id,
                    "reported_path": spec.reported_path,
                    "captured_at_ns": captured_at_ns,
                    "value": value,
                }
            ),
            quality="device_reported_cloud_twin",
            uncertainty=spec.uncertainty,
            wall_clock_source=clock_source,
            source_epoch=self._identity,
            source_sequence=sequence,
        )

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("azure_dt_resource_not_bound")
        if get_active_governance() is None and governance_runtime_active():
            decision = local_internal_decision(
                "azure_digital_twins.device_reported_readback",
                domain="environment_action",
                constraints={
                    "read_only": True,
                    "resource_sha256": _digest(resource_id),
                    "twin_sha256": _digest(spec.twin_id),
                },
            )
            async with governed_scope(decision):
                return await self._read_governed(spec)
        return await self._read_governed(spec)

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        spec = self._resources.get(resource_id)
        if spec is None or not spec.writable:
            raise PermissionError("azure_dt_resource_not_writable")
        target = _finite(value, name="Azure twin desired property")
        if not spec.domain.contains(target):
            raise AzureDigitalTwinsConnectorError(
                "azure_dt_desired_property_outside_manifest_domain"
            )
        key = str(idempotency_key or "").strip()
        if not key or len(key.encode("utf-8")) > 256:
            raise ValueError("azure_dt_idempotency_key_invalid")
        require_governance(
            f"azure_digital_twins.write_scalar:{resource_id}",
            strict=True,
            allowed_domains=_CONTROL_DOMAINS,
        )
        async with self._lock:
            previous = self._idempotency.get(key)
            if previous is not None:
                old_resource, old_value, result = previous
                if old_resource != resource_id or old_value != target:
                    raise AzureDigitalTwinsConnectorError("azure_dt_idempotency_key_conflict")
                return result
            etag = self._etags.get(spec.twin_id)
            if not etag:
                raise AzureDigitalTwinsConnectorError("azure_dt_write_requires_fresh_etag_readback")
            patch = canonical_json(
                [{"op": spec.patch_operation, "path": spec.desired_path, "value": target}]
            )
            response = await self._request_governed(
                method="PATCH",
                twin_id=spec.twin_id,
                headers={
                    "Content-Type": "application/json-patch+json",
                    "If-Match": etag,
                },
                data=patch,
                read_only=False,
            )
            status = int(response.get("status_code") or 0)
            self._etags.pop(spec.twin_id, None)
            if not response.get("ok") or status != 204:
                if status == 412:
                    raise AzureDigitalTwinsConnectorError("azure_dt_etag_precondition_failed")
                if status == 0 or status >= 500:
                    raise AzureDigitalTwinsConnectorError(
                        "azure_dt_desired_property_effect_indeterminate"
                    )
                raise AzureDigitalTwinsConnectorError("azure_dt_desired_property_rejected")
            result = ScalarWriteResult(
                accepted=True,
                transport_completed=True,
                receipt={
                    "protocol": self.transport_id,
                    "resource_id": resource_id,
                    "instance_identity_sha256": self._identity,
                    "twin_id_sha256": _digest(spec.twin_id),
                    "desired_path_sha256": _digest(spec.desired_path),
                    "target_sha256": _digest(target),
                    "precondition_etag_sha256": _digest(etag),
                    "idempotency_sha256": _digest(key),
                    "recovery": recovery,
                    "transport_status": status,
                    "effect_verified": False,
                },
            )
            if len(self._idempotency) >= _MAX_IDEMPOTENCY_RECEIPTS:
                self._idempotency.pop(next(iter(self._idempotency)))
            self._idempotency[key] = (resource_id, target, result)
            return result


class AzureDigitalTwinsConnector:
    """Discover and attach declared Azure twin properties."""

    connector_id = "azure.digital_twins"

    def __init__(
        self,
        transport: AzureDigitalTwinsScalarTransport,
        resources: tuple[AzureDigitalTwinResourceSpec, ...],
        *,
        candidate_ttl_s: float = 180.0,
        discovery_budget_s: float = 30.0,
    ) -> None:
        if not resources:
            raise ValueError("resources must not be empty")
        self._transport = transport
        self._resources = {item.resource_id: item for item in resources}
        self._ttl_s = max(30.0, min(_finite(candidate_ttl_s, name="candidate_ttl_s"), 3600.0))
        budget = _finite(discovery_budget_s, name="discovery_budget_s")
        if not 0.01 <= budget <= 300.0:
            raise ValueError("discovery_budget_s must lie inside [0.01, 300]")
        self._discovery_budget_s = budget

    def _profile(self, spec: AzureDigitalTwinResourceSpec) -> ScalarResourceProfile:
        return ScalarResourceProfile(
            resource_id=spec.resource_id,
            observable=spec.observable,
            unit=spec.unit,
            domain=spec.domain,
            resolution=spec.resolution,
            writable=spec.writable,
            physical_identity_sha256=_digest(
                {
                    "instance": self._transport.instance_identity_sha256,
                    "twin_id": spec.twin_id,
                    "model_id": spec.expected_model_id,
                    "reported_path": spec.reported_path,
                    "desired_path": spec.desired_path,
                }
            ),
            owner="core.embodiment.azure_digital_twins_connector",
            protocol="azure_digital_twins",
            safe_value=spec.safe_value,
            tolerance=spec.tolerance,
            max_commands_per_minute=spec.max_commands_per_minute,
            cooldown_s=spec.cooldown_s,
            stale_after_s=spec.stale_after_s,
            readback_distinct_from_command=spec.writable,
        )

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        candidates: list[DeviceCandidate] = []
        now_ns = max(1, time.time_ns())
        deadline = time.monotonic() + self._discovery_budget_s
        for spec in self._resources.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                async with asyncio.timeout(remaining):
                    sample = await self._transport.read_scalar(spec.resource_id)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                continue
            profile = self._profile(spec)
            if not profile.domain.contains(sample.value):
                continue
            manifest = _digest(
                {
                    "spec_sha256": spec.sha256,
                    "profile_sha256": profile.sha256,
                    "instance_identity_sha256": self._transport.instance_identity_sha256,
                }
            )
            access = (
                (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                if profile.writable
                else (AttachmentAccess.OBSERVE,)
            )
            candidates.append(
                DeviceCandidate(
                    candidate_id="azure.dt.candidate." + manifest.removeprefix("sha256:")[:32],
                    connector_id=self.connector_id,
                    device_id=f"azure.dt.{spec.resource_id}",
                    display_name=f"Azure twin {spec.twin_id}: {spec.observable}"[:160],
                    transport=self._transport.transport_id,
                    identity_fingerprint=profile.physical_identity_sha256,
                    manifest_sha256=manifest,
                    access=access,
                    discovered_at_ns=now_ns,
                    expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                    persistent_identity=True,
                    proposal_salience=0.3,
                    metadata={
                        "resource_id": spec.resource_id,
                        "twin_id_sha256": _digest(spec.twin_id),
                        "model_id_sha256": _digest(spec.expected_model_id),
                        "spec_sha256": spec.sha256,
                        "profile_sha256": profile.sha256,
                        "control_available": profile.writable,
                        "independent_device_reported_readback": spec.writable,
                    },
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("azure_dt_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("azure_dt_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("azure_dt_control_requires_observation")
        resource_id = str(candidate.metadata.get("resource_id") or "")
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("azure_dt_candidate_resource_missing")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("azure_dt_candidate_changed_before_attachment")
        profile = self._profile(spec)
        if AttachmentAccess.CONTROL not in requested:
            profile = replace(profile, writable=False, safe_value=None)
        sample = await self._transport.read_scalar(resource_id)
        return ScalarRealityAdapter(self._transport, profile, initial_sample=sample)

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        return None


def build_configured_azure_digital_twins_connector() -> AzureDigitalTwinsConnector:
    raw = str(os.getenv("AURA_AZURE_DT_RESOURCES_JSON") or "").strip()
    if not raw:
        raise AzureDigitalTwinsConnectorError("azure_dt_resource_manifest_missing")
    resources = parse_azure_digital_twin_manifest(raw)
    provider = EntraClientCredentialsTokenProvider()
    return AzureDigitalTwinsConnector(
        AzureDigitalTwinsScalarTransport(resources, provider),
        resources,
    )


__all__ = [
    "AzureAccessTokenProvider",
    "AzureDigitalTwinResourceSpec",
    "AzureDigitalTwinsConnector",
    "AzureDigitalTwinsConnectorError",
    "AzureDigitalTwinsScalarTransport",
    "EntraClientCredentialsTokenProvider",
    "build_configured_azure_digital_twins_connector",
    "parse_azure_digital_twin_manifest",
]
