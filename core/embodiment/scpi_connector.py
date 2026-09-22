"""Manifest-bound SCPI laboratory measurements and verified setpoints.

The connector intentionally exposes no arbitrary SCPI console.  Every query,
numeric domain, command template, safe value, and readback relationship is
declared before boot.  Commands complete only after ``*OPC?`` and a clear
instrument error queue; physical success still requires the shared scalar
adapter's fresh independent readback.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from core.governance_context import require_governance
from core.reality_reach.attachments import AttachmentAccess, DeviceCandidate
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import LiveChannelAdapter
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import NetworkEffectDenied
from core.runtime.lockdep import checked_async_lock
from core.runtime.network_gateway import StreamAdmission, get_network_gateway

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCPI_LINE = re.compile(r"^[A-Za-z*][A-Za-z0-9:*? .,_+\-/{}()@\[\]]*$")
_MAX_LINE_BYTES = 65_536
_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_IDEMPOTENCY_RECEIPTS = 512
_CONTROL_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
)


class SCPIConnectorError(RuntimeError):
    """A SCPI manifest, transport, identity, or effect contract failed."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _text_digest(value: str) -> str:
    return str(sha256_hex(value.encode("utf-8")))


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


def _scpi_line(value: object, *, name: str, query: bool) -> str:
    line = str(value or "").strip()
    if (
        not line
        or len(line.encode("ascii", errors="ignore")) > 256
        or not line.isascii()
        or any(token in line for token in ("\n", "\r", "\x00", ";"))
        or not _SCPI_LINE.fullmatch(line)
        or line.count("?") != int(query)
    ):
        kind = "query" if query else "command"
        raise ValueError(f"{name} must be one bounded single SCPI {kind}")
    return line


def _command_template(value: object) -> str:
    template = str(value or "").strip()
    remainder = template.replace("{value}", "")
    if (
        template.count("{value}") != 1
        or "{" in remainder
        or "}" in remainder
    ):
        raise ValueError("command_template must contain exactly one {value} placeholder")
    probe = template.replace("{value}", "0")
    _scpi_line(probe, name="command_template", query=False)
    return template


def _allow_plaintext() -> bool:
    from core.runtime.flags import FlagKind, declare

    return str(
        declare(
            "AURA_SCPI_ALLOW_PLAINTEXT",
            kind=FlagKind.STRING,
            default="",
            description="Permit observation-only plaintext SCPI streams",
            owner="core.embodiment.scpi_connector",
        ).value()
    ).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class SCPIResourceSpec:
    resource_id: str
    device_id: str
    observable: str
    unit: str
    read_query: str
    domain: NumericDomain
    resolution: float
    command_template: str = ""
    safe_value: float | None = None
    tolerance: float | None = None
    uncertainty: float | None = None
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("resource_id", "device_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "read_query",
            _scpi_line(self.read_query, name="read_query", query=True),
        )
        command = str(self.command_template or "").strip()
        if command:
            command = _command_template(command)
        object.__setattr__(self, "command_template", command)
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be NumericDomain")
        resolution = _finite(self.resolution, name="resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        object.__setattr__(self, "resolution", resolution)
        tolerance = resolution if self.tolerance is None else _finite(
            self.tolerance,
            name="tolerance",
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
            if not command or not self.domain.contains(safe):
                raise ValueError("safe_value requires a writable in-domain resource")
            object.__setattr__(self, "safe_value", safe)
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("SCPI timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def writable(self) -> bool:
        return bool(self.command_template)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "device_id": self.device_id,
            "observable": self.observable,
            "unit": self.unit,
            "read_query": self.read_query,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "command_template": self.command_template,
            "safe_value": self.safe_value,
            "tolerance": self.tolerance,
            "uncertainty": self.uncertainty,
            "max_commands_per_minute": self.max_commands_per_minute,
            "cooldown_s": self.cooldown_s,
            "stale_after_s": self.stale_after_s,
        }

    def decode(self, response: str) -> float:
        if len(response.encode("utf-8")) > _MAX_LINE_BYTES:
            raise SCPIConnectorError("scpi_measurement_response_too_large")
        number = _finite(response.strip(), name="SCPI measurement")
        if not self.domain.contains(number):
            raise SCPIConnectorError("scpi_measurement_outside_manifest_domain")
        return number

    def command(self, value: float) -> str:
        number = _finite(value, name="SCPI command")
        if not self.writable or not self.domain.contains(number):
            raise SCPIConnectorError("scpi_command_outside_manifest_contract")
        return _scpi_line(
            self.command_template.replace("{value}", format(number, ".17g")),
            name="compiled command",
            query=False,
        )


def parse_scpi_resource_manifest(raw: object) -> tuple[SCPIResourceSpec, ...]:
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > _MAX_MANIFEST_BYTES:
            raise SCPIConnectorError("scpi_manifest_too_large")
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SCPIConnectorError("scpi_manifest_invalid_json") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SCPIConnectorError("scpi_manifest_must_be_a_list")
    if not 1 <= len(raw) <= 512:
        raise SCPIConnectorError("scpi_manifest_size_invalid")
    resources: list[SCPIResourceSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise SCPIConnectorError("scpi_manifest_entry_invalid")
        resources.append(
            SCPIResourceSpec(
                resource_id=str(item.get("resource_id") or ""),
                device_id=str(item.get("device_id") or ""),
                observable=str(item.get("observable") or ""),
                unit=str(item.get("unit") or ""),
                read_query=str(item.get("read_query") or ""),
                domain=NumericDomain(
                    _finite(item.get("minimum"), name="minimum"),
                    _finite(item.get("maximum"), name="maximum"),
                ),
                resolution=_finite(item.get("resolution"), name="resolution"),
                command_template=str(item.get("command_template") or ""),
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
                stale_after_s=_finite(item.get("stale_after_s") or 30.0, name="stale_after_s"),
            )
        )
    if len({item.resource_id for item in resources}) != len(resources):
        raise SCPIConnectorError("scpi_resource_id_duplicate")
    if len({item.read_query for item in resources}) != len(resources):
        raise SCPIConnectorError("scpi_read_query_duplicate")
    return tuple(sorted(resources, key=lambda item: item.resource_id))


@runtime_checkable
class SCPIScalarTransport(Protocol):
    @property
    def transport_id(self) -> str: ...

    @property
    def instrument_identity_sha256(self) -> str: ...

    @property
    def identity_stable(self) -> bool: ...

    async def read_scalar(self, resource_id: str) -> ScalarSample: ...

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult: ...


class SCPIStreamTransport:
    """Serialized IEEE-488.2/SCPI line transport over governed TCP or TLS."""

    transport_id = "scpi.stream"

    def __init__(self, resources: tuple[SCPIResourceSpec, ...]) -> None:
        if not resources:
            raise ValueError("resources must not be empty")
        endpoint = str(os.getenv("AURA_SCPI_ENDPOINT") or "").strip()
        installation = _identifier(
            os.getenv("AURA_SCPI_INSTALLATION_ID"),
            name="AURA_SCPI_INSTALLATION_ID",
        )
        expected_idn = str(os.getenv("AURA_SCPI_EXPECTED_IDN_SHA256") or "").strip().lower()
        certificate_pin = str(
            os.getenv("AURA_SCPI_SERVER_CERT_SHA256") or ""
        ).strip().lower()
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"tcp", "tls"} or not parsed.hostname or parsed.port is None:
            raise SCPIConnectorError("scpi_endpoint_invalid")
        if not _DIGEST.fullmatch(expected_idn):
            raise SCPIConnectorError("scpi_expected_idn_sha256_required")
        if parsed.scheme == "tcp" and not _allow_plaintext():
            raise SCPIConnectorError("scpi_plaintext_requires_explicit_opt_in")
        if any(item.writable for item in resources):
            if parsed.scheme != "tls" or not _DIGEST.fullmatch(certificate_pin):
                raise SCPIConnectorError("scpi_control_requires_tls_certificate_pin")
        elif certificate_pin and not _DIGEST.fullmatch(certificate_pin):
            raise SCPIConnectorError("scpi_server_certificate_pin_invalid")
        timeout_s = _finite(os.getenv("AURA_SCPI_TIMEOUT_S") or 5.0, name="timeout")
        if not 0.1 <= timeout_s <= 120.0:
            raise ValueError("SCPI timeout must lie inside [0.1, 120]")
        self._resources = {item.resource_id: item for item in resources}
        self._endpoint = endpoint
        self._installation = installation
        self._expected_idn = expected_idn
        self._certificate_pin = certificate_pin
        self._secure = parsed.scheme == "tls"
        self._timeout_s = timeout_s
        self._admission: StreamAdmission | None = None
        self._connect_lock = checked_async_lock("scpi.connect")
        self._io_lock = checked_async_lock("scpi.io")
        self._sequence = 0
        self._idempotency: dict[str, tuple[str, float, ScalarWriteResult]] = {}
        self._instrument_identity = _digest(
            {
                "installation": installation,
                "endpoint": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}",
                "expected_idn_sha256": expected_idn,
                "certificate_pin": certificate_pin,
            }
        )

    @property
    def instrument_identity_sha256(self) -> str:
        return self._instrument_identity

    @property
    def identity_stable(self) -> bool:
        return self._secure and bool(self._certificate_pin)

    async def _ensure_connected(self, *, read_only: bool = True) -> StreamAdmission:
        current = self._admission
        if (
            current is not None
            and not current.writer.is_closing()
            and (read_only or not current.read_only)
        ):
            return current
        async with self._connect_lock:
            current = self._admission
            if (
                current is not None
                and not current.writer.is_closing()
                and (read_only or not current.read_only)
            ):
                return current
            if current is not None:
                self._admission = None
                await self._close_admission(current)
            admission: StreamAdmission | None = None
            try:
                admission = await get_network_gateway().connect_stream(
                    self._endpoint,
                    open_timeout=self._timeout_s,
                    read_limit=_MAX_LINE_BYTES,
                    source="reality_reach:scpi.stream",
                    read_only=read_only,
                    allow_private_target=True,
                    expected_certificate_sha256=self._certificate_pin,
                )
                idn = await self._query_on(admission, "*IDN?")
            except (NetworkEffectDenied, OSError, RuntimeError, TimeoutError, ValueError) as exc:
                if admission is not None:
                    await self._close_admission(admission)
                raise SCPIConnectorError("scpi_connect_or_identity_failed") from exc
            if _text_digest(idn.strip()) != self._expected_idn:
                await self._close_admission(admission)
                raise SCPIConnectorError("scpi_instrument_identity_mismatch")
            self._admission = admission
            return admission

    async def _query_on(self, admission: StreamAdmission, command: str) -> str:
        admission.writer.write((command + "\n").encode("ascii"))
        async with asyncio.timeout(self._timeout_s):
            await admission.writer.drain()
            payload = await admission.reader.readline()
        if not payload or len(payload) > _MAX_LINE_BYTES or not payload.endswith(b"\n"):
            raise SCPIConnectorError("scpi_response_missing_or_unterminated")
        try:
            return payload.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise SCPIConnectorError("scpi_response_not_ascii") from exc

    async def _query(self, command: str) -> str:
        async with self._io_lock:
            admission = await self._ensure_connected()
            try:
                return await self._query_on(admission, command)
            except (OSError, RuntimeError, TimeoutError, UnicodeError) as exc:
                await self._invalidate(admission)
                raise SCPIConnectorError("scpi_query_failed") from exc

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("scpi_resource_not_bound")
        response = await self._query(spec.read_query)
        value = spec.decode(response)
        self._sequence += 1
        captured_at_ns = max(1, time.time_ns())
        return ScalarSample(
            value=value,
            captured_at_ns=captured_at_ns,
            source_event_id=_digest(
                {
                    "instrument_identity_sha256": self._instrument_identity,
                    "resource_id": resource_id,
                    "value": value,
                    "sequence": self._sequence,
                }
            ),
            quality="instrument_reported",
            uncertainty=spec.uncertainty,
            source_epoch=self._instrument_identity,
            source_sequence=self._sequence,
        )

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
            raise PermissionError("scpi_resource_not_writable")
        stable_key = str(idempotency_key or "").strip()
        if not stable_key or len(stable_key.encode("utf-8")) > 256:
            raise ValueError("scpi_idempotency_key_invalid")
        command = spec.command(value)
        require_governance(
            f"scpi.write_scalar:{resource_id}",
            strict=True,
            allowed_domains=_CONTROL_DOMAINS,
        )
        async with self._io_lock:
            previous = self._idempotency.get(stable_key)
            if previous is not None:
                old_resource, old_value, result = previous
                if old_resource != resource_id or old_value != float(value):
                    raise SCPIConnectorError("scpi_idempotency_key_conflict")
                return result
            admission = await self._ensure_connected(read_only=False)
            dispatched = False
            try:
                admission.writer.write((command + "\n").encode("ascii"))
                dispatched = True
                async with asyncio.timeout(self._timeout_s):
                    await admission.writer.drain()
                completed = await self._query_on(admission, "*OPC?")
                error = await self._query_on(admission, "SYST:ERR?")
            except (OSError, RuntimeError, TimeoutError, UnicodeError) as exc:
                await self._invalidate(admission)
                reason = "scpi_command_effect_indeterminate" if dispatched else "scpi_command_failed"
                raise SCPIConnectorError(reason) from exc
            if completed.strip() != "1":
                raise SCPIConnectorError("scpi_operation_not_complete")
            error_code = error.split(",", 1)[0].strip()
            if error_code not in {"0", "+0", "-0"}:
                raise SCPIConnectorError("scpi_instrument_error_queue_nonzero")
            result = ScalarWriteResult(
                accepted=True,
                transport_completed=True,
                receipt={
                    "protocol": self.transport_id,
                    "resource_id": resource_id,
                    "instrument_identity_sha256": self._instrument_identity,
                    "command_sha256": _digest(command),
                    "idempotency_sha256": _digest(stable_key),
                    "operation_complete": True,
                    "error_queue_clear": True,
                    "recovery": recovery,
                },
            )
            if len(self._idempotency) >= _MAX_IDEMPOTENCY_RECEIPTS:
                self._idempotency.pop(next(iter(self._idempotency)))
            self._idempotency[stable_key] = (resource_id, float(value), result)
            return result

    async def _invalidate(self, admission: StreamAdmission) -> None:
        if self._admission is admission:
            self._admission = None
        await self._close_admission(admission)

    @staticmethod
    async def _close_admission(admission: StreamAdmission) -> None:
        admission.writer.close()
        try:
            await asyncio.wait_for(admission.writer.wait_closed(), timeout=2.0)
        # not a failure: a connection already closing, or one that will not close inside
        # its bound, is the state this teardown reaches.
        except (ConnectionError, OSError, RuntimeError):
            return
        # not a failure: the wait ran out, which is what the timeout was set to decide.
        except TimeoutError:
            return

    async def stop(self) -> None:
        async with self._io_lock:
            async with self._connect_lock:
                admission = self._admission
                self._admission = None
        if admission is not None:
            await self._close_admission(admission)


class SCPIConnector:
    """Expose declared laboratory readings and verified setpoints for attachment."""

    connector_id = "scpi.manifest"

    def __init__(
        self,
        transport: SCPIScalarTransport,
        resources: tuple[SCPIResourceSpec, ...],
        *,
        candidate_ttl_s: float = 180.0,
        discovery_budget_s: float = 30.0,
    ) -> None:
        if not isinstance(transport, SCPIScalarTransport):
            raise TypeError("transport must satisfy SCPIScalarTransport")
        if not resources:
            raise ValueError("resources must not be empty")
        self._transport = transport
        self._resources = {item.resource_id: item for item in resources}
        self._ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))
        budget = _finite(discovery_budget_s, name="discovery_budget_s")
        if not 0.01 <= budget <= 300.0:
            raise ValueError("discovery_budget_s must lie inside [0.01, 300]")
        self._discovery_budget_s = budget

    def _profile(self, spec: SCPIResourceSpec) -> ScalarResourceProfile:
        writable = spec.writable and self._transport.identity_stable
        return ScalarResourceProfile(
            resource_id=spec.resource_id,
            observable=spec.observable,
            unit=spec.unit,
            domain=spec.domain,
            resolution=spec.resolution,
            writable=writable,
            physical_identity_sha256=_digest(
                {
                    "instrument": self._transport.instrument_identity_sha256,
                    "device_id": spec.device_id,
                    "resource_id": spec.resource_id,
                    "read_query_sha256": _digest(spec.read_query),
                    "command_template_sha256": (
                        _digest(spec.command_template) if spec.command_template else ""
                    ),
                }
            ),
            owner="core.embodiment.scpi_connector",
            protocol="scpi",
            safe_value=spec.safe_value if writable else None,
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
                    "instrument_identity_sha256": self._transport.instrument_identity_sha256,
                }
            )
            access = (
                (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                if profile.writable
                else (AttachmentAccess.OBSERVE,)
            )
            candidates.append(
                DeviceCandidate(
                    candidate_id="scpi.candidate." + manifest.removeprefix("sha256:")[:32],
                    connector_id=self.connector_id,
                    device_id=f"scpi.{spec.device_id}.{spec.resource_id}",
                    display_name=f"{spec.device_id}: {spec.observable}"[:160],
                    transport=self._transport.transport_id,
                    identity_fingerprint=profile.physical_identity_sha256,
                    manifest_sha256=manifest,
                    access=access,
                    discovered_at_ns=now_ns,
                    expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                    persistent_identity=self._transport.identity_stable,
                    proposal_salience=0.35,
                    metadata={
                        "resource_id": spec.resource_id,
                        "device_id": spec.device_id,
                        "spec_sha256": spec.sha256,
                        "profile_sha256": profile.sha256,
                        "control_available": profile.writable,
                        "independent_readback": spec.writable,
                        "instrument_identity_sha256": self._transport.instrument_identity_sha256,
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
            raise ValueError("scpi_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("scpi_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("scpi_control_requires_observation")
        resource_id = str(candidate.metadata.get("resource_id") or "")
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("scpi_candidate_resource_missing")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("scpi_candidate_changed_before_attachment")
        profile = self._profile(spec)
        if AttachmentAccess.CONTROL not in requested:
            profile = replace(profile, writable=False, safe_value=None)
        sample = await self._transport.read_scalar(resource_id)
        return ScalarRealityAdapter(self._transport, profile, initial_sample=sample)

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        return None

    async def stop(self) -> None:
        stop = getattr(self._transport, "stop", None)
        if not callable(stop):
            return
        result = stop()
        if asyncio.iscoroutine(result):
            await result


def build_configured_scpi_connector() -> SCPIConnector:
    raw = str(os.getenv("AURA_SCPI_RESOURCES_JSON") or "").strip()
    if not raw:
        raise SCPIConnectorError("scpi_resource_manifest_missing")
    resources = parse_scpi_resource_manifest(raw)
    return SCPIConnector(SCPIStreamTransport(resources), resources)


__all__ = [
    "SCPIConnector",
    "SCPIConnectorError",
    "SCPIResourceSpec",
    "SCPIScalarTransport",
    "SCPIStreamTransport",
    "build_configured_scpi_connector",
    "parse_scpi_resource_manifest",
]
