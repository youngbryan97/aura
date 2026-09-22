"""openHAB REST discovery and bidirectional Reality Reach connector."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

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

_ITEM_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_NUMBER = re.compile(r"^[\s]*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_SUPPORTED_TYPES = frozenset({"Contact", "Dimmer", "Number", "Rollershutter", "Switch"})


class OpenHABConnectorError(RuntimeError):
    """The configured openHAB endpoint or item contract is invalid."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _csv_env(name: str) -> frozenset[str]:
    return frozenset(
        item.strip()
        for item in str(os.getenv(name) or "").split(",")
        if item.strip()
    )


def _item_mapping_env(name: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for raw in str(os.getenv(name) or "").split(","):
        if not raw.strip():
            continue
        command, separator, feedback = raw.partition(":")
        command = command.strip()
        feedback = feedback.strip()
        if (
            separator != ":"
            or not _ITEM_NAME.fullmatch(command)
            or not _ITEM_NAME.fullmatch(feedback)
            or command == feedback
            or command in mappings
        ):
            raise OpenHABConnectorError("openhab_feedback_item_mapping_invalid")
        mappings[command] = feedback
    if len(set(mappings.values())) != len(mappings):
        raise OpenHABConnectorError("openhab_feedback_item_reused")
    return mappings


def _allow_insecure_http() -> bool:
    from core.runtime.flags import FlagKind, declare

    return str(
        declare(
            "AURA_OPENHAB_ALLOW_HTTP",
            kind=FlagKind.STRING,
            default="",
            description="Permit plain-http openHAB URLs",
            owner="core.embodiment.openhab_connector",
        ).value()
    ).strip().lower() in {"1", "true", "yes", "on"}


def _canonical_resource_id(item_name: str) -> str:
    if not _ITEM_NAME.fullmatch(item_name):
        raise OpenHABConnectorError("openhab_item_name_invalid")
    normalized = re.sub(r"[^a-z0-9]+", "_", item_name.lower()).strip("_")
    if not normalized or len(normalized) > 96:
        raise OpenHABConnectorError("openhab_resource_id_invalid")
    return f"item.{normalized}"


def _state_number(item_type: str, raw_state: object) -> float:
    state = str(raw_state or "").strip()
    if item_type == "Switch":
        if state == "ON":
            return 1.0
        if state == "OFF":
            return 0.0
    if item_type == "Contact":
        if state == "OPEN":
            return 1.0
        if state == "CLOSED":
            return 0.0
    match = _NUMBER.match(state)
    if match is None:
        raise OpenHABConnectorError("openhab_item_state_not_scalar")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise OpenHABConnectorError("openhab_item_state_not_finite")
    return value


def _unit_from_item(item: Mapping[str, Any]) -> str:
    state = str(item.get("state") or "").strip()
    match = _NUMBER.match(state)
    suffix = state[match.end() :].strip() if match is not None else ""
    aliases = {
        "%": "percent",
        "°c": "celsius",
        "°f": "fahrenheit",
        "w": "watt",
        "kw": "kilowatt",
        "wh": "watt_hour",
        "kwh": "kilowatt_hour",
        "v": "volt",
        "a": "ampere",
        "hz": "hertz",
        "lx": "lux",
    }
    if suffix.lower() in aliases:
        return aliases[suffix.lower()]
    if suffix:
        normalized = re.sub(r"[^a-z0-9]+", "_", suffix.lower()).strip("_")
        if normalized:
            return normalized[:64]
    item_type = str(item.get("type") or "")
    if item_type in {"Contact", "Switch"}:
        return "binary"
    if item_type in {"Dimmer", "Rollershutter"}:
        return "percent"
    return "scalar"


def _optional_finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _profile_domain(
    item: Mapping[str, Any],
    *,
    unit: str,
) -> tuple[NumericDomain, float, float | None]:
    item_type = str(item.get("type") or "")
    if item_type in {"Contact", "Switch"}:
        return NumericDomain(0.0, 1.0), 1.0, 0.0
    if item_type in {"Dimmer", "Rollershutter"}:
        return NumericDomain(0.0, 100.0), 1.0, 0.0
    description = item.get("stateDescription")
    if isinstance(description, Mapping):
        minimum = _optional_finite(description.get("minimum"))
        maximum = _optional_finite(description.get("maximum"))
        step = _optional_finite(description.get("step"))
        if minimum is not None and maximum is not None and minimum <= maximum:
            return (
                NumericDomain(minimum, maximum),
                step if step is not None and step > 0.0 else 1e-6,
                None,
            )
    # These bounds describe the declared quantity class, not its current value.
    # A changing reading must never mutate the manifest or physical identity.
    stable_domains = {
        "percent": (0.0, 100.0, 0.01),
        "celsius": (-273.15, 1_000_000.0, 0.001),
        "fahrenheit": (-459.67, 1_800_032.0, 0.001),
        "kelvin": (0.0, 1_000_273.15, 0.001),
        "hertz": (0.0, 1e15, 1e-6),
        "lux": (0.0, 1e12, 1e-6),
        "watt": (-1e15, 1e15, 1e-6),
        "kilowatt": (-1e12, 1e12, 1e-9),
        "watt_hour": (-1e18, 1e18, 1e-6),
        "kilowatt_hour": (-1e15, 1e15, 1e-9),
        "volt": (-1e12, 1e12, 1e-6),
        "ampere": (-1e12, 1e12, 1e-6),
    }
    minimum, maximum, resolution = stable_domains.get(
        unit,
        (-1e15, 1e15, 1e-6),
    )
    return NumericDomain(minimum, maximum), resolution, None


class OpenHABTransport:
    """Credentialed openHAB REST transport through Aura's network gateway."""

    transport_id = "openhab.rest"

    def __init__(self) -> None:
        self._base = str(os.getenv("AURA_OPENHAB_URL") or "").strip().rstrip("/")
        self._token = str(os.getenv("AURA_OPENHAB_TOKEN") or "").strip()
        self._auth_mode = str(os.getenv("AURA_OPENHAB_AUTH_MODE") or "token_basic").strip()
        if not self._base or not self._token:
            raise RuntimeError("openhab_credentials_missing")
        parsed = urllib.parse.urlparse(self._base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise RuntimeError("openhab_url_must_be_origin_only_http_or_https")
        if parsed.scheme == "http" and not _allow_insecure_http():
            raise RuntimeError("openhab_insecure_http_requires_explicit_opt_in")
        if self._auth_mode not in {"bearer", "token_basic"}:
            raise RuntimeError("openhab_auth_mode_invalid")
        self._resources: dict[str, tuple[str, str, str, str]] = {}

    @property
    def base(self) -> str:
        return self._base

    def _headers(self, *, content_type: str = "") -> dict[str, str]:
        if self._auth_mode == "bearer":
            authorization = f"Bearer {self._token}"
        else:
            encoded = base64.b64encode(f"{self._token}:".encode()).decode("ascii")
            authorization = f"Basic {encoded}"
        headers = {"Accept": "application/json", "Authorization": authorization}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def bind_resource(
        self,
        resource_id: str,
        *,
        item_name: str,
        item_type: str,
        readback_item_name: str | None = None,
        readback_item_type: str | None = None,
    ) -> None:
        canonical = _canonical_resource_id(item_name)
        if resource_id != canonical or item_type not in _SUPPORTED_TYPES:
            raise OpenHABConnectorError("openhab_resource_binding_invalid")
        readback_name = readback_item_name or item_name
        readback_type = readback_item_type or item_type
        if (
            not _ITEM_NAME.fullmatch(readback_name)
            or readback_type not in _SUPPORTED_TYPES
        ):
            raise OpenHABConnectorError("openhab_readback_binding_invalid")
        existing = self._resources.get(resource_id)
        binding = (item_name, item_type, readback_name, readback_type)
        if existing is not None and existing != binding:
            raise OpenHABConnectorError("openhab_resource_binding_collision")
        self._resources[resource_id] = binding

    def _binding(self, resource_id: str) -> tuple[str, str, str, str]:
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise OpenHABConnectorError("openhab_resource_not_bound") from exc

    async def discover_items(self) -> list[dict[str, Any]]:
        response = await ActionExecutor.request_network_transport(
            method="GET",
            url=f"{self.base}/rest/items",
            headers=self._headers(),
            timeout_s=8.0,
            source="reality_reach:openhab.discover",
            read_only=True,
        )
        if not response.get("ok"):
            raise OpenHABConnectorError(
                str(response.get("error") or "openhab_discovery_failed")[:300]
            )
        content = response.get("content") or b"[]"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="strict")
        decoded = json.loads(str(content))
        if not isinstance(decoded, list):
            raise OpenHABConnectorError("openhab_discovery_response_not_list")
        return [dict(item) for item in decoded if isinstance(item, Mapping)][:5000]

    async def read_item(self, item_name: str) -> dict[str, Any]:
        if not _ITEM_NAME.fullmatch(item_name):
            raise OpenHABConnectorError("openhab_item_name_invalid")
        quoted = urllib.parse.quote(item_name, safe="")
        response = await ActionExecutor.request_network_transport(
            method="GET",
            url=f"{self.base}/rest/items/{quoted}",
            headers=self._headers(),
            timeout_s=8.0,
            source="reality_reach:openhab.readback",
            read_only=True,
        )
        if not response.get("ok"):
            raise OpenHABConnectorError(
                str(response.get("error") or "openhab_readback_failed")[:300]
            )
        content = response.get("content") or b"{}"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="strict")
        decoded = json.loads(str(content))
        if not isinstance(decoded, dict) or decoded.get("name") != item_name:
            raise OpenHABConnectorError("openhab_readback_identity_mismatch")
        return decoded

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        item_name, item_type, readback_name, readback_type = self._binding(resource_id)
        item = await self.read_item(readback_name)
        if str(item.get("type") or "") != readback_type:
            raise OpenHABConnectorError("openhab_item_type_drift")
        return ScalarSample(
            value=_state_number(readback_type, item.get("state")),
            captured_at_ns=max(1, time.time_ns()),
            source_event_id=_digest(
                {
                    "command_item": item_name,
                    "readback_item": readback_name,
                    "state": item.get("state"),
                    "type": readback_type,
                }
            ),
            quality="uncertain_source_timestamp",
        )

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        item_name, item_type, _readback_name, _readback_type = self._binding(resource_id)
        number = float(value)
        if item_type == "Switch":
            command = "ON" if number >= 0.5 else "OFF"
        elif item_type in {"Dimmer", "Rollershutter"}:
            command = f"{number:.6f}".rstrip("0").rstrip(".")
        elif item_type == "Number":
            command = repr(number)
        else:
            raise OpenHABConnectorError("openhab_item_is_read_only")
        quoted = urllib.parse.quote(item_name, safe="")
        response = await ActionExecutor.request_network_transport(
            method="POST",
            url=f"{self.base}/rest/items/{quoted}",
            headers={
                **self._headers(content_type="text/plain; charset=utf-8"),
                "X-Aura-Idempotency-Key": idempotency_key,
            },
            data=command,
            timeout_s=8.0,
            source=(
                "reality_reach:openhab.recovery"
                if recovery
                else "reality_reach:openhab.actuate"
            ),
            read_only=False,
        )
        status = int(response.get("status_code") or 0)
        accepted = bool(response.get("ok") and 200 <= status < 300)
        return ScalarWriteResult(
            accepted=accepted,
            transport_completed=accepted,
            receipt={
                "protocol": self.transport_id,
                "resource_id": resource_id,
                "status_code": status,
                "recovery": recovery,
                "accepted": accepted,
                "error": "" if accepted else str(response.get("error") or "refused")[:240],
            },
        )


class OpenHABConnector:
    """Discover stable openHAB items and attach typed scalar adapters."""

    connector_id = "openhab.local"

    def __init__(
        self,
        transport: OpenHABTransport,
        *,
        candidate_ttl_s: float = 180.0,
    ) -> None:
        if not isinstance(transport, OpenHABTransport):
            raise TypeError("transport must be OpenHABTransport")
        self._transport = transport
        self._ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))
        self._installation_id = str(os.getenv("AURA_OPENHAB_INSTALLATION_ID") or "").strip()
        self._observe_items = _csv_env("AURA_OPENHAB_ITEMS")
        self._control_items = _csv_env("AURA_OPENHAB_CONTROL_ITEMS")
        self._feedback_items = _item_mapping_env("AURA_OPENHAB_FEEDBACK_ITEMS")

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        items = await self._transport.discover_items()
        items_by_name = {
            str(item.get("name") or ""): item
            for item in items
            if _ITEM_NAME.fullmatch(str(item.get("name") or ""))
        }
        candidates: list[DeviceCandidate] = []
        seen_resources: set[str] = set()
        for item in items:
            item_name = str(item.get("name") or "")
            if self._observe_items and item_name not in self._observe_items:
                continue
            if (
                item_name in self._feedback_items.values()
                and item_name not in self._observe_items
            ):
                continue
            feedback_name = self._feedback_items.get(item_name, "")
            feedback_item = items_by_name.get(feedback_name) if feedback_name else None
            bound_feedback_name = feedback_name if feedback_item is not None else ""
            try:
                candidate, profile = self._candidate_and_profile(
                    item,
                    feedback_item=feedback_item,
                )
            except (OpenHABConnectorError, TypeError, ValueError):
                continue
            if profile.resource_id in seen_resources:
                continue
            seen_resources.add(profile.resource_id)
            self._transport.bind_resource(
                profile.resource_id,
                item_name=item_name,
                item_type=str(item.get("type") or ""),
                readback_item_name=bound_feedback_name or None,
                readback_item_type=(
                    str(feedback_item.get("type") or "")
                    if feedback_item is not None
                    else None
                ),
            )
            candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))

    async def attach(
        self,
        candidate: DeviceCandidate,
        access: tuple[AttachmentAccess, ...],
    ) -> LiveChannelAdapter:
        if candidate.connector_id != self.connector_id:
            raise ValueError("openhab_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("openhab_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("openhab_control_requires_observation")
        item_name = str(candidate.metadata.get("item_name") or "")
        item = await self._transport.read_item(item_name)
        feedback_name = str(candidate.metadata.get("feedback_item_name") or "")
        feedback_item = (
            await self._transport.read_item(feedback_name)
            if feedback_name
            else None
        )
        current, profile = self._candidate_and_profile(
            item,
            feedback_item=feedback_item,
        )
        if (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("openhab_candidate_changed_before_attachment")
        self._transport.bind_resource(
            profile.resource_id,
            item_name=item_name,
            item_type=str(item.get("type") or ""),
            readback_item_name=feedback_name or None,
            readback_item_type=(
                str(feedback_item.get("type") or "")
                if feedback_item is not None
                else None
            ),
        )
        if AttachmentAccess.CONTROL not in requested:
            profile = replace(profile, writable=False, safe_value=None)
        sample = await self._transport.read_scalar(profile.resource_id)
        return ScalarRealityAdapter(self._transport, profile, initial_sample=sample)

    async def detach(self, _adapter: LiveChannelAdapter) -> None:
        return None

    def _candidate_and_profile(
        self,
        item: Mapping[str, Any],
        *,
        feedback_item: Mapping[str, Any] | None = None,
    ) -> tuple[DeviceCandidate, ScalarResourceProfile]:
        item_name = str(item.get("name") or "")
        item_type = str(item.get("type") or "")
        if item_type not in _SUPPORTED_TYPES or not _ITEM_NAME.fullmatch(item_name):
            raise OpenHABConnectorError("openhab_item_type_unsupported")
        value = _state_number(item_type, item.get("state"))
        unit = _unit_from_item(item)
        domain, resolution, safe_value = _profile_domain(item, unit=unit)
        if not domain.contains(value):
            raise OpenHABConnectorError("openhab_item_state_outside_declared_domain")
        resource_id = _canonical_resource_id(item_name)
        feedback_name = str((feedback_item or {}).get("name") or "")
        feedback_type = str((feedback_item or {}).get("type") or "")
        feedback_unit = _unit_from_item(feedback_item or {}) if feedback_item else ""
        feedback_domain = (
            _profile_domain(feedback_item, unit=feedback_unit)[0]
            if feedback_item is not None and feedback_type in _SUPPORTED_TYPES
            else None
        )
        feedback_compatible = bool(
            feedback_name == self._feedback_items.get(item_name)
            and feedback_name != item_name
            and feedback_type in _SUPPORTED_TYPES
            and feedback_unit == unit
            and feedback_domain == domain
        )
        writable = bool(
            item_name in self._control_items
            and item_type != "Contact"
            and feedback_compatible
        )
        installation = self._installation_id or f"session:{self._transport.base}"
        identity = _digest(
            {
                "connector": self.connector_id,
                "installation": installation,
                "item_name": item_name,
                "feedback_item_name": feedback_name,
            }
        )
        profile = ScalarResourceProfile(
            resource_id=resource_id,
            observable=f"openhab_{resource_id.replace('.', '_')}",
            unit=unit,
            domain=domain,
            resolution=resolution,
            writable=writable,
            physical_identity_sha256=identity,
            owner="core.embodiment.openhab_connector",
            protocol="openhab",
            safe_value=safe_value if writable else None,
            max_commands_per_minute=12,
            cooldown_s=0.1,
            stale_after_s=30.0,
            readback_distinct_from_command=feedback_compatible,
        )
        manifest = _digest(
            {
                "profile_sha256": profile.sha256,
                "item_type": item_type,
                "feedback_item_name": feedback_name,
                "feedback_item_type": feedback_type,
                "groups": sorted(str(group) for group in item.get("groupNames") or []),
                "tags": sorted(str(tag) for tag in item.get("tags") or []),
            }
        )
        now_ns = max(1, time.time_ns())
        candidate_id = f"openhab.candidate.{manifest.removeprefix('sha256:')[:32]}"
        access = (
            (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
            if writable
            else (AttachmentAccess.OBSERVE,)
        )
        return (
            DeviceCandidate(
                candidate_id=candidate_id,
                connector_id=self.connector_id,
                device_id=f"openhab.{resource_id}",
                display_name=str(item.get("label") or item_name)[:160],
                transport=self._transport.transport_id,
                identity_fingerprint=identity,
                manifest_sha256=manifest,
                access=access,
                discovered_at_ns=now_ns,
                expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                persistent_identity=bool(self._installation_id),
                proposal_salience=0.4,
                metadata={
                    "item_name": item_name,
                    "item_type": item_type,
                    "feedback_item_name": feedback_name,
                    "resource_id": resource_id,
                    "profile_sha256": profile.sha256,
                    "control_available": writable,
                    "independent_readback": feedback_compatible,
                    "identity_strength": (
                        "installation_scoped"
                        if self._installation_id
                        else "session_endpoint_scoped"
                    ),
                },
            ),
            profile,
        )


__all__ = [
    "OpenHABConnector",
    "OpenHABConnectorError",
    "OpenHABTransport",
]
