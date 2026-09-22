"""Typed Home Assistant transport and Reality Reach entity adapters."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import urllib.parse
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, cast

from core.reality_reach.actuation import (
    ActuationCommand,
    ActuationLease,
    ActuationReceipt,
    ActuationState,
    ActuatorCapability,
    EffectReceipt,
    PreparedActuation,
    Reversibility,
    RollbackReceipt,
)
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, ReadingStatus
from core.runtime.action_executor import ActionExecutor
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_ENTITY_ID = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_ENTITY_ID_LENGTH = 80
_MAX_PAYLOAD_BYTES = 16 * 1024
_POWER_DOMAINS = frozenset({"fan", "input_boolean", "light", "switch"})
_LIGHT_PARAMETERS = frozenset(
    {
        "brightness",
        "brightness_pct",
        "color_temp",
        "color_temp_kelvin",
        "rgb_color",
        "transition",
    }
)
_FAN_PARAMETERS = frozenset({"percentage"})
_UNIT_ALIASES = {
    "%": "percent",
    "°c": "celsius",
    "c": "celsius",
    "°f": "fahrenheit",
    "f": "fahrenheit",
    "w": "watt",
    "kw": "kilowatt",
    "wh": "watt_hour",
    "kwh": "kilowatt_hour",
    "v": "volt",
    "mv": "millivolt",
    "a": "ampere",
    "ma": "milliampere",
    "hz": "hertz",
    "pa": "pascal",
    "hpa": "hectopascal",
    "ppm": "parts_per_million",
    "lx": "lux",
    "s": "second",
    "ms": "millisecond",
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km/h": "kilometer_per_hour",
    "m/s": "meter_per_second",
}
_SENSOR_DOMAINS: dict[str, tuple[float, float]] = {
    "apparent_power": (0.0, 10_000_000.0),
    "aqi": (0.0, 1000.0),
    "atmospheric_pressure": (0.0, 2000.0),
    "battery": (0.0, 100.0),
    "carbon_dioxide": (0.0, 100_000.0),
    "carbon_monoxide": (0.0, 100_000.0),
    "current": (-100_000.0, 100_000.0),
    "data_rate": (0.0, 1e15),
    "data_size": (0.0, 1e18),
    "distance": (0.0, 1e12),
    "duration": (0.0, 1e12),
    "energy": (-1e15, 1e15),
    "frequency": (0.0, 1e12),
    "gas": (0.0, 100_000.0),
    "humidity": (0.0, 100.0),
    "illuminance": (0.0, 10_000_000.0),
    "moisture": (0.0, 100.0),
    "nitrogen_dioxide": (0.0, 100_000.0),
    "nitrogen_monoxide": (0.0, 100_000.0),
    "nitrous_oxide": (0.0, 100_000.0),
    "ozone": (0.0, 100_000.0),
    "pm1": (0.0, 1_000_000.0),
    "pm10": (0.0, 1_000_000.0),
    "pm25": (0.0, 1_000_000.0),
    "power": (-10_000_000.0, 10_000_000.0),
    "power_factor": (-1.0, 1.0),
    "precipitation": (0.0, 1_000_000.0),
    "pressure": (0.0, 10_000_000.0),
    "reactive_power": (-10_000_000.0, 10_000_000.0),
    "signal_strength": (-300.0, 300.0),
    "sound_pressure": (0.0, 300.0),
    "speed": (-1_000_000.0, 1_000_000.0),
    "sulphur_dioxide": (0.0, 100_000.0),
    "temperature": (-273.15, 5000.0),
    "volatile_organic_compounds": (0.0, 100_000.0),
    "voltage": (-1_000_000.0, 1_000_000.0),
    "volume": (0.0, 1e15),
    "volume_flow_rate": (0.0, 1e15),
    "water": (0.0, 1e15),
    "weight": (0.0, 1e15),
    "wind_speed": (0.0, 1_000_000.0),
}


def _home_assistant_event_time_ns(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    timestamp_ns = int(parsed.timestamp() * 1_000_000_000)
    return timestamp_ns if timestamp_ns > 0 else None


class HomeAssistantRealityError(RuntimeError):
    """Stable refusal raised at the Home Assistant physical-effect boundary."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _home_assistant_source_lineage(
    state: Mapping[str, Any],
    *,
    entity_id: str,
) -> tuple[int, str, str, str, str]:
    if str(state.get("entity_id") or "") != entity_id:
        raise HomeAssistantRealityError("hass_sensor_entity_identity_mismatch")
    raw_source_time = state.get("last_updated") or state.get("last_changed")
    source_time_ns = _home_assistant_event_time_ns(raw_source_time)
    raw_context = state.get("context")
    context_id = (
        str(raw_context.get("id") or "")[:128]
        if isinstance(raw_context, Mapping)
        else ""
    )
    source_event_id = (
        _digest(
            {
                "context_id": context_id,
                "entity_id": entity_id,
                "source_time": str(raw_source_time),
            }
        )
        if source_time_ns is not None
        else ""
    )
    return (
        source_time_ns or max(1, time.time_ns()),
        (
            "home_assistant.last_updated"
            if source_time_ns is not None
            else "system.time_ns.fallback"
        ),
        f"hass.{entity_id}",
        source_event_id,
        "good" if source_time_ns is not None else "uncertain",
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    number = _finite(value, name=name)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} lies outside [{minimum}, {maximum}]")
    return number


def _canonical_entity_id(value: object) -> str:
    entity_id = str(value or "").strip().lower()
    if len(entity_id) > _MAX_ENTITY_ID_LENGTH or not _ENTITY_ID.fullmatch(entity_id):
        raise ValueError("home_assistant_entity_id_invalid")
    return entity_id


def _canonical_identifier(value: object, *, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    alias = _UNIT_ALIASES.get(raw)
    if alias is not None:
        return alias
    normalized = re.sub(r"[^a-z0-9_.:-]+", "_", raw).strip("_.:-")
    if not normalized:
        normalized = fallback
    if not _IDENTIFIER.fullmatch(normalized[:128]):
        raise ValueError("home_assistant_identifier_invalid")
    return normalized[:128]


def _frozen_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        raw = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("home_assistant_payload_not_canonical_json") from exc
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise ValueError("home_assistant_payload_too_large")
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("home_assistant_payload_not_mapping")
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class HomeAssistantEffect:
    target: str
    op: str
    payload: Mapping[str, Any]
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _canonical_entity_id(self.target))
        operation = str(self.op or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9_]+", operation):
            raise ValueError("home_assistant_operation_invalid")
        object.__setattr__(self, "op", operation)
        object.__setattr__(self, "payload", _frozen_payload(self.payload))
        object.__setattr__(self, "reason", str(self.reason or "")[:240])

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self.op,
            "payload": dict(self.payload),
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class _EntityProfile:
    kind: str
    observable: str
    unit: str
    domain: NumericDomain
    tolerance: float
    allowed_operations: tuple[str, ...]
    max_commands_per_minute: int
    cooldown_s: float

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "kind": self.kind,
                "observable": self.observable,
                "unit": self.unit,
                "domain": self.domain.to_dict(),
                "tolerance": self.tolerance,
                "allowed_operations": list(self.allowed_operations),
                "max_commands_per_minute": self.max_commands_per_minute,
                "cooldown_s": self.cooldown_s,
            }
        )


@dataclass(frozen=True, slots=True)
class _SensorProfile:
    observable: str
    unit: str
    domain: NumericDomain
    resolution: float
    device_class: str
    domain_source: str
    binary: bool = False

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "observable": self.observable,
                "unit": self.unit,
                "domain": self.domain.to_dict(),
                "resolution": self.resolution,
                "device_class": self.device_class,
                "domain_source": self.domain_source,
                "binary": self.binary,
            }
        )


def _profile_for_state(entity_id: str, state: Mapping[str, Any]) -> _EntityProfile:
    domain = entity_id.partition(".")[0]
    if domain in _POWER_DOMAINS:
        return _EntityProfile(
            kind="power",
            observable="home_assistant_power_state",
            unit="binary",
            domain=NumericDomain(0.0, 1.0),
            tolerance=0.0,
            allowed_operations=("turn_on", "turn_off"),
            max_commands_per_minute=20,
            cooldown_s=0.2,
        )
    if domain == "climate":
        attributes = state.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        minimum = _finite(attributes.get("min_temp", 5.0), name="climate.min_temp")
        maximum = _finite(attributes.get("max_temp", 35.0), name="climate.max_temp")
        if minimum >= maximum or maximum - minimum > 100.0:
            raise ValueError("home_assistant_climate_range_invalid")
        return _EntityProfile(
            kind="temperature",
            observable="home_assistant_temperature_setpoint",
            unit=_canonical_identifier(
                attributes.get("temperature_unit"),
                fallback="temperature",
            ),
            domain=NumericDomain(minimum, maximum),
            tolerance=0.5,
            allowed_operations=("set_temperature",),
            max_commands_per_minute=6,
            cooldown_s=1.0,
        )
    raise HomeAssistantRealityError(
        f"home_assistant_domain_has_no_physical_manifest:{domain}"
    )


def _sensor_profile_for_state(
    entity_id: str,
    state: Mapping[str, Any],
) -> _SensorProfile:
    domain = entity_id.partition(".")[0]
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}
    device_class = _canonical_identifier(
        attributes.get("device_class"),
        fallback=("power_state" if domain in _POWER_DOMAINS else domain),
    )
    if domain in _POWER_DOMAINS or domain == "binary_sensor":
        state_value = str(state.get("state") or "").strip().lower()
        if state_value not in {"on", "off"}:
            raise HomeAssistantRealityError(
                "home_assistant_binary_sensor_state_unavailable"
            )
        return _SensorProfile(
            observable=f"home_assistant_{device_class}_state"[:128],
            unit="binary",
            domain=NumericDomain(0.0, 1.0),
            resolution=1.0,
            device_class=device_class,
            domain_source="binary_contract",
            binary=True,
        )
    if domain == "climate":
        minimum = _finite(attributes.get("min_temp", 5.0), name="climate.min_temp")
        maximum = _finite(attributes.get("max_temp", 35.0), name="climate.max_temp")
        if minimum >= maximum or maximum - minimum > 100.0:
            raise HomeAssistantRealityError("home_assistant_climate_range_invalid")
        _bounded_number(
            attributes.get("temperature"),
            name="temperature",
            minimum=minimum,
            maximum=maximum,
        )
        return _SensorProfile(
            observable="home_assistant_temperature_setpoint",
            unit=_canonical_identifier(
                attributes.get("temperature_unit"),
                fallback="temperature",
            ),
            domain=NumericDomain(minimum, maximum),
            resolution=0.5,
            device_class="temperature",
            domain_source="home_assistant_limits",
        )
    if domain != "sensor":
        raise HomeAssistantRealityError(
            f"home_assistant_domain_has_no_sensor_manifest:{domain}"
        )
    observed = _finite(state.get("state"), name="sensor.state")
    explicit_min = attributes.get("min_value")
    explicit_max = attributes.get("max_value")
    if explicit_min is not None and explicit_max is not None:
        minimum = _finite(explicit_min, name="sensor.min_value")
        maximum = _finite(explicit_max, name="sensor.max_value")
        if minimum >= maximum:
            raise HomeAssistantRealityError("home_assistant_sensor_range_invalid")
        domain_source = "entity_metadata"
    elif device_class in _SENSOR_DOMAINS:
        minimum, maximum = _SENSOR_DOMAINS[device_class]
        domain_source = "device_class_standard"
    else:
        magnitude = max(100.0, abs(observed) * 4.0)
        span = 10.0 ** math.ceil(math.log10(magnitude))
        minimum, maximum = -span, span
        domain_source = "inferred_magnitude_bucket"
    if not minimum <= observed <= maximum:
        raise HomeAssistantRealityError("home_assistant_sensor_outside_declared_range")
    resolution_raw = attributes.get("step")
    if resolution_raw is None:
        precision = attributes.get("suggested_display_precision")
        try:
            resolution = 10.0 ** -max(0, min(int(cast(Any, precision)), 9))
        except (TypeError, ValueError):
            resolution = max(1e-9, (maximum - minimum) / 1_000_000.0)
    else:
        resolution = _bounded_number(
            resolution_raw,
            name="sensor.step",
            minimum=0.0,
            maximum=max(1.0, maximum - minimum),
        )
    return _SensorProfile(
        observable=f"home_assistant_{device_class}"[:128],
        unit=_canonical_identifier(
            attributes.get("unit_of_measurement"),
            fallback="scalar",
        ),
        domain=NumericDomain(minimum, maximum),
        resolution=resolution,
        device_class=device_class,
        domain_source=domain_source,
    )


def _sensor_value(state: Mapping[str, Any], profile: _SensorProfile) -> float:
    if profile.binary:
        value = str(state.get("state") or "").strip().lower()
        if value not in {"on", "off"}:
            raise HomeAssistantRealityError(
                "home_assistant_binary_sensor_state_unavailable"
            )
        return 1.0 if value == "on" else 0.0
    if str(state.get("entity_id") or "").startswith("climate."):
        attributes = state.get("attributes")
        if not isinstance(attributes, Mapping):
            raise HomeAssistantRealityError("home_assistant_attributes_unavailable")
        raw_value = attributes.get("temperature")
    else:
        raw_value = state.get("state")
    return _bounded_number(
        raw_value,
        name="sensor.value",
        minimum=profile.domain.minimum,
        maximum=profile.domain.maximum,
    )


def _validated_effect(
    effect: HomeAssistantEffect,
    profile: _EntityProfile,
) -> HomeAssistantEffect:
    if effect.op not in profile.allowed_operations:
        raise HomeAssistantRealityError(
            f"home_assistant_operation_not_manifested:{effect.op}"
        )
    domain = effect.target.partition(".")[0]
    payload = dict(effect.payload)
    if profile.kind == "temperature":
        if set(payload) != {"temperature"}:
            raise HomeAssistantRealityError(
                "home_assistant_temperature_requires_only_temperature"
            )
        payload["temperature"] = _bounded_number(
            payload["temperature"],
            name="temperature",
            minimum=profile.domain.minimum,
            maximum=profile.domain.maximum,
        )
        return HomeAssistantEffect(effect.target, effect.op, payload, effect.reason)

    allowed: frozenset[str]
    if domain == "light":
        allowed = _LIGHT_PARAMETERS
    elif domain == "fan":
        allowed = _FAN_PARAMETERS
    else:
        allowed = frozenset()
    unknown = set(payload) - set(allowed)
    if unknown:
        raise HomeAssistantRealityError(
            "home_assistant_parameters_not_manifested:" + ",".join(sorted(unknown))
        )
    if effect.op == "turn_off":
        permitted_when_off = {"transition"} if domain == "light" else set()
        if set(payload) - permitted_when_off:
            raise HomeAssistantRealityError(
                "home_assistant_turn_off_parameters_not_manifested"
            )
    if "brightness" in payload:
        payload["brightness"] = int(
            _bounded_number(payload["brightness"], name="brightness", minimum=0, maximum=255)
        )
    if "brightness_pct" in payload:
        payload["brightness_pct"] = _bounded_number(
            payload["brightness_pct"], name="brightness_pct", minimum=0, maximum=100
        )
    if "color_temp" in payload:
        payload["color_temp"] = int(
            _bounded_number(payload["color_temp"], name="color_temp", minimum=100, maximum=1000)
        )
    if "color_temp_kelvin" in payload:
        payload["color_temp_kelvin"] = int(
            _bounded_number(
                payload["color_temp_kelvin"],
                name="color_temp_kelvin",
                minimum=1000,
                maximum=10000,
            )
        )
    if "color_temp" in payload and "color_temp_kelvin" in payload:
        raise HomeAssistantRealityError("home_assistant_color_temperature_ambiguous")
    if "transition" in payload:
        payload["transition"] = _bounded_number(
            payload["transition"], name="transition", minimum=0, maximum=60
        )
    if "percentage" in payload:
        payload["percentage"] = int(
            _bounded_number(payload["percentage"], name="percentage", minimum=0, maximum=100)
        )
    if "rgb_color" in payload:
        rgb = payload["rgb_color"]
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise HomeAssistantRealityError("home_assistant_rgb_color_invalid")
        payload["rgb_color"] = [
            int(_bounded_number(item, name="rgb_color", minimum=0, maximum=255))
            for item in rgb
        ]
    return HomeAssistantEffect(effect.target, effect.op, payload, effect.reason)


def _expected_attributes(effect: HomeAssistantEffect) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for key, value in effect.payload.items():
        if key == "transition":
            continue
        if key == "brightness_pct":
            expected["brightness"] = round(float(value) * 255.0 / 100.0)
            continue
        expected[key] = value
    return expected


def _values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(observed) - float(expected)) <= 2.0
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return list(observed or []) == expected
    return str(observed) == str(expected)


def state_matches_effect(state: Mapping[str, Any], effect: HomeAssistantEffect) -> bool:
    if str(state.get("entity_id") or "") != effect.target:
        return False
    state_value = str(state.get("state") or "").lower()
    if effect.op == "turn_on" and state_value != "on":
        return False
    if effect.op == "turn_off" and state_value != "off":
        return False
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}
    return all(
        _values_match(expected, attributes.get(key))
        for key, expected in _expected_attributes(effect).items()
    )


def _primary_value(state: Mapping[str, Any], profile: _EntityProfile) -> float:
    if profile.kind == "power":
        value = str(state.get("state") or "").lower()
        if value not in {"on", "off"}:
            raise HomeAssistantRealityError("home_assistant_power_state_unavailable")
        return 1.0 if value == "on" else 0.0
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        raise HomeAssistantRealityError("home_assistant_attributes_unavailable")
    return _bounded_number(
        attributes.get("temperature"),
        name="temperature",
        minimum=profile.domain.minimum,
        maximum=profile.domain.maximum,
    )


def _effect_target(effect: HomeAssistantEffect, profile: _EntityProfile) -> float:
    if profile.kind == "power":
        return 1.0 if effect.op == "turn_on" else 0.0
    return _bounded_number(
        effect.payload.get("temperature"),
        name="temperature",
        minimum=profile.domain.minimum,
        maximum=profile.domain.maximum,
    )


def _state_projection(state: Mapping[str, Any], effect: HomeAssistantEffect) -> dict[str, Any]:
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}
    keys = set(_expected_attributes(effect))
    domain = effect.target.partition(".")[0]
    if domain == "light":
        keys.update({"brightness", "color_temp", "color_temp_kelvin", "rgb_color"})
    elif domain == "fan":
        keys.add("percentage")
    elif domain == "climate":
        keys.add("temperature")
    return {
        "entity_id": str(state.get("entity_id") or "")[:_MAX_ENTITY_ID_LENGTH],
        "state": str(state.get("state") or "")[:80],
        "attributes": {key: attributes.get(key) for key in sorted(keys) if key in attributes},
    }


def _rollback_effect(
    state: Mapping[str, Any],
    effect: HomeAssistantEffect,
    profile: _EntityProfile,
) -> HomeAssistantEffect:
    if profile.kind == "temperature":
        return HomeAssistantEffect(
            effect.target,
            "set_temperature",
            {"temperature": _primary_value(state, profile)},
            "restore_pre_actuation_temperature",
        )
    state_value = str(state.get("state") or "").lower()
    if state_value != "on":
        return HomeAssistantEffect(
            effect.target,
            "turn_off",
            {},
            "restore_pre_actuation_power_state",
        )
    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}
    domain = effect.target.partition(".")[0]
    payload: dict[str, Any] = {}
    if domain == "light":
        for key in ("brightness", "color_temp_kelvin", "rgb_color"):
            if attributes.get(key) is not None:
                payload[key] = attributes[key]
    elif domain == "fan" and attributes.get("percentage") is not None:
        payload["percentage"] = attributes["percentage"]
    return _validated_effect(
        HomeAssistantEffect(
            effect.target,
            "turn_on",
            payload,
            "restore_pre_actuation_power_state",
        ),
        profile,
    )


class HomeAssistantTransport:
    """Credentialed Home Assistant REST transport with typed dispatch entrypoints."""

    name = "home_assistant"

    def __init__(self) -> None:
        self._token = str(
            os.getenv("AURA_HASS_TOKEN") or os.getenv("HASS_TOKEN") or ""
        ).strip()
        self._base = str(
            os.getenv("AURA_HASS_URL")
            or os.getenv("HASS_URL")
            or ("https://homeassistant.local:8123" if self._token else "")
        ).strip().rstrip("/")
        if not self._base or not self._token:
            raise RuntimeError("hass_credentials_missing")
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
            raise RuntimeError("hass_url_must_be_origin_only_http_or_https")
        if parsed.scheme == "http" and str(
            __import__("core.runtime.flags", fromlist=["declare", "FlagKind"]).declare(
                "AURA_HASS_ALLOW_HTTP",
                kind=__import__(
                    "core.runtime.flags", fromlist=["FlagKind"]
                ).FlagKind.STRING,
                default="",
                description="Permit plain-http Home Assistant URLs",
                owner="core.embodiment.home_assistant_reality",
            ).value()
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("hass_insecure_http_requires_explicit_opt_in")

    @property
    def base(self) -> str:
        """Return the non-secret configured origin used for stable identity."""

        return self._base

    def _headers(self, *, json_content: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._token}"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    async def read_state(self, entity_id: str) -> dict[str, Any]:
        target = _canonical_entity_id(entity_id)
        from core.governance_context import (
            get_active_governance,
            governance_runtime_active,
            governed_scope,
            local_internal_decision,
        )

        if get_active_governance() is None and governance_runtime_active():
            decision = local_internal_decision(
                "home_assistant.reality_readback",
                domain="environment_action",
                constraints={
                    "read_only": True,
                    "configured_transport": "home_assistant",
                    "target_sha256": _digest(target),
                },
            )
            async with governed_scope(decision):
                return await self._read_state_governed(target)
        return await self._read_state_governed(target)

    async def _read_state_governed(self, target: str) -> dict[str, Any]:
        response = await ActionExecutor.request_network_transport(
            method="GET",
            url=f"{self.base}/api/states/{target}",
            headers=self._headers(),
            timeout_s=8.0,
            source="world_bridge:iot.home_assistant.readback",
            read_only=True,
        )
        status = int(response.get("status_code") or 0)
        if not response.get("ok") or not 200 <= status < 300:
            raise HomeAssistantRealityError(
                str(response.get("error") or f"hass_state_http_{status}")[:300]
            )
        content = response.get("content") or b"{}"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="strict")
        decoded = json.loads(str(content))
        if not isinstance(decoded, dict):
            raise HomeAssistantRealityError("hass_state_response_not_mapping")
        if str(decoded.get("entity_id") or "") != target:
            raise HomeAssistantRealityError("hass_state_entity_identity_mismatch")
        return decoded

    async def discover(self) -> list[dict[str, Any]]:
        response = await ActionExecutor.request_network_transport(
            method="GET",
            url=f"{self.base}/api/states",
            headers=self._headers(),
            timeout_s=8.0,
            source="world_bridge:iot.home_assistant.discover",
            read_only=True,
        )
        if not response.get("ok"):
            raise HomeAssistantRealityError(
                str(response.get("error") or "hass_discovery_failed")[:300]
            )
        content = response.get("content") or b"[]"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="strict")
        decoded = json.loads(str(content))
        if not isinstance(decoded, list):
            raise HomeAssistantRealityError("hass_discovery_response_not_list")
        return [item for item in decoded if isinstance(item, dict)][:5000]

    async def observe(self) -> dict[str, Any] | None:
        return None

    async def apply(self, _effect: HomeAssistantEffect) -> dict[str, Any]:
        raise HomeAssistantRealityError(
            "direct_home_assistant_apply_requires_reality_reach_transaction"
        )

    async def dispatch(
        self,
        effect: HomeAssistantEffect,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> dict[str, Any]:
        if (
            command.sha256 != lease.command_sha256
            or prepared.command_sha256 != command.sha256
            or prepared.lease_sha256 != lease.sha256
            or command.parameters.get("effect_sha256") != effect.sha256
        ):
            raise HomeAssistantRealityError("hass_typed_dispatch_identity_mismatch")
        return await self._post_effect(effect, source="world_bridge:iot.home_assistant.actuate")

    async def dispatch_recovery(
        self,
        effect: HomeAssistantEffect,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> dict[str, Any]:
        if actuation is not None and actuation.command_sha256 != command.sha256:
            raise HomeAssistantRealityError("hass_recovery_identity_mismatch")
        return await self._post_effect(effect, source="world_bridge:iot.home_assistant.recover")

    async def _post_effect(
        self,
        effect: HomeAssistantEffect,
        *,
        source: str,
    ) -> dict[str, Any]:
        domain = effect.target.partition(".")[0]
        response = await ActionExecutor.request_network_transport(
            method="POST",
            url=f"{self.base}/api/services/{domain}/{effect.op}",
            headers=self._headers(json_content=True),
            data=json.dumps(
                {"entity_id": effect.target, **dict(effect.payload)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            timeout_s=8.0,
            source=source,
            read_only=False,
        )
        status = int(response.get("status_code") or 0)
        content = response.get("content") or b""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        accepted = bool(response.get("ok") and 200 <= status < 300)
        return {
            "accepted": accepted,
            "transport_completed": accepted or status > 0,
            "status": status,
            "body": str(content)[:1024],
            "error": "" if accepted else str(response.get("error") or "hass_service_refused")[:300],
        }

    async def create_adapter(self, entity_id: str) -> HomeAssistantRealityAdapter:
        state = await self.read_state(entity_id)
        return HomeAssistantRealityAdapter(self, entity_id, initial_state=state)

    async def create_sensor_adapter(
        self,
        entity_id: str,
    ) -> HomeAssistantSensorAdapter:
        state = await self.read_state(entity_id)
        return HomeAssistantSensorAdapter(self, entity_id, initial_state=state)


class HomeAssistantSensorAdapter:
    """Read-only Reality Reach adapter for one numeric or binary entity."""

    def __init__(
        self,
        transport: HomeAssistantTransport,
        entity_id: str,
        *,
        initial_state: Mapping[str, Any],
    ) -> None:
        if not isinstance(transport, HomeAssistantTransport):
            raise TypeError("transport must be HomeAssistantTransport")
        self._transport = transport
        self._entity_id = _canonical_entity_id(entity_id)
        self._profile = _sensor_profile_for_state(self._entity_id, initial_state)
        prefix = f"hass.{self._entity_id}.sensor"
        tags = ["home_assistant", "read_only_sensor"]
        tags.append(self._profile.domain_source)
        if self._profile.binary:
            tags.append("binary_sensor")
        self._declaration = ChannelDeclaration(
            channel_id=f"{prefix}.reading",
            kind=ChannelKind.SENSOR,
            observable=self._profile.observable,
            unit=self._profile.unit,
            domain=self._profile.domain,
            coupling=CouplingClass.NETWORK,
            reality_layers=(RealityLayer.EFFECTIVE, RealityLayer.AMBIENT),
            evidence_level=EvidenceLevel.P1,
            owner="core.embodiment.home_assistant_reality",
            resolution=self._profile.resolution,
            sample_rate_hz=0.5,
            max_latency_s=8.0,
            stale_after_s=15.0,
            reference_id=f"{prefix}.state_api",
            compliance_tags=tuple(tags),
            coupling_validated=True,
        )
        self._last_observation = self._reading_from_state(initial_state)

    @property
    def adapter_id(self) -> str:
        return f"hass.{self._entity_id}.sensor.adapter"

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def manifest_sha256(self) -> str:
        return _digest(
            {
                "adapter_id": self.adapter_id,
                "declaration": self._declaration.to_dict(),
                "profile": self._profile.sha256,
            }
        )

    @property
    def domain_source(self) -> str:
        return self._profile.domain_source

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self._declaration,)

    def read(self) -> tuple[ChannelReading, ...]:
        return (self._last_observation,)

    def _reading_from_state(self, state: Mapping[str, Any]) -> ChannelReading:
        (
            captured_at_ns,
            wall_clock_source,
            source_epoch,
            source_event_id,
            source_quality,
        ) = _home_assistant_source_lineage(
            state,
            entity_id=self.entity_id,
        )
        return ChannelReading(
            channel_id=self._declaration.channel_id,
            value=_sensor_value(state, self._profile),
            unit=self._declaration.unit,
            captured_at_ns=captured_at_ns,
            status=ReadingStatus.AVAILABLE,
            source=f"{self.adapter_id}.state_api",
            uncertainty=self._declaration.resolution,
            wall_clock_source=wall_clock_source,
            source_epoch=source_epoch,
            source_event_id=source_event_id,
            source_quality=source_quality,
        )

    async def refresh_readback(self) -> ChannelReading:
        try:
            state = await self._transport.read_state(self.entity_id)
            reading = self._reading_from_state(state)
        except (
            HomeAssistantRealityError,
            OSError,
            RuntimeError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            reading = ChannelReading(
                channel_id=self._declaration.channel_id,
                value=None,
                unit=self._declaration.unit,
                captured_at_ns=max(1, time.time_ns()),
                status=ReadingStatus.DEGRADED,
                source=f"{self.adapter_id}.state_api",
                error=f"{type(exc).__name__}:{exc}"[:300],
            )
        self._last_observation = reading
        return reading


class HomeAssistantRealityAdapter:
    """Manifest-bound Reality Reach adapter for one Home Assistant entity."""

    def __init__(
        self,
        transport: HomeAssistantTransport,
        entity_id: str,
        *,
        initial_state: Mapping[str, Any],
    ) -> None:
        if not isinstance(transport, HomeAssistantTransport):
            raise TypeError("transport must be HomeAssistantTransport")
        self._transport = transport
        self._entity_id = _canonical_entity_id(entity_id)
        self._profile = _profile_for_state(self._entity_id, initial_state)
        prefix = f"hass.{self._entity_id}.{self._profile.kind}"
        self._adapter_id = f"{prefix}.adapter"
        self._actuator = ChannelDeclaration(
            channel_id=f"{prefix}.command",
            kind=ChannelKind.ACTUATOR,
            observable=self._profile.observable,
            unit=self._profile.unit,
            domain=self._profile.domain,
            coupling=CouplingClass.NETWORK,
            reality_layers=(RealityLayer.EFFECTIVE,),
            evidence_level=EvidenceLevel.P1,
            owner="core.embodiment.home_assistant_reality",
            stale_after_s=15.0,
            compliance_tags=("home_assistant", "typed_actuation"),
            coupling_validated=True,
        )
        self._observation = ChannelDeclaration(
            channel_id=f"{prefix}.readback",
            kind=ChannelKind.SENSOR,
            observable=self._profile.observable,
            unit=self._profile.unit,
            domain=self._profile.domain,
            coupling=CouplingClass.NETWORK,
            reality_layers=(RealityLayer.EFFECTIVE,),
            evidence_level=EvidenceLevel.P1,
            owner="core.embodiment.home_assistant_reality",
            resolution=self._profile.tolerance,
            sample_rate_hz=1.0,
            max_latency_s=8.0,
            stale_after_s=15.0,
            reference_id=f"{prefix}.state_api",
            compliance_tags=("home_assistant", "transport_distinct_readback"),
            coupling_validated=True,
        )
        self._capability = ActuatorCapability(
            adapter_id=self._adapter_id,
            channel_id=self._actuator.channel_id,
            reversibility=Reversibility.REVERSIBLE,
            magnitude_domain=self._profile.domain,
            max_commands_per_minute=self._profile.max_commands_per_minute,
            observation_channels=(self._observation.channel_id,),
            required_permissions=("environment.physical", "network.local"),
            failure_modes=(
                "transport_failure",
                "readback_mismatch",
                "common_driver_false_confirmation",
            ),
            cooldown_s=self._profile.cooldown_s,
            watchdog_timeout_s=10.0,
            exclusive=True,
            supports_cancel=True,
            supports_safe_state=self._profile.kind == "power",
            supports_rollback=True,
            compensation_action="restore_previous_home_assistant_state",
        )
        self._cached_state = dict(initial_state)
        self._last_observation = self._reading_from_state(initial_state)
        self._prepared: dict[str, dict[str, Any]] = {}
        self._dispatch_times: deque[float] = deque()
        self._last_dispatch_monotonic = 0.0
        self._lock = checked_async_lock("home_assistant_reality")

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def entity_id(self) -> str:
        return self._entity_id

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self._actuator, self._observation)

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]:
        return (self._capability,)

    def read(self) -> tuple[ChannelReading, ...]:
        actuator = ChannelReading(
            channel_id=self._actuator.channel_id,
            value=None,
            unit=self._actuator.unit,
            captured_at_ns=max(1, time.time_ns()),
            status=ReadingStatus.UNAVAILABLE,
            source=f"{self.adapter_id}.command_api",
            error="actuator_channels_do_not_self_report_effects",
        )
        return (actuator, self._last_observation)

    def _reading_from_state(self, state: Mapping[str, Any]) -> ChannelReading:
        (
            captured_at_ns,
            wall_clock_source,
            source_epoch,
            source_event_id,
            source_quality,
        ) = _home_assistant_source_lineage(
            state,
            entity_id=self.entity_id,
        )
        return ChannelReading(
            channel_id=self._observation.channel_id,
            value=_primary_value(state, self._profile),
            unit=self._observation.unit,
            captured_at_ns=captured_at_ns,
            status=ReadingStatus.AVAILABLE,
            source=f"{self.adapter_id}.state_api",
            uncertainty=self._observation.resolution,
            wall_clock_source=wall_clock_source,
            source_epoch=source_epoch,
            source_event_id=source_event_id,
            source_quality=source_quality,
        )

    async def refresh_readback(self) -> ChannelReading:
        try:
            state = await self._transport.read_state(self.entity_id)
            reading = self._reading_from_state(state)
            self._cached_state = dict(state)
        except (
            HomeAssistantRealityError,
            OSError,
            RuntimeError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            reading = ChannelReading(
                channel_id=self._observation.channel_id,
                value=None,
                unit=self._observation.unit,
                captured_at_ns=max(1, time.time_ns()),
                status=ReadingStatus.DEGRADED,
                source=f"{self.adapter_id}.state_api",
                error=f"{type(exc).__name__}:{exc}"[:300],
            )
        self._last_observation = reading
        return reading

    async def compile_effect(
        self,
        effect: HomeAssistantEffect,
        *,
        inventory_sha256: str,
        deadline_s: float,
        idempotency_key: str,
        source: str,
    ) -> ActuationCommand:
        effect = _validated_effect(effect, self._profile)
        if effect.target != self.entity_id:
            raise HomeAssistantRealityError("home_assistant_effect_entity_mismatch")
        reading = await self.refresh_readback()
        if reading.status != ReadingStatus.AVAILABLE:
            raise HomeAssistantRealityError("home_assistant_compile_readback_unavailable")
        target = _effect_target(effect, self._profile)
        identifier = str(idempotency_key or "").strip()
        if not _IDENTIFIER.fullmatch(identifier):
            identifier = f"hass.idem.{_digest(identifier).removeprefix('sha256:')[:32]}"
        bounded_deadline = max(0.1, min(float(deadline_s), 120.0))
        return ActuationCommand(
            command_id=f"hass.command.{uuid.uuid4().hex}",
            request_id=f"hass.request.{uuid.uuid4().hex}",
            adapter_id=self.adapter_id,
            channel_id=self._actuator.channel_id,
            observable=self._actuator.observable,
            unit=self._actuator.unit,
            target=target,
            tolerance=self._profile.tolerance,
            magnitude=target,
            idempotency_key=identifier,
            inventory_sha256=inventory_sha256,
            deadline_ns=time.time_ns() + int(bounded_deadline * 1_000_000_000),
            safe_envelope=self._profile.domain,
            parameters={
                "entity_id": self.entity_id,
                "operation": effect.op,
                "payload": dict(effect.payload),
                "reason": effect.reason,
                "source": str(source or "iot_bridge")[:128],
                "effect_sha256": effect.sha256,
                "profile_sha256": self._profile.sha256,
            },
            preconditions=("entity_readback_available", "entity_state_stable"),
            expected_effects=("home_assistant_state_api_matches",),
            abort_predicates=("entity_state_changed_before_dispatch",),
        )

    def _effect_from_command(self, command: ActuationCommand) -> HomeAssistantEffect:
        if (
            command.adapter_id != self.adapter_id
            or command.channel_id != self._actuator.channel_id
            or command.observable != self._actuator.observable
            or command.unit != self._actuator.unit
            or command.parameters.get("entity_id") != self.entity_id
            or command.parameters.get("profile_sha256") != self._profile.sha256
        ):
            raise HomeAssistantRealityError("home_assistant_command_identity_mismatch")
        payload = command.parameters.get("payload")
        if not isinstance(payload, Mapping):
            raise HomeAssistantRealityError("home_assistant_command_payload_invalid")
        effect = _validated_effect(
            HomeAssistantEffect(
                self.entity_id,
                str(command.parameters.get("operation") or ""),
                payload,
                str(command.parameters.get("reason") or ""),
            ),
            self._profile,
        )
        if (
            command.parameters.get("effect_sha256") != effect.sha256
            or command.target != _effect_target(effect, self._profile)
            or command.magnitude != command.target
        ):
            raise HomeAssistantRealityError("home_assistant_command_compilation_mismatch")
        return effect

    @staticmethod
    def _lease_valid(command: ActuationCommand, lease: ActuationLease) -> bool:
        return bool(
            lease.command_sha256 == command.sha256
            and lease.adapter_id == command.adapter_id
            and lease.is_valid(
                now_ns=time.time_ns(),
                monotonic_now_ns=time.monotonic_ns(),
                session_id=lease.session_id,
            )
        )

    def _check_rate_limit(self, *, now: float) -> None:
        while self._dispatch_times and now - self._dispatch_times[0] >= 60.0:
            self._dispatch_times.popleft()
        if len(self._dispatch_times) >= self._capability.max_commands_per_minute:
            raise HomeAssistantRealityError("home_assistant_command_rate_limited")
        if now - self._last_dispatch_monotonic < self._capability.cooldown_s:
            raise HomeAssistantRealityError("home_assistant_command_cooldown_active")

    async def prepare(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
    ) -> PreparedActuation:
        effect = self._effect_from_command(command)
        if not self._lease_valid(command, lease):
            raise HomeAssistantRealityError("home_assistant_lease_invalid")
        async with self._lock:
            self._check_rate_limit(now=time.monotonic())
            state = await self._transport.read_state(self.entity_id)
            reading = self._reading_from_state(state)
            self._cached_state = dict(state)
            self._last_observation = reading
            projection = _state_projection(state, effect)
            rollback = _rollback_effect(state, effect, self._profile)
            precondition_sha256 = _digest(projection)
            context = {
                "precondition_sha256": precondition_sha256,
                "rollback_effect": rollback.to_dict(),
                "rollback_sha256": rollback.sha256,
            }
            self._prepared[command.sha256] = context
            while len(self._prepared) > 256:
                self._prepared.pop(next(iter(self._prepared)))
        return PreparedActuation(
            preparation_id=f"hass.prepare.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            lease_sha256=lease.sha256,
            adapter_id=self.adapter_id,
            capability_sha256=self._capability.sha256,
            precondition_sha256=precondition_sha256,
            rollback_token_sha256=rollback.sha256,
            prepared_at_ns=time.time_ns(),
        )

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt:
        effect = self._effect_from_command(command)
        if (
            prepared.command_sha256 != command.sha256
            or prepared.lease_sha256 != lease.sha256
            or prepared.adapter_id != self.adapter_id
            or not self._lease_valid(command, lease)
        ):
            raise HomeAssistantRealityError("home_assistant_preparation_invalid")
        async with self._lock:
            context = self._prepared.get(command.sha256)
            if not isinstance(context, Mapping):
                raise HomeAssistantRealityError("home_assistant_preparation_context_missing")
            state = await self._transport.read_state(self.entity_id)
            if _digest(_state_projection(state, effect)) != prepared.precondition_sha256:
                raise HomeAssistantRealityError(
                    "home_assistant_state_changed_before_dispatch"
                )
            now = time.monotonic()
            self._check_rate_limit(now=now)
            result = await self._transport.dispatch(effect, command, lease, prepared)
            executed = result.get("accepted") is True
            if executed:
                self._dispatch_times.append(now)
                self._last_dispatch_monotonic = now
        return ActuationReceipt(
            receipt_id=f"hass.actuate.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.EXECUTED if executed else ActuationState.FAILED,
            accepted=True,
            transport_completed=bool(result.get("transport_completed")),
            executed=executed,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest(dict(result)),
        )

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt:
        effect = self._effect_from_command(command)
        reading = self._last_observation
        matched = False
        readback_failed = False
        for attempt in range(3):
            try:
                state = await self._transport.read_state(self.entity_id)
                reading = self._reading_from_state(state)
                self._cached_state = dict(state)
                self._last_observation = reading
                readback_failed = False
                matched = state_matches_effect(state, effect)
            except (
                HomeAssistantRealityError,
                OSError,
                RuntimeError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                readback_failed = True
                reading = ChannelReading(
                    channel_id=self._observation.channel_id,
                    value=None,
                    unit=self._observation.unit,
                    captured_at_ns=max(1, time.time_ns()),
                    status=ReadingStatus.DEGRADED,
                    source=f"{self.adapter_id}.state_api",
                    error=f"{type(exc).__name__}:{exc}"[:300],
                )
                self._last_observation = reading
                matched = False
            if matched:
                break
            if attempt < 2:
                await asyncio.sleep(0.2)
        target_error = (
            abs(float(reading.value) - command.target)
            if reading.value is not None
            else None
        )
        independent_readback = (
            not readback_failed and reading.status == ReadingStatus.AVAILABLE
        )
        verified = bool(
            actuation.executed
            and matched
            and independent_readback
            and target_error is not None
            and target_error <= command.tolerance
        )
        return EffectReceipt(
            receipt_id=f"hass.effect.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256,
            observation_channel_id=self._observation.channel_id,
            observation_sha256=reading.sha256,
            state=ActuationState.EFFECT_VERIFIED if verified else ActuationState.FAILED,
            target_error=target_error,
            independently_observed=independent_readback,
            recorded_at_ns=time.time_ns(),
        )

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt:
        self._effect_from_command(command)
        self._prepared.pop(command.sha256, None)
        return ActuationReceipt(
            receipt_id=f"hass.cancel.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256 if prepared is not None else command.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.CANCELLED,
            accepted=False,
            transport_completed=False,
            executed=False,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest({"cancelled_before_dispatch": True}),
        )

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        effect = self._effect_from_command(command)
        if self._profile.kind != "power":
            return self._indeterminate_recovery(command, actuation)
        safe_effect = HomeAssistantEffect(
            effect.target,
            "turn_off",
            {},
            "reality_reach_safe_state",
        )
        return await self._recover(
            command,
            actuation,
            effect=safe_effect,
            success_state=ActuationState.SAFE_STATE,
        )

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt:
        self._effect_from_command(command)
        context = self._prepared.get(command.sha256)
        rollback = context.get("rollback_effect") if isinstance(context, Mapping) else None
        if not isinstance(rollback, Mapping):
            return self._indeterminate_recovery(command, actuation)
        rollback_payload = rollback.get("payload")
        if not isinstance(rollback_payload, Mapping):
            rollback_payload = {}
        effect = _validated_effect(
            HomeAssistantEffect(
                str(rollback.get("target") or ""),
                str(rollback.get("op") or ""),
                cast(Mapping[str, Any], rollback_payload),
                str(rollback.get("reason") or ""),
            ),
            self._profile,
        )
        return await self._recover(
            command,
            actuation,
            effect=effect,
            success_state=ActuationState.ROLLED_BACK,
        )

    def _indeterminate_recovery(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        return RollbackReceipt(
            receipt_id=f"hass.indeterminate.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256 if actuation is not None else command.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.INDETERMINATE,
            safe_state_observation_sha256=self._last_observation.sha256,
            independently_observed=False,
            recorded_at_ns=time.time_ns(),
        )

    async def _recover(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
        *,
        effect: HomeAssistantEffect,
        success_state: ActuationState,
    ) -> RollbackReceipt:
        async with self._lock:
            result = await self._transport.dispatch_recovery(effect, command, actuation)
            try:
                state = await self._transport.read_state(self.entity_id)
                reading = self._reading_from_state(state)
                self._cached_state = dict(state)
                self._last_observation = reading
                observed = result.get("accepted") is True and state_matches_effect(state, effect)
            except (
                HomeAssistantRealityError,
                OSError,
                RuntimeError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                reading = self._last_observation
                observed = False
            self._prepared.pop(command.sha256, None)
        return RollbackReceipt(
            receipt_id=f"hass.{success_state.value}.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256 if actuation is not None else command.sha256,
            adapter_id=self.adapter_id,
            state=success_state if observed else ActuationState.INDETERMINATE,
            safe_state_observation_sha256=reading.sha256,
            independently_observed=observed,
            recorded_at_ns=time.time_ns(),
        )


__all__ = [
    "HomeAssistantEffect",
    "HomeAssistantRealityAdapter",
    "HomeAssistantRealityError",
    "HomeAssistantSensorAdapter",
    "HomeAssistantTransport",
    "state_matches_effect",
]
