"""Manifest-bound MQTT sensing and verified scalar actuation for Reality Reach."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import ssl
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

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
from core.runtime.lockdep import checked_lock

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_TOPIC_FORBIDDEN = frozenset({"#", "+", "\x00"})


class MQTTConnectorError(RuntimeError):
    """MQTT configuration or broker evidence violated its declared contract."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _topic(value: object, *, name: str) -> str:
    topic = str(value or "").strip()
    if (
        not topic
        or len(topic.encode("utf-8")) > 1024
        or any(token in topic for token in _TOPIC_FORBIDDEN)
    ):
        raise ValueError(f"{name} must be a bounded concrete MQTT topic")
    return topic


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _allow_plaintext() -> bool:
    from core.runtime.flags import FlagKind, declare

    return str(
        declare(
            "AURA_MQTT_ALLOW_PLAINTEXT",
            kind=FlagKind.STRING,
            default="",
            description="Permit plaintext MQTT broker connections",
            owner="core.embodiment.mqtt_connector",
        ).value()
    ).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class MQTTResourceSpec:
    resource_id: str
    device_id: str
    observable: str
    unit: str
    state_topic: str
    domain: NumericDomain
    resolution: float
    command_topic: str = ""
    safe_value: float | None = None
    tolerance: float | None = None
    encoding: str = "number"
    value_field: str = "value"
    device_reported_feedback: bool = False
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("resource_id", "device_id", "observable", "unit"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        object.__setattr__(self, "state_topic", _topic(self.state_topic, name="state_topic"))
        command_topic = str(self.command_topic or "").strip()
        if command_topic:
            command_topic = _topic(command_topic, name="command_topic")
        object.__setattr__(self, "command_topic", command_topic)
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
        if self.safe_value is not None:
            safe = _finite(self.safe_value, name="safe_value")
            if not self.domain.contains(safe):
                raise ValueError("safe_value lies outside the domain")
            object.__setattr__(self, "safe_value", safe)
        if self.encoding not in {"number", "on_off", "json"}:
            raise ValueError("encoding must be number, on_off, or json")
        object.__setattr__(self, "value_field", _identifier(self.value_field, name="value_field"))
        if not isinstance(self.device_reported_feedback, bool):
            raise TypeError("device_reported_feedback must be boolean")
        if self.command_topic and (
            self.command_topic == self.state_topic or not self.device_reported_feedback
        ):
            raise ValueError(
                "writable MQTT resources require a distinct device-reported state topic"
            )
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("MQTT timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def writable(self) -> bool:
        return bool(self.command_topic)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "device_id": self.device_id,
            "observable": self.observable,
            "unit": self.unit,
            "state_topic": self.state_topic,
            "domain": self.domain.to_dict(),
            "resolution": self.resolution,
            "command_topic": self.command_topic,
            "safe_value": self.safe_value,
            "tolerance": self.tolerance,
            "encoding": self.encoding,
            "value_field": self.value_field,
            "device_reported_feedback": self.device_reported_feedback,
            "max_commands_per_minute": self.max_commands_per_minute,
            "cooldown_s": self.cooldown_s,
            "stale_after_s": self.stale_after_s,
        }

    def decode(self, payload: bytes) -> float:
        if len(payload) > 64 * 1024:
            raise MQTTConnectorError("mqtt_state_payload_too_large")
        text = payload.decode("utf-8", errors="strict").strip()
        if self.encoding == "json":
            decoded = json.loads(text)
            if not isinstance(decoded, Mapping) or self.value_field not in decoded:
                raise MQTTConnectorError("mqtt_json_value_field_missing")
            value = decoded[self.value_field]
        elif self.encoding == "on_off":
            normalized = text.upper()
            if normalized not in {"ON", "OFF"}:
                raise MQTTConnectorError("mqtt_binary_state_invalid")
            value = 1.0 if normalized == "ON" else 0.0
        else:
            value = text
        number = _finite(value, name="mqtt state")
        if not self.domain.contains(number):
            raise MQTTConnectorError("mqtt_state_outside_manifest_domain")
        return number

    def encode(self, value: float) -> bytes:
        number = _finite(value, name="mqtt command")
        if not self.domain.contains(number):
            raise MQTTConnectorError("mqtt_command_outside_manifest_domain")
        if self.encoding == "json":
            return canonical_json({self.value_field: number})
        if self.encoding == "on_off":
            return b"ON" if number >= 0.5 else b"OFF"
        return repr(number).encode("ascii")


def parse_mqtt_resource_manifest(raw: object) -> tuple[MQTTResourceSpec, ...]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MQTTConnectorError("mqtt_resource_manifest_invalid_json") from exc
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MQTTConnectorError("mqtt_resource_manifest_must_be_a_list")
    if not 1 <= len(raw) <= 512:
        raise MQTTConnectorError("mqtt_resource_manifest_size_invalid")
    resources: list[MQTTResourceSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise MQTTConnectorError("mqtt_resource_manifest_entry_invalid")
        domain = NumericDomain(
            _finite(item.get("minimum"), name="minimum"),
            _finite(item.get("maximum"), name="maximum"),
        )
        resources.append(
            MQTTResourceSpec(
                resource_id=str(item.get("resource_id") or ""),
                device_id=str(item.get("device_id") or ""),
                observable=str(item.get("observable") or ""),
                unit=str(item.get("unit") or ""),
                state_topic=str(item.get("state_topic") or ""),
                domain=domain,
                resolution=_finite(item.get("resolution"), name="resolution"),
                command_topic=str(item.get("command_topic") or ""),
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
                encoding=str(item.get("encoding") or "number").strip().lower(),
                value_field=str(item.get("value_field") or "value"),
                device_reported_feedback=item.get("device_reported_feedback") is True,
                max_commands_per_minute=int(item.get("max_commands_per_minute") or 12),
                cooldown_s=_finite(item.get("cooldown_s") or 0.0, name="cooldown_s"),
                stale_after_s=_finite(item.get("stale_after_s") or 30.0, name="stale_after_s"),
            )
        )
    if len({item.resource_id for item in resources}) != len(resources):
        raise MQTTConnectorError("mqtt_resource_id_duplicate")
    if len({item.state_topic for item in resources}) != len(resources):
        raise MQTTConnectorError("mqtt_state_topic_duplicate")
    return tuple(sorted(resources, key=lambda item: item.resource_id))


@runtime_checkable
class MQTTScalarTransport(Protocol):
    @property
    def transport_id(self) -> str: ...

    @property
    def broker_identity_sha256(self) -> str: ...

    async def read_scalar(self, resource_id: str) -> ScalarSample: ...

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult: ...


class PahoMQTTScalarTransport:
    """TLS MQTT v5 transport with retained device-state readback."""

    transport_id = "mqtt.v5"

    def __init__(self, resources: tuple[MQTTResourceSpec, ...]) -> None:
        broker_url = str(os.getenv("AURA_MQTT_BROKER_URL") or "").strip()
        parsed = urllib.parse.urlparse(broker_url)
        if (
            parsed.scheme not in {"mqtt", "mqtts"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise MQTTConnectorError("mqtt_broker_url_invalid")
        if parsed.scheme == "mqtt" and not _allow_plaintext():
            raise MQTTConnectorError("mqtt_plaintext_requires_explicit_opt_in")
        installation = _identifier(
            os.getenv("AURA_MQTT_INSTALLATION_ID") or "",
            name="AURA_MQTT_INSTALLATION_ID",
        )
        self._broker_identity = _digest(
            {
                "scheme": parsed.scheme,
                "host": parsed.hostname.lower(),
                "port": parsed.port or (8883 if parsed.scheme == "mqtts" else 1883),
                "installation": installation,
            }
        )
        self._host = parsed.hostname
        self._port = parsed.port or (8883 if parsed.scheme == "mqtts" else 1883)
        self._tls = parsed.scheme == "mqtts"
        self._resources = {item.resource_id: item for item in resources}
        self._topic_resources = {item.state_topic: item.resource_id for item in resources}
        self._samples: dict[str, ScalarSample] = {}
        self._events = {item.resource_id: threading.Event() for item in resources}
        self._lock = checked_lock("mqtt_transport.samples", reentrant=True)
        self._lifecycle_lock = checked_lock("mqtt_transport.lifecycle", reentrant=True)
        self._client: Any | None = None
        self._mqtt: Any | None = None
        self._started = False
        self._connected = False
        self._sequence = 0

    @property
    def broker_identity_sha256(self) -> str:
        return self._broker_identity

    def _start_sync(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return
            self._start_unlocked()

    def _start_unlocked(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise MQTTConnectorError(
                "mqtt_transport_dependency_missing:paho-mqtt"
            ) from exc
        client_id = f"aura-{self._broker_identity.removeprefix('sha256:')[:20]}"
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv5,
        )
        username = str(os.getenv("AURA_MQTT_USERNAME") or "").strip()
        password = str(os.getenv("AURA_MQTT_PASSWORD") or "")
        if username:
            client.username_pw_set(username, password)
        if self._tls:
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

        def on_connect(
            callback_client: Any,
            _userdata: Any,
            _flags: Any,
            reason_code: Any,
            _properties: Any,
        ) -> None:
            self._connected = not bool(reason_code.is_failure)
            if not self._connected:
                return
            for topic in self._topic_resources:
                callback_client.subscribe(topic, qos=1)

        def on_disconnect(
            _client: Any,
            _userdata: Any,
            _flags: Any,
            _reason_code: Any,
            _properties: Any,
        ) -> None:
            self._connected = False

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            resource_id = self._topic_resources.get(str(message.topic))
            if resource_id is None:
                return
            spec = self._resources[resource_id]
            try:
                value = spec.decode(bytes(message.payload))
            # not a failure: a payload this spec cannot decode is not a reading for this
            # resource.
            except (UnicodeError, MQTTConnectorError, TypeError, ValueError):
                return
            with self._lock:
                self._sequence += 1
                self._samples[resource_id] = ScalarSample(
                    value=value,
                    captured_at_ns=max(1, time.time_ns()),
                    source_event_id=_digest(
                        {
                            "broker": self._broker_identity,
                            "topic": message.topic,
                            "payload_sha256": _digest(bytes(message.payload).hex()),
                            "sequence": self._sequence,
                        }
                    ),
                    quality="device_reported_broker_timestamp",
                    source_epoch=self._broker_identity,
                    source_sequence=self._sequence,
                )
                self._events[resource_id].set()

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.connect_async(self._host, self._port, keepalive=60)
        client.loop_start()
        self._client = client
        self._mqtt = mqtt
        self._started = True

    async def _ensure_started(self) -> None:
        await asyncio.to_thread(self._start_sync)

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        if resource_id not in self._resources:
            raise MQTTConnectorError("mqtt_resource_not_bound")
        await self._ensure_started()
        event = self._events[resource_id]
        if not await asyncio.to_thread(event.wait, 8.0):
            raise TimeoutError("mqtt_state_sample_timeout")
        with self._lock:
            sample = self._samples.get(resource_id)
        if sample is None:
            raise MQTTConnectorError("mqtt_state_sample_missing")
        if time.time_ns() - sample.captured_at_ns > int(
            self._resources[resource_id].stale_after_s * 1_000_000_000
        ):
            event.clear()
            if self._client is not None and self._connected:
                self._client.subscribe(self._resources[resource_id].state_topic, qos=1)
            raise TimeoutError("mqtt_state_sample_stale")
        return sample

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        spec = self._resources.get(resource_id)
        if spec is None or not spec.command_topic:
            raise MQTTConnectorError("mqtt_resource_not_writable")
        await self._ensure_started()
        if self._client is None or not self._connected:
            raise MQTTConnectorError("mqtt_broker_not_connected")
        payload = spec.encode(value)
        properties = None
        if self._mqtt is not None:
            properties = self._mqtt.Properties(self._mqtt.PacketTypes.PUBLISH)
            properties.UserProperty = [
                ("aura-idempotency-key", idempotency_key),
                ("aura-recovery", "true" if recovery else "false"),
            ]
        info = self._client.publish(
            spec.command_topic,
            payload=payload,
            qos=1,
            retain=False,
            properties=properties,
        )
        published = await asyncio.to_thread(info.wait_for_publish, 8.0)
        accepted = bool(published is None or published) and bool(info.is_published())
        return ScalarWriteResult(
            accepted=accepted,
            transport_completed=accepted,
            receipt={
                "protocol": self.transport_id,
                "resource_id": resource_id,
                "broker_identity_sha256": self._broker_identity,
                "message_id": int(info.mid),
                "qos": 1,
                "retained": False,
                "recovery": recovery,
                "idempotency_sha256": _digest(idempotency_key),
            },
        )

    async def stop(self) -> None:
        def _stop_sync() -> None:
            with self._lifecycle_lock:
                client = self._client
                self._client = None
                self._mqtt = None
                self._started = False
                self._connected = False
            if client is not None:
                try:
                    client.disconnect()
                finally:
                    client.loop_stop()

        await asyncio.to_thread(_stop_sync)


class MQTTConnector:
    """Expose manifest-bound MQTT resources as attachable physical channels."""

    connector_id = "mqtt.manifest"

    def __init__(
        self,
        transport: MQTTScalarTransport,
        resources: tuple[MQTTResourceSpec, ...],
        *,
        candidate_ttl_s: float = 180.0,
    ) -> None:
        if not isinstance(transport, MQTTScalarTransport):
            raise TypeError("transport must satisfy MQTTScalarTransport")
        if not resources:
            raise ValueError("resources must not be empty")
        self._transport = transport
        self._resources = {item.resource_id: item for item in resources}
        self._ttl_s = max(30.0, min(float(candidate_ttl_s), 3600.0))

    def _profile(self, spec: MQTTResourceSpec) -> ScalarResourceProfile:
        return ScalarResourceProfile(
            resource_id=spec.resource_id,
            observable=spec.observable,
            unit=spec.unit,
            domain=spec.domain,
            resolution=spec.resolution,
            writable=spec.writable,
            physical_identity_sha256=_digest(
                {
                    "broker": self._transport.broker_identity_sha256,
                    "device_id": spec.device_id,
                    "resource_id": spec.resource_id,
                }
            ),
            owner="core.embodiment.mqtt_connector",
            protocol="mqtt",
            safe_value=spec.safe_value,
            tolerance=spec.tolerance,
            max_commands_per_minute=spec.max_commands_per_minute,
            cooldown_s=spec.cooldown_s,
            stale_after_s=spec.stale_after_s,
            readback_distinct_from_command=spec.device_reported_feedback,
        )

    async def discover(self) -> tuple[DeviceCandidate, ...]:
        candidates: list[DeviceCandidate] = []
        now_ns = max(1, time.time_ns())
        for spec in self._resources.values():
            try:
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
                    "broker_identity_sha256": self._transport.broker_identity_sha256,
                }
            )
            access = (
                (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
                if spec.writable
                else (AttachmentAccess.OBSERVE,)
            )
            candidates.append(
                DeviceCandidate(
                    candidate_id=(
                        "mqtt.candidate."
                        + manifest.removeprefix("sha256:")[:32]
                    ),
                    connector_id=self.connector_id,
                    device_id=f"mqtt.{spec.device_id}.{spec.resource_id}",
                    display_name=f"{spec.device_id}: {spec.observable}"[:160],
                    transport=self._transport.transport_id,
                    identity_fingerprint=profile.physical_identity_sha256,
                    manifest_sha256=manifest,
                    access=access,
                    discovered_at_ns=now_ns,
                    expires_at_ns=now_ns + int(self._ttl_s * 1_000_000_000),
                    persistent_identity=True,
                    proposal_salience=0.4,
                    metadata={
                        "resource_id": spec.resource_id,
                        "device_id": spec.device_id,
                        "spec_sha256": spec.sha256,
                        "profile_sha256": profile.sha256,
                        "control_available": spec.writable,
                        "independent_readback": spec.device_reported_feedback,
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
            raise ValueError("mqtt_candidate_connector_mismatch")
        requested = set(access)
        if not requested or not requested.issubset(set(candidate.access)):
            raise PermissionError("mqtt_attachment_access_not_declared")
        if AttachmentAccess.CONTROL in requested and AttachmentAccess.OBSERVE not in requested:
            raise PermissionError("mqtt_control_requires_observation")
        resource_id = str(candidate.metadata.get("resource_id") or "")
        spec = self._resources.get(resource_id)
        if spec is None:
            raise LookupError("mqtt_candidate_resource_missing")
        current = next(
            (item for item in await self.discover() if item.candidate_id == candidate.candidate_id),
            None,
        )
        if current is None or (
            current.identity_fingerprint != candidate.identity_fingerprint
            or current.manifest_sha256 != candidate.manifest_sha256
        ):
            raise RuntimeError("mqtt_candidate_changed_before_attachment")
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


def build_configured_mqtt_connector() -> MQTTConnector:
    raw = str(os.getenv("AURA_MQTT_RESOURCES_JSON") or "").strip()
    if not raw:
        raise MQTTConnectorError("mqtt_resource_manifest_missing")
    resources = parse_mqtt_resource_manifest(raw)
    return MQTTConnector(PahoMQTTScalarTransport(resources), resources)


__all__ = [
    "MQTTConnector",
    "MQTTConnectorError",
    "MQTTResourceSpec",
    "MQTTScalarTransport",
    "PahoMQTTScalarTransport",
    "build_configured_mqtt_connector",
    "parse_mqtt_resource_manifest",
]
