"""Bounded sensory routing from physical channels into Aura's cognition.

Reality Reach owns physical declarations and metrology.  This module turns
those readings into a controllable exteroceptive stream without allowing a
fast sensor, a large installation, or a failing device to monopolize the event
loop or Aura's attention.  Raw device payloads remain with their adapters;
only bounded scalar claims and provenance cross the cognitive boundary.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from core.perception.multimodal_sync import (
    Calibration,
    Modality,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.digital_twin import RealityDigitalTwinGraph, TwinReceipt
from core.reality_reach.historian import (
    HistorianDisposition,
    HistorianError,
    HistorianHeadSnapshot,
    RealityHistorian,
)
from core.reality_reach.live import (
    ChannelReading,
    ReadingStatus,
    RealityReachService,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock, checked_semaphore
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SELECTOR = re.compile(r"^(?:\*|[a-z0-9][a-z0-9_.:-]{0,127}\*?)$")
_AVAILABLE = frozenset({ReadingStatus.AVAILABLE, ReadingStatus.SIMULATED})
_OBSERVATION_SCHEMA = "aura.reality-observation.v2"
_LEGACY_OBSERVATION_SCHEMA = "aura.reality-observation.v1"
_HISTORIAN_EVIDENCE_SCHEMA = "aura.reality-historian-evidence.v1"
_HISTORIAN_EVIDENCE_KEYS = frozenset(
    {
        "alarm_codes",
        "binding_sha256",
        "order_basis",
        "order_gap",
        "quality",
        "reason",
        "record_id",
        "schema",
    }
)
_SQLITE_INT_MAX = (1 << 63) - 1
_BACKFILL_SUBSCRIPTION_ID = "reality.backfill.current_head"
_BACKFILL_PAGE_SIZE = 512
_BACKFILL_MAX_PAGES = 256


@dataclass(frozen=True, slots=True)
class _RequiredSink:
    sink_id: str
    dependency_key: str
    callable_attribute: str
    health_attribute: str
    health_contract: str


_REQUIRED_SINK_REGISTRY = (
    _RequiredSink(
        sink_id="digital_twin",
        dependency_key="reality_digital_twin",
        callable_attribute="observe_observation",
        health_attribute="health_snapshot",
        health_contract="ready_flag",
    ),
    _RequiredSink(
        sink_id="multimodal",
        dependency_key="multimodal_synchronizer",
        callable_attribute="ingest",
        health_attribute="get_status",
        health_contract="bounded_multimodal_status",
    ),
    _RequiredSink(
        sink_id="advanced_cognition",
        dependency_key="advanced_cognition",
        callable_attribute="observe_state",
        health_attribute="health_report",
        health_contract="ok_flag",
    ),
)


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, _finite(value, name="value")))


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _stored_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _stored_sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _stored_int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = _SQLITE_INT_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if value > maximum:
        raise ValueError(f"{name} exceeds the signed 64-bit storage contract")
    return int(value)


def _observation_identifier(
    *,
    adapter_id: str,
    channel_id: str,
    reading_sha256: str,
    received_monotonic_ns: int,
) -> str:
    digest = _digest(
        {
            "adapter_id": adapter_id,
            "channel_id": channel_id,
            "reading_sha256": reading_sha256,
            "received_monotonic_ns": received_monotonic_ns,
        }
    )
    return f"reality.obs.{digest.removeprefix('sha256:')[:32]}"


def _backfill_received_monotonic_ns(
    *,
    record_id: str,
    source_sha256: str,
    target_binding_sha256: str,
) -> int:
    digest = _digest(
        {
            "record_id": record_id,
            "source_sha256": source_sha256,
            "target_binding_sha256": target_binding_sha256,
        }
    )
    return max(1, int(digest.removeprefix("sha256:")[:15], 16))


@dataclass(frozen=True, slots=True)
class ObservationSubscription:
    """One attention policy selected by Aura or a trusted runtime owner.

    ``selector`` is an exact channel id, a prefix ending in ``*``, or ``*``.
    The most specific active selector wins.  Expiring focus subscriptions let
    cognition temporarily inspect a device at higher cadence without changing
    the conservative background budget.
    """

    subscription_id: str
    selector: str = "*"
    max_rate_hz: float = 0.5
    min_delta: float = 0.0
    min_salience: float = 0.12
    enabled: bool = True
    expires_monotonic: float | None = None
    retain_for_memory: bool = False

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.subscription_id):
            raise ValueError("subscription_id must be a canonical identifier")
        if not _SELECTOR.fullmatch(self.selector):
            raise ValueError("selector must be an exact channel, prefix*, or *")
        rate = _finite(self.max_rate_hz, name="max_rate_hz")
        delta = _finite(self.min_delta, name="min_delta")
        salience = _clamp01(self.min_salience)
        if not 0.01 <= rate <= 20.0:
            raise ValueError("max_rate_hz must lie inside [0.01, 20]")
        if delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.expires_monotonic is not None:
            expires = _finite(self.expires_monotonic, name="expires_monotonic")
            if expires <= 0.0:
                raise ValueError("expires_monotonic must be positive")
            object.__setattr__(self, "expires_monotonic", expires)
        object.__setattr__(self, "max_rate_hz", rate)
        object.__setattr__(self, "min_delta", delta)
        object.__setattr__(self, "min_salience", salience)

    def matches(self, channel_id: str, *, now: float) -> bool:
        if not self.enabled:
            return False
        if self.expires_monotonic is not None and now >= self.expires_monotonic:
            return False
        if self.selector == "*":
            return True
        if self.selector.endswith("*"):
            return channel_id.startswith(self.selector[:-1])
        return channel_id == self.selector

    @property
    def specificity(self) -> tuple[int, int]:
        if self.selector == "*":
            return (0, 0)
        if self.selector.endswith("*"):
            return (1, len(self.selector))
        return (2, len(self.selector))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "selector": self.selector,
            "max_rate_hz": self.max_rate_hz,
            "min_delta": self.min_delta,
            "min_salience": self.min_salience,
            "enabled": self.enabled,
            "expires_monotonic": self.expires_monotonic,
            "retain_for_memory": self.retain_for_memory,
        }


@dataclass(frozen=True, slots=True)
class RealityObservation:
    observation_id: str
    adapter_id: str
    declaration: ChannelDeclaration
    reading: ChannelReading
    salience: float
    received_at_ns: int
    received_monotonic_ns: int
    subscription_id: str
    historian_record_id: str = ""
    historian_quality: str = "ephemeral"
    historian_order_basis: str = "ephemeral"
    historian_order_gap: bool = False
    historian_alarm_codes: tuple[str, ...] = ()
    twin_id: str = ""
    attachment_generation: int = 0
    attachment_bound_at_ns: int = 0
    topology_revision: int = 0

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.observation_id):
            raise ValueError("observation_id must be a canonical identifier")
        if not _IDENTIFIER.fullmatch(self.adapter_id):
            raise ValueError("adapter_id must be a canonical identifier")
        if self.declaration.kind != ChannelKind.SENSOR:
            raise ValueError("only sensor declarations can become observations")
        if self.reading.channel_id != self.declaration.channel_id:
            raise ValueError("reading and declaration channel identities differ")
        if self.reading.unit != self.declaration.unit:
            raise ValueError("reading and declaration units differ")
        if self.received_at_ns <= 0 or self.received_monotonic_ns <= 0:
            raise ValueError("observation receipt clocks must be positive")
        if not _IDENTIFIER.fullmatch(self.subscription_id):
            raise ValueError("subscription_id must be a canonical identifier")
        if self.historian_record_id and not _IDENTIFIER.fullmatch(self.historian_record_id):
            raise ValueError("historian_record_id must be a canonical identifier")
        if self.historian_quality not in {
            "ephemeral",
            "good",
            "uncertain",
            "bad",
            "stale",
            "simulated",
        }:
            raise ValueError("historian_quality differs from its bounded ontology")
        if not self.historian_order_basis or len(self.historian_order_basis) > 80:
            raise ValueError("historian_order_basis must be present and bounded")
        if not isinstance(self.historian_order_gap, bool):
            raise TypeError("historian_order_gap must be a boolean")
        if len(self.historian_alarm_codes) > 8 or any(
            not _IDENTIFIER.fullmatch(item) for item in self.historian_alarm_codes
        ):
            raise ValueError("historian_alarm_codes differ from their bounded contract")
        fenced = bool(self.twin_id)
        if fenced:
            if not _IDENTIFIER.fullmatch(self.twin_id):
                raise ValueError("twin_id must be a canonical identifier")
            for name, value in (
                ("attachment_generation", self.attachment_generation),
                ("attachment_bound_at_ns", self.attachment_bound_at_ns),
                ("topology_revision", self.topology_revision),
            ):
                _stored_int(value, name=name, minimum=1)
        elif any(
            value
            for value in (
                self.attachment_generation,
                self.attachment_bound_at_ns,
                self.topology_revision,
            )
        ):
            raise ValueError("partial digital-twin attachment fence is invalid")
        expected_id = _observation_identifier(
            adapter_id=self.adapter_id,
            channel_id=self.declaration.channel_id,
            reading_sha256=self.reading.sha256,
            received_monotonic_ns=self.received_monotonic_ns,
        )
        if self.observation_id != expected_id:
            raise ValueError("observation_id differs from its evidence")
        object.__setattr__(self, "salience", _clamp01(self.salience))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": _OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "adapter_id": self.adapter_id,
            "declaration": self.declaration.to_dict(),
            "declaration_sha256": self.declaration.sha256,
            "channel_id": self.declaration.channel_id,
            "observable": self.declaration.observable,
            "unit": self.declaration.unit,
            "reading": self.reading.to_dict(),
            "reading_sha256": self.reading.sha256,
            "salience": self.salience,
            "received_at_ns": self.received_at_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "subscription_id": self.subscription_id,
        }
        twin_binding: dict[str, Any] = {
            "twin_id": self.twin_id,
            "attachment_generation": self.attachment_generation,
            "attachment_bound_at_ns": self.attachment_bound_at_ns,
            "topology_revision": self.topology_revision,
        }
        twin_binding["binding_sha256"] = _digest(
            {"observation_id": self.observation_id, "binding": twin_binding}
        )
        payload["twin_binding"] = twin_binding
        historian: dict[str, Any] = {
            "schema": _HISTORIAN_EVIDENCE_SCHEMA,
            "record_id": self.historian_record_id,
            "quality": self.historian_quality,
            "order_basis": self.historian_order_basis,
            "order_gap": self.historian_order_gap,
            "alarm_codes": list(self.historian_alarm_codes),
            "reason": (
                "ephemeral"
                if self.historian_quality == "ephemeral"
                else "accepted_with_source_gap"
                if self.historian_order_gap
                else "accepted"
            ),
        }
        historian["binding_sha256"] = _digest({"observation": payload, "historian": historian})
        payload["historian"] = historian
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RealityObservation:
        """Reconstruct one bounded durable observation without executable input."""

        if not isinstance(payload, Mapping):
            raise TypeError("Reality observation payload must be a mapping")
        schema = str(payload.get("schema") or "")
        if schema not in {_OBSERVATION_SCHEMA, _LEGACY_OBSERVATION_SCHEMA}:
            raise ValueError("unsupported Reality observation schema")
        raw_declaration = payload.get("declaration")
        raw_reading = payload.get("reading")
        if not isinstance(raw_declaration, Mapping) or not isinstance(
            raw_reading,
            Mapping,
        ):
            raise ValueError("Reality observation evidence is incomplete")
        raw_domain = raw_declaration.get("domain")
        if not isinstance(raw_domain, Mapping):
            raise ValueError("Reality observation domain is missing")
        declaration = ChannelDeclaration(
            channel_id=str(raw_declaration.get("channel_id") or ""),
            kind=ChannelKind(str(raw_declaration.get("kind") or "")),
            observable=str(raw_declaration.get("observable") or ""),
            unit=str(raw_declaration.get("unit") or ""),
            domain=NumericDomain(
                raw_domain.get("minimum"),
                raw_domain.get("maximum"),
            ),
            coupling=CouplingClass(str(raw_declaration.get("coupling") or "")),
            reality_layers=tuple(
                RealityLayer(str(value))
                for value in _stored_sequence(
                    raw_declaration.get("reality_layers", ()),
                    name="declaration.reality_layers",
                )
            ),
            evidence_level=EvidenceLevel(str(raw_declaration.get("evidence_level") or "")),
            owner=str(raw_declaration.get("owner") or ""),
            resolution=raw_declaration.get("resolution", 0.0),
            sample_rate_hz=raw_declaration.get("sample_rate_hz", 0.0),
            max_latency_s=raw_declaration.get("max_latency_s", 0.0),
            stale_after_s=raw_declaration.get("stale_after_s", 30.0),
            reference_id=str(raw_declaration.get("reference_id") or ""),
            calibration_id=str(raw_declaration.get("calibration_id") or ""),
            calibration_valid_until_ns=raw_declaration.get("calibration_valid_until_ns"),
            compliance_tags=tuple(
                str(value)
                for value in _stored_sequence(
                    raw_declaration.get("compliance_tags", ()),
                    name="declaration.compliance_tags",
                )
            ),
            external_metrology=_stored_bool(
                raw_declaration.get("external_metrology", False),
                name="declaration.external_metrology",
            ),
            coupling_validated=_stored_bool(
                raw_declaration.get("coupling_validated", False),
                name="declaration.coupling_validated",
            ),
            enabled=_stored_bool(
                raw_declaration.get("enabled", True),
                name="declaration.enabled",
            ),
        )
        reading = ChannelReading(
            channel_id=str(raw_reading.get("channel_id") or ""),
            value=raw_reading.get("value"),
            unit=str(raw_reading.get("unit") or ""),
            captured_at_ns=_stored_int(
                raw_reading.get("captured_at_ns"),
                name="reading.captured_at_ns",
                minimum=1,
            ),
            status=ReadingStatus(str(raw_reading.get("status") or "")),
            source=str(raw_reading.get("source") or ""),
            scenario_id=str(raw_reading.get("scenario_id") or ""),
            uncertainty=raw_reading.get("uncertainty"),
            error=str(raw_reading.get("error") or ""),
            ingested_at_ns=_stored_int(
                raw_reading.get("ingested_at_ns", 0),
                name="reading.ingested_at_ns",
            ),
            ingested_monotonic_ns=_stored_int(
                raw_reading.get("ingested_monotonic_ns", 0),
                name="reading.ingested_monotonic_ns",
            ),
            session_id=str(raw_reading.get("session_id") or ""),
            sequence=_stored_int(
                raw_reading.get("sequence", 0),
                name="reading.sequence",
            ),
            wall_clock_source=str(raw_reading.get("wall_clock_source") or "system.time_ns"),
            source_epoch=str(raw_reading.get("source_epoch") or ""),
            source_sequence=_stored_int(
                raw_reading.get("source_sequence", 0),
                name="reading.source_sequence",
            ),
            source_event_id=str(raw_reading.get("source_event_id") or ""),
            source_quality=str(raw_reading.get("source_quality") or ""),
            adapter_identity_sha256=str(
                raw_reading.get("adapter_identity_sha256") or ""
            ),
            adapter_registration_generation=_stored_int(
                raw_reading.get("adapter_registration_generation", 0),
                name="reading.adapter_registration_generation",
            ),
            adapter_identity_stable=_stored_bool(
                raw_reading.get("adapter_identity_stable", False),
                name="reading.adapter_identity_stable",
            ),
        )
        if payload.get("channel_id") not in {None, declaration.channel_id}:
            raise ValueError("Reality observation channel summary conflicts with evidence")
        if payload.get("unit") not in {None, declaration.unit}:
            raise ValueError("Reality observation unit summary conflicts with evidence")
        if payload.get("declaration_sha256") != declaration.sha256:
            raise ValueError("Reality observation declaration digest differs")
        if payload.get("reading_sha256") != reading.sha256:
            raise ValueError("Reality observation reading digest differs")
        received_at_ns = _stored_int(
            payload.get("received_at_ns"),
            name="received_at_ns",
            minimum=1,
        )
        received_monotonic_ns = _stored_int(
            payload.get("received_monotonic_ns"),
            name="received_monotonic_ns",
            minimum=1,
        )
        raw_historian = payload.get("historian", {})
        if not isinstance(raw_historian, Mapping):
            raise ValueError("Reality observation historian evidence is invalid")
        if set(raw_historian) != _HISTORIAN_EVIDENCE_KEYS:
            raise ValueError("Reality observation historian evidence manifest differs")
        if raw_historian.get("schema") != _HISTORIAN_EVIDENCE_SCHEMA:
            raise ValueError("Reality observation historian schema differs")
        historian_record_id = str(raw_historian.get("record_id") or "")
        historian_quality = str(raw_historian.get("quality") or "")
        historian_order_basis = str(raw_historian.get("order_basis") or "")
        historian_order_gap = _stored_bool(
            raw_historian.get("order_gap", False),
            name="historian.order_gap",
        )
        historian_alarm_codes = tuple(
            str(value)
            for value in _stored_sequence(
                raw_historian.get("alarm_codes", ()),
                name="historian.alarm_codes",
            )
        )
        expected_reason = (
            "ephemeral"
            if historian_quality == "ephemeral"
            else "accepted_with_source_gap"
            if historian_order_gap
            else "accepted"
        )
        if str(raw_historian.get("reason") or "") != expected_reason:
            raise ValueError("Reality observation historian reason differs")
        if historian_quality == "ephemeral":
            if (
                historian_record_id
                or historian_order_basis != "ephemeral"
                or historian_order_gap
                or historian_alarm_codes
            ):
                raise ValueError("Ephemeral observation carries durable historian claims")
        elif not historian_record_id:
            raise ValueError("Durable observation has no historian record binding")
        binding_payload = dict(payload)
        binding_payload.pop("historian", None)
        binding_evidence = dict(raw_historian)
        supplied_binding = str(binding_evidence.pop("binding_sha256", ""))
        expected_binding = _digest({"observation": binding_payload, "historian": binding_evidence})
        if supplied_binding != expected_binding:
            raise ValueError("Reality observation historian evidence binding differs")
        twin_id = ""
        attachment_generation = 0
        attachment_bound_at_ns = 0
        topology_revision = 0
        if schema == _OBSERVATION_SCHEMA:
            raw_twin_binding = payload.get("twin_binding")
            if not isinstance(raw_twin_binding, Mapping):
                raise ValueError("Reality observation twin binding is missing")
            expected_keys = {
                "attachment_bound_at_ns",
                "attachment_generation",
                "binding_sha256",
                "topology_revision",
                "twin_id",
            }
            if set(raw_twin_binding) != expected_keys:
                raise ValueError("Reality observation twin binding manifest differs")
            twin_binding = dict(raw_twin_binding)
            supplied_twin_digest = str(twin_binding.pop("binding_sha256", ""))
            if supplied_twin_digest != _digest(
                {
                    "observation_id": str(payload.get("observation_id") or ""),
                    "binding": twin_binding,
                }
            ):
                raise ValueError("Reality observation twin binding digest differs")
            twin_id = str(twin_binding.get("twin_id") or "")
            attachment_generation = _stored_int(
                twin_binding.get("attachment_generation", 0),
                name="twin_binding.attachment_generation",
                minimum=1 if twin_id else 0,
            )
            attachment_bound_at_ns = _stored_int(
                twin_binding.get("attachment_bound_at_ns", 0),
                name="twin_binding.attachment_bound_at_ns",
                minimum=1 if twin_id else 0,
            )
            topology_revision = _stored_int(
                twin_binding.get("topology_revision", 0),
                name="twin_binding.topology_revision",
                minimum=1 if twin_id else 0,
            )
        return cls(
            observation_id=str(payload.get("observation_id") or ""),
            adapter_id=str(payload.get("adapter_id") or ""),
            declaration=declaration,
            reading=reading,
            salience=payload.get("salience", 0.0),
            received_at_ns=received_at_ns,
            received_monotonic_ns=received_monotonic_ns,
            subscription_id=str(payload.get("subscription_id") or ""),
            historian_record_id=historian_record_id,
            historian_quality=historian_quality,
            historian_order_basis=historian_order_basis,
            historian_order_gap=historian_order_gap,
            historian_alarm_codes=historian_alarm_codes,
            twin_id=twin_id,
            attachment_generation=attachment_generation,
            attachment_bound_at_ns=attachment_bound_at_ns,
            topology_revision=topology_revision,
        )


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    observation_id: str
    accepted: bool
    reason: str
    queue_depth: int
    salience: float
    evicted_observation_id: str = ""


@dataclass(slots=True)
class _ChannelState:
    status: ReadingStatus
    value: float | None
    accepted_monotonic: float
    reading_sha256: str


@dataclass(slots=True)
class _SourceState:
    source_epoch: str
    source_sequence: int


@dataclass(slots=True)
class _Sampler:
    adapter_id: str
    declarations: dict[str, ChannelDeclaration]
    callback: Callable[[], Awaitable[ChannelReading | tuple[ChannelReading, ...]]]
    sample_rate_hz: float
    next_due_monotonic: float = 0.0


class RealityObservationRouter:
    """Backpressured, salience-aware physical observation service."""

    def __init__(
        self,
        service: RealityReachService,
        *,
        historian: RealityHistorian | None = None,
        digital_twin: RealityDigitalTwinGraph | None = None,
        queue_limit: int = 256,
        poll_interval_s: float = 2.0,
        sampler_timeout_s: float = 8.5,
        max_delivery_rate_hz: float = 20.0,
    ) -> None:
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not 8 <= int(queue_limit) <= 8192:
            raise ValueError("queue_limit must lie inside [8, 8192]")
        self._service = service
        if historian is not None and not isinstance(historian, RealityHistorian):
            raise TypeError("historian must be a RealityHistorian")
        self._historian = historian
        if digital_twin is not None and not isinstance(digital_twin, RealityDigitalTwinGraph):
            raise TypeError("digital_twin must be a RealityDigitalTwinGraph")
        self._digital_twin = digital_twin
        self._required_sinks = tuple(item.sink_id for item in _REQUIRED_SINK_REGISTRY)
        self._queue_limit = int(queue_limit)
        self._poll_interval_s = max(0.1, min(float(poll_interval_s), 60.0))
        self._sampler_timeout_s = max(0.1, min(float(sampler_timeout_s), 30.0))
        self._max_delivery_rate_hz = max(
            0.5,
            min(_finite(max_delivery_rate_hz, name="max_delivery_rate_hz"), 100.0),
        )
        self._queue: deque[RealityObservation] = deque()
        self._latest: dict[str, RealityObservation] = {}
        self._channel_state: dict[str, _ChannelState] = {}
        self._source_state: dict[str, _SourceState] = {}
        self._subscriptions: dict[str, ObservationSubscription] = {
            "reality.default": ObservationSubscription(
                subscription_id="reality.default",
                selector="*",
                max_rate_hz=0.5,
                min_delta=0.0,
                min_salience=0.12,
            )
        }
        self._samplers: dict[str, _Sampler] = {}
        self._lock = checked_lock("reality_observation_router.state", reentrant=True)
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[Any] | None = None
        self._poll_task: asyncio.Task[Any] | None = None
        self._running = False
        self._sequence = 0
        self._accepted = 0
        self._delivered = 0
        self._deduplicated = 0
        self._rate_limited = 0
        self._below_salience = 0
        self._overflow_drops = 0
        self._coalesced = 0
        self._delivery_failures = 0
        self._sampler_failures = 0
        self._last_delivery_ns = 0
        self._historian_admitted = 0
        self._historian_rejected = 0
        self._historian_replays = 0
        self._historian_failures = 0
        self._historian_backoff_s = 0.25
        self._durable_queue_depth = 0
        self._last_historian_probe_monotonic = 0.0
        self._last_historian_maintenance_monotonic = 0.0
        self._last_twin_probe_monotonic = 0.0
        self._last_backfill_probe_monotonic = 0.0
        self._twin_backfill_ready = historian is None
        self._twin_backfill_receipts = 0
        self._twin_backfill_failures = 0
        self._attention_paused = False
        self._acquisition_paused = False

    def configure_subscription(self, subscription: ObservationSubscription) -> None:
        if not isinstance(subscription, ObservationSubscription):
            raise TypeError("subscription must be an ObservationSubscription")
        with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription

    def remove_subscription(self, subscription_id: str) -> None:
        if subscription_id == "reality.default":
            raise ValueError("the bounded default subscription cannot be removed")
        with self._lock:
            if self._subscriptions.pop(subscription_id, None) is None:
                raise LookupError(f"unknown observation subscription: {subscription_id}")

    def focus(
        self,
        channel_or_prefix: str,
        *,
        duration_s: float = 30.0,
        max_rate_hz: float = 4.0,
        min_salience: float = 0.0,
    ) -> ObservationSubscription:
        selector = str(channel_or_prefix or "").strip().lower()
        duration = max(0.1, min(_finite(duration_s, name="duration_s"), 3600.0))
        digest = _digest({"selector": selector, "at": time.monotonic_ns()})
        subscription = ObservationSubscription(
            subscription_id=f"reality.focus.{digest.removeprefix('sha256:')[:24]}",
            selector=selector,
            max_rate_hz=max_rate_hz,
            min_salience=min_salience,
            expires_monotonic=time.monotonic() + duration,
        )
        self.configure_subscription(subscription)
        return subscription

    def pause_attention(self) -> None:
        with self._lock:
            self._attention_paused = True

    def resume_attention(self) -> None:
        with self._lock:
            self._attention_paused = False

    def pause_acquisition(self) -> None:
        with self._lock:
            self._acquisition_paused = True

    def resume_acquisition(self) -> None:
        with self._lock:
            self._acquisition_paused = False

    def pause(self) -> None:
        """Compatibility control: stop acquiring new physical samples."""

        self.pause_acquisition()

    def resume(self) -> None:
        self.resume_acquisition()

    def register_sampler(self, adapter: Any) -> None:
        adapter_id = str(getattr(adapter, "adapter_id", "") or "")
        if not _IDENTIFIER.fullmatch(adapter_id):
            raise ValueError("sampled adapter requires a canonical adapter_id")
        callback = getattr(adapter, "refresh_readback", None)
        declarations_fn = getattr(adapter, "declarations", None)
        if not callable(callback) or not asyncio.iscoroutinefunction(callback):
            raise TypeError("sampled adapter requires async refresh_readback")
        if not callable(declarations_fn):
            raise TypeError("sampled adapter requires declarations")
        declarations = {
            item.channel_id: item
            for item in tuple(declarations_fn())
            if isinstance(item, ChannelDeclaration) and item.kind == ChannelKind.SENSOR
        }
        if not declarations:
            raise ValueError("sampled adapter declares no sensor channel")
        rate = max(
            0.01,
            min(20.0, max(item.sample_rate_hz for item in declarations.values())),
        )
        with self._lock:
            existing = self._samplers.get(adapter_id)
            if existing is not None and existing.callback != callback:
                raise ValueError(f"sampler already registered: {adapter_id}")
            self._samplers[adapter_id] = _Sampler(
                adapter_id=adapter_id,
                declarations=declarations,
                callback=callback,
                sample_rate_hz=rate,
            )

    def unregister_sampler(self, adapter_id: str) -> None:
        with self._lock:
            if self._samplers.pop(adapter_id, None) is None:
                raise LookupError(f"sampler is not registered: {adapter_id}")

    async def submit(
        self,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        *,
        adapter_id: str,
    ) -> ObservationReceipt:
        with self._lock:
            acquisition_paused = self._acquisition_paused
            queue_depth = (
                self._durable_queue_depth if self._historian is not None else len(self._queue)
            )
        if acquisition_paused:
            return ObservationReceipt(
                "",
                False,
                "acquisition_paused",
                queue_depth,
                0.0,
            )
        if declaration.kind != ChannelKind.SENSOR:
            raise ValueError("only sensor declarations can be submitted")
        if reading.channel_id != declaration.channel_id or reading.unit != declaration.unit:
            raise ValueError("reading differs from its declaration")
        now = time.monotonic()
        policy = self._policy_for(declaration.channel_id, now=now)
        reading_sha256 = reading.sha256
        event_sha256 = reading.event_sha256
        with self._lock:
            previous = self._channel_state.get(declaration.channel_id)
            previous_source = self._source_state.get(declaration.channel_id)
        attention_reason = ""
        salience = 0.0
        if policy is None:
            attention_reason = "not_subscribed"
        elif previous is not None and previous.reading_sha256 == event_sha256:
            self._deduplicated += 1
            attention_reason = "duplicate"
        else:
            salience = self._salience(
                declaration,
                reading,
                previous,
                previous_source,
            )
        if policy is not None and not attention_reason and previous is not None:
            interval = 1.0 / policy.max_rate_hz
            if (
                now - previous.accepted_monotonic < interval
                and previous.status == reading.status
                and salience < 0.9
            ):
                self._rate_limited += 1
                attention_reason = "rate_limited"
            elif (
                previous.value is not None
                and reading.value is not None
                and abs(reading.value - previous.value) < policy.min_delta
                and previous.status == reading.status
            ):
                self._deduplicated += 1
                attention_reason = "below_min_delta"
        if policy is not None and not attention_reason and salience < policy.min_salience:
            self._below_salience += 1
            attention_reason = "below_salience"
        observation: RealityObservation | None = None
        if not attention_reason and policy is not None:
            twin_binding: Mapping[str, Any] = {}
            if self._digital_twin is not None:
                twin_binding = await asyncio.to_thread(
                    self._digital_twin.binding_context,
                    adapter_id,
                    declaration,
                )
            received_at_ns = max(1, time.time_ns())
            received_monotonic_ns = max(1, time.monotonic_ns())
            self._sequence += 1
            observation = RealityObservation(
                observation_id=_observation_identifier(
                    adapter_id=adapter_id,
                    channel_id=declaration.channel_id,
                    reading_sha256=reading_sha256,
                    received_monotonic_ns=received_monotonic_ns,
                ),
                adapter_id=adapter_id,
                declaration=declaration,
                reading=reading,
                salience=salience,
                received_at_ns=received_at_ns,
                received_monotonic_ns=received_monotonic_ns,
                subscription_id=policy.subscription_id,
                twin_id=str(twin_binding.get("twin_id") or ""),
                attachment_generation=int(twin_binding.get("attachment_generation") or 0),
                attachment_bound_at_ns=int(twin_binding.get("attachment_bound_at_ns") or 0),
                topology_revision=int(twin_binding.get("topology_revision") or 0),
            )
        if self._historian is not None:
            admission = await self._historian.admit(
                declaration,
                reading,
                adapter_id=adapter_id,
                delivery_observation_id=(
                    observation.observation_id if observation is not None else ""
                ),
                delivery_payload=(observation.to_dict() if observation is not None else None),
                delivery_queue_limit=self._queue_limit,
                delivery_salience=salience,
                delivery_required_sinks=self._required_sinks,
            )
            if admission.disposition in {
                HistorianDisposition.ACCEPTED,
                HistorianDisposition.DEADBAND,
            }:
                self._advance_source_state(reading)
            if not admission.accepted:
                self._historian_rejected += 1
                return ObservationReceipt(
                    "",
                    False,
                    f"historian_{admission.reason}",
                    self._durable_queue_depth,
                    salience,
                )
            self._historian_admitted += 1
            if observation is not None:
                self._durable_queue_depth = admission.delivery_queue_depth
        else:
            self._advance_source_state(reading)
        if attention_reason:
            return ObservationReceipt(
                "",
                False,
                attention_reason,
                (self._durable_queue_depth if self._historian is not None else len(self._queue)),
                salience,
            )
        if observation is None:
            raise RuntimeError("accepted Reality observation was not constructed")
        if self._historian is not None:
            observation = replace(
                observation,
                historian_record_id=admission.record_id,
                historian_quality=admission.quality.value,
                historian_order_basis=admission.order_basis or "unknown",
                historian_order_gap=admission.order_gap,
                historian_alarm_codes=admission.alarm_codes,
            )
            if not admission.delivery_accepted:
                self._overflow_drops += 1
                return ObservationReceipt(
                    observation.observation_id,
                    False,
                    admission.delivery_reason,
                    admission.delivery_queue_depth,
                    salience,
                )
            superseded = admission.superseded_delivery_ids
            evicted = superseded[0] if superseded else ""
            if superseded:
                self._coalesced += len(superseded)
            with self._lock:
                self._latest[declaration.channel_id] = observation
                self._channel_state[declaration.channel_id] = _ChannelState(
                    status=reading.status,
                    value=reading.value,
                    accepted_monotonic=now,
                    reading_sha256=event_sha256,
                )
                self._accepted += 1
            self._wake.set()
            return ObservationReceipt(
                observation.observation_id,
                True,
                "accepted",
                admission.delivery_queue_depth,
                salience,
                evicted_observation_id=evicted,
            )
        evicted = ""
        rejected_for_capacity = False
        with self._lock:
            for pending_index, pending in enumerate(self._queue):
                if pending.declaration.channel_id != declaration.channel_id:
                    continue
                evicted = pending.observation_id
                del self._queue[pending_index]
                self._coalesced += 1
                break
            if len(self._queue) >= self._queue_limit:
                least_index, least = min(
                    enumerate(self._queue),
                    key=lambda item: (item[1].salience, item[1].received_monotonic_ns),
                )
                if least.salience >= observation.salience:
                    self._overflow_drops += 1
                    rejected_for_capacity = True
                else:
                    evicted = least.observation_id
                    del self._queue[least_index]
                    self._overflow_drops += 1
            if not rejected_for_capacity:
                self._queue.append(observation)
                self._latest[declaration.channel_id] = observation
                self._channel_state[declaration.channel_id] = _ChannelState(
                    status=reading.status,
                    value=reading.value,
                    accepted_monotonic=now,
                    reading_sha256=event_sha256,
                )
            depth = len(self._queue)
            if not rejected_for_capacity:
                self._accepted += 1
        if rejected_for_capacity:
            return ObservationReceipt(
                observation.observation_id,
                False,
                "queue_full_lower_priority",
                depth,
                salience,
            )
        self._wake.set()
        return ObservationReceipt(
            observation.observation_id,
            True,
            "accepted",
            depth,
            salience,
            evicted_observation_id=evicted,
        )

    async def poll_once(self) -> int:
        with self._lock:
            if self._acquisition_paused:
                return 0
        readings = await asyncio.to_thread(self._service.refresh)
        declarations = {
            item.channel_id: item
            for item in self._service.declarations()
            if item.kind == ChannelKind.SENSOR
        }
        ownership = self._service.adapter_channels()
        owner_by_channel = {
            channel_id: adapter_id
            for adapter_id, channel_ids in ownership.items()
            for channel_id in channel_ids
        }
        with self._lock:
            sampled_adapter_ids = set(self._samplers)
        accepted = 0
        for channel_id, declaration in declarations.items():
            if owner_by_channel.get(channel_id) in sampled_adapter_ids:
                # Async sampled adapters refresh below. Submitting the cached
                # service reading first would rate-limit their fresh readback.
                continue
            reading = readings.get(channel_id)
            if reading is None:
                continue
            receipt = await self.submit(
                declaration,
                reading,
                adapter_id=owner_by_channel.get(channel_id, "reality.unknown"),
            )
            accepted += int(receipt.accepted)
        accepted += await self._poll_samplers()
        return accepted

    async def _poll_samplers(self) -> int:
        now = time.monotonic()
        with self._lock:
            due = [
                sampler for sampler in self._samplers.values() if now >= sampler.next_due_monotonic
            ]
            for sampler in due:
                sampler.next_due_monotonic = now + (1.0 / sampler.sample_rate_hz)
        if not due:
            return 0
        semaphore = checked_semaphore("observation_router", 8)

        async def _sample(sampler: _Sampler) -> int:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        sampler.callback(),
                        timeout=self._sampler_timeout_s,
                    )
                    readings = result if isinstance(result, tuple) else (result,)
                    normalized = await asyncio.to_thread(
                        self._service.ingest_sensor_readings,
                        sampler.adapter_id,
                        readings,
                    )
                    accepted = 0
                    for reading in normalized.values():
                        if not isinstance(reading, ChannelReading):
                            raise TypeError("sampler returned a non-reading")
                        declaration = sampler.declarations.get(reading.channel_id)
                        if declaration is None:
                            raise ValueError("sampler returned an undeclared channel")
                        receipt = await self.submit(
                            declaration,
                            reading,
                            adapter_id=sampler.adapter_id,
                        )
                        accepted += int(receipt.accepted)
                    return accepted
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    self._sampler_failures += 1
                    record_degradation(
                        "reality_observation_router.sampler",
                        exc,
                        action=f"retained other physical samplers after {sampler.adapter_id} failed",
                    )
                    return 0

        return sum(await asyncio.gather(*(_sample(item) for item in due)))

    def _resolve_sink_dependency(self, requirement: _RequiredSink) -> Any:
        if requirement.sink_id == "digital_twin":
            return self._digital_twin
        from core.container import ServiceContainer

        return ServiceContainer.get(requirement.dependency_key, default=None)

    @staticmethod
    def _sink_health_ready(
        requirement: _RequiredSink,
        payload: Any,
    ) -> tuple[bool, str]:
        if not isinstance(payload, Mapping):
            return False, "health_payload_invalid"
        if requirement.health_contract == "ready_flag":
            ready = payload.get("ready") is True
            return ready, "ready" if ready else "health_report_unready"
        if requirement.health_contract == "ok_flag":
            ready = payload.get("ok") is True
            return ready, "ready" if ready else "health_report_unready"
        if requirement.health_contract == "bounded_multimodal_status":
            queue_limit = payload.get("queue_limit")
            queue_depths = payload.get("queue_depths")
            counters = (
                payload.get("accepted_events"),
                payload.get("rejected_events"),
                payload.get("queue_overflow_drops"),
                payload.get("late_events"),
                payload.get("fusions"),
            )
            bounded_queue_limit = (
                queue_limit
                if isinstance(queue_limit, int)
                and not isinstance(queue_limit, bool)
                and 1 <= queue_limit <= 1_000_000
                else None
            )
            valid_depths = (
                isinstance(queue_depths, Mapping)
                and bool(queue_depths)
                and bounded_queue_limit is not None
                and all(
                    isinstance(depth, int)
                    and not isinstance(depth, bool)
                    and 0 <= depth <= bounded_queue_limit
                    for depth in queue_depths.values()
                )
            )
            valid_counters = all(
                isinstance(counter, int)
                and not isinstance(counter, bool)
                and counter >= 0
                for counter in counters
            )
            ready = bool(valid_depths and valid_counters)
            return ready, "ready" if ready else "health_report_unready"
        return False, "health_contract_unknown"

    def _required_sink_status(self) -> dict[str, dict[str, Any]]:
        status: dict[str, dict[str, Any]] = {}
        for requirement in _REQUIRED_SINK_REGISTRY:
            try:
                dependency = self._resolve_sink_dependency(requirement)
                callback = getattr(dependency, requirement.callable_attribute, None)
                configured = dependency is not None and callable(callback)
                ready = False
                reason = "dependency_unavailable"
                health_probe = getattr(dependency, requirement.health_attribute, None)
                if configured and callable(health_probe):
                    ready, reason = self._sink_health_ready(
                        requirement,
                        health_probe(),
                    )
                elif configured:
                    ready = False
                    reason = "health_probe_unavailable"
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                configured = False
                ready = False
                reason = f"probe_failed:{type(exc).__name__}"
            status[requirement.sink_id] = {
                "dependency_key": requirement.dependency_key,
                "callable_attribute": requirement.callable_attribute,
                "health_attribute": requirement.health_attribute,
                "health_contract": requirement.health_contract,
                "configured": configured,
                "ready": ready,
                "reason": reason,
            }
        return status

    @staticmethod
    def _backfill_observation(
        snapshot: HistorianHeadSnapshot,
        binding: Mapping[str, Any],
    ) -> tuple[RealityObservation, str]:
        binding_evidence = {
            "twin_id": str(binding.get("twin_id") or ""),
            "attachment_generation": int(binding.get("attachment_generation") or 0),
            "attachment_bound_at_ns": int(binding.get("attachment_bound_at_ns") or 0),
            "topology_revision": int(binding.get("topology_revision") or 0),
        }
        target_binding_sha256 = _digest(binding_evidence)
        received_monotonic_ns = _backfill_received_monotonic_ns(
            record_id=snapshot.record_id,
            source_sha256=snapshot.source_sha256,
            target_binding_sha256=target_binding_sha256,
        )
        observation_id = _observation_identifier(
            adapter_id=snapshot.adapter_id,
            channel_id=snapshot.channel_id,
            reading_sha256=_digest(snapshot.reading),
            received_monotonic_ns=received_monotonic_ns,
        )
        twin_binding = dict(binding_evidence)
        twin_binding["binding_sha256"] = _digest(
            {"observation_id": observation_id, "binding": binding_evidence}
        )
        captured_at_ns = _stored_int(
            snapshot.reading.get("captured_at_ns"),
            name="backfill.reading.captured_at_ns",
            minimum=1,
        )
        payload: dict[str, Any] = {
            "schema": _OBSERVATION_SCHEMA,
            "observation_id": observation_id,
            "adapter_id": snapshot.adapter_id,
            "declaration": dict(snapshot.declaration),
            "declaration_sha256": _digest(snapshot.declaration),
            "channel_id": snapshot.channel_id,
            "observable": str(snapshot.declaration.get("observable") or ""),
            "unit": str(snapshot.declaration.get("unit") or ""),
            "reading": dict(snapshot.reading),
            "reading_sha256": _digest(snapshot.reading),
            "salience": 0.0,
            "received_at_ns": max(
                1,
                int(snapshot.reading.get("ingested_at_ns") or captured_at_ns),
            ),
            "received_monotonic_ns": received_monotonic_ns,
            "subscription_id": _BACKFILL_SUBSCRIPTION_ID,
            "twin_binding": twin_binding,
        }
        historian = {
            "schema": _HISTORIAN_EVIDENCE_SCHEMA,
            "record_id": snapshot.record_id,
            "quality": snapshot.quality,
            "order_basis": snapshot.order_basis,
            "order_gap": snapshot.order_gap,
            "alarm_codes": list(snapshot.alarm_codes),
            "reason": "accepted_with_source_gap" if snapshot.order_gap else "accepted",
        }
        historian["binding_sha256"] = _digest(
            {"observation": payload, "historian": historian}
        )
        payload["historian"] = historian
        return RealityObservation.from_dict(payload), target_binding_sha256

    async def _backfill_legacy_twin_heads(self) -> int:
        if self._historian is None:
            self._twin_backfill_ready = True
            return 0
        if self._digital_twin is None or not self._digital_twin.is_ready():
            self._twin_backfill_ready = False
            return 0
        inventory = self._service.adapter_inventory()
        cursor = ""
        completed = 0
        exhausted = False
        for _page_number in range(_BACKFILL_MAX_PAGES):
            page = await self._historian.legacy_twin_head_page(
                adapter_inventory=inventory,
                after_channel_id=cursor,
                limit=_BACKFILL_PAGE_SIZE,
            )
            for snapshot in page.snapshots:
                current_inventory = self._service.adapter_inventory()
                current_entry = current_inventory.get(snapshot.adapter_id)
                if (
                    current_entry is None
                    or snapshot.channel_id not in current_entry.channel_ids
                    or snapshot.adapter_identity_sha256 != current_entry.identity_sha256
                    or snapshot.adapter_identity_stable != current_entry.stable_identity
                    or (
                        not current_entry.stable_identity
                        and snapshot.adapter_registration_generation
                        != current_entry.registration_generation
                    )
                ):
                    self._twin_backfill_ready = False
                    return completed
                binding = await asyncio.to_thread(
                    self._digital_twin.binding_context,
                    snapshot.adapter_id,
                    RealityObservationRouter._declaration_from_snapshot(snapshot),
                )
                observation, binding_sha256 = self._backfill_observation(snapshot, binding)
                twin_receipt = await asyncio.to_thread(
                    self._digital_twin.observe_observation,
                    observation,
                )
                if (
                    not isinstance(twin_receipt, TwinReceipt)
                    or not twin_receipt.accepted
                    or twin_receipt.twin_id != observation.twin_id
                ):
                    raise RuntimeError("reality_digital_twin_backfill_rejected")
                await self._historian.record_backfill_receipt(
                    backfill_id=observation.observation_id,
                    record_id=snapshot.record_id,
                    channel_id=snapshot.channel_id,
                    source_sha256=snapshot.source_sha256,
                    target_binding_sha256=binding_sha256,
                    sink_receipt_id=twin_receipt.receipt_id,
                )
                completed += 1
            cursor = page.next_channel_id
            if page.exhausted:
                exhausted = True
                break
        if not exhausted:
            self._twin_backfill_ready = False
            self._twin_backfill_receipts += completed
            return completed
        if self._service.adapter_inventory() != inventory:
            self._twin_backfill_ready = False
            self._twin_backfill_receipts += completed
            return completed
        remaining = await self._historian.legacy_twin_head_page(
            adapter_inventory=inventory,
            limit=1,
        )
        self._twin_backfill_receipts += completed
        self._twin_backfill_ready = not remaining.snapshots and remaining.exhausted
        return completed

    @staticmethod
    def _declaration_from_snapshot(snapshot: HistorianHeadSnapshot) -> ChannelDeclaration:
        received_monotonic_ns = max(
            1,
            int(snapshot.reading.get("ingested_monotonic_ns") or 1),
        )
        probe_payload = {
            "schema": _LEGACY_OBSERVATION_SCHEMA,
            "observation_id": _observation_identifier(
                adapter_id=snapshot.adapter_id,
                channel_id=snapshot.channel_id,
                reading_sha256=_digest(snapshot.reading),
                received_monotonic_ns=received_monotonic_ns,
            ),
            "adapter_id": snapshot.adapter_id,
            "declaration": dict(snapshot.declaration),
            "declaration_sha256": _digest(snapshot.declaration),
            "reading": dict(snapshot.reading),
            "reading_sha256": _digest(snapshot.reading),
            "channel_id": snapshot.channel_id,
            "unit": str(snapshot.declaration.get("unit") or ""),
            "salience": 0.0,
            "received_at_ns": max(
                1,
                int(snapshot.reading.get("ingested_at_ns") or snapshot.reading["captured_at_ns"]),
            ),
            "received_monotonic_ns": received_monotonic_ns,
            "subscription_id": _BACKFILL_SUBSCRIPTION_ID,
        }
        historian = {
            "schema": _HISTORIAN_EVIDENCE_SCHEMA,
            "record_id": snapshot.record_id,
            "quality": snapshot.quality,
            "order_basis": snapshot.order_basis,
            "order_gap": snapshot.order_gap,
            "alarm_codes": list(snapshot.alarm_codes),
            "reason": "accepted_with_source_gap" if snapshot.order_gap else "accepted",
        }
        historian["binding_sha256"] = _digest(
            {"observation": probe_payload, "historian": historian}
        )
        probe_payload["historian"] = historian
        return RealityObservation.from_dict(probe_payload).declaration

    async def start(self) -> None:
        if self._running:
            return
        if self._digital_twin is not None:
            twin_ready = await asyncio.to_thread(self._digital_twin.probe_health)
            if not twin_ready:
                record_degradation(
                    "reality_observation_router.digital_twin_start",
                    RuntimeError("Reality digital twin health probe failed"),
                    action=(
                        "Started router supervision in an explicitly unready state; "
                        "durable twin delivery will retry without losing observations"
                    ),
                )
        if self._historian is not None:
            try:
                self._historian_replays += await self._historian.recover_inflight()
                historian_status = await asyncio.to_thread(self._historian.status)
                delivery_counts = historian_status.get("delivery_counts", {})
                if isinstance(delivery_counts, Mapping):
                    self._durable_queue_depth = sum(
                        int(delivery_counts.get(state, 0) or 0)
                        for state in ("queued", "delivering")
                    )
            except HistorianError as exc:
                self._historian_failures += 1
                record_degradation(
                    "reality_observation_router.historian_start",
                    exc,
                    action=(
                        "Started live sensing in an explicitly unready state; "
                        "the supervised durable worker will retry with backoff"
                    ),
                )
        try:
            await self._backfill_legacy_twin_heads()
        except (HistorianError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._twin_backfill_ready = False
            self._twin_backfill_failures += 1
            record_degradation(
                "reality_observation_router.digital_twin_backfill_start",
                exc,
                action=(
                    "Kept authoritative historian heads intact and left router readiness "
                    "closed until the idempotent twin backfill can be receipted"
                ),
            )
        for sink_id, sink_status in self._required_sink_status().items():
            if not bool(sink_status["ready"]):
                record_degradation(
                    f"reality_observation_router.required_sink.{sink_id}",
                    RuntimeError(str(sink_status["reason"])),
                    action=(
                        "Started supervised store-and-forward with the required sink "
                        "visible as unready; no sink was removed from admission"
                    ),
                )
        self._running = True
        self._worker_task = get_task_tracker().create_task(
            self._worker_loop(),
            name="RealityObservationRouter",
        )
        self._poll_task = get_task_tracker().create_task(
            self._poll_loop(),
            name="RealityObservationPoll",
        )

    async def stop(self) -> None:
        self._running = False
        self._wake.set()
        poll_task = self._poll_task
        if poll_task is not None:
            await cancel_and_join(poll_task, owner="core.reality_reach.observation_router")
        worker_task = self._worker_task
        if worker_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(worker_task), timeout=2.0)
            except TimeoutError:
                await cancel_and_join(worker_task, owner="core.reality_reach.observation_router")
        self._poll_task = None
        self._worker_task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
                now = time.monotonic()
                if (
                    self._historian is not None
                    and now - self._last_historian_maintenance_monotonic >= 30.0
                ):
                    await self._historian.maintain()
                    self._last_historian_maintenance_monotonic = now
                if (
                    self._historian is not None
                    and now - self._last_historian_probe_monotonic >= 30.0
                ):
                    prior_ready = self._historian.is_ready()
                    historian_ready = await self._historian.probe_health()
                    self._last_historian_probe_monotonic = now
                    if not historian_ready:
                        self._historian_failures += 1
                        if prior_ready:
                            record_degradation(
                                "reality_observation_router.historian_probe",
                                HistorianError("Reality historian periodic probe failed"),
                                action=(
                                    "Kept live router supervision active and "
                                    "backed off durable delivery pending recovery"
                                ),
                            )
                if self._digital_twin is not None and now - self._last_twin_probe_monotonic >= 30.0:
                    prior_ready = self._digital_twin.is_ready()
                    twin_ready = await asyncio.to_thread(self._digital_twin.probe_health)
                    self._last_twin_probe_monotonic = now
                    if prior_ready and not twin_ready:
                        record_degradation(
                            "reality_observation_router.digital_twin_probe",
                            RuntimeError("Reality digital twin periodic probe failed"),
                            action=(
                                "Retained durable observations for retry while the "
                                "canonical physical graph recovers"
                            ),
                        )
                if (
                    not self._twin_backfill_ready
                    and now - self._last_backfill_probe_monotonic >= 30.0
                ):
                    await self._backfill_legacy_twin_heads()
                    self._last_backfill_probe_monotonic = now
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                if isinstance(exc, HistorianError):
                    self._historian_failures += 1
                record_degradation(
                    "reality_observation_router.poll",
                    exc,
                    action="continued bounded physical sensing after one inventory poll failed",
                )
            await asyncio.sleep(self._poll_interval_s)

    async def _worker_loop(self) -> None:
        next_delivery = time.monotonic()
        while self._running:
            observation: RealityObservation | None = None
            claimed_id = ""
            lease_token = ""
            sink_states: dict[str, dict[str, str]] = {}
            try:
                (
                    observation,
                    claimed_id,
                    lease_token,
                    sink_states,
                ) = await self._next_observation()
                reached_the_person = False
                if observation is None:
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                    # not a failure: the wait ran out, which is what the timeout was set to decide.
                    except TimeoutError:
                        pass
                    continue
                now = time.monotonic()
                if now < next_delivery:
                    await asyncio.sleep(next_delivery - now)
                next_delivery = max(next_delivery, time.monotonic()) + (
                    1.0 / self._max_delivery_rate_hz
                )
                await self._deliver(
                    observation,
                    claimed_id=claimed_id,
                    lease_token=lease_token,
                    sink_states=sink_states,
                )
                # The observation has now REACHED the person. Everything
                # below is bookkeeping about an event that already happened,
                # and the failure handler must not treat it as a delivery
                # that did not occur — `mark_delivery_failed` requeues, and
                # requeuing something already seen shows it to them twice.
                reached_the_person = True
                if self._historian is not None and claimed_id:
                    await self._historian.mark_delivered(
                        claimed_id,
                        lease_token=lease_token,
                    )
                    self._durable_queue_depth = max(
                        0,
                        self._durable_queue_depth - 1,
                    )
                self._delivered += 1
                self._last_delivery_ns = max(1, time.time_ns())
                self._historian_backoff_s = 0.25
            except asyncio.CancelledError:
                if self._historian is not None and claimed_id:
                    try:
                        cancelled_state = await asyncio.shield(
                            self._historian.mark_delivery_failed(
                                claimed_id,
                                error_code="delivery_cancelled",
                                lease_token=lease_token,
                            )
                        )
                        if cancelled_state == "quarantined":
                            self._durable_queue_depth = max(
                                0,
                                self._durable_queue_depth - 1,
                            )
                    except (RuntimeError, ValueError):
                        pass
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                self._delivery_failures += 1
                if isinstance(exc, HistorianError):
                    self._historian_failures += 1
                if self._historian is not None and claimed_id:
                    if reached_the_person:
                        # Delivered, then the RECORD failed. Two different
                        # events wearing one exception: the person has seen
                        # this observation, and only the bookkeeping is
                        # broken. Requeuing here — which is what
                        # `mark_delivery_failed` does — would deliver it to
                        # them a second time and blame the sink for a
                        # historian fault.
                        #
                        # One bounded retry of the RECORD, then quarantine.
                        # Quarantine is the honest terminal state: it means
                        # "do not deliver again", which is exactly right for
                        # something already delivered, and it keeps the row
                        # for the recovery audit instead of losing it.
                        await self._settle_unconfirmed_delivery(
                            claimed_id, lease_token=lease_token, cause=exc
                        )
                    else:
                        try:
                            failed_state = await self._historian.mark_delivery_failed(
                                claimed_id,
                                error_code=f"{type(exc).__name__}_delivery_failure".lower(),
                                lease_token=lease_token,
                            )
                            if failed_state == "quarantined":
                                self._durable_queue_depth = max(
                                    0,
                                    self._durable_queue_depth - 1,
                                )
                        except (RuntimeError, ValueError) as receipt_exc:
                            record_degradation(
                                "reality_observation_router.delivery_receipt",
                                receipt_exc,
                                action="left durable delivery failure visible for recovery audit",
                            )
                record_degradation(
                    "reality_observation_router.delivery",
                    exc,
                    action=(
                        "recorded an unconfirmed delivery that already reached the person"
                        if reached_the_person
                        else "retained and scheduled retry for a failed physical observation delivery"
                    ),
                )
                if isinstance(exc, HistorianError):
                    delay = self._historian_backoff_s
                    self._historian_backoff_s = min(30.0, delay * 2.0)
                    await asyncio.sleep(delay)

    async def _settle_unconfirmed_delivery(
        self,
        claimed_id: str,
        *,
        lease_token: str,
        cause: BaseException,
    ) -> None:
        """Close out an observation the person SAW but we failed to record.

        Never requeues. The delivery happened; the only open question is the
        row, and answering it by scheduling a redelivery would make a
        historian fault into a duplicate ambient interruption for the person.

        One retry of the record — most historian faults here are a transient
        lock or a WAL checkpoint — and then quarantine, which means "do not
        deliver again". That is the honest terminal state for something
        already delivered, and it keeps the row for the recovery audit
        instead of dropping it on the floor.
        """
        if self._historian is None or not claimed_id:
            return
        try:
            await self._historian.mark_delivered(claimed_id, lease_token=lease_token)
            self._durable_queue_depth = max(0, self._durable_queue_depth - 1)
            return
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as retry_exc:
            record_degradation(
                "reality_observation_router.delivery_receipt",
                retry_exc,
                severity="error",
                action="could not confirm a delivery that already reached the person",
                extra={"observation_id": str(claimed_id)[:120]},
            )
        try:
            await self._historian.quarantine_delivery(
                claimed_id,
                reason="delivered_but_unconfirmed",
                lease_token=lease_token,
            )
            self._durable_queue_depth = max(0, self._durable_queue_depth - 1)
        except (AttributeError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as quarantine_exc:
            # Even quarantine failed. The lease will expire and the recovery
            # sweep will requeue it, so say plainly that a duplicate is now
            # possible rather than letting it look like an ordinary retry.
            record_degradation(
                "reality_observation_router.delivery_receipt",
                quarantine_exc,
                severity="critical",
                action=(
                    "left a delivered observation unquarantined; lease expiry "
                    "will requeue it and the person may see it twice"
                ),
                extra={"observation_id": str(claimed_id)[:120], "cause": repr(cause)[:160]},
            )

    async def _next_observation(
        self,
    ) -> tuple[
        RealityObservation | None,
        str,
        str,
        dict[str, dict[str, str]],
    ]:
        while self._running:
            with self._lock:
                observation = self._queue.popleft() if self._queue else None
                if not self._queue:
                    self._wake.clear()
            if observation is not None:
                if self._historian is None:
                    return observation, "", "", {}
                delivery = await self._historian.claim_delivery(observation.observation_id)
                if delivery is None:
                    continue
                try:
                    restored = RealityObservation.from_dict(delivery.payload)
                except (KeyError, TypeError, ValueError) as exc:
                    await self._historian.mark_delivery_failed(
                        delivery.observation_id,
                        error_code="durable_payload_invalid",
                        lease_token=delivery.lease_token,
                    )
                    raise ValueError("durable Reality observation is invalid") from exc
                return (
                    restored,
                    delivery.observation_id,
                    delivery.lease_token,
                    delivery.sink_states,
                )
            if self._historian is None:
                return None, "", "", {}
            due = await self._historian.claim_due_deliveries(limit=1)
            if not due:
                return None, "", "", {}
            delivery = due[0]
            self._historian_replays += 1
            try:
                restored = RealityObservation.from_dict(delivery.payload)
            except (KeyError, TypeError, ValueError) as exc:
                await self._historian.mark_delivery_failed(
                    delivery.observation_id,
                    error_code="durable_payload_invalid",
                    lease_token=delivery.lease_token,
                )
                raise ValueError("durable Reality observation is invalid") from exc
            return (
                restored,
                delivery.observation_id,
                delivery.lease_token,
                delivery.sink_states,
            )
        return None, "", "", {}

    async def _deliver(
        self,
        observation: RealityObservation,
        *,
        claimed_id: str,
        lease_token: str,
        sink_states: Mapping[str, Mapping[str, str]],
    ) -> None:
        twin_required = "digital_twin" in sink_states
        twin_delivered = str(sink_states.get("digital_twin", {}).get("state") or "") == "delivered"
        twin_requirement = _REQUIRED_SINK_REGISTRY[0]
        digital_twin = self._resolve_sink_dependency(twin_requirement)
        if twin_required and digital_twin is None:
            raise RuntimeError("reality_digital_twin_unavailable")
        if digital_twin is not None and not twin_delivered:
            if not observation.twin_id:
                if twin_required:
                    raise RuntimeError("reality_observation_attachment_fence_missing")
            else:
                twin_receipt = await asyncio.to_thread(
                    digital_twin.observe_observation,
                    observation,
                )
                if not isinstance(twin_receipt, TwinReceipt):
                    raise RuntimeError("reality_digital_twin_missing_receipt")
                if not twin_receipt.accepted:
                    raise RuntimeError(
                        f"reality_digital_twin_rejected:{twin_receipt.disposition.value}"
                    )
                if twin_receipt.twin_id != observation.twin_id:
                    raise RuntimeError("reality_digital_twin_receipt_identity_mismatch")
                if twin_required and self._historian is not None:
                    await self._historian.mark_sink_delivered(
                        claimed_id,
                        sink="digital_twin",
                        receipt_id=twin_receipt.receipt_id,
                        lease_token=lease_token,
                    )

        synchronizer = self._resolve_sink_dependency(_REQUIRED_SINK_REGISTRY[1])
        multimodal_required = "multimodal" in sink_states
        multimodal_delivered = (
            str(sink_states.get("multimodal", {}).get("state") or "") == "delivered"
        )
        ingest = getattr(synchronizer, "ingest", None)
        if multimodal_required and not callable(ingest):
            raise RuntimeError("multimodal_synchronizer_unavailable")
        if callable(ingest) and not multimodal_delivered:
            reading = observation.reading
            claims: list[PerceptualClaim] = [
                PerceptualClaim(
                    key=f"{observation.declaration.channel_id}.status",
                    value=reading.status.value,
                    confidence=1.0,
                )
            ]
            if reading.value is not None:
                claims.append(
                    PerceptualClaim(
                        key=observation.declaration.channel_id,
                        value=reading.value,
                        confidence=self._confidence(observation),
                    )
                )
            calibration_id = (
                observation.declaration.calibration_id
                or observation.declaration.reference_id
                or f"{observation.adapter_id}.uncalibrated"
            )[:160]
            event = PerceptualEvent(
                event_id=observation.observation_id,
                modality=Modality.DEVICE,
                source=f"reality:{observation.adapter_id}"[:160],
                sequence=max(0, reading.sequence or self._sequence),
                observed_at=max(0.001, reading.captured_at_ns / 1_000_000_000),
                observed_monotonic_ns=observation.received_monotonic_ns,
                summary=(
                    f"{observation.declaration.observable} "
                    f"{reading.status.value}"
                    + (f" {reading.value:g} {reading.unit}" if reading.value is not None else "")
                )[:320],
                confidence=self._confidence(observation),
                claims=tuple(claims),
                calibration=Calibration(
                    calibration_id=calibration_id,
                    status=("valid" if observation.declaration.coupling_validated else "unknown"),
                    reliability=self._confidence(observation),
                ),
                provenance=(
                    observation.reading.sha256,
                    observation.declaration.sha256,
                    observation.adapter_id,
                ),
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.PRIVATE,
                    retention=(
                        "bounded_private_historian" if self._historian is not None else "ephemeral"
                    ),
                    consent_scope="reality_reach.sensor_summary",
                    redacted=True,
                    raw_retained=False,
                ),
                quality_flags=(
                    f"status:{reading.status.value}",
                    f"evidence:{observation.declaration.evidence_level.value}",
                    f"historian_quality:{observation.historian_quality}",
                    f"historian_order:{observation.historian_order_basis}",
                    f"historian_gap:{str(observation.historian_order_gap).lower()}",
                    *tuple(f"historian_alarm:{code}" for code in observation.historian_alarm_codes),
                ),
            )
            receipt = ingest(event)
            accepted = bool(getattr(receipt, "accepted", False))
            reason = str(getattr(receipt, "reason", "") or "")
            event_id = str(getattr(receipt, "event_id", "") or "")
            if not accepted and reason != "duplicate_event":
                raise RuntimeError(f"multimodal_ingest_rejected:{reason or 'missing_receipt'}")
            if event_id != observation.observation_id:
                raise RuntimeError("multimodal_ingest_receipt_identity_mismatch")
            if multimodal_required and self._historian is not None:
                await self._historian.mark_sink_delivered(
                    claimed_id,
                    sink="multimodal",
                    receipt_id=f"{event_id}:{reason or 'accepted'}",
                    lease_token=lease_token,
                )

        advanced = self._resolve_sink_dependency(_REQUIRED_SINK_REGISTRY[2])
        observe_state = getattr(advanced, "observe_state", None)
        advanced_required = "advanced_cognition" in sink_states
        advanced_delivered = (
            str(sink_states.get("advanced_cognition", {}).get("state") or "") == "delivered"
        )
        if advanced_required and not callable(observe_state):
            raise RuntimeError("advanced_cognition_observer_unavailable")
        if callable(observe_state) and not advanced_delivered:
            advanced_receipt = await asyncio.to_thread(
                observe_state,
                "physical_environment",
                {
                    "observation_id": observation.observation_id,
                    "adapter_id": observation.adapter_id,
                    "channel_id": observation.declaration.channel_id,
                    "observable": observation.declaration.observable,
                    "value": observation.reading.value,
                    "unit": observation.reading.unit,
                    "status": observation.reading.status.value,
                    "uncertainty": observation.reading.uncertainty,
                    "salience": observation.salience,
                    "evidence_level": observation.declaration.evidence_level.value,
                    "reality_layers": [
                        layer.value for layer in observation.declaration.reality_layers
                    ],
                    "observation_sha256": observation.reading.sha256,
                    "historian": {
                        "record_id": observation.historian_record_id,
                        "quality": observation.historian_quality,
                        "order_basis": observation.historian_order_basis,
                        "order_gap": observation.historian_order_gap,
                        "alarm_codes": list(observation.historian_alarm_codes),
                    },
                },
                source=f"reality:{observation.adapter_id}",
                confidence=self._confidence(observation),
                observed_at=(observation.reading.captured_at_ns / 1_000_000_000),
                idempotency_key=observation.observation_id,
            )
            if not isinstance(advanced_receipt, Mapping):
                raise RuntimeError("advanced_cognition_missing_receipt")
            advanced_receipt_id = str(advanced_receipt.get("receipt_id") or "")
            if not advanced_receipt_id:
                raise RuntimeError("advanced_cognition_missing_receipt_id")
            if advanced_required and self._historian is not None:
                await self._historian.mark_sink_delivered(
                    claimed_id,
                    sink="advanced_cognition",
                    receipt_id=advanced_receipt_id,
                    lease_token=lease_token,
                )

    def latest(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                channel_id: observation.to_dict()
                for channel_id, observation in self._latest.items()
            }

    def subscriptions(self) -> tuple[ObservationSubscription, ...]:
        now = time.monotonic()
        with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._subscriptions.values()
                        if item.expires_monotonic is None or item.expires_monotonic > now
                    ),
                    key=lambda item: item.subscription_id,
                )
            )

    def status(self) -> dict[str, Any]:
        historian_status = (
            self._historian.health_snapshot() if self._historian is not None else None
        )
        twin_status = (
            self._digital_twin.health_snapshot() if self._digital_twin is not None else None
        )
        sink_registry = self._required_sink_status()
        alive = self.is_alive()
        subscriptions = self.subscriptions()
        historian_ready = self._historian is None or bool(
            isinstance(historian_status, dict) and historian_status.get("ready")
        )
        required_sinks_ready = all(
            bool(item["ready"]) for item in sink_registry.values()
        )
        with self._lock:
            ready = bool(
                alive
                and historian_ready
                and required_sinks_ready
                and self._twin_backfill_ready
                and any(item.enabled for item in subscriptions)
            )
            return {
                "status": "active" if ready else "degraded",
                "alive": alive,
                "ready": ready,
                "queue_depth": (
                    self._durable_queue_depth if self._historian is not None else len(self._queue)
                ),
                "queue_limit": self._queue_limit,
                "latest_channels": len(self._latest),
                "subscriptions": len(subscriptions),
                "samplers": len(self._samplers),
                "accepted": self._accepted,
                "delivered": self._delivered,
                "deduplicated": self._deduplicated,
                "rate_limited": self._rate_limited,
                "below_salience": self._below_salience,
                "overflow_drops": self._overflow_drops,
                "coalesced": self._coalesced,
                "delivery_failures": self._delivery_failures,
                "sampler_failures": self._sampler_failures,
                "last_delivery_ns": self._last_delivery_ns,
                "durable_store_forward": self._historian is not None,
                "historian_admitted": self._historian_admitted,
                "historian_rejected": self._historian_rejected,
                "historian_replays": self._historian_replays,
                "historian_failures": self._historian_failures,
                "historian": historian_status,
                "digital_twin": twin_status,
                "required_sinks": list(self._required_sinks),
                "required_sink_registry": sink_registry,
                "twin_backfill_ready": self._twin_backfill_ready,
                "twin_backfill_receipts": self._twin_backfill_receipts,
                "twin_backfill_failures": self._twin_backfill_failures,
                "attention_paused": self._attention_paused,
                "acquisition_paused": self._acquisition_paused,
            }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def is_alive(self) -> bool:
        return bool(
            self._running
            and self._worker_task is not None
            and not self._worker_task.done()
            and self._poll_task is not None
            and not self._poll_task.done()
        )

    def is_ready(self) -> bool:
        historian_ready = self._historian is None or self._historian.is_ready()
        required_sinks_ready = all(
            bool(item["ready"]) for item in self._required_sink_status().values()
        )
        return bool(
            self.is_alive()
            and historian_ready
            and required_sinks_ready
            and self._twin_backfill_ready
            and any(item.enabled for item in self.subscriptions())
        )

    def _policy_for(
        self,
        channel_id: str,
        *,
        now: float,
    ) -> ObservationSubscription | None:
        with self._lock:
            if self._attention_paused:
                return None
            matches = [
                item for item in self._subscriptions.values() if item.matches(channel_id, now=now)
            ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.specificity, item.max_rate_hz))

    @staticmethod
    def _confidence(observation: RealityObservation) -> float:
        reading = observation.reading
        if reading.status not in _AVAILABLE:
            return 0.0
        domain_width = max(
            1e-12,
            observation.declaration.domain.maximum - observation.declaration.domain.minimum,
        )
        uncertainty = max(0.0, float(reading.uncertainty or 0.0))
        uncertainty_penalty = min(0.75, uncertainty / domain_width)
        evidence = 0.4 + 0.08 * observation.declaration.evidence_level.rank
        if not observation.declaration.coupling_validated:
            evidence *= 0.7
        quality_factor = {
            "good": 1.0,
            "simulated": 0.75,
            "uncertain": 0.5,
            "stale": 0.2,
            "bad": 0.0,
            "ephemeral": 0.8,
        }[observation.historian_quality]
        if observation.historian_order_gap:
            quality_factor *= 0.55
        return _clamp01(evidence * (1.0 - uncertainty_penalty) * quality_factor)

    @staticmethod
    def _salience(
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        previous: _ChannelState | None,
        previous_source: _SourceState | None,
    ) -> float:
        if previous is None:
            base = 0.42 if reading.status in _AVAILABLE else 0.65
        elif previous.status != reading.status:
            base = 0.92
        elif reading.value is None or previous.value is None:
            base = 0.2
        else:
            width = max(1e-12, declaration.domain.maximum - declaration.domain.minimum)
            resolution = max(0.0, declaration.resolution)
            meaningful = max(width * 0.01, resolution, 1e-12)
            normalized = abs(reading.value - previous.value) / meaningful
            base = min(0.85, 0.08 + 0.18 * normalized)
        tags = set(declaration.compliance_tags)
        if tags & {"safety_critical", "life_safety", "interlock", "alarm"}:
            base = max(base, 0.95)
        if (
            previous_source is not None
            and reading.source_epoch
            and reading.source_epoch == previous_source.source_epoch
            and reading.source_sequence > 0
            and previous_source.source_sequence > 0
            and reading.source_sequence > previous_source.source_sequence + 1
        ):
            base = max(base, 0.95)
        if str(reading.source_quality or "").strip().lower() in {
            "bad",
            "stale",
            "uncertain",
        }:
            base = max(base, 0.9)
        if reading.status in {
            ReadingStatus.DEGRADED,
            ReadingStatus.PERMISSION_DENIED,
            ReadingStatus.UNAVAILABLE,
        }:
            base = max(base, 0.75)
        return _clamp01(base)

    def _advance_source_state(self, reading: ChannelReading) -> None:
        if not reading.source_epoch or reading.source_sequence <= 0:
            return
        with self._lock:
            previous = self._source_state.get(reading.channel_id)
            if (
                previous is not None
                and previous.source_epoch == reading.source_epoch
                and reading.source_sequence <= previous.source_sequence
            ):
                return
            self._source_state[reading.channel_id] = _SourceState(
                source_epoch=reading.source_epoch,
                source_sequence=reading.source_sequence,
            )


__all__ = [
    "ObservationReceipt",
    "ObservationSubscription",
    "RealityObservation",
    "RealityObservationRouter",
]
