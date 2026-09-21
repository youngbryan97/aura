"""Durable SCADA-style history, alarm, and store-and-forward semantics.

The historian is the persistence boundary beneath RealityObservationRouter.
It records bounded scalar observations before cognitive attention filtering,
enforces per-source ordering, journals quality/alarm transitions, quarantines
conflicting evidence, and owns restart-safe delivery state. It never actuates a
device and it never creates a second authority or cognition plane.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from core.reality_reach.contracts import ChannelDeclaration, ChannelKind
from core.reality_reach.live import AdapterInventoryEntry, ChannelReading, ReadingStatus
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_lock
from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import state_root


def _roll_back(connection: Any) -> None:
    """Undo the open transaction on the way out of a failed write.

    A rollback that itself fails must not replace the exception on its way
    up: that one says what actually went wrong. Its reason is attached to
    that exception instead, so it travels with the failure rather than
    nowhere. Written once because it was written fourteen times, and each of
    the fourteen was an empty handler.
    """
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error as exc:
        unwinding = sys.exception()
        if unwinding is not None:
            unwinding.add_note(f"the rollback after this also failed: {exc}")


_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,159}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_DELIVERY_STATES = frozenset({"queued", "delivering", "delivered", "superseded", "quarantined"})
_MAX_DELIVERY_PAYLOAD_BYTES = 128 * 1024
_MAX_EVIDENCE_PAYLOAD_BYTES = 128 * 1024
_SQLITE_INT_MAX = (1 << 63) - 1
_DELIVERY_SINK_STATES = frozenset({"pending", "delivered"})
_DELIVERY_SINK_ENVELOPE_SCHEMA = "aura.reality-delivery-sink-state.v2"
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
_OBSERVATION_SCHEMAS = frozenset({"aura.reality-observation.v1", "aura.reality-observation.v2"})
_TWIN_BINDING_KEYS = frozenset(
    {
        "attachment_bound_at_ns",
        "attachment_generation",
        "binding_sha256",
        "topology_revision",
        "twin_id",
    }
)
_DELIVERY_SELECT = (
    "SELECT d.*, o.adapter_id AS evidence_adapter_id, "
    "o.declaration_sha256 AS evidence_declaration_sha256, "
    "o.reading_sha256 AS evidence_reading_sha256, "
    "o.status AS evidence_status, o.quality AS evidence_quality, "
    "o.order_basis AS evidence_order_basis, "
    "o.order_gap AS evidence_order_gap, "
    "o.twin_id AS evidence_twin_id, "
    "o.attachment_generation AS evidence_attachment_generation, "
    "o.attachment_bound_at_ns AS evidence_attachment_bound_at_ns, "
    "o.topology_revision AS evidence_topology_revision "
    "FROM reality_deliveries AS d LEFT JOIN reality_observations AS o "
    "ON o.record_id=d.record_id"
)
_META_COUNTERS = (
    "alarm_events_pruned_total",
    "capacity_refusals_total",
    "observations_pruned_total",
    "quarantine_pruned_total",
    "recovered_inflight_total",
    "terminal_deliveries_pruned_total",
)
_LEGACY_OBSERVATION_COLUMNS = (
    "record_id",
    "adapter_id",
    "channel_id",
    "declaration_sha256",
    "reading_sha256",
    "event_sha256",
    "captured_at_ns",
    "ingested_at_ns",
    "ingested_monotonic_ns",
    "session_id",
    "sequence",
    "status",
    "quality",
    "order_basis",
    "order_gap",
    "value",
    "unit",
    "source",
    "uncertainty",
    "error_code",
    "declaration_json",
    "reading_json",
    "recorded_at",
)
_SCHEMA_COLUMNS: dict[str, tuple[str, ...]] = {
    "reality_historian_meta": ("key", "value"),
    "reality_observations": (
        "record_id",
        "adapter_id",
        "channel_id",
        "declaration_sha256",
        "reading_sha256",
        "event_sha256",
        "captured_at_ns",
        "ingested_at_ns",
        "ingested_monotonic_ns",
        "session_id",
        "sequence",
        "status",
        "quality",
        "order_basis",
        "order_gap",
        "value",
        "unit",
        "source",
        "uncertainty",
        "error_code",
        "declaration_json",
        "reading_json",
        "recorded_at",
        "twin_id",
        "attachment_generation",
        "attachment_bound_at_ns",
        "topology_revision",
    ),
    "reality_channel_heads": (
        "channel_id",
        "adapter_id",
        "last_seen_session_id",
        "last_seen_sequence",
        "last_seen_captured_at_ns",
        "last_seen_reading_sha256",
        "last_seen_event_sha256",
        "last_source_epoch",
        "last_source_sequence",
        "last_source_event_id",
        "last_stored_value",
        "last_stored_status",
        "last_stored_quality",
        "last_stored_record_id",
        "last_stored_at",
        "updated_at",
    ),
    "reality_quarantine": (
        "quarantine_id",
        "adapter_id",
        "channel_id",
        "reason",
        "reading_sha256",
        "payload_json",
        "created_at",
    ),
    "reality_alarm_events": (
        "event_id",
        "channel_id",
        "alarm_code",
        "severity",
        "state",
        "record_id",
        "actor",
        "created_at",
    ),
    "reality_alarm_heads": (
        "channel_id",
        "alarm_code",
        "severity",
        "active",
        "acknowledged",
        "active_since",
        "last_record_id",
        "updated_at",
    ),
    "reality_deliveries": (
        "observation_id",
        "record_id",
        "channel_id",
        "salience",
        "payload_json",
        "sink_states_json",
        "state",
        "attempts",
        "replay_count",
        "available_at",
        "last_error",
        "replacement_observation_id",
        "lease_owner",
        "lease_token",
        "lease_expires_at",
        "created_at",
        "updated_at",
    ),
    "reality_backfill_receipts": (
        "backfill_id",
        "sink",
        "record_id",
        "channel_id",
        "source_sha256",
        "target_binding_sha256",
        "sink_receipt_id",
        "receipt_id",
        "completed_at",
        "row_sha256",
    ),
}
_SCHEMA_INDEXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "reality_observations_channel_time": (
        "reality_observations",
        ("channel_id", "captured_at_ns", "record_id"),
    ),
    "reality_deliveries_state_due": (
        "reality_deliveries",
        ("state", "available_at", "created_at"),
    ),
    "reality_deliveries_channel_state": (
        "reality_deliveries",
        ("channel_id", "state", "created_at"),
    ),
}
_SCHEMA_SQL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "reality_observations": (
        "CHECK (captured_at_ns > 0 AND captured_at_ns <= 9223372036854775807)",
        "CHECK (sequence >= 0 AND sequence <= 9223372036854775807)",
        "CHECK (attachment_generation >= 0 AND attachment_generation <= 9223372036854775807)",
        "CHECK (attachment_bound_at_ns >= 0 AND attachment_bound_at_ns <= 9223372036854775807)",
        "CHECK (topology_revision >= 0 AND topology_revision <= 9223372036854775807)",
    ),
    "reality_channel_heads": (
        "CHECK (last_seen_sequence >= 0 AND last_seen_sequence <= 9223372036854775807)",
        "CHECK (last_source_sequence >= 0 AND last_source_sequence <= 9223372036854775807)",
    ),
    "reality_deliveries": (
        "FOREIGN KEY(record_id) REFERENCES reality_observations(record_id)",
        "CHECK (salience >= 0.0 AND salience <= 1.0)",
    ),
    "reality_backfill_receipts": (
        "UNIQUE(sink, record_id, target_binding_sha256)",
    ),
}


class HistorianError(RuntimeError):
    """The historian could not establish trustworthy durable state."""


class HistorianCorruptionError(HistorianError):
    """Existing durable state failed ownership, schema, or integrity checks."""


class ObservationQuality(StrEnum):
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    SIMULATED = "simulated"


class HistorianDisposition(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    DEADBAND = "deadband"
    QUARANTINED = "quarantined"
    CAPACITY_EXHAUSTED = "capacity_exhausted"


@dataclass(frozen=True, slots=True)
class HistorianAdmission:
    record_id: str
    disposition: HistorianDisposition
    reason: str
    quality: ObservationQuality
    alarm_event_ids: tuple[str, ...] = ()
    alarm_codes: tuple[str, ...] = ()
    order_basis: str = ""
    order_gap: bool = False
    delivery_observation_id: str = ""
    delivery_accepted: bool = False
    delivery_reason: str = "not_requested"
    delivery_queue_depth: int = 0
    superseded_delivery_ids: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.disposition == HistorianDisposition.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "quality": self.quality.value,
            "alarm_event_ids": list(self.alarm_event_ids),
            "alarm_codes": list(self.alarm_codes),
            "order_basis": self.order_basis,
            "order_gap": self.order_gap,
            "delivery_observation_id": self.delivery_observation_id,
            "delivery_accepted": self.delivery_accepted,
            "delivery_reason": self.delivery_reason,
            "delivery_queue_depth": self.delivery_queue_depth,
            "superseded_delivery_ids": list(self.superseded_delivery_ids),
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class HistorianDelivery:
    observation_id: str
    record_id: str
    payload: dict[str, Any]
    sink_states: dict[str, dict[str, str]]
    attempts: int
    replay_count: int
    lease_token: str
    lease_expires_at: float


@dataclass(frozen=True, slots=True)
class HistorianHeadSnapshot:
    """One authoritative current channel head eligible for projection repair."""

    record_id: str
    adapter_id: str
    adapter_identity_sha256: str
    adapter_registration_generation: int
    adapter_identity_stable: bool
    channel_id: str
    declaration: dict[str, Any]
    reading: dict[str, Any]
    quality: str
    order_basis: str
    order_gap: bool
    alarm_codes: tuple[str, ...]
    recorded_at: float
    source_sha256: str


@dataclass(frozen=True, slots=True)
class HistorianHeadPage:
    """One stable page of unreceipted current heads for live adapters."""

    snapshots: tuple[HistorianHeadSnapshot, ...]
    next_channel_id: str
    exhausted: bool
    scanned: int


@dataclass(frozen=True, slots=True)
class _DeliveryAdmission:
    accepted: bool
    reason: str
    queue_depth: int
    superseded_ids: tuple[str, ...] = ()


def default_reality_historian_path() -> Path:
    override = str(os.environ.get("AURA_REALITY_HISTORIAN_DB") or "").strip()
    if override:
        return Path(override).expanduser()
    test_root = str(os.environ.get("AURA_TEST_RUNTIME_ROOT") or "").strip()
    if test_root:
        return Path(test_root).expanduser() / "reality_historian.sqlite3"
    return cast(Path, state_root()) / "data" / "reality_historian.sqlite3"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _normalized_schema_sql(value: Any) -> str:
    normalized = " ".join(str(value or "").split())
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)
    return normalized


def _identifier(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _authoritative_twin_binding(
    payload: Mapping[str, Any] | None,
    *,
    observation_id: str,
) -> tuple[str, int, int, int]:
    if payload is None or str(payload.get("schema") or "") != "aura.reality-observation.v2":
        return "", 0, 0, 0
    raw_binding = payload.get("twin_binding")
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != _TWIN_BINDING_KEYS:
        raise HistorianCorruptionError("Reality delivery twin binding manifest differs")
    binding = dict(raw_binding)
    supplied_digest = str(binding.pop("binding_sha256", ""))
    twin_id = str(binding.get("twin_id") or "")
    if twin_id:
        _identifier(twin_id, name="twin_id")
    values: list[int] = []
    for field in (
        "attachment_generation",
        "attachment_bound_at_ns",
        "topology_revision",
    ):
        value = binding.get(field)
        minimum = 1 if twin_id else 0
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not (minimum <= value <= _SQLITE_INT_MAX)
        ):
            raise HistorianCorruptionError(f"Reality delivery twin {field} is invalid")
        if not twin_id and value != 0:
            raise HistorianCorruptionError("Reality delivery carries a partial twin attachment fence")
        values.append(int(value))
    expected_digest = _digest({"observation_id": observation_id, "binding": binding})
    if not secrets.compare_digest(supplied_digest, expected_digest):
        raise HistorianCorruptionError("Reality delivery twin binding digest differs")
    return twin_id, values[0], values[1], values[2]


def _backfill_row_sha256(values: Mapping[str, Any]) -> str:
    evidence = {key: value for key, value in values.items() if key != "row_sha256"}
    return _digest({"schema": "aura.reality-head-backfill-receipt.v1", **evidence})


def _validated_sink_states(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise HistorianCorruptionError("Reality delivery sink state is invalid")
    sink_states: dict[str, dict[str, str]] = {}
    for raw_sink, raw_state in value.items():
        sink = _identifier(raw_sink, name="delivery_sink")
        if not isinstance(raw_state, dict) or set(raw_state) != {
            "receipt_id",
            "state",
        }:
            raise HistorianCorruptionError("Reality delivery sink receipt is invalid")
        state = str(raw_state.get("state") or "")
        receipt_id = str(raw_state.get("receipt_id") or "")
        if state not in _DELIVERY_SINK_STATES or len(receipt_id) > 256:
            raise HistorianCorruptionError(
                "Reality delivery sink receipt differs from its contract"
            )
        if state == "pending" and receipt_id:
            raise HistorianCorruptionError("Pending Reality delivery sink carries a receipt")
        if state == "delivered" and not receipt_id:
            raise HistorianCorruptionError("Delivered Reality delivery sink has no receipt")
        sink_states[sink] = {"state": state, "receipt_id": receipt_id}
    return sink_states


def _sink_state_envelope(
    sink_states: Mapping[str, Mapping[str, str]],
    *,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": _DELIVERY_SINK_ENVELOPE_SCHEMA,
        "payload_sha256": _digest(dict(payload)),
        "sinks": {sink: dict(state) for sink, state in sink_states.items()},
    }


def _decode_sink_state_envelope(
    value: Any,
    *,
    payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {
        "payload_sha256",
        "schema",
        "sinks",
    }:
        raise HistorianCorruptionError(
            "Reality delivery sink-state envelope differs from its contract"
        )
    if value.get("schema") != _DELIVERY_SINK_ENVELOPE_SCHEMA:
        raise HistorianCorruptionError("Reality delivery sink-state schema differs")
    expected_digest = _digest(dict(payload))
    if not secrets.compare_digest(
        str(value.get("payload_sha256") or ""),
        expected_digest,
    ):
        raise HistorianCorruptionError("Reality delivery payload integrity binding differs")
    return _validated_sink_states(value.get("sinks"))


def _validate_observation_delivery_binding(
    row: sqlite3.Row,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    observation_id = _identifier(row["observation_id"], name="observation_id")
    record_id = _identifier(row["record_id"], name="record_id")
    if payload.get("observation_id") not in {None, observation_id}:
        raise HistorianCorruptionError("Reality delivery observation identity differs from its row")
    observation_schema = str(payload.get("schema") or "")
    if observation_schema not in _OBSERVATION_SCHEMAS:
        return observation_id, record_id
    historian = payload.get("historian")
    if not isinstance(historian, dict) or set(historian) != _HISTORIAN_EVIDENCE_KEYS:
        raise HistorianCorruptionError("Reality delivery historian evidence manifest differs")
    if historian.get("schema") != _HISTORIAN_EVIDENCE_SCHEMA:
        raise HistorianCorruptionError("Reality delivery historian evidence schema differs")
    order_gap = historian.get("order_gap")
    if not isinstance(order_gap, bool):
        raise HistorianCorruptionError("Reality delivery historian order-gap evidence is invalid")
    alarm_codes = historian.get("alarm_codes")
    if (
        not isinstance(alarm_codes, list)
        or len(alarm_codes) > 8
        or any(not _IDENTIFIER.fullmatch(str(item)) for item in alarm_codes)
    ):
        raise HistorianCorruptionError("Reality delivery historian alarm evidence is invalid")
    quality = str(historian.get("quality") or "")
    try:
        authoritative_status = ReadingStatus(str(row["evidence_status"] or ""))
        authoritative_quality = ObservationQuality(str(row["evidence_quality"] or ""))
    except ValueError as exc:
        raise HistorianCorruptionError(
            "Reality delivery authoritative alarm evidence is invalid"
        ) from exc
    _severity, authoritative_alarm_code = _alarm_condition(
        authoritative_status,
        authoritative_quality,
        order_gap=bool(row["evidence_order_gap"]),
    )
    authoritative_alarm_codes = [authoritative_alarm_code] if authoritative_alarm_code else []
    expected_reason = "accepted_with_source_gap" if order_gap else "accepted"
    expected = {
        "record_id": record_id,
        "quality": str(row["evidence_quality"] or ""),
        "order_basis": str(row["evidence_order_basis"] or ""),
        "order_gap": bool(row["evidence_order_gap"]),
        "adapter_id": str(row["evidence_adapter_id"] or ""),
        "declaration_sha256": str(row["evidence_declaration_sha256"] or ""),
        "reading_sha256": str(row["evidence_reading_sha256"] or ""),
    }
    if (
        historian.get("record_id") != expected["record_id"]
        or quality != expected["quality"]
        or historian.get("order_basis") != expected["order_basis"]
        or order_gap is not expected["order_gap"]
        or historian.get("reason") != expected_reason
        or payload.get("adapter_id") != expected["adapter_id"]
        or payload.get("declaration_sha256") != expected["declaration_sha256"]
        or payload.get("reading_sha256") != expected["reading_sha256"]
        or alarm_codes != authoritative_alarm_codes
    ):
        raise HistorianCorruptionError(
            "Reality delivery historian evidence differs from its record"
        )
    base_payload = dict(payload)
    base_payload.pop("historian", None)
    binding_evidence = dict(historian)
    supplied_binding = str(binding_evidence.pop("binding_sha256", ""))
    expected_binding = _digest({"observation": base_payload, "historian": binding_evidence})
    if not secrets.compare_digest(supplied_binding, expected_binding):
        raise HistorianCorruptionError("Reality delivery historian evidence binding differs")
    if observation_schema == "aura.reality-observation.v2":
        twin_id, generation, bound_at_ns, revision = _authoritative_twin_binding(
            payload,
            observation_id=observation_id,
        )
        authoritative = (
            str(row["evidence_twin_id"] or ""),
            int(row["evidence_attachment_generation"] or 0),
            int(row["evidence_attachment_bound_at_ns"] or 0),
            int(row["evidence_topology_revision"] or 0),
        )
        if (twin_id, generation, bound_at_ns, revision) != authoritative:
            raise HistorianCorruptionError(
                "Reality delivery twin binding differs from its authoritative record"
            )
    return observation_id, record_id


def _quality(reading: ChannelReading) -> ObservationQuality:
    if reading.status == ReadingStatus.AVAILABLE:
        status_quality = ObservationQuality.GOOD
    elif reading.status == ReadingStatus.SIMULATED:
        status_quality = ObservationQuality.SIMULATED
    elif reading.status == ReadingStatus.STALE:
        status_quality = ObservationQuality.STALE
    elif reading.status in {ReadingStatus.DEGRADED, ReadingStatus.UNCALIBRATED}:
        status_quality = ObservationQuality.UNCERTAIN
    else:
        status_quality = ObservationQuality.BAD
    native = str(reading.source_quality or "").strip().lower()
    if not native:
        return status_quality
    try:
        native_quality = ObservationQuality(native)
    except ValueError:
        native_quality = ObservationQuality.UNCERTAIN
    rank = {
        ObservationQuality.GOOD: 0,
        ObservationQuality.SIMULATED: 1,
        ObservationQuality.UNCERTAIN: 2,
        ObservationQuality.STALE: 3,
        ObservationQuality.BAD: 4,
    }
    return max((status_quality, native_quality), key=rank.__getitem__)


def _alarm_severity(status: ReadingStatus) -> str:
    if status in {ReadingStatus.PERMISSION_DENIED, ReadingStatus.UNAVAILABLE}:
        return "high"
    if status in {
        ReadingStatus.DEGRADED,
        ReadingStatus.STALE,
        ReadingStatus.UNCALIBRATED,
    }:
        return "medium"
    return "none"


def _alarm_condition(
    status: ReadingStatus,
    quality: ObservationQuality,
    *,
    order_gap: bool,
) -> tuple[str, str]:
    status_severity = _alarm_severity(status)
    if status_severity != "none":
        return status_severity, f"reading_{status.value}"
    if quality == ObservationQuality.BAD:
        return "high", "source_quality_bad"
    if order_gap:
        return "medium", "source_order_gap"
    if quality == ObservationQuality.STALE:
        return "medium", "source_quality_stale"
    if quality == ObservationQuality.UNCERTAIN:
        return "medium", "source_quality_uncertain"
    return "none", ""


class RealityHistorian:
    """Crash-consistent history and delivery owner for physical observations."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        clock: Callable[[], float] = lambda: time.time(),
        busy_timeout_s: float = 3.0,
        max_records: int = 100_000,
        max_quarantine: int = 5_000,
        max_alarm_events: int = 10_000,
        max_delivery_attempts: int = 8,
        max_silence_s: float = 300.0,
        delivery_lease_s: float = 30.0,
        max_age_s: float = 30.0 * 86_400.0,
        max_storage_bytes: int = 256 * 1024 * 1024,
        min_free_bytes: int = 512 * 1024 * 1024,
        disk_free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self.db_path = Path(db_path or default_reality_historian_path()).expanduser()
        self._clock = clock
        self._busy_timeout_s = max(0.1, min(float(busy_timeout_s), 30.0))
        self._max_records = max(8, min(int(max_records), 1_000_000))
        self._max_quarantine = max(8, min(int(max_quarantine), 100_000))
        self._max_alarm_events = max(8, min(int(max_alarm_events), 100_000))
        self._max_delivery_attempts = max(1, min(int(max_delivery_attempts), 32))
        self._max_silence_s = max(1.0, min(float(max_silence_s), 86_400.0))
        self._delivery_lease_s = max(5.0, min(float(delivery_lease_s), 300.0))
        self._max_age_s = max(60.0, min(float(max_age_s), 10.0 * 365.25 * 86_400.0))
        self._max_storage_bytes = max(
            128 * 1024,
            min(int(max_storage_bytes), 16 * 1024 * 1024 * 1024),
        )
        self._min_free_bytes = max(
            0,
            min(int(min_free_bytes), 1024 * 1024 * 1024 * 1024),
        )
        self._disk_free_bytes = disk_free_bytes or (
            lambda path: int(get_resource_observer().disk(path).free_bytes)
        )
        self._consumer_id = f"reality.historian.{os.getpid()}.{secrets.token_hex(8)}"
        self._write_lock = checked_lock("reality_historian.write", reentrant=True)
        self._health_lock = checked_lock("reality_historian.health", reentrant=True)
        self._healthy = False
        self._last_error = "not_initialized"
        self._consecutive_failures = 0
        self._last_success_at = 0.0
        self._last_failure_at = 0.0
        self._last_probe_at = 0.0
        self._last_probe_error = ""
        self._last_maintenance_at = 0.0
        try:
            self._initialize()
        except HistorianError as exc:
            self._record_health_failure(exc)
            raise
        except sqlite3.Error as exc:
            wrapped = HistorianError(
                f"Reality historian initialization failed: {type(exc).__name__}"
            )
            self._record_health_failure(wrapped)
            raise wrapped from exc
        self._record_health_success()

    def _run_serialized(self, callback: Callable[..., Any], *args: Any) -> Any:
        with self._write_lock:
            try:
                result = callback(*args)
            except HistorianError as exc:
                self._record_health_failure(exc)
                raise
            except sqlite3.Error as exc:
                wrapped = HistorianError(
                    f"Reality historian database operation failed: {type(exc).__name__}"
                )
                self._record_health_failure(wrapped)
                raise wrapped from exc
            else:
                self._record_health_success()
                return result

    def _record_health_success(self) -> None:
        with self._health_lock:
            self._healthy = True
            self._last_error = ""
            self._consecutive_failures = 0
            self._last_success_at = float(self._clock())

    def _record_health_failure(self, error: BaseException) -> None:
        with self._health_lock:
            self._healthy = False
            self._last_error = f"{type(error).__name__}:{error}"[:512]
            self._consecutive_failures += 1
            self._last_failure_at = float(self._clock())

    def _connect(self) -> sqlite3.Connection:
        self._assert_storage_paths_safe()
        connection = sqlite3.connect(
            self.db_path,
            timeout=self._busy_timeout_s,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_s * 1000)}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        return connection

    def _assert_storage_paths_safe(self) -> None:
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.is_symlink():
                raise HistorianCorruptionError(
                    f"Reality historian storage path must not be a symlink: {path.name}"
                )
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise HistorianError(
                    f"Reality historian storage path could not be inspected: {path.name}"
                ) from exc
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                raise HistorianCorruptionError(
                    f"Reality historian storage permissions are unsafe: {path.name}"
                )

    def _initialize(self) -> None:
        if self.db_path.is_symlink():
            raise HistorianCorruptionError("Reality historian path must not be a symlink")
        existed = self.db_path.exists() and self.db_path.stat().st_size > 0
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "reality_historian.initialize",
            domain="file_write",
        ):
            gateway = get_file_write_gateway()
            gateway.ensure_directory(
                self.db_path.parent,
                source="core.reality_reach.historian.initialize",
            )
            with gateway.open_owned_binary(
                self.db_path,
                mode="a+b",
                permissions=0o600,
                source="core.reality_reach.historian.initialize",
            ):
                pass
        stat = self.db_path.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise HistorianCorruptionError("Reality historian ownership or permissions are unsafe")
        connection = self._connect()
        try:
            quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
            if quick_check is None or str(quick_check[0]).casefold() != "ok":
                raise HistorianCorruptionError("Reality historian quick_check failed")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not str(row[0]).startswith("sqlite_")
            }
            if existed and tables and "reality_historian_meta" not in tables:
                raise HistorianCorruptionError("Existing Reality historian has no schema identity")
            if existed and "reality_historian_meta" in tables:
                version = connection.execute(
                    "SELECT value FROM reality_historian_meta WHERE key='schema_version'"
                ).fetchone()
                if version is None:
                    raise HistorianCorruptionError(
                        "Existing Reality historian has no schema version"
                    )
            if not tables:
                connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
                connection.execute("VACUUM")
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode is None or str(mode[0]).casefold() != "wal":
                raise HistorianError("Reality historian could not enable WAL durability")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA secure_delete=ON")
            self._migrate_schema(connection)
            self._create_schema(connection)
            self._verify_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            self._migrate_delivery_sink_envelopes(connection)
            recovered = connection.execute(
                "UPDATE reality_deliveries SET state='queued', "
                "replay_count=replay_count+1, available_at=?, "
                "last_error='recovered_expired_delivery', lease_owner='', "
                "lease_token='', lease_expires_at=0, updated_at=? "
                "WHERE state='delivering' AND lease_expires_at<=?",
                (
                    float(self._clock()),
                    float(self._clock()),
                    float(self._clock()),
                ),
            ).rowcount
            if recovered:
                self._increment_meta(
                    connection,
                    "recovered_inflight_total",
                    int(recovered),
                )
            connection.execute("COMMIT")
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not str(row[0]).startswith("sqlite_")
        }
        if "reality_historian_meta" not in tables:
            return
        version_row = connection.execute(
            "SELECT value FROM reality_historian_meta WHERE key='schema_version'"
        ).fetchone()
        if version_row is None:
            raise HistorianCorruptionError("Existing Reality historian has no schema version")
        try:
            version = int(version_row[0])
        except (TypeError, ValueError) as exc:
            raise HistorianCorruptionError("Reality historian schema version is invalid") from exc
        if version == _SCHEMA_VERSION:
            return
        if version != _LEGACY_SCHEMA_VERSION:
            raise HistorianCorruptionError("Unsupported Reality historian schema")
        expected_legacy_tables = set(_SCHEMA_COLUMNS) - {"reality_backfill_receipts"}
        if tables != expected_legacy_tables:
            raise HistorianCorruptionError(
                "Legacy Reality historian table manifest differs from schema identity"
            )
        legacy_columns = tuple(
            str(row[1])
            for row in connection.execute(
                'PRAGMA table_info("reality_observations")'
            ).fetchall()
        )
        if legacy_columns != _LEGACY_OBSERVATION_COLUMNS:
            raise HistorianCorruptionError(
                "Legacy Reality historian observation contract differs"
            )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "ALTER TABLE reality_observations "
                "ADD COLUMN twin_id TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "ALTER TABLE reality_observations ADD COLUMN attachment_generation "
                "INTEGER NOT NULL DEFAULT 0 CHECK (attachment_generation >= 0 "
                "AND attachment_generation <= 9223372036854775807)"
            )
            connection.execute(
                "ALTER TABLE reality_observations ADD COLUMN attachment_bound_at_ns "
                "INTEGER NOT NULL DEFAULT 0 CHECK (attachment_bound_at_ns >= 0 "
                "AND attachment_bound_at_ns <= 9223372036854775807)"
            )
            connection.execute(
                "ALTER TABLE reality_observations ADD COLUMN topology_revision "
                "INTEGER NOT NULL DEFAULT 0 CHECK (topology_revision >= 0 "
                "AND topology_revision <= 9223372036854775807)"
            )
            updated = connection.execute(
                "UPDATE reality_historian_meta SET value=? "
                "WHERE key='schema_version' AND value=?",
                (str(_SCHEMA_VERSION), str(_LEGACY_SCHEMA_VERSION)),
            )
            if updated.rowcount != 1:
                raise HistorianCorruptionError(
                    "Reality historian schema migration lost its compare-and-swap"
                )
            connection.execute("COMMIT")
        except Exception:
            _roll_back(connection)
            raise

    @staticmethod
    def _migrate_delivery_sink_envelopes(connection: sqlite3.Connection) -> None:
        rows = connection.execute(_DELIVERY_SELECT).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
                raw_sink_state = json.loads(str(row["sink_states_json"]))
            except (TypeError, ValueError) as exc:
                raise HistorianCorruptionError("Reality delivery durable JSON is invalid") from exc
            if not isinstance(payload, dict):
                raise HistorianCorruptionError("Reality delivery payload is not an object")
            _validate_observation_delivery_binding(row, payload)
            if (
                isinstance(raw_sink_state, dict)
                and raw_sink_state.get("schema") == _DELIVERY_SINK_ENVELOPE_SCHEMA
            ):
                _decode_sink_state_envelope(raw_sink_state, payload=payload)
                continue
            sink_states = _validated_sink_states(raw_sink_state)
            envelope = _sink_state_envelope(sink_states, payload=payload)
            updated = connection.execute(
                "UPDATE reality_deliveries SET sink_states_json=? "
                "WHERE observation_id=? AND sink_states_json=?",
                (
                    canonical_json(envelope).decode("utf-8"),
                    str(row["observation_id"]),
                    str(row["sink_states_json"]),
                ),
            )
            if updated.rowcount != 1:
                raise HistorianCorruptionError("Reality delivery sink-state migration lost its row")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reality_historian_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        version = connection.execute(
            "SELECT value FROM reality_historian_meta WHERE key='schema_version'"
        ).fetchone()
        if version is not None:
            try:
                schema_version = int(version[0])
            except (TypeError, ValueError) as exc:
                raise HistorianCorruptionError(
                    "Reality historian schema version is invalid"
                ) from exc
            if schema_version != _SCHEMA_VERSION:
                raise HistorianCorruptionError("Unsupported Reality historian schema")
        connection.execute(
            "INSERT OR IGNORE INTO reality_historian_meta(key, value) VALUES('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO reality_historian_meta(key, value) VALUES(?, '0')",
            ((key,) for key in _META_COUNTERS),
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_observations (
                record_id TEXT PRIMARY KEY,
                adapter_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                declaration_sha256 TEXT NOT NULL,
                reading_sha256 TEXT NOT NULL,
                event_sha256 TEXT NOT NULL,
                captured_at_ns INTEGER NOT NULL CHECK (
                    captured_at_ns > 0 AND captured_at_ns <= 9223372036854775807
                ),
                ingested_at_ns INTEGER NOT NULL CHECK (
                    ingested_at_ns >= 0 AND ingested_at_ns <= 9223372036854775807
                ),
                ingested_monotonic_ns INTEGER NOT NULL CHECK (
                    ingested_monotonic_ns >= 0 AND ingested_monotonic_ns <= 9223372036854775807
                ),
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK (
                    sequence >= 0 AND sequence <= 9223372036854775807
                ),
                status TEXT NOT NULL,
                quality TEXT NOT NULL,
                order_basis TEXT NOT NULL,
                order_gap INTEGER NOT NULL CHECK (order_gap IN (0,1)),
                value REAL,
                unit TEXT NOT NULL,
                source TEXT NOT NULL,
                uncertainty REAL,
                error_code TEXT NOT NULL,
                declaration_json TEXT NOT NULL,
                reading_json TEXT NOT NULL,
                recorded_at REAL NOT NULL,
                twin_id TEXT NOT NULL DEFAULT '',
                attachment_generation INTEGER NOT NULL DEFAULT 0 CHECK (
                    attachment_generation >= 0
                    AND attachment_generation <= 9223372036854775807
                ),
                attachment_bound_at_ns INTEGER NOT NULL DEFAULT 0 CHECK (
                    attachment_bound_at_ns >= 0
                    AND attachment_bound_at_ns <= 9223372036854775807
                ),
                topology_revision INTEGER NOT NULL DEFAULT 0 CHECK (
                    topology_revision >= 0
                    AND topology_revision <= 9223372036854775807
                )
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reality_observations_channel_time "
            "ON reality_observations(channel_id, captured_at_ns, record_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_channel_heads (
                channel_id TEXT PRIMARY KEY,
                adapter_id TEXT NOT NULL,
                last_seen_session_id TEXT NOT NULL,
                last_seen_sequence INTEGER NOT NULL CHECK (
                    last_seen_sequence >= 0 AND last_seen_sequence <= 9223372036854775807
                ),
                last_seen_captured_at_ns INTEGER NOT NULL CHECK (
                    last_seen_captured_at_ns > 0
                    AND last_seen_captured_at_ns <= 9223372036854775807
                ),
                last_seen_reading_sha256 TEXT NOT NULL,
                last_seen_event_sha256 TEXT NOT NULL,
                last_source_epoch TEXT NOT NULL,
                last_source_sequence INTEGER NOT NULL CHECK (
                    last_source_sequence >= 0
                    AND last_source_sequence <= 9223372036854775807
                ),
                last_source_event_id TEXT NOT NULL,
                last_stored_value REAL,
                last_stored_status TEXT NOT NULL,
                last_stored_quality TEXT NOT NULL,
                last_stored_record_id TEXT NOT NULL,
                last_stored_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_quarantine (
                quarantine_id TEXT PRIMARY KEY,
                adapter_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                reading_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_alarm_events (
                event_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                alarm_code TEXT NOT NULL,
                severity TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('active','acknowledged','cleared')),
                record_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_alarm_heads (
                channel_id TEXT PRIMARY KEY,
                alarm_code TEXT NOT NULL,
                severity TEXT NOT NULL,
                active INTEGER NOT NULL CHECK (active IN (0,1)),
                acknowledged INTEGER NOT NULL CHECK (acknowledged IN (0,1)),
                active_since REAL NOT NULL,
                last_record_id TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_deliveries (
                observation_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL UNIQUE,
                channel_id TEXT NOT NULL,
                salience REAL NOT NULL CHECK (salience >= 0.0 AND salience <= 1.0),
                payload_json TEXT NOT NULL,
                sink_states_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('queued','delivering','delivered','superseded','quarantined')
                ),
                attempts INTEGER NOT NULL CHECK (attempts >= 0),
                replay_count INTEGER NOT NULL CHECK (replay_count >= 0),
                available_at REAL NOT NULL,
                last_error TEXT NOT NULL,
                replacement_observation_id TEXT NOT NULL,
                lease_owner TEXT NOT NULL,
                lease_token TEXT NOT NULL,
                lease_expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(record_id) REFERENCES reality_observations(record_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reality_deliveries_state_due "
            "ON reality_deliveries(state, available_at, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reality_deliveries_channel_state "
            "ON reality_deliveries(channel_id, state, created_at)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reality_backfill_receipts (
                backfill_id TEXT PRIMARY KEY,
                sink TEXT NOT NULL,
                record_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                target_binding_sha256 TEXT NOT NULL,
                sink_receipt_id TEXT NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                completed_at REAL NOT NULL,
                row_sha256 TEXT NOT NULL,
                UNIQUE(sink, record_id, target_binding_sha256)
            )
            """
        )

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        auto_vacuum = connection.execute("PRAGMA auto_vacuum").fetchone()
        if auto_vacuum is None or int(auto_vacuum[0]) != 2:
            raise HistorianCorruptionError("Reality historian incremental-vacuum contract differs")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not str(row[0]).startswith("sqlite_")
        }
        expected_tables = set(_SCHEMA_COLUMNS)
        if tables != expected_tables:
            raise HistorianCorruptionError(
                "Reality historian table manifest differs from schema identity"
            )
        reference = sqlite3.connect(":memory:", isolation_level=None)
        reference.row_factory = sqlite3.Row
        try:
            RealityHistorian._create_schema(reference)
            for table, expected_columns in _SCHEMA_COLUMNS.items():
                actual_info = tuple(
                    (
                        int(row[0]),
                        str(row[1]),
                        str(row[2]).upper(),
                        int(row[3]),
                        row[4],
                        int(row[5]),
                    )
                    for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                )
                expected_info = tuple(
                    (
                        int(row[0]),
                        str(row[1]),
                        str(row[2]).upper(),
                        int(row[3]),
                        row[4],
                        int(row[5]),
                    )
                    for row in reference.execute(f'PRAGMA table_info("{table}")').fetchall()
                )
                columns = tuple(item[1] for item in actual_info)
                if columns != expected_columns or actual_info != expected_info:
                    raise HistorianCorruptionError(
                        f"Reality historian table contract differs: {table}"
                    )
                actual_foreign_keys = tuple(
                    tuple(row)
                    for row in connection.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                )
                expected_foreign_keys = tuple(
                    tuple(row)
                    for row in reference.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                )
                if actual_foreign_keys != expected_foreign_keys:
                    raise HistorianCorruptionError(
                        f"Reality historian foreign-key contract differs: {table}"
                    )
            expected_schema_sql = {
                str(row[0]): _normalized_schema_sql(row[1])
                for row in reference.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            reference.close()
        schema_sql = {
            str(row[0]): _normalized_schema_sql(row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in expected_tables:
            if schema_sql.get(table, "") != expected_schema_sql.get(table, ""):
                raise HistorianCorruptionError(f"Reality historian canonical DDL differs: {table}")
        for table, required_fragments in _SCHEMA_SQL_REQUIREMENTS.items():
            table_sql = schema_sql.get(table, "")
            if any(
                _normalized_schema_sql(fragment) not in table_sql for fragment in required_fragments
            ):
                raise HistorianCorruptionError(
                    f"Reality historian constraint contract differs: {table}"
                )
        indexes = {
            str(row[0]): (str(row[1]), str(row[2] or ""))
            for row in connection.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'"
            ).fetchall()
        }
        if set(indexes) != set(_SCHEMA_INDEXES):
            raise HistorianCorruptionError(
                "Reality historian index manifest differs from schema identity"
            )
        for index_name, (table, expected_columns) in _SCHEMA_INDEXES.items():
            if indexes[index_name][0] != table:
                raise HistorianCorruptionError(
                    f"Reality historian index owner differs: {index_name}"
                )
            columns = tuple(
                str(row[2])
                for row in connection.execute(f'PRAGMA index_info("{index_name}")').fetchall()
            )
            if columns != expected_columns:
                raise HistorianCorruptionError(
                    f"Reality historian index contract differs: {index_name}"
                )
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise HistorianCorruptionError("Reality historian foreign-key integrity check failed")
        for row in connection.execute("SELECT * FROM reality_backfill_receipts").fetchall():
            values = dict(row)
            supplied = str(values.get("row_sha256") or "")
            if not _DIGEST.fullmatch(supplied) or not secrets.compare_digest(
                supplied,
                _backfill_row_sha256(values),
            ):
                raise HistorianCorruptionError(
                    "Reality historian backfill receipt integrity differs"
                )
        counter_rows = {
            str(row[0]): row[1]
            for row in connection.execute(
                "SELECT key, value FROM reality_historian_meta WHERE key IN ("
                + ",".join("?" for _ in _META_COUNTERS)
                + ")",
                _META_COUNTERS,
            ).fetchall()
        }
        if set(counter_rows) != set(_META_COUNTERS):
            raise HistorianCorruptionError(
                "Reality historian counter manifest differs from schema identity"
            )
        for key, raw_value in counter_rows.items():
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise HistorianCorruptionError(
                    f"Reality historian counter is invalid: {key}"
                ) from exc
            if value < 0:
                raise HistorianCorruptionError(f"Reality historian counter is invalid: {key}")

    async def admit(
        self,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        *,
        adapter_id: str,
        source_deadband: float | None = None,
        delivery_observation_id: str = "",
        delivery_payload: Mapping[str, Any] | None = None,
        delivery_queue_limit: int = 8192,
        delivery_salience: float = 0.0,
        delivery_required_sinks: tuple[str, ...] = (),
    ) -> HistorianAdmission:
        return await asyncio.to_thread(
            self._run_serialized,
            self._admit_sync,
            declaration,
            reading,
            adapter_id,
            source_deadband,
            delivery_observation_id,
            delivery_payload,
            delivery_queue_limit,
            delivery_salience,
            delivery_required_sinks,
        )

    def _admit_sync(
        self,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        adapter_id: str,
        source_deadband: float | None,
        delivery_observation_id: str,
        delivery_payload: Mapping[str, Any] | None,
        delivery_queue_limit: int,
        delivery_salience: float,
        delivery_required_sinks: tuple[str, ...],
    ) -> HistorianAdmission:
        if declaration.kind != ChannelKind.SENSOR:
            raise ValueError("Reality historian accepts sensor declarations only")
        if reading.channel_id != declaration.channel_id or reading.unit != declaration.unit:
            raise ValueError("Reality historian reading differs from its declaration")
        owner = _identifier(adapter_id, name="adapter_id")
        deadband = declaration.resolution if source_deadband is None else float(source_deadband)
        if not math.isfinite(deadband) or deadband < 0.0:
            raise ValueError("source_deadband must be finite and non-negative")
        deadband = max(deadband, declaration.resolution)
        quality = _quality(reading)
        reading_sha256 = reading.sha256
        event_sha256 = reading.event_sha256
        declaration_sha256 = declaration.sha256
        if (
            not _DIGEST.fullmatch(reading_sha256)
            or not _DIGEST.fullmatch(declaration_sha256)
            or not _DIGEST.fullmatch(event_sha256)
        ):
            raise ValueError("Reality historian requires canonical evidence digests")
        declaration_json = canonical_json(declaration.to_dict())
        reading_json = canonical_json(reading.to_dict())
        if (
            len(declaration_json) > _MAX_EVIDENCE_PAYLOAD_BYTES
            or len(reading_json) > _MAX_EVIDENCE_PAYLOAD_BYTES
        ):
            raise ValueError("Reality historian evidence exceeds its bounded contract")
        delivery_id = ""
        delivery_payload_dict: dict[str, Any] | None = None
        authoritative_twin = ("", 0, 0, 0)
        queue_limit = max(1, min(int(delivery_queue_limit), 8192))
        salience = float(delivery_salience)
        if not math.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("delivery_salience must lie inside [0, 1]")
        required_sinks = tuple(
            sorted(
                {
                    _identifier(value, name="delivery_required_sink")
                    for value in delivery_required_sinks
                }
            )
        )
        if len(required_sinks) > 16:
            raise ValueError("Reality delivery sink manifest exceeds its bound")
        if delivery_observation_id or delivery_payload is not None:
            if not delivery_observation_id or delivery_payload is None:
                raise ValueError("Reality delivery identity and payload must be supplied together")
            delivery_id = _identifier(
                delivery_observation_id,
                name="delivery_observation_id",
            )
            delivery_payload_dict = dict(delivery_payload)
            if delivery_payload_dict.get("observation_id") not in {None, delivery_id}:
                raise ValueError("Reality delivery observation identity differs")
            authoritative_twin = _authoritative_twin_binding(
                delivery_payload_dict,
                observation_id=delivery_id,
            )
            encoded_delivery = canonical_json(delivery_payload_dict)
            if not encoded_delivery or len(encoded_delivery) > _MAX_DELIVERY_PAYLOAD_BYTES:
                raise ValueError("Reality delivery payload exceeds its bounded contract")
        estimated_bytes = (
            len(declaration_json)
            + len(reading_json)
            + (
                len(canonical_json(delivery_payload_dict))
                if delivery_payload_dict is not None
                else 0
            )
            + 16 * 1024
        )
        self._prepare_storage_budget(
            now=float(self._clock()),
            estimated_bytes=estimated_bytes,
        )
        record_id = (
            "reality.hist."
            + _digest(
                {
                    "adapter_id": owner,
                    "captured_at_ns": reading.captured_at_ns,
                    "channel_id": reading.channel_id,
                    "reading_sha256": reading_sha256,
                    "sequence": reading.sequence,
                    "session_id": reading.session_id,
                }
            ).removeprefix("sha256:")[:40]
        )
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_expired_records(connection, now=now)
            if not self._storage_budget_available(estimated_bytes=estimated_bytes):
                self._increment_meta(connection, "capacity_refusals_total", 1)
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id="",
                    disposition=HistorianDisposition.CAPACITY_EXHAUSTED,
                    reason="historian_storage_budget_exhausted",
                    quality=quality,
                )
            head = connection.execute(
                "SELECT * FROM reality_channel_heads WHERE channel_id=?",
                (reading.channel_id,),
            ).fetchone()
            order_reason, order_gap, order_basis = self._order_verdict(
                head,
                reading,
                event_sha256,
                owner,
            )
            if order_reason == "duplicate":
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id=str(head["last_stored_record_id"] if head else ""),
                    disposition=HistorianDisposition.DUPLICATE,
                    reason="duplicate_source_sample",
                    quality=quality,
                    order_basis=order_basis,
                    order_gap=order_gap,
                )
            if order_reason:
                self._insert_quarantine(
                    connection,
                    adapter_id=owner,
                    declaration=declaration,
                    reading=reading,
                    reason=order_reason,
                    now=now,
                )
                self._prune(connection)
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id="",
                    disposition=HistorianDisposition.QUARANTINED,
                    reason=order_reason,
                    quality=quality,
                    order_basis=order_basis,
                    order_gap=order_gap,
                )
            if order_gap and quality == ObservationQuality.GOOD:
                quality = ObservationQuality.UNCERTAIN
            if self._is_deadbanded(
                head,
                reading,
                quality,
                deadband,
                order_gap=order_gap,
                now=now,
            ):
                self._upsert_head(
                    connection,
                    owner,
                    reading,
                    reading_sha256,
                    event_sha256,
                    now,
                    stored_value=head["last_stored_value"],
                    stored_status=str(head["last_stored_status"]),
                    stored_quality=str(head["last_stored_quality"]),
                    stored_record_id=str(head["last_stored_record_id"]),
                    stored_at=float(head["last_stored_at"]),
                )
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id=str(head["last_stored_record_id"]),
                    disposition=HistorianDisposition.DEADBAND,
                    reason="below_source_deadband",
                    quality=quality,
                    order_basis=order_basis,
                    order_gap=order_gap,
                )
            connection.execute("SAVEPOINT candidate_admission")
            connection.execute(
                "INSERT INTO reality_observations("
                "record_id, adapter_id, channel_id, declaration_sha256, reading_sha256, "
                "event_sha256, "
                "captured_at_ns, ingested_at_ns, ingested_monotonic_ns, session_id, "
                "sequence, status, quality, order_basis, order_gap, value, unit, source, "
                "uncertainty, error_code, "
                "declaration_json, reading_json, recorded_at, twin_id, "
                "attachment_generation, attachment_bound_at_ns, topology_revision) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?)",
                (
                    record_id,
                    owner,
                    reading.channel_id,
                    declaration_sha256,
                    reading_sha256,
                    event_sha256,
                    reading.captured_at_ns,
                    reading.ingested_at_ns,
                    reading.ingested_monotonic_ns,
                    reading.session_id,
                    reading.sequence,
                    reading.status.value,
                    quality.value,
                    order_basis,
                    int(order_gap),
                    reading.value,
                    reading.unit,
                    reading.source,
                    reading.uncertainty,
                    str(reading.error or "")[:512],
                    declaration_json.decode("utf-8"),
                    reading_json.decode("utf-8"),
                    now,
                    authoritative_twin[0],
                    authoritative_twin[1],
                    authoritative_twin[2],
                    authoritative_twin[3],
                ),
            )
            self._upsert_head(
                connection,
                owner,
                reading,
                reading_sha256,
                event_sha256,
                now,
                stored_value=reading.value,
                stored_status=reading.status.value,
                stored_quality=quality.value,
                stored_record_id=record_id,
                stored_at=now,
            )
            alarm_ids = self._reconcile_alarm(
                connection,
                reading=reading,
                quality=quality,
                order_gap=order_gap,
                record_id=record_id,
                now=now,
            )
            alarm_row = connection.execute(
                "SELECT alarm_code FROM reality_alarm_heads WHERE channel_id=? AND active=1",
                (reading.channel_id,),
            ).fetchone()
            alarm_codes = (str(alarm_row[0]),) if alarm_row is not None else ()
            delivery_admission = _DeliveryAdmission(
                accepted=False,
                reason="not_requested",
                queue_depth=0,
            )
            if delivery_id:
                if delivery_payload_dict is None:
                    raise HistorianCorruptionError(
                        "Reality delivery payload disappeared before admission"
                    )
                bound_payload = dict(delivery_payload_dict)
                base_payload = dict(bound_payload)
                base_payload.pop("historian", None)
                historian_evidence: dict[str, Any] = {
                    "schema": _HISTORIAN_EVIDENCE_SCHEMA,
                    "record_id": record_id,
                    "quality": quality.value,
                    "order_basis": order_basis,
                    "order_gap": bool(order_gap),
                    "alarm_codes": list(alarm_codes),
                    "reason": ("accepted_with_source_gap" if order_gap else "accepted"),
                }
                historian_evidence["binding_sha256"] = _digest(
                    {
                        "observation": base_payload,
                        "historian": historian_evidence,
                    }
                )
                bound_payload = base_payload
                bound_payload["historian"] = historian_evidence
                encoded_delivery = canonical_json(bound_payload)
                if len(encoded_delivery) > _MAX_DELIVERY_PAYLOAD_BYTES:
                    raise ValueError("Historian-bound delivery payload exceeds its contract")
                delivery_admission = self._insert_delivery_row(
                    connection,
                    observation_id=delivery_id,
                    record_id=record_id,
                    channel_id=reading.channel_id,
                    salience=salience,
                    payload_json=encoded_delivery.decode("utf-8"),
                    required_sinks=required_sinks,
                    queue_limit=queue_limit,
                    now=now,
                )
            if not self._storage_budget_available(estimated_bytes=0):
                connection.execute("ROLLBACK TO candidate_admission")
                connection.execute("RELEASE candidate_admission")
                self._increment_meta(connection, "capacity_refusals_total", 1)
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id="",
                    disposition=HistorianDisposition.CAPACITY_EXHAUSTED,
                    reason="historian_storage_budget_exhausted",
                    quality=quality,
                    order_basis=order_basis,
                    order_gap=order_gap,
                )
            if not self._ensure_record_capacity(
                connection,
                protected_record_id=record_id,
            ):
                connection.execute("ROLLBACK TO candidate_admission")
                connection.execute("RELEASE candidate_admission")
                self._increment_meta(connection, "capacity_refusals_total", 1)
                self._insert_quarantine(
                    connection,
                    adapter_id=owner,
                    declaration=declaration,
                    reading=reading,
                    reason="historian_capacity_exhausted",
                    now=now,
                )
                self._prune(connection)
                connection.execute("COMMIT")
                return HistorianAdmission(
                    record_id="",
                    disposition=HistorianDisposition.CAPACITY_EXHAUSTED,
                    reason="historian_capacity_exhausted",
                    quality=quality,
                    order_basis=order_basis,
                    order_gap=order_gap,
                )
            connection.execute("RELEASE candidate_admission")
            self._prune(connection)
            connection.execute("COMMIT")
            return HistorianAdmission(
                record_id=record_id,
                disposition=HistorianDisposition.ACCEPTED,
                reason=("accepted_with_source_gap" if order_gap else "accepted"),
                quality=quality,
                alarm_event_ids=alarm_ids,
                alarm_codes=alarm_codes,
                order_basis=order_basis,
                order_gap=order_gap,
                delivery_observation_id=(delivery_id if delivery_admission.accepted else ""),
                delivery_accepted=delivery_admission.accepted,
                delivery_reason=delivery_admission.reason,
                delivery_queue_depth=delivery_admission.queue_depth,
                superseded_delivery_ids=delivery_admission.superseded_ids,
            )
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _order_verdict(
        head: sqlite3.Row | None,
        reading: ChannelReading,
        event_sha256: str,
        adapter_id: str,
    ) -> tuple[str, bool, str]:
        if head is None:
            basis = (
                "source_sequence"
                if reading.source_epoch and reading.source_sequence > 0
                else "source_event_time"
                if reading.source_event_id or reading.source_epoch
                else "ingest_sequence"
            )
            return "", False, basis
        if str(head["adapter_id"]) != adapter_id:
            return "source_adapter_changed_without_migration", False, "identity"
        prior_event_sha256 = str(head["last_seen_event_sha256"])
        if prior_event_sha256 == event_sha256:
            return "duplicate", False, "event_digest"
        if reading.source_epoch and reading.source_sequence > 0:
            prior_epoch = str(head["last_source_epoch"])
            prior_source_sequence = int(head["last_source_sequence"])
            if prior_epoch == reading.source_epoch and prior_source_sequence > 0:
                if reading.source_sequence < prior_source_sequence:
                    return "source_sequence_regressed", False, "source_sequence"
                if reading.source_sequence == prior_source_sequence:
                    return (
                        "duplicate"
                        if prior_event_sha256 == event_sha256
                        else "source_sequence_conflict",
                        False,
                        "source_sequence",
                    )
                return (
                    "",
                    reading.source_sequence > prior_source_sequence + 1,
                    "source_sequence",
                )
            if prior_epoch and prior_epoch != reading.source_epoch:
                prior_capture = int(head["last_seen_captured_at_ns"])
                if reading.captured_at_ns < prior_capture:
                    return "source_epoch_time_regressed", False, "source_sequence"
                if reading.captured_at_ns == prior_capture:
                    return "", True, "source_epoch_time_tie"
            return "", False, "source_sequence"
        if reading.source_event_id or reading.source_epoch:
            prior_source_event_id = str(head["last_source_event_id"])
            if reading.source_event_id and prior_source_event_id == reading.source_event_id:
                return (
                    "duplicate"
                    if prior_event_sha256 == event_sha256
                    else "source_event_id_conflict",
                    False,
                    "source_event_time",
                )
            prior_capture = int(head["last_seen_captured_at_ns"])
            if reading.captured_at_ns < prior_capture:
                return "source_event_time_regressed", False, "source_event_time"
            if reading.captured_at_ns == prior_capture:
                return "", True, "source_event_time_tie"
            return "", False, "source_event_time"
        if str(head["last_source_epoch"]) or str(head["last_source_event_id"]):
            return "", True, "source_lineage_downgraded"
        same_session = str(head["last_seen_session_id"]) == reading.session_id
        prior_sequence = int(head["last_seen_sequence"])
        if same_session and reading.sequence > 0 and prior_sequence > 0:
            if reading.sequence < prior_sequence:
                return "ingest_sequence_regressed", False, "ingest_sequence"
            if reading.sequence == prior_sequence:
                return (
                    (
                        "duplicate"
                        if prior_event_sha256 == event_sha256
                        else "ingest_sequence_conflict"
                    ),
                    False,
                    "ingest_sequence",
                )
        if same_session and reading.sequence == 0 and prior_sequence == 0:
            prior_capture = int(head["last_seen_captured_at_ns"])
            if reading.captured_at_ns < prior_capture:
                return "ingest_capture_time_regressed", False, "ingest_time"
            if reading.captured_at_ns == prior_capture:
                return "", True, "ingest_capture_time_tie"
        if not same_session:
            prior_capture = int(head["last_seen_captured_at_ns"])
            if reading.captured_at_ns < prior_capture:
                return "ingest_session_time_regressed", False, "ingest_time"
            if reading.captured_at_ns == prior_capture:
                return "", True, "ingest_session_time_tie"
        return "", False, "ingest_sequence"

    def _is_deadbanded(
        self,
        head: sqlite3.Row | None,
        reading: ChannelReading,
        quality: ObservationQuality,
        deadband: float,
        *,
        order_gap: bool,
        now: float,
    ) -> bool:
        if head is None or reading.value is None or head["last_stored_value"] is None:
            return False
        if order_gap or now - float(head["last_stored_at"]) >= self._max_silence_s:
            return False
        if str(head["last_stored_status"]) != reading.status.value:
            return False
        if str(head["last_stored_quality"]) != quality.value:
            return False
        if reading.status not in {ReadingStatus.AVAILABLE, ReadingStatus.SIMULATED}:
            return False
        return abs(float(reading.value) - float(head["last_stored_value"])) < deadband

    @staticmethod
    def _upsert_head(
        connection: sqlite3.Connection,
        adapter_id: str,
        reading: ChannelReading,
        reading_sha256: str,
        event_sha256: str,
        now: float,
        *,
        stored_value: float | None,
        stored_status: str,
        stored_quality: str,
        stored_record_id: str,
        stored_at: float,
    ) -> None:
        connection.execute(
            "INSERT INTO reality_channel_heads("
            "channel_id, adapter_id, last_seen_session_id, last_seen_sequence, "
            "last_seen_captured_at_ns, last_seen_reading_sha256, last_seen_event_sha256, "
            "last_source_epoch, last_source_sequence, last_source_event_id, "
            "last_stored_value, last_stored_status, last_stored_quality, "
            "last_stored_record_id, last_stored_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET adapter_id=excluded.adapter_id, "
            "last_seen_session_id=excluded.last_seen_session_id, "
            "last_seen_sequence=excluded.last_seen_sequence, "
            "last_seen_captured_at_ns=excluded.last_seen_captured_at_ns, "
            "last_seen_reading_sha256=excluded.last_seen_reading_sha256, "
            "last_seen_event_sha256=excluded.last_seen_event_sha256, "
            "last_source_epoch=excluded.last_source_epoch, "
            "last_source_sequence=excluded.last_source_sequence, "
            "last_source_event_id=excluded.last_source_event_id, "
            "last_stored_value=excluded.last_stored_value, "
            "last_stored_status=excluded.last_stored_status, "
            "last_stored_quality=excluded.last_stored_quality, "
            "last_stored_record_id=excluded.last_stored_record_id, "
            "last_stored_at=excluded.last_stored_at, updated_at=excluded.updated_at",
            (
                reading.channel_id,
                adapter_id,
                reading.session_id,
                reading.sequence,
                reading.captured_at_ns,
                reading_sha256,
                event_sha256,
                reading.source_epoch,
                reading.source_sequence,
                reading.source_event_id,
                stored_value,
                stored_status,
                stored_quality,
                stored_record_id,
                stored_at,
                now,
            ),
        )

    @staticmethod
    def _insert_quarantine(
        connection: sqlite3.Connection,
        *,
        adapter_id: str,
        declaration: ChannelDeclaration,
        reading: ChannelReading,
        reason: str,
        now: float,
    ) -> str:
        payload = {
            "adapter_id": adapter_id,
            "declaration": declaration.to_dict(),
            "reading": reading.to_dict(),
            "reason": reason,
        }
        quarantine_id = (
            "reality.quarantine."
            + _digest({"payload": payload, "created_at": now}).removeprefix("sha256:")[:36]
        )
        connection.execute(
            "INSERT INTO reality_quarantine("
            "quarantine_id, adapter_id, channel_id, reason, reading_sha256, "
            "payload_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                quarantine_id,
                adapter_id,
                reading.channel_id,
                reason,
                reading.sha256,
                canonical_json(payload).decode("utf-8"),
                now,
            ),
        )
        return quarantine_id

    def _reconcile_alarm(
        self,
        connection: sqlite3.Connection,
        *,
        reading: ChannelReading,
        quality: ObservationQuality,
        order_gap: bool,
        record_id: str,
        now: float,
    ) -> tuple[str, ...]:
        severity, alarm_code = _alarm_condition(
            reading.status,
            quality,
            order_gap=order_gap,
        )
        head = connection.execute(
            "SELECT * FROM reality_alarm_heads WHERE channel_id=?",
            (reading.channel_id,),
        ).fetchone()
        event_ids: list[str] = []
        if head is not None and bool(head["active"]):
            prior_code = str(head["alarm_code"])
            if severity == "none" or prior_code != alarm_code:
                event_ids.append(
                    self._insert_alarm_event(
                        connection,
                        channel_id=reading.channel_id,
                        alarm_code=prior_code,
                        severity=str(head["severity"]),
                        state="cleared",
                        record_id=record_id,
                        actor="reality_historian",
                        now=now,
                    )
                )
        if severity == "none":
            if head is not None:
                connection.execute(
                    "UPDATE reality_alarm_heads SET active=0, acknowledged=0, "
                    "last_record_id=?, updated_at=? WHERE channel_id=?",
                    (record_id, now, reading.channel_id),
                )
            return tuple(event_ids)
        if head is None or not bool(head["active"]) or str(head["alarm_code"]) != alarm_code:
            event_ids.append(
                self._insert_alarm_event(
                    connection,
                    channel_id=reading.channel_id,
                    alarm_code=alarm_code,
                    severity=severity,
                    state="active",
                    record_id=record_id,
                    actor="reality_historian",
                    now=now,
                )
            )
            connection.execute(
                "INSERT INTO reality_alarm_heads("
                "channel_id, alarm_code, severity, active, acknowledged, active_since, "
                "last_record_id, updated_at) VALUES(?, ?, ?, 1, 0, ?, ?, ?) "
                "ON CONFLICT(channel_id) DO UPDATE SET alarm_code=excluded.alarm_code, "
                "severity=excluded.severity, active=1, acknowledged=0, "
                "active_since=excluded.active_since, last_record_id=excluded.last_record_id, "
                "updated_at=excluded.updated_at",
                (reading.channel_id, alarm_code, severity, now, record_id, now),
            )
        else:
            connection.execute(
                "UPDATE reality_alarm_heads SET last_record_id=?, updated_at=? WHERE channel_id=?",
                (record_id, now, reading.channel_id),
            )
        return tuple(event_ids)

    @staticmethod
    def _insert_alarm_event(
        connection: sqlite3.Connection,
        *,
        channel_id: str,
        alarm_code: str,
        severity: str,
        state: str,
        record_id: str,
        actor: str,
        now: float,
    ) -> str:
        event_id = (
            "reality.alarm."
            + _digest(
                {
                    "actor": actor,
                    "alarm_code": alarm_code,
                    "channel_id": channel_id,
                    "record_id": record_id,
                    "state": state,
                    "time": now,
                }
            ).removeprefix("sha256:")[:40]
        )
        connection.execute(
            "INSERT INTO reality_alarm_events("
            "event_id, channel_id, alarm_code, severity, state, record_id, actor, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, channel_id, alarm_code, severity, state, record_id, actor, now),
        )
        return event_id

    def _prepare_storage_budget(self, *, now: float, estimated_bytes: int) -> None:
        current_bytes = sum(self._storage_file_bytes().values())
        maintenance_due = (
            now - self._last_maintenance_at >= 60.0
            or current_bytes + estimated_bytes > int(self._max_storage_bytes * 0.9)
        )
        if maintenance_due:
            self._maintain_storage(now=now)
            self._last_maintenance_at = now

    def _storage_budget_available(self, *, estimated_bytes: int) -> bool:
        current_bytes = sum(self._storage_file_bytes().values())
        free_bytes = self._current_disk_free_bytes()
        return bool(
            current_bytes + estimated_bytes <= self._max_storage_bytes
            and free_bytes - estimated_bytes >= self._min_free_bytes
        )

    def _prune_expired_records(
        self,
        connection: sqlite3.Connection,
        *,
        now: float,
    ) -> int:
        cutoff = now - self._max_age_s
        pruned_total = 0
        while True:
            expired_ids = [
                str(row[0])
                for row in connection.execute(
                    "SELECT o.record_id FROM reality_observations AS o "
                    "LEFT JOIN reality_deliveries AS d ON d.record_id=o.record_id "
                    "WHERE o.recorded_at<? AND (d.record_id IS NULL OR "
                    "d.state IN ('delivered','superseded','quarantined')) "
                    "ORDER BY o.recorded_at ASC LIMIT 500",
                    (cutoff,),
                ).fetchall()
            ]
            if not expired_ids:
                return pruned_total
            pruned_total += self._delete_observation_records(
                connection,
                expired_ids,
            )

    def _maintain_storage(self, now: float) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._prune_expired_records(connection, now=now)
            self._prune(connection)
            connection.execute("COMMIT")
            connection.execute("PRAGMA incremental_vacuum(256)")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def maintain(self) -> None:
        """Enforce retention while acquisition or attention is idle."""

        now = float(self._clock())
        await asyncio.to_thread(
            self._run_serialized,
            self._maintain_storage,
            now,
        )
        self._last_maintenance_at = now

    def _increment_capacity_refusal_best_effort(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            self._increment_meta(connection, "capacity_refusals_total", 1)
            connection.execute("COMMIT")
        except (HistorianError, OSError, sqlite3.Error):
            if connection is not None:
                _roll_back(connection)
        finally:
            if connection is not None:
                connection.close()

    def _delete_observation_records(
        self,
        connection: sqlite3.Connection,
        record_ids: list[str],
    ) -> int:
        if not record_ids:
            return 0
        placeholders = ",".join("?" for _ in record_ids)
        terminal_deliveries = int(
            connection.execute(
                "SELECT COUNT(*) FROM reality_deliveries WHERE record_id IN ("
                + placeholders
                + ") AND state IN ('delivered','superseded','quarantined')",
                tuple(record_ids),
            ).fetchone()[0]
        )
        connection.execute(
            "UPDATE reality_channel_heads SET last_stored_value=NULL, "
            "last_stored_status='', last_stored_quality='', "
            "last_stored_record_id='', last_stored_at=0 "
            "WHERE last_stored_record_id IN (" + placeholders + ")",
            tuple(record_ids),
        )
        connection.execute(
            "UPDATE reality_alarm_heads SET last_record_id='' "
            "WHERE last_record_id IN (" + placeholders + ")",
            tuple(record_ids),
        )
        connection.execute(
            "UPDATE reality_alarm_events SET record_id='' WHERE record_id IN ("
            + placeholders
            + ")",
            tuple(record_ids),
        )
        pruned = int(
            connection.execute(
                "DELETE FROM reality_observations WHERE record_id IN (" + placeholders + ")",
                tuple(record_ids),
            ).rowcount
        )
        self._increment_meta(connection, "observations_pruned_total", pruned)
        self._increment_meta(
            connection,
            "terminal_deliveries_pruned_total",
            terminal_deliveries,
        )
        return pruned

    def _ensure_record_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        protected_record_id: str,
    ) -> bool:
        count = int(connection.execute("SELECT COUNT(*) FROM reality_observations").fetchone()[0])
        if count <= self._max_records:
            return True
        removable = count - self._max_records
        removable_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT o.record_id FROM reality_observations AS o "
                "LEFT JOIN reality_deliveries AS d ON d.record_id=o.record_id "
                "WHERE o.record_id<>? AND (d.record_id IS NULL OR "
                "d.state IN ('delivered','superseded','quarantined')) "
                "ORDER BY o.recorded_at ASC LIMIT ?",
                (protected_record_id, removable),
            ).fetchall()
        ]
        self._delete_observation_records(connection, removable_ids)
        count = int(connection.execute("SELECT COUNT(*) FROM reality_observations").fetchone()[0])
        available = count <= self._max_records
        return available

    def _prune(self, connection: sqlite3.Connection) -> None:
        quarantine_count = int(
            connection.execute("SELECT COUNT(*) FROM reality_quarantine").fetchone()[0]
        )
        if quarantine_count > self._max_quarantine:
            pruned = connection.execute(
                "DELETE FROM reality_quarantine WHERE quarantine_id IN ("
                "SELECT quarantine_id FROM reality_quarantine ORDER BY created_at ASC LIMIT ?)",
                (quarantine_count - self._max_quarantine,),
            ).rowcount
            self._increment_meta(
                connection,
                "quarantine_pruned_total",
                int(pruned),
            )
        alarm_count = int(
            connection.execute("SELECT COUNT(*) FROM reality_alarm_events").fetchone()[0]
        )
        if alarm_count > self._max_alarm_events:
            pruned = connection.execute(
                "DELETE FROM reality_alarm_events WHERE event_id IN ("
                "SELECT event_id FROM reality_alarm_events ORDER BY created_at ASC LIMIT ?)",
                (alarm_count - self._max_alarm_events,),
            ).rowcount
            self._increment_meta(
                connection,
                "alarm_events_pruned_total",
                int(pruned),
            )

    @staticmethod
    def _increment_meta(
        connection: sqlite3.Connection,
        key: str,
        amount: int,
    ) -> None:
        if key not in _META_COUNTERS:
            raise ValueError("unknown Reality historian counter")
        increment = int(amount)
        if increment <= 0:
            return
        row = connection.execute(
            "SELECT value FROM reality_historian_meta WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            raise HistorianCorruptionError(f"Reality historian counter is missing: {key}")
        try:
            current = int(row[0])
        except (TypeError, ValueError) as exc:
            raise HistorianCorruptionError(f"Reality historian counter is invalid: {key}") from exc
        if current < 0:
            raise HistorianCorruptionError(f"Reality historian counter is invalid: {key}")
        connection.execute(
            "UPDATE reality_historian_meta SET value=? WHERE key=?",
            (str(current + increment), key),
        )

    async def enqueue_delivery(
        self,
        *,
        observation_id: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> HistorianDelivery:
        return await asyncio.to_thread(
            self._run_serialized,
            self._enqueue_delivery_sync,
            observation_id,
            record_id,
            payload,
        )

    def _insert_delivery_row(
        self,
        connection: sqlite3.Connection,
        *,
        observation_id: str,
        record_id: str,
        channel_id: str,
        salience: float,
        payload_json: str,
        required_sinks: tuple[str, ...],
        queue_limit: int,
        now: float,
    ) -> _DeliveryAdmission:
        existing = connection.execute(
            "SELECT record_id, payload_json, state FROM reality_deliveries WHERE observation_id=?",
            (observation_id,),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["record_id"]) != record_id
                or str(existing["payload_json"]) != payload_json
            ):
                raise HistorianCorruptionError(
                    "Reality delivery idempotency key conflicts with existing evidence"
                )
            return _DeliveryAdmission(
                accepted=str(existing["state"]) in {"queued", "delivering", "delivered"},
                reason="idempotent_existing_delivery",
                queue_depth=self._active_delivery_count(connection),
            )
        record_owner = connection.execute(
            "SELECT observation_id FROM reality_deliveries WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if record_owner is not None:
            raise HistorianCorruptionError(
                "Reality history record already belongs to another delivery"
            )
        superseded: list[str] = []
        pending_same_channel = connection.execute(
            "SELECT observation_id FROM reality_deliveries "
            "WHERE channel_id=? AND state='queued' ORDER BY created_at ASC",
            (channel_id,),
        ).fetchall()
        for row in pending_same_channel:
            prior_id = str(row["observation_id"])
            updated = connection.execute(
                "UPDATE reality_deliveries SET state='superseded', "
                "last_error='attention_channel_coalesced', "
                "replacement_observation_id=?, updated_at=? "
                "WHERE observation_id=? AND state='queued'",
                (observation_id, now, prior_id),
            )
            if updated.rowcount == 1:
                superseded.append(prior_id)
        active_count = self._active_delivery_count(connection)
        if active_count >= queue_limit:
            least = connection.execute(
                "SELECT observation_id, salience FROM reality_deliveries "
                "WHERE state='queued' "
                "ORDER BY salience ASC, created_at ASC LIMIT 1"
            ).fetchone()
            if least is None or float(least["salience"]) >= salience:
                return _DeliveryAdmission(
                    accepted=False,
                    reason="queue_full_lower_priority",
                    queue_depth=active_count,
                    superseded_ids=tuple(superseded),
                )
            least_id = str(least["observation_id"])
            updated = connection.execute(
                "UPDATE reality_deliveries SET state='superseded', "
                "last_error='attention_queue_evicted', "
                "replacement_observation_id=?, updated_at=? "
                "WHERE observation_id=? AND state='queued'",
                (observation_id, now, least_id),
            )
            if updated.rowcount != 1:
                raise HistorianCorruptionError(
                    "Reality delivery queue eviction lost its transaction"
                )
            superseded.append(least_id)
        sink_states = {sink: {"state": "pending", "receipt_id": ""} for sink in required_sinks}
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError) as exc:
            raise HistorianCorruptionError(
                "Reality delivery payload is invalid before admission"
            ) from exc
        if not isinstance(payload, dict):
            raise HistorianCorruptionError(
                "Reality delivery payload is not an object before admission"
            )
        sink_state_envelope = _sink_state_envelope(
            sink_states,
            payload=payload,
        )
        connection.execute(
            "INSERT INTO reality_deliveries("
            "observation_id, record_id, channel_id, salience, payload_json, "
            "sink_states_json, state, attempts, replay_count, "
            "available_at, last_error, replacement_observation_id, lease_owner, "
            "lease_token, lease_expires_at, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?, '', '', '', '', 0, ?, ?)",
            (
                observation_id,
                record_id,
                channel_id,
                salience,
                payload_json,
                canonical_json(sink_state_envelope).decode("utf-8"),
                now,
                now,
                now,
            ),
        )
        return _DeliveryAdmission(
            accepted=True,
            reason=("accepted_with_supersession" if superseded else "accepted"),
            queue_depth=self._active_delivery_count(connection),
            superseded_ids=tuple(superseded),
        )

    @staticmethod
    def _active_delivery_count(connection: sqlite3.Connection) -> int:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM reality_deliveries WHERE state IN ('queued','delivering')"
            ).fetchone()[0]
        )

    def _enqueue_delivery_sync(
        self,
        observation_id: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> HistorianDelivery:
        observation = _identifier(observation_id, name="observation_id")
        record = _identifier(record_id, name="record_id")
        encoded = canonical_json(dict(payload))
        if not encoded or len(encoded) > _MAX_DELIVERY_PAYLOAD_BYTES:
            raise ValueError("Reality delivery payload exceeds its bounded contract")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            observation_row = connection.execute(
                "SELECT channel_id FROM reality_observations WHERE record_id=?",
                (record,),
            ).fetchone()
            if observation_row is None:
                raise HistorianCorruptionError("Reality delivery references no observation")
            self._insert_delivery_row(
                connection,
                observation_id=observation,
                record_id=record,
                channel_id=str(observation_row["channel_id"]),
                salience=0.0,
                payload_json=encoded.decode("utf-8"),
                required_sinks=(),
                queue_limit=8192,
                now=now,
            )
            row = connection.execute(
                _DELIVERY_SELECT + " WHERE d.observation_id=?",
                (observation,),
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise HistorianCorruptionError("Reality delivery admission disappeared")
            return self._delivery_from_row(row)
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def claim_delivery(self, observation_id: str) -> HistorianDelivery | None:
        return await asyncio.to_thread(
            self._run_serialized,
            self._claim_delivery_sync,
            observation_id,
        )

    async def recover_inflight(self) -> int:
        """Return interrupted claims to the durable queue before a router starts."""

        return await asyncio.to_thread(
            self._run_serialized,
            self._recover_inflight_sync,
        )

    def _recover_inflight_sync(self) -> int:
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            recovered = connection.execute(
                "UPDATE reality_deliveries SET state='queued', "
                "replay_count=replay_count+1, available_at=?, "
                "last_error='recovered_expired_delivery', lease_owner='', "
                "lease_token='', lease_expires_at=0, updated_at=? "
                "WHERE state='delivering' AND lease_expires_at<=?",
                (now, now, now),
            ).rowcount
            self._increment_meta(
                connection,
                "recovered_inflight_total",
                int(recovered),
            )
            connection.execute("COMMIT")
            return int(recovered)
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    def _claim_delivery_sync(self, observation_id: str) -> HistorianDelivery | None:
        observation = _identifier(observation_id, name="observation_id")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_leases(connection, now=now)
            row = connection.execute(
                _DELIVERY_SELECT + " WHERE d.observation_id=? AND d.state='queued' "
                "AND d.available_at<=?",
                (observation, now),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            try:
                self._delivery_from_row(row)
            except HistorianCorruptionError:
                self._quarantine_corrupt_delivery(
                    connection,
                    observation_id=observation,
                    now=now,
                )
                connection.execute("COMMIT")
                raise
            lease_token = secrets.token_hex(24)
            lease_expires_at = now + self._delivery_lease_s
            updated = connection.execute(
                "UPDATE reality_deliveries SET state='delivering', attempts=attempts+1, "
                "lease_owner=?, lease_token=?, lease_expires_at=?, updated_at=? "
                "WHERE observation_id=? AND state='queued' AND available_at<=?",
                (
                    self._consumer_id,
                    lease_token,
                    lease_expires_at,
                    now,
                    observation,
                    now,
                ),
            )
            row = connection.execute(
                _DELIVERY_SELECT + " WHERE d.observation_id=?",
                (observation,),
            ).fetchone()
            if row is None or updated.rowcount != 1:
                raise HistorianCorruptionError("Reality delivery claim lost its durable row")
            delivery = self._delivery_from_row(row)
            connection.execute("COMMIT")
            return delivery
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def claim_due_deliveries(self, *, limit: int = 32) -> tuple[HistorianDelivery, ...]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._claim_due_deliveries_sync,
            limit,
        )

    def _claim_due_deliveries_sync(self, limit: int) -> tuple[HistorianDelivery, ...]:
        bounded = max(1, min(int(limit), 128))
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_leases(connection, now=now)
            candidate_rows = connection.execute(
                _DELIVERY_SELECT + " WHERE d.state='queued' AND d.available_at<=? "
                "ORDER BY d.available_at ASC, d.created_at ASC LIMIT ?",
                (now, bounded),
            ).fetchall()
            deliveries: list[HistorianDelivery] = []
            for candidate_row in candidate_rows:
                observation_id = str(candidate_row["observation_id"])
                try:
                    self._delivery_from_row(candidate_row)
                except HistorianCorruptionError:
                    self._quarantine_corrupt_delivery(
                        connection,
                        observation_id=observation_id,
                        now=now,
                    )
                    continue
                lease_token = secrets.token_hex(24)
                lease_expires_at = now + self._delivery_lease_s
                updated = connection.execute(
                    "UPDATE reality_deliveries SET state='delivering', attempts=attempts+1, "
                    "lease_owner=?, lease_token=?, lease_expires_at=?, updated_at=? "
                    "WHERE observation_id=? AND state='queued'",
                    (
                        self._consumer_id,
                        lease_token,
                        lease_expires_at,
                        now,
                        observation_id,
                    ),
                )
                if updated.rowcount != 1:
                    continue
                row = connection.execute(
                    _DELIVERY_SELECT + " WHERE d.observation_id=?",
                    (observation_id,),
                ).fetchone()
                if row is None:
                    raise HistorianCorruptionError("Reality delivery claim lost its durable row")
                deliveries.append(self._delivery_from_row(row))
            connection.execute("COMMIT")
            return tuple(deliveries)
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _quarantine_corrupt_delivery(
        connection: sqlite3.Connection,
        *,
        observation_id: str,
        now: float,
    ) -> None:
        updated = connection.execute(
            "UPDATE reality_deliveries SET state='quarantined', "
            "attempts=attempts+1, available_at=?, "
            "last_error='corrupt_delivery_payload', lease_owner='', "
            "lease_token='', lease_expires_at=0, updated_at=? "
            "WHERE observation_id=? AND state='queued'",
            (now, now, observation_id),
        )
        if updated.rowcount != 1:
            raise HistorianCorruptionError("Reality corrupt-delivery quarantine lost its row")

    @staticmethod
    def _recover_expired_leases(
        connection: sqlite3.Connection,
        *,
        now: float,
    ) -> int:
        recovered = int(
            connection.execute(
                "UPDATE reality_deliveries SET state='queued', "
                "replay_count=replay_count+1, available_at=?, "
                "last_error='recovered_expired_delivery', lease_owner='', "
                "lease_token='', lease_expires_at=0, updated_at=? "
                "WHERE state='delivering' AND lease_expires_at<=?",
                (now, now, now),
            ).rowcount
        )
        RealityHistorian._increment_meta(
            connection,
            "recovered_inflight_total",
            recovered,
        )
        return recovered

    async def mark_sink_delivered(
        self,
        observation_id: str,
        *,
        sink: str,
        receipt_id: str,
        lease_token: str,
    ) -> bool:
        return await asyncio.to_thread(
            self._run_serialized,
            self._mark_sink_delivered_sync,
            observation_id,
            sink,
            receipt_id,
            lease_token,
        )

    def _mark_sink_delivered_sync(
        self,
        observation_id: str,
        sink: str,
        receipt_id: str,
        lease_token: str,
    ) -> bool:
        observation = _identifier(observation_id, name="observation_id")
        sink_id = _identifier(sink, name="delivery_sink")
        receipt = str(receipt_id or "").strip()
        if not receipt or len(receipt) > 256:
            raise ValueError("delivery sink receipt_id must be present and bounded")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, lease_token, payload_json, sink_states_json "
                "FROM reality_deliveries WHERE observation_id=?",
                (observation,),
            ).fetchone()
            if row is None:
                raise HistorianCorruptionError("Reality delivery receipt is missing")
            if str(row["state"]) != "delivering":
                raise HistorianCorruptionError(
                    "only a claimed Reality delivery may accept a sink receipt"
                )
            if not secrets.compare_digest(str(row["lease_token"]), str(lease_token)):
                raise HistorianCorruptionError("Reality delivery lease token differs")
            try:
                payload = json.loads(str(row["payload_json"]))
                sink_envelope = json.loads(str(row["sink_states_json"]))
            except (TypeError, ValueError) as exc:
                raise HistorianCorruptionError("Reality delivery durable JSON is invalid") from exc
            if not isinstance(payload, dict):
                raise HistorianCorruptionError("Reality delivery payload is not an object")
            sink_states = _decode_sink_state_envelope(
                sink_envelope,
                payload=payload,
            )
            if sink_id not in sink_states:
                raise HistorianCorruptionError(f"Reality delivery sink is not required: {sink_id}")
            prior = sink_states[sink_id]
            if not isinstance(prior, dict):
                raise HistorianCorruptionError("Reality delivery sink receipt is invalid")
            if str(prior.get("state") or "") == "delivered":
                if not secrets.compare_digest(
                    str(prior.get("receipt_id") or ""),
                    receipt,
                ):
                    raise HistorianCorruptionError(
                        "Reality delivery sink receipt conflicts with prior success"
                    )
                connection.execute("COMMIT")
                return False
            sink_states[sink_id] = {
                "state": "delivered",
                "receipt_id": receipt,
            }
            updated = connection.execute(
                "UPDATE reality_deliveries SET sink_states_json=?, updated_at=? "
                "WHERE observation_id=? AND state='delivering' AND lease_token=?",
                (
                    canonical_json(_sink_state_envelope(sink_states, payload=payload)).decode(
                        "utf-8"
                    ),
                    float(self._clock()),
                    observation,
                    lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise HistorianCorruptionError("Reality delivery sink receipt lost its lease")
            connection.execute("COMMIT")
            return True
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def mark_delivered(
        self,
        observation_id: str,
        *,
        lease_token: str,
    ) -> None:
        await asyncio.to_thread(
            self._run_serialized,
            self._set_delivery_state_sync,
            observation_id,
            "delivered",
            "",
            "",
            frozenset({"delivering"}),
            True,
            lease_token,
        )

    async def quarantine_delivery(
        self,
        observation_id: str,
        *,
        reason: str,
        lease_token: str,
    ) -> None:
        """Terminate a delivery WITHOUT scheduling it again.

        The state machine had `mark_delivered` (it worked), and
        `mark_delivery_failed` (it did not, so requeue) — and no way to say
        the third thing that actually happens: the observation REACHED the
        person and the record of that failed. The router was forced to call
        `mark_delivery_failed`, which requeued something already seen and
        showed it to them twice.

        Quarantine is the honest terminal state for that: it means "do not
        deliver again", which is exactly right for something already
        delivered, and it keeps the row for the recovery audit rather than
        dropping it.
        """
        await asyncio.to_thread(
            self._run_serialized,
            self._set_delivery_state_sync,
            observation_id,
            "quarantined",
            _identifier(reason, name="reason"),
            "",
            frozenset({"delivering"}),
            True,
            lease_token,
        )

    async def supersede_delivery(
        self,
        observation_id: str,
        *,
        replacement_observation_id: str = "",
        reason: str = "attention_superseded",
    ) -> None:
        await asyncio.to_thread(
            self._run_serialized,
            self._set_delivery_state_sync,
            observation_id,
            "superseded",
            reason,
            replacement_observation_id,
            frozenset({"queued"}),
            False,
            "",
        )

    def _set_delivery_state_sync(
        self,
        observation_id: str,
        state: str,
        error: str,
        replacement: str,
        expected_states: frozenset[str],
        strict: bool,
        lease_token: str,
    ) -> bool:
        observation = _identifier(observation_id, name="observation_id")
        if state not in _DELIVERY_STATES:
            raise ValueError("invalid Reality delivery state")
        replacement_id = (
            _identifier(replacement, name="replacement_observation_id") if replacement else ""
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, lease_token FROM reality_deliveries WHERE observation_id=?",
                (observation,),
            ).fetchone()
            if row is None:
                raise HistorianCorruptionError("Reality delivery receipt is missing")
            current = str(row["state"])
            if current == state:
                connection.execute("COMMIT")
                return False
            if current not in expected_states:
                if strict:
                    raise HistorianCorruptionError(
                        f"invalid Reality delivery transition: {current}->{state}"
                    )
                connection.execute("COMMIT")
                return False
            if lease_token:
                token = str(row["lease_token"])
                if not secrets.compare_digest(token, str(lease_token)):
                    raise HistorianCorruptionError("Reality delivery lease token differs")
            if state == "delivered":
                sink_row = connection.execute(
                    "SELECT payload_json, sink_states_json FROM reality_deliveries "
                    "WHERE observation_id=?",
                    (observation,),
                ).fetchone()
                try:
                    payload = json.loads(str(sink_row["payload_json"])) if sink_row else None
                    sink_envelope = (
                        json.loads(str(sink_row["sink_states_json"])) if sink_row else None
                    )
                except (TypeError, ValueError) as exc:
                    raise HistorianCorruptionError(
                        "Reality delivery durable JSON is invalid"
                    ) from exc
                if not isinstance(payload, dict):
                    raise HistorianCorruptionError("Reality delivery payload is not an object")
                sink_states = _decode_sink_state_envelope(
                    sink_envelope,
                    payload=payload,
                )
                if any(
                    not isinstance(item, dict) or str(item.get("state") or "") != "delivered"
                    for item in sink_states.values()
                ):
                    raise HistorianCorruptionError(
                        "Reality delivery cannot complete before every required sink"
                    )
            updated = connection.execute(
                "UPDATE reality_deliveries SET state=?, last_error=?, "
                "replacement_observation_id=?, lease_owner='', lease_token='', "
                "lease_expires_at=0, updated_at=? "
                "WHERE observation_id=? AND state=?",
                (
                    state,
                    str(error or "")[:160],
                    replacement_id,
                    float(self._clock()),
                    observation,
                    current,
                ),
            )
            if updated.rowcount != 1:
                raise HistorianCorruptionError("Reality delivery transition lost a race")
            connection.execute("COMMIT")
            return True
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def mark_delivery_failed(
        self,
        observation_id: str,
        *,
        error_code: str,
        lease_token: str,
    ) -> str:
        return await asyncio.to_thread(
            self._run_serialized,
            self._mark_delivery_failed_sync,
            observation_id,
            error_code,
            lease_token,
        )

    def _mark_delivery_failed_sync(
        self,
        observation_id: str,
        error_code: str,
        lease_token: str,
    ) -> str:
        observation = _identifier(observation_id, name="observation_id")
        error = _identifier(error_code, name="error_code")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempts, state, lease_token FROM reality_deliveries "
                "WHERE observation_id=?",
                (observation,),
            ).fetchone()
            if row is None:
                raise HistorianCorruptionError("Reality delivery receipt is missing")
            if str(row["state"]) != "delivering":
                raise HistorianCorruptionError("only a claimed Reality delivery may be failed")
            if not secrets.compare_digest(
                str(row["lease_token"]),
                str(lease_token),
            ):
                raise HistorianCorruptionError("Reality delivery lease token differs")
            attempts = int(row["attempts"])
            if attempts >= self._max_delivery_attempts:
                state = "quarantined"
                available_at = now
            else:
                state = "queued"
                available_at = now + min(60.0, float(2 ** max(0, attempts - 1)))
            connection.execute(
                "UPDATE reality_deliveries SET state=?, available_at=?, last_error=?, "
                "lease_owner='', lease_token='', lease_expires_at=0, updated_at=? "
                "WHERE observation_id=?",
                (state, available_at, error, now, observation),
            )
            connection.execute("COMMIT")
            return state
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _delivery_from_row(row: sqlite3.Row) -> HistorianDelivery:
        try:
            payload = json.loads(str(row["payload_json"]))
            sink_envelope = json.loads(str(row["sink_states_json"]))
        except (TypeError, ValueError) as exc:
            raise HistorianCorruptionError("Reality delivery payload is invalid") from exc
        if not isinstance(payload, dict):
            raise HistorianCorruptionError("Reality delivery payload is not an object")
        sink_states = _decode_sink_state_envelope(
            sink_envelope,
            payload=payload,
        )
        observation_id, record_id = _validate_observation_delivery_binding(
            row,
            payload,
        )
        return HistorianDelivery(
            observation_id=observation_id,
            record_id=record_id,
            payload=payload,
            sink_states=sink_states,
            attempts=int(row["attempts"]),
            replay_count=int(row["replay_count"]),
            lease_token=str(row["lease_token"]),
            lease_expires_at=float(row["lease_expires_at"]),
        )

    async def legacy_twin_head_snapshots(
        self,
        *,
        limit: int = 512,
    ) -> tuple[HistorianHeadSnapshot, ...]:
        """Return real current heads that predate an authoritative twin fence."""

        page = await self.legacy_twin_head_page(limit=limit)
        return page.snapshots

    async def legacy_twin_head_page(
        self,
        *,
        adapter_inventory: Mapping[str, AdapterInventoryEntry] | None = None,
        after_channel_id: str = "",
        limit: int = 512,
        sink: str = "digital_twin",
    ) -> HistorianHeadPage:
        """Page unreceipted legacy heads in stable channel order.

        When ``adapter_inventory`` is supplied, only heads whose adapter,
        channel, and authoritative physical or registration identity still
        match the live Reality Reach inventory are eligible. Historical
        adapters and a replacement device reusing an adapter ID are excluded.
        """

        return await asyncio.to_thread(
            self._run_serialized,
            self._legacy_twin_head_page_sync,
            adapter_inventory,
            after_channel_id,
            limit,
            sink,
        )

    def _legacy_twin_head_page_sync(
        self,
        adapter_inventory: Mapping[str, AdapterInventoryEntry] | None,
        after_channel_id: str,
        limit: int,
        sink: str,
    ) -> HistorianHeadPage:
        bounded = max(1, min(int(limit), 4096))
        cursor = (
            ""
            if not str(after_channel_id or "")
            else _identifier(after_channel_id, name="after_channel_id")
        )
        target_sink = _identifier(sink, name="sink")
        inventory = self._normalize_adapter_inventory(adapter_inventory)
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS active_reality_inventory ("
                "adapter_id TEXT NOT NULL, channel_id TEXT NOT NULL, "
                "adapter_identity_sha256 TEXT NOT NULL, "
                "registration_generation INTEGER NOT NULL, "
                "stable_identity INTEGER NOT NULL, "
                "PRIMARY KEY(adapter_id, channel_id)) WITHOUT ROWID"
            )
            connection.execute("DELETE FROM active_reality_inventory")
            if inventory is not None:
                connection.executemany(
                    "INSERT INTO active_reality_inventory("
                    "adapter_id, channel_id, adapter_identity_sha256, "
                    "registration_generation, stable_identity) VALUES(?, ?, ?, ?, ?)",
                    inventory,
                )
            inventory_join = (
                "JOIN active_reality_inventory AS i "
                "ON i.adapter_id=o.adapter_id AND i.channel_id=o.channel_id "
                "AND i.adapter_identity_sha256="
                "json_extract(o.reading_json, '$.adapter_identity_sha256') "
                "AND ((i.stable_identity=1 AND "
                "json_extract(o.reading_json, '$.adapter_identity_stable')=1) "
                "OR (i.stable_identity=0 AND "
                "json_extract(o.reading_json, '$.adapter_identity_stable')=0 "
                "AND i.registration_generation="
                "json_extract(o.reading_json, '$.adapter_registration_generation'))) "
                if inventory is not None
                else ""
            )
            receipt_rows = connection.execute(
                "SELECT b.* FROM reality_backfill_receipts AS b "
                "JOIN reality_observations AS o ON o.record_id=b.record_id "
                + inventory_join
                + "WHERE b.sink=? AND o.twin_id='' "
                "AND o.attachment_generation=0 AND o.attachment_bound_at_ns=0 "
                "AND o.topology_revision=0",
                (target_sink,),
            ).fetchall()
            for receipt_row in receipt_rows:
                self._verify_backfill_receipt_row(receipt_row)
            query = (
                "SELECT o.*, a.alarm_code AS active_alarm_code "
                "FROM reality_channel_heads AS h "
                "JOIN reality_observations AS o "
                "ON o.record_id=h.last_stored_record_id "
                + inventory_join
                +
                "LEFT JOIN reality_alarm_heads AS a "
                "ON a.channel_id=o.channel_id AND a.active=1 "
                "WHERE o.twin_id='' AND o.attachment_generation=0 "
                "AND o.attachment_bound_at_ns=0 AND o.topology_revision=0 "
                "AND o.channel_id>? "
                "AND NOT EXISTS ("
                "SELECT 1 FROM reality_backfill_receipts AS b "
                "WHERE b.sink=? AND b.record_id=o.record_id) "
                "ORDER BY o.channel_id ASC LIMIT ?"
            )
            rows = connection.execute(
                query,
                (cursor, target_sink, bounded),
            ).fetchall()
            snapshots = tuple(self._head_snapshot_from_row(row) for row in rows)
            next_channel_id = str(rows[-1]["channel_id"]) if rows else cursor
            return HistorianHeadPage(
                snapshots=snapshots,
                next_channel_id=next_channel_id,
                exhausted=len(rows) < bounded,
                scanned=len(rows),
            )
        finally:
            connection.close()

    @staticmethod
    def _normalize_adapter_inventory(
        adapter_inventory: Mapping[str, AdapterInventoryEntry] | None,
    ) -> tuple[tuple[str, str, str, int, int], ...] | None:
        if adapter_inventory is None:
            return None
        pairs: set[tuple[str, str, str, int, int]] = set()
        for raw_adapter_id, raw_entry in adapter_inventory.items():
            adapter_id = _identifier(raw_adapter_id, name="adapter_id")
            if not isinstance(raw_entry, AdapterInventoryEntry):
                raise TypeError("adapter inventory must contain AdapterInventoryEntry values")
            if raw_entry.adapter_id != adapter_id:
                raise ValueError("adapter inventory key differs from entry identity")
            for raw_channel_id in raw_entry.channel_ids:
                channel_id = _identifier(raw_channel_id, name="channel_id")
                pairs.add(
                    (
                        adapter_id,
                        channel_id,
                        raw_entry.identity_sha256,
                        raw_entry.registration_generation,
                        int(raw_entry.stable_identity),
                    )
                )
        if len(pairs) > 65_536:
            raise ValueError("adapter channel inventory exceeds bounded backfill contract")
        return tuple(sorted(pairs))

    @staticmethod
    def _verify_backfill_receipt_row(row: Mapping[str, Any]) -> None:
        values = dict(row)
        supplied = str(values.get("row_sha256") or "")
        if not _DIGEST.fullmatch(supplied) or not secrets.compare_digest(
            supplied,
            _backfill_row_sha256(values),
        ):
            raise HistorianCorruptionError(
                "Reality historian backfill receipt integrity differs"
            )

    @staticmethod
    def _head_snapshot_from_row(row: sqlite3.Row) -> HistorianHeadSnapshot:
        try:
            declaration = json.loads(str(row["declaration_json"]))
            reading = json.loads(str(row["reading_json"]))
        except (TypeError, ValueError) as exc:
            raise HistorianCorruptionError(
                "Reality historian head contains invalid evidence JSON"
            ) from exc
        if not isinstance(declaration, dict) or not isinstance(reading, dict):
            raise HistorianCorruptionError(
                "Reality historian head evidence is not an object"
            )
        declaration_sha256 = str(row["declaration_sha256"] or "")
        reading_sha256 = str(row["reading_sha256"] or "")
        if not secrets.compare_digest(_digest(declaration), declaration_sha256):
            raise HistorianCorruptionError(
                "Reality historian head declaration evidence differs"
            )
        if not secrets.compare_digest(_digest(reading), reading_sha256):
            raise HistorianCorruptionError("Reality historian head reading evidence differs")
        record_id = _identifier(row["record_id"], name="record_id")
        adapter_id = _identifier(row["adapter_id"], name="adapter_id")
        adapter_identity_sha256 = str(
            reading.get("adapter_identity_sha256") or ""
        )
        adapter_registration_generation = int(
            reading.get("adapter_registration_generation") or 0
        )
        adapter_identity_stable = reading.get("adapter_identity_stable", False)
        if (
            not _DIGEST.fullmatch(adapter_identity_sha256)
            or adapter_registration_generation <= 0
            or not isinstance(adapter_identity_stable, bool)
        ):
            raise HistorianCorruptionError(
                "Reality historian head lacks authoritative adapter identity"
            )
        channel_id = _identifier(row["channel_id"], name="channel_id")
        if (
            str(declaration.get("channel_id") or "") != channel_id
            or str(reading.get("channel_id") or "") != channel_id
            or str(reading.get("unit") or "") != str(row["unit"] or "")
            or int(reading.get("captured_at_ns") or 0) != int(row["captured_at_ns"])
            or int(reading.get("sequence") or 0) != int(row["sequence"])
            or str(reading.get("status") or "") != str(row["status"] or "")
        ):
            raise HistorianCorruptionError(
                "Reality historian head scalar evidence differs from its record"
            )
        try:
            quality = ObservationQuality(str(row["quality"] or "")).value
        except ValueError as exc:
            raise HistorianCorruptionError("Reality historian head quality is invalid") from exc
        order_basis = str(row["order_basis"] or "")
        if not order_basis or len(order_basis) > 128:
            raise HistorianCorruptionError("Reality historian head order basis is invalid")
        alarm_code = str(row["active_alarm_code"] or "")
        alarm_codes = () if not alarm_code else (_identifier(alarm_code, name="alarm_code"),)
        evidence = {
            "record_id": record_id,
            "adapter_id": adapter_id,
            "adapter_identity_sha256": adapter_identity_sha256,
            "adapter_registration_generation": adapter_registration_generation,
            "adapter_identity_stable": adapter_identity_stable,
            "channel_id": channel_id,
            "declaration_sha256": declaration_sha256,
            "reading_sha256": reading_sha256,
            "event_sha256": str(row["event_sha256"] or ""),
            "quality": quality,
            "order_basis": order_basis,
            "order_gap": bool(row["order_gap"]),
            "alarm_codes": list(alarm_codes),
            "recorded_at": float(row["recorded_at"]),
        }
        return HistorianHeadSnapshot(
            record_id=record_id,
            adapter_id=adapter_id,
            adapter_identity_sha256=adapter_identity_sha256,
            adapter_registration_generation=adapter_registration_generation,
            adapter_identity_stable=adapter_identity_stable,
            channel_id=channel_id,
            declaration=declaration,
            reading=reading,
            quality=quality,
            order_basis=order_basis,
            order_gap=bool(row["order_gap"]),
            alarm_codes=alarm_codes,
            recorded_at=float(row["recorded_at"]),
            source_sha256=_digest(
                {"schema": "aura.reality-authoritative-head.v1", **evidence}
            ),
        )

    async def backfill_receipt(self, backfill_id: str) -> str:
        return await asyncio.to_thread(
            self._run_serialized,
            self._backfill_receipt_sync,
            backfill_id,
        )

    def _backfill_receipt_sync(self, backfill_id: str) -> str:
        canonical = _identifier(backfill_id, name="backfill_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM reality_backfill_receipts WHERE backfill_id=?",
                (canonical,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return ""
        values = dict(row)
        supplied = str(values.get("row_sha256") or "")
        if not secrets.compare_digest(supplied, _backfill_row_sha256(values)):
            raise HistorianCorruptionError(
                "Reality historian backfill receipt integrity differs"
            )
        return str(row["receipt_id"])

    async def record_backfill_receipt(
        self,
        *,
        backfill_id: str,
        record_id: str,
        channel_id: str,
        source_sha256: str,
        target_binding_sha256: str,
        sink_receipt_id: str,
        sink: str = "digital_twin",
    ) -> str:
        return await asyncio.to_thread(
            self._run_serialized,
            self._record_backfill_receipt_sync,
            backfill_id,
            record_id,
            channel_id,
            source_sha256,
            target_binding_sha256,
            sink_receipt_id,
            sink,
        )

    def _record_backfill_receipt_sync(
        self,
        backfill_id: str,
        record_id: str,
        channel_id: str,
        source_sha256: str,
        target_binding_sha256: str,
        sink_receipt_id: str,
        sink: str,
    ) -> str:
        backfill = _identifier(backfill_id, name="backfill_id")
        record = _identifier(record_id, name="record_id")
        channel = _identifier(channel_id, name="channel_id")
        target = _identifier(sink, name="sink")
        source_digest = str(source_sha256 or "")
        binding_digest = str(target_binding_sha256 or "")
        if not _DIGEST.fullmatch(source_digest) or not _DIGEST.fullmatch(binding_digest):
            raise ValueError("Reality backfill source and target require canonical digests")
        sink_receipt = _identifier(sink_receipt_id, name="sink_receipt_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM reality_backfill_receipts WHERE backfill_id=? "
                "OR (sink=? AND record_id=? AND target_binding_sha256=?)",
                (backfill, target, record, binding_digest),
            ).fetchone()
            if existing is not None:
                values = dict(existing)
                if (
                    str(existing["backfill_id"]) != backfill
                    or str(existing["channel_id"]) != channel
                    or str(existing["source_sha256"]) != source_digest
                    or str(existing["sink_receipt_id"]) != sink_receipt
                    or not secrets.compare_digest(
                        str(existing["row_sha256"] or ""),
                        _backfill_row_sha256(values),
                    )
                ):
                    raise HistorianCorruptionError(
                        "Reality historian backfill idempotency evidence conflicts"
                    )
                connection.execute("COMMIT")
                return str(existing["receipt_id"])
            current = connection.execute(
                "SELECT o.*, a.alarm_code AS active_alarm_code "
                "FROM reality_channel_heads AS h "
                "JOIN reality_observations AS o "
                "ON o.record_id=h.last_stored_record_id "
                "LEFT JOIN reality_alarm_heads AS a "
                "ON a.channel_id=o.channel_id AND a.active=1 "
                "WHERE h.channel_id=? AND o.record_id=? AND o.twin_id='' "
                "AND o.attachment_generation=0 AND o.attachment_bound_at_ns=0 "
                "AND o.topology_revision=0",
                (channel, record),
            ).fetchone()
            if current is None:
                raise HistorianCorruptionError(
                    "Reality historian backfill source is no longer the authoritative head"
                )
            snapshot = self._head_snapshot_from_row(current)
            if not secrets.compare_digest(snapshot.source_sha256, source_digest):
                raise HistorianCorruptionError(
                    "Reality historian backfill source changed before receipt"
                )
            completed_at = float(self._clock())
            receipt_id = (
                "reality.backfill.receipt."
                + _digest(
                    {
                        "backfill_id": backfill,
                        "sink_receipt_id": sink_receipt,
                        "source_sha256": source_digest,
                        "target_binding_sha256": binding_digest,
                    }
                ).removeprefix("sha256:")[:32]
            )
            values = {
                "backfill_id": backfill,
                "sink": target,
                "record_id": record,
                "channel_id": channel,
                "source_sha256": source_digest,
                "target_binding_sha256": binding_digest,
                "sink_receipt_id": sink_receipt,
                "receipt_id": receipt_id,
                "completed_at": completed_at,
            }
            row_sha256 = _backfill_row_sha256(values)
            connection.execute(
                "INSERT INTO reality_backfill_receipts("
                "backfill_id, sink, record_id, channel_id, source_sha256, "
                "target_binding_sha256, sink_receipt_id, receipt_id, completed_at, "
                "row_sha256) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    backfill,
                    target,
                    record,
                    channel,
                    source_digest,
                    binding_digest,
                    sink_receipt,
                    receipt_id,
                    completed_at,
                    row_sha256,
                ),
            )
            connection.execute("COMMIT")
            return receipt_id
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def replay_history(
        self,
        *,
        channel_id: str | None = None,
        before_row_id: int | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._replay_history_sync,
            channel_id,
            before_row_id,
            limit,
        )

    def _replay_history_sync(
        self,
        channel_id: str | None,
        before_row_id: int | None,
        limit: int,
    ) -> dict[str, Any]:
        bounded = max(1, min(int(limit), 500))
        params: list[Any] = []
        where: list[str] = []
        if channel_id:
            where.append("channel_id=?")
            params.append(_identifier(channel_id, name="channel_id"))
        if before_row_id is not None:
            cursor = int(before_row_id)
            if cursor <= 0:
                raise ValueError("before_row_id must be positive")
            where.append("rowid<?")
            params.append(cursor)
        clause = " WHERE " + " AND ".join(where) if where else ""
        params.append(bounded)
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT rowid, * FROM reality_observations"
                + clause
                + " ORDER BY rowid DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        finally:
            connection.close()
        records = [
            {
                "row_id": int(row["rowid"]),
                "record_id": str(row["record_id"]),
                "adapter_id": str(row["adapter_id"]),
                "channel_id": str(row["channel_id"]),
                "captured_at_ns": int(row["captured_at_ns"]),
                "status": str(row["status"]),
                "quality": str(row["quality"]),
                "value": row["value"],
                "unit": str(row["unit"]),
                "uncertainty": row["uncertainty"],
                "recorded_at": float(row["recorded_at"]),
                "reading_sha256": str(row["reading_sha256"]),
                "event_sha256": str(row["event_sha256"]),
                "declaration_sha256": str(row["declaration_sha256"]),
                "source": str(row["source"]),
                "session_id": str(row["session_id"]),
                "sequence": int(row["sequence"]),
                "order_basis": str(row["order_basis"]),
                "order_gap": bool(row["order_gap"]),
                "error_code": str(row["error_code"]),
                "twin_id": str(row["twin_id"]),
                "attachment_generation": int(row["attachment_generation"]),
                "attachment_bound_at_ns": int(row["attachment_bound_at_ns"]),
                "topology_revision": int(row["topology_revision"]),
            }
            for row in reversed(rows)
        ]
        return {
            "records": records,
            "count": len(records),
            "limit": bounded,
            "next_before_row_id": min((item["row_id"] for item in records), default=None),
        }

    async def active_alarms(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._active_alarms_sync,
            limit,
        )

    async def alarm_history(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._alarm_history_sync,
            limit,
        )

    def _alarm_history_sync(self, limit: int) -> tuple[dict[str, Any], ...]:
        bounded = max(1, min(int(limit), 500))
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT event_id, channel_id, alarm_code, severity, state, "
                "record_id, actor, created_at FROM reality_alarm_events "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(dict(row) for row in rows)

    def _active_alarms_sync(self, limit: int) -> tuple[dict[str, Any], ...]:
        bounded = max(1, min(int(limit), 500))
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM reality_alarm_heads WHERE active=1 "
                "ORDER BY CASE severity WHEN 'high' THEN 0 ELSE 1 END, active_since ASC "
                "LIMIT ?",
                (bounded,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            {
                "channel_id": str(row["channel_id"]),
                "alarm_code": str(row["alarm_code"]),
                "severity": str(row["severity"]),
                "acknowledged": bool(row["acknowledged"]),
                "active_since": float(row["active_since"]),
                "last_record_id": str(row["last_record_id"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        )

    async def acknowledge_alarm(self, channel_id: str, *, actor: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._acknowledge_alarm_sync,
            channel_id,
            actor,
        )

    def _acknowledge_alarm_sync(self, channel_id: str, actor: str) -> dict[str, Any]:
        channel = _identifier(channel_id, name="channel_id")
        principal = _identifier(actor, name="actor")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            head = connection.execute(
                "SELECT * FROM reality_alarm_heads WHERE channel_id=? AND active=1",
                (channel,),
            ).fetchone()
            if head is None:
                raise LookupError("no active Reality alarm for channel")
            if not bool(head["acknowledged"]):
                event_id = self._insert_alarm_event(
                    connection,
                    channel_id=channel,
                    alarm_code=str(head["alarm_code"]),
                    severity=str(head["severity"]),
                    state="acknowledged",
                    record_id=str(head["last_record_id"]),
                    actor=principal,
                    now=now,
                )
                connection.execute(
                    "UPDATE reality_alarm_heads SET acknowledged=1, updated_at=? "
                    "WHERE channel_id=?",
                    (now, channel),
                )
            else:
                event_id = ""
            connection.execute("COMMIT")
            return {
                "channel_id": channel,
                "acknowledged": True,
                "event_id": event_id,
            }
        except Exception:
            _roll_back(connection)
            raise
        finally:
            connection.close()

    async def quarantine(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        return await asyncio.to_thread(
            self._run_serialized,
            self._quarantine_sync,
            limit,
        )

    def _quarantine_sync(self, limit: int) -> tuple[dict[str, Any], ...]:
        bounded = max(1, min(int(limit), 500))
        connection = self._connect()
        try:
            sensor_rows = connection.execute(
                "SELECT quarantine_id, adapter_id, channel_id, reason, reading_sha256, "
                "created_at FROM reality_quarantine ORDER BY created_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            delivery_rows = connection.execute(
                "SELECT observation_id, record_id, last_error, attempts, replay_count, "
                "updated_at FROM reality_deliveries WHERE state='quarantined' "
                "ORDER BY updated_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        finally:
            connection.close()
        combined = [{"kind": "source_evidence", **dict(row)} for row in sensor_rows] + [
            {
                "kind": "cognitive_delivery",
                "quarantine_id": str(row["observation_id"]),
                "record_id": str(row["record_id"]),
                "reason": str(row["last_error"]),
                "attempts": int(row["attempts"]),
                "replay_count": int(row["replay_count"]),
                "created_at": float(row["updated_at"]),
            }
            for row in delivery_rows
        ]
        combined.sort(key=lambda item: float(item["created_at"]), reverse=True)
        return tuple(combined[:bounded])

    def _status_sync(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            observation_count = int(
                connection.execute("SELECT COUNT(*) FROM reality_observations").fetchone()[0]
            )
            quarantine_count = int(
                connection.execute("SELECT COUNT(*) FROM reality_quarantine").fetchone()[0]
            )
            active_alarm_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM reality_alarm_heads WHERE active=1"
                ).fetchone()[0]
            )
            backfill_receipt_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM reality_backfill_receipts"
                ).fetchone()[0]
            )
            delivery_counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM reality_deliveries GROUP BY state"
                ).fetchall()
            }
            sink_status: dict[str, dict[str, float | int]] = {}
            now = time.time()
            for delivery in connection.execute(
                """
                SELECT payload_json, sink_states_json, state, created_at
                FROM reality_deliveries
                WHERE state IN ('queued', 'delivering', 'delivered')
                """
            ):
                try:
                    payload = json.loads(str(delivery["payload_json"]))
                    envelope = json.loads(str(delivery["sink_states_json"]))
                except (TypeError, ValueError) as exc:
                    raise HistorianCorruptionError(
                        "Reality delivery status found malformed sink evidence"
                    ) from exc
                if not isinstance(payload, dict):
                    raise HistorianCorruptionError(
                        "Reality delivery status payload is not an object"
                    )
                states = _decode_sink_state_envelope(envelope, payload=payload)
                for sink, state in states.items():
                    sink_counters = sink_status.setdefault(
                        sink,
                        {
                            "pending": 0,
                            "delivered": 0,
                            "oldest_pending_age_s": 0.0,
                        },
                    )
                    sink_state = str(state.get("state") or "")
                    sink_counters[sink_state] = int(sink_counters.get(sink_state, 0)) + 1
                    if sink_state == "pending":
                        age = max(0.0, now - float(delivery["created_at"]))
                        sink_counters["oldest_pending_age_s"] = max(
                            float(sink_counters["oldest_pending_age_s"]), age
                        )
            counters: dict[str, int] = {}
            for key in _META_COUNTERS:
                row = connection.execute(
                    "SELECT value FROM reality_historian_meta WHERE key=?",
                    (key,),
                ).fetchone()
                if row is None:
                    raise HistorianCorruptionError(f"Reality historian counter is missing: {key}")
                try:
                    value = int(row[0])
                except (TypeError, ValueError) as exc:
                    raise HistorianCorruptionError(
                        f"Reality historian counter is invalid: {key}"
                    ) from exc
                if value < 0:
                    raise HistorianCorruptionError(f"Reality historian counter is invalid: {key}")
                counters[key] = value
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        finally:
            connection.close()
        storage_files = self._storage_file_bytes()
        return {
            "ready": True,
            "schema_version": _SCHEMA_VERSION,
            "observation_count": observation_count,
            "max_records": self._max_records,
            "quarantine_count": quarantine_count,
            "max_quarantine": self._max_quarantine,
            "active_alarm_count": active_alarm_count,
            "backfill_receipt_count": backfill_receipt_count,
            "delivery_counts": delivery_counts,
            "delivery_sink_status": sink_status,
            **counters,
            "db_mode": "private_sqlite_wal_full",
            "storage_bytes": sum(storage_files.values()),
            "storage_files": storage_files,
            "allocated_page_bytes": page_size * page_count,
            "active_page_bytes": page_size * max(0, page_count - freelist_count),
            "max_storage_bytes": self._max_storage_bytes,
            "min_free_bytes": self._min_free_bytes,
            "disk_free_bytes": self._current_disk_free_bytes(),
            "max_age_s": self._max_age_s,
            "max_silence_s": self._max_silence_s,
            "delivery_lease_s": self._delivery_lease_s,
        }

    def status(self) -> dict[str, Any]:
        try:
            status = cast(
                dict[str, Any],
                self._run_serialized(self._status_sync),
            )
        except HistorianError:
            status = {
                "ready": False,
                "schema_version": _SCHEMA_VERSION,
                "db_mode": "private_sqlite_wal_full",
                "storage_bytes": sum(self._storage_file_bytes().values()),
                "storage_files": self._storage_file_bytes(),
                "max_storage_bytes": self._max_storage_bytes,
                "min_free_bytes": self._min_free_bytes,
                "max_age_s": self._max_age_s,
            }
        with self._health_lock:
            status.update(
                {
                    "ready": bool(status.get("ready")) and self._healthy,
                    "last_error": self._last_error,
                    "consecutive_failures": self._consecutive_failures,
                    "last_success_at": self._last_success_at,
                    "last_failure_at": self._last_failure_at,
                }
            )
        status["status"] = "active" if status["ready"] else "degraded"
        return status

    def _storage_file_bytes(self) -> dict[str, int]:
        files = {
            "database": self.db_path,
            "wal": Path(f"{self.db_path}-wal"),
            "shm": Path(f"{self.db_path}-shm"),
        }
        sizes: dict[str, int] = {}
        for label, path in files.items():
            try:
                sizes[label] = int(path.stat().st_size)
            except OSError:
                # not a failure: a WAL or shm file that is not there occupies
                # no bytes, which is what this sum is asking.
                sizes[label] = 0
        return sizes

    def _current_disk_free_bytes(self) -> int:
        try:
            value = int(self._disk_free_bytes(self.db_path.parent))
        except (OSError, TypeError, ValueError) as exc:
            raise HistorianError("Reality historian could not measure disk headroom") from exc
        if value < 0:
            raise HistorianError("Reality historian disk headroom was negative")
        return value

    def _probe_sync(self) -> None:
        connection = self._connect()
        try:
            version = connection.execute(
                "SELECT value FROM reality_historian_meta WHERE key='schema_version'"
            ).fetchone()
            try:
                schema_version = int(version[0]) if version is not None else -1
            except (TypeError, ValueError) as exc:
                raise HistorianCorruptionError(
                    "Reality historian health probe found an invalid schema version"
                ) from exc
            if schema_version != _SCHEMA_VERSION:
                raise HistorianCorruptionError(
                    "Reality historian health probe found a schema mismatch"
                )
            connection.execute("SELECT record_id FROM reality_observations LIMIT 1").fetchone()
        finally:
            connection.close()

    def get_status(self) -> dict[str, Any]:
        return self.health_snapshot()

    def health_snapshot(self) -> dict[str, Any]:
        """Return lock-only health for event-loop status aggregation."""

        with self._health_lock:
            ready = self._healthy and self.is_alive()
            return {
                "status": "active" if ready else "degraded",
                "ready": ready,
                "last_error": self._last_error,
                "consecutive_failures": self._consecutive_failures,
                "last_success_at": self._last_success_at,
                "last_failure_at": self._last_failure_at,
                "last_probe_error": self._last_probe_error,
                "storage_bytes": sum(self._storage_file_bytes().values()),
                "max_storage_bytes": self._max_storage_bytes,
            }

    def is_alive(self) -> bool:
        try:
            stat = self.db_path.stat()
        except OSError:
            # not a failure: a database file that cannot be stat'd is not a
            # live historian, and that verdict is the question asked.
            return False
        return stat.st_uid == os.getuid() and not bool(stat.st_mode & 0o077)

    def is_ready(self) -> bool:
        if not self.is_alive():
            return False
        with self._health_lock:
            return self._healthy

    async def probe_health(self) -> bool:
        try:
            await asyncio.to_thread(self._run_serialized, self._probe_sync)
        except HistorianError as exc:
            # The probe's whole job is to answer healthy or not, and the
            # caller gets only the verdict. The cause is attached where a
            # later reader can still find it.
            exc.add_note("reported by probe_health as not healthy")
            self._last_probe_error = str(exc)[:200]
            return False
        with self._health_lock:
            self._last_probe_at = float(self._clock())
        return True

    def close(self) -> None:
        """No persistent connection is retained; lifecycle hook is explicit."""

    async def on_stop_async(self) -> None:
        self.close()


__all__ = [
    "HistorianAdmission",
    "HistorianCorruptionError",
    "HistorianDelivery",
    "HistorianDisposition",
    "HistorianError",
    "ObservationQuality",
    "RealityHistorian",
    "default_reality_historian_path",
]
