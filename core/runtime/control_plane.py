"""Canonical desired-state and resource-admission control plane.

This module owns two runtime decisions that were previously spread across
resource governors, supervisors, boot code, and individual callers:

* whether a unit of work may consume a constrained runtime resource now;
* whether a managed service matches its declared desired lifecycle state.

It does not replace domain-specific samplers, evictors, or process transports.
Those become observations and adapters behind this policy owner.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.RuntimeControlPlane")

CONTROL_PLANE_RECONCILE_TASK_NAME = "runtime_control_plane_reconcile"
CONTROL_PLANE_RECONCILE_INTERVAL_S = 5.0
CONTROL_PLANE_RECONCILE_TIMEOUT_S = 30.0

_ADMISSION_RECEIPT_HEARTBEAT_FLAG = declare(
    "AURA_ADMISSION_RECEIPT_HEARTBEAT_S",
    kind=FlagKind.FLOAT,
    default=3600.0,
    description=(
        "Maximum interval between durable receipts for an unchanged unaudited "
        "admission denial; 0 disables coalescing"
    ),
    owner="core.runtime.control_plane",
)


class WorkClass(StrEnum):
    INFERENCE = "inference"
    MODEL_LOAD = "model_load"
    EVOLUTION = "evolution"
    DESKTOP = "desktop"
    EXTERNAL_IO = "external_io"
    SERVICE_START = "service_start"
    MAINTENANCE = "maintenance"
    BACKGROUND = "background"


class AdmissionPriority(IntEnum):
    CRITICAL = 0
    FOREGROUND = 10
    INTERACTIVE = 20
    MAINTENANCE = 50
    BACKGROUND = 80


class AdmissionOutcome(StrEnum):
    ADMITTED = "admitted"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    RELEASED = "released"
    PREEMPTED = "preempted"
    EXPIRED = "expired"


class DesiredServiceState(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class ObservedServiceState(StrEnum):
    UNKNOWN = "unknown"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    BACKING_OFF = "backing_off"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True)
class PressureSnapshot:
    captured_at: float = field(default_factory=time.time)
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    host_observed: bool = False
    qualifies_as_live_pressure: bool = False
    resource_observation_available: bool = False
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    process_tree_rss_mb: float = 0.0
    thermal_level: int = 0
    thermal_provider: str = "blind"
    disk_percent: float = 0.0
    disk_free_bytes: int = 0
    loop_lag_s: float = 0.0
    loop_lag_sample_age_s: float = 0.0
    loop_lag_sample_fresh: bool = True
    loop_monitor_alive: bool | None = None
    loop_monitor_running: bool | None = None
    loop_monitor_healthy: bool | None = None
    loop_monitor_incident_active: bool = False
    shutdown_requested: bool = False
    red_zones: tuple[str, ...] = ()
    suspended_capabilities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PressureSnapshot:
        raw = dict(value or {})
        return cls(
            captured_at=float(raw.get("captured_at") or raw.get("at_unix") or time.time()),
            observation_source=str(raw.get("observation_source") or "unavailable"),
            observation_scenario_id=str(raw.get("observation_scenario_id") or ""),
            host_observed=bool(raw.get("host_observed", False)),
            qualifies_as_live_pressure=bool(raw.get("qualifies_as_live_pressure", False)),
            resource_observation_available=bool(
                raw.get("resource_observation_available", False)
            ),
            memory_percent=max(
                0.0,
                float(raw.get("memory_percent") or raw.get("memory_pct") or 0.0),
            ),
            memory_rss_mb=max(0.0, float(raw.get("memory_rss_mb") or 0.0)),
            process_tree_rss_mb=max(
                0.0,
                float(raw.get("process_tree_rss_mb") or 0.0),
            ),
            thermal_level=max(0, int(raw.get("thermal_level") or 0)),
            thermal_provider=str(raw.get("thermal_provider") or "blind"),
            disk_percent=max(0.0, float(raw.get("disk_percent") or 0.0)),
            disk_free_bytes=max(0, int(raw.get("disk_free_bytes") or 0)),
            loop_lag_s=max(0.0, float(raw.get("loop_lag_s") or 0.0)),
            loop_lag_sample_age_s=max(
                0.0,
                float(raw.get("loop_lag_sample_age_s") or 0.0),
            ),
            loop_lag_sample_fresh=bool(
                raw.get("loop_lag_sample_fresh", True)
            ),
            loop_monitor_alive=(
                bool(raw.get("loop_monitor_alive"))
                if raw.get("loop_monitor_alive") is not None
                else None
            ),
            loop_monitor_running=(
                bool(raw.get("loop_monitor_running"))
                if raw.get("loop_monitor_running") is not None
                else (
                    bool(raw.get("loop_monitor_alive"))
                    if raw.get("loop_monitor_alive") is not None
                    else None
                )
            ),
            loop_monitor_healthy=(
                bool(raw.get("loop_monitor_healthy"))
                if raw.get("loop_monitor_healthy") is not None
                else (
                    bool(raw.get("loop_monitor_alive"))
                    if raw.get("loop_monitor_alive") is not None
                    else None
                )
            ),
            loop_monitor_incident_active=bool(
                raw.get("loop_monitor_incident_active", False)
            ),
            shutdown_requested=bool(raw.get("shutdown_requested", False)),
            red_zones=tuple(str(item) for item in raw.get("red_zones") or ()),
            suspended_capabilities=tuple(
                str(item) for item in raw.get("suspended_capabilities") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["red_zones"] = list(self.red_zones)
        payload["suspended_capabilities"] = list(self.suspended_capabilities)
        return payload


@dataclass(frozen=True)
class AdmissionRequest:
    owner: str
    work_class: WorkClass
    lane: str = "default"
    priority: AdmissionPriority = AdmissionPriority.INTERACTIVE
    timeout_s: float = 30.0
    lease_ttl_s: float = 300.0
    preemptible: bool = False
    receipt_required: bool = False
    estimated_memory_mb: float = 0.0
    request_id: str = field(default_factory=lambda: f"admission-{uuid.uuid4()}")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.owner).strip():
            raise ValueError("admission owner must be non-empty")
        if not str(self.lane).strip():
            raise ValueError("admission lane must be non-empty")
        if not str(self.request_id).strip():
            raise ValueError("admission request_id must be non-empty")
        if float(self.timeout_s) < 0:
            raise ValueError("admission timeout_s must be non-negative")
        if float(self.lease_ttl_s) <= 0:
            raise ValueError("admission lease_ttl_s must be positive")
        if float(self.estimated_memory_mb) < 0:
            raise ValueError("estimated_memory_mb must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "owner": self.owner,
            "work_class": self.work_class.value,
            "lane": self.lane,
            "priority": int(self.priority),
            "priority_name": self.priority.name.lower(),
            "timeout_s": float(self.timeout_s),
            "lease_ttl_s": float(self.lease_ttl_s),
            "preemptible": bool(self.preemptible),
            "receipt_required": bool(self.receipt_required),
            "estimated_memory_mb": float(self.estimated_memory_mb),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AdmissionDecision:
    request_id: str
    outcome: AdmissionOutcome
    reason: str
    decided_at: float
    lease_id: str = ""
    receipt_id: str = ""
    blocking_lease_ids: tuple[str, ...] = ()
    pressure: PressureSnapshot = field(default_factory=PressureSnapshot)
    replayed: bool = False
    receipt_replayed: bool = False

    @property
    def admitted(self) -> bool:
        return self.outcome == AdmissionOutcome.ADMITTED and bool(self.lease_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "lease_id": self.lease_id,
            "receipt_id": self.receipt_id,
            "blocking_lease_ids": list(self.blocking_lease_ids),
            "pressure": self.pressure.to_dict(),
            "replayed": self.replayed,
            "receipt_replayed": self.receipt_replayed,
        }


@dataclass
class _LeaseRecord:
    lease_id: str
    request: AdmissionRequest
    admitted_at: float
    expires_at: float
    preempt_requested: bool = False
    preempt_reason: str = ""
    on_preempt: Callable[[str], Any] | None = None
    holder_task: asyncio.Task[Any] | None = None


@dataclass(frozen=True)
class _Waiter:
    sequence: int
    request: AdmissionRequest
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class _ReceiptState:
    fingerprint: tuple[str, str]
    receipt_id: str
    emitted_at: float
    coalesced_count: int = 0


PressureProvider = Callable[[], PressureSnapshot | Mapping[str, Any]]


def _default_pressure_provider() -> PressureSnapshot:
    raw: dict[str, Any] = {}
    try:
        from core.runtime.runtime_pressure import get_unified_runtime_pressure

        raw.update(get_unified_runtime_pressure().runtime_pressure_snapshot())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("runtime pressure provider unavailable: %s", exc)

    try:
        from core.resource.resource_governor import get_resource_governor

        resource = get_resource_governor().get_snapshot()
        if resource is not None:
            raw["memory_percent"] = float(resource.memory_percent)
            raw["memory_rss_mb"] = float(resource.memory_rss_mb)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("resource sampler unavailable: %s", exc)

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        raw["shutdown_requested"] = bool(is_shutdown_requested())
    except (ImportError, RuntimeError) as exc:
        logger.debug("shutdown state unavailable: %s", exc)

    try:
        from core.runtime.service_registry import get_runtime_service

        stakes = get_runtime_service("resource_stakes", default=None)
        if stakes is not None and hasattr(stakes, "state"):
            state = stakes.state()
            raw["suspended_capabilities"] = tuple(
                getattr(state, "suspended_capabilities", ()) or ()
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("resource stakes unavailable: %s", exc)

    return PressureSnapshot.from_mapping(raw)


class ResourceAdmissionController:
    """Loop-agnostic, priority-aware owner of constrained work leases."""

    MEMORY_MODERATE_PERCENT = 85.0
    MEMORY_CRITICAL_PERCENT = 92.0
    LOOP_LAG_BACKGROUND_S = 1.0
    THERMAL_SERIOUS_LEVEL = 2
    THERMAL_CRITICAL_LEVEL = 3

    def __init__(
        self,
        *,
        pressure_provider: PressureProvider | None = None,
        history_limit: int = 512,
        poll_interval_s: float = 0.05,
        receipt_store: Any | None = None,
    ) -> None:
        self._pressure_provider = pressure_provider or _default_pressure_provider
        self._history: deque[dict[str, Any]] = deque(maxlen=max(16, int(history_limit)))
        self._poll_interval_s = min(1.0, max(0.01, float(poll_interval_s)))
        self._receipt_store = receipt_store
        self._lock = threading.RLock()
        self._receipt_lock = threading.RLock()
        self._last_pressure: PressureSnapshot | None = None
        self._last_pressure_at = 0.0
        self._pressure_cache_s = 0.5
        self._leases: dict[str, _LeaseRecord] = {}
        self._request_to_lease: dict[str, str] = {}
        self._waiters: dict[str, _Waiter] = {}
        self._sequence = 0
        self._admitted = 0
        self._deferred = 0
        self._rejected = 0
        self._timed_out = 0
        self._preemptions = 0
        self._expired = 0
        self._receipt_states: dict[tuple[str, str, str], _ReceiptState] = {}
        self._receipt_state_limit = max(64, int(history_limit))
        self._receipt_coalesced = 0

    def pressure_snapshot(self) -> PressureSnapshot:
        now = time.monotonic()
        with self._lock:
            if (
                self._last_pressure is not None
                and now - self._last_pressure_at <= self._pressure_cache_s
            ):
                return self._last_pressure
        try:
            raw = self._pressure_provider()
            snapshot = (
                raw
                if isinstance(raw, PressureSnapshot)
                else PressureSnapshot.from_mapping(raw)
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "runtime_control_plane",
                exc,
                severity="warning",
                action="resource pressure sampling failed; background admission fails closed",
                receipt_required=False,
            )
            snapshot = PressureSnapshot(
                observation_source="unavailable",
                red_zones=("pressure_provider_unavailable",),
            )
        with self._lock:
            self._last_pressure = snapshot
            self._last_pressure_at = time.monotonic()
        self._publish_pressure_conditions(snapshot)
        return snapshot

    @staticmethod
    def _publish_pressure_conditions(snapshot: PressureSnapshot) -> None:
        try:
            from core.runtime.conditions import (
                ConditionType,
                get_component_conditions,
            )

            conditions = get_component_conditions("resource_admission")
            unavailable = "pressure_provider_unavailable" in snapshot.red_zones
            loop_signal_unavailable = (
                snapshot.loop_monitor_running is False
                or not snapshot.loop_lag_sample_fresh
            )
            ready = (
                not snapshot.shutdown_requested
                and not unavailable
                and not loop_signal_unavailable
            )
            conditions.set(
                ConditionType.READY,
                ready,
                reason=(
                    "ShutdownRequested"
                    if snapshot.shutdown_requested
                    else "PressureProviderUnavailable"
                    if unavailable
                    else "EventLoopSignalUnavailable"
                    if loop_signal_unavailable
                    else "PressureObserved"
                ),
                message=(
                    f"memory={snapshot.memory_percent:.1f}% "
                    f"thermal={snapshot.thermal_level} lag={snapshot.loop_lag_s:.3f}s "
                    f"source={snapshot.observation_source}"
                ),
            )
            degraded = bool(
                snapshot.red_zones
                or snapshot.suspended_capabilities
                or snapshot.memory_percent
                >= ResourceAdmissionController.MEMORY_CRITICAL_PERCENT
                or snapshot.thermal_level
                >= ResourceAdmissionController.THERMAL_CRITICAL_LEVEL
            )
            conditions.set(
                ConditionType.DEGRADED,
                degraded,
                reason="ResourcePressure" if degraded else "WithinEnvelope",
                message=(
                    f"red_zones={','.join(snapshot.red_zones) or 'none'}; "
                    "suspended="
                    f"{','.join(snapshot.suspended_capabilities) or 'none'}"
                ),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return

    async def pressure_snapshot_async(self) -> PressureSnapshot:
        with self._lock:
            fresh = (
                self._last_pressure is not None
                and time.monotonic() - self._last_pressure_at <= self._pressure_cache_s
            )
            cached = self._last_pressure
        if fresh and cached is not None:
            return cached
        return await asyncio.to_thread(self.pressure_snapshot)

    @staticmethod
    def _work_conflicts(left: AdmissionRequest, right: AdmissionRequest) -> bool:
        if left.work_class == WorkClass.INFERENCE and right.work_class == WorkClass.INFERENCE:
            if bool(left.metadata.get("global_inference_scope")) or bool(
                right.metadata.get("global_inference_scope")
            ):
                return True
            return left.lane == right.lane
        if left.work_class == WorkClass.EVOLUTION and right.work_class in {
            WorkClass.INFERENCE,
            WorkClass.EVOLUTION,
            WorkClass.MODEL_LOAD,
        }:
            return True
        if right.work_class == WorkClass.EVOLUTION and left.work_class in {
            WorkClass.INFERENCE,
            WorkClass.EVOLUTION,
            WorkClass.MODEL_LOAD,
        }:
            return True
        if left.work_class == WorkClass.MODEL_LOAD and right.work_class == WorkClass.MODEL_LOAD:
            return True
        if left.work_class == WorkClass.MODEL_LOAD and right.work_class == WorkClass.INFERENCE:
            if left.lane == right.lane:
                return False
            # Cross-lane: a user-facing inference (serving the current turn on the
            # fallback ladder) must not be blocked by a model load that only warms a
            # model for future turns — serving now outranks warming for later. Memory
            # is still enforced by lane_admission and GPU compute by the single-slot
            # local semaphore, so background inference keeps yielding to the load.
            return right.priority > AdmissionPriority.INTERACTIVE
        if right.work_class == WorkClass.MODEL_LOAD and left.work_class == WorkClass.INFERENCE:
            if left.lane == right.lane:
                return False
            return left.priority > AdmissionPriority.INTERACTIVE
        if left.work_class in {WorkClass.DESKTOP, WorkClass.SERVICE_START}:
            return left.work_class == right.work_class and left.lane == right.lane
        if left.work_class == WorkClass.MAINTENANCE:
            return right.work_class == WorkClass.MAINTENANCE and left.lane == right.lane
        if left.work_class == WorkClass.BACKGROUND:
            return right.work_class == WorkClass.BACKGROUND and left.lane == right.lane
        return False

    @classmethod
    def _pressure_block_reason(
        cls,
        request: AdmissionRequest,
        pressure: PressureSnapshot,
    ) -> tuple[str, bool] | None:
        if pressure.shutdown_requested:
            return "runtime_shutdown_requested", True

        suspended = set(pressure.suspended_capabilities)
        if request.work_class == WorkClass.BACKGROUND and "background_exploration" in suspended:
            return "background_capability_suspended", True
        if request.work_class == WorkClass.MODEL_LOAD and "large_model_cortex" in suspended:
            return "large_model_capability_suspended", True

        is_foreground = request.priority <= AdmissionPriority.FOREGROUND
        if pressure.memory_percent >= cls.MEMORY_CRITICAL_PERCENT:
            if request.work_class in {WorkClass.EVOLUTION, WorkClass.MODEL_LOAD, WorkClass.BACKGROUND}:
                return f"critical_memory_pressure_{pressure.memory_percent:.1f}", False
            if not is_foreground:
                return f"critical_memory_pressure_{pressure.memory_percent:.1f}", False
        elif pressure.memory_percent >= cls.MEMORY_MODERATE_PERCENT:
            if request.priority >= AdmissionPriority.MAINTENANCE or request.work_class in {
                WorkClass.EVOLUTION,
                WorkClass.MODEL_LOAD,
                WorkClass.BACKGROUND,
            }:
                return f"moderate_memory_pressure_{pressure.memory_percent:.1f}", False

        if pressure.thermal_level >= cls.THERMAL_CRITICAL_LEVEL:
            if not is_foreground or request.work_class in {
                WorkClass.EVOLUTION,
                WorkClass.MODEL_LOAD,
                WorkClass.BACKGROUND,
            }:
                return f"critical_thermal_pressure_{pressure.thermal_level}", False
        elif pressure.thermal_level >= cls.THERMAL_SERIOUS_LEVEL:
            if request.priority >= AdmissionPriority.MAINTENANCE:
                return f"serious_thermal_pressure_{pressure.thermal_level}", False

        if (
            (
                pressure.loop_monitor_running is False
                or not pressure.loop_lag_sample_fresh
            )
            and request.priority >= AdmissionPriority.MAINTENANCE
        ):
            return "event_loop_signal_unavailable", False
        if (
            pressure.loop_lag_s >= cls.LOOP_LAG_BACKGROUND_S
            and request.priority >= AdmissionPriority.MAINTENANCE
        ):
            return f"event_loop_lag_{pressure.loop_lag_s:.3f}s", False
        if "pressure_provider_unavailable" in pressure.red_zones and request.priority > AdmissionPriority.FOREGROUND:
            return "pressure_provider_unavailable", False
        return None

    def _expire_leases_locked(self, now: float) -> list[_LeaseRecord]:
        expired: list[_LeaseRecord] = []
        for lease_id, lease in list(self._leases.items()):
            # A held resource does not become available because its estimated
            # duration elapsed. The task's cancellation/fault policy owns work;
            # admission owns mutual exclusion until release or owner exit.
            if lease.holder_task is not None and not lease.holder_task.done():
                continue
            if lease.holder_task is None and lease.expires_at > now:
                continue
            expired.append(lease)
            self._leases.pop(lease_id, None)
            self._request_to_lease.pop(lease.request.request_id, None)
            self._expired += 1
            self._append_history_locked(
                lease.request,
                AdmissionOutcome.EXPIRED,
                "holder_task_finished" if lease.holder_task is not None else "lease_ttl_expired",
                lease_id=lease_id,
            )
        return expired

    def _blocking_leases_locked(self, request: AdmissionRequest) -> list[_LeaseRecord]:
        return [
            lease
            for lease in self._leases.values()
            if self._work_conflicts(request, lease.request)
        ]

    def _older_waiter_blocks_locked(self, waiter: _Waiter) -> bool:
        for other in self._waiters.values():
            if other.request.request_id == waiter.request.request_id:
                continue
            if other.sequence >= waiter.sequence:
                continue
            if other.request.priority > waiter.request.priority:
                continue
            if self._work_conflicts(waiter.request, other.request):
                return True
        return False

    def _append_history_locked(
        self,
        request: AdmissionRequest,
        outcome: AdmissionOutcome,
        reason: str,
        *,
        lease_id: str = "",
        blocking_lease_ids: tuple[str, ...] = (),
    ) -> None:
        self._history.append(
            {
                "at": time.time(),
                "request_id": request.request_id,
                "owner": request.owner,
                "work_class": request.work_class.value,
                "lane": request.lane,
                "priority": int(request.priority),
                "outcome": outcome.value,
                "reason": reason,
                "lease_id": lease_id,
                "blocking_lease_ids": list(blocking_lease_ids),
            }
        )

    @staticmethod
    def _receipt_state_key(request: AdmissionRequest) -> tuple[str, str, str]:
        return (request.owner, request.work_class.value, request.lane)

    @staticmethod
    def _receipt_reason_class(reason: str) -> str:
        normalized = str(reason or "unknown")
        for prefix in (
            "critical_memory_pressure_",
            "moderate_memory_pressure_",
            "critical_thermal_pressure_",
            "serious_thermal_pressure_",
            "event_loop_lag_",
        ):
            if normalized.startswith(prefix):
                return prefix.rstrip("_")
        return normalized

    def _emit_receipt(
        self,
        request: AdmissionRequest,
        outcome: AdmissionOutcome,
        reason: str,
        pressure: PressureSnapshot,
        *,
        lease_id: str = "",
        blocking_lease_ids: tuple[str, ...] = (),
        force: bool = False,
    ) -> tuple[str, bool]:
        state_key = self._receipt_state_key(request)
        if not (request.receipt_required or force):
            if outcome == AdmissionOutcome.ADMITTED:
                with self._receipt_lock:
                    self._receipt_states.pop(state_key, None)
            return "", False

        coalescible = (
            not request.receipt_required
            and outcome
            in {
                AdmissionOutcome.DEFERRED,
                AdmissionOutcome.REJECTED,
                AdmissionOutcome.TIMED_OUT,
            }
        )
        fingerprint = (outcome.value, self._receipt_reason_class(reason))
        heartbeat_s = max(0.0, float(_ADMISSION_RECEIPT_HEARTBEAT_FLAG.value()))
        now = time.time()

        with self._receipt_lock:
            prior = self._receipt_states.get(state_key)
            if (
                coalescible
                and heartbeat_s > 0.0
                and prior is not None
                and prior.fingerprint == fingerprint
                and now - prior.emitted_at < heartbeat_s
            ):
                prior.coalesced_count += 1
                self._receipt_coalesced += 1
                return prior.receipt_id, True

            try:
                from core.runtime.receipts import (
                    ResourceAdmissionReceipt,
                    get_receipt_store,
                )

                metadata = {
                    **dict(request.metadata),
                    "blocking_lease_ids": list(blocking_lease_ids),
                }
                if prior is not None and prior.coalesced_count:
                    metadata.update(
                        {
                            "coalesced_since_prior": prior.coalesced_count,
                            "prior_receipt_id": prior.receipt_id,
                        }
                    )
                store = self._receipt_store or get_receipt_store()
                receipt = store.emit(
                    ResourceAdmissionReceipt(
                        cause=f"{request.owner}:{request.work_class.value}",
                        request_id=request.request_id,
                        owner=request.owner,
                        work_class=request.work_class.value,
                        lane=request.lane,
                        priority=int(request.priority),
                        decision=outcome.value,
                        reason=reason,
                        lease_id=lease_id,
                        pressure=pressure.to_dict(),
                        metadata=metadata,
                    )
                )
                receipt_id = str(receipt.receipt_id)
                if coalescible:
                    self._receipt_states[state_key] = _ReceiptState(
                        fingerprint=fingerprint,
                        receipt_id=receipt_id,
                        emitted_at=now,
                    )
                    while len(self._receipt_states) > self._receipt_state_limit:
                        oldest_key = min(
                            self._receipt_states,
                            key=lambda key: self._receipt_states[key].emitted_at,
                        )
                        self._receipt_states.pop(oldest_key, None)
                elif outcome == AdmissionOutcome.ADMITTED:
                    self._receipt_states.pop(state_key, None)
                return receipt_id, False
            except (
                ImportError,
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                record_degradation(
                    "runtime_control_plane",
                    exc,
                    severity="warning",
                    action=(
                        "resource admission decision remained in bounded memory "
                        "but durable receipt failed"
                    ),
                    receipt_required=False,
                )
                return "", False

    async def _emit_receipt_async(
        self,
        request: AdmissionRequest,
        outcome: AdmissionOutcome,
        reason: str,
        pressure: PressureSnapshot,
        *,
        lease_id: str = "",
        blocking_lease_ids: tuple[str, ...] = (),
        force: bool = False,
    ) -> tuple[str, bool]:
        if not (request.receipt_required or force):
            if outcome == AdmissionOutcome.ADMITTED:
                with self._receipt_lock:
                    self._receipt_states.pop(self._receipt_state_key(request), None)
            return "", False
        return await asyncio.to_thread(
            self._emit_receipt,
            request,
            outcome,
            reason,
            pressure,
            lease_id=lease_id,
            blocking_lease_ids=blocking_lease_ids,
            force=force,
        )

    async def acquire(
        self,
        request: AdmissionRequest,
        *,
        on_preempt: Callable[[str], Any] | None = None,
        holder_task: asyncio.Task[Any] | None = None,
    ) -> AdmissionDecision:
        if holder_task is not None and holder_task is not asyncio.current_task():
            raise ValueError("admission holder must be the acquiring task")
        started = time.monotonic()
        deadline = started + float(request.timeout_s)
        initial_pressure = await self.pressure_snapshot_async()
        with self._lock:
            self._expire_leases_locked(time.monotonic())
            existing_id = self._request_to_lease.get(request.request_id)
            existing = self._leases.get(existing_id or "")
            if existing is not None:
                if holder_task is not None and existing.holder_task is not holder_task:
                    raise ValueError("active admission lease belongs to another task")
                return AdmissionDecision(
                    request_id=request.request_id,
                    outcome=AdmissionOutcome.ADMITTED,
                    reason="idempotent_active_lease",
                    decided_at=time.time(),
                    lease_id=existing.lease_id,
                    pressure=initial_pressure,
                    replayed=True,
                )
            if request.request_id in self._waiters:
                raise ValueError(f"duplicate in-flight admission request_id: {request.request_id}")
            self._sequence += 1
            waiter = _Waiter(self._sequence, request)
            self._waiters[request.request_id] = waiter

        preemption_notified: set[str] = set()
        attempt_limit = max(
            1,
            int(math.ceil(float(request.timeout_s) / self._poll_interval_s)) + 2,
        )
        try:
            for _attempt in range(attempt_limit):
                pressure = await self.pressure_snapshot_async()
                pressure_block = self._pressure_block_reason(request, pressure)
                callbacks: list[tuple[_LeaseRecord, Callable[[str], Any]]] = []
                admitted_lease_id = ""
                with self._lock:
                    self._expire_leases_locked(time.monotonic())
                    blockers = self._blocking_leases_locked(request)
                    fairness_blocked = self._older_waiter_blocks_locked(waiter)

                    if pressure_block is None and not blockers and not fairness_blocked:
                        lease_id = f"lease-{uuid.uuid4()}"
                        now = time.monotonic()
                        self._leases[lease_id] = _LeaseRecord(
                            lease_id=lease_id,
                            request=request,
                            admitted_at=time.time(),
                            expires_at=now + float(request.lease_ttl_s),
                            on_preempt=on_preempt,
                            holder_task=holder_task,
                        )
                        self._request_to_lease[request.request_id] = lease_id
                        self._waiters.pop(request.request_id, None)
                        self._admitted += 1
                        self._append_history_locked(
                            request,
                            AdmissionOutcome.ADMITTED,
                            "capacity_available",
                            lease_id=lease_id,
                        )
                        admitted_lease_id = lease_id

                    if (
                        not admitted_lease_id
                        and blockers
                        and request.priority <= AdmissionPriority.FOREGROUND
                    ):
                        for lease in blockers:
                            if (
                                lease.request.preemptible
                                and lease.request.priority > request.priority
                                and lease.on_preempt is not None
                                and lease.lease_id not in preemption_notified
                            ):
                                lease.preempt_requested = True
                                lease.preempt_reason = f"preempted_by:{request.request_id}"
                                callbacks.append((lease, lease.on_preempt))
                                preemption_notified.add(lease.lease_id)
                                self._preemptions += 1

                if admitted_lease_id:
                    receipt_id, receipt_replayed = await self._emit_receipt_async(
                        request,
                        AdmissionOutcome.ADMITTED,
                        "capacity_available",
                        pressure,
                        lease_id=admitted_lease_id,
                    )
                    return AdmissionDecision(
                        request_id=request.request_id,
                        outcome=AdmissionOutcome.ADMITTED,
                        reason="capacity_available",
                        decided_at=time.time(),
                        lease_id=admitted_lease_id,
                        receipt_id=receipt_id,
                        receipt_replayed=receipt_replayed,
                        pressure=pressure,
                    )

                for lease, callback in callbacks:
                    try:
                        result = callback(lease.preempt_reason)
                        if inspect.isawaitable(result):
                            await result
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        record_degradation(
                            "runtime_control_plane",
                            exc,
                            severity="warning",
                            action="preemption callback failed; blocking lease remains active",
                            receipt_required=False,
                            extra={"lease_id": lease.lease_id},
                        )

                now = time.monotonic()
                if pressure_block is not None and pressure_block[1]:
                    outcome = AdmissionOutcome.REJECTED
                    reason = pressure_block[0]
                elif request.timeout_s == 0:
                    outcome = AdmissionOutcome.DEFERRED
                    reason = pressure_block[0] if pressure_block else (
                        "fairness_wait" if fairness_blocked else "resource_busy"
                    )
                elif now >= deadline:
                    outcome = AdmissionOutcome.TIMED_OUT
                    reason = pressure_block[0] if pressure_block else (
                        "fairness_timeout" if fairness_blocked else "resource_timeout"
                    )
                else:
                    await asyncio.sleep(min(self._poll_interval_s, max(0.0, deadline - now)))
                    continue

                blocking_ids = tuple(lease.lease_id for lease in blockers)
                with self._lock:
                    self._waiters.pop(request.request_id, None)
                    if outcome == AdmissionOutcome.REJECTED:
                        self._rejected += 1
                    elif outcome == AdmissionOutcome.TIMED_OUT:
                        self._timed_out += 1
                    else:
                        self._deferred += 1
                    self._append_history_locked(
                        request,
                        outcome,
                        reason,
                        blocking_lease_ids=blocking_ids,
                    )
                receipt_id, receipt_replayed = await self._emit_receipt_async(
                    request,
                    outcome,
                    reason,
                    pressure,
                    blocking_lease_ids=blocking_ids,
                    force=True,
                )
                return AdmissionDecision(
                    request_id=request.request_id,
                    outcome=outcome,
                    reason=reason,
                    decided_at=time.time(),
                    receipt_id=receipt_id,
                    receipt_replayed=receipt_replayed,
                    blocking_lease_ids=blocking_ids,
                    pressure=pressure,
                )
            raise RuntimeError("admission attempt budget exhausted without a terminal decision")
        finally:
            with self._lock:
                self._waiters.pop(request.request_id, None)

    async def release(
        self,
        lease_id: str,
        *,
        reason: str = "completed",
        preempted: bool = False,
    ) -> AdmissionDecision:
        pressure = await self.pressure_snapshot_async()
        with self._lock:
            lease = self._leases.pop(str(lease_id), None)
            if lease is None:
                raise KeyError(f"unknown or already released admission lease: {lease_id}")
            self._request_to_lease.pop(lease.request.request_id, None)
            outcome = AdmissionOutcome.PREEMPTED if preempted else AdmissionOutcome.RELEASED
            self._append_history_locked(
                lease.request,
                outcome,
                reason,
                lease_id=lease.lease_id,
            )
        receipt_id, receipt_replayed = await self._emit_receipt_async(
            lease.request,
            outcome,
            reason,
            pressure,
            lease_id=lease.lease_id,
            force=preempted,
        )
        return AdmissionDecision(
            request_id=lease.request.request_id,
            outcome=outcome,
            reason=reason,
            decided_at=time.time(),
            lease_id=lease.lease_id,
            receipt_id=receipt_id,
            receipt_replayed=receipt_replayed,
            pressure=pressure,
        )

    def release_sync(
        self,
        lease_id: str,
        *,
        reason: str = "completed",
    ) -> AdmissionDecision:
        """Release a receipt-free compatibility lease from synchronous APIs.

        New runtime callers must use :meth:`release`. This narrow bridge exists
        for legacy synchronous ``release()`` methods whose acquisition is
        already async. Receipt-bearing leases are rejected so durable I/O can
        never be smuggled onto an event-loop thread through this adapter.
        """

        pressure = self.pressure_snapshot()
        with self._lock:
            lease = self._leases.get(str(lease_id))
            if lease is None:
                raise KeyError(f"unknown or already released admission lease: {lease_id}")
            if lease.request.receipt_required:
                raise RuntimeError(
                    "receipt-bearing admission leases require async release"
                )
            self._leases.pop(str(lease_id), None)
            self._request_to_lease.pop(lease.request.request_id, None)
            self._append_history_locked(
                lease.request,
                AdmissionOutcome.RELEASED,
                reason,
                lease_id=lease.lease_id,
            )
        return AdmissionDecision(
            request_id=lease.request.request_id,
            outcome=AdmissionOutcome.RELEASED,
            reason=reason,
            decided_at=time.time(),
            lease_id=lease.lease_id,
            pressure=pressure,
        )

    def reap_dead_holder_leases_sync(
        self,
        *,
        lane: str,
        work_class: WorkClass = WorkClass.MODEL_LOAD,
        reason: str = "holder_died",
    ) -> int:
        """Release leases whose holder is known dead — the 2026-07-15 soak P0.

        A MODEL_LOAD lease conflicts with every other MODEL_LOAD lease, so a
        cortex-load lease that dies without release (worker killed mid-load,
        handshake failure) walls every recovery load behind its TTL while
        each retry burns its own ``timeout_s`` into ``resource_timeout`` —
        and the K1 reconciler retries into the same wall forever. The
        worker-death seam (``_note_lane_worker_death``) calls this so the
        wall falls within one poll interval, not one TTL.

        Thread-safe and receipt-free: history carries the honest outcome
        (``holder_died:<reason>``), and the holder that would have received
        the release receipt is dead. The dead holder's own late ``release()``
        becomes the already-tolerated KeyError path ("lease expired before
        release"). Worst case a late death note reaps a NEWER same-lane
        load's lease: that load keeps running (the global spawn mutex still
        serializes real spawns beneath admission) and its release logs the
        same tolerated line — bounded, and strictly better than the
        deadlock. Never raises.
        """
        reaped = 0
        try:
            with self._lock:
                victims = [
                    lease
                    for lease in self._leases.values()
                    if lease.request.work_class is work_class
                    and str(lease.request.lane) == str(lane)
                ]
                for lease in victims:
                    self._leases.pop(lease.lease_id, None)
                    self._request_to_lease.pop(lease.request.request_id, None)
                    self._expired += 1
                    self._append_history_locked(
                        lease.request,
                        AdmissionOutcome.EXPIRED,
                        f"holder_died:{reason}"[:120],
                        lease_id=lease.lease_id,
                    )
                    reaped += 1
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "resource_admission",
                exc,
                action="continued without reaping dead-holder leases",
            )
        return reaped

    def active_lease_count(self, work_class: WorkClass | None = None) -> int:
        with self._lock:
            self._expire_leases_locked(time.monotonic())
            if work_class is None:
                return len(self._leases)
            return sum(
                lease.request.work_class == work_class for lease in self._leases.values()
            )

    def is_alive(self) -> bool:
        try:
            self.pressure_snapshot()
            return True
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def is_ready(self) -> bool:
        return self.is_alive()

    def status(self) -> dict[str, Any]:
        pressure = self.pressure_snapshot()
        now_monotonic = time.monotonic()
        with self._receipt_lock:
            receipt_state_count = len(self._receipt_states)
            receipt_coalesced = self._receipt_coalesced
        with self._lock:
            self._expire_leases_locked(time.monotonic())
            leases = [
                {
                    "lease_id": lease.lease_id,
                    "request_id": lease.request.request_id,
                    "owner": lease.request.owner,
                    "work_class": lease.request.work_class.value,
                    "lane": lease.request.lane,
                    "priority": int(lease.request.priority),
                    "admitted_at": lease.admitted_at,
                    "expires_at_monotonic": lease.expires_at,
                    "ttl_remaining_s": max(0.0, lease.expires_at - now_monotonic),
                    "lifetime": "holder_task" if lease.holder_task is not None else "ttl",
                    "holder_task_active": (
                        not lease.holder_task.done() if lease.holder_task is not None else None
                    ),
                    "preempt_requested": lease.preempt_requested,
                    "preempt_reason": lease.preempt_reason,
                }
                for lease in self._leases.values()
            ]
            waiters = [
                {
                    "request_id": waiter.request.request_id,
                    "owner": waiter.request.owner,
                    "work_class": waiter.request.work_class.value,
                    "lane": waiter.request.lane,
                    "priority": int(waiter.request.priority),
                    "sequence": waiter.sequence,
                    "enqueued_at": waiter.enqueued_at,
                    "wait_s": max(0.0, time.time() - waiter.enqueued_at),
                    "timeout_s": float(waiter.request.timeout_s),
                }
                for waiter in sorted(self._waiters.values(), key=lambda item: item.sequence)
            ]
            return {
                "alive": True,
                "pressure": pressure.to_dict(),
                "active_leases": leases,
                "waiters": waiters,
                "history": list(self._history),
                "counters": {
                    "admitted": self._admitted,
                    "deferred": self._deferred,
                    "rejected": self._rejected,
                    "timed_out": self._timed_out,
                    "preemptions": self._preemptions,
                    "expired": self._expired,
                    "receipt_coalesced": receipt_coalesced,
                },
                "receipt_state_count": receipt_state_count,
            }


StartCallback = Callable[[], Any | Awaitable[Any]]
StopCallback = Callable[[], Any | Awaitable[Any]]
ProbeCallback = Callable[[], Any | Awaitable[Any]]


@dataclass(frozen=True)
class DesiredServiceSpec:
    name: str
    critical: bool = False
    dependencies: tuple[str, ...] = ()
    desired_state: DesiredServiceState = DesiredServiceState.RUNNING
    start_timeout_s: float = 10.0
    stop_timeout_s: float = 5.0
    restart_limit: int = 3
    restart_window_s: float = 60.0
    backoff_initial_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 30.0
    restart_on_unhealthy: bool = True
    admission_class: WorkClass = WorkClass.SERVICE_START
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("service name must be non-empty")
        if self.name in self.dependencies:
            raise ValueError(f"service {self.name} cannot depend on itself")
        if self.start_timeout_s <= 0 or self.stop_timeout_s <= 0:
            raise ValueError("service timeouts must be positive")
        if self.restart_limit < 0:
            raise ValueError("restart_limit must be non-negative")
        if self.restart_window_s <= 0:
            raise ValueError("restart_window_s must be positive")


@dataclass
class ServiceObservation:
    name: str
    desired_state: DesiredServiceState
    observed_state: ObservedServiceState = ObservedServiceState.UNKNOWN
    reason: str = "registered"
    generation: int = 0
    last_transition_at: float = field(default_factory=time.time)
    last_probe_at: float = 0.0
    restart_times: list[float] = field(default_factory=list)
    next_retry_at: float = 0.0
    last_error: str = ""
    admission_receipt_id: str = ""
    #: Independent named claims about this service. A single collapsed
    #: observed_state cannot express "loaded but not accepting foreground
    #: work" or "degraded while recovering normally" — those are separate
    #: facts, and squashing them forces every consumer to re-derive the
    #: distinction from a prose reason string.
    conditions: Any = None
    #: Cleanups that must complete before this service counts as stopped.
    finalizers: Any = None

    def condition_set(self) -> Any:
        """The live condition set, created on first use."""
        if self.conditions is None:
            from core.runtime.service_conditions import ConditionSet

            self.conditions = ConditionSet(generation=self.generation)
        return self.conditions

    def finalizer_set(self) -> Any:
        if self.finalizers is None:
            from core.runtime.service_conditions import FinalizerSet

            self.finalizers = FinalizerSet()
        return self.finalizers

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["desired_state"] = self.desired_state.value
        payload["observed_state"] = self.observed_state.value
        payload["conditions"] = (
            self.conditions.to_dict() if self.conditions is not None else None
        )
        payload["finalizers"] = (
            self.finalizers.to_dict() if self.finalizers is not None else None
        )
        return payload


@dataclass
class _ServiceBinding:
    spec: DesiredServiceSpec
    start: StartCallback
    stop: StopCallback
    probe: ProbeCallback
    observation: ServiceObservation


class RuntimeControlPlane:
    """Reconciles registered service adapters toward declared desired state."""

    def __init__(self, *, admission: ResourceAdmissionController | None = None) -> None:
        self.admission = admission or ResourceAdmissionController()
        self._lock = threading.RLock()
        self._reconcile_lock = threading.Lock()
        self._services: dict[str, _ServiceBinding] = {}
        self._last_report: dict[str, Any] = {}
        self._closed = False
        self._reconcile_count = 0

    @staticmethod
    async def _call(callback: Callable[[], Any], timeout_s: float) -> Any:
        started = time.monotonic()
        if inspect.iscoroutinefunction(callback):
            return await asyncio.wait_for(callback(), timeout=timeout_s)
        result = await asyncio.wait_for(asyncio.to_thread(callback), timeout=timeout_s)
        if inspect.isawaitable(result):
            remaining = max(0.01, timeout_s - (time.monotonic() - started))
            return await asyncio.wait_for(result, timeout=remaining)
        return result

    def register_service(
        self,
        spec: DesiredServiceSpec,
        *,
        start: StartCallback,
        stop: StopCallback,
        probe: ProbeCallback,
        adopt_running: bool = False,
    ) -> None:
        with self._lock:
            if spec.name in self._services:
                raise ValueError(f"service already registered with control plane: {spec.name}")
            observation = ServiceObservation(
                name=spec.name,
                desired_state=spec.desired_state,
                observed_state=(
                    ObservedServiceState.READY
                    if adopt_running
                    else ObservedServiceState.STOPPED
                    if spec.desired_state == DesiredServiceState.STOPPED
                    else ObservedServiceState.UNKNOWN
                ),
                reason=(
                    "adopted_running"
                    if adopt_running
                    else "registered_stopped"
                    if spec.desired_state == DesiredServiceState.STOPPED
                    else "registered"
                ),
            )
            self._services[spec.name] = _ServiceBinding(
                spec=spec,
                start=start,
                stop=stop,
                probe=probe,
                observation=observation,
            )
            try:
                self._topological_order_locked()
            except ValueError:
                self._services.pop(spec.name, None)
                raise

    def set_desired_state(self, name: str, desired: DesiredServiceState) -> None:
        with self._lock:
            binding = self._services.get(name)
            if binding is None:
                raise KeyError(name)
            binding.observation.desired_state = DesiredServiceState(desired)
            binding.observation.last_transition_at = time.time()
            binding.observation.reason = f"desired_{desired.value}"

    def has_service(self, name: str) -> bool:
        with self._lock:
            return str(name) in self._services

    def _topological_order_locked(self) -> list[str]:
        temporary: set[str] = set()
        permanent: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError(f"control-plane dependency cycle includes {name}")
            temporary.add(name)
            binding = self._services[name]
            for dependency in binding.spec.dependencies:
                if dependency in self._services:
                    visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            order.append(name)

        for service_name in sorted(self._services):
            visit(service_name)
        return order

    @staticmethod
    def _transition(
        observation: ServiceObservation,
        state: ObservedServiceState,
        reason: str,
        *,
        error: str = "",
    ) -> None:
        if observation.observed_state != state:
            observation.last_transition_at = time.time()
        observation.observed_state = state
        observation.reason = reason
        observation.last_error = error
        # Derive the independent claims from the state the reconciler just
        # observed, so conditions are populated by real reconciliation rather
        # than being a parallel surface someone has to remember to update.
        RuntimeControlPlane._sync_conditions(observation, state, reason, error)

    @staticmethod
    def _sync_conditions(
        observation: ServiceObservation,
        state: ObservedServiceState,
        reason: str,
        error: str,
    ) -> None:
        """Project an observed state onto independent conditions.

        Never raises: conditions are an observability surface, and a reporting
        failure must not break the reconciliation that produced it.
        """
        try:
            from core.runtime.service_conditions import (
                ConditionStatus,
                ConditionType,
            )

            conditions = observation.condition_set()
            value = getattr(state, "value", str(state))

            alive = value in {"running", "degraded", "starting", "stopping"}
            ready = value in {"running", "degraded"}
            terminating = value == "stopping"

            conditions.set(
                ConditionType.ALIVE,
                ConditionStatus.TRUE if alive else (
                    ConditionStatus.UNKNOWN if value == "unknown"
                    else ConditionStatus.FALSE
                ),
                reason=reason,
                message=error,
            )
            conditions.set(
                ConditionType.READY,
                ConditionStatus.TRUE if ready else (
                    ConditionStatus.UNKNOWN if value == "unknown"
                    else ConditionStatus.FALSE
                ),
                reason=reason,
            )
            # Degraded is running-but-not-fully-capable; it is NOT down, and a
            # consumer that treats it as down takes a working service offline.
            conditions.set(
                ConditionType.DEGRADED,
                ConditionStatus.TRUE if value == "degraded" else ConditionStatus.FALSE,
                reason=reason,
            )
            conditions.set(
                ConditionType.TERMINATING,
                ConditionStatus.TRUE if terminating else ConditionStatus.FALSE,
                reason=reason,
            )
            # A service still inside its restart budget is recovering, not
            # merely failing — the distinction decides whether to escalate.
            recovering = bool(observation.next_retry_at) and value != "running"
            conditions.set(
                ConditionType.RECOVERING,
                ConditionStatus.TRUE if recovering else ConditionStatus.FALSE,
                reason=f"next_retry_at={observation.next_retry_at:.0f}" if recovering else reason,
            )
        except (ImportError, AttributeError, TypeError, ValueError, KeyError):
            return

    @staticmethod
    def _probe_ok(value: Any) -> bool:
        if isinstance(value, Mapping):
            if "ok" in value:
                return bool(value["ok"])
            if "alive" in value:
                return bool(value["alive"])
            for key in ("ready", "healthy", "operational", "pressure_ok"):
                if key in value:
                    return bool(value[key])
        return bool(value)

    async def _probe(self, binding: _ServiceBinding) -> bool:
        binding.observation.last_probe_at = time.time()
        result = await self._call(binding.probe, max(0.1, binding.spec.start_timeout_s))
        return self._probe_ok(result)

    def _restart_budget_available(self, binding: _ServiceBinding, now: float) -> bool:
        observation = binding.observation
        cutoff = now - binding.spec.restart_window_s
        observation.restart_times = [stamp for stamp in observation.restart_times if stamp >= cutoff]
        if (
            observation.generation == 0
            and not observation.restart_times
            and observation.observed_state
            in {
                ObservedServiceState.UNKNOWN,
                ObservedServiceState.STOPPED,
                ObservedServiceState.BLOCKED,
            }
        ):
            return True
        return len(observation.restart_times) < binding.spec.restart_limit

    def _schedule_restart(
        self,
        binding: _ServiceBinding,
        *,
        reason: str,
        error: str = "",
    ) -> ObservedServiceState:
        observation = binding.observation
        now = time.time()
        cutoff = now - binding.spec.restart_window_s
        observation.restart_times = [stamp for stamp in observation.restart_times if stamp >= cutoff]
        observation.restart_times.append(now)
        count = len(observation.restart_times)
        if count >= binding.spec.restart_limit:
            observation.next_retry_at = 0.0
            state = ObservedServiceState.CIRCUIT_OPEN
        else:
            delay = min(
                binding.spec.backoff_max_s,
                binding.spec.backoff_initial_s
                * (binding.spec.backoff_factor ** max(0, count - 1)),
            )
            observation.next_retry_at = now + delay
            state = ObservedServiceState.BACKING_OFF
        self._transition(observation, state, reason, error=error)
        return state

    async def _stop_binding(
        self,
        binding: _ServiceBinding,
        actions: list[dict[str, Any]],
    ) -> bool:
        observation = binding.observation
        self._transition(observation, ObservedServiceState.STOPPING, "stop_requested")
        try:
            await self._call(binding.stop, binding.spec.stop_timeout_s)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            self._transition(
                observation,
                ObservedServiceState.FAILED,
                "stop_failed",
                error=str(exc),
            )
            actions.append({"service": binding.spec.name, "action": "stop", "ok": False, "error": str(exc)})
            return False
        self._transition(observation, ObservedServiceState.STOPPED, "stopped")
        actions.append({"service": binding.spec.name, "action": "stop", "ok": True})
        return True

    async def _start_binding(self, binding: _ServiceBinding, actions: list[dict[str, Any]]) -> None:
        observation = binding.observation
        request = AdmissionRequest(
            owner=f"runtime_control_plane:{binding.spec.name}",
            work_class=binding.spec.admission_class,
            lane=binding.spec.name,
            priority=(
                AdmissionPriority.CRITICAL
                if binding.spec.critical
                else AdmissionPriority.MAINTENANCE
            ),
            timeout_s=min(2.0, binding.spec.start_timeout_s),
            lease_ttl_s=max(5.0, binding.spec.start_timeout_s + 2.0),
            receipt_required=True,
            metadata={"service": binding.spec.name, **dict(binding.spec.metadata)},
        )
        decision = await self.admission.acquire(request)
        observation.admission_receipt_id = decision.receipt_id
        if not decision.admitted:
            self._transition(
                observation,
                ObservedServiceState.BLOCKED,
                f"admission_{decision.reason}",
            )
            actions.append(
                {
                    "service": binding.spec.name,
                    "action": "start",
                    "ok": False,
                    "reason": decision.reason,
                    "admission_receipt_id": decision.receipt_id,
                }
            )
            return

        self._transition(observation, ObservedServiceState.STARTING, "start_admitted")
        try:
            await self._call(binding.start, binding.spec.start_timeout_s)
            if not await self._probe(binding):
                raise RuntimeError("service start returned without passing liveness probe")
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            start_error = str(exc)
            actions.append(
                {
                    "service": binding.spec.name,
                    "action": "start",
                    "ok": False,
                    "error": start_error,
                }
            )
            cleaned = await self._stop_binding(binding, actions)
            if cleaned:
                self._schedule_restart(
                    binding,
                    reason="start_failed",
                    error=start_error,
                )
            else:
                cleanup_error = observation.last_error
                self._transition(
                    observation,
                    ObservedServiceState.FAILED,
                    "start_cleanup_failed",
                    error=(
                        f"start failed: {start_error}; cleanup failed: {cleanup_error}"
                    ),
                )
        else:
            observation.generation += 1
            observation.next_retry_at = 0.0
            self._transition(observation, ObservedServiceState.READY, "probe_passed")
            actions.append(
                {
                    "service": binding.spec.name,
                    "action": "start",
                    "ok": True,
                    "generation": observation.generation,
                    "admission_receipt_id": decision.receipt_id,
                }
            )
        finally:
            try:
                await self.admission.release(decision.lease_id, reason="service_start_finished")
            except KeyError:
                logger.warning("service-start admission lease expired before release: %s", decision.lease_id)

    async def reconcile_once(self) -> dict[str, Any]:
        if not self._reconcile_lock.acquire(blocking=False):
            return {
                "schema": "aura.runtime_control_plane.reconcile.v1",
                "converged": False,
                "reason": "reconcile_already_running",
                "services": self.service_status(),
            }
        started = time.monotonic()
        actions: list[dict[str, Any]] = []
        try:
            with self._lock:
                order = self._topological_order_locked()
                bindings = {name: self._services[name] for name in order}

            for name in reversed(order):
                binding = bindings[name]
                if binding.observation.desired_state != DesiredServiceState.STOPPED:
                    continue
                if binding.observation.observed_state in {
                    ObservedServiceState.READY,
                    ObservedServiceState.DEGRADED,
                    ObservedServiceState.BLOCKED,
                    ObservedServiceState.BACKING_OFF,
                }:
                    await self._stop_binding(binding, actions)

            for name in order:
                binding = bindings[name]
                observation = binding.observation
                if observation.desired_state != DesiredServiceState.RUNNING:
                    continue
                blockers = [
                    dependency
                    for dependency in binding.spec.dependencies
                    if dependency not in bindings
                    or bindings[dependency].observation.observed_state
                    != ObservedServiceState.READY
                ]
                if blockers:
                    blocked_reason = "dependency_blocked:" + ",".join(sorted(blockers))
                    if observation.observed_state in {
                        ObservedServiceState.READY,
                        ObservedServiceState.DEGRADED,
                    }:
                        stopped = await self._stop_binding(binding, actions)
                        if not stopped:
                            continue
                    self._transition(observation, ObservedServiceState.BLOCKED, blocked_reason)
                    continue

                if observation.observed_state in {
                    ObservedServiceState.READY,
                    ObservedServiceState.DEGRADED,
                }:
                    try:
                        healthy = await self._probe(binding)
                    except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
                        healthy = False
                        observation.last_error = str(exc)
                    if healthy:
                        self._transition(observation, ObservedServiceState.READY, "probe_passed")
                        continue
                    if not binding.spec.restart_on_unhealthy:
                        self._transition(observation, ObservedServiceState.DEGRADED, "probe_failed")
                        continue
                    probe_error = observation.last_error or "liveness probe returned false"
                    restart_state = self._schedule_restart(
                        binding,
                        reason="probe_failed",
                        error=probe_error,
                    )
                    stopped = await self._stop_binding(binding, actions)
                    if not stopped:
                        continue
                    self._transition(
                        observation,
                        restart_state,
                        "probe_failed",
                        error=probe_error,
                    )

                now = time.time()
                if observation.observed_state in {
                    ObservedServiceState.CIRCUIT_OPEN,
                    ObservedServiceState.FAILED,
                }:
                    continue
                if observation.next_retry_at > now:
                    self._transition(observation, ObservedServiceState.BACKING_OFF, "restart_backoff")
                    continue
                if not self._restart_budget_available(binding, now):
                    self._transition(observation, ObservedServiceState.CIRCUIT_OPEN, "restart_budget_exhausted")
                    continue
                await self._start_binding(binding, actions)

            self._reconcile_count += 1
            services = self.service_status()
            critical_ready = all(
                not binding.spec.critical
                or binding.observation.observed_state == ObservedServiceState.READY
                for binding in bindings.values()
            )
            converged = all(
                (
                    binding.observation.desired_state == DesiredServiceState.RUNNING
                    and binding.observation.observed_state == ObservedServiceState.READY
                )
                or (
                    binding.observation.desired_state == DesiredServiceState.STOPPED
                    and binding.observation.observed_state == ObservedServiceState.STOPPED
                )
                for binding in bindings.values()
            )
            report = {
                "schema": "aura.runtime_control_plane.reconcile.v1",
                "reconcile_count": self._reconcile_count,
                "converged": converged,
                "critical_ready": critical_ready,
                "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                "actions": actions,
                "services": services,
                "admission": self.admission.status(),
            }
            self._last_report = report
            self._publish_conditions(report)
            return report
        finally:
            self._reconcile_lock.release()

    @staticmethod
    def _publish_conditions(report: Mapping[str, Any]) -> None:
        try:
            from core.runtime.conditions import (
                ConditionType,
                get_component_conditions,
            )

            services = report.get("services")
            service_map = services if isinstance(services, Mapping) else {}
            circuits = sorted(
                str(name)
                for name, status in service_map.items()
                if isinstance(status, Mapping)
                and status.get("observed_state")
                in {
                    ObservedServiceState.CIRCUIT_OPEN.value,
                    ObservedServiceState.FAILED.value,
                }
            )
            conditions = get_component_conditions("runtime_control_plane")
            critical_ready = bool(report.get("critical_ready", False))
            converged = bool(report.get("converged", False))
            conditions.set(
                ConditionType.READY,
                critical_ready,
                reason="CriticalServicesReady" if critical_ready else "CriticalServiceBlocked",
                message=f"managed_services={len(service_map)}",
            )
            conditions.set(
                ConditionType.PROGRESSING,
                not converged and not circuits,
                reason="Reconciling" if not converged and not circuits else "SteadyState",
                message=f"actions={len(report.get('actions') or [])}",
            )
            conditions.set(
                ConditionType.DEGRADED,
                bool(circuits),
                reason="CircuitOpen" if circuits else "NoOpenCircuits",
                message=",".join(circuits),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return

    def service_status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                name: {
                    **binding.observation.to_dict(),
                    "critical": binding.spec.critical,
                    "dependencies": list(binding.spec.dependencies),
                    "restart_limit": binding.spec.restart_limit,
                    "metadata": dict(binding.spec.metadata),
                }
                for name, binding in sorted(self._services.items())
            }

    def is_alive(self) -> bool:
        if self._closed:
            return False
        try:
            self.admission.pressure_snapshot()
            return True
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
            return False

    def is_ready(self) -> bool:
        if not self.is_alive():
            return False
        with self._lock:
            return all(
                not binding.spec.critical
                or binding.observation.observed_state == ObservedServiceState.READY
                for binding in self._services.values()
            )

    def get_status(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "ready": self.is_ready(),
            "closed": self._closed,
            "reconcile_count": self._reconcile_count,
            "services": self.service_status(),
            "admission": self.admission.status(),
            "last_report_digest": hashlib.sha256(
                json.dumps(self._last_report, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if self._last_report
            else "",
        }


_CONTROL_PLANE: RuntimeControlPlane | None = None
_CONTROL_PLANE_LOCK = threading.Lock()


def get_runtime_control_plane() -> RuntimeControlPlane:
    global _CONTROL_PLANE
    if _CONTROL_PLANE is None:
        with _CONTROL_PLANE_LOCK:
            if _CONTROL_PLANE is None:
                _CONTROL_PLANE = RuntimeControlPlane()
    return _CONTROL_PLANE


def reset_runtime_control_plane() -> None:
    global _CONTROL_PLANE
    with _CONTROL_PLANE_LOCK:
        _CONTROL_PLANE = None


async def reconcile_registered_runtime_control_plane() -> dict[str, Any]:
    """Advance the initialized control plane without constructing one implicitly."""
    from core.container import ServiceContainer

    control_plane = ServiceContainer.peek("runtime_control_plane", default=None)
    if control_plane is None:
        raise RuntimeError("runtime control plane is not registered")
    reconcile = getattr(control_plane, "reconcile_once", None)
    if not callable(reconcile):
        raise TypeError("registered runtime control plane lacks reconcile_once")
    report = reconcile()
    if inspect.isawaitable(report):
        report = await report
    if not isinstance(report, dict):
        raise TypeError("runtime control plane reconcile_once returned a non-dictionary report")
    return report


async def register_runtime_control_plane_reconciler(scheduler: Any) -> None:
    """Register the canonical desired-state heartbeat with Aura's scheduler."""
    from core.scheduler import TaskSpec

    register = getattr(scheduler, "register", None)
    if not callable(register):
        raise TypeError("runtime scheduler lacks register")
    await register(
        TaskSpec(
            name=CONTROL_PLANE_RECONCILE_TASK_NAME,
            coro=reconcile_registered_runtime_control_plane,
            tick_interval=CONTROL_PLANE_RECONCILE_INTERVAL_S,
            timeout_s=CONTROL_PLANE_RECONCILE_TIMEOUT_S,
            critical=True,
            priority=100,
            metadata={
                "owner": "core.runtime.control_plane",
                "contract": "desired_state_convergence",
            },
        )
    )


__all__ = [
    "AdmissionDecision",
    "AdmissionOutcome",
    "AdmissionPriority",
    "AdmissionRequest",
    "DesiredServiceSpec",
    "DesiredServiceState",
    "ObservedServiceState",
    "PressureSnapshot",
    "ResourceAdmissionController",
    "RuntimeControlPlane",
    "WorkClass",
    "CONTROL_PLANE_RECONCILE_INTERVAL_S",
    "CONTROL_PLANE_RECONCILE_TASK_NAME",
    "CONTROL_PLANE_RECONCILE_TIMEOUT_S",
    "get_runtime_control_plane",
    "reconcile_registered_runtime_control_plane",
    "register_runtime_control_plane_reconciler",
    "reset_runtime_control_plane",
]
