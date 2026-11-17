"""core/resource/resource_governor.py
=====================================
Unified resource governor for Aura production runtime.

Manages:
  - Thermal throttle integration with WorldState
  - Inference concurrency limiter (priority queue, limit=1)
  - Memory watchdog with tiered eviction
  - CPU pressure detection and adaptive throttling

All resource decisions are published to the MetricsCollector for
Prometheus observability.
"""
from __future__ import annotations

import enum
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.resource_observation import ResourceObserver, get_resource_observer

logger = logging.getLogger("Aura.Resource.Governor")


class ThermalState(enum.Enum):
    """macOS-aligned thermal pressure states."""
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"


class EvictionTier(enum.Enum):
    """Memory eviction escalation tiers."""
    NONE = "none"
    SOFT = "soft"        # Drop caches, trim histories
    MODERATE = "moderate" # Evict non-essential services
    AGGRESSIVE = "aggressive"  # Shed background tasks, force GC


@dataclass
class ResourceSnapshot:
    """Point-in-time resource state."""
    timestamp: float = field(default_factory=lambda: time.time())
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    thermal_state: ThermalState = ThermalState.NOMINAL
    thermal_pressure_raw: float = 0.0
    inference_queue_depth: int = 0
    inference_active: bool = False
    eviction_tier: EvictionTier = EvictionTier.NONE
    throttle_active: bool = False
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    observation_available: bool = False
    thermal_provider: str = "blind"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "memory_percent": round(self.memory_percent, 1),
            "memory_rss_mb": round(self.memory_rss_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "thermal_state": self.thermal_state.value,
            "thermal_pressure_raw": round(self.thermal_pressure_raw, 3),
            "inference_queue_depth": self.inference_queue_depth,
            "inference_active": self.inference_active,
            "eviction_tier": self.eviction_tier.value,
            "throttle_active": self.throttle_active,
            "observation_source": self.observation_source,
            "observation_scenario_id": self.observation_scenario_id,
            "observation_available": self.observation_available,
            "thermal_provider": self.thermal_provider,
        }


class InferenceSemaphore:
    """Compatibility API backed by canonical resource-admission leases.

    Production policy lives in ``RuntimeControlPlane.admission``. This class
    retains the historical async-acquire/sync-release shape for old callers
    and requests global inference scope so it preserves the old limit of one.
    """

    def __init__(self, max_concurrent: int = 1, *, admission: Any | None = None):
        if int(max_concurrent) != 1:
            raise ValueError(
                "InferenceSemaphore compatibility mode supports max_concurrent=1; "
                "use ResourceAdmissionController for lane-aware concurrency"
            )
        self._admission = admission
        self._queue_depth = 0
        self._active = False
        self._active_source: str = ""
        self._lease_ids: list[str] = []
        self._total_acquired: int = 0
        self._total_timeouts: int = 0
        self._total_wait_ms: float = 0.0

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def is_active(self) -> bool:
        return self._active

    async def acquire(
        self,
        source: str = "unknown",
        timeout: float = 120.0,  # noqa: ASYNC109 - compatibility API accepts a wait bound.
        priority: bool = False,
    ) -> bool:
        """Acquire inference slot.

        Args:
            source: Who's requesting (for logging).
            timeout: Max wait time in seconds.
            priority: If True, this is user-facing and should not be shed.

        Returns:
            True if acquired, False if timed out.
        """
        self._queue_depth += 1
        start = time.monotonic()
        try:
            from core.runtime.control_plane import (
                AdmissionPriority,
                AdmissionRequest,
                WorkClass,
                get_runtime_control_plane,
            )

            admission = self._admission or get_runtime_control_plane().admission
            decision = await admission.acquire(
                AdmissionRequest(
                    owner=f"resource_governor.inference:{source}",
                    work_class=WorkClass.INFERENCE,
                    lane="legacy_global_inference",
                    priority=(
                        AdmissionPriority.FOREGROUND
                        if priority
                        else AdmissionPriority.INTERACTIVE
                    ),
                    timeout_s=max(0.0, float(timeout)),
                    lease_ttl_s=max(900.0, float(timeout) + 600.0),
                    metadata={
                        "global_inference_scope": True,
                        "compatibility_facade": "InferenceSemaphore",
                    },
                )
            )
            if not decision.admitted:
                self._total_timeouts += 1
                logger.warning(
                    "InferenceSemaphore: %s denied after %.1fs (priority=%s, reason=%s)",
                    source,
                    timeout,
                    priority,
                    decision.reason,
                )
                return False
            self._admission = admission
            self._lease_ids.append(decision.lease_id)
            self._active = True
            self._active_source = source
            self._total_acquired += 1
            wait_ms = (time.monotonic() - start) * 1000
            self._total_wait_ms += wait_ms
            if wait_ms > 1000:
                logger.info(
                    "InferenceSemaphore: %s acquired after %.0fms wait (priority=%s)",
                    source, wait_ms, priority,
                )
            return True
        finally:
            self._queue_depth = max(0, self._queue_depth - 1)

    def release(self) -> None:
        """Release inference slot."""
        lease_id = self._lease_ids.pop() if self._lease_ids else ""
        if not lease_id or self._admission is None:
            logger.warning("InferenceSemaphore release called without an active lease")
            self._active = bool(self._lease_ids)
            if not self._active:
                self._active_source = ""
            return
        try:
            self._admission.release_sync(
                lease_id,
                reason="legacy_inference_finished",
            )
        except KeyError:
            logger.warning(
                "InferenceSemaphore lease expired before synchronous release: %s",
                lease_id,
            )
        self._active = False
        self._active_source = ""

    def get_stats(self) -> dict[str, Any]:
        avg_wait = (
            self._total_wait_ms / self._total_acquired
            if self._total_acquired > 0
            else 0.0
        )
        return {
            "active": self._active,
            "active_source": self._active_source,
            "queue_depth": self._queue_depth,
            "total_acquired": self._total_acquired,
            "total_timeouts": self._total_timeouts,
            "avg_wait_ms": round(avg_wait, 1),
        }


class ResourceGovernor:
    """Unified resource governor for Aura runtime.

    Periodically samples system resources, computes thermal/memory
    pressure, and provides throttling decisions to the metabolic
    coordinator and inference gate.
    """

    # Memory thresholds (percent of system memory)
    MEMORY_SOFT_THRESHOLD = 75.0
    MEMORY_MODERATE_THRESHOLD = 85.0
    MEMORY_AGGRESSIVE_THRESHOLD = 92.0

    # Thermal thresholds (mapped from macOS thermal pressure)
    THERMAL_THRESHOLDS = {
        ThermalState.NOMINAL: 0.0,
        ThermalState.FAIR: 0.3,
        ThermalState.SERIOUS: 0.6,
        ThermalState.CRITICAL: 0.85,
    }

    def __init__(self, *, observer: ResourceObserver | None = None) -> None:
        self._observer = observer
        self._inference_semaphore = InferenceSemaphore(max_concurrent=1)
        self._history: deque[ResourceSnapshot] = deque(maxlen=60)
        self._last_snapshot: ResourceSnapshot | None = None
        self._last_sample_time: float = 0.0
        self._sample_interval_s: float = 5.0
        self._eviction_callbacks: list[Callable[[EvictionTier], None]] = []
        self._eviction_callback_failures: dict[int, int] = {}
        self._throttle_active: bool = False
        self._consecutive_pressure_samples: int = 0
        # The tier last reported, so a run of evictions at one tier is reported
        # once. A sample that finds no pressure clears it.
        self._tier_reported: EvictionTier | None = None

    @property
    def inference(self) -> InferenceSemaphore:
        return self._inference_semaphore

    def register_eviction_callback(
        self,
        callback: Callable[[EvictionTier], None],
    ) -> None:
        """Register a callback for memory eviction events.

        Callback signature: callback(tier: EvictionTier) -> None
        """
        self._eviction_callbacks.append(callback)

    def sample(self) -> ResourceSnapshot:
        """Take a point-in-time resource snapshot."""
        snap = ResourceSnapshot()
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        snap.observation_source = provenance.source.value
        snap.observation_scenario_id = provenance.scenario_id

        # Memory
        try:
            memory = observer.memory()
            snap.memory_percent = float(memory.percent) if memory.available else 100.0
            snap.memory_rss_mb = float(memory.process_rss_bytes) / float(1024**2)
            snap.observation_available = bool(memory.available)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            snap.memory_percent = 100.0

        # CPU
        try:
            compute = observer.compute()
            snap.cpu_percent = float(compute.cpu_percent)
            snap.observation_available = snap.observation_available and compute.cpu_percent_readable
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.resource.resource_governor: %s", type(_exc).__name__, _exc)

        # Thermal
        try:
            thermal = observer.thermal()
            snap.thermal_state, snap.thermal_pressure_raw = (
                self._thermal_state_from_observation(thermal)
            )
            snap.thermal_provider = thermal.provider
            snap.observation_available = snap.observation_available and thermal.available
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Thermal resource observation failed: %s", exc)
            snap.thermal_state = ThermalState.CRITICAL
            snap.thermal_pressure_raw = 1.0
            snap.thermal_provider = "unavailable"
            snap.observation_available = False

        # Inference
        snap.inference_queue_depth = self._inference_semaphore.queue_depth
        snap.inference_active = self._inference_semaphore.is_active

        # Compute eviction tier
        snap.eviction_tier = self._compute_eviction_tier(snap)
        if snap.eviction_tier == EvictionTier.NONE:
            self._tier_reported = None

        # Throttle decision
        snap.throttle_active = self._should_throttle(snap)
        self._throttle_active = snap.throttle_active

        self._last_snapshot = snap
        self._history.append(snap)
        self._last_sample_time = time.time()

        # Record to metrics
        try:
            from core.observability.metrics import get_metrics
            m = get_metrics()
            m.set_gauge("thermal_pressure", snap.thermal_pressure_raw)
            m.set_gauge("eviction_tier", float(
                [EvictionTier.NONE, EvictionTier.SOFT,
                 EvictionTier.MODERATE, EvictionTier.AGGRESSIVE].index(snap.eviction_tier)
            ))
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.resource.resource_governor: %s", type(_exc).__name__, _exc)

        return snap

    def _read_thermal_state(self) -> tuple[ThermalState, float]:
        """Map the injected canonical thermal observation into governor state."""
        try:
            reading = (self._observer or get_resource_observer()).thermal()
            return self._thermal_state_from_observation(reading)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return ThermalState.CRITICAL, 1.0

    def _thermal_state_from_observation(self, reading: Any) -> tuple[ThermalState, float]:
        if not reading.available:
            return ThermalState.CRITICAL, 1.0
        state = {
            0: ThermalState.NOMINAL,
            1: ThermalState.FAIR,
            2: ThermalState.SERIOUS,
            3: ThermalState.CRITICAL,
        }.get(max(0, min(3, int(reading.level))), ThermalState.CRITICAL)
        return state, self.THERMAL_THRESHOLDS[state]

    def _compute_eviction_tier(self, snap: ResourceSnapshot) -> EvictionTier:
        """Determine memory eviction tier from current snapshot."""
        mem = snap.memory_percent

        if mem >= self.MEMORY_AGGRESSIVE_THRESHOLD:
            return EvictionTier.AGGRESSIVE
        elif mem >= self.MEMORY_MODERATE_THRESHOLD:
            return EvictionTier.MODERATE
        elif mem >= self.MEMORY_SOFT_THRESHOLD:
            return EvictionTier.SOFT
        return EvictionTier.NONE

    def _should_throttle(self, snap: ResourceSnapshot) -> bool:
        """Determine if background processing should be throttled."""
        if snap.thermal_state in (ThermalState.SERIOUS, ThermalState.CRITICAL):
            self._consecutive_pressure_samples += 1
            return True
        if snap.eviction_tier in (EvictionTier.MODERATE, EvictionTier.AGGRESSIVE):
            self._consecutive_pressure_samples += 1
            return True
        self._consecutive_pressure_samples = 0
        return False

    def _threshold_for(self, tier: EvictionTier) -> float:
        """The memory usage this tier fires at, so a shed has a line to get
        back under rather than a number somebody chose."""
        return {
            EvictionTier.SOFT: self.MEMORY_SOFT_THRESHOLD,
            EvictionTier.MODERATE: self.MEMORY_MODERATE_THRESHOLD,
            EvictionTier.AGGRESSIVE: self.MEMORY_AGGRESSIVE_THRESHOLD,
        }.get(tier, 100.0)

    def _shed_the_order(self, tier: EvictionTier) -> int:
        """Ask the registered organs to free memory. Returns how many shed."""
        if tier == EvictionTier.NONE:
            return 0
        try:
            from core.runtime.oom_policy import get_oom_policy
            from core.runtime.resource_observation import get_resource_observer
        except ImportError as exc:
            record_degradation(
                "resource_governor", exc, severity="debug",
                action="skipped the shed order; the OOM policy is unavailable",
            )
            return 0
        observer = self._observer or get_resource_observer()

        def free_bytes_now() -> int:
            reading = observer.memory()
            return int(getattr(reading, "available_bytes", 0) or 0)

        try:
            reading = observer.memory()
            total = int(getattr(reading, "total_bytes", 0) or 0)
            if total <= 0:
                return 0
            head_room = 1.0 - (self._threshold_for(tier) / 100.0)
            steps = [
                EvictionTier.SOFT, EvictionTier.MODERATE, EvictionTier.AGGRESSIVE,
            ]
            events = get_oom_policy().shed_until(
                target_free_bytes=int(total * head_room),
                free_bytes_now=free_bytes_now,
                reason=f"resource_governor:{tier.value}",
                max_victims=steps.index(tier) + 1 if tier in steps else 1,
            )
            return len(events)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "resource_governor", exc, severity="warning",
                action="continued after the shed order failed",
                extra={"tier": tier.value},
            )
            return 0

    def execute_eviction(self, tier: EvictionTier) -> int:
        """Execute memory eviction at the specified tier.

        Returns number of callbacks invoked.
        """
        if tier == EvictionTier.NONE:
            return 0

        invoked = 0
        for cb in self._eviction_callbacks:
            callback_id = id(cb)
            if self._eviction_callback_failures.get(callback_id, 0) >= 3:
                logger.debug("ResourceGovernor: skipping quarantined eviction callback %r", cb)
                continue
            try:
                cb(tier)
                invoked += 1
                self._eviction_callback_failures.pop(callback_id, None)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                failure_count = self._eviction_callback_failures.get(callback_id, 0) + 1
                self._eviction_callback_failures[callback_id] = failure_count
                action = (
                    "quarantined failing eviction callback and continued eviction"
                    if failure_count >= 3
                    else "continued eviction after callback failed"
                )
                record_degradation(
                    "resource_governor",
                    exc,
                    severity="warning",
                    action=action,
                    extra={
                        "tier": tier.value,
                        "callback": repr(cb),
                        "failure_count": failure_count,
                    },
                )
                logger.debug("Eviction callback failed: %s", exc)

        # Force garbage collection at moderate+ tiers
        if tier in (EvictionTier.MODERATE, EvictionTier.AGGRESSIVE):
            try:
                import gc
                collected = gc.collect(1)
                logger.info(
                    "ResourceGovernor: GC collected %d objects (tier=%s)",
                    collected, tier.value,
                )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                logger.debug("Suppressed %s in core.resource.resource_governor: %s", type(_exc).__name__, _exc)

        # And the shed order that exists, which this never asked.
        #
        # `register_eviction_callback` had no caller anywhere in the runtime, so
        # every eviction ran an empty list and reported "callbacks=0" at
        # warning — a mechanism that announced itself while freeing nothing.
        # LIVE 2026-09-28 00:43, mid-page-run at 92% resource pressure: "
        # Eviction tier=moderate, callbacks=0" then "tier=soft, callbacks=0".
        #
        # The organs that CAN free memory are registered with the OOM policy by
        # the container, ranked by badness, each with its own shed. The target
        # comes from the line that fired: shed until usage is back under this
        # tier's own threshold, and stop as soon as it is. The victim cap is the
        # escalation itself — one at the first tier, one more at each step.
        shed = self._shed_the_order(tier)
        logger.log(
            logging.WARNING if tier != getattr(self, "_tier_reported", None) else logging.INFO,
            "ResourceGovernor: Eviction tier=%s, callbacks=%d, shed=%d",
            tier.value, invoked, shed,
        )
        self._tier_reported = tier

        # Report to incident manager at aggressive tier
        if tier == EvictionTier.AGGRESSIVE:
            try:
                from core.resilience.incident_manager import get_incident_manager
                get_incident_manager().report(
                    source="resource_governor",
                    title=f"Memory eviction: {tier.value}",
                    detail=f"mem={self._last_snapshot.memory_percent:.1f}%"
                    if self._last_snapshot else "unknown",
                    severity="critical",
                )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                logger.debug("Suppressed %s in core.resource.resource_governor: %s", type(_exc).__name__, _exc)

        return invoked

    def get_snapshot(self) -> ResourceSnapshot | None:
        """Get the latest snapshot (may sample if stale)."""
        now = time.time()
        if (
            self._last_snapshot is None
            or now - self._last_sample_time > self._sample_interval_s
        ):
            return self.sample()
        return self._last_snapshot

    def is_throttled(self) -> bool:
        """Quick check: should background work be throttled?"""
        return self._throttle_active

    def is_alive(self) -> bool:
        """The sampler is live when it can produce a current snapshot."""
        try:
            return self.get_snapshot() is not None
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "no snapshot, so the resource sampler counts as not alive (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def get_status(self) -> dict[str, Any]:
        """Return full resource status for observability."""
        snap = self._last_snapshot
        return {
            "snapshot": snap.to_dict() if snap else None,
            "throttle_active": self._throttle_active,
            "consecutive_pressure_samples": self._consecutive_pressure_samples,
            "inference": self._inference_semaphore.get_stats(),
            "history_length": len(self._history),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: ResourceGovernor | None = None


def get_resource_governor() -> ResourceGovernor:
    """Get the singleton ResourceGovernor instance."""
    global _instance
    if _instance is None:
        _instance = ResourceGovernor()
    return _instance


def reset_resource_governor_for_test() -> None:
    global _instance
    _instance = None
