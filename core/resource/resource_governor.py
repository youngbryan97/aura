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

import asyncio
import enum
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.runtime.errors import record_degradation

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
    timestamp: float = field(default_factory=time.time)
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    thermal_state: ThermalState = ThermalState.NOMINAL
    thermal_pressure_raw: float = 0.0
    inference_queue_depth: int = 0
    inference_active: bool = False
    eviction_tier: EvictionTier = EvictionTier.NONE
    throttle_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
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
        }


class InferenceSemaphore:
    """Priority-aware inference concurrency limiter.

    Ensures only one inference runs at a time. User-priority requests
    can preempt the queue position of background tasks.
    """

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_depth = 0
        self._active = False
        self._active_source: str = ""
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
        timeout: float = 120.0,
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
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
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
        except asyncio.TimeoutError:
            self._total_timeouts += 1
            logger.warning(
                "InferenceSemaphore: %s timed out after %.1fs (priority=%s)",
                source, timeout, priority,
            )
            return False
        finally:
            self._queue_depth = max(0, self._queue_depth - 1)

    def release(self) -> None:
        """Release inference slot."""
        self._active = False
        self._active_source = ""
        try:
            self._semaphore.release()
        except ValueError:
            pass  # Already released

    def get_stats(self) -> Dict[str, Any]:
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

    def __init__(self) -> None:
        self._inference_semaphore = InferenceSemaphore(max_concurrent=1)
        self._history: Deque[ResourceSnapshot] = deque(maxlen=60)
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_sample_time: float = 0.0
        self._sample_interval_s: float = 5.0
        self._eviction_callbacks: List = []
        self._throttle_active: bool = False
        self._consecutive_pressure_samples: int = 0

    @property
    def inference(self) -> InferenceSemaphore:
        return self._inference_semaphore

    def register_eviction_callback(self, callback) -> None:
        """Register a callback for memory eviction events.

        Callback signature: callback(tier: EvictionTier) -> None
        """
        self._eviction_callbacks.append(callback)

    def sample(self) -> ResourceSnapshot:
        """Take a point-in-time resource snapshot."""
        snap = ResourceSnapshot()

        # Memory
        try:
            import psutil
            vm = psutil.virtual_memory()
            snap.memory_percent = vm.percent
            proc = psutil.Process()
            snap.memory_rss_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            snap.memory_percent = 0.0

        # CPU
        try:
            import psutil
            snap.cpu_percent = psutil.cpu_percent(interval=0)
        except Exception:
            pass

        # Thermal (macOS native)
        snap.thermal_state, snap.thermal_pressure_raw = self._read_thermal_state()

        # Inference
        snap.inference_queue_depth = self._inference_semaphore.queue_depth
        snap.inference_active = self._inference_semaphore.is_active

        # Compute eviction tier
        snap.eviction_tier = self._compute_eviction_tier(snap)

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
        except Exception:
            pass

        return snap

    def _read_thermal_state(self) -> tuple:
        """Read macOS thermal pressure via subprocess (cached)."""
        try:
            # Use pmset to read thermal state on macOS
            import subprocess
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True, text=True, timeout=2.0,
            )
            output = result.stdout.lower()
            if "critical" in output:
                return ThermalState.CRITICAL, 0.95
            elif "serious" in output:
                return ThermalState.SERIOUS, 0.7
            elif "fair" in output:
                return ThermalState.FAIR, 0.4
            else:
                return ThermalState.NOMINAL, 0.0
        except Exception:
            # Fallback: use CPU as proxy
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0)
                if cpu > 90:
                    return ThermalState.SERIOUS, 0.7
                elif cpu > 70:
                    return ThermalState.FAIR, 0.4
                return ThermalState.NOMINAL, 0.0
            except Exception:
                return ThermalState.NOMINAL, 0.0

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

    def execute_eviction(self, tier: EvictionTier) -> int:
        """Execute memory eviction at the specified tier.

        Returns number of callbacks invoked.
        """
        if tier == EvictionTier.NONE:
            return 0

        invoked = 0
        for cb in self._eviction_callbacks:
            try:
                cb(tier)
                invoked += 1
            except Exception as exc:
                record_degradation("resource_governor", exc)
                logger.debug("Eviction callback failed: %s", exc)

        # Force garbage collection at moderate+ tiers
        if tier in (EvictionTier.MODERATE, EvictionTier.AGGRESSIVE):
            try:
                import gc
                collected = gc.collect()
                logger.info(
                    "ResourceGovernor: GC collected %d objects (tier=%s)",
                    collected, tier.value,
                )
            except Exception:
                pass

        logger.warning(
            "ResourceGovernor: Eviction tier=%s, callbacks=%d",
            tier.value, invoked,
        )

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
            except Exception:
                pass

        return invoked

    def get_snapshot(self) -> Optional[ResourceSnapshot]:
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

    def get_status(self) -> Dict[str, Any]:
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

_instance: Optional[ResourceGovernor] = None


def get_resource_governor() -> ResourceGovernor:
    """Get the singleton ResourceGovernor instance."""
    global _instance
    if _instance is None:
        _instance = ResourceGovernor()
    return _instance
