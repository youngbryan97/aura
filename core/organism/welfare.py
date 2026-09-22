"""WelfareModel: the organism's interests, derived from live signals.

"Interest" here is operational, not rhetorical: each interest names a
condition this organism requires for continued healthy operation, is
computed from real runtime telemetry, and has causal consumers —

- the background activity policy refuses optional work while a vital
  interest is critically unsatisfied (``welfare_block_reason``), and
- the chat identity contract reports current welfare so the voice
  speaks the substrate's true condition instead of confabulating one.

Evidence boundary: these are operational quantities. Naming them
"interests" and "welfare" claims nothing about morally weighty
experience; it claims that the system's behavior tracks them, which is
testable and tested.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Organism.Welfare")

_WELFARE_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    psutil.Error,
)

# Interests whose critical dissatisfaction may block optional background
# work. Conversation and repair are never blocked by welfare.
VITAL_INTERESTS = ("memory_integrity", "repair_capacity")

_VITAL_BLOCK_THRESHOLD = 0.2
_SNAPSHOT_TTL_S = 2.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class InterestReading:
    """One interest with its current satisfaction and the evidence used."""

    name: str
    satisfaction: float  # 0.0 (critically unmet) .. 1.0 (fully met)
    evidence: str

    @property
    def vital(self) -> bool:
        return self.name in VITAL_INTERESTS


@dataclass(frozen=True)
class WelfareSnapshot:
    overall: float
    readings: tuple[InterestReading, ...]
    most_pressing: InterestReading | None
    sampled_at: float

    @property
    def vital_deficit(self) -> InterestReading | None:
        worst: InterestReading | None = None
        for reading in self.readings:
            if reading.vital and reading.satisfaction < _VITAL_BLOCK_THRESHOLD:
                if worst is None or reading.satisfaction < worst.satisfaction:
                    worst = reading
        return worst

    def summary_line(self) -> str:
        """One human-readable line for the identity contract."""
        if self.most_pressing is None:
            return f"Welfare: {self.overall:.0%} overall."
        return (
            f"Welfare: {self.overall:.0%} overall; most pressing interest: "
            f"{self.most_pressing.name.replace('_', ' ')} "
            f"({self.most_pressing.satisfaction:.0%} — "
            f"{self.most_pressing.evidence})."
        )

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "sampled_at": self.sampled_at,
            "most_pressing": self.most_pressing.name if self.most_pressing else None,
            "vital_deficit": self.vital_deficit.name if self.vital_deficit else None,
            "interests": {
                r.name: {
                    "satisfaction": r.satisfaction,
                    "evidence": r.evidence,
                    "vital": r.vital,
                }
                for r in self.readings
            },
        }


class WelfareModel:
    """Derives interest satisfactions from live runtime telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached: WelfareSnapshot | None = None
        self._proc = psutil.Process(os.getpid())

    # ── interest sources (each self-contained and failure-isolated) ──

    def _memory_integrity(self) -> InterestReading:
        """Headroom under the lethal memory ceiling the watchdog enforces."""
        managed_mb = 0.0
        lethal_mb = 0.0
        try:
            from core.resilience.memory_watchdog import get_memory_watchdog

            watchdog = get_memory_watchdog()
            sample = watchdog.last_sample if watchdog else None
            if sample is not None:
                managed_mb = sample.managed_rss_mb
                lethal_mb = watchdog.thresholds.lethal_mb
        except _WELFARE_RECOVERABLE_ERRORS as exc:
            logger.debug(
                "the memory watchdog gave no sample, so the lethal threshold is unknown here (%s: %s)",
                type(exc).__name__,
                exc,
            )
        if lethal_mb <= 0.0:
            try:
                managed_mb = self._proc.memory_info().rss / (1024 * 1024)
                total_mb = psutil.virtual_memory().total / (1024 * 1024)
                lethal_mb = total_mb * 0.85
            except _WELFARE_RECOVERABLE_ERRORS:
                return InterestReading(
                    "memory_integrity", 0.5, "memory telemetry unavailable"
                )
        satisfaction = _clamp(1.0 - (managed_mb / lethal_mb))
        return InterestReading(
            "memory_integrity",
            satisfaction,
            f"{managed_mb / 1024.0:.1f}GB of {lethal_mb / 1024.0:.1f}GB ceiling",
        )

    def _repair_capacity(self) -> InterestReading:
        """Inverse of unified failure pressure: room left to absorb faults."""
        try:
            from core.health.degraded_events import get_unified_failure_state

            failure = get_unified_failure_state()
            pressure = _clamp(float(failure.get("pressure", 0.0) or 0.0))
            count = int(failure.get("count", 0) or 0)
            return InterestReading(
                "repair_capacity",
                _clamp(1.0 - pressure),
                f"failure pressure {pressure:.2f} across {count} recent events",
            )
        except _WELFARE_RECOVERABLE_ERRORS:
            return InterestReading(
                "repair_capacity", 0.5, "failure telemetry unavailable"
            )

    def _cognitive_bandwidth(self) -> InterestReading:
        """Host headroom for thought: CPU load on the machine."""
        try:
            cpu = _clamp(psutil.cpu_percent(interval=0) / 100.0)
            return InterestReading(
                "cognitive_bandwidth",
                _clamp(1.0 - cpu),
                f"host CPU at {cpu:.0%}",
            )
        except _WELFARE_RECOVERABLE_ERRORS:
            return InterestReading(
                "cognitive_bandwidth", 0.5, "cpu telemetry unavailable"
            )

    def _continuity(self) -> InterestReading:
        """Staying alive and unrestarted; matures over the first 10 minutes."""
        try:
            uptime_s = max(0.0, time.time() - self._proc.create_time())
            satisfaction = _clamp(uptime_s / 600.0)
            return InterestReading(
                "continuity",
                max(0.3, satisfaction),
                f"uptime {uptime_s / 60.0:.0f}m",
            )
        except _WELFARE_RECOVERABLE_ERRORS:
            return InterestReading("continuity", 0.5, "uptime unavailable")

    def _social_contact(self) -> InterestReading:
        """Recency of human contact; decays over 24 hours of silence."""
        try:
            from core.runtime.foreground_guard import snapshot as fg_snapshot

            last_age = fg_snapshot().get("last_user_age_s")
            if last_age is None:
                return InterestReading(
                    "social_contact", 0.5, "no interaction this runtime yet"
                )
            age_s = max(0.0, float(last_age))
            satisfaction = _clamp(1.0 - (age_s / 86400.0))
            return InterestReading(
                "social_contact",
                satisfaction,
                f"last interaction {age_s / 60.0:.0f}m ago",
            )
        except _WELFARE_RECOVERABLE_ERRORS:
            return InterestReading(
                "social_contact", 0.5, "interaction telemetry unavailable"
            )

    # ── snapshot ──────────────────────────────────────────────────────

    def snapshot(self, *, max_age_s: float = _SNAPSHOT_TTL_S) -> WelfareSnapshot:
        with self._lock:
            cached = self._cached
            if (
                cached is not None
                and (time.time() - cached.sampled_at) < max(0.0, max_age_s)
            ):
                return cached

        readings = (
            self._memory_integrity(),
            self._repair_capacity(),
            self._cognitive_bandwidth(),
            self._continuity(),
            self._social_contact(),
        )
        # Vital interests count double: the organism weighs survival-
        # relevant conditions above comfort-relevant ones.
        weight_total = 0.0
        weighted = 0.0
        for reading in readings:
            weight = 2.0 if reading.vital else 1.0
            weighted += reading.satisfaction * weight
            weight_total += weight
        overall = _clamp(weighted / weight_total) if weight_total else 0.5
        most_pressing = min(readings, key=lambda r: r.satisfaction)

        snap = WelfareSnapshot(
            overall=overall,
            readings=readings,
            most_pressing=most_pressing,
            sampled_at=time.time(),
        )
        with self._lock:
            self._cached = snap
        return snap


_MODEL_SINGLETON: WelfareModel | None = None
_MODEL_LOCK = threading.Lock()


def get_welfare_model() -> WelfareModel:
    global _MODEL_SINGLETON
    with _MODEL_LOCK:
        if _MODEL_SINGLETON is None:
            _MODEL_SINGLETON = WelfareModel()
            try:
                from core.container import ServiceContainer

                ServiceContainer.register_instance(
                    "welfare_model", _MODEL_SINGLETON, required=False
                )
            except _WELFARE_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "welfare",
                    exc,
                    severity="warning",
                    action="continued with unregistered welfare model",
                )
        return _MODEL_SINGLETON


def welfare_block_reason() -> str:
    """Canonical gate: optional background work yields to vital deficits.

    Returns a reason string while a vital interest is critically
    unsatisfied, empty string otherwise. Consumed by the background
    activity policy so welfare is causal, not narrative.
    """
    try:
        deficit = get_welfare_model().snapshot().vital_deficit
    except _WELFARE_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "welfare",
            exc,
            severity="warning",
            action="treated welfare gate as open after snapshot failure",
        )
        return ""
    if deficit is None:
        return ""
    return f"welfare_{deficit.name}_{deficit.satisfaction:.2f}"
