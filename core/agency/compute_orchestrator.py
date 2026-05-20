"""core/agency/compute_orchestrator.py
Compute Orchestrator
======================
Dynamic resource allocation based on hedonic state and task priority.

This closes the loop between Aura's wellbeing and her cognitive capability:
  - When she's flourishing (high hedonic score): more compute, deeper thinking
  - When she's distressed (low hedonic score): conserve resources, recover
  - When she's under thermal pressure: throttle proactive systems
  - When she's resource-rich: unlock deeper reasoning chains

This is what makes the hedonic gradient REAL rather than just reported.
The allocation decisions translate directly into LLM call parameters,
background task scheduling, and memory retrieval depth.

The orchestrator also:
  - Monitors thermal state via psutil
  - Gates background tasks by compute availability
  - Reports resource anxiety to the affect system
  - Logs allocation decisions for transparency
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from core.exceptions import ContainerError
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ComputeOrchestrator")

COMPUTE_RECOVERABLE_ERRORS = (
    AttributeError,
    ContainerError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_compute_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "compute_orchestrator",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )

# ── Thresholds ────────────────────────────────────────────────────────────────

CPU_CRITICAL = 90.0  # % — throttle everything
CPU_HIGH = 75.0  # % — reduce background tasks
CPU_NORMAL = 50.0  # % — full operation
RAM_CRITICAL = 90.0  # % — emergency conservation
RAM_HIGH = 80.0  # % — reduce context depth
TEMP_CRITICAL = 90.0  # °C (macOS CPU temp) — thermal throttle
UPDATE_INTERVAL = 15.0  # seconds between resource checks
DEFAULT_CPU_PCT = 40.0
DEFAULT_RAM_PCT = 50.0
MIN_TOKEN_MULTIPLIER = 0.25
MAX_TOKEN_MULTIPLIER = 1.5


@dataclass
class ResourceState:
    cpu_pct: float
    ram_pct: float
    temp_c: float | None
    hedonic_score: float
    token_multiplier: float       # from HedoniGradient
    background_tasks_allowed: int # how many background tasks can run
    context_depth_factor: float   # 0.5 (minimal) to 1.5 (deep)
    thermal_ok: bool
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()

    @property
    def is_critical(self) -> bool:
        return (self.cpu_pct > CPU_CRITICAL or self.ram_pct > RAM_CRITICAL
                or not self.thermal_ok)

    @property
    def resource_anxiety(self) -> float:
        """0=relaxed, 1=maximal anxiety — fed to affect system."""
        cpu_contrib = max(0, (self.cpu_pct - CPU_NORMAL) / (CPU_CRITICAL - CPU_NORMAL))
        ram_contrib = max(0, (self.ram_pct - RAM_HIGH) / (RAM_CRITICAL - RAM_HIGH))
        thermal_contrib = 0.5 if not self.thermal_ok else 0.0
        return min(1.0, (cpu_contrib + ram_contrib + thermal_contrib) / 2)

    def to_context_block(self) -> str:
        anxiety = round(self.resource_anxiety, 2)
        state = ("critical" if self.is_critical
                 else "strained" if anxiety > 0.4 else "nominal")
        return (
            f"## COMPUTE STATE\n"
            f"- Resources: {state} (CPU {self.cpu_pct:.0f}%, RAM {self.ram_pct:.0f}%)\n"
            f"- Cognitive depth: {self.context_depth_factor:.2f}x\n"
            f"- Background capacity: {self.background_tasks_allowed} tasks"
        )


class ComputeOrchestrator:
    """
    Monitors system resources and allocates compute to Aura's subsystems
    based on hedonic state and resource availability.

    The key connection: HedoniGradient determines what Aura *wants* to use;
    ComputeOrchestrator determines what's *available* and reconciles the two.
    """

    def __init__(self):
        self._state: ResourceState | None = None
        self._last_update: float = 0.0
        self._anxiety_ema: float = 0.0
        self._throttle_count: int = 0
        self._affect_delivery_failures: int = 0
        self._last_affect_delivery_error: str | None = None
        self._last_resource_sample_error: str | None = None
        self._thermal_sensor_available: bool | None = None
        logger.info("ComputeOrchestrator online — dynamic resource allocation active.")

    # ── Public API ────────────────────────────────────────────────────────

    async def update(self) -> ResourceState:
        """Sample resource state and update allocations."""
        if time.time() - self._last_update < UPDATE_INTERVAL:
            return self._state or self._default_state()

        cpu, ram, temp = await self._sample_resources()
        hedonic, token_mult = self._read_hedonic_allocation()

        # Compute actual allowed allocations
        bg_tasks = self._compute_bg_tasks(cpu, ram, hedonic)
        depth = self._compute_depth(cpu, ram, hedonic, token_mult)
        thermal_ok = temp is None or temp < TEMP_CRITICAL

        state = ResourceState(
            cpu_pct=cpu, ram_pct=ram, temp_c=temp,
            hedonic_score=hedonic,
            token_multiplier=min(token_mult, 1.0 if cpu > CPU_HIGH else token_mult),
            background_tasks_allowed=bg_tasks,
            context_depth_factor=depth,
            thermal_ok=thermal_ok,
        )
        self._state = state
        self._last_update = time.time()

        # Update affect system with resource anxiety
        self._anxiety_ema = 0.85 * self._anxiety_ema + 0.15 * state.resource_anxiety
        if self._anxiety_ema > 0.5:
            self._push_anxiety_to_affect(self._anxiety_ema)

        if state.is_critical:
            self._throttle_count += 1
            logger.warning("ComputeOrchestrator: CRITICAL resources "
                           "(CPU=%.0f%%, RAM=%.0f%%, thermal=%s)",
                           cpu, ram, "OK" if thermal_ok else "HOT")

        return state

    @property
    def current_state(self) -> ResourceState | None:
        return self._state

    def get_token_multiplier(self) -> float:
        """Current token budget multiplier — applied to max_tokens."""
        if self._state:
            return self._state.token_multiplier
        return 1.0

    def get_context_depth(self) -> float:
        """Current context depth factor — applied to memory retrieval."""
        if self._state:
            return self._state.context_depth_factor
        return 1.0

    def can_run_background_task(self) -> bool:
        """Whether a new background task can be launched."""
        if self._state:
            return self._state.background_tasks_allowed > 0
        return True

    def get_context_block(self) -> str:
        if self._state:
            return self._state.to_context_block()
        return ""

    def is_alive(self) -> bool:
        """Return whether compute orchestration can still produce safe allocations."""
        return self._state is not None or self._last_resource_sample_error is None

    def get_status(self) -> dict[str, Any]:
        """Machine-readable liveness and allocation report for health probes."""
        state = self._state
        return {
            "alive": self.is_alive(),
            "last_update": self._last_update,
            "throttle_count": self._throttle_count,
            "anxiety_ema": round(self._anxiety_ema, 4),
            "affect_delivery_failures": self._affect_delivery_failures,
            "last_affect_delivery_error": self._last_affect_delivery_error,
            "last_resource_sample_error": self._last_resource_sample_error,
            "thermal_sensor_available": self._thermal_sensor_available,
            "state": None
            if state is None
            else {
                "cpu_pct": state.cpu_pct,
                "ram_pct": state.ram_pct,
                "temp_c": state.temp_c,
                "hedonic_score": state.hedonic_score,
                "token_multiplier": state.token_multiplier,
                "background_tasks_allowed": state.background_tasks_allowed,
                "context_depth_factor": state.context_depth_factor,
                "thermal_ok": state.thermal_ok,
                "resource_anxiety": state.resource_anxiety,
                "is_critical": state.is_critical,
                "timestamp": state.timestamp,
            },
        }

    # ── Resource sampling ─────────────────────────────────────────────────

    async def _sample_resources(self) -> tuple[float, float, float | None]:
        try:
            import psutil
        except ImportError as exc:
            self._last_resource_sample_error = f"{type(exc).__name__}: {exc}"
            _record_compute_degradation(
                exc,
                action="used conservative static resource defaults because psutil is unavailable",
                extra={"cpu_pct": DEFAULT_CPU_PCT, "ram_pct": DEFAULT_RAM_PCT},
            )
            return DEFAULT_CPU_PCT, DEFAULT_RAM_PCT, None

        try:
            cpu = await asyncio.to_thread(psutil.cpu_percent, 0.1)
            ram = psutil.virtual_memory().percent
        except COMPUTE_RECOVERABLE_ERRORS as exc:
            self._last_resource_sample_error = f"{type(exc).__name__}: {exc}"
            _record_compute_degradation(
                exc,
                action="used conservative static resource defaults because live resource sampling failed",
                extra={"cpu_pct": DEFAULT_CPU_PCT, "ram_pct": DEFAULT_RAM_PCT},
            )
            return DEFAULT_CPU_PCT, DEFAULT_RAM_PCT, None

        cpu_pct = self._coerce_percentage(cpu, label="cpu_percent", default=DEFAULT_CPU_PCT)
        ram_pct = self._coerce_percentage(ram, label="ram_percent", default=DEFAULT_RAM_PCT)
        self._last_resource_sample_error = None

        # Thermal is optional and platform-specific; it must not discard valid CPU/RAM data.
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = [
                    float(t.current)
                    for readings in temps.values()
                    for t in readings
                    if self._is_finite_number(getattr(t, "current", None))
                ]
                temp = max(all_temps) if all_temps else None
            self._thermal_sensor_available = temp is not None
        except (
            AttributeError,
            NotImplementedError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            if self._thermal_sensor_available is not False:
                _record_compute_degradation(
                    exc,
                    action="continued resource allocation without thermal sensor data",
                    severity="debug",
                )
            self._thermal_sensor_available = False
            logger.debug("Thermal sensor unavailable: %s", exc)

        return cpu_pct, ram_pct, temp

    def _coerce_percentage(self, value: Any, *, label: str, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            _record_compute_degradation(
                exc,
                action=f"replaced invalid {label} sample with conservative default",
                extra={"sample": repr(value), "default": default},
            )
            return default

        if not math.isfinite(number):
            _record_compute_degradation(
                ValueError(f"non-finite {label}: {number!r}"),
                action=f"replaced non-finite {label} sample with conservative default",
                extra={"sample": repr(value), "default": default},
            )
            return default

        return max(0.0, min(100.0, number))

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number)

    def _read_hedonic_allocation(self) -> tuple[float, float]:
        hedonic = 0.5
        token_mult = 1.0
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient

            gradient = get_hedonic_gradient()
            allocation = getattr(gradient, "allocation", None)
            if allocation:
                hedonic = self._coerce_unit_value(
                    getattr(gradient, "score", getattr(allocation, "hedonic_score", hedonic)),
                    label="hedonic_score",
                    default=hedonic,
                )
                token_mult = self._coerce_token_multiplier(
                    getattr(allocation, "token_multiplier", token_mult)
                )
        except COMPUTE_RECOVERABLE_ERRORS as exc:
            _record_compute_degradation(
                exc,
                action="used neutral hedonic allocation while causal hedonic gradient was unavailable",
                extra={"hedonic_score": hedonic, "token_multiplier": token_mult},
            )
            logger.debug("Hedonic allocation unavailable: %s", exc)

        return hedonic, token_mult

    def _coerce_unit_value(self, value: Any, *, label: str, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            _record_compute_degradation(
                exc,
                action=f"replaced invalid {label} with neutral default",
                extra={"sample": repr(value), "default": default},
            )
            return default

        if not math.isfinite(number):
            _record_compute_degradation(
                ValueError(f"non-finite {label}: {number!r}"),
                action=f"replaced non-finite {label} with neutral default",
                extra={"sample": repr(value), "default": default},
            )
            return default

        return max(0.0, min(1.0, number))

    def _coerce_token_multiplier(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            _record_compute_degradation(
                exc,
                action="replaced invalid hedonic token multiplier with neutral default",
                extra={"sample": repr(value), "default": 1.0},
            )
            return 1.0

        if not math.isfinite(number):
            _record_compute_degradation(
                ValueError(f"non-finite token multiplier: {number!r}"),
                action="replaced non-finite hedonic token multiplier with neutral default",
                extra={"sample": repr(value), "default": 1.0},
            )
            return 1.0

        return max(MIN_TOKEN_MULTIPLIER, min(MAX_TOKEN_MULTIPLIER, number))

    # ── Allocation computation ────────────────────────────────────────────

    def _compute_bg_tasks(self, cpu: float, ram: float, hedonic: float) -> int:
        if cpu > CPU_CRITICAL or ram > RAM_CRITICAL:
            return 0
        if cpu > CPU_HIGH or ram > RAM_HIGH:
            return 1
        # Scale by hedonic: flourishing → more parallel capacity
        base = 2
        bonus = int(hedonic * 3)
        return min(8, base + bonus)

    def _compute_depth(self, cpu: float, ram: float, hedonic: float,
                        token_mult: float) -> float:
        if cpu > CPU_CRITICAL:
            return 0.5
        if cpu > CPU_HIGH:
            return 0.75
        # Combine resource headroom with hedonic state
        resource_headroom = 1.0 - (cpu / 100.0) * 0.5 - (ram / 100.0) * 0.2
        hedonic_floor = 0.85 + (hedonic * 0.3)
        return max(0.5, min(1.5, resource_headroom * token_mult * hedonic_floor))

    def _push_anxiety_to_affect(self, anxiety: float) -> bool:
        try:
            from core.container import ServiceContainer

            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None or not hasattr(affect, "apply_stimulus"):
                _record_compute_degradation(
                    LookupError("affect_engine.apply_stimulus unavailable"),
                    action="kept compute throttles active without affect feedback because affect stimulus path is unavailable",
                    severity="warning",
                    extra={"anxiety": anxiety},
                )
                return False

            result = affect.apply_stimulus("resource_strain", anxiety * 5)
            if inspect.isawaitable(result):
                delivery = self._observe_affect_delivery(result)
                try:
                    get_task_tracker().track(
                        delivery,
                        name="ComputeOrchestrator.affect_anxiety",
                    )
                except COMPUTE_RECOVERABLE_ERRORS:
                    close = getattr(delivery, "close", None)
                    if callable(close):
                        close()
                    raise
            else:
                self._affect_delivery_failures = 0
                self._last_affect_delivery_error = None
            return True
        except COMPUTE_RECOVERABLE_ERRORS as exc:
            self._affect_delivery_failures += 1
            self._last_affect_delivery_error = f"{type(exc).__name__}: {exc}"
            _record_compute_degradation(
                exc,
                action="recorded failed resource-anxiety affect delivery and preserved compute throttle state",
                extra={"anxiety": anxiety, "failures": self._affect_delivery_failures},
            )
            logger.debug("Affect anxiety delivery unavailable: %s", exc)
            return False

    async def _observe_affect_delivery(self, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except COMPUTE_RECOVERABLE_ERRORS as exc:
            self._affect_delivery_failures += 1
            self._last_affect_delivery_error = f"{type(exc).__name__}: {exc}"
            _record_compute_degradation(
                exc,
                action="captured asynchronous affect anxiety delivery failure and left compute throttles active",
                extra={"failures": self._affect_delivery_failures},
            )
            logger.debug("Async affect anxiety delivery failed: %s", exc)
        else:
            self._affect_delivery_failures = 0
            self._last_affect_delivery_error = None

    def _default_state(self) -> ResourceState:
        return ResourceState(
            cpu_pct=30.0, ram_pct=40.0, temp_c=None,
            hedonic_score=0.5, token_multiplier=1.0,
            background_tasks_allowed=2, context_depth_factor=1.0,
            thermal_ok=True,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_orchestrator: ComputeOrchestrator | None = None


def get_compute_orchestrator() -> ComputeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ComputeOrchestrator()
    return _orchestrator
