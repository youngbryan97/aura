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

from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("Aura.ComputeOrchestrator")

# ── Thresholds ────────────────────────────────────────────────────────────────

CPU_CRITICAL    = 90.0   # % — throttle everything
CPU_HIGH        = 75.0   # % — reduce background tasks
CPU_NORMAL      = 50.0   # % — full operation
RAM_CRITICAL    = 90.0   # % — emergency conservation
RAM_HIGH        = 80.0   # % — reduce context depth
TEMP_CRITICAL   = 90.0   # °C (macOS CPU temp) — thermal throttle
UPDATE_INTERVAL = 15.0   # seconds between resource checks


@dataclass
class ResourceState:
    cpu_pct: float
    ram_pct: float
    temp_c: Optional[float]
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
        self._state: Optional[ResourceState] = None
        self._last_update: float = 0.0
        self._anxiety_ema: float = 0.0
        self._throttle_count: int = 0
        logger.info("ComputeOrchestrator online — dynamic resource allocation active.")

    # ── Public API ────────────────────────────────────────────────────────

    async def update(self) -> ResourceState:
        """Sample resource state and update allocations."""
        if time.time() - self._last_update < UPDATE_INTERVAL:
            return self._state or self._default_state()

        cpu, ram, temp = await self._sample_resources()
        hedonic = 0.5
        token_mult = 1.0
        try:
            from core.consciousness.hedonic_gradient import get_hedonic_gradient
            hg = get_hedonic_gradient()
            if hg.allocation:
                hedonic = hg.score
                token_mult = hg.allocation.token_multiplier
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

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
    def current_state(self) -> Optional[ResourceState]:
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

    # ── Resource sampling ─────────────────────────────────────────────────

    async def _sample_resources(self):
        try:
            import psutil
            cpu = await asyncio.to_thread(psutil.cpu_percent, 0.1)
            ram = psutil.virtual_memory().percent

            # Thermal — macOS only, graceful fallback
            temp = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    all_temps = [t.current for readings in temps.values()
                                 for t in readings if t.current]
                    temp = max(all_temps) if all_temps else None
            except (AttributeError, NotImplementedError):
                logger.debug("Suppressed bare exception")
                pass

            return float(cpu), float(ram), temp
        except Exception:
            return 40.0, 50.0, None

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
        return max(0.5, min(1.5, resource_headroom * token_mult))

    def _push_anxiety_to_affect(self, anxiety: float):
        try:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect and hasattr(affect, "apply_stimulus"):
                get_task_tracker().track(
                    affect.apply_stimulus("resource_strain", anxiety * 5)
                )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def _default_state(self) -> ResourceState:
        return ResourceState(
            cpu_pct=30.0, ram_pct=40.0, temp_c=None,
            hedonic_score=0.5, token_multiplier=1.0,
            background_tasks_allowed=2, context_depth_factor=1.0,
            thermal_ok=True,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_orchestrator: Optional[ComputeOrchestrator] = None


def get_compute_orchestrator() -> ComputeOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ComputeOrchestrator()
    return _orchestrator
