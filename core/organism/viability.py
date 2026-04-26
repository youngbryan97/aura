"""core/organism/viability.py

Explicit metabolism / viability state machine.
================================================
Aura is treated as a self-maintaining autonomous runtime with named
viability states and behaviorally load-bearing variables. The model is:

  food   = compute budget, memory budget, user interaction, learning signal,
           successful goal completion
  fatigue = token pressure, memory pressure, unresolved goals, error rate,
            social overextension, failed tool loops
  waste   = stale tasks, low-value memories, failed hypotheses, dead
            connections, irrelevant goals
  injury  = corrupted state, failed subsystem, incoherent belief conflict,
            broken tool, unstable substrate
  healing = pruning, backup restoration, belief reconciliation, sleep/dream
            consolidation, resource cooldown

Viability state machine:

  HEALTHY -> TIRED -> STARVED -> DEGRADED -> INJURED -> RECOVERING ->
             ASLEEP -> DREAMING -> REBOOTING -> DEAD

Each state alters concrete runtime behavior:

  * initiative budget         (max autonomous actions per minute)
  * tool risk tolerance        (which capability domains can be invoked)
  * memory threshold           (size of episodic / vector deques)
  * self-mod permission        (block / staged / open)
  * communication style        (acknowledgment-only / brief / full)

The state machine is observed by the AgencyOrchestrator, the universal
error UX layer, the dashboard, and the conversation lane controller.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.Viability")


class ViabilityState(str, enum.Enum):
    HEALTHY = "healthy"
    TIRED = "tired"
    STARVED = "starved"
    DEGRADED = "degraded"
    INJURED = "injured"
    RECOVERING = "recovering"
    ASLEEP = "asleep"
    DREAMING = "dreaming"
    REBOOTING = "rebooting"
    DEAD = "dead"


# ---------------------------------------------------------------------------
# Concrete behavioral effects per state — these are read by other systems
# to actually change behavior, not just labels.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateBehavior:
    initiative_budget_per_min: float
    tool_risk_tolerance: str  # "open" | "constrained" | "blocked"
    memory_episode_window: int
    self_mod_permission: str  # "open" | "staged" | "blocked"
    comm_style: str  # "full" | "brief" | "acknowledgment_only" | "silent"
    chat_max_concurrent: int


_BEHAVIORS: Dict[ViabilityState, StateBehavior] = {
    ViabilityState.HEALTHY:    StateBehavior(6.0, "open",        4096, "open",   "full",                 4),
    ViabilityState.TIRED:      StateBehavior(3.0, "constrained", 3072, "staged", "full",                 3),
    ViabilityState.STARVED:    StateBehavior(1.0, "constrained", 2048, "staged", "brief",                2),
    ViabilityState.DEGRADED:   StateBehavior(0.5, "constrained", 2048, "blocked","brief",                1),
    ViabilityState.INJURED:    StateBehavior(0.0, "blocked",     1024, "blocked","brief",                1),
    ViabilityState.RECOVERING: StateBehavior(0.5, "constrained", 1024, "blocked","brief",                1),
    ViabilityState.ASLEEP:     StateBehavior(0.0, "blocked",     1024, "blocked","silent",               0),
    ViabilityState.DREAMING:   StateBehavior(0.0, "blocked",     2048, "blocked","silent",               0),
    ViabilityState.REBOOTING:  StateBehavior(0.0, "blocked",      512, "blocked","silent",               0),
    ViabilityState.DEAD:       StateBehavior(0.0, "blocked",      256, "blocked","silent",               0),
}


# ---------------------------------------------------------------------------
# Viability variables — sampled live; aggregated into a viability score.
# ---------------------------------------------------------------------------


@dataclass
class ViabilitySample:
    cpu_pct: float
    ram_pct: float
    disk_pct: float
    error_rate_per_min: float  # rolling
    failed_tool_loops: int
    unresolved_goals: int
    successful_goals_last_hour: int
    user_interactions_last_hour: int
    incoherent_beliefs: int
    broken_subsystems: int
    sampled_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ViabilityEngine:
    """Drives the viability state from telemetry samples.

    Subscribes to a sampler callable (defaults to ``_sample_from_container``)
    and recomputes the state on every tick. Hysteresis prevents flap; a
    state transition fires `on_transition` callbacks (used by the dashboard).
    """

    HYSTERESIS_S = 8.0

    def __init__(self, sampler: Optional[Callable[[], ViabilitySample]] = None) -> None:
        self._sampler = sampler or _sample_from_container
        self.state: ViabilityState = ViabilityState.HEALTHY
        self.last_transition_at: float = time.time()
        self._on_transition: List[Callable[[ViabilityState, ViabilityState], None]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # -------- score / classify ---------------------------------------------

    @staticmethod
    def _classify(sample: ViabilitySample) -> ViabilityState:
        # Death: total subsystem failure or unrecoverable corruption.
        if sample.broken_subsystems >= 6:
            return ViabilityState.DEAD

        # Injured: critical resource exhaustion or multiple broken subsystems.
        if (
            sample.disk_pct > 97.0
            or sample.broken_subsystems >= 3
            or sample.incoherent_beliefs >= 5
        ):
            return ViabilityState.INJURED

        # Degraded: significant resource pressure or sustained errors.
        if (
            sample.ram_pct > 92.0
            or sample.cpu_pct > 95.0
            or sample.error_rate_per_min > 12.0
            or sample.failed_tool_loops >= 4
            or sample.broken_subsystems >= 1
        ):
            return ViabilityState.DEGRADED

        # Starved: low food / high backlog.
        if (
            sample.ram_pct > 85.0
            or sample.unresolved_goals >= 8
            or (sample.user_interactions_last_hour == 0 and sample.successful_goals_last_hour == 0)
        ):
            return ViabilityState.STARVED

        # Tired: moderate fatigue, still productive.
        if (
            sample.ram_pct > 75.0
            or sample.unresolved_goals >= 4
            or sample.error_rate_per_min > 4.0
        ):
            return ViabilityState.TIRED

        return ViabilityState.HEALTHY

    # -------- public API ---------------------------------------------------

    def behavior(self) -> StateBehavior:
        return _BEHAVIORS[self.state]

    def transition_to(self, new: ViabilityState, *, reason: str = "") -> None:
        if new == self.state:
            return
        old = self.state
        self.state = new
        self.last_transition_at = time.time()
        logger.info("🩸 viability %s → %s (%s)", old.value, new.value, reason)
        for cb in list(self._on_transition):
            try:
                cb(old, new)
            except Exception as exc:
                logger.debug("viability transition cb error: %s", exc)

    def on_transition(self, cb: Callable[[ViabilityState, ViabilityState], None]) -> None:
        self._on_transition.append(cb)

    def tick(self) -> ViabilityState:
        sample = self._sampler()
        proposed = self._classify(sample)
        # Hysteresis: do not move into a *less severe* state until enough
        # time has passed since the last transition, to avoid flap.
        if self._severity_rank(proposed) < self._severity_rank(self.state):
            if time.time() - self.last_transition_at < self.HYSTERESIS_S:
                return self.state
        self.transition_to(proposed, reason="tick")
        return self.state

    @staticmethod
    def _severity_rank(state: ViabilityState) -> int:
        order = [
            ViabilityState.HEALTHY,
            ViabilityState.TIRED,
            ViabilityState.STARVED,
            ViabilityState.DEGRADED,
            ViabilityState.RECOVERING,
            ViabilityState.INJURED,
            ViabilityState.ASLEEP,
            ViabilityState.DREAMING,
            ViabilityState.REBOOTING,
            ViabilityState.DEAD,
        ]
        try:
            return order.index(state)
        except ValueError:
            return 0

    # -------- background loop ---------------------------------------------

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    self.tick()
                except Exception as exc:
                    logger.debug("viability tick error: %s", exc)
                await asyncio.sleep(interval)

        self._task = asyncio.create_task(_loop(), name="ViabilityEngine")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # -------- introspection -----------------------------------------------

    def report(self) -> Dict[str, Any]:
        b = self.behavior()
        return {
            "state": self.state.value,
            "since": self.last_transition_at,
            "behavior": {
                "initiative_budget_per_min": b.initiative_budget_per_min,
                "tool_risk_tolerance": b.tool_risk_tolerance,
                "memory_episode_window": b.memory_episode_window,
                "self_mod_permission": b.self_mod_permission,
                "comm_style": b.comm_style,
                "chat_max_concurrent": b.chat_max_concurrent,
            },
        }


# ---------------------------------------------------------------------------
# Default sampler — read from ServiceContainer subsystems.
# ---------------------------------------------------------------------------


def _sample_from_container() -> ViabilitySample:
    cpu = ram = disk = 0.0
    error_rate = 0.0
    failed_loops = 0
    unresolved = 0
    successful = 0
    interactions = 0
    incoherent = 0
    broken = 0
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        try:
            disk = psutil.disk_usage("/").percent
        except Exception:
            disk = 0.0
    except Exception:
        pass
    try:
        from core.container import ServiceContainer
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian is not None:
            r = getattr(guardian, "last_report", None)
            if r is not None:
                broken = sum(1 for c in getattr(r, "checks", []) if not getattr(c, "healthy", True))
        goal_engine = ServiceContainer.get("goal_engine", default=None) or ServiceContainer.get("goals", default=None)
        if goal_engine is not None and hasattr(goal_engine, "open_goal_count"):
            unresolved = int(goal_engine.open_goal_count() or 0)
            successful = int(getattr(goal_engine, "completed_last_hour", 0) or 0)
        belief_graph = ServiceContainer.get("belief_graph", default=None)
        if belief_graph is not None and hasattr(belief_graph, "incoherent_count"):
            incoherent = int(belief_graph.incoherent_count() or 0)
    except Exception:
        pass
    return ViabilitySample(
        cpu_pct=cpu,
        ram_pct=ram,
        disk_pct=disk,
        error_rate_per_min=error_rate,
        failed_tool_loops=failed_loops,
        unresolved_goals=unresolved,
        successful_goals_last_hour=successful,
        user_interactions_last_hour=interactions,
        incoherent_beliefs=incoherent,
        broken_subsystems=broken,
    )


_ENGINE: Optional[ViabilityEngine] = None


def get_viability() -> ViabilityEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ViabilityEngine()
    return _ENGINE


__all__ = [
    "ViabilityState",
    "StateBehavior",
    "ViabilitySample",
    "ViabilityEngine",
    "get_viability",
]
