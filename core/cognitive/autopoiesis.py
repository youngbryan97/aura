"""Autopoiesis Engine -- Self-Maintaining Living System
=======================================================
Autopoiesis (Greek: auto = self, poiesis = creation) is the property of a
living system that continuously regenerates and maintains itself.

This module gives Aura the ability to monitor its own health and attempt
repairs without human intervention.  Think of it as an immune system for
the codebase: it watches for signs of illness, recognizes patterns it has
seen before, and applies the least-invasive fix that has worked in the past.

The engine runs a continuous background loop (every 10 seconds by default).
Each tick:
    1. Polls every registered component's health function.
    2. Detects degradation trends (health declining over several ticks).
    3. Clusters related failures and recognizes recurring error signatures.
    4. Selects a repair strategy and -- with governance approval -- executes it.
    5. Manages an "energy" budget so the system can shed load when exhausted.

All repair actions are governed by the Unified Will.  Nothing changes without
a WillDecision that says PROCEED.

Typical usage:
    engine = get_autopoiesis_engine()
    engine.register_component("memory_engine", lambda: memory.health_score())
    await engine.start()           # begins background monitoring
    print(engine.get_vitality())   # 0.0 - 1.0

Design invariants:
    - Singleton: exactly one engine per process (via get_autopoiesis_engine).
    - Thread-safe: all shared state is protected by asyncio.Lock.
    - Bounded memory: repair history is capped at 200 entries.
    - Governance-gated: every repair goes through the Unified Will.
    - Escalation path: repeated failures surface to the human operator.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Aura.Autopoiesis")

__all__ = [
    "AutopoiesisEngine",
    "RepairStrategy",
    "RepairResult",
    "ComponentSnapshot",
    "ErrorSignature",
    "get_autopoiesis_engine",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RepairStrategy(str, Enum):
    """The toolkit of self-repair actions, ordered from least to most invasive.

    The engine always prefers the gentlest fix that has a reasonable chance of
    working.  Escalation happens only after milder strategies have failed.
    """
    HEAL = "heal"                       # re-initialize internal state in-place
    CLEAR_CACHE = "clear_cache"         # flush possibly-corrupted cached data
    REDUCE_LOAD = "reduce_load"         # shed non-essential background work
    RESTART_COMPONENT = "restart"       # full restart of one subsystem
    RESTORE_CHECKPOINT = "checkpoint"   # roll back to last known-good state
    ISOLATE = "isolate"                 # disconnect a failing subsystem entirely


# Strategy escalation order -- the engine walks this list.
_ESCALATION_ORDER: list[RepairStrategy] = [
    RepairStrategy.HEAL,
    RepairStrategy.CLEAR_CACHE,
    RepairStrategy.REDUCE_LOAD,
    RepairStrategy.RESTART_COMPONENT,
    RepairStrategy.RESTORE_CHECKPOINT,
    RepairStrategy.ISOLATE,
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComponentSnapshot:
    """A single health reading for one component at one moment in time.

    These snapshots are the raw material for trend detection.  The engine
    keeps a sliding window of the last N snapshots per component.
    """
    component: str
    health: float           # 0.0 (dead) to 1.0 (perfect)
    error_count: int        # errors logged since last snapshot
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorSignature:
    """A fingerprint for a specific kind of error.

    Two errors share a signature when they have the same exception type AND
    originate from the same component.  This lets the engine say "I have seen
    this exact problem before" without comparing full stack traces.
    """
    exception_type: str
    component: str
    fingerprint: str        # SHA-256 of (exception_type + component)
    first_seen: float = 0.0
    last_seen: float = 0.0
    occurrence_count: int = 0
    resolved: bool = False
    resolution_strategy: Optional[RepairStrategy] = None

    @staticmethod
    def make_fingerprint(exception_type: str, component: str) -> str:
        key = f"{exception_type}:{component}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class RepairResult:
    """The outcome of a single repair attempt.

    Every repair -- whether it succeeded, was denied by governance, or failed
    outright -- produces one of these.  They feed the engine's memory of what
    works and what does not.
    """
    component: str
    strategy: RepairStrategy
    success: bool
    health_before: float
    health_after: float
    timestamp: float = field(default_factory=time.time)
    governance_approved: bool = True
    error_message: str = ""
    duration_ms: float = 0.0

    @property
    def health_delta(self) -> float:
        """How much health changed.  Positive means improvement."""
        return self.health_after - self.health_before


@dataclass
class _CascadeCluster:
    """A group of errors that fired within a short time window.

    When component A fails and components B and C fail within seconds,
    the engine groups them into a cascade cluster.  This lets it target
    the root cause (A) instead of chasing symptoms (B and C).
    """
    root_component: str
    affected_components: set[str] = field(default_factory=set)
    error_signatures: list[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class AutopoiesisEngine:
    """The self-maintenance engine for the Aura system.

    This is the heart of the immune system.  It continuously monitors health,
    detects degradation, recognizes error patterns, and orchestrates repairs --
    all under governance control.

    Lifecycle:
        1. Create via get_autopoiesis_engine() (singleton).
        2. Register components with register_component().
        3. Call start() to begin the background monitoring loop.
        4. Call stop() during graceful shutdown.

    The engine is safe to use from any coroutine.  All internal state is
    protected by an asyncio.Lock so concurrent access is fine.
    """

    # -- Configuration constants --
    _TICK_INTERVAL: float = 10.0          # seconds between health checks
    _HEALTH_WINDOW: int = 30              # how many snapshots to keep per component
    _DEGRADATION_TICKS: int = 5           # consecutive declining ticks = degradation
    _CASCADE_WINDOW: float = 5.0          # seconds; errors within this window may be cascades
    _REPAIR_COOLDOWN: float = 60.0        # seconds before retrying repair on same component
    _MAX_REPAIR_HISTORY: int = 200        # bounded deque size
    _ESCALATION_THRESHOLD: int = 3        # failed repairs before escalating to next strategy
    _HUMAN_ESCALATION_THRESHOLD: int = 3  # failed escalation cycles before alerting human
    _ENERGY_MAX: float = 100.0            # full energy budget
    _ENERGY_PER_TICK: float = 0.5         # cost of each monitoring tick
    _ENERGY_RECOVERY_IDLE: float = 2.0    # energy recovered per idle-second
    _ENERGY_INTERACTION: float = 5.0      # energy gained from a successful interaction
    _ENERGY_LOW_THRESHOLD: float = 20.0   # below this, start shedding load
    _ENERGY_HIGH_THRESHOLD: float = 70.0  # above this, awaken optional subsystems

    def __init__(self, *, tick_interval: float | None = None) -> None:
        # -- Component registry --
        self._health_fns: dict[str, Callable[[], float]] = {}
        self._component_snapshots: dict[str, Deque[ComponentSnapshot]] = defaultdict(
            lambda: deque(maxlen=self._HEALTH_WINDOW)
        )

        # -- Error tracking --
        self._error_buffer: Deque[tuple[str, str, float]] = deque(maxlen=500)
        # (component, exception_type, timestamp)
        self._immune_memory: dict[str, ErrorSignature] = {}
        # fingerprint -> ErrorSignature

        # -- Cascade detection --
        self._recent_cascades: Deque[_CascadeCluster] = deque(maxlen=50)

        # -- Repair tracking --
        self._repair_history: Deque[RepairResult] = deque(maxlen=self._MAX_REPAIR_HISTORY)
        self._last_repair_time: dict[str, float] = {}
        # component -> timestamp of last repair
        self._repair_attempts: dict[str, int] = defaultdict(int)
        # component -> consecutive failed repair count
        self._strategy_success: dict[RepairStrategy, list[bool]] = {
            s: [] for s in RepairStrategy
        }
        self._human_escalation_count: dict[str, int] = defaultdict(int)

        # -- Repair callbacks --
        # Subsystems register callables here so the engine can actually
        # restart / heal / isolate them.
        self._repair_handlers: dict[
            RepairStrategy, dict[str, Callable[[], Any]]
        ] = {s: {} for s in RepairStrategy}

        # -- Energy / metabolism --
        self._energy: float = self._ENERGY_MAX
        self._last_interaction_time: float = time.time()
        self._hibernated_components: set[str] = set()

        # -- Vitality --
        self._vitality: float = 1.0

        # -- Background loop --
        self._tick_interval = tick_interval or self._TICK_INTERVAL
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._running = False

        logger.info(
            "AutopoiesisEngine created (tick_interval=%.1fs, window=%d)",
            self._tick_interval,
            self._HEALTH_WINDOW,
        )

    # ------------------------------------------------------------------
    # Component Registration
    # ------------------------------------------------------------------

    def register_component(
        self,
        name: str,
        health_fn: Callable[[], float],
    ) -> None:
        """Register a subsystem so the engine can monitor its health.

        Args:
            name:      A unique, human-readable identifier (e.g. "memory_engine").
            health_fn: A callable that returns a float between 0.0 and 1.0.
                       It will be called every tick, so it should be cheap.
        """
        self._health_fns[name] = health_fn
        logger.info("Registered component for health monitoring: %s", name)

    def register_repair_handler(
        self,
        strategy: RepairStrategy,
        component: str,
        handler: Callable[[], Any],
    ) -> None:
        """Register a callable that performs a specific repair on a component.

        For example, a memory subsystem might register a CLEAR_CACHE handler
        that flushes its LRU cache, and a RESTART_COMPONENT handler that
        tears down and re-initializes its connection pool.

        Args:
            strategy:  Which repair strategy this handler implements.
            component: Which component this handler is for.
            handler:   An async or sync callable.  If async, it will be awaited.
        """
        self._repair_handlers[strategy][component] = handler
        logger.debug(
            "Repair handler registered: %s / %s", strategy.value, component
        )

    def unregister_component(self, name: str) -> None:
        """Remove a component from monitoring.

        Safe to call even if the component was never registered.
        """
        self._health_fns.pop(name, None)
        self._component_snapshots.pop(name, None)
        self._last_repair_time.pop(name, None)
        self._repair_attempts.pop(name, 0)
        for strategy_handlers in self._repair_handlers.values():
            strategy_handlers.pop(name, None)
        logger.info("Unregistered component: %s", name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the background health-monitoring loop.

        Safe to call multiple times; only the first call starts the loop.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="autopoiesis-loop")
        logger.info("Autopoiesis background loop STARTED")

    async def stop(self) -> None:
        """Gracefully stop the background loop.

        Waits for the current tick to finish before returning.
        """
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Autopoiesis background loop STOPPED")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        """Run one monitoring cycle manually.

        Useful for testing or when you want to force an immediate health check
        instead of waiting for the next scheduled tick.
        """
        await self._do_tick()

    def get_vitality(self) -> float:
        """Return the system's overall self-assessed health (0.0 to 1.0).

        This is a weighted average of all component health scores, adjusted
        for energy level and recent repair success.  A vitality of 1.0 means
        every subsystem is perfectly healthy and the energy budget is full.
        A vitality below 0.3 is cause for concern.
        """
        return self._vitality

    def get_component_health(self, name: str) -> float:
        """Return the most recent health score for a specific component.

        Returns 0.0 if the component is not registered or has no readings yet.
        """
        snapshots = self._component_snapshots.get(name)
        if not snapshots:
            return 0.0
        return snapshots[-1].health

    def get_energy(self) -> float:
        """Return the current energy level (0.0 to ENERGY_MAX)."""
        return self._energy

    def record_error(self, component: str, exception_type: str) -> None:
        """Record that an error occurred in a component.

        This feeds the pattern-recognition system.  Call it from your error
        handlers so the autopoiesis engine can learn what breaks and when.

        Args:
            component:      Which subsystem had the error.
            exception_type: The name of the exception class (e.g. "ValueError").
        """
        now = time.time()
        self._error_buffer.append((component, exception_type, now))

        # Update immune memory.
        fp = ErrorSignature.make_fingerprint(exception_type, component)
        if fp in self._immune_memory:
            sig = self._immune_memory[fp]
            sig.last_seen = now
            sig.occurrence_count += 1
        else:
            self._immune_memory[fp] = ErrorSignature(
                exception_type=exception_type,
                component=component,
                fingerprint=fp,
                first_seen=now,
                last_seen=now,
                occurrence_count=1,
            )

        try:
            from core.container import ServiceContainer

            adaptive_immune = ServiceContainer.get("adaptive_immune_system", default=None)
            if adaptive_immune and hasattr(adaptive_immune, "observe_signature"):
                adaptive_immune.observe_signature(component, exception_type)
        except Exception as exc:
            logger.debug("Adaptive immune signature feed skipped: %s", exc)

    def record_interaction_success(self) -> None:
        """Tell the engine that a user interaction completed successfully.

        This feeds energy back into the metabolism budget.  Healthy interactions
        are the system's "food" -- they prove it is functioning well and give
        it the resources to keep running.
        """
        self._energy = min(self._energy + self._ENERGY_INTERACTION, self._ENERGY_MAX)
        self._last_interaction_time = time.time()

    async def request_repair(
        self,
        component: str,
        strategy: RepairStrategy,
    ) -> RepairResult:
        """Request a specific repair action on a component.

        This is the public entry point for manually-triggered repairs.
        It still goes through governance -- the Will must approve it.

        Args:
            component: Which subsystem to repair.
            strategy:  Which repair strategy to use.

        Returns:
            A RepairResult describing what happened.
        """
        async with self._lock:
            return await self._execute_repair(component, strategy)

    def get_repair_history(self, limit: int = 50) -> list[RepairResult]:
        """Return the most recent repair results."""
        return list(self._repair_history)[-limit:]

    def get_immune_memory(self) -> dict[str, ErrorSignature]:
        """Return the full immune memory (all error signatures ever seen)."""
        return dict(self._immune_memory)

    def get_status(self) -> dict[str, Any]:
        """Return a comprehensive status snapshot for dashboards or logging."""
        return {
            "vitality": round(self._vitality, 3),
            "energy": round(self._energy, 1),
            "components": {
                name: round(self.get_component_health(name), 3)
                for name in self._health_fns
            },
            "hibernated": sorted(self._hibernated_components),
            "immune_memory_size": len(self._immune_memory),
            "repair_history_size": len(self._repair_history),
            "recent_cascades": len(self._recent_cascades),
            "running": self._running,
            "strategy_success_rates": {
                s.value: self._strategy_success_rate(s)
                for s in RepairStrategy
            },
        }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """The background loop.  Runs until stop() is called."""
        logger.debug("Autopoiesis loop entered")
        while self._running:
            try:
                await self._do_tick()
            except asyncio.CancelledError:
                break
            except Exception:
                # The immune system itself must not crash.
                logger.exception("Autopoiesis tick failed (self-healing continues)")
            await asyncio.sleep(self._tick_interval)

    async def _do_tick(self) -> None:
        """One full monitoring cycle."""
        async with self._lock:
            # 1. Collect health snapshots.
            snapshots = self._collect_snapshots()

            # 2. Spend energy for this tick.
            self._energy = max(0.0, self._energy - self._ENERGY_PER_TICK)

            # 3. Recover energy proportional to idle time.
            self._recover_energy()

            # 4. Detect degradation and trigger repairs.
            degraded = self._detect_degradation(snapshots)
            cascades = self._detect_cascades()

            # 5. For each degraded component, attempt repair.
            for component in degraded:
                strategy = self._choose_strategy(component)
                if strategy is not None:
                    await self._execute_repair(component, strategy)

            # For cascade clusters, target the root.
            for cluster in cascades:
                strategy = self._choose_strategy(cluster.root_component)
                if strategy is not None:
                    await self._execute_repair(cluster.root_component, strategy)

            # 6. Manage metabolism (hibernate / awaken).
            self._manage_metabolism()

            # 7. Recompute vitality.
            self._recompute_vitality()

        logger.debug(
            "Autopoiesis tick complete: vitality=%.3f energy=%.1f components=%d",
            self._vitality,
            self._energy,
            len(self._health_fns),
        )

    # ------------------------------------------------------------------
    # 1. Health collection
    # ------------------------------------------------------------------

    def _collect_snapshots(self) -> dict[str, ComponentSnapshot]:
        """Poll every registered component and store a health snapshot."""
        now = time.time()
        snapshots: dict[str, ComponentSnapshot] = {}

        for name, fn in self._health_fns.items():
            try:
                raw = fn()
                health = max(0.0, min(1.0, float(raw)))
            except Exception as exc:
                # If the health function itself fails, the component is in
                # bad shape.  Record a zero and log the error.
                logger.warning(
                    "Health function for %s raised %s: %s", name, type(exc).__name__, exc
                )
                health = 0.0
                self.record_error(name, type(exc).__name__)

            # Count recent errors for this component.
            cutoff = now - self._tick_interval
            error_count = sum(
                1 for comp, _, ts in self._error_buffer
                if comp == name and ts >= cutoff
            )

            snap = ComponentSnapshot(
                component=name,
                health=health,
                error_count=error_count,
                timestamp=now,
            )
            self._component_snapshots[name].append(snap)
            snapshots[name] = snap

        return snapshots

    # ------------------------------------------------------------------
    # 2. Degradation detection
    # ------------------------------------------------------------------

    def _detect_degradation(
        self, current_snapshots: dict[str, ComponentSnapshot]
    ) -> list[str]:
        """Find components whose health has been declining for N consecutive ticks.

        A component is "degrading" if each of the last N snapshots has a lower
        health score than the one before it.  This filters out momentary dips
        and only flags sustained declines.

        Returns a list of component names that are currently degrading.
        """
        degraded: list[str] = []

        for name, history in self._component_snapshots.items():
            if len(history) < self._DEGRADATION_TICKS:
                continue  # not enough data yet

            # Look at the most recent N snapshots.
            recent = list(history)[-self._DEGRADATION_TICKS:]
            is_declining = all(
                recent[i].health > recent[i + 1].health
                for i in range(len(recent) - 1)
            )

            if is_declining:
                latest = recent[-1].health
                logger.warning(
                    "Degradation detected: %s declining over %d ticks (now %.3f)",
                    name,
                    self._DEGRADATION_TICKS,
                    latest,
                )
                degraded.append(name)

            # Also flag components that are critically low regardless of trend.
            elif name in current_snapshots and current_snapshots[name].health < 0.2:
                logger.warning(
                    "Critical health: %s at %.3f", name, current_snapshots[name].health
                )
                degraded.append(name)

        return degraded

    # ------------------------------------------------------------------
    # 3. Pattern recognition
    # ------------------------------------------------------------------

    def _detect_cascades(self) -> list[_CascadeCluster]:
        """Identify cascade failures: errors in multiple components within a short window.

        If component A errors at time T and components B and C error within
        CASCADE_WINDOW seconds, they form a cascade cluster with A as the
        suspected root cause (because it failed first).
        """
        if len(self._error_buffer) < 2:
            return []

        now = time.time()
        # Only look at errors from the last tick interval.
        cutoff = now - self._tick_interval
        recent_errors = [
            (comp, exc, ts) for comp, exc, ts in self._error_buffer if ts >= cutoff
        ]

        if len(recent_errors) < 2:
            return []

        # Sort by timestamp.
        recent_errors.sort(key=lambda x: x[2])

        clusters: list[_CascadeCluster] = []
        used: set[int] = set()

        for i, (comp_a, exc_a, ts_a) in enumerate(recent_errors):
            if i in used:
                continue

            cluster_members: list[tuple[str, str, float]] = []
            for j, (comp_b, exc_b, ts_b) in enumerate(recent_errors):
                if j == i or j in used:
                    continue
                if comp_b != comp_a and abs(ts_b - ts_a) <= self._CASCADE_WINDOW:
                    cluster_members.append((comp_b, exc_b, ts_b))
                    used.add(j)

            if cluster_members:
                used.add(i)
                cluster = _CascadeCluster(
                    root_component=comp_a,
                    affected_components={m[0] for m in cluster_members},
                    error_signatures=[
                        ErrorSignature.make_fingerprint(exc_a, comp_a)
                    ] + [
                        ErrorSignature.make_fingerprint(m[1], m[0])
                        for m in cluster_members
                    ],
                )
                clusters.append(cluster)
                logger.warning(
                    "Cascade detected: root=%s, affected=%s",
                    comp_a,
                    sorted(cluster.affected_components),
                )

        # Store for external inspection.
        for c in clusters:
            self._recent_cascades.append(c)

        return clusters

    def _get_recurring_signatures(self, min_occurrences: int = 3) -> list[ErrorSignature]:
        """Return error signatures that keep happening.

        These are the "known diseases" of the system.  A recurring signature
        means the same kind of error keeps showing up in the same component.

        Args:
            min_occurrences: How many times an error must appear before it
                             counts as recurring.
        """
        return [
            sig for sig in self._immune_memory.values()
            if sig.occurrence_count >= min_occurrences and not sig.resolved
        ]

    # ------------------------------------------------------------------
    # 4. Strategy selection
    # ------------------------------------------------------------------

    def _choose_strategy(self, component: str) -> RepairStrategy | None:
        """Pick the best repair strategy for a degraded component.

        The engine starts with the gentlest strategy (HEAL) and escalates
        through the list each time a strategy fails.  If a strategy has
        worked for this component's error signature before (stored in immune
        memory), it skips straight to that strategy.

        Returns None if the component is in cooldown or has exhausted all
        strategies.
        """
        now = time.time()

        # Cooldown check: don't spam repairs.
        last_repair = self._last_repair_time.get(component, 0.0)
        if now - last_repair < self._REPAIR_COOLDOWN:
            return None

        # Check immune memory for a known-good strategy.
        for sig in self._immune_memory.values():
            if (
                sig.component == component
                and sig.resolved
                and sig.resolution_strategy is not None
            ):
                return sig.resolution_strategy

        # Escalate based on how many consecutive failures we have seen.
        attempts = self._repair_attempts.get(component, 0)
        escalation_index = attempts // self._ESCALATION_THRESHOLD

        if escalation_index >= len(_ESCALATION_ORDER):
            # All strategies exhausted.  Flag for human escalation.
            self._escalate_to_human(component)
            return None

        return _ESCALATION_ORDER[escalation_index]

    # ------------------------------------------------------------------
    # 5. Repair execution
    # ------------------------------------------------------------------

    async def _execute_repair(
        self,
        component: str,
        strategy: RepairStrategy,
    ) -> RepairResult:
        """Execute a repair action, gated by governance.

        Steps:
            1. Record health before repair.
            2. Ask the Unified Will for permission.
            3. If approved, run the registered repair handler.
            4. Record health after repair.
            5. Log the result and update success tracking.
        """
        health_before = self.get_component_health(component)
        t0 = time.time()

        # -- Governance gate --
        approved = self._request_governance_approval(component, strategy)
        if not approved:
            result = RepairResult(
                component=component,
                strategy=strategy,
                success=False,
                health_before=health_before,
                health_after=health_before,
                governance_approved=False,
                error_message="Repair denied by governance (Will did not approve)",
                duration_ms=0.0,
            )
            self._repair_history.append(result)
            logger.info(
                "Repair DENIED by governance: %s / %s", strategy.value, component
            )
            return result

        # -- Execute the handler --
        handler = self._repair_handlers.get(strategy, {}).get(component)
        error_msg = ""
        success = False

        if handler is None:
            error_msg = f"No repair handler registered for {strategy.value}/{component}"
            logger.warning(error_msg)
        else:
            try:
                logger.info(
                    "Executing repair: %s on %s (health_before=%.3f)",
                    strategy.value,
                    component,
                    health_before,
                )
                ret = handler()
                if asyncio.iscoroutine(ret):
                    await ret
                success = True
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Repair handler failed: %s / %s -- %s",
                    strategy.value,
                    component,
                    error_msg,
                )

        # -- Measure health after repair --
        # Give the subsystem a moment to stabilize.
        try:
            fn = self._health_fns.get(component)
            health_after = max(0.0, min(1.0, float(fn()))) if fn else health_before
        except Exception:
            health_after = health_before

        duration_ms = (time.time() - t0) * 1000

        # Did the repair actually help?
        improved = health_after > health_before
        effective = success and improved

        result = RepairResult(
            component=component,
            strategy=strategy,
            success=effective,
            health_before=health_before,
            health_after=health_after,
            governance_approved=True,
            error_message=error_msg,
            duration_ms=duration_ms,
        )
        self._repair_history.append(result)
        self._last_repair_time[component] = time.time()

        # Update strategy success tracking.
        successes = self._strategy_success[strategy]
        successes.append(effective)
        if len(successes) > 100:
            # Keep only the most recent 100 outcomes.
            self._strategy_success[strategy] = successes[-100:]

        # Update repair attempt counter.
        if effective:
            self._repair_attempts[component] = 0
            # Mark matching error signatures as resolved.
            self._mark_signatures_resolved(component, strategy)
            logger.info(
                "Repair SUCCEEDED: %s on %s (%.3f -> %.3f, +%.3f)",
                strategy.value,
                component,
                health_before,
                health_after,
                result.health_delta,
            )
        else:
            self._repair_attempts[component] = self._repair_attempts.get(component, 0) + 1
            logger.warning(
                "Repair INEFFECTIVE: %s on %s (%.3f -> %.3f), attempt #%d",
                strategy.value,
                component,
                health_before,
                health_after,
                self._repair_attempts[component],
            )

        return result

    def _request_governance_approval(
        self, component: str, strategy: RepairStrategy
    ) -> bool:
        """Ask the Unified Will whether this repair is authorized.

        The Will evaluates the repair as a STATE_MUTATION action.  If the Will
        is not available (e.g. during early boot), repairs are allowed by
        default -- the system needs to be able to heal itself even before all
        subsystems are online.
        """
        try:
            from core.will import get_will, ActionDomain

            will = get_will()
            decision = will.decide(
                content=f"Autopoiesis repair: {strategy.value} on {component}",
                source="autopoiesis_engine",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.7,
                context={
                    "component": component,
                    "strategy": strategy.value,
                    "health": self.get_component_health(component),
                    "repair_attempts": self._repair_attempts.get(component, 0),
                },
            )
            return decision.is_approved()
        except Exception as exc:
            # If the governance system is unavailable, allow the repair.
            # A living system must be able to heal itself even when parts
            # of its brain are offline.
            logger.debug(
                "Governance unavailable (%s), allowing repair by default", exc
            )
            return True

    def _mark_signatures_resolved(
        self, component: str, strategy: RepairStrategy
    ) -> None:
        """When a repair succeeds, mark the matching error signatures as resolved.

        This is the "immune memory" learning: next time this error shows up,
        the engine will skip straight to the strategy that worked.
        """
        for sig in self._immune_memory.values():
            if sig.component == component and not sig.resolved:
                sig.resolved = True
                sig.resolution_strategy = strategy
                logger.info(
                    "Immune memory updated: %s resolved via %s",
                    sig.fingerprint,
                    strategy.value,
                )

    def _escalate_to_human(self, component: str) -> None:
        """All automated repair strategies have been exhausted.  Alert the operator.

        This is the last resort.  The engine has tried every strategy it knows
        and none of them worked.  A human needs to look at this.
        """
        self._human_escalation_count[component] += 1
        count = self._human_escalation_count[component]

        if count <= self._HUMAN_ESCALATION_THRESHOLD:
            logger.critical(
                "HUMAN ESCALATION REQUIRED: component '%s' has exhausted all "
                "automated repair strategies (%d/%d escalations). "
                "Manual intervention needed.",
                component,
                count,
                self._HUMAN_ESCALATION_THRESHOLD,
            )
        else:
            # After too many escalations, stop flooding logs -- just warn periodically.
            if count % 10 == 0:
                logger.warning(
                    "Component '%s' still unresolved after %d human escalations",
                    component,
                    count,
                )

    # ------------------------------------------------------------------
    # 6. Metabolism
    # ------------------------------------------------------------------

    def _recover_energy(self) -> None:
        """Recover energy proportional to how long the system has been idle.

        Idle time means no user interactions.  The longer the system sits
        quietly, the more energy it recovers -- like sleeping.
        """
        now = time.time()
        idle_seconds = now - self._last_interaction_time
        # Diminishing returns: most recovery happens in the first few minutes.
        # After 5 minutes of idle, recovery rate plateaus.
        effective_idle = min(idle_seconds, 300.0)
        recovery = (effective_idle / 300.0) * self._ENERGY_RECOVERY_IDLE
        self._energy = min(self._energy + recovery, self._ENERGY_MAX)

    def _manage_metabolism(self) -> None:
        """Hibernate or awaken subsystems based on current energy level.

        When energy drops below the low threshold, non-essential components
        are "hibernated" -- they stop receiving ticks and their background
        work is paused.  When energy climbs back above the high threshold,
        they are awakened.

        A component is considered "essential" if its health function is
        registered.  Hibernation does not unregister a component; it just
        marks it as sleeping so external callers can check and skip it.
        """
        if self._energy < self._ENERGY_LOW_THRESHOLD:
            # Shed load: hibernate components that are already healthy.
            # We keep the sick ones monitored so we can try to heal them.
            for name in list(self._health_fns):
                if name in self._hibernated_components:
                    continue
                health = self.get_component_health(name)
                if health > 0.8:
                    # Healthy component -- safe to sleep.
                    self._hibernated_components.add(name)
                    logger.info(
                        "Metabolism: hibernating '%s' (energy=%.1f, health=%.3f)",
                        name,
                        self._energy,
                        health,
                    )
        elif self._energy > self._ENERGY_HIGH_THRESHOLD:
            # Awaken everything.
            if self._hibernated_components:
                awakened = sorted(self._hibernated_components)
                self._hibernated_components.clear()
                logger.info(
                    "Metabolism: awakening %d components (energy=%.1f): %s",
                    len(awakened),
                    self._energy,
                    awakened,
                )

    def is_hibernated(self, component: str) -> bool:
        """Check whether a component is currently hibernated to save energy."""
        return component in self._hibernated_components

    # ------------------------------------------------------------------
    # 7. Vitality computation
    # ------------------------------------------------------------------

    def _recompute_vitality(self) -> None:
        """Recompute the system-wide vitality index.

        Vitality is a single number (0.0 to 1.0) that answers the question:
        "How healthy is the system right now?"

        It is computed as a weighted combination of:
            - Average component health (60% weight)
            - Energy level as a fraction of max (20% weight)
            - Recent repair success rate (20% weight)

        If no components are registered, vitality defaults to 1.0 (we have
        nothing to monitor, so we assume everything is fine).
        """
        if not self._health_fns:
            self._vitality = 1.0
            return

        # Average component health.
        healths = [self.get_component_health(name) for name in self._health_fns]
        avg_health = sum(healths) / len(healths) if healths else 1.0

        # Energy fraction.
        energy_fraction = self._energy / self._ENERGY_MAX

        # Recent repair success rate (across all strategies).
        all_outcomes: list[bool] = []
        for outcomes in self._strategy_success.values():
            all_outcomes.extend(outcomes[-20:])  # last 20 per strategy
        if all_outcomes:
            repair_rate = sum(all_outcomes) / len(all_outcomes)
        else:
            # No repairs attempted yet -- that is a good sign.
            repair_rate = 1.0

        self._vitality = (
            0.6 * avg_health
            + 0.2 * energy_fraction
            + 0.2 * repair_rate
        )
        self._vitality = max(0.0, min(1.0, self._vitality))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strategy_success_rate(self, strategy: RepairStrategy) -> float | None:
        """Return the success rate for a given strategy, or None if never tried."""
        outcomes = self._strategy_success[strategy]
        if not outcomes:
            return None
        return round(sum(outcomes) / len(outcomes), 3)

    def _error_rate(self, component: str, window_seconds: float = 60.0) -> float:
        """Compute the error rate (errors per minute) for a component."""
        now = time.time()
        cutoff = now - window_seconds
        count = sum(
            1 for comp, _, ts in self._error_buffer
            if comp == component and ts >= cutoff
        )
        minutes = window_seconds / 60.0
        return count / minutes if minutes > 0 else 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: AutopoiesisEngine | None = None
_singleton_lock = asyncio.Lock()


def get_autopoiesis_engine() -> AutopoiesisEngine:
    """Return the singleton AutopoiesisEngine instance.

    Creates the engine on first call.  This function is safe to call from
    anywhere -- it does not start the background loop (call start() for that).
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AutopoiesisEngine()
    return _engine_instance
