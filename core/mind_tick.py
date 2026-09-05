import asyncio
import hashlib
import logging
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.brain.metacognitive_monitor import MetacognitiveMonitor
from core.brain.predictive_engine import PredictiveEngine
from core.config import get_config
from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.predictive.trajectory_predictor import TrajectoryPredictor
from core.runtime.errors import record_degradation
from core.runtime.pipeline_blueprint import instantiate_legacy_runtime_phases
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.utils.resilience import CircuitBreaker
from core.utils.task_tracker import get_task_tracker

from .state.aura_state import AuraState

config = get_config()
logger = logging.getLogger(__name__)

_MIND_SUBSYSTEM = "mind_tick"
_MIND_BOUNDARY_ERRORS = (
    AttributeError, ImportError, LookupError, OSError,
    RuntimeError, TimeoutError, TypeError, ValueError,
    sqlite3.Error,
    asyncio.InvalidStateError,
)


def _record_mind_degradation(
    error: BaseException,
    *,
    action: str = "mind tick operation degraded and isolated",
    severity: str = "degraded",
) -> None:
    record_degradation(_MIND_SUBSYSTEM, error, severity=severity, action=action)


def _close_if_possible(awaitable: Any) -> None:
    try:
        close = awaitable.close
    except AttributeError:
        return
    try:
        close()
    except _MIND_BOUNDARY_ERRORS as exc:
        _record_mind_degradation(exc, action="unscheduled mind tick awaitable close failed")


def _schedule_mind_task(awaitable: Any, *, name: str, tracker: Any = None) -> asyncio.Task | None:
    try:
        task_owner = tracker if tracker is not None else get_task_tracker()
        try:
            schedule = task_owner.create_task
        except AttributeError:
            schedule = task_owner.track_task
        return schedule(awaitable, name=name)
    except RuntimeError as exc:
        _close_if_possible(awaitable)
        logger.debug("MindTick background task %s deferred outside an event loop: %s", name, exc)
        return None
    except _MIND_BOUNDARY_ERRORS as exc:
        _close_if_possible(awaitable)
        _record_mind_degradation(exc, action=f"mind tick background task {name} was not scheduled")
        logger.debug("MindTick background task %s scheduling failed: %s", name, exc)
        return None


def _dead_tiers_are_policy_deferred_cortex(gate: Any, dead_tiers: list[str]) -> bool:
    """Return true for the deliberate desktop cold-Cortex standby state.

    Desktop safe boot may defer the 32B lane until the first foreground user
    turn to avoid launch-time memory spikes. That state must keep boot health
    unready, but it is not an incident and must not start a repair storm.
    """
    normalized = {str(tier or "").strip().lower() for tier in dead_tiers if tier}
    if normalized != {"cortex"}:
        return False
    try:
        desktop_safe = bool(getattr(gate, "_desktop_safe_boot_enabled", lambda: False)())
        if not desktop_safe:
            return False
        lane = (
            gate.get_conversation_status()
            if hasattr(gate, "get_conversation_status")
            else {}
        )
        if bool((lane or {}).get("conversation_ready", False)):
            return False
        lane_state = str((lane or {}).get("state", "") or "").strip().lower()
        warmup_attempted = bool((lane or {}).get("warmup_attempted", False))
        cold_or_starting = lane_state in {
            "",
            "cold",
            "spawning",
            "handshaking",
            "warming",
            "recovering",
        } or not warmup_attempted
        if not cold_or_starting:
            return False
        should_schedule = getattr(gate, "_boot_should_schedule_deferred_prewarm", None)
        if callable(should_schedule) and bool(should_schedule()):
            return False
        return True
    except _MIND_BOUNDARY_ERRORS as exc:
        _record_mind_degradation(
            exc,
            action="continued LLM health accounting after deferred-cortex policy probe failed",
            severity="warning",
        )
        logger.debug("MindTick: deferred Cortex policy probe failed: %s", exc)
        return False


class CognitiveMode(Enum):
    CONVERSATIONAL = "conversational"
    REFLECTIVE = "reflective"
    SLEEP = "sleep"
    CRITICAL = "critical"

TICK_INTERVALS = {
    CognitiveMode.CONVERSATIONAL: 2.0,
    CognitiveMode.REFLECTIVE: 4.0,
    CognitiveMode.SLEEP: 10.0,
    CognitiveMode.CRITICAL: 0.5,
}

# The longest budget any single tick stage is allowed: the dream-research
# window. Every stage in the loop is bounded by an explicit wait_for, so a loop
# that has made no progress for longer than its own longest bounded stage is
# not slow — it is wedged past a bound it declared.
#
# The stall thresholds are derived from it rather than picked. They were 600s
# stale and 900s hard against a 2s conversational tick: a loop could miss three
# hundred consecutive beats and still report alive, which is far past any
# conversation-readiness budget. Both stay overridable by env.
MAX_BOUNDED_TICK_STAGE_S = 120.0

# How many consecutive ticks may run on the second phase pipeline before the
# kernel's absence is an error rather than a boot-time condition. One minute of
# CognitiveMode.NORMAL ticks: long enough to cover a slow kernel boot, short
# enough that a kernel which never arrives is reported while someone is still
# looking at the logs.
MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS = 30
DEFAULT_STALE_PROGRESS_S = MAX_BOUNDED_TICK_STAGE_S * 1.5
DEFAULT_HARD_STALL_S = MAX_BOUNDED_TICK_STAGE_S * 2.0

PhaseCallable = Callable[[AuraState], Awaitable[AuraState]]

@dataclass
class TickMetadata:
    tick_id: int
    mode: CognitiveMode
    start_time: float
    duration: float = 0.0
    phases_executed: list[str] = field(default_factory=list)
    phase_durations: dict[str, float] = field(default_factory=dict)


def _authorize_state_mutation_through_will(
    content: str,
    source: str,
    *,
    priority: float = 0.5,
    context: dict | None = None,
):
    """Fail-closed canonical admission for background state mutations."""
    try:
        from core.runtime.action_executor import ActionExecutor
        from core.will import ActionDomain

        source_text = str(source or "").strip()
        if not source_text or source_text.casefold() == "unknown":
            raise ValueError("mind tick state mutation requires an attributable source")
        bound_context = dict(context or {})
        declared_source = str(bound_context.get("source") or "").strip()
        if declared_source and declared_source != source_text:
            raise ValueError(
                "mind tick state mutation context source does not match its admission source"
            )
        bound_context["source"] = source_text
        admission = ActionExecutor.authorize_action(
            action_name=source_text,
            params={"purpose": str(content or "")[:500]},
            source=source_text,
            domain=ActionDomain.STATE_MUTATION,
            priority=priority,
            context=bound_context,
        )
        return admission.decision
    except _MIND_BOUNDARY_ERRORS as exc:
        _record_mind_degradation(exc)
        logger.warning("MindTick: UnifiedWill unavailable for %s; state mutation blocked: %s", source, exc)
        return None

class MindTick:
    """
    The unified cognitive rhythm of Aura.
    
    MindTick executes a sequence of registered 'phases' against the current state
    at a regular interval determined by the CognitiveMode.
    """
    
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.mode = CognitiveMode.CONVERSATIONAL
        self.phases: list[tuple[str, PhaseCallable]] = []
        self._running = False
        self._tick_count = 0
        self._task: asyncio.Task | None = None
        self._last_tick_metadata: TickMetadata | None = None
        self._started_at = 0.0
        self._last_successful_tick_at = 0.0
        self._last_loop_progress_at = 0.0
        self._last_progress_label = "not_started"
        self._active_tick_started_at = 0.0
        self._active_tick_stage = "idle"
        self._consecutive_loop_failures = 0
        self._last_deferred_cortex_health_log_at = 0.0
        self._last_liveness_repair_at = 0.0
        self._liveness_repair_count = 0
        #: True while a cancelled loop is unwinding and its replacement has
        #: not started yet. Exactly one _run_loop exists at any instant.
        self._repair_pending = False
        self._owner_loop = None
        
        # Cognitive Deepening Components
        self.predictive_engine = PredictiveEngine()
        self.metacognitive_monitor = MetacognitiveMonitor()
        self.trajectory_predictor = TrajectoryPredictor(ServiceContainer)
        self._last_prediction_time = 0.0
        self._last_trajectory_time = 0.0
        self._last_audit_time = 0.0
        
        # Circuit Breakers for fault isolation
        self.breakers = {
            "prediction": CircuitBreaker("predictive_engine", max_failures=3, reset_timeout=30.0),
            "audit": CircuitBreaker("metacognitive_monitor", max_failures=5, reset_timeout=60.0),  # More tolerant on 32B
            "immune_audit": CircuitBreaker("immune_pulse_audit", max_failures=2, reset_timeout=60.0),
        }
        self.phase_breakers: dict[str, CircuitBreaker] = {}
        
        # Initiative re-promotion cooldown
        self._last_initiative_goal: str | None = None
        self._last_initiative_time: float = 0.0
        self._initiative_cooldown: float = 30.0  # seconds
        self._missing_state_streak: int = 0
        self._last_missing_state_log: float = 0.0
        self._max_missing_state_backoff: float = 5.0
        self._objective_attempt_key = ""
        self._objective_attempt_inflight = ""
        self._objective_next_attempt_at = 0.0
        
        # Per-phase timeout budgets
        self.phase_timeouts = {
            "response_generation": 120.0,
            "memory_retrieval": 30.0,     # Increased for DB stability
            "cognitive_routing": 120.0,   # Matched to MLX load deadline
            "memory_consolidation": 20.0,
        }
        self.default_timeout = 5.0
        
        self._bootstrap_phases()

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    def _mark_loop_progress(self, label: str) -> None:
        """Record supervised-loop progress without claiming a completed tick.

        This also beats the system watchdog, because LIVENESS IS NOT THROUGHPUT.
        The heartbeat used to fire only once per iteration, at tick_start, under
        a 30s timeout — so a loop that was alive and deliberately backing off
        (foreground headroom reserved, allostasis protecting loop_lag_s, a
        bounded state-read timeout) looked dead. Measured live: "SYSTEM STALL
        DETECTED: Component 'mind_tick' has not responded for 34.4s!" while the
        runtime was healthy and the loop was choosing not to work. That is the
        same false-death this repo has been bitten by before.

        Every call site of this method is already a point where the loop
        demonstrably progressed, which is exactly what a heartbeat should mean.
        Slowness is still reported — by the tick-rate metric, which is the right
        instrument for it.
        """
        self._last_loop_progress_at = time.time()
        self._last_progress_label = str(label or "progress")[:80]
        try:
            from infrastructure.watchdog import get_watchdog

            get_watchdog().heartbeat("mind_tick")
        except _MIND_BOUNDARY_ERRORS as exc:
            logger.debug("MindTick: watchdog heartbeat unavailable: %s", exc)

    @staticmethod
    def _objective_key(objective: str) -> str:
        return " ".join(str(objective or "").split())[:2000]

    def _objective_attempt_defer_reason(
        self,
        objective: str,
        *,
        now: float | None = None,
    ) -> str:
        key = self._objective_key(objective)
        if not key:
            return "no_objective"
        if key != self._objective_attempt_key:
            self._objective_attempt_key = key
            self._objective_next_attempt_at = 0.0
        if self._objective_attempt_inflight == key:
            return "objective_inflight"
        current = time.monotonic() if now is None else float(now)
        if current < self._objective_next_attempt_at:
            return "objective_cooldown"
        return ""

    def _begin_objective_attempt(self, objective: str) -> None:
        key = self._objective_key(objective)
        self._objective_attempt_key = key
        self._objective_attempt_inflight = key

    def _finish_objective_attempt(
        self,
        objective: str,
        *,
        retry_after_s: float,
        now: float | None = None,
    ) -> None:
        key = self._objective_key(objective)
        if self._objective_attempt_inflight == key:
            self._objective_attempt_inflight = ""
        if self._objective_attempt_key == key:
            current = time.monotonic() if now is None else float(now)
            self._objective_next_attempt_at = current + max(0.0, float(retry_after_s))

    def _bootstrap_phases(self):
        """Initialize and register the 8 core cognitive phases."""
        from .container import get_container
        container = get_container()
        kernel = container.get("aura_kernel", default=None)

        for name, phase in instantiate_legacy_runtime_phases(
            kernel or container,
            include_executive_closure=True,
        ):
            self.register_phase(name, phase)

        from core.supervisor.registry import get_task_registry
        self.registry = get_task_registry()
        logger.info("📋 MindTick: TaskRegistry heartbeat wired.")

    def _install_loop_done_callback(self, task: asyncio.Task | None, *, name: str) -> None:
        """Attach a restart hook to the supervised MindTick loop."""
        if task is None:
            return

        def _on_done(done_task: asyncio.Task) -> None:
            if not self._running or is_shutdown_requested():
                return
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                exc = None
            except _MIND_BOUNDARY_ERRORS as err:
                exc = err
            if exc is not None:
                logger.error(
                    "MindTick supervised loop %s exited unexpectedly: %s",
                    name,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            else:
                logger.warning("MindTick supervised loop %s exited while runtime was still active.", name)
            self._attempt_liveness_repair()

        task.add_done_callback(_on_done)

    def register_phase(self, name: str, phase_fn: PhaseCallable):
        """Register a new cognitive phase to execute every tick."""
        self.phases.append((name, phase_fn))
        logger.info("🧠 MindTick: Registered phase '%s'", name)

    def reload_phases(self):
        """Dynamically reloads all phase modules and re-bootstraps the pipeline."""
        logger.info("🔄 MindTick: Hot-reloading cognitive phases...")
        import sys
        
        # Clear cached phase modules to force fresh import
        phase_modules = [m for m in sys.modules if m.startswith("core.phases.")]
        for mod_name in phase_modules:
            del sys.modules[mod_name]
        
        if "core.phases" in sys.modules:
            del sys.modules["core.phases"]
            
        # Re-bootstrap
        self.phases = []
        self._bootstrap_phases()
        logger.info("✅ MindTick: Hot-reload complete. %d phases active.", len(self.phases))

    def _background_reasoning_pause_reason(self, state: AuraState | None = None) -> str:
        # Yield the background kernel tick while the foreground conversation lane
        # holds the model. Otherwise a soak's back-to-back turns saturate the
        # generation gate, this tick blocks on kernel.tick() for minutes, the
        # iteration never marks progress, and mind_tick is falsely declared dead
        # → runtime DEGRADED → GUI reverts to "Connecting to runtime"
        # (observed 2026-07-06).
        try:
            from core.runtime.backpressure import foreground_inference_active

            if foreground_inference_active():
                return "foreground_inference_active"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        try:
            from core.runtime.background_policy import (
                IDLE_COGNITION_BACKGROUND_POLICY,
                background_activity_reason,
            )

            reason = background_activity_reason(
                self.orchestrator,
                profile=IDLE_COGNITION_BACKGROUND_POLICY,
                # [FIX] The old threshold of 0.25 was too aggressive: transient
                # errors blocked all cognitive work, preventing
                # _last_successful_tick_at from advancing, which triggered the
                # health contract to report mind_tick as an IMPORTANT-tier
                # failure — cascading to orchestrator UNHEALTHY, meta-evolution
                # abort, and chat repair failures.  0.70 keeps the safety gate
                # active for genuine cascading failures while letting the tick
                # survive normal transient degradation.
                max_failure_pressure=0.70,
                allow_no_user_anchor=False,
            )
            if reason:
                return reason
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(exc)
            logger.debug("MindTick background policy probe failed: %s", exc)

        try:
            flow = getattr(self.orchestrator, "_flow_controller", None)
            if flow is not None:
                snap = flow.snapshot(self.orchestrator)
                if float(getattr(snap, "lag_seconds", 0.0) or 0.0) >= 0.15:
                    return "event_loop_lag"
                if bool(getattr(snap, "overloaded", False)) or float(getattr(snap, "load", 0.0) or 0.0) >= 0.65:
                    return "flow_overload"
                if str(getattr(snap, "governor_mode", "") or "").upper() == "DEGRADED_CORE_ONLY":
                    return "degraded_core_only"
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(exc)
            logger.debug("MindTick background flow probe failed: %s", exc)

        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(exc)
            logger.debug("MindTick router pressure probe failed: %s", exc)

        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin="mind_tick") or "").strip()
                if reason:
                    return reason
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(exc)
            logger.debug("MindTick gate pressure probe failed: %s", exc)

        objective = str(getattr(getattr(state, "cognition", None), "current_objective", "") or "").strip() if state is not None else ""
        active_goals = list(getattr(getattr(state, "cognition", None), "active_goals", []) or []) if state is not None else []
        last_user = float(getattr(self.orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
        recent_user_context = last_user > 0.0 and (time.time() - last_user) <= 180.0
        if not objective and not active_goals and not recent_user_context:
            return "no_reasoning_context"

        return ""

    async def start(self):
        """Start the cognitive rhythm."""
        if self._running:
            return
        
        from infrastructure.watchdog import get_watchdog
        get_watchdog().register_component("mind_tick", timeout=30.0)
        
        self._running = True
        self._started_at = time.time()
        self._active_tick_started_at = 0.0
        self._active_tick_stage = "starting"
        self._mark_loop_progress("start")
        self._task = _schedule_mind_task(self._run_loop(), name="mind_tick.run_loop")
        if self._task is None:
            self._running = False
            logger.warning("💓 MindTick: Cognitive rhythm scheduling deferred.")
            return
        try:
            self._owner_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._owner_loop = None
        self._install_loop_done_callback(self._task, name="mind_tick.run_loop")
        try:
            ServiceContainer.register_instance("mind_tick", self, required=False)
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(
                exc,
                action="continued after authoritative MindTick service publish failed",
                severity="warning",
            )
        logger.info("💓 MindTick: Cognitive rhythm started.")

    def is_alive(self) -> bool:
        """Report whether the supervised loop is running and progressing.

        A QUERY. It records what it observes and changes nothing: a health
        probe that restarts the service it is inspecting turns every reader —
        a dashboard, a contract sweep, a status endpoint — into an actuator,
        and the reader has no idea it did anything. ``ensure_alive`` is the
        one that repairs, and callers that want self-healing ask for it.
        """
        return self._liveness(repair=False)

    def ensure_alive(self) -> bool:
        """Report liveness AND repair a lost loop, returning the result.

        This is what the old ``is_alive`` did to every caller. It is a
        deliberate action now, taken by whoever wants it.
        """
        return self._liveness(repair=True)

    def _liveness(self, *, repair: bool) -> bool:
        task_alive = bool(self._task and not self._task.done())
        if not self._running or not task_alive:
            if not repair or not self._attempt_liveness_repair():
                return False
            task_alive = bool(self._task and not self._task.done())
        if (
            float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0) <= 0.0
            and float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0) <= 0.0
        ):
            if int(getattr(self, "_consecutive_loop_failures", 0) or 0) >= 3:
                if not repair or not self._attempt_liveness_repair(
                    reason="repeated loop failures before first progress"
                ):
                    return False
                return bool(self._task and not self._task.done())
            return bool(self._started_at and (time.time() - self._started_at) <= 180.0)
        now = time.time()
        freshest_progress = max(
            float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0),
            float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0),
        )
        stale_progress_s = self._float_env(
            "AURA_MIND_TICK_STALE_PROGRESS_S", DEFAULT_STALE_PROGRESS_S
        )
        if freshest_progress > 0.0 and (now - freshest_progress) <= stale_progress_s:
            return True
        active_started = float(getattr(self, "_active_tick_started_at", 0.0) or 0.0)
        hard_stall_s = self._float_env(
            "AURA_MIND_TICK_HARD_STALL_S", DEFAULT_HARD_STALL_S
        )
        if (
            task_alive
            and active_started > 0.0
            and (now - active_started) <= hard_stall_s
            and int(getattr(self, "_consecutive_loop_failures", 0) or 0) < 3
        ):
            return True
        if int(getattr(self, "_consecutive_loop_failures", 0) or 0) >= 3:
            if not repair or not self._attempt_liveness_repair(
                reason="repeated loop failures without fresh progress"
            ):
                return False
            return bool(self._task and not self._task.done())
        if task_alive:
            age = now - freshest_progress if freshest_progress > 0.0 else now - float(getattr(self, "_started_at", now) or now)
            # Name the wedge: the contract line 'is_alive() returned False'
            # told an operator nothing for two hours. Every stale-progress
            # verdict now records WHERE the rhythm is stuck.
            record_degraded_event(
                "mind_tick",
                "rhythm_stale",
                detail=(
                    f"stage={getattr(self, '_active_tick_stage', '?')} "
                    f"progress_age={age:.0f}s"
                ),
                severity="warning",
                classification="background_degraded",
                context={
                    "stage": str(getattr(self, "_active_tick_stage", "") or ""),
                    "progress_age_s": round(age, 1),
                },
            )
            if repair:
                self._attempt_liveness_repair(
                    reason=f"stale progress for {age:.1f}s",
                    cancel_existing=True,
                )
        return False


    def _schedule_restart_after(self, old_task: Any, *, reason: str = "") -> None:
        """Start the replacement loop only once the old one has actually ended.

        The whole point of the fix for CP126 e98446be: there must never be
        two ``_run_loop`` coroutines alive at the same time. A done-callback
        is the only way to know the cancelled one has really unwound,
        because ``cancel()`` returns long before that happens.
        """

        def _restart(finished: Any) -> None:
            try:
                if is_shutdown_requested():
                    self._repair_pending = False
                    return
                self._consecutive_loop_failures = 0
                self._running = True
                self._started_at = time.time()
                self._active_tick_started_at = 0.0
                self._active_tick_stage = "repair_after_cancel"
                self._last_successful_tick_at = 0.0
                self._mark_loop_progress("repair_after_cancel")
                count = int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
                name = f"mind_tick.run_loop.recovered.{count}"
                self._task = _schedule_mind_task(self._run_loop(), name=name)
                if self._task is None:
                    self._running = False
                    self._repair_pending = False
                    record_degraded_event(
                        "mind_tick",
                        "liveness_repair_failed",
                        detail="replacement loop could not be scheduled after cancel",
                        severity="error",
                        classification="background_degraded",
                    )
                    return
                self._install_loop_done_callback(self._task, name=name)
                self._liveness_repair_count = count
                self._repair_pending = False
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair",
                    detail=reason or "stale loop cancelled and replaced after it unwound",
                    severity="warning",
                    classification="runtime_recovered",
                    context={"repair_count": count},
                )
                logger.warning(
                    "💓 MindTick: stale loop unwound; replacement started (repair %d).",
                    count,
                )
            except _MIND_BOUNDARY_ERRORS as exc:
                self._repair_pending = False
                _record_mind_degradation(
                    exc,
                    action="MindTick replacement loop could not start after the stale loop ended",
                    severity="critical",
                )

        try:
            old_task.add_done_callback(_restart)
        except _MIND_BOUNDARY_ERRORS as exc:
            self._repair_pending = False
            _record_mind_degradation(
                exc,
                action="could not chain the MindTick restart to the cancelled loop",
                severity="critical",
            )

    def _attempt_liveness_repair(self, *, reason: str = "", cancel_existing: bool = False) -> bool:
        """Restart the supervised cognitive loop when a live runtime loses it.

        This is deliberately narrow: it only repairs a runtime that already
        marked MindTick running or has a finished task, it rate-limits repair
        attempts, and it never runs during coordinated shutdown. The goal is to
        turn the health-contract failure into an internal recovery attempt
        instead of leaving the desktop path degraded until the next user prompt.
        """
        if is_shutdown_requested():
            return False
        if not self._running and self._task is None:
            return False
        if not callable(getattr(self, "_run_loop", None)):
            return False
        now = time.monotonic()
        if now - float(getattr(self, "_last_liveness_repair_at", 0.0) or 0.0) < 10.0:
            return False
        self._last_liveness_repair_at = now

        existing_task = getattr(self, "_task", None)
        if cancel_existing and existing_task is not None and not existing_task.done():
            # CP126 e98446be: this cancelled and immediately started a new
            # loop. `cancel()` only REQUESTS cancellation — the old
            # coroutine keeps running until it next reaches an await point,
            # so both loops ran concurrently, each mutating the same state
            # object and committing over the other. A repair that produces
            # two minds is worse than the stall it was repairing.
            #
            # The restart is now chained to the old task's completion, so
            # there is exactly one loop at every instant. If the old task
            # will not die, no new one is started and that is recorded —
            # a stuck loop must not be masked by a second one.
            try:
                existing_task.cancel()
            except _MIND_BOUNDARY_ERRORS as exc:
                _record_mind_degradation(
                    exc,
                    action="continued MindTick liveness repair after stale loop cancel failed",
                    severity="warning",
                )
            else:
                self._repair_pending = True
                self._schedule_restart_after(existing_task, reason=reason)
                return True

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Health checks call is_alive() from plain threads, where task
            # creation is impossible — so a dead loop could never be revived
            # by the pulse that detected it (observed live 2026-07-05: the
            # runtime sat DEGRADED for 84 minutes with repair machinery
            # present). Hand the repair to the owning loop instead.
            owner_loop = getattr(self, "_owner_loop", None)
            if owner_loop is not None and not owner_loop.is_closed():
                def _threadsafe_repair() -> None:
                    self._consecutive_loop_failures = 0
                    self._running = True
                    self._started_at = time.time()
                    self._active_tick_started_at = 0.0
                    self._active_tick_stage = "threadsafe_repair"
                    self._last_successful_tick_at = 0.0
                    self._mark_loop_progress("threadsafe_repair")
                    self._task = _schedule_mind_task(
                        self._run_loop(), name="mind_tick.run_loop.recovered.threadsafe"
                    )
                    self._install_loop_done_callback(
                        self._task,
                        name="mind_tick.run_loop.recovered.threadsafe",
                    )
                    self._liveness_repair_count = (
                        int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
                    )
                    self._repair_pending = False
                    logger.warning("💓 MindTick: loop revived via owning-loop repair.")

                self._repair_pending = True
                owner_loop.call_soon_threadsafe(_threadsafe_repair)
                logger.info("MindTick repair scheduled onto owning loop from thread.")
                # CP126 c76abf56: this returned False immediately, so health
                # read "unhealthy" while a recovery was already in flight and
                # nothing recorded that one had been started. The pending flag
                # is the receipt; get_health_status reports it.
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair_scheduled",
                    detail=reason or "repair handed to the owning loop from a probe thread",
                    severity="warning",
                    classification="runtime_recovering",
                    context={"stage": str(getattr(self, "_active_tick_stage", "") or "")},
                )
            else:
                # A repair that silently cannot run is how a runtime sits
                # DEGRADED for hours with 'repair machinery present'. Say so.
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair_unreachable",
                    detail=(
                        "no usable owner loop from thread context"
                        if owner_loop is None
                        else "owner loop closed"
                    ),
                    severity="error",
                    classification="background_degraded",
                    context={"stage": str(getattr(self, "_active_tick_stage", "") or "")},
                )
            return False

        finished_error = None
        if self._task and self._task.done():
            try:
                finished_error = self._task.exception()
            except (asyncio.CancelledError, RuntimeError, AttributeError) as exc:
                finished_error = exc

        self._consecutive_loop_failures = 0
        self._running = True
        self._started_at = time.time()
        self._active_tick_started_at = 0.0
        self._active_tick_stage = "repair"
        self._last_successful_tick_at = 0.0
        self._mark_loop_progress("repair")
        self._task = _schedule_mind_task(
            self._run_loop(),
            name=f"mind_tick.run_loop.recovered.{int(getattr(self, '_liveness_repair_count', 0) or 0) + 1}",
        )
        if self._task is None:
            self._running = False
            return False
        self._install_loop_done_callback(
            self._task,
            name=f"mind_tick.run_loop.recovered.{int(getattr(self, '_liveness_repair_count', 0) or 0) + 1}",
        )

        self._liveness_repair_count = int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
        detail = (
            f"{type(finished_error).__name__}: {finished_error}"
            if finished_error is not None
            else (reason or "liveness probe found MindTick loop missing, stopped, or stale")
        )
        logger.warning("💓 MindTick: Repaired stalled cognitive rhythm (%s).", detail)
        record_degraded_event(
            "mind_tick",
            "liveness_repair",
            detail=detail,
            severity="warning",
            classification="runtime_recovered",
            context={"repair_count": self._liveness_repair_count},
            exc=finished_error if isinstance(finished_error, BaseException) else None,
        )
        return True

    def get_health_status(self) -> dict[str, Any]:
        """Expose causal loop progress without treating heartbeat transport as health."""
        return {
            "healthy": self.is_alive(),
            "running": self._running,
            "task_alive": bool(self._task and not self._task.done()),
            "tick_count": int(getattr(self, "_tick_count", 0) or 0),
            "consecutive_failures": int(getattr(self, "_consecutive_loop_failures", 0) or 0),
            "last_successful_tick_at": float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0),
            "last_loop_progress_at": float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0),
            "last_progress_label": str(getattr(self, "_last_progress_label", "") or ""),
            "active_tick_started_at": float(getattr(self, "_active_tick_started_at", 0.0) or 0.0),
            "active_tick_stage": str(getattr(self, "_active_tick_stage", "") or ""),
            "liveness_repair_count": int(getattr(self, "_liveness_repair_count", 0) or 0),
            # CP126 c76abf56: a recovery already in flight used to be
            # invisible, so the surface said "unhealthy" with no indication
            # that anything was being done about it. "Recovering" and
            # "broken and unattended" are different operational states.
            "repair_pending": bool(getattr(self, "_repair_pending", False)),
        }

    async def stop(self):
        """Stop the cognitive rhythm."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("MindTick: Shutdown requested.")
            except _MIND_BOUNDARY_ERRORS as exc:
                _record_mind_degradation(
                    exc,
                    action="continued shutdown after MindTick background loop failed while draining",
                    severity="warning",
                )
                logger.debug("MindTick: Background loop ended during shutdown: %s", exc)
        logger.info("🛑 MindTick: Cognitive rhythm stopped.")

    async def _run_loop(self):
        """The main execution loop for the cognitive rhythm."""
        while self._running:
            sleep_time_override: float | None = None
            try:
                start_time = asyncio.get_running_loop().time()
                self._active_tick_started_at = time.time()
                self._active_tick_stage = "tick_start"
                self._mark_loop_progress("tick_start")
                
                # 1. Get the latest state
                from infrastructure.watchdog import get_watchdog
                get_watchdog().heartbeat("mind_tick")
                
                # Bounded: the rhythm loop must never hand its liveness to a
                # dependency. A wedged state read parks the whole tick — and
                # with it every downstream organ — until someone notices.
                try:
                    state = await asyncio.wait_for(
                        self.orchestrator.state_repo.get_current(), timeout=30.0
                    )
                except TimeoutError:
                    record_degraded_event(
                        "mind_tick",
                        "tick_stage_timeout",
                        detail="state_repo.get_current>30s",
                        severity="warning",
                        classification="background_degraded",
                        context={"stage": "state_load", "tick_count": self._tick_count},
                    )
                    # Liveness is preserved by the loop-progress marker (which
                    # is_alive() reads via max(success, progress)); do NOT also
                    # advance _last_successful_tick_at — no tick processed state,
                    # so the "completed tick" marker must stay honest.
                    self._mark_loop_progress("state_load_timeout_yield")
                    sleep_time_override = 5.0
                    continue
                self._mark_loop_progress("state_loaded" if state else "state_missing")
                if not state:
                    self._missing_state_streak += 1
                    base_interval = max(0.5, TICK_INTERVALS.get(self.mode, 1.0))
                    sleep_time_override = min(
                        self._max_missing_state_backoff,
                        base_interval * (2 ** min(self._missing_state_streak - 1, 3)),
                    )
                    now = time.monotonic()
                    should_log = (
                        self._missing_state_streak == 1
                        or (now - self._last_missing_state_log) >= 5.0
                    )
                    if should_log:
                        logger.warning(
                            "💓 MindTick: No current state found. Deferring tick for %.1fs (streak=%d).",
                            sleep_time_override,
                            self._missing_state_streak,
                        )
                        self._last_missing_state_log = now
                    continue
                if self._missing_state_streak:
                    logger.info(
                        "💓 MindTick: State became available after %d deferred tick(s).",
                        self._missing_state_streak,
                    )
                    self._missing_state_streak = 0

                # ── UNIFIED WILL: Ensure Will is started and refresh identity ──
                try:
                    from core.will import get_will
                    _will = get_will()
                    if not _will._started:
                        self._active_tick_stage = "will_start"
                        self._mark_loop_progress("will_start")
                        await _will.start()
                except _MIND_BOUNDARY_ERRORS as _will_boot:
                    _record_mind_degradation(_will_boot)
                    if self._tick_count <= 1:
                        logger.debug("MindTick: Unified Will boot deferred: %s", _will_boot)

                # ── WORLD STATE: Update telemetry every tick ──
                try:
                    from core.world_state import get_world_state
                    self._active_tick_stage = "world_state"
                    # psutil's CPU, memory, battery and thermal probes are
                    # blocking syscalls behind a lock. On the event loop they
                    # stall every other coroutine in the process, and a slow
                    # sensor read — a thermal probe on a throttling machine —
                    # stalls the whole rhythm. Off-thread and bounded: this is
                    # telemetry, and stale telemetry beats a stopped heartbeat.
                    await asyncio.wait_for(
                        asyncio.to_thread(get_world_state().update), timeout=5.0
                    )
                    self._mark_loop_progress("world_state")
                except TimeoutError:
                    record_degraded_event(
                        "mind_tick",
                        "tick_stage_timeout",
                        detail="world_state.update>5s",
                        severity="warning",
                        classification="background_degraded",
                        context={"stage": "world_state", "tick_count": self._tick_count},
                    )
                    self._mark_loop_progress("world_state_timeout_yield")
                except _MIND_BOUNDARY_ERRORS as exc:
                    _record_mind_degradation(exc)
                    logger.debug("MindTick: World state update failed: %s", exc)

                # ── LLM HEALTH: Proactive recovery for ALL tiers every 10 ticks ──
                #
                # Never while the conversation lane holds the model. The sweep
                # probes every tier and its recovery paths spawn and load
                # workers, so running it during a turn puts a maintenance probe
                # in contention with the person waiting for an answer — on the
                # one process that serves them. The bound below keeps a wedge
                # from stopping the rhythm; it does nothing about contention,
                # because a probe that finishes inside its budget has still
                # taken the lane. The kernel tick already yields on this exact
                # signal; the sweep that can load a 32B did not.
                health_pause = (
                    self._background_reasoning_pause_reason()
                    if self._tick_count % 10 == 0
                    else ""
                )
                if health_pause:
                    self._mark_loop_progress(f"llm_health_deferred:{health_pause}")
                if self._tick_count % 10 == 0 and not health_pause:
                    try:
                        gate = ServiceContainer.get("inference_gate", default=None)
                        if gate and hasattr(gate, "ensure_all_tiers_healthy"):
                            self._active_tick_stage = "llm_health"
                            self._mark_loop_progress("llm_health")
                            # Bounded: tier health can trigger recovery paths
                            # that spawn/probe model workers (minutes under
                            # load). Observed live 2026-07-07: the tick loop
                            # sat wedged for 40+ minute stretches on an idle
                            # instance, flagging mind_tick dead while the rest
                            # of the organism was healthy. The rhythm yields
                            # and retries next cycle instead of wedging.
                            tier_statuses = await asyncio.wait_for(
                                gate.ensure_all_tiers_healthy(), timeout=45.0
                            )
                            self._mark_loop_progress("llm_health_done")
                            dead_tiers = [t for t, s in tier_statuses.items() if s == "dead"]
                            if dead_tiers and self._tick_count % 30 == 0:
                                if _dead_tiers_are_policy_deferred_cortex(gate, dead_tiers):
                                    now = time.monotonic()
                                    if now - self._last_deferred_cortex_health_log_at > 300.0:
                                        logger.info(
                                            "LLM health: Cortex is cold by desktop prewarm policy; "
                                            "foreground demand will warm the lane."
                                        )
                                        self._last_deferred_cortex_health_log_at = now
                                    else:
                                        logger.debug(
                                            "MindTick: deferred cold-Cortex health notice coalesced."
                                        )
                                    # A cold Cortex by prewarm policy is not a
                                    # fault, so the ESCALATION is skipped. This
                                    # was `continue`, which skipped the whole
                                    # tick: every thirtieth tick under an
                                    # entirely normal condition abandoned every
                                    # phase and the commit, discarding the state
                                    # work the tick had already done.
                                else:
                                    logger.warning("LLM health: dead tiers=%s", dead_tiers)
                                    # Report persistent dead tiers to incident manager
                                    try:
                                        from core.resilience.incident_manager import (
                                            get_incident_manager,
                                        )
                                        get_incident_manager().report(
                                            source="mind_tick",
                                            title=f"LLM tiers dead: {', '.join(dead_tiers)}",
                                            detail=f"Dead tiers detected at tick {self._tick_count}",
                                            severity="warning",
                                        )
                                    except _MIND_BOUNDARY_ERRORS as exc:
                                        _record_mind_degradation(exc)
                                        logger.debug("MindTick: LLM health incident report failed: %s", exc)
                        elif gate and hasattr(gate, "_ensure_cortex_recovery"):
                            await asyncio.wait_for(gate._ensure_cortex_recovery(), timeout=45.0)
                    except TimeoutError:
                        # The health sweep wedged past its budget — name the
                        # stage, count the yield as rhythm progress, move on.
                        record_degraded_event(
                            "mind_tick",
                            "tick_stage_timeout",
                            detail="ensure_all_tiers_healthy>45s",
                            severity="warning",
                            classification="background_degraded",
                            context={"stage": "llm_health", "tick_count": self._tick_count},
                        )
                        # Loop-progress marker keeps the rhythm alive without
                        # fabricating a completed tick (is_alive reads both).
                        self._mark_loop_progress("llm_health_timeout_yield")
                    except _MIND_BOUNDARY_ERRORS as exc:
                        _record_mind_degradation(exc)
                        logger.debug("MindTick: LLM health recovery check failed: %s", exc)

                # ── RESOURCE GOVERNOR: Check throttle state ──
                if self._tick_count % 10 == 0:
                    try:
                        from core.resource.resource_governor import get_resource_governor
                        gov = get_resource_governor()
                        if gov.is_throttled():
                            if self._tick_count % 30 == 0:
                                logger.info(
                                    "💓 MindTick: Resource governor throttled — "
                                    "deferring heavy background work."
                                )
                    except _MIND_BOUNDARY_ERRORS as exc:
                        _record_mind_degradation(exc)
                        logger.debug("MindTick: Resource governor check failed: %s", exc)

                # ── BINDING ENGINE: Run coherence tick before phases ──
                _coherence_report = None
                _bg_pause_pre = self._background_reasoning_pause_reason(state)
                if not _bg_pause_pre:
                    try:
                        from core.coherence.binding_engine import get_binding_engine
                        _binding = get_binding_engine()
                        self._active_tick_stage = "binding_engine"
                        self._mark_loop_progress("binding_engine")
                        _coherence_report = await asyncio.wait_for(_binding.tick(state), timeout=3.0)
                        self._mark_loop_progress("binding_engine_done")
                    except _MIND_BOUNDARY_ERRORS as _be:
                        _record_mind_degradation(_be)
                        logger.debug("MindTick: BindingEngine tick skipped: %s", _be)
                else:
                    if self._tick_count % 30 == 0:
                        logger.debug("MindTick: BindingEngine deferred (%s).", _bg_pause_pre)

                # ── GOAL-DRIVEN INITIATIVE GENERATION ────────────────────────
                # If there are active goals but no pending initiatives, generate
                # an initiative from the highest-priority goal. This is what makes
                # Aura proactively pursue her goals during idle background ticks
                # instead of only reacting to user input.
                if not state.cognition.current_objective and not state.cognition.pending_initiatives:
                    if self._tick_count % 10 == 0:  # Check every 10 ticks, not every tick
                        try:
                            goal_engine = ServiceContainer.get("goal_engine", default=None)
                            if goal_engine and hasattr(goal_engine, "get_active_goals"):
                                active = goal_engine.get_active_goals(
                                    limit=3,
                                    include_external=False,
                                    actionable_only=True,
                                )
                                for goal in active:
                                    objective = str(goal.get("objective") or goal.get("name") or "")
                                    if not objective:
                                        continue
                                    status = str(goal.get("status", "")).lower()
                                    if status not in ("queued", "in_progress"):
                                        continue
                                    # Don't re-promote if we just tried this goal
                                    if objective == self._last_initiative_goal:
                                        continue
                                    # Use governed proposal path (constitutional compliance)
                                    from core.runtime.proposal_governance import (
                                        propose_governed_initiative_to_state,
                                    )
                                    state, _ = await propose_governed_initiative_to_state(
                                        state,
                                        objective,
                                        source="goal_engine",
                                        urgency=float(goal.get("priority", 0.5)),
                                        triggered_by="proactive_goal_pursuit",
                                    )
                                    break  # Only inject one goal per cycle
                        except _MIND_BOUNDARY_ERRORS as _ge:
                            _record_mind_degradation(_ge)
                            logger.debug("MindTick: goal-driven initiative generation failed: %s", _ge)

                # ── INITIATIVE ARBITRATION: Replace FIFO with scored selection ──
                if not state.cognition.current_objective and state.cognition.pending_initiatives:
                    initiative_pause = ""
                    try:
                        from core.runtime.background_policy import (
                            IDLE_COGNITION_BACKGROUND_POLICY,
                            background_activity_reason,
                        )

                        initiative_pause = background_activity_reason(
                            self.orchestrator,
                            profile=IDLE_COGNITION_BACKGROUND_POLICY,
                            max_failure_pressure=0.25,
                        )
                        if initiative_pause:
                            logger.debug("MindTick: initiative promotion paused: %s", initiative_pause)
                    except _MIND_BOUNDARY_ERRORS as exc:
                        _record_mind_degradation(exc)
                        logger.debug("MindTick: initiative background-policy probe failed: %s", exc)

                    # Cooldown: don't re-promote the same initiative within 30s
                    top_goal = ""
                    if state.cognition.pending_initiatives:
                        top_init = state.cognition.pending_initiatives[0]
                        top_goal = top_init.get("goal", "") if isinstance(top_init, dict) else str(top_init)
                    now_init = time.time()
                    cooldown_active = (
                        top_goal == self._last_initiative_goal
                        and (now_init - self._last_initiative_time) < self._initiative_cooldown
                    )
                    if cooldown_active:
                        logger.debug("MindTick: initiative promotion cooling down for %s.", top_goal[:80])
                    if not initiative_pause and not cooldown_active:
                        from core.consciousness.executive_authority import get_executive_authority

                        authority = get_executive_authority(self.orchestrator)
                        state, initiative, decision = await authority.promote_next_initiative(state, source="mind_tick")
                        if initiative:
                            self._last_initiative_goal = initiative.get("goal", "")
                            self._last_initiative_time = now_init
                            logger.info(
                                "⚡ MindTick: Promoted initiative via executive authority: %s... (%s)",
                                str(initiative.get("goal", ""))[:50],
                                decision.get("reason", "initiative_promoted"),
                            )
                
                # 2. Prediction Step (Active Inference)
                prediction = None
                prediction_interval = config.autonomous_thought_interval_s if not config.skeletal_mode else 300.0
                if hasattr(self, 'predictive_engine') and self.predictive_engine:
                    breaker = self.breakers["prediction"]
                    reasoning_pause = self._background_reasoning_pause_reason(state)
                    if breaker.is_available and not reasoning_pause and (time.time() - self._last_prediction_time > prediction_interval):
                        try:
                            self._active_tick_stage = "predictive_engine"
                            self._mark_loop_progress("predictive_engine")
                            prediction = await asyncio.wait_for(
                                self.predictive_engine.predict(state, prefer_tier="tertiary", is_background=True), 
                                timeout=30.0
                            )
                            self._mark_loop_progress("predictive_engine_done")
                            self._last_prediction_time = time.time()
                            breaker.record_success()
                            logger.info("🔮 MindTick: Predicted: %s...", f"{prediction.content[:50]}")
                        except _MIND_BOUNDARY_ERRORS as e:
                            detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                            logger.warning("⚠️ MindTick: Prediction failed/stalled: %s", detail)
                            breaker.record_failure()
                            prediction = None
                    elif reasoning_pause and (self._tick_count % 20 == 0):
                        logger.debug("💓 MindTick: Skipping predictive background reasoning (%s).", reasoning_pause)

                if hasattr(self, 'trajectory_predictor') and self.trajectory_predictor:
                    reasoning_pause = self._background_reasoning_pause_reason(state)
                    if not reasoning_pause and time.time() - self._last_trajectory_time > 60.0: # Every minute
                        self._active_tick_stage = "trajectory_predictor"
                        self._mark_loop_progress("trajectory_predictor")
                        _schedule_mind_task(
                            self.trajectory_predictor.predict_path(
                                state.cognition.current_objective or "General Processing",
                                state,
                            ),
                            name="MindTick.trajectory_predict",
                        )
                        self._last_trajectory_time = time.time()

                # 3. Execute all registered phases within a Mycelial rooted_flow.
                mycelium = ServiceContainer.get("mycelium", default=None)
                
                metadata = TickMetadata(
                    tick_id=self._tick_count,
                    mode=self.mode,
                    start_time=time.time()
                )

                current_state = state

                async def execute_tick(tick_metadata=metadata):
                    nonlocal current_state
                    # ── CONSTITUTIONAL UNIFICATION ──────────────────────────────
                    # MindTick is the heartbeat; the kernel is the sole authority.
                    # We collect context above (binding, initiatives, prediction),
                    # then delegate to kernel.tick() for all phase execution.
                    # MindTick's own phases are fallback-only for early boot.
                    # ───────────────────────────────────────────────────────────
                    objective = str(current_state.cognition.current_objective or "").strip()
                    current_origin = str(getattr(current_state.cognition, "current_origin", "") or "").strip().lower()
                    quiet_until = float(getattr(self.orchestrator, "_foreground_user_quiet_until", 0.0) or 0.0)
                    try:
                        from core.continuity import _is_generic_continuity_reentry_goal

                        if _is_generic_continuity_reentry_goal(objective):
                            current_state.cognition.current_objective = None
                            current_state.cognition.pending_initiatives = [
                                item for item in list(getattr(current_state.cognition, "pending_initiatives", []) or [])
                                if not _is_generic_continuity_reentry_goal(
                                    item.get("goal", "") if isinstance(item, dict) else str(item)
                                )
                            ]
                            logger.debug("💓 MindTick: Cleared generic continuity re-entry objective.")
                            return current_state
                    except _MIND_BOUNDARY_ERRORS as exc:
                        _record_mind_degradation(exc)
                        logger.debug("MindTick continuity objective scrub failed: %s", exc)
                    if not objective:
                        return current_state
                    if current_origin in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external", "test", "benchmark"}:
                        logger.debug("💓 MindTick: Skipping background tick for foreground-owned objective from origin=%s.", current_origin)
                        return current_state
                    if quiet_until > time.time():
                        logger.debug("💓 MindTick: Skipping background tick during foreground quiet window.")
                        return current_state

                    # Try kernel-sovereign path first
                    kernel = ServiceContainer.get("aura_kernel", default=None)
                    kernel_status = getattr(kernel, "status", None) if kernel else None
                    kernel_live = bool(
                        kernel
                        and hasattr(kernel, "tick")
                        and (
                            getattr(kernel, "_running", False)
                            or getattr(kernel_status, "running", False)
                            or int(getattr(kernel_status, "cycle_count", 0) or 0) > 0
                        )
                    )
                    if kernel and hasattr(kernel, "tick") and kernel_live:
                        # If a user message is already waiting
                        # for the kernel lock, skip this background tick entirely.
                        # The user's tick will run as soon as the lock is free.
                        if getattr(kernel, "_user_priority_pending", None) and kernel._user_priority_pending.is_set():
                            logger.debug("💓 MindTick: Skipping background tick — user priority message pending.")
                            return current_state
                        # If the loop is already lagging,
                        # don't add a full kernel tick (which invokes the LLM).
                        _bg_pause = self._background_reasoning_pause_reason(current_state)
                        if _bg_pause:
                            if self._tick_count % 30 == 0:
                                logger.debug("💓 MindTick: Deferring background kernel tick (%s).", _bg_pause)
                            self._mark_loop_progress(f"kernel_deferred:{_bg_pause}")
                            return current_state
                        objective_defer = self._objective_attempt_defer_reason(objective)
                        if objective_defer:
                            self._mark_loop_progress(f"kernel_deferred:{objective_defer}")
                            return current_state
                        try:
                            # Bound the background kernel tick. A background
                            # cognition step must never freeze the whole tick
                            # iteration: if it can't finish promptly the model is
                            # contended, so abort and let the loop iterate (which
                            # marks progress and keeps mind_tick reported alive).
                            _bg_tick_timeout = float(
                                os.getenv("AURA_MIND_TICK_KERNEL_TIMEOUT_S", "45")
                            )
                            self._active_tick_stage = "kernel_tick"
                            self._mark_loop_progress("kernel_tick")
                            # Advertise that the cognition lane holds the shared 32B
                            # worker, so slow background phenomenology yields instead
                            # of contending this tick and blowing the tick SLO.
                            from core.runtime.backpressure import primary_inference_lease

                            self._begin_objective_attempt(objective)
                            with primary_inference_lease():
                                entry = await asyncio.wait_for(
                                    kernel.tick(objective, priority=False),
                                    timeout=_bg_tick_timeout,
                                )
                            self._mark_loop_progress("kernel_tick_done")
                            if entry is not None:
                                # Kernel ran successfully — fetch the committed state
                                committed = await self.orchestrator.state_repo.get_current()
                                if committed:
                                    current_state = committed
                                    tick_metadata.phases_executed.append("kernel_sovereign_tick")
                                    logger.debug("💓 MindTick: Kernel sovereign tick completed (cycle %d).", self._tick_count)
                                self._finish_objective_attempt(
                                    objective,
                                    retry_after_s=self._float_env(
                                        "AURA_MIND_TICK_OBJECTIVE_SUCCESS_RETRY_S",
                                        60.0,
                                    ),
                                )
                                return current_state
                            else:
                                self._finish_objective_attempt(objective, retry_after_s=10.0)
                                logger.warning("💓 MindTick: Kernel tick returned None (lock contention?).")
                                record_degraded_event(
                                    "mind_tick",
                                    "kernel_tick_lock_contention",
                                    detail="kernel tick returned None while kernel was live",
                                    severity="warning",
                                    classification="background_degraded",
                                    context={"tick_count": self._tick_count},
                                )
                                return current_state
                        except TimeoutError:
                            self._finish_objective_attempt(objective, retry_after_s=30.0)
                            # Expected backpressure: the model is contended by the
                            # foreground lane. Not a failure — yield and let the
                            # next iteration try. NO hard degradation (that would
                            # climb _consecutive_loop_failures and self-inflict the
                            # false-dead state this bound exists to prevent).
                            if self._tick_count % 30 == 0:
                                logger.debug(
                                    "💓 MindTick: background kernel tick yielded to a contended model."
                                )
                            # A correct yield to a contended/headroom-deferred model
                            # IS forward progress of the organism rhythm — record it
                            # via the loop-progress marker (which is_alive() reads
                            # alongside the completed-tick marker). This preserves
                            # the false-death protection from the 2026-07-06 respawn
                            # cascade WITHOUT fabricating a completed tick: a mind
                            # that is merely memory-backpressured stays alive through
                            # loop progress, while _last_successful_tick_at keeps its
                            # honest "a tick actually processed state" meaning.
                            # Kernel death itself is caught by the separate REQUIRED
                            # kernel probe, not here.
                            self._consecutive_loop_failures = 0
                            self._mark_loop_progress("kernel_tick_timeout_yield")
                            return current_state
                        except _MIND_BOUNDARY_ERRORS as _kt_err:
                            self._finish_objective_attempt(objective, retry_after_s=60.0)
                            if is_shutdown_requested():
                                logger.info(
                                    "💓 MindTick: Kernel tick ended during requested shutdown (%s).",
                                    _kt_err,
                                )
                                return current_state
                            _record_mind_degradation(_kt_err)
                            logger.warning("💓 MindTick: Kernel tick failed (%s).", _kt_err)
                            record_degraded_event(
                                "mind_tick",
                                "kernel_tick_failed",
                                detail=f"{type(_kt_err).__name__}: {_kt_err}",
                                severity="error",
                                classification="background_degraded",
                                context={"tick_count": self._tick_count},
                                exc=_kt_err,
                            )
                            return current_state
                        finally:
                            if self._objective_attempt_inflight == self._objective_key(objective):
                                self._finish_objective_attempt(objective, retry_after_s=10.0)

                    # Once the kernel has booted, degraded self-execution is a
                    # constitutional violation, not a convenience fallback.
                    if kernel_live:
                        self._ticks_without_kernel = 0
                        logger.debug("💓 MindTick: Kernel is live; skipping degraded-mode self-execution.")
                        return current_state

                    # ── DEGRADED MODE: MindTick runs its own phases ──
                    # Only reached when the kernel has not booted. Two pipelines
                    # can execute cognition in this runtime, and which one ran
                    # decides what the answer was made of, so the second one
                    # never runs quietly: every entry is recorded against the
                    # kernel's absence, and after a bounded grace it escalates
                    # rather than settling in. A fallback nobody escalates is
                    # how a kernel that never boots becomes the normal case.
                    self._ticks_without_kernel = int(
                        getattr(self, "_ticks_without_kernel", 0) or 0
                    ) + 1
                    logger.debug("💓 MindTick: Running degraded-mode phase pipeline (kernel unavailable).")
                    if self._ticks_without_kernel == 1 or (
                        self._ticks_without_kernel % MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS == 0
                    ):
                        record_degraded_event(
                            "mind_tick",
                            "second_phase_pipeline_active",
                            detail=(
                                "kernel absent; MindTick is executing cognition on its own "
                                f"phase pipeline (tick {self._ticks_without_kernel} without a kernel)"
                            ),
                            severity=(
                                "warning"
                                if self._ticks_without_kernel
                                < MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS
                                else "error"
                            ),
                            classification="background_degraded",
                            context={
                                "ticks_without_kernel": self._ticks_without_kernel,
                                "tick_count": self._tick_count,
                            },
                        )
                    # Sequential on purpose: each phase takes the state the
                    # previous one returned, so there is nothing to run in
                    # parallel. This was wrapped in `async with
                    # asyncio.TaskGroup():` that never created a task — the
                    # construct named a concurrency that does not exist, and
                    # it was not inert: a TaskGroup collects an exception
                    # escaping its body into an ExceptionGroup, which none of
                    # this method's `except _MIND_BOUNDARY_ERRORS` clauses
                    # match. The decoration was disabling the tick's own
                    # error handling.
                    for name, phase_fn in self.phases:
                        # Relaxed failure threshold for complex phases
                        max_f = 5 if name == "response_generation" else 2
                        breaker = self.phase_breakers.setdefault(name, CircuitBreaker(f"phase_{name}", max_failures=max_f, reset_timeout=60.0))
                            
                        if not breaker.is_available:
                            logger.warning("⚠️ MindTick: Phase '%s' SKIPPED (Circuit Open)", name)
                            continue

                        try:
                            # Per-phase timeouts — adaptive during early boot
                            timeout = self.phase_timeouts.get(name, self.default_timeout)
                            if self._tick_count < 20:
                                if name == "response_generation":
                                    timeout = min(timeout, 60.0)
                                elif name == "cognitive_routing":
                                    timeout = min(timeout, 120.0) 
                                else:
                                    timeout = min(timeout, 10.0)
                            phase_start = time.perf_counter()
                            self._active_tick_stage = f"phase:{name}"
                            self._mark_loop_progress(f"phase:{name}")
                            current_state = await asyncio.wait_for(phase_fn(current_state), timeout=timeout)
                            self._mark_loop_progress(f"phase_done:{name}")
                            phase_duration = time.perf_counter() - phase_start
                                
                            tick_metadata.phases_executed.append(name)
                            tick_metadata.phase_durations[name] = phase_duration
                                
                            if phase_duration > 1.0:
                                logger.warning("🐢 MindTick: Slow phase detected: '%s' took %ss", name, f"{phase_duration:.3f}")
                                
                            breaker.record_success()
                        except TimeoutError:
                            logger.error("🛑 MindTick: Phase '%s' STALLED (timeout). Tripping circuit.", name)
                            breaker.record_failure()
                        except _MIND_BOUNDARY_ERRORS as phase_err:
                            _record_mind_degradation(phase_err)
                            logger.error("❌ MindTick: Phase '%s' failed: %s", name, phase_err)
                            breaker.record_failure()
                                
                        # Auto-reset breakers if system has been stable for 100+ cycles.
                        if self._tick_count % 100 == 0:
                            for b in self.phase_breakers.values():
                                if not b.is_available:
                                    logger.info("♻️ MindTick: Periodic recovery - Resetting circuit for phase %s", b.name)
                                    b.reset()
                    
                    reflex_fallback_used = False
                    if "response_generation" not in tick_metadata.phases_executed:
                        user_origins = ("user", "voice", "admin", "external", "gui", "api", "websocket", "direct", "test", "benchmark")
                        current_origin = getattr(current_state.cognition, "current_origin", None)
                        if current_origin in user_origins:
                            # A placeholder is not an answer, and the previous
                            # version could not be told apart from one: a canned
                            # sentence went into working memory as an ordinary
                            # assistant turn, the objective was completed right
                            # below, and the request was finished without ever
                            # having been answered. Three things are missing from
                            # a holding line and each is restored here — which
                            # phases did not run, that this is a placeholder, and
                            # a continuation so the real answer still arrives.
                            missing = [
                                name
                                for name, _fn in self.phases
                                if name not in tick_metadata.phases_executed
                            ]
                            logger.warning(
                                "🛡️ MindTick: Emergency Fallback — no response_generation; missing phases=%s",
                                missing,
                            )
                            latest = (
                                current_state.cognition.working_memory[-1]
                                if current_state.cognition.working_memory
                                else {}
                            )
                            if latest.get("origin") != "mind_tick_fallback":
                                current_state.cognition.working_memory.append({
                                    "role": "assistant",
                                    "content": "Give me a moment — I'm thinking through something.",
                                    "timestamp": time.time(),
                                    "origin": "mind_tick_fallback",
                                    "ephemeral": True,
                                    # Machine-readable, so a consumer never
                                    # renders this as the reply to the request.
                                    "placeholder": True,
                                    "answers_request": False,
                                    "failure_disclosure": {
                                        "reason": "cognitive_phases_incomplete",
                                        "missing_phases": missing,
                                        "kernel_present": False,
                                        "tick_count": self._tick_count,
                                    },
                                    "continuation": "objective_retained_for_next_tick",
                                })
                                current_state = current_state.derive("reflexive_fallback")
                            reflex_fallback_used = True
                            record_degraded_event(
                                "mind_tick",
                                "reflex_fallback_served",
                                detail=(
                                    "returned a placeholder instead of an answer; "
                                    f"missing phases={missing}"
                                ),
                                severity="error",
                                classification="background_degraded",
                                context={
                                    "missing_phases": missing,
                                    "origin": str(current_origin or ""),
                                    "tick_count": self._tick_count,
                                },
                            )
                        else:
                            logger.debug(
                                "🛡️ MindTick: Skipping reflexive fallback for non-user origin %r.",
                                current_origin,
                            )
                            
                    try:
                        if current_state and not reflex_fallback_used:
                            from core.consciousness.executive_authority import (
                                get_executive_authority,
                            )

                            authority = get_executive_authority(self.orchestrator)
                            current_state, _ = await authority.complete_current_objective(
                                current_state,
                                reason="tick_cycle_complete",
                                source="mind_tick",
                            )
                        elif reflex_fallback_used:
                            # The continuation the placeholder promised. Closing
                            # the objective here would have retired a request the
                            # runtime never answered, and the next tick would find
                            # nothing left to do about it.
                            logger.info(
                                "🛡️ MindTick: Objective retained after reflex fallback; the next tick answers it."
                            )
                    except _MIND_BOUNDARY_ERRORS as e:
                        _record_mind_degradation(e)
                        logger.debug("MindTick: Objective cleanup failed: %s", e)
                    return current_state

                if mycelium:
                    async with mycelium.rooted_flow(
                        source="mind_tick",
                        target="cognitive_phases",
                        activity="cognitive_cycle",
                    ) as flow:
                        current_state = await execute_tick()
                    flow.raise_for_status()
                else:
                    current_state = await execute_tick()
                
                # 4. Bridge to Event Bus (for UI/Observability)
                # Circuit-breaker: after repeated failures, back off to avoid
                # flooding the resilience engine with degradation events.
                if not hasattr(self, "_bus_fail_count"):
                    self._bus_fail_count = 0
                    self._bus_backoff_until_tick = 0

                if self._tick_count < self._bus_backoff_until_tick:
                    pass  # Skip publish during backoff
                else:
                    from core.event_bus import get_event_bus
                    bus = get_event_bus()
                    try:
                        # metadata.duration is not assigned until the end of the
                        # tick (after this publish), so reading it here always
                        # emitted a stale 0.0. Publish the elapsed-so-far
                        # measured from this tick's start instead.
                        _elapsed_so_far = asyncio.get_running_loop().time() - start_time
                        # Wrap in a 5.0s timeout to prevent Redis stalls from blocking the tick.
                        await asyncio.wait_for(bus.publish("aura/events/mind_tick", {
                            "tick_id": self._tick_count,
                            "mode": self.mode.value,
                            "phases": metadata.phases_executed,
                            "durations": metadata.phase_durations,
                            "total_duration": _elapsed_so_far,
                            "timestamp": time.time()
                        }), timeout=5.0)
                        # Reset on success
                        if self._bus_fail_count > 0:
                            logger.info("⚠️ MindTick: EventBus publish recovered after %d failures.", self._bus_fail_count)
                            self._bus_fail_count = 0
                            self._bus_backoff_until_tick = 0
                    except TimeoutError:
                        self._bus_fail_count += 1
                        if self._bus_fail_count <= 2:
                            # Suppress false-positive degradation logging for non-critical telemetry publishing during heavy LLM load
                            # _record_mind_degradation(TimeoutError("event_bus_publish_timeout"))
                            logger.warning("⚠️ MindTick: EventBus publish stalled (timeout). Continuing tick.")
                        elif self._bus_fail_count == 3:
                            logger.warning(
                                "⚠️ MindTick: EventBus publish timing out repeatedly (%d). "
                                "Backing off EventBus publish retries for runtime stability.",
                                self._bus_fail_count,
                            )
                        self._record_bus_outage("timeout")
                        backoff = min(30, 10 * self._bus_fail_count)
                        self._bus_backoff_until_tick = self._tick_count + backoff
                    except _MIND_BOUNDARY_ERRORS as e:
                        self._bus_fail_count += 1
                        # Only record degradation on first failure, then back off
                        if self._bus_fail_count <= 2:
                            # Suppress false-positive degradation logging for non-critical telemetry publishing during heavy LLM load
                            # _record_mind_degradation(e)
                            logger.error("⚠️ MindTick: EventBus publish failed: %s", e)
                        elif self._bus_fail_count == 3:
                            logger.warning(
                                "⚠️ MindTick: EventBus publish failing repeatedly (%d). "
                                "Backing off retries. Will retry every 30 ticks.",
                                self._bus_fail_count,
                            )
                        self._record_bus_outage(type(e).__name__)
                        # Exponential backoff: skip 10, 20, 30 ticks (capped at 30)
                        backoff = min(30, 10 * self._bus_fail_count)
                        self._bus_backoff_until_tick = self._tick_count + backoff
                
                # 4. Metacognitive Audit
                audit_interval = 60.0 # 1 minute base audit
                if hasattr(self, 'metacognitive_monitor') and self.metacognitive_monitor and current_state.state_id != state.state_id:
                    if len(current_state.cognition.working_memory) > len(state.cognition.working_memory):
                        new_msg = current_state.cognition.working_memory[-1]
                        if new_msg.get("role") == "assistant" and (time.time() - self._last_audit_time > audit_interval):
                            breaker = self.breakers["audit"]
                            reasoning_pause = self._background_reasoning_pause_reason(current_state)
                            if breaker.is_available and not reasoning_pause:
                                try:
                                    report = await asyncio.wait_for(self.metacognitive_monitor.evaluate(new_msg["content"], current_state), timeout=15.0)  # 32B needs more than 3s
                                    self._last_audit_time = time.time()
                                    breaker.record_success()
                                    if report.revision_needed and report.revised_response:
                                        logger.warning("⚖️ MindTick: Metacognitive violation! Revising...")
                                        # A revision REPLACED the committed
                                        # message in place. What she actually
                                        # said was gone — no original to
                                        # compare against, no record that a
                                        # revision happened, and nothing a
                                        # reader could use to see the
                                        # correction. Overwriting a memory is
                                        # how a system stops being able to
                                        # audit itself.
                                        revised_msg = current_state.cognition.working_memory[-1]
                                        original_content = revised_msg.get("content", "")
                                        revised_msg["content"] = report.revised_response
                                        revised_msg["revision"] = {
                                            "schema": "aura.mind_tick.metacognitive_revision.v1",
                                            "original_content": original_content,
                                            "original_sha256": hashlib.sha256(
                                                str(original_content).encode("utf-8")
                                            ).hexdigest(),
                                            "revised_by": "metacognitive_monitor",
                                            "reason": str(
                                                getattr(report, "reason", "") or "metacognitive_violation"
                                            )[:240],
                                            "revised_at_unix": time.time(),
                                            "tick": self._tick_count,
                                        }
                                        current_state = current_state.derive("metacognitive_revision")
                                except _MIND_BOUNDARY_ERRORS as e:
                                    detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                                    logger.warning("⚠️ MindTick: Metacognitive audit failed: %s", detail)
                                    breaker.record_failure()
                            elif reasoning_pause and (self._tick_count % 20 == 0):
                                logger.debug("💓 MindTick: Skipping metacognitive audit (%s).", reasoning_pause)

                # 5. Evaluate Prediction Error (if state changed)
                if prediction and current_state.state_id != state.state_id and hasattr(self, 'predictive_engine') and self.predictive_engine:
                    actual = self._get_actual_from_state(current_state)
                    if actual:
                        try:
                            error = await self.predictive_engine.evaluate(prediction, actual, current_state)
                        except _MIND_BOUNDARY_ERRORS as exc:
                            _record_mind_degradation(
                                exc,
                                action="continued MindTick after predictive evaluation failed",
                                severity="warning",
                            )
                            logger.warning("⚠️ MindTick: Prediction evaluation failed: %s", exc)
                            error = None
                        # A missing error only means there is no surprise to
                        # process — it must NOT `continue` out of the tick,
                        # which skipped goal evaluation, resource stakes, the
                        # state commit, and the heartbeat, losing already-
                        # computed state changes. Guard the surprise block and
                        # let the rest of the tick run.
                        if error is not None:
                            logger.info("💥 MindTick: Surprise signal: %s", f"{error.surprise_signal:.2f}")
                            # Feed surprise into affect update (arousal/curiosity)
                            current_state.affect.arousal = min(1.0, current_state.affect.arousal + error.surprise_signal * 0.2)
                            current_state.affect.curiosity = min(1.0, current_state.affect.curiosity + error.surprise_signal * 0.1)

                            # A prediction and the surprise that followed it
                            # are a CORRELATION. The world model already knows
                            # that — edges enter as `correlates_with` and are
                            # upgraded to `causes` only by an intervention with
                            # a receipt — but this call said "causal link" and
                            # reported no reporter, so the distinct-reporter
                            # confidence added in CP126 462e62cb could not see
                            # who observed it and every tick looked anonymous.
                            try:
                                cwm = ServiceContainer.get("causal_world_model", default=None)
                                if cwm and error.surprise_signal > 0.4:
                                    safe_prediction = str(prediction.content).strip()[:30] if hasattr(prediction, 'content') else "unknown"
                                    safe_actual = str(actual).strip()[:30]
                                    cwm.add_observation(
                                        source=safe_prediction,
                                        target=safe_actual,
                                        correlation=error.surprise_signal,
                                        reported_by="mind_tick.prediction_surprise",
                                    )
                                    logger.info(
                                        "🌐 CausalWorldModel recorded a CORRELATION from a surprise "
                                        "signal; it stays correlational until an intervention says otherwise."
                                    )
                            except _MIND_BOUNDARY_ERRORS as cwm_e:
                                _record_mind_degradation(cwm_e)
                                logger.error("Failed to record causal observation in MindTick: %s", cwm_e)

                # 5.5 Goal evaluation — check for goal completion every ~30 ticks
                if self._tick_count % 30 == 0:
                    try:
                        goal_engine = ServiceContainer.get("goal_engine", default=None)
                        if goal_engine and hasattr(goal_engine, "evaluate_goals"):
                            await asyncio.wait_for(goal_engine.evaluate_goals(), timeout=5.0)

                            # Feed completed/failed goals to intrinsic motivation
                            try:
                                im = ServiceContainer.get("intrinsic_motivation", default=None)
                                if im and hasattr(goal_engine, "get_recent_completions"):
                                    for goal in goal_engine.get_recent_completions(limit=5):
                                        name = str(goal.get("name", goal.get("objective", "unknown")))
                                        success = str(goal.get("status", "")).lower() in ("completed", "succeeded")
                                        im.record_competence(name, success)
                            except (TypeError, ValueError, AttributeError) as exc:
                                logger.debug("MindTick: Intrinsic motivation competence update skipped: %s", exc)

                    except TimeoutError:
                        logger.debug("MindTick: Goal evaluation timed out.")
                    except _MIND_BOUNDARY_ERRORS as _ge_err:
                        _record_mind_degradation(_ge_err)
                        logger.debug("MindTick: Goal evaluation failed: %s", _ge_err)

                # 5.6 Resource stakes — tick the digital mortality engine
                try:
                    from core.consciousness.resource_stakes import get_resource_stakes
                    get_resource_stakes().tick()
                except _MIND_BOUNDARY_ERRORS as exc:
                    _record_mind_degradation(exc)
                    logger.debug("MindTick: Resource stakes tick failed: %s", exc)

                # 6. Synchronize Persistence Metrics
                # Save all circuit breaker states into the state object before commit
                current_state.health["circuits"] = {
                    name: breaker.to_dict() for name, breaker in self.breakers.items()
                }
                for name, breaker in self.phase_breakers.items():
                    current_state.health["circuits"][f"phase_{name}"] = breaker.to_dict()
                
                # Check sidecar process health
                local_runtime_state = "offline"
                try:
                    gate = ServiceContainer.get("inference_gate", default=None)
                    lane = gate.get_conversation_status() if gate and hasattr(gate, "get_conversation_status") else {}
                    if isinstance(lane, dict) and lane:
                        lane_state = str(lane.get("state", "") or "").strip().lower()
                        if bool(lane.get("conversation_ready", False)):
                            local_runtime_state = "online"
                        elif lane_state in {"warming", "recovering", "spawning", "handshaking", "ready"}:
                            local_runtime_state = "warming"
                except _MIND_BOUNDARY_ERRORS as exc:
                    _record_mind_degradation(exc)
                    logger.debug("MindTick local runtime health probe via gate failed: %s", exc)
                if local_runtime_state == "offline":
                    # ASK the registered client; never BUILD one. get_mlx_client()
                    # resolves the model path and constructs a backend, which
                    # can initialize MLX inside the rhythm process — the one
                    # process that must stay free of model work. A health probe
                    # that spawns the thing it is probing is not a probe.
                    mlx_client = ServiceContainer.get("mlx_client", default=None)
                    if mlx_client is not None and hasattr(mlx_client, "is_alive"):
                        local_runtime_state = "online" if mlx_client.is_alive() else "offline"
                    else:
                        # No client registered: nobody has stood the lane up.
                        # "unknown" is the truth; "offline" would be a verdict
                        # this tick has no evidence for.
                        local_runtime_state = "unknown"
                current_state.health["capabilities"]["local_runtime"] = local_runtime_state
                
                # Unlike the MLX client, constructing this one starts nothing:
                # __init__ only sets up the IPC handles for a sidecar that may
                # or may not be running, so asking it is a real probe.
                from core.senses.sensory_client import get_sensory_client
                sensory_client = get_sensory_client()
                current_state.health["capabilities"]["sensory_worker"] = "online" if sensory_client.is_alive() else "offline"
                
                current_state.health["watchdog_timestamp"] = time.time()
                
                # 6.5. Systemic pulse audit.
                if self._tick_count % 100 == 0: # Every ~50s in conversational mode
                    await self._immune_pulse_audit()

                # 7. If state changed, commit it
                if current_state and state and current_state.state_id != state.state_id:
                    from .state.state_repository import StateVersionConflictError
                    if not self._running or is_shutdown_requested():
                        logger.debug("💓 MindTick: Skipping final commit during shutdown.")
                        break
                    try:
                        await self.orchestrator.state_repo.commit(current_state, "mind_tick")
                        
                        # Autonomous Response Emission
                        # Only emit responses that background ticks actually produced —
                        # NOT responses from foreground user ticks that were committed
                        # before this tick read the state.
                        if len(current_state.cognition.working_memory) > len(state.cognition.working_memory):
                            last_msg = current_state.cognition.working_memory[-1]
                            origin = str(last_msg.get("origin", "") or last_msg.get("source", "") or "").lower()
                            is_foreground_response = origin in (
                                "user", "voice", "admin", "api", "gui", "ws",
                                "websocket", "direct", "external", "response_generation",
                                "response_generation_user", "tick",
                            )
                            if last_msg.get("role") == "assistant" and not is_foreground_response:
                                logger.info("🗣️ MindTick: Routing autonomous response through ExecutiveAuthority.")

                                content = last_msg.get("content", "")
                                # This text is about to be SPOKEN to him,
                                # unprompted. The old gate was
                                # `len > 5 or any alphabetic`, which the `or`
                                # made vacuous: a single letter passed. It goes
                                # through the same assessment a foreground
                                # reply does — the gate built to decide whether
                                # text is fit to serve a person — rather than a
                                # length heuristic invented here.
                                from core.brain.llm.latent_cortex.output_quality import (
                                    _terminal_complete,
                                )
                                from core.conversation.response_reliability import (
                                    assess_user_facing_reply,
                                )

                                # No question gives this utterance its meaning,
                                # so it has to stand on its own: a finished
                                # sentence, not a token. The reply assessor
                                # declines to judge without a request (its own
                                # no-context-no-verdict rule), so completeness
                                # is what carries the weight here and the
                                # assessor's hard failures still apply on top.
                                assessment = assess_user_facing_reply("", content)
                                stripped = content.strip()
                                is_meaty = (
                                    not assessment.hard_failure
                                    and _terminal_complete(stripped)
                                    and len(stripped.split()) >= 4
                                )
                                has_null = "null" in content.lower()
                                has_action = "say '" in content.lower() or "do '" in content.lower()
                                if not is_meaty:
                                    logger.debug(
                                        "MindTick: withheld autonomous utterance (%s).",
                                        ",".join(assessment.reasons[:3]) or "incomplete",
                                    )

                                if is_meaty and not has_null and not has_action:
                                    try:
                                        from core.consciousness.executive_authority import (
                                            get_executive_authority,
                                        )
                                        authority = get_executive_authority(self.orchestrator)
                                        await authority.release_expression(
                                            content,
                                            source="mind_tick_autonomous",
                                            urgency=0.5,
                                            target="primary",
                                            metadata={"autonomous": True, "spontaneous": True},
                                        )
                                    except _MIND_BOUNDARY_ERRORS as _ea_err:
                                        _record_mind_degradation(_ea_err)
                                        logger.debug("MindTick: ExecutiveAuthority emission failed: %s", _ea_err)
                    except StateVersionConflictError:
                        # For MindTick, we can safely ignore conflicts; the next tick will catch up
                        logger.debug("💓 MindTick: Skipping commit due to concurrent update (Atomic Guard).")
                    except _MIND_BOUNDARY_ERRORS as e:
                        _record_mind_degradation(e)
                        logger.error("❌ MindTick: Commit failed: %s", e)
                        
                # Subconscious Memory Consolidation ("Dreaming")
                if self.mode != CognitiveMode.SLEEP and current_state.cognition.working_memory:
                    last_user_time = 0.0
                    for msg in reversed(current_state.cognition.working_memory):
                        if msg.get("role") == "user":
                            last_user_time = msg.get("timestamp", 0.0)
                            break

                    # Only consolidate after a real user interaction in the current
                    # working set for this session; inherited/restored state should
                    # not look "idle" enough to trigger dreaming immediately after boot.
                    session_start = float(getattr(self.orchestrator, "start_time", 0.0) or 0.0)
                    if last_user_time == 0.0 or (session_start and last_user_time < session_start):
                        idle_time = 0.0
                    else:
                        idle_time = time.time() - last_user_time
                    # Increased idle threshold to 20 minutes (1200s) to reduce state churn
                    if idle_time > 1200.0 and len(current_state.cognition.working_memory) >= 5:
                        logger.info("🌙 MindTick: 20+ minutes of idle time. Triggering Subconscious Consolidation.")
                        # Need the actual coordinator, not just the facade interface if we are calling a new method
                        memory_coord = self.orchestrator.memory
                        if memory_coord and hasattr(memory_coord, "consolidate_working_memory"):
                            decision = _authorize_state_mutation_through_will(
                                "dream_consolidation: consolidate working memory during idle; identity-affecting writes must stay governed",
                                "mind_tick.dream_consolidation",
                                priority=0.55,
                                # Sleep-class restoration context: without it the
                                # welfare recovery-drive rule deferred the very
                                # act that lowers recovery drive — 2,428 blocked
                                # consolidations across the 7/17-7/21 sessions,
                                # freezing memory writes for whole sessions.
                                context={
                                    "source": "mind_tick.dream_consolidation",
                                    "effect_scope": "internal_restoration",
                                    "no_external_effects": True,
                                },
                            )
                            if not decision or not decision.is_approved():
                                self.set_mode(CognitiveMode.CONVERSATIONAL)
                            else:
                                try:
                                    current_state.response_modifiers["dream_consolidation_will_receipt"] = decision.receipt_id
                                except _MIND_BOUNDARY_ERRORS as exc:
                                    _record_mind_degradation(exc)
                                    logger.debug("MindTick: Dream consolidation receipt annotation failed: %s", exc)
                                _schedule_mind_task(
                                    memory_coord.consolidate_working_memory(current_state, is_background=True),
                                    name="mind_tick.consolidate_working_memory",
                                )
                                # These run during dream consolidation, not on
                                # every tick — but they ran ON THE LOOP:
                                # metacognitive assessment, value-graph
                                # evolution, a hidden eval suite and STDP
                                # diagnostics, each of which can take seconds.
                                # A dream is background work by definition, so
                                # it goes off-thread and bounded rather than
                                # parking the rhythm for the duration.
                                try:
                                    await asyncio.wait_for(
                                        asyncio.to_thread(self._dream_research_modules),
                                        timeout=120.0,
                                    )
                                except TimeoutError:
                                    record_degraded_event(
                                        "mind_tick",
                                        "tick_stage_timeout",
                                        detail="dream_research_modules>120s",
                                        severity="warning",
                                        classification="background_degraded",
                                        context={
                                            "stage": "dream_research",
                                            "tick_count": self._tick_count,
                                        },
                                    )
                                    self._mark_loop_progress("dream_research_timeout_yield")
                                except (TypeError, ValueError, RuntimeError, ImportError) as _drm:
                                    logger.debug("MindTick: Dream research modules skipped: %s", _drm)
                                _schedule_mind_task(
                                    self._replay_deferred_memory_writes(),
                                    name="mind_tick.replay_deferred_memory_writes",
                                )
                                self.set_mode(CognitiveMode.SLEEP)
                
                # NOTE: tick_count and cycle_count increment moved to finally block
                # to guarantee the heartbeat always advances.

                metadata.duration = asyncio.get_running_loop().time() - start_time
                self._last_tick_metadata = metadata
                self._last_successful_tick_at = time.time()
                self._active_tick_stage = "tick_complete"
                self._mark_loop_progress("tick_complete")
                self._consecutive_loop_failures = 0
                
            except asyncio.CancelledError:
                if not self._running or is_shutdown_requested():
                    break
                # Cancellation is a DIRECTIVE. Swallowing it and continuing let
                # the loop outlive the thing that asked it to stop, and there
                # was no owner to restart it deliberately. There is one now:
                # the task's done-callback repair, chained so exactly one loop
                # exists at any instant. Record who was cancelled and let it
                # propagate.
                record_degraded_event(
                    "mind_tick",
                    "loop_cancelled",
                    detail=f"stage={self._active_tick_stage or '?'}",
                    severity="warning",
                    classification="background_degraded",
                    context={
                        "stage": str(self._active_tick_stage or ""),
                        "tick_count": self._tick_count,
                    },
                )
                raise
            except _MIND_BOUNDARY_ERRORS as e:
                self._consecutive_loop_failures += 1
                _record_mind_degradation(e)
                logger.error(
                    "⚠️ MindTick Loop Error: %s",
                    e,
                    exc_info=(type(e), e, e.__traceback__),
                )
                try:
                    record_degraded_event(
                        "mind_tick",
                        "loop_error",
                        detail=f"{type(e).__name__}: {e}",
                        severity="error",
                        classification="background_degraded",
                        context={
                            "tick_count": self._tick_count,
                            "mode": getattr(self.mode, "value", str(self.mode)),
                        },
                        exc=e,
                    )
                except _MIND_BOUNDARY_ERRORS as _exc:
                    _record_mind_degradation(_exc)
                    logger.debug("MindTick degraded-event receipt failed: %s", _exc)
            finally:
                # Always advance heartbeat counters, even on degraded ticks.
                self._tick_count += 1
                try:
                    if hasattr(self.orchestrator, 'status') and self.orchestrator.status:
                        current_c = getattr(self.orchestrator.status, 'cycle_count', 0)
                        self.orchestrator.status.cycle_count = current_c + 1
                except _MIND_BOUNDARY_ERRORS as exc:
                    _record_mind_degradation(exc)
                    logger.debug("MindTick: Cycle count increment failed: %s", exc)
                
                # Wait for the next tick based on mode.
                # Adaptive pacing: if the last tick was slow (> 5s), back off
                # proportionally to give the event loop breathing room and
                # prevent the "mean tick too slow" stability guardian alert.
                interval = sleep_time_override or TICK_INTERVALS.get(self.mode, 2.0)
                elapsed = asyncio.get_running_loop().time() - start_time
                # Black-box flight recorder (roadmap A5): one mind-moment per
                # tick, degraded ticks included — the ring is what survives a
                # hard death. A single bounded memcpy; never blocks the loop.
                try:
                    from core.runtime.flight_recorder import record_mind_moment
                    record_mind_moment(
                        tick=self._tick_count,
                        stage=str(self._active_tick_stage or ""),
                        mode=getattr(self.mode, "value", str(self.mode)),
                        tick_duration_ms=elapsed * 1000.0,
                        consecutive_failures=self._consecutive_loop_failures,
                    )
                except _MIND_BOUNDARY_ERRORS as exc:
                    _record_mind_degradation(exc)
                if elapsed > 5.0:
                    # Proportional backoff is a sleep DURATION, not a deadline.
                    # Subtracting elapsed from it inverted the intent exactly
                    # when it mattered: a 20s tick with a 10s backoff slept
                    # max(1, 10-20) = 1s, so the slowest ticks got the
                    # shortest rest and the loop never got the breathing room
                    # the backoff exists to give it.
                    sleep_time = max(1.0, elapsed * 0.5)
                    if self._tick_count % 10 == 0:
                        logger.debug(
                            "MindTick: adaptive backoff — last tick %.1fs, sleeping %.1fs.",
                            elapsed, sleep_time,
                        )
                else:
                    sleep_time = max(1.0, interval - elapsed)
                if is_shutdown_requested():
                    self._running = False
                if self._running:
                    self._active_tick_stage = "sleep"
                    self._mark_loop_progress("sleep")
                    await asyncio.sleep(sleep_time)
                else:
                    self._active_tick_stage = "stopped"

    #: While the event bus is down the tick keeps a standing degraded record
    #: rather than going quiet. Re-asserted on this cadence so the state stays
    #: truthful and recovery pressure persists, without one line per tick.
    _BUS_OUTAGE_REASSERT_TICKS = 30

    def _record_bus_outage(self, kind: str) -> None:
        """Keep a dark telemetry lane visible.

        Log backoff is right — a line per tick is noise. Silence is not: the
        degradation record was commented out entirely and then suppressed
        after three failures, so a runtime publishing nothing looked healthy
        and nothing applied pressure to fix it.
        """

        # `x or default` throws away a legitimate zero, and tick 0 is exactly
        # when the first outage is reported — so the second report would come
        # straight back through the quiet window.
        raw_last = getattr(self, "_last_bus_outage_report_tick", None)
        last = int(raw_last) if isinstance(raw_last, int) else -10_000
        if self._bus_fail_count > 1 and (
            self._tick_count - last
        ) < self._BUS_OUTAGE_REASSERT_TICKS:
            return
        self._last_bus_outage_report_tick = self._tick_count
        try:
            record_degraded_event(
                "mind_tick",
                "event_bus_publish_failing",
                detail=f"{kind} x{self._bus_fail_count}",
                severity="warning",
                classification="background_degraded",
                context={
                    "consecutive_failures": int(self._bus_fail_count),
                    "tick_count": int(self._tick_count),
                },
            )
        except _MIND_BOUNDARY_ERRORS as exc:
            _record_mind_degradation(exc)

    def _get_actual_from_state(self, state: AuraState) -> str | None:
        """Extract the last actual cognitive output for prediction evaluation."""
        if state.cognition.working_memory:
            # We look for the most recent message
            last_msg = state.cognition.working_memory[-1]
            return last_msg.get("content")
        return None

    def _dream_research_modules(self) -> None:
        """Run research modules during dream consolidation.

        Called once per dream cycle (20+ min idle). Executes:
          1. MetaCognitive assessment → strategy adjustments
          2. Intrinsic motivation → feed to DynamicValueGraph
          3. DVG evolution cycle
          4. Hidden eval suite → drift detection
          5. STDP MESU diagnostics logging

        Uses container-registered singletons (same instances that
        collect observations during regular ticks).
        """
        # 1. Meta-cognitive reflection (uses boot-registered singleton)
        try:
            metacog = ServiceContainer.get("metacognitive_monitor", default=None)
            if metacog is not None:
                reflection = metacog.assess()
                logger.info(
                    "🧠 Dream: MetaCognitive assessment: %s → %s",
                    reflection.condition.value,
                    [a.value for a in reflection.recommended_actions],
                )
                metacog.execute_actions(reflection)
        except (TypeError, ValueError) as exc:
            logger.debug("Dream: MetaCognitive skipped: %s", exc)

        # 2. Intrinsic motivation → feed to value graph
        try:
            im = ServiceContainer.get("intrinsic_motivation", default=None)
            if im is not None:
                count = im.feed_to_value_graph()
                if count > 0:
                    logger.info("🧠 Dream: Fed %d intrinsic motivation signals to DVG", count)
        except (TypeError, ValueError) as exc:
            logger.debug("Dream: IntrinsicMotivation skipped: %s", exc)

        # 3. Dynamic value graph evolution
        try:
            from core.adaptation.dynamic_value_graph import get_dynamic_value_graph
            dvg = get_dynamic_value_graph()
            mutations = dvg.evolve()
            if mutations:
                logger.info("🧠 Dream: DVG evolved: %d mutation(s)", len(mutations))
        except (ImportError, TypeError, ValueError) as exc:
            logger.debug("Dream: DVG evolution skipped: %s", exc)

        # 4. Hidden eval suite
        try:
            from core.architect.hidden_eval import HiddenEvalRunner
            if not hasattr(self, '_research_eval'):
                self._research_eval = HiddenEvalRunner.create_default_suite()
            result = self._research_eval.run_suite()
            if result.drift_detected:
                logger.warning(
                    "⚠️ Dream: Hidden eval DRIFT DETECTED! health=%.2f",
                    result.overall_health,
                )
            elif result.failed > 0:
                logger.warning(
                    "⚠️ Dream: Hidden eval: %d/%d failed, health=%.2f",
                    result.failed, result.total_scenarios, result.overall_health,
                )
        except (ImportError, TypeError, ValueError) as exc:
            logger.debug("Dream: HiddenEval skipped: %s", exc)

        # 5. STDP MESU diagnostics
        try:
            from core.consciousness.stdp_learning import get_stdp_engine

            stdp = (
                ServiceContainer.get("stdp_engine", default=None)
                if ServiceContainer.has("stdp_engine")
                else None
            )
            if stdp is None:
                stdp = get_stdp_engine()
            if stdp is not None and hasattr(stdp, 'get_mesu_diagnostics'):
                diag = stdp.get_mesu_diagnostics()
                logger.info(
                    "🧠 Dream: MESU diagnostics: locked=%d (%.1f%%), uncertainty=%.4f",
                    diag["locked_count"],
                    diag["locked_fraction"] * 100,
                    diag["uncertainty_mean"],
                )
        except (TypeError, ValueError) as exc:
            logger.debug("Dream: MESU diagnostics skipped: %s", exc)

        # 6. EWC consolidation (trigger during dream for weight protection)
        try:
            gov = ServiceContainer.get("plasticity_governor", default=None)
            if gov is not None and hasattr(gov, 'consolidate'):
                gov.consolidate()
                logger.info("🧠 Dream: EWC plasticity governor consolidated.")
        except (TypeError, ValueError) as exc:
            logger.debug("Dream: EWC consolidation skipped: %s", exc)

    async def _replay_deferred_memory_writes(self) -> None:
        """Replay retained writes on the mind loop that owns async services.

        The blocking dream-research bundle runs in ``asyncio.to_thread``.
        Scheduling this coroutine from inside that bundle left no running loop,
        so every dream reported a degradation and retained work never moved.
        The caller creates this task after returning to the owner loop.
        """

        try:
            from core.memory.deferred_retention import get_deferred_retention_queue

            queue = get_deferred_retention_queue()
            report = await queue.replay()
            if report.committed or report.refused or report.expired:
                logger.info(
                    "🧠 Dream: deferred memory writes — %s.", report.narrative()
                )
        except asyncio.CancelledError:
            raise
        except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
            _record_mind_degradation(
                exc, severity="warning",
                action="deferred memory writes were not replayed this dream cycle",
            )

    def set_mode(self, mode: CognitiveMode):
        """Update the cognitive mode and tick interval."""
        if mode != self.mode:
            self.mode = mode
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_task("mind_tick", f"Switching mode to {mode.value}", {"mode": mode.value})

    async def _immune_pulse_audit(self):
        """Perform a deterministic Python-based health audit of the system environment."""
        breaker = self.breakers.get("immune_audit")
        if breaker and not breaker.is_available:
            return

        try:
            from core.resilience.immunity_hyphae import get_immunity
            immunity = get_immunity()
            
            # 1. PID File Integrity
            # Anchored, not cwd-relative: this reads a pid and signals it.
            pid_file = get_config().paths.pid_file
            if await asyncio.to_thread(pid_file.exists):
                try:
                    pid_text = await asyncio.to_thread(pid_file.read_text)
                    pid = int(pid_text.strip())
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        logger.warning("💉 [IMMUNE] Pulse Audit: Found stale PID file for non-existent process %d", pid)
                        immunity.registry.match_and_repair("PID file already exists")
                except (ValueError, OSError) as e:
                    logger.debug("MindTick: PID check failed (likely stale/malformed): %s", e)

            # 2. Resource Leak Probe (Memory)
            from core.runtime.resource_observation import get_resource_observer

            process = await asyncio.to_thread(
                get_resource_observer().process, os.getpid()
            )
            mem_pct = process.memory_percent if process is not None else 100.0
            if mem_pct > 25.0: # Trigger cleanup if one process exceeds 25% RAM
                logger.warning("💉 [IMMUNE] Pulse Audit: High memory usage detected (%.1f%%). Triggering conservative sweep.", mem_pct)
                # A full gc pass on a 20GB resident process is a stop-the-world
                # pause measured in seconds. On the event loop that is the
                # rhythm stopping, every organ with it, to tidy up.
                import gc

                await asyncio.to_thread(gc.collect)

            # 3. Log Sieve (Look for systemic issues)
            from core.config import config
            log_dir = config.paths.data_dir / "error_logs"

            def _sieve_logs() -> list:
                # Directory listing and the sieve's own file reads are
                # filesystem work. Bounded to the newest logs: an unbounded
                # glob over a directory that grows with every incident makes
                # the audit slower exactly when there is most to read.
                if not log_dir.exists():
                    return []
                logs = sorted(
                    log_dir.glob("*.log"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )[:64]
                return immunity.registry.log_sieve(logs)

            hidden = await asyncio.to_thread(_sieve_logs)
            if hidden:
                logger.warning("💉 [IMMUNE] Log Sieve detected %d hidden issues.", len(hidden))

            breaker.record_success()
        except _MIND_BOUNDARY_ERRORS as e:
            _record_mind_degradation(e)
            logger.error("💉 [IMMUNE] Pulse Audit failed: %s", e)
            if breaker:
                breaker.record_failure()
