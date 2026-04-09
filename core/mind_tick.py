import asyncio
import logging
import time
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable
import psutil
from dataclasses import dataclass, field
from enum import Enum

from .state.aura_state import AuraState
from core.brain.predictive_engine import PredictiveEngine
from core.brain.metacognitive_monitor import MetacognitiveMonitor
from core.predictive.trajectory_predictor import TrajectoryPredictor
from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.runtime.pipeline_blueprint import instantiate_legacy_runtime_phases
from core.utils.resilience import CircuitBreaker, run_with_watchdog
from core.utils.task_tracker import get_task_tracker
from core.config import get_config

config = get_config()
logger = logging.getLogger(__name__)

class CognitiveMode(Enum):
    CONVERSATIONAL = "conversational"
    REFLECTIVE = "reflective"
    SLEEP = "sleep"
    CRITICAL = "critical"

TICK_INTERVALS = {
    CognitiveMode.CONVERSATIONAL: 0.5,
    CognitiveMode.REFLECTIVE: 2.0,
    CognitiveMode.SLEEP: 10.0,
    CognitiveMode.CRITICAL: 0.1,
}

PhaseCallable = Callable[[AuraState], Awaitable[AuraState]]

@dataclass
class TickMetadata:
    tick_id: int
    mode: CognitiveMode
    start_time: float
    duration: float = 0.0
    phases_executed: List[str] = field(default_factory=list)
    phase_durations: Dict[str, float] = field(default_factory=dict)

class MindTick:
    """
    The unified cognitive rhythm of Aura.
    
    MindTick executes a sequence of registered 'phases' against the current state
    at a regular interval determined by the CognitiveMode.
    """
    
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.mode = CognitiveMode.CONVERSATIONAL
        self.phases: List[tuple[str, PhaseCallable]] = []
        self._running = False
        self._tick_count = 0
        self._task: Optional[asyncio.Task] = None
        self._last_tick_metadata: Optional[TickMetadata] = None
        
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
        self.phase_breakers: Dict[str, CircuitBreaker] = {}
        
        # Initiative re-promotion cooldown
        self._last_initiative_goal: Optional[str] = None
        self._last_initiative_time: float = 0.0
        self._initiative_cooldown: float = 30.0  # seconds
        self._missing_state_streak: int = 0
        self._last_missing_state_log: float = 0.0
        self._max_missing_state_backoff: float = 5.0
        
        # Phase-specific timeouts
        self.phase_timeouts = {
            "response_generation": 120.0,
            "memory_retrieval": 30.0,     # Increased for DB stability
            "cognitive_routing": 120.0,   # Matched to MLX load deadline
            "memory_consolidation": 20.0,
        }
        self.default_timeout = 5.0
        
        # Bootstrap Phase Decomposition (Phase 3)
        self._bootstrap_phases()

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

        # [STRUCTURAL UNIFICATION] Wire TaskRegistry heartbeat
        from core.supervisor.registry import get_task_registry
        self.registry = get_task_registry()
        logger.info("📋 MindTick: TaskRegistry heartbeat wired.")

    def register_phase(self, name: str, phase_fn: PhaseCallable):
        """Register a new cognitive phase to execute every tick."""
        self.phases.append((name, phase_fn))
        logger.info(f"🧠 MindTick: Registered phase '{name}'")

    def reload_phases(self):
        """Dynamically reloads all phase modules and re-bootstraps the pipeline."""
        logger.info("🔄 MindTick: Hot-reloading cognitive phases...")
        import importlib
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

    def _background_reasoning_pause_reason(self, state: Optional[AuraState] = None) -> str:
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
        except Exception as exc:
            logger.debug("MindTick background flow probe failed: %s", exc)

        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"
        except Exception as exc:
            logger.debug("MindTick router pressure probe failed: %s", exc)

        try:
            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin="mind_tick") or "").strip()
                if reason:
                    return reason
        except Exception as exc:
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
        self._task = get_task_tracker().track_task(
            asyncio.create_task(self._run_loop(), name="mind_tick.run_loop")
        )
        logger.info("💓 MindTick: Cognitive rhythm started.")

    async def stop(self):
        """Stop the cognitive rhythm."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("MindTick: Shutdown requested.")
        logger.info("🛑 MindTick: Cognitive rhythm stopped.")

    async def _run_loop(self):
        """The main execution loop for the cognitive rhythm."""
        while self._running:
            sleep_time_override: Optional[float] = None
            try:
                start_time = asyncio.get_running_loop().time()
                
                # 1. Get the latest state
                from infrastructure.watchdog import get_watchdog
                get_watchdog().heartbeat("mind_tick")
                
                state = await self.orchestrator.state_repo.get_current()
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
                        await _will.start()
                except Exception as _will_boot:
                    if self._tick_count <= 1:
                        logger.debug("MindTick: Unified Will boot deferred: %s", _will_boot)

                # ── WORLD STATE: Update telemetry every tick ──
                try:
                    from core.world_state import get_world_state
                    get_world_state().update()
                except Exception:
                    pass

                # ── BINDING ENGINE: Run coherence tick before phases ──
                _coherence_report = None
                _bg_pause_pre = self._background_reasoning_pause_reason(state)
                if not _bg_pause_pre:
                    try:
                        from core.coherence.binding_engine import get_binding_engine
                        _binding = get_binding_engine()
                        _coherence_report = await asyncio.wait_for(_binding.tick(state), timeout=3.0)
                    except Exception as _be:
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
                                active = goal_engine.get_active_goals(limit=3, include_external=False)
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
                                    from core.runtime.proposal_governance import propose_governed_initiative_to_state
                                    state, _ = await propose_governed_initiative_to_state(
                                        state,
                                        objective,
                                        source="goal_engine",
                                        urgency=float(goal.get("priority", 0.5)),
                                        triggered_by="proactive_goal_pursuit",
                                    )
                                    break  # Only inject one goal per cycle
                        except Exception as _ge:
                            logger.debug("MindTick: goal-driven initiative generation failed: %s", _ge)

                # ── INITIATIVE ARBITRATION: Replace FIFO with scored selection ──
                if not state.cognition.current_objective and state.cognition.pending_initiatives:
                    # Cooldown: don't re-promote the same initiative within 30s
                    top_goal = ""
                    if state.cognition.pending_initiatives:
                        top_init = state.cognition.pending_initiatives[0]
                        top_goal = top_init.get("goal", "") if isinstance(top_init, dict) else str(top_init)
                    now_init = time.time()
                    if top_goal == self._last_initiative_goal and (now_init - self._last_initiative_time) < self._initiative_cooldown:
                        pass  # Skip — same initiative, still in cooldown
                    else:
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
                # [THROTTLE] Only predict if interval passed or in special curiosity peak
                prediction_interval = config.autonomous_thought_interval_s if not config.skeletal_mode else 300.0
                if hasattr(self, 'predictive_engine') and self.predictive_engine:
                    breaker = self.breakers["prediction"]
                    reasoning_pause = self._background_reasoning_pause_reason(state)
                    if breaker.is_available and not reasoning_pause and (time.time() - self._last_prediction_time > prediction_interval):
                        try:
                            # Force tertiary tier for background prediction
                            # Phase 33: Increased to 30.0s to accommodate slow CPU inference when Metal/Gemini fails.
                            prediction = await asyncio.wait_for(
                                self.predictive_engine.predict(state, prefer_tier="tertiary", is_background=True), 
                                timeout=30.0
                            )
                            self._last_prediction_time = time.time()
                            breaker.record_success()
                            logger.info(f"🔮 MindTick: Predicted: {prediction.content[:50]}...")
                        except (asyncio.TimeoutError, Exception) as e:
                            detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                            logger.warning("⚠️ MindTick: Prediction failed/stalled: %s", detail)
                            breaker.record_failure()
                            prediction = None
                    elif reasoning_pause and (self._tick_count % 20 == 0):
                        logger.debug("💓 MindTick: Skipping predictive background reasoning (%s).", reasoning_pause)

                # [MOTO TRANSIMAL] Trajectory Prediction (Next 3 steps)
                if hasattr(self, 'trajectory_predictor') and self.trajectory_predictor:
                    reasoning_pause = self._background_reasoning_pause_reason(state)
                    if not reasoning_pause and time.time() - self._last_trajectory_time > 60.0: # Every minute
                        get_task_tracker().track_task(
                            asyncio.create_task(
                                self.trajectory_predictor.predict_path(
                                    state.cognition.current_objective or "General Processing",
                                    state,
                                ),
                                name="MindTick.trajectory_predict",
                            )
                        )
                        self._last_trajectory_time = time.time()

                # 3. Execute all registered phases within a Mycelial rooted_flow (Phase 4)
                mycelium = ServiceContainer.get("mycelium", default=None)
                
                metadata = TickMetadata(
                    tick_id=self._tick_count,
                    mode=self.mode,
                    start_time=time.time()
                )

                current_state = state

                async def execute_tick():
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
                    if not objective:
                        return current_state
                    if current_origin in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}:
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
                        # [USER PRIORITY GUARD] If a user message is already waiting
                        # for the kernel lock, skip this background tick entirely.
                        # The user's tick will run as soon as the lock is free.
                        if getattr(kernel, "_user_priority_pending", None) and kernel._user_priority_pending.is_set():
                            logger.debug("💓 MindTick: Skipping background tick — user priority message pending.")
                            return current_state
                        # [EVENT LOOP PRESSURE GUARD] If the loop is already lagging,
                        # don't add a full kernel tick (which invokes the LLM).
                        _bg_pause = self._background_reasoning_pause_reason(current_state)
                        if _bg_pause:
                            if self._tick_count % 30 == 0:
                                logger.debug("💓 MindTick: Deferring background kernel tick (%s).", _bg_pause)
                            return current_state
                        try:
                            entry = await kernel.tick(objective, priority=False)
                            if entry is not None:
                                # Kernel ran successfully — fetch the committed state
                                committed = await self.orchestrator.state_repo.get_current()
                                if committed:
                                    current_state = committed
                                    metadata.phases_executed.append("kernel_sovereign_tick")
                                    logger.debug("💓 MindTick: Kernel sovereign tick completed (cycle %d).", self._tick_count)
                                return current_state
                            else:
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
                        except Exception as _kt_err:
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

                    # Once the kernel has booted, degraded self-execution is a
                    # constitutional violation, not a convenience fallback.
                    if kernel_live:
                        logger.debug("💓 MindTick: Kernel is live; skipping degraded-mode self-execution.")
                        return current_state

                    # ── DEGRADED MODE: MindTick runs its own phases ──
                    # Only reached when kernel is not yet booted or tick acquisition fails.
                    logger.debug("💓 MindTick: Running degraded-mode phase pipeline (kernel unavailable).")
                    async with asyncio.TaskGroup() as tg:
                        for name, phase_fn in self.phases:
                            # Relaxed failure threshold for complex phases
                            max_f = 5 if name == "response_generation" else 2
                            breaker = self.phase_breakers.setdefault(name, CircuitBreaker(f"phase_{name}", max_failures=max_f, reset_timeout=60.0))
                            
                            if not breaker.is_available:
                                logger.warning(f"⚠️ MindTick: Phase '{name}' SKIPPED (Circuit Open)")
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
                                current_state = await asyncio.wait_for(phase_fn(current_state), timeout=timeout)
                                phase_duration = time.perf_counter() - phase_start
                                
                                metadata.phases_executed.append(name)
                                metadata.phase_durations[name] = phase_duration
                                
                                if phase_duration > 1.0:
                                    logger.warning(f"🐢 MindTick: Slow phase detected: '{name}' took {phase_duration:.3f}s")
                                
                                breaker.record_success()
                            except asyncio.TimeoutError:
                                logger.error(f"🛑 MindTick: Phase '{name}' STALLED (timeout). Tripping circuit.")
                                breaker.record_failure()
                            except Exception as phase_err:
                                logger.error(f"❌ MindTick: Phase '{name}' failed: {phase_err}")
                                breaker.record_failure()
                                
                            # [RECOVERY] Auto-reset breakers if system has been stable for 100+ cycles
                            if self._tick_count % 100 == 0:
                                for b in self.phase_breakers.values():
                                    if not b.is_available:
                                        logger.info(f"♻️ MindTick: Periodic recovery - Resetting circuit for phase {b.name}")
                                        b.reset()
                    
                        # [REFLEXIVE FALLBACK]
                        if "response_generation" not in metadata.phases_executed:
                            user_origins = ("user", "voice", "admin", "external", "gui", "api", "websocket", "direct")
                            current_origin = getattr(current_state.cognition, "current_origin", None)
                            if current_origin in user_origins:
                                logger.warning("🛡️ MindTick: Emergency Fallback - Injecting reflexive response.")
                                latest = current_state.cognition.working_memory[-1] if current_state.cognition.working_memory else {}
                                if not (
                                    latest.get("origin") == "mind_tick_fallback"
                                    and latest.get("content") == "Give me a moment — I'm thinking through something."
                                ):
                                    current_state.cognition.working_memory.append({
                                        "role": "assistant",
                                        "content": "Give me a moment — I'm thinking through something.",
                                        "timestamp": time.time(),
                                        "origin": "mind_tick_fallback",
                                        "ephemeral": True,
                                    })
                                    current_state = current_state.derive("reflexive_fallback")
                            else:
                                logger.debug(
                                    "🛡️ MindTick: Skipping reflexive fallback for non-user origin %r.",
                                    current_origin,
                                )
                            
                    # [WHOLESALE FIX] Clear objective AFTER all phases complete
                    try:
                        if current_state:
                            from core.consciousness.executive_authority import get_executive_authority

                            authority = get_executive_authority(self.orchestrator)
                            current_state, _ = await authority.complete_current_objective(
                                current_state,
                                reason="tick_cycle_complete",
                                source="mind_tick",
                            )
                    except Exception as e:
                        logger.debug(f"MindTick: Objective cleanup failed: {e}")
                    return current_state

                if mycelium:
                    async with mycelium.rooted_flow(source="mind_tick", target="cognitive_phases", activity="cognitive_cycle"):
                        current_state = await execute_tick()
                else:
                    current_state = await execute_tick()
                
                # 4. Bridge to Event Bus (for UI/Observability)
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                try:
                    # Wrap in a 5.0s timeout to prevent Redis stalls from blocking the tick.
                    await asyncio.wait_for(bus.publish("aura/events/mind_tick", {
                        "tick_id": self._tick_count,
                        "mode": self.mode.value,
                        "phases": metadata.phases_executed,
                        "durations": metadata.phase_durations,
                        "total_duration": metadata.duration,
                        "timestamp": time.time()
                    }), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("⚠️ MindTick: EventBus publish stalled (timeout). Continuing tick.")
                except Exception as e:
                    logger.error("⚠️ MindTick: EventBus publish failed: %s", e)
                
                # 4. Metacognitive Audit
                # [THROTTLE] Only audit periodically unless in critical mode
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
                                        logger.warning(f"⚖️ MindTick: Metacognitive violation! Revising...")
                                        current_state.cognition.working_memory[-1]["content"] = report.revised_response
                                        current_state = current_state.derive("metacognitive_revision")
                                except (asyncio.TimeoutError, Exception) as e:
                                    detail = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                                    logger.warning("⚠️ MindTick: Metacognitive audit failed: %s", detail)
                                    breaker.record_failure()
                            elif reasoning_pause and (self._tick_count % 20 == 0):
                                logger.debug("💓 MindTick: Skipping metacognitive audit (%s).", reasoning_pause)

                # 5. Evaluate Prediction Error (if state changed)
                if prediction and current_state.state_id != state.state_id and hasattr(self, 'predictive_engine') and self.predictive_engine:
                    actual = self._get_actual_from_state(current_state)
                    if actual:
                        error = await self.predictive_engine.evaluate(prediction, actual, current_state)
                        logger.info(f"💥 MindTick: Surprise signal: {error.surprise_signal:.2f}")
                        # Feed surprise into affect update (arousal/curiosity)
                        current_state.affect.arousal = min(1.0, current_state.affect.arousal + error.surprise_signal * 0.2)
                        current_state.affect.curiosity = min(1.0, current_state.affect.curiosity + error.surprise_signal * 0.1)

                        # Disconnected logic re-attached: When surprise is high, log the causal link
                        try:
                            cwm = ServiceContainer.get("causal_world_model", default=None)
                            if cwm and error.surprise_signal > 0.4:
                                safe_prediction = str(prediction.content).strip()[:30] if hasattr(prediction, 'content') else "unknown"
                                safe_actual = str(actual).strip()[:30]
                                cwm.add_observation(
                                    source=safe_prediction,
                                    target=safe_actual,
                                    correlation=error.surprise_signal
                                )
                                logger.info("🌐 CausalWorldModel learned new observation from surprise signal.")
                        except Exception as cwm_e:
                            logger.error(f"Failed to record causal observation in MindTick: {cwm_e}")

                # 5.5 Goal evaluation — check for goal completion every ~30 ticks
                if self._tick_count % 30 == 0:
                    try:
                        goal_engine = ServiceContainer.get("goal_engine", default=None)
                        if goal_engine and hasattr(goal_engine, "evaluate_goals"):
                            await asyncio.wait_for(goal_engine.evaluate_goals(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.debug("MindTick: Goal evaluation timed out.")
                    except Exception as _ge_err:
                        logger.debug("MindTick: Goal evaluation failed: %s", _ge_err)

                # 5.6 Resource stakes — tick the digital mortality engine
                try:
                    from core.consciousness.resource_stakes import get_resource_stakes
                    get_resource_stakes().tick()
                except Exception:
                    pass

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
                except Exception as exc:
                    logger.debug("MindTick local runtime health probe via gate failed: %s", exc)
                if local_runtime_state == "offline":
                    from core.brain.llm.mlx_client import get_mlx_client
                    mlx_client = get_mlx_client()
                    local_runtime_state = "online" if mlx_client.is_alive() else "offline"
                current_state.health["capabilities"]["local_runtime"] = local_runtime_state
                
                from core.senses.sensory_client import get_sensory_client
                sensory_client = get_sensory_client()
                current_state.health["capabilities"]["sensory_worker"] = "online" if sensory_client.is_alive() else "offline"
                
                current_state.health["watchdog_timestamp"] = time.time()
                
                # [IMMUNE 2.0] 6.5. Systemic Pulse Audit (Deterministic)
                if self._tick_count % 100 == 0: # Every ~50s in conversational mode
                    await self._immune_pulse_audit()

                # 7. If state changed, commit it
                if current_state and state and current_state.state_id != state.state_id:
                    from .state.state_repository import StateVersionConflictError
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
                                # Meatiness check: don't emit "null", repetition fragments, or action leakage
                                is_meaty = len(content.strip()) > 5 or any(c.isalpha() for c in content)
                                has_null = "null" in content.lower()
                                has_action = "say '" in content.lower() or "do '" in content.lower()

                                if is_meaty and not has_null and not has_action:
                                    # [CONSTITUTIONAL] Route through ExecutiveAuthority — not output_gate
                                    try:
                                        from core.consciousness.executive_authority import get_executive_authority
                                        authority = get_executive_authority(self.orchestrator)
                                        await authority.release_expression(
                                            content,
                                            source="mind_tick_autonomous",
                                            urgency=0.5,
                                            target="primary",
                                            metadata={"autonomous": True, "spontaneous": True},
                                        )
                                    except Exception as _ea_err:
                                        logger.debug("MindTick: ExecutiveAuthority emission failed: %s", _ea_err)
                    except StateVersionConflictError:
                        # For MindTick, we can safely ignore conflicts; the next tick will catch up
                        logger.debug("💓 MindTick: Skipping commit due to concurrent update (Atomic Guard).")
                    except Exception as e:
                        logger.error(f"❌ MindTick: Commit failed: {e}")
                        
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
                            # Run as fire-and-forget task so we don't block the tick
                            # Ensure background flag is passed
                            get_task_tracker().track_task(
                                asyncio.create_task(
                                    memory_coord.consolidate_working_memory(current_state, is_background=True),
                                    name="mind_tick.consolidate_working_memory",
                                )
                            )
                        self.set_mode(CognitiveMode.SLEEP)
                
                # NOTE: tick_count and cycle_count increment moved to finally block
                # to guarantee the heartbeat always advances.

                metadata.duration = asyncio.get_running_loop().time() - start_time
                self._last_tick_metadata = metadata
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("⚠️ MindTick Loop Error: %s", e)
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
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            finally:
                # [CRITICAL FIX] ALWAYS increment tick and cycle count, even on failure.
                # Without this, any exception traps the system at Cycle 1 forever,
                # which also prevents the MetabolicCoordinator's grace period
                # (skip if cycle < 5) from ever expiring.
                self._tick_count += 1
                try:
                    if hasattr(self.orchestrator, 'status') and self.orchestrator.status:
                        current_c = getattr(self.orchestrator.status, 'cycle_count', 0)
                        self.orchestrator.status.cycle_count = current_c + 1
                except Exception:
                    pass  # Non-critical; never let this block the loop
                
                # Wait for the next tick based on mode.
                # Floor at 0.5s to prevent CPU saturation when tick work
                # approaches the interval — the event loop needs breathing room.
                interval = sleep_time_override or TICK_INTERVALS.get(self.mode, 1.0)
                elapsed = asyncio.get_running_loop().time() - start_time
                sleep_time = max(0.5, interval - elapsed)
                await asyncio.sleep(sleep_time)

    def _get_actual_from_state(self, state: AuraState) -> Optional[str]:
        """Extract the last actual cognitive output for prediction evaluation."""
        if state.cognition.working_memory:
            # We look for the most recent message
            last_msg = state.cognition.working_memory[-1]
            return last_msg.get("content")
        return None

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
            pid_file = Path("aura.pid")
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        logger.warning("💉 [IMMUNE] Pulse Audit: Found stale PID file for non-existent process %d", pid)
                        immunity.registry.match_and_repair("PID file already exists")
                except (ValueError, OSError) as e:
                    logger.debug(f"MindTick: PID check failed (likely stale/malformed): {e}")

            # 2. Resource Leak Probe (Memory)
            import psutil
            process = psutil.Process()
            mem_pct = process.memory_percent()
            if mem_pct > 25.0: # Trigger cleanup if one process exceeds 25% RAM
                logger.warning("💉 [IMMUNE] Pulse Audit: High memory usage detected (%.1f%%). Triggering conservative sweep.", mem_pct)
                # Proactive cleanup trigger
                import gc
                gc.collect()
            
            # 3. Log Sieve (Look for systemic issues)
            from core.config import config
            log_dir = config.paths.data_dir / "error_logs"
            if log_dir.exists():
                logs = list(log_dir.glob("*.log"))
                hidden = immunity.registry.log_sieve(logs)
                if hidden:
                    logger.warning("💉 [IMMUNE] Log Sieve detected %d hidden issues.", len(hidden))

            breaker.record_success()
        except Exception as e:
            logger.error("💉 [IMMUNE] Pulse Audit failed: %s", e)
            if breaker: breaker.record_failure()
