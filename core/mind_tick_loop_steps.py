"""The steps one mind tick is assembled from.

Lifted whole out of `mind_tick`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any


class _RunsTheTickLoopSteps:
    """Lifted whole out of MindTick; see mind_tick.py."""

    async def _run_loop_part_1(self) -> Any:
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            _record_mind_degradation,
            logger,
            record_degraded_event,
            run_on_a_thread_while_it_works,
        )

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
            await run_on_a_thread_while_it_works(
                get_world_state().update, stall_s=5.0, name="mind_tick.world_state"
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
        return health_pause

    async def _run_loop_gate(self) -> Any:
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            _dead_tiers_are_policy_deferred_cortex,
            _record_mind_degradation,
            logger,
            resolve_inference_gate,
        )

        gate = resolve_inference_gate()
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
        return gate

    async def _run_loop_bridge_event_bus(self, metadata: Any, start_time: Any) -> float:
        # 4. Bridge to Event Bus (for UI/Observability)
        # Circuit-breaker: after repeated failures, back off to avoid
        # flooding the resilience engine with degradation events.
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            logger,
        )

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
        return audit_interval

    @staticmethod
    def _run_loop_check_sidecar_process() -> Any:
        # Check sidecar process health
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            ServiceContainer,
            _record_mind_degradation,
            logger,
            resolve_inference_gate,
        )

        local_runtime_state = "offline"
        try:
            gate = resolve_inference_gate()
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
        return local_runtime_state

    async def _run_loop_these_run_during(self) -> None:
        # These run during dream consolidation, not on
        # every tick — but they ran ON THE LOOP:
        # metacognitive assessment, value-graph
        # evolution, a hidden eval suite and STDP
        # diagnostics, each of which can take seconds.
        # A dream is background work by definition, so
        # it goes off-thread and bounded rather than
        # parking the rhythm for the duration.
        from .mind_tick import (
            CognitiveMode,
            _schedule_mind_task,
            logger,
            record_degraded_event,
        )

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

    def _run_loop_always_advance_heartbeat(
        self,
        sleep_time_override: Any,
        start_time: Any,
    ) -> tuple[Any, Any]:
        # Always advance heartbeat counters, even on degraded ticks.
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            TICK_INTERVALS,
            _record_mind_degradation,
            logger,
        )

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
        return elapsed, interval

    async def _run_loop_health_pause(self) -> None:
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            _record_mind_degradation,
            logger,
            record_degraded_event,
        )

        health_pause = await self._run_loop_part_1()
        if self._tick_count % 10 == 0 and not health_pause:
            try:
                await self._run_loop_gate()
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

