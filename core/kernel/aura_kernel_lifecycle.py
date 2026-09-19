"""The body of a tick, and the way the kernel comes down.

Lifted whole out of `aura_kernel`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any


class _TicksAndShutsDown:
    """Lifted whole out of AuraKernel; see aura_kernel.py."""

    def _tick_body_part_1(self, bound_objective, priority, state):
        from .aura_kernel import (
            _begin_pass_run,
            get_executive_authority,
            open_tick,
        )

        state.cognition.current_objective = bound_objective
        get_executive_authority().record_objective_binding(
            state,
            bound_objective,
            source="aura_kernel.tick",
            mode="unitary_tick",
            reason="kernel_tick_bound",
        )

        # Linear Pipeline execution
        volition = self.volition_level

        # Phases that only belong in background autonomous ticks.
        # Running them during a user-facing (priority) tick blocks the response
        # for up to 60s per phase and is never needed for conversation.
        #
        # Pass numbering restarts here. It used to be monotonic for the
        # process, which made AURA_PASS_BISECT_LIMIT=5 mean "the first
        # five passes since boot" — right on the first tick and total
        # silence on every tick after it. The documented behaviour, and
        # the only useful one, is per-tick.
        _begin_pass_run(f"kernel_tick/{'priority' if priority else 'background'}")
        # Same shape and the same reason as the pass run above: one record
        # per tick, opened here, so "why did she do that" can be answered
        # from what the runtime measured rather than from what the model
        # would say about itself afterwards.
        _provenance = open_tick(objective=bound_objective, priority=priority)
        return _provenance, volition

    async def _tick_body_part_2(self, _provenance, entry, objective, start_time, turn_origin):
        from .aura_kernel import (
            ServiceContainer,
            _record_kernel_degradation,
            close_tick,
            logger,
        )

        close_tick(_provenance)

        # Cognitive health is a materialized projection of the completed
        # state, not a write owned by every phase that derives a state.
        # Refresh it once outside phase provenance so contracts attribute
        # only the transformations each phase actually performed.
        refresh_cognitive_health = getattr(
            self.state,
            "_refresh_cognitive_health",
            None,
        )
        if callable(refresh_cognitive_health):
            try:
                refresh_cognitive_health()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_kernel_degradation(
                    exc,
                    action="completed tick with the prior cognitive-health projection",
                    severity="warning",
                )

        # Flush deferred storage side-effects (eternal_append, db_write, etc.)
        # [STABILITY v53] Timeout guard — storage intents can hang on slow I/O
        try:
            await asyncio.wait_for(self._process_storage_intents(), timeout=10.0)
        except TimeoutError:
            logger.warning(
                "⚠️ [STABILITY] Storage intents timed out (10s) — skipping for this tick."
            )

        # ── CONSTITUTIONAL CLOSURE ──────────────────────────────────────
        # Stamp this tick's arbitration into the canonical state before commit.
        # Every committed state is self-documenting about the decision chain.
        try:
            self.state.cognition.last_kernel_cycle_id = entry.tick_id if entry else None
            self.state.cognition.last_action_source = (
                self.state.cognition.current_origin or "kernel"
            )

            from core.executive.executive_core import get_executive_core

            _exec = get_executive_core()
            if _exec is not None:
                _exec_stats = _exec.get_stats() if hasattr(_exec, "get_stats") else {}
                self.state.cognition.kernel_decision_count = int(
                    _exec_stats.get("approved", 0) or 0
                )
                self.state.cognition.kernel_veto_count = int(
                    _exec_stats.get("rejected", 0) or 0
                )
                _recent = _exec_stats.get("recent_decisions", []) or []
                self.state.cognition.last_veto_reasons = [
                    str(d.get("reason", ""))
                    for d in _recent
                    if isinstance(d, dict) and d.get("outcome") == "rejected"
                ][-5:]
        except (ImportError, AttributeError, RuntimeError) as _cc_err:
            _record_kernel_degradation(
                _cc_err,
                action="continued tick without constitutional closure state stamp",
                severity="error",
            )
            logger.error("Constitutional closure stamp failed: %s", _cc_err, exc_info=True)
        # ────────────────────────────────────────────────────────────────

        # A foreground objective is a live turn, not a durable autonomous
        # goal. Close it before persistence so proxy serialization cannot
        # race the post-return cleanup.
        self._finalize_foreground_turn_state(
            objective=objective,
            turn_origin=turn_origin,
        )

        # Persistence
        # [STABILITY v53] Timeout guard — vault commit can hang on slow disk/network
        try:
            await asyncio.wait_for(self._commit_vault(objective), timeout=10.0)
        except TimeoutError:
            logger.warning(
                "⚠️ [STABILITY] Vault commit timed out (10s) — state not persisted this tick."
            )

        # Cognitive Ledger: record this tick as a structured transition
        try:
            from core.resilience.cognitive_ledger import (
                Transition,
                TransitionType,
                compute_state_hash,
                get_cognitive_ledger,
            )

            ledger = get_cognitive_ledger()
            state_hash = compute_state_hash(self.state)
            ledger.append(
                Transition.create(
                    ttype=TransitionType.TICK_COMPLETE,
                    subsystem="kernel",
                    cause=objective[:120] if objective else "tick",
                    payload={
                        "phi": round(self.state.phi, 4),
                        "valence": round(self.state.affect.valence, 3),
                        "mode": self.state.cognition.current_mode.value,
                        "response_len": len(self.state.cognition.last_response or ""),
                        "cycle": self.status.cycle_count,
                    },
                    prior_hash=state_hash,
                    confidence=1.0
                    - (self.state.free_energy if hasattr(self.state, "free_energy") else 0.0),
                )
            )
        except (ImportError, AttributeError, RuntimeError) as _ledger_err:
            _record_kernel_degradation(
                _ledger_err,
                action="completed tick without cognitive ledger transition",
            )
            logger.debug("Ledger tick record failed (non-critical): %s", _ledger_err)

        # Visual Update
        await self._pulse_mirror()

        # 2. Feedback Loop: End
        response = self.state.cognition.last_response
        self.feedback_observer.end_tick(entry, response, self.state, start_time)

        # Record phase health in StabilityGuardian
        try:
            if self._guardian is None:
                from core.container import ServiceContainer

                self._guardian = ServiceContainer.get("stability_guardian", default=None)

            if self._guardian:
                self._guardian.record_tick_health(entry)
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_kernel_degradation(
                e,
                action="completed tick without stability guardian health record",
            )
            logger.debug("StabilityGuardian: Health record skipped: %s", e)

        # Log the loop summary
        logger.info("LOOP| %s", entry.summary())

        await self._trace_the_tick(objective, response)

        # Record completion timestamp for telemetry staleness detection
        self._last_tick_completed_at = time.time()

    async def _tick_body(self, objective, priority, turn_origin, state):
        """Body lifted verbatim out of ``AuraKernel.tick``.

        Moved by tools/extract_seam.py, which refuses to write unless the
        relocated body diffs clean against the original. The seam was
        5 names in, 0 out, 1 early return(s), 5 awaits.
        """
        from .aura_kernel import (
            BondingPhase,
            EternalGrowthEngine,
            LearningPhase,
            RepairPhase,
            SelfReviewPhase,
            TrueEvolutionPhase,
            _await_phase_completion,
            _pass_instrumentation,
            _record_kernel_degradation,
            _record_pass,
            get_task_tracker,
            is_shutdown_requested,
            logger,
            sys,
        )

        try:
            # Priority request acquired the lock — clear the pending flag
            if priority:
                self._user_priority_pending.clear()

            start_time = time.time()
            logger.info("🌀 Unitary Tick Initiated: '%s' (priority=%s)", objective, priority)

            # 1. Feedback Loop: Begin
            entry = self.feedback_observer.begin_tick(
                state,
                objective,
                origin=str(getattr(state.cognition, "current_origin", "") or ""),
                priority=bool(priority),
            )

            # Initial derivation for the tick itself
            state = await state.derive_async(f"tick_start: {objective[:50]}", origin="tick")
            # Objective/initiative hygiene is a tick-lifecycle operation. It
            # must happen before phase provenance starts; doing it inside
            # AuraState.derive() falsely attributed the cleanup to whichever
            # phase happened to derive next.
            state.prepare_tick_boundary()
            # Clear per-turn prompt/runtime modifiers from previous ticks so
            # stale tool results, open social-thread directives, recovery
            # flags, or proof contracts cannot leak into an unrelated turn.
            try:
                from core.runtime.proof_policy import (
                    clear_transient_response_modifiers,
                    is_proof_repair_prompt,
                    proof_persistent_objective,
                    proof_run_active,
                )

                proof_active = proof_run_active(origin=turn_origin)
                bound_proof_objective = proof_persistent_objective(
                    objective,
                    origin=turn_origin,
                )
                clear_transient_response_modifiers(
                    state.response_modifiers,
                    strict=proof_active,
                )
                if proof_active:
                    state.response_modifiers["proof_turn_objective"] = bound_proof_objective
                    if is_proof_repair_prompt(objective, origin=turn_origin):
                        state.response_modifiers["proof_repair_turn"] = True
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_kernel_degradation(
                    exc,
                    action="continued tick after transient response-modifier scrub failed",
                    severity="error",
                )
                for _stale_key in (
                    "last_skill_run",
                    "last_skill_ok",
                    "last_skill_result_payload",
                    "matched_skills",
                    "intent_type",
                    "precomputed_grounded_reply",
                    "last_task_outcome",
                    "last_task_id",
                    "auto_browse_urls",
                    "conversational_dynamics",
                    "conv_dynamics_state",
                    "response_contract",
                ):
                    state.response_modifiers.pop(_stale_key, None)
            self.state = state

            # CASIE: Score user objective for strategy
            tricorder = self.organs.get("tricorder")
            if tricorder and tricorder.instance:
                casie = tricorder.instance.score_user_message(objective)
                logger.info("🎭 [CASIE] Strategy: %s - %s", casie["strategy"], casie["description"])

            # [SEVERANCE] Apply Persona Masking to Cognitive Cycle
            partition = state.context_partition
            if state.partition_mask:
                logger.info(
                    "🎭 [SEVERANCE] Executing in %s partition. Field masking ACTIVE.", partition
                )

            try:
                from core.runtime.proof_policy import proof_persistent_objective

                bound_objective = proof_persistent_objective(
                    objective,
                    origin=turn_origin,
                )
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_kernel_degradation(
                    exc,
                    action="continued tick after proof objective binding normalization failed",
                    severity="error",
                )
                bound_objective = objective
            _provenance, volition = self._tick_body_part_1(bound_objective, priority, state)
            for phase in self._phases:
                phase_name = phase.__class__.__name__

                # [PRIORITY PREEMPTION] If a user message is now waiting for the
                # kernel lock, yield immediately after the current phase completes.
                if not priority and self._user_priority_pending.is_set():
                    logger.info(
                        "⚡ Background tick yielding to priority user request — aborting remaining phases after %s.",
                        phase_name,
                    )
                    await self._record_dream_fragment(objective, phase, phase_name)
                    break

                # Skip background-only phases during user-facing ticks so the
                # response pipeline runs without waiting for slow autonomous work.
                if self._should_skip_priority_phase(phase_name, priority=priority):
                    continue

                # Volition-based Gating
                # Level 0: Lockdown (Standard pipeline only)
                # Level 1: Reflective (Adds Self-Review)
                # Level 2: Perceptive (Adds Learning/Repair)
                # Level 3: Agentic (Adds Growth/Evolution)
                if volition < 3 and isinstance(phase, (EternalGrowthEngine, TrueEvolutionPhase)):
                    continue
                if volition < 2 and isinstance(phase, (LearningPhase, RepairPhase, BondingPhase)):
                    continue
                if volition < 1 and isinstance(phase, SelfReviewPhase):
                    continue

                # [PASS INSTRUMENTATION] The one seam every phase announces
                # itself through. It carries pass timing, and it carries
                # `-opt-bisect-limit`: when a turn comes out wrong, binary-
                # searching the limit finds which of ~30 phases did it in a
                # handful of runs instead of an afternoon of guessing.
                # See core/pipeline/pass_manager.py.
                _pass_run, _pass_ordinal, _pass_reason = _pass_instrumentation().should_run(
                    f"kernel_tick/{phase_name}"
                )
                if not _pass_run:
                    _record_pass(phase_name, _pass_ordinal, 0.0, skipped=True, reason=_pass_reason)
                    continue
                _pass_started = time.perf_counter()

                # Strict Lineage: Each phase execution derives a new state version.
                # Use asyncio.shield() so that if the outer task is cancelled the
                # inner phase coroutine is NOT cancelled — preventing CancelledError
                # from reaching MLX workers and triggering unnecessary worker reboots.
                # Each MLX call has its own internal timeout (45 s for background),
                # so phases will complete or fail on their own without kernel-level
                # cancellation.
                try:
                    phase_task = get_task_tracker().create_task(
                        self._execute_phase_with_timing(
                            phase,
                            phase_name,
                            entry,
                            objective=objective,
                            priority=priority,
                        ),
                        name=f"AuraKernel.{phase_name}",
                    )
                    try:
                        phase_timeout = self._phase_timeout_seconds(phase_name, priority=priority)

                        # [STABILITY v50] FAST PREEMPTION: When a priority user
                        # request is pending, cap background phase budgets at 5s
                        # so the user doesn't wait 45s+ for a background tick to
                        # finish. Response phases get a hard 5s cap; other phases
                        # get 8s. This is the #1 fix for kernel lock contention.
                        if not priority and self._user_priority_pending.is_set():
                            if phase_name in {"UnitaryResponsePhase", "ResponseGenerationPhase"}:
                                phase_timeout = min(phase_timeout, 5.0)
                            else:
                                phase_timeout = min(phase_timeout, 8.0)

                        result = await _await_phase_completion(
                            phase_task, phase_name=phase_name, priority=priority,
                            origin=turn_origin, budget_s=phase_timeout,
                        )
                        self.state = result
                    except TimeoutError:
                        logger.error(
                            "⏰ Phase '%s' timed out after %.0fs — skipping",
                            phase_name,
                            phase_timeout,
                        )
                        try:
                            if "phase_task" in locals() and not phase_task.done():
                                phase_task.cancel()
                        except (AttributeError, RuntimeError) as _exc:
                            logger.debug(
                                "Suppressed %s while cancelling timed-out phase task: %s",
                                type(_exc).__name__,
                                _exc,
                            )
                        if not priority and phase_name in {
                            "UnitaryResponsePhase",
                            "ResponseGenerationPhase",
                        }:
                            logger.info(
                                "⚡ Background tick ending early after %s timeout so stale response generation does not pin the foreground lane.",
                                phase_name,
                            )
                            await self._record_dream_fragment(objective, phase, phase_name)
                            break
                        if not priority and self._user_priority_pending.is_set():
                            logger.info(
                                "⚡ Background tick releasing kernel lock after timed-out %s for a waiting priority request.",
                                phase_name,
                            )
                            await self._record_dream_fragment(objective, phase, phase_name)
                            break
                        continue
                except asyncio.CancelledError as phase_err:
                    if priority or is_shutdown_requested():
                        try:
                            if "phase_task" in locals() and not phase_task.done():
                                phase_task.cancel()
                        except (AttributeError, RuntimeError) as _exc:
                            logger.debug("Suppressed %s in core.kernel.aura_kernel: %s", type(_exc).__name__, _exc)
                        logger.warning(
                            "⏹️ Priority kernel tick cancelled during %s; propagating caller timeout/cancellation.",
                            phase_name,
                        )
                        raise
                    _record_kernel_degradation(
                        phase_err,
                        action=f"skipped background phase {phase_name} after cancellation",
                        severity="warning",
                    )
                    logger.warning(
                        "Background phase '%s' was cancelled; continuing tick.", phase_name
                    )
                    continue
                except (
                    RuntimeError,
                    TimeoutError,
                    AttributeError,
                ) as phase_err:
                    _record_kernel_degradation(
                        phase_err,
                        action=f"skipped phase {phase_name} after unexpected phase failure",
                        severity="error",
                    )
                    logger.error(
                        "🔥 Phase '%s' raised unexpected error: %s",
                        phase_name,
                        phase_err,
                        exc_info=True,
                    )
                    # Don't let a single phase crash the entire tick — skip and continue
                    continue
                finally:
                    # Runs on the success path and on every early exit, so a
                    # phase that times out is still timed. Skipped phases are
                    # recorded above and never reach here.
                    _record_pass(
                        phase_name,
                        _pass_ordinal,
                        time.perf_counter() - _pass_started,
                        skipped=False,
                        error=repr(sys.exc_info()[1]) if sys.exc_info()[1] is not None else "",
                    )

                if self.state is None:
                    raise RuntimeError(f"Phase {phase_name} returned None state")

                self.state.updated_at = time.time()

            await self._tick_body_part_2(_provenance, entry, objective, start_time, turn_origin)

            return entry
        finally:
            if self._lock.locked():
                self._lock.release()

    async def _shutdown_impl(self, *, finalize_process_runtime: bool) -> None:
        """Execute the kernel-owned portion of shutdown exactly once."""
        from .aura_kernel import (
            _KERNEL_OPTIONAL_PERCEPTION_ERRORS,
            _record_kernel_degradation,
            logger,
        )

        if (
            not self._running
            and not self._background_tasks
            and not any(
                getattr(organ, "instance", None) is not None for organ in self.organs.values()
            )
        ):
            return

        try:
            from core.runtime.shutdown_coordinator import request_shutdown

            request_shutdown("aura_kernel.shutdown")
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued kernel shutdown without setting global shutdown request",
            )

        logger.info("🛑 [KERNEL] Initiating graceful shutdown...")
        self._running = False
        self.status.running = False

        # Stop PerceptionDaemon
        try:
            from core.perception.perception_daemon import get_perception_daemon
            daemon = get_perception_daemon()
            await daemon.stop()
            logger.info("📡 [PERCEPTION] PerceptionDaemon OFFLINE")
        except _KERNEL_OPTIONAL_PERCEPTION_ERRORS as e:
            _record_kernel_degradation(
                e,
                action="continued kernel shutdown after PerceptionDaemon stop failed",
                severity="warning",
            )
            logger.error("Failed to stop PerceptionDaemon: %s", e)

        # 1. Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks = []

        # 2. Shutdown organs
        for name, organ in self.organs.items():
            try:
                if hasattr(organ, "shutdown"):
                    await organ.shutdown()
                logger.info("🫀 Organ %s shut down.", name)
            except (RuntimeError, AttributeError, TypeError) as e:
                _record_kernel_degradation(
                    e,
                    action=f"continued shutdown after organ {name} shutdown failed",
                    severity="error",
                )
                logger.error("Error shutting down organ %s: %s", name, e)

        # 3. Stop singleton runtime services booted or activated by kernel ticks.
        await self._shutdown_rubicon_runtime()

        # 4. Close the state vault owned by this kernel instance.
        await self._close_kernel_vault()

        # 5. Stop process-wide event/task/runtime hygiene surfaces so an isolated
        # kernel boot leaves no background loops behind after shutdown.  The
        # orchestrator defers these finalizers until its service owners stop.
        await self._shutdown_process_runtime(
            finalize_process_runtime=finalize_process_runtime,
        )

        logger.info("✅ [KERNEL] Shutdown complete.")

    async def _shutdown_rubicon_runtime(self) -> None:
        """Stop Rubicon singletons that are started from AuraKernel.boot()."""
        from .aura_kernel import (
            _record_kernel_degradation,
        )

        runtime_targets: list[tuple[str, Any]] = []
        target_specs = (
            ("feedback_processor", "core.somatic.action_feedback", "_feedback_processor_instance"),
            ("motor_cortex", "core.somatic.motor_cortex", "_motor_cortex_instance"),
            ("pre_linguistic", "core.cognition.pre_linguistic", "_pre_linguistic_instance"),
        )
        for label, module_name, attr_name in target_specs:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                target = getattr(module, attr_name, None)
                if target is not None:
                    runtime_targets.append((label, target))
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_kernel_degradation(
                    exc,
                    action=f"skipped {label} shutdown after singleton lookup failed",
                )

        for label, target in runtime_targets:
            await self._call_shutdown_hook(label, target, "shutdown", "stop", "close")

    async def _shutdown_process_runtime(
        self,
        *,
        finalize_process_runtime: bool = True,
    ) -> None:
        """Drain process services without violating root-owner teardown order."""
        from .aura_kernel import (
            _record_kernel_degradation,
            get_task_tracker,
            logger,
        )

        try:
            import core.learning.live_learner as live_learner_module

            learner = getattr(live_learner_module, "_learner", None)
            if learner is not None:
                await self._call_shutdown_hook("live_learner", learner, "shutdown", "stop", "close")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="skipped live learner shutdown after singleton lookup failed",
            )

        try:
            from core.resilience.lock_watchdog import get_lock_watchdog

            await self._call_shutdown_hook("lock_watchdog", get_lock_watchdog(), "shutdown", "stop", "close")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="skipped lock watchdog shutdown after singleton lookup failed",
            )

        if not finalize_process_runtime:
            logger.info(
                "Kernel process-wide finalizers deferred to orchestrator root shutdown."
            )
            return

        try:
            from core.event_bus import get_event_bus

            await self._call_shutdown_hook("event_bus", get_event_bus(), "shutdown", "stop", "close")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="skipped event bus shutdown after singleton lookup failed",
            )

        try:
            await get_task_tracker().shutdown(timeout=5.0)
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            _record_kernel_degradation(
                exc,
                action="continued shutdown after task tracker drain failed",
                severity="error",
            )
            logger.warning("Task tracker shutdown failed: %s", exc)

        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            await self._call_shutdown_hook("runtime_hygiene", get_runtime_hygiene(), "shutdown", "stop", "close")
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_kernel_degradation(
                exc,
                action="skipped runtime hygiene shutdown after singleton lookup failed",
            )

