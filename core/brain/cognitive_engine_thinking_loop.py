"""The background thinking loop and the clock it keeps.

Lifted whole out of `cognitive_engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cognitive_engine import (
        AuraState,
        ThinkingMode,
        Thought,
    )


class _RunsTheThinkingLoop:
    """Lifted whole out of CognitiveEngine; see cognitive_engine.py."""

    def _run_thinking_loop_part_1(self, context, objective, origin, state):
        """
        Internal method to execute the core cognitive phase loop.
        Extracted from `think` to allow pre/post-processing in `think`.
        """
        from .cognitive_engine import (
            _bind_live_mind_generation_contract,
        )

        if not isinstance(context, dict):
            context = {}
        foreground_turn_objective = str(objective or "")
        from core.conversation.user_surface_contract import (
            bind_user_surface_prompt,
            resolve_user_surface_prompt,
        )

        surface_prompt = resolve_user_surface_prompt(context)
        if not surface_prompt.bound:
            bind_user_surface_prompt(
                context,
                surface_prompt.prompt or objective,
                source="cognitive_engine.visible_user_message",
                overwrite=True,
            )
        _bind_live_mind_generation_contract(context)

        append_user_message = True
        append_user_message = not bool(
            context.get("suppress_user_memory_append")
            or context.get("suppress_working_memory_user_append")
        )
        self._thinking_loop_surface_prompt(append_user_message, context, objective, origin, state, surface_prompt)
        return context, foreground_turn_objective

    def _run_thinking_loop_part_2(self, context, cycle_timeout, is_background, objective, origin):
        from .cognitive_engine import (
            _DEFAULT_COGNITIVE_CYCLE_MAX_S,
            _time_the_answer_needs,
            response_policy,
        )

        if cycle_timeout <= 0.0:
            if self._is_user_facing_origin(origin):
                cycle_timeout = 180.0
            elif is_background:
                cycle_timeout = 90.0
            else:
                cycle_timeout = 240.0
        cycle_timeout_cap = _DEFAULT_COGNITIVE_CYCLE_MAX_S
        if self._is_user_facing_origin(origin):
            # The ceiling a turn somebody is waiting for may reach, whether or
            # not the caller named a timeout. It used to depend on that, so an
            # ordinary question — nobody passes a timeout for one — was held
            # to 240 seconds while the machinery below it agreed the answer
            # needed longer, and the wait it is nested in already allows 480.
            #
            # A ceiling, not a target: the floor below still decides what this
            # particular turn asks for, and a turn that finishes sooner
            # finishes sooner.
            cycle_timeout_cap = response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S
        # This is the clock that OWNS the turn, and it was a flat number chosen
        # before anything knew what the answer would cost. Everything below it
        # is nested inside it, so extending any of those could not help: the
        # inference gate raised its own deadline to 345 seconds and the turn
        # still ended at 186, because 180 was set here.
        #
        # LIVE, 2026-08-28: a tool loop reached code_repl, the code raised
        # NameError for a missing import, the error was handed back for the
        # model to fix, and the turn was cut before it could. Four clocks were
        # involved and only one of them was the owner.
        #
        # It takes its floor from the same reading the others do now: what the
        # answer this request needs costs to decode, at the rate this machine
        # has been measured at. An unmeasured rate raises nothing.
        if self._is_user_facing_origin(origin) and not is_background:
            cycle_timeout = max(cycle_timeout, _time_the_answer_needs(objective))
        cycle_timeout = max(8.0, min(cycle_timeout_cap, cycle_timeout))
        cycle_deadline_at = time.monotonic() + cycle_timeout
        context["cognitive_cycle_deadline_monotonic"] = cycle_deadline_at
        return cycle_deadline_at, cycle_timeout

    def _run_thinking_loop_part_3(self, direct_quick_reply, origin, state, success, temp_state):
        from .cognitive_engine import (
            record_response_path,
        )

        if direct_quick_reply is not None:
            # The quick lane returned before any phase executed. Whether the
            # model was called depends on which branch inside it answered: the
            # canonical floors return pre-rendered text and never reach it.
            record_response_path(
                str(
                    (direct_quick_reply.metadata or {}).get("response_path")
                    or "desktop_quick_reply"
                ),
                model_generation=bool(
                    (direct_quick_reply.metadata or {}).get(
                        "live_mind_generation_required", True
                    )
                ),
            )
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": direct_quick_reply.content,
                    "timestamp": time.time(),
                    "origin": origin,
                }
            )
            if self._is_user_facing_origin(origin):
                state.transition_origin = origin
                state.cognition.current_origin = origin
            temp_state = state
            success = True
        return success, temp_state

    def _run_thinking_loop__clock_keeper(self, _cycle_clock, context, is_background, objective, origin):
        from .cognitive_engine import (
            _begin_pass_run,
            _keep_the_cycle_open_while_it_is_working,
            _open_provenance_tick,
            _the_longest_this_turn_may_take,
            create_owned_asyncio_task,
            response_policy,
        )

        _clock_keeper = create_owned_asyncio_task(
            _keep_the_cycle_open_while_it_is_working(
                _cycle_clock,
                ceiling_at=time.monotonic()
                + _the_longest_this_turn_may_take(
                    float(
                        response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S
                    ),
                    user_facing=bool(
                        self._is_user_facing_origin(origin)
                        and not is_background
                    ),
                ),
                user_facing=bool(
                    self._is_user_facing_origin(origin) and not is_background
                ),
                runtime_context=context,
            )
        )
        _begin_pass_run("legacy_pipeline")
        # The provenance graph had the same asymmetry the pass
        # instrumentation had, for the same reason: it was opened
        # in AuraKernel.tick, and chat drives THIS loop. So the
        # causal record that answers "why did she do that" existed
        # for the three turns a day the kernel runs and not for the
        # several hundred a person has. Same seam, same graph.
        _provenance_tick = _open_provenance_tick(
            objective=objective, priority=self._is_user_facing_origin(origin)
        )
        return _clock_keeper, _provenance_tick

    @staticmethod
    async def _run_thinking_loop_started_at(context, kwargs, objective, ordinal, phase, phase_name, temp_state):
        from .cognitive_engine import (
            _begin_provenance,
            _complete_provenance,
            _record_legacy_pass,
            record_latency,
            record_phase,
        )

        started_at = time.perf_counter()
        # Measured around the phase, never reported by it.
        _transformation = _begin_provenance(phase_name, temp_state)
        _phase_error = ""
        try:
            # Pass through kwargs like is_background if phases support it
            temp_state = await phase.execute(
                temp_state,
                objective=objective,
                context=context,
                **kwargs,
            )
        except BaseException as phase_exc:
            _phase_error = f"{type(phase_exc).__name__}: {phase_exc}"
            _record_legacy_pass(
                phase_name,
                ordinal,
                time.perf_counter() - started_at,
                skipped=False,
                error=_phase_error,
            )
            raise
        finally:
            _complete_provenance(
                _transformation,
                temp_state,
                error=_phase_error,
                objective=objective,
            )
        phase_elapsed = time.perf_counter() - started_at
        _record_legacy_pass(
            phase_name,
            ordinal,
            phase_elapsed,
            skipped=False,
        )
        # Marked after the phase returns, so a phase that timed
        # out mid-execution is not recorded as having run.
        record_phase(phase_name)
        # The retrieval phase is one of the five components a
        # turn's latency is split into (R11), and its duration
        # was only ever a "phase latency exceeded budget" line.
        if "retrieval" in str(phase_name).lower():
            try:
                record_latency("retrieval", phase_elapsed)
            except ValueError:
                pass
        return temp_state

    async def _run_thinking_loop_closed_rather_after(self, _clock_keeper, _provenance_tick, backup_state, state, success):
        # Closed here rather than after the loop so a tick that timed
        # out or crashed still lands in the ring. Those are the ticks
        # somebody most wants to read afterwards, and the version that
        # closed on the success path recorded only the turns that went
        # well.
        #
        # And the task holding the cycle clock open, however the turn
        # ended. It is bounded on its own, but a turn that finished in
        # two seconds should not leave something watching for eight
        # minutes.
        from .cognitive_engine import (
            _close_provenance_tick,
            logger,
            record_degradation,
            suppress,
        )

        if _clock_keeper is not None:
            _clock_keeper.cancel()
            with suppress(asyncio.CancelledError):
                await _clock_keeper
        _close_provenance_tick(_provenance_tick)
        try:
            # vResilience: Avoid locals().get() for type stability
            if not success and "backup_state" in locals():
                # This restores a LOCAL REFERENCE and nothing else.
                #
                # A deep copy of the state object cannot undo what a
                # phase already did outside it: events published, tools
                # invoked, rows written, in-place mutations to
                # collaborators the phase was handed. Calling this
                # "rollback" invites the next reader to rely on a
                # transaction that does not exist, so the receipt says
                # what was and was not restored.
                state = backup_state
                self._last_phase_rollback = {
                    "restored": "cognitive_state_snapshot",
                    "not_restored": [
                        "external_service_writes",
                        "published_events",
                        "tool_invocations",
                        "database_rows",
                        "in_place_collaborator_mutations",
                    ],
                    "at": time.time(),
                }
                # A receipt about the recovery, not a second fault.
                #
                # cognitive_engine is a required subsystem, so a
                # warning here escalates: this note became CRITICAL
                # SERVICE FAILURE, which aborted the whole turn and
                # answered the person "I couldn't get to an answer
                # I'd stand behind." The phase failure itself is
                # recorded where it happens; this line only says what
                # the restore did and did not cover, and saying so
                # must not cost the turn it was trying to save.
                record_degradation(
                    "cognitive_engine",
                    RuntimeError("phase_failure_partial_rollback"),
                    severity="info",
                    action="restored the cognitive state snapshot; external phase effects are not reversible here",
                )
        except (OSError, ConnectionError, TimeoutError) as _e:
            record_degradation(
                "cognitive_engine",
                _e,
                severity="warning",
                action="continued with current state after backup restore check failed",
            )
            logger.debug("Ignored Exception in cognitive_engine.py: %s", _e)
        return state

    async def _run_thinking_loop_should_bypass_commit(self, context, cycle_deadline_at, is_test_run, origin, pre_turn_cognition, state, temp_state):
        from .cognitive_engine import (
            _commit_the_thought_with_retries,
        )

        should_bypass_commit = is_test_run or self.state_repository is None
        # The watchdog above wraps PHASE EXECUTION only. Repository reads,
        # advisors, spine checks, augmentors, the deep copy, this commit loop
        # and feedback learning all run outside it, so a configured cycle
        # timeout was never the end-to-end budget it reads as. The commit loop
        # is the largest of those — three attempts, each a database round trip
        # — so it gets what is left of the same deadline instead of an
        # unbounded wait after the budget is already gone.


        # What actually happened to durable state. Every exit from this loop
        # used to be a bare `break`, after which extraction returned a
        # 0.9-confidence "completed successfully" thought whether the commit
        # had landed, been bypassed, exhausted its retries, or raised.
        commit_outcome = "not_attempted"
        max_retries = 3
        commit_outcome, state = await _commit_the_thought_with_retries(
            commit_outcome=commit_outcome,
            cycle_deadline_at=cycle_deadline_at,
            runtime_context=context,
            is_test_run=is_test_run,
            max_retries=max_retries,
            origin=origin,
            pre_turn_cognition=pre_turn_cognition,
            self=self,
            should_bypass_commit=should_bypass_commit,
            state=state,
            temp_state=temp_state,
        )
        return commit_outcome, state

    def _run_thinking_loop_imagination_feedback(self, _cycle_reward, commit_outcome, context, feedback, last_msg, mode, state):
        from .cognitive_engine import (
            _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
            Thought,
            _truncation_verdict,
            get_container,
            logger,
            normalize_live_mind_surface_control_receipt,
            uuid,
        )

        imagination_feedback = self._learn_imagination_workspace_outcome(
            context,
            outcome="assistant_response",
            reward=_cycle_reward,
        )
        bicameral_feedback = self._learn_bicameral_advisory_outcome(
            context,
            outcome="assistant_response",
            reward=_cycle_reward,
        )
        generation_controls = context.get("live_mind_generation_controls")
        if not isinstance(generation_controls, dict):
            generation_controls = {}
        surface_control_receipt = state.response_modifiers.get(
            "live_mind_surface_control_receipt"
        )
        if not isinstance(surface_control_receipt, dict):
            surface_control_receipt = {}
        if not surface_control_receipt:
            try:
                router = get_container().get("llm_router", default=None)
                if router is not None and hasattr(
                    router, "get_last_generation_metadata"
                ):
                    generation_metadata = router.get_last_generation_metadata()
                    if isinstance(generation_metadata, dict):
                        candidate = generation_metadata.get(
                            "surface_control_receipt"
                        )
                        if isinstance(candidate, dict):
                            surface_control_receipt = dict(candidate)
            except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
                logger.debug(
                    "Could not read full-phase surface-control receipt: %s",
                    exc,
                )
        context_controls_bound = bool(
            context.get("live_mind_controls_bound", False)
            and generation_controls
        )
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=context_controls_bound,
            generation_controls=generation_controls,
            source="cognitive_engine_full_phase_controls",
        )
        latent_final_quality = state.response_modifiers.get(
            "latent_cortex_final_output_quality"
        )
        latent_quality_reasons = (
            tuple(latent_final_quality.get("reasons") or ())
            if isinstance(latent_final_quality, dict)
            else ()
        )
        surface_reasons = tuple(
            dict.fromkeys(
                (
                    *tuple(
                        surface_control_receipt.get(
                            "surface_quality_gate_reasons"
                        )
                        or ()
                    ),
                    *latent_quality_reasons,
                )
            )
        )
        generation_stop_reason = str(
            surface_control_receipt.get("generation_stop_reason") or ""
        )
        semantic_completion_incomplete = bool(
            surface_control_receipt.get("semantic_completion_incomplete", False)
        )
        full_phase_text = str(last_msg.get("content") or "").strip()
        generation_failure_class = str(
            state.response_modifiers.get("generation_failure_class") or ""
        ).lower()
        reply_generation_incomplete = bool(
            semantic_completion_incomplete
            or set(surface_reasons)
            & {
                "truncated_tail",
                "final_answer_missing",
                "missing_final_answer",
                "incomplete_code_response",
            }
            or generation_stop_reason
            in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
            or any(
                reason in generation_failure_class
                for reason in (
                    "truncated_tail",
                    "final_answer_missing",
                    "missing_final_answer",
                    "incomplete_code_response",
                )
            )
            or _truncation_verdict(
                full_phase_text,
                generation_stop_reason=generation_stop_reason,
            )
        )
        latent_metadata = {
            key: state.response_modifiers.get(key)
            for key in (
                "latent_cortex_selected",
                "latent_cortex_selection_reason",
                "latent_cortex_depth_worthy",
                "latent_cortex_prompt_shape",
                "latent_cortex_attempted",
                "latent_cortex_succeeded",
                "latent_cortex_fallback_used",
                "latent_cortex_failure_reason",
                "latent_cortex_identity_bound",
                "latent_cortex_final_text_transformed",
                "latent_cortex_final_output_quality",
                "latent_cortex_raw_final_quality_hash_match",
                "latent_cortex_receipt",
                "latent_cortex_ingress",
                "latent_cortex_progress",
            )
            if key in state.response_modifiers
        }
        _degraded_count = len(
            [
                key
                for key in state.response_modifiers
                if str(key).endswith("_degraded")
                and state.response_modifiers.get(key)
            ]
        )
        thought = Thought(
            id=str(uuid.uuid4()),
            content=last_msg["content"],
            mode=mode,
            confidence=self._cycle_confidence(
                commit_outcome=commit_outcome,
                degraded_subsystems=_degraded_count,
            ),
            reasoning=[
                "Phase-based cognitive cycle completed.",
                f"State commit: {commit_outcome}.",
            ],
            metadata={
                "state_commit_outcome": commit_outcome,
                "spiking_active_inference": context.get("spiking_active_inference")
                if isinstance(context, dict)
                else None,
                "spiking_active_inference_feedback": feedback,
                "imagination_workspace_feedback": imagination_feedback,
                "bicameral_advisory": context.get("bicameral_advisory")
                if isinstance(context, dict)
                else None,
                "bicameral_advisory_feedback": bicameral_feedback,
                "cognitive_situation_frame": context.get("cognitive_situation_frame")
                if isinstance(context, dict)
                else None,
                "live_mind_controls_bound": context_controls_bound,
                "live_mind_generation_controls": dict(generation_controls),
                "live_mind_snapshot_ready": bool(
                    context.get("live_mind_snapshot_ready", False)
                ),
                "live_mind_required_subsystems_ok": bool(
                    context.get("live_mind_required_subsystems_ok", False)
                ),
                "live_mind_context_required": bool(
                    context.get("live_mind_context_required", False)
                ),
                "live_mind_surface_control_receipt": dict(
                    surface_control_receipt
                ),
                "live_mind_controls_worker_applied": bool(
                    surface_control_receipt.get("live_mind_controls_bound")
                    and surface_control_receipt.get("applied")
                ),
                "reply_generation_incomplete": reply_generation_incomplete,
                "reply_generation_stop_reason": generation_stop_reason,
                "reply_generation_failure_reasons": surface_reasons,
                "reply_original_chars": len(full_phase_text),
                **latent_metadata,
                "response_path": str(
                    state.response_modifiers.get("response_path")
                    or (
                        "cognitive_engine_latent_cortex"
                        if state.response_modifiers.get(
                            "latent_cortex_succeeded"
                        )
                        is True
                        else "cognitive_engine"
                    )
                ),
            },
        )
        return thought

    def _run_thinking_loop_record_pressure_read(self, context, objective):
        # Record the pressure, then READ it back. Until this, the graph had
        # two writers and no reader anywhere in the codebase: it accumulated
        # friction that could not influence any output, which makes the
        # signal unmeasurable rather than merely unused.
        from .cognitive_engine import (
            logger,
            record_degradation,
        )

        friction_key = objective[:20]
        self.autopoiesis.experience_friction(friction_key, 0.45)
        if self.autopoiesis.is_under_pressure(friction_key):
            logger.warning(
                "Objective '%s' keeps failing to resolve (friction %.2f); "
                "repeated failures on one kind of request are a defect signal, "
                "not noise",
                friction_key,
                self.autopoiesis.friction_for(friction_key),
            )
            record_degradation(
                "cognitive_engine",
                RuntimeError(f"objective repeatedly unresolved: {friction_key}"),
                severity="warning",
                action="recorded sustained objective friction",
                extra=self.autopoiesis.pressure_report(),
                # Friction is a durable learning/diagnostic observation, not
                # an exception from the service contract. Letting the generic
                # fail-closed policy enforce this warning turned a useful
                # signal into CRITICAL SERVICE FAILURE and killed the repair
                # pass that was supposed to resolve it.
                enforce_failure_policy=False,
            )
        self._learn_spiking_active_inference_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_imagination_workspace_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )
        self._learn_bicameral_advisory_outcome(
            context,
            outcome="no_assistant_response",
            reward=-0.65,
        )

    @staticmethod
    def _run_thinking_loop_generation_metadata(context, origin, salvaged, state):
        from .cognitive_engine import (
            _truncation_verdict,
            logger,
            normalize_live_mind_surface_control_receipt,
        )

        generation_metadata = dict(state.response_modifiers)
        generation_controls = context.get("live_mind_generation_controls")
        if not isinstance(generation_controls, dict):
            generation_controls = {}
        surface_control_receipt = generation_metadata.get(
            "live_mind_surface_control_receipt"
        )
        if not isinstance(surface_control_receipt, dict):
            surface_control_receipt = {}
        context_controls_bound = bool(
            context.get("live_mind_controls_bound", False)
            and generation_controls
        )
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=context_controls_bound,
            generation_controls=generation_controls,
            source="cognitive_engine_recoverable_draft_controls",
        )
        latent_final_quality = generation_metadata.get(
            "latent_cortex_final_output_quality"
        )
        latent_quality_reasons = (
            tuple(latent_final_quality.get("reasons") or ())
            if isinstance(latent_final_quality, dict)
            else ()
        )
        surface_reasons = tuple(
            dict.fromkeys(
                (
                    *tuple(
                        surface_control_receipt.get(
                            "surface_quality_gate_reasons"
                        )
                        or ()
                    ),
                    *latent_quality_reasons,
                )
            )
        )
        generation_stop_reason = str(
            surface_control_receipt.get("generation_stop_reason") or ""
        )
        generation_failure_class = str(
            generation_metadata.get("generation_failure_class") or ""
        ).lower()
        reply_generation_incomplete = bool(
            surface_control_receipt.get(
                "semantic_completion_incomplete", False
            )
            or set(surface_reasons)
            & {
                "truncated_tail",
                "final_answer_missing",
                "missing_final_answer",
                "incomplete_code_response",
            }
            or generation_stop_reason
            in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
            or any(
                reason in generation_failure_class
                for reason in (
                    "truncated_tail",
                    "final_answer_missing",
                    "missing_final_answer",
                    "incomplete_code_response",
                )
            )
            or _truncation_verdict(
                salvaged,
                generation_stop_reason=generation_stop_reason,
            )
        )
        recoverable_metadata = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in generation_metadata.items()
            if key
            in {
                "generation_failure_class",
                "latent_cortex_selected",
                "latent_cortex_selection_reason",
                "latent_cortex_depth_worthy",
                "latent_cortex_prompt_shape",
                "latent_cortex_attempted",
                "latent_cortex_succeeded",
                "latent_cortex_fallback_used",
                "latent_cortex_failure_reason",
                "latent_cortex_identity_bound",
                "latent_cortex_final_text_transformed",
                "latent_cortex_final_output_quality",
                "latent_cortex_raw_final_quality_hash_match",
                "latent_cortex_receipt",
                "latent_cortex_ingress",
                "latent_cortex_progress",
                "response_path",
            }
        }
        recoverable_metadata.update(
            {
                "recovered_from_suppression": True,
                "live_mind_controls_bound": context_controls_bound,
                "live_mind_generation_controls": dict(generation_controls),
                "live_mind_snapshot_ready": bool(
                    context.get("live_mind_snapshot_ready", False)
                ),
                "live_mind_required_subsystems_ok": bool(
                    context.get("live_mind_required_subsystems_ok", False)
                ),
                "live_mind_context_required": bool(
                    context.get("live_mind_context_required", False)
                ),
                "live_mind_surface_control_receipt": dict(
                    surface_control_receipt
                ),
                "live_mind_controls_worker_applied": bool(
                    surface_control_receipt.get("live_mind_controls_bound")
                    and surface_control_receipt.get("applied")
                ),
                "reply_generation_incomplete": reply_generation_incomplete,
                "reply_generation_stop_reason": generation_stop_reason,
                "reply_generation_failure_reasons": surface_reasons,
                "reply_original_chars": len(salvaged),
            }
        )
        logger.warning(
            "🩹 CognitiveEngine: no answer-quality response for origin=%s, but the "
            "turn still held a recoverable %d-char draft; serving it rather than "
            "reporting an empty cycle.",
            origin,
            len(salvaged),
        )
        return recoverable_metadata

    async def _run_thinking_loop(
        self,
        state: AuraState,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] = None,
        **kwargs,
    ) -> Thought:
        from .cognitive_engine import (
            _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
            ThinkingMode,
            Thought,
            _pass_instrumentation,
            _record_legacy_pass,
            _skip_provenance,
            finalize_foreground_turn_state,
            get_container,
            is_foreground_objective_origin,
            logger,
            record_degradation,
            record_response_path,
            recoverable_answer,
            sqlite3,
            uuid,
        )

        context, foreground_turn_objective = self._run_thinking_loop_part_1(context, objective, origin, state)

        is_background = bool(kwargs.get("is_background", False))
        explicit_timeout = kwargs.get("timeout_s", kwargs.get("timeout"))
        try:
            cycle_timeout = float(explicit_timeout) if explicit_timeout is not None else 0.0
        except (TypeError, ValueError):
            cycle_timeout = 0.0
        cycle_deadline_at, cycle_timeout = self._run_thinking_loop_part_2(context, cycle_timeout, is_background, objective, origin)

        # 4. Phase Execution Loop with Watchdog
        import copy

        backup_state = copy.deepcopy(state)
        temp_state = state
        success = False
        # Where this turn's working memory begins. An assistant message at or
        # before this mark belongs to an EARLIER turn: a duplicate or
        # suppressed user append leaves one sitting at the end, and extraction
        # checked only `role == "assistant"`, so the previous answer went out
        # again as this one's.
        turn_memory_mark = len(getattr(state.cognition, "working_memory", []) or [])
        # What this turn changed, so a rebase can restore only that. Restoring
        # everything from the per-turn snapshot overwrote whatever a concurrent
        # writer had done to goals, initiatives, focus and modifiers.
        pre_turn_cognition = {
            "active_goals": list(getattr(state.cognition, "active_goals", []) or []),
            "pending_initiatives": list(
                getattr(state.cognition, "pending_initiatives", []) or []
            ),
            "attention_focus": getattr(state.cognition, "attention_focus", None),
            "phenomenal_state": getattr(state.cognition, "phenomenal_state", None),
            "modifiers": dict(getattr(state.cognition, "modifiers", {}) or {}),
        }

        direct_quick_reply = kwargs.pop("precomputed_direct_reply", None)
        if direct_quick_reply is None:
            direct_quick_reply = await self._direct_desktop_quick_reply(
                objective,
                mode,
                origin,
                context,
                timeout_s=cycle_timeout,
            )
        success, temp_state = self._run_thinking_loop_part_3(direct_quick_reply, origin, state, success, temp_state)

        if not success:
            # Bound before the try, because the finally below reads it and the
            # timeout context manager can fail on entry.
            _provenance_tick = None
            _clock_keeper: Any = None
            try:
                from core.runtime.completion_admission import bind_completion_admission

                async with asyncio.timeout(cycle_timeout) as _cycle_clock, bind_completion_admission(
                    _cycle_clock, context,
                    enabled=bool(
                        self._is_user_facing_origin(origin) and not is_background
                        and str(origin).lower() not in {"proof", "eval", "evaluation", "benchmark", "test"}
                        and not any(context.get(key) for key in (
                            "proof_or_benchmark", "proof_run", "benchmark_run", "proof_answer_run",
                        ))
                    ),
                ):
                    _clock_keeper, _provenance_tick = self._run_thinking_loop__clock_keeper(_cycle_clock, context, is_background, objective, origin)
                    for phase in self._phases:
                        phase_name = phase.__class__.__name__
                        # [PASS INSTRUMENTATION] AURA_PASS_BISECT_LIMIT and
                        # AURA_PASS_TRACE existed only in pass_manager and the
                        # kernel tick — and chat drives THIS loop, not the
                        # kernel, by roughly 479 turns to 3. The documented way
                        # to find which phase ruined an answer therefore did
                        # nothing on the path that produces almost every
                        # answer. Same seam, same flag, same ordinals.
                        run_it, ordinal, reason = _pass_instrumentation().should_run(
                            f"legacy_pipeline/{phase_name}"
                        )
                        if not run_it:
                            _record_legacy_pass(
                                phase_name, ordinal, 0.0, skipped=True, reason=reason
                            )
                            # A skipped phase is part of the causal record. A
                            # graph that shows only what ran cannot answer why
                            # something did NOT happen, which is half of what
                            # "why did you do that" usually means.
                            _skip_provenance(phase_name, temp_state, reason)
                            continue
                        temp_state = await self._run_thinking_loop_started_at(context, kwargs, objective, ordinal, phase, phase_name, temp_state)

                    state = temp_state
                    record_response_path(
                        "full_phase_pipeline", model_generation=True
                    )
                    if self._is_user_facing_origin(origin):
                        state.transition_origin = origin
                        final_origin = getattr(state.cognition, "current_origin", "")
                        if is_foreground_objective_origin(final_origin) or not str(
                            final_origin or ""
                        ).strip():
                            state.cognition.current_origin = origin
                    success = True
            except TimeoutError:
                logger.error("🛑 [COGNITION] Watchdog: Cognitive cycle TIMEOUT (%.1fs).", cycle_timeout)
                record_response_path("reactive_recovery_timeout", model_generation=False)
                # Immediate Reactive Recovery
                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    "timeout",
                    context=context,
                    # The version THIS turn derived. A rollback that cannot
                    # match it is undoing somebody else's work.
                    authored_version=int(getattr(state, "version", 0) or 0),
                )
            except (sqlite3.Error, *_COGNITIVE_ENGINE_RECOVERABLE_ERRORS) as e:
                # This caught sqlite3.Error and OSError only, so the failures a
                # phase ACTUALLY produces — RuntimeError, AttributeError,
                # TypeError, ValueError from a malformed return or an
                # implementation defect — escaped the cognitive API entirely.
                # The caller got a raw exception where reactive recovery was
                # the designed behaviour, and the degradation record that names
                # the phase was never written.
                record_degradation(
                    "cognitive_engine",
                    e,
                    severity="critical",
                    action="downshifted or entered reactive recovery after phase failure",
                )
                logger.error("🚨 [COGNITION] Fatal error in phase logic: %s", e)
                # v14.1 HARDENING: Rollback & Downshift
                if mode == ThinkingMode.DEEP:
                    logger.warning(
                        "🔄 [COGNITION] Downshifting to REACTIVE mode due to Deep Failure..."
                    )
                    # WITH the context. The downshift used to call think()
                    # without it, so the retry lost the desktop-required
                    # flags, the live-mind evidence, the scoped request
                    # metadata and the recent exchanges — and then answered a
                    # question it could no longer see the terms of. A retry
                    # that drops the contract is a different request.
                    return await self.think(
                        objective,
                        context=dict(context or {}),
                        mode=ThinkingMode.FAST,
                        origin=origin,
                        **kwargs,
                    )

                record_response_path("reactive_recovery_crash", model_generation=False)
                return await self._reactive_recovery(
                    objective,
                    mode,
                    origin,
                    f"crash: {e}",
                    context=context,
                    authored_version=int(getattr(state, "version", 0) or 0),
                )
            finally:
                state = await self._run_thinking_loop_closed_rather_after(_clock_keeper, _provenance_tick, backup_state, state, success)

        # Capture the routed objective before closing a foreground turn. Response
        # extraction still needs it for action-imperative validation, but durable
        # state must not retain a completed chat turn as autonomous work.
        routed_obj = str(getattr(state.cognition, "current_objective", "") or "")
        is_action_imperative = (
            "[ACTION IMPERATIVE]" in objective or "[ACTION IMPERATIVE]" in routed_obj
        )
        # finalize_foreground_turn_state mutates the state about to be
        # committed, so it belongs here. The CLOSURE notification does not: it
        # tells external lifecycle state that the turn completed, and it used
        # to fire before persistence, so a bypassed or failed commit left the
        # rest of the runtime believing a turn had completed that durable state
        # has no record of. It moves below the commit loop.
        _is_foreground_turn = self._is_user_facing_origin(origin) and not is_background
        if _is_foreground_turn:
            finalize_foreground_turn_state(
                state,
                objective=foreground_turn_objective,
                origin=origin,
            )

        # ─── SUCCESS PATH (Unreachable before fix) ──────────────────────────
        # 5. Final State Commit
        # HF12: Handle concurrent version conflicts with a mini-retry loop
        is_test_run = self._is_test_run(origin)
        commit_outcome, state = await self._run_thinking_loop_should_bypass_commit(context, cycle_deadline_at, is_test_run, origin, pre_turn_cognition, state, temp_state)

        # The turn completed durably (or was legitimately isolated). Only now
        # may external lifecycle state be told it finished.
        if _is_foreground_turn and commit_outcome in {
            "committed",
            "bypassed_test_isolation",
        }:
            closure = get_container().get("executive_closure", default=None)
            if closure is not None and hasattr(closure, "complete_foreground_turn"):
                closure.complete_foreground_turn(foreground_turn_objective, origin)
        elif _is_foreground_turn:
            record_degradation(
                "cognitive_engine",
                RuntimeError(f"foreground_turn_uncommitted:{commit_outcome}"),
                severity="warning",
                action="withheld foreground closure because cognitive state did not commit",
                # Withholding closure is the whole response to this. It was
                # also escalated to fatal by the fail-closed policy, which
                # discarded the reply this function goes on to extract four
                # lines below — an answer thrown away over bookkeeping.
                enforce_failure_policy=False,
            )

        # 6. Extract Response
        last_msg = self._turn_response_message(
            state.cognition.working_memory, mark=turn_memory_mark
        )
        if last_msg:
            self.autopoiesis.experience_friction(objective[:20], 0.05)
            # Reward is no longer 1.0 for the mere existence of an
            # assistant-shaped message. Three learning systems were given the
            # maximum positive signal with no user feedback, no correctness
            # check, no tool postcondition and no persistence outcome — an
            # answer that failed to commit taught them it had gone perfectly.
            _cycle_reward = 1.0 if commit_outcome in {
                "committed",
                "bypassed_test_isolation",
            } else 0.5
            feedback = self._learn_spiking_active_inference_outcome(
                context,
                outcome="assistant_response",
                reward=_cycle_reward,
            )
            if direct_quick_reply is not None:
                thought = direct_quick_reply
                quick_metadata = dict(thought.metadata or {})
                imagination_feedback = quick_metadata.get(
                    "imagination_workspace_feedback"
                )
                if not isinstance(imagination_feedback, dict):
                    imagination_feedback = self._learn_imagination_workspace_outcome(
                        context,
                        outcome="assistant_response",
                        reward=_cycle_reward,
                    )
                bicameral_feedback = quick_metadata.get("bicameral_advisory_feedback")
                if not isinstance(bicameral_feedback, dict):
                    bicameral_feedback = self._learn_bicameral_advisory_outcome(
                        context,
                        outcome="assistant_response",
                        reward=_cycle_reward,
                    )
                thought.metadata = {
                    **quick_metadata,
                    "spiking_active_inference_feedback": feedback,
                    "imagination_workspace_feedback": imagination_feedback,
                    "bicameral_advisory_feedback": bicameral_feedback,
                }
            else:
                thought = self._run_thinking_loop_imagination_feedback(_cycle_reward, commit_outcome, context, feedback, last_msg, mode, state)
            self.thoughts.append(thought)
            return thought

        deferred = str(state.response_modifiers.get("background_suppression") or "")
        if deferred:
            # Held back for admission, not unresolved: no friction, no alarm.
            logger.info(
                "CognitiveEngine: background objective deferred for origin=%s (%s).",
                origin,
                deferred,
            )
            return self._empty_thought(mode, "background_deferred")
        self._run_thinking_loop_record_pressure_read(context, objective)

        # ── ACTION IMPERATIVE FALLBACK ──
        #
        # This used to emit [SOMATIC:key='.'] for ANY turn whose objective
        # contained "[ACTION IMPERATIVE]" — text a user can type and injected
        # content can carry — and called it a safe no-op. A keystroke is not a
        # no-op: it goes to whatever holds focus, which may be a terminal, an
        # editor, or a form. The legitimate somatic reflex
        # (orchestrator/mixins/incoming_logic.py) fires only on a message
        # carrying [EMBODIED CONTROL CONTRACT] and only when a real CLI prompt
        # pattern matched. The same condition governs it here: without the
        # contract, a turn that produced no response says so.
        if is_action_imperative:
            embodied_control = "[EMBODIED CONTROL CONTRACT]" in objective or (
                "[EMBODIED CONTROL CONTRACT]" in routed_obj
            )
            if embodied_control:
                logger.warning(
                    "⚠️ [COGNITION] Embodied control turn produced no response. "
                    "Falling back to the pager-advance key."
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content="[SOMATIC:key='.']",
                    mode=mode,
                    confidence=0.5,
                    reasoning=["Embodied control fallback (pager advance)."],
                    metadata={"embodied_control_contract": True},
                )
            record_degradation(
                "cognitive_engine",
                RuntimeError("action_imperative_without_embodied_contract"),
                severity="warning",
                action="refused a motor fallback for an action imperative with no embodied control contract",
            )
            logger.warning(
                "⚠️ [COGNITION] Action Imperative active but no response generated, "
                "and no embodied control contract authorises a keystroke."
            )
            return self._empty_thought(mode, "action_imperative_no_response")

        if is_background:
            logger.debug(
                "🛡️ CognitiveEngine: background cycle for origin=%s produced no response; returning quiet no-op.",
                origin,
            )
            return self._empty_thought(mode, "background_cycle_no_response")

        structured = self._structured_evaluation_thought(
            objective,
            state=state,
            mode=mode,
            origin=origin,
            fast_path=False,
            context=context,
        )
        if structured is not None:
            return structured

        # If the objective requires a strict answer format, do not return conversational evasive fallbacks.
        # Instead, attempt a direct, single-turn LLM generation as a high-fidelity recovery mechanism.
        # A literal "<answer>" ANYWHERE in the objective used to activate this
        # recovery — text a person can type, and text injected content can
        # carry — and the recovery sends the full objective to a cloud
        # provider. Routing to a third party is not something the prompt gets
        # to decide. The caller's answer_format kwarg is an explicit contract
        # and still counts; a substring in the user's words does not.
        is_strict_answer = "answer_format" in kwargs or bool(
            context.get("strict_answer_contract", False)
        )
        if "<answer>" in objective.lower() and not is_strict_answer:
            logger.info(
                "🛡️ [COGNITION] '<answer>' appears in the objective but no caller "
                "declared a strict-answer contract; not activating cloud recovery."
            )
        if is_strict_answer:
            logger.warning("⚠️ [COGNITION] Structured answer required but phase execution produced no response. Running last-resort direct recovery...")
            try:
                from core.brain.llm_health_router import get_llm_router
                from core.runtime.proof_policy import proof_model_tier
                router = get_llm_router()
                from core.brain.llm.an_envelope_the_decoder_enforces import (
                    AN_ANSWER,
                    the_request_for,
                )

                # The envelope is structural here, not requested.
                #
                # This asked — "Put your final answer strictly inside
                # <answer>...</answer> tags" — and then checked whether the
                # envelope had arrived, failing the turn when it had not. A
                # model asked for a delimiter produces one most of the time,
                # and most of the time is what becomes a retry and then a
                # person told the runtime could not get to an answer. The
                # assistant turn now opens with the marker, so the model is
                # inside the envelope and can only continue, and the decoder
                # stops at the closing one.
                system_prompt = "Solve the user's problem."
                recovery_tier = proof_model_tier() if is_test_run else "primary"
                # Last-resort recovery remains on the selected local lane.
                content = await router.think(
                    **the_request_for(
                        AN_ANSWER,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": objective},
                        ],
                    ),
                    origin=f"recovery_{origin}",
                    allow_cloud_fallback=False,
                    prefer_tier=recovery_tier,
                    protected_foreground_lane=recovery_tier == "primary",
                    proof_primary_lane_required=is_test_run and recovery_tier == "primary",
                    proof_evaluation_contract=is_test_run,
                    foreground_request=True,
                )
                # Nonempty was the whole postcondition: the recovery claimed
                # success at confidence 0.8 for any text at all, including
                # text with no <answer> envelope — the one thing the contract
                # promised. A strict answer that does not carry its envelope
                # did not satisfy the contract, and saying it did is what the
                # caller then parses and fails on.
                # Put back around what came out, because what came out is the
                # contents: the opening marker was the prompt's last token and
                # the closing one ended the generation, so neither is in the
                # text. Idempotent, so a model that wrote its own markers
                # anyway does not end up with two envelopes.
                content = AN_ANSWER.around(content)
                cleaned = str(content or "").strip()
                envelope_ok = AN_ANSWER.holds(cleaned)
                if cleaned and envelope_ok:
                    thought = Thought(
                        id=str(uuid.uuid4()),
                        content=content,
                        mode=mode,
                        confidence=0.8,
                        reasoning=["Last-resort direct structured recovery succeeded."],
                        metadata={"strict_answer_envelope_verified": True},
                    )
                    self.thoughts.append(thought)
                    return thought
                if cleaned:
                    record_degradation(
                        "cognitive_engine",
                        ValueError("strict answer recovery returned no <answer> envelope"),
                        severity="warning",
                        action="refused a strict-answer recovery that did not carry its envelope",
                    )
            except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
                record_degradation(
                    "cognitive_engine",
                    rec_err,
                    severity="degraded",
                    action="returned strict answer recovery failure after direct recovery failed",
                )
                logger.error("Failed last-resort structured recovery: %s", rec_err)
            return self._empty_thought(mode, "strict_answer_recovery_failed")

        # Last stop before this turn tells a person it has nothing. Ask the
        # ledger: did a gate suppress a draft that is still servable? Measured
        # live, the answer was sometimes yes — a complete 240-character reply
        # rejected for `truncated_tail` while the person got an apology.
        #
        # This does NOT overrule the gates. Text they marked unrecoverable
        # (prompt leaks, corrupted output, policy refusals) never comes back;
        # `best_recoverable_candidate` excludes it. What comes back is a draft
        # a heuristic merely disliked, and only when the alternative is
        # nothing at all.
        salvaged = recoverable_answer()
        if not salvaged:
            try:
                from core.conversation.surface_disposition import best_available_reply

                salvaged = best_available_reply(question=objective)
            except (ImportError, RuntimeError, TypeError, ValueError):
                salvaged = ""
        if salvaged:
            recoverable_metadata = self._run_thinking_loop_generation_metadata(context, origin, salvaged, state)
            return Thought(
                id=str(uuid.uuid4()),
                content=salvaged,
                mode=mode,
                confidence=0.4,
                reasoning=["Recovered a gate-suppressed draft; nothing else survived."],
                metadata=recoverable_metadata,
            )

        if bool(
            context.get("desktop_cognitive_engine_required", False)
            or context.get("cognitive_engine_required", False)
        ):
            return self._desktop_cognitive_failure_thought(
                mode,
                str(
                    state.response_modifiers.get("generation_failure_class")
                    or "user_cycle_no_response"
                ),
                generation_metadata=dict(state.response_modifiers),
            )

        logger.warning(
            "🛡️ CognitiveEngine: user-facing cycle for origin=%s produced no answer-quality response.",
            origin,
        )
        return self._empty_thought(mode, "user_cycle_no_response")

