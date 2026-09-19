"""What a turn decides before the prompt is assembled.

Lifted whole out of `inference_gate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import math
from typing import Any


class _SetsTheTurnUp:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    def _generate_with_metadata_sink_part_1(self, context, prompt):
        from .inference_gate import (
            bind_user_surface_prompt,
            logger,
            resolve_user_surface_prompt,
        )

        if "user_surface_grounding_evidence" not in context:
            try:
                from core.conversation.turn_evidence_custody import (
                    turn_grounding_evidence,
                )

                turn_grounding = turn_grounding_evidence()
                if turn_grounding:
                    # Snapshot process-local custody before routing can cross
                    # task, thread, or worker boundaries. The worker receives
                    # evidence, never authority to read ambient conversation.
                    context["user_surface_grounding_evidence"] = list(turn_grounding)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Turn grounding custody unavailable, carrying no evidence: %s", exc)
        self._clear_last_generation_metadata()
        initial_messages = context.get("messages")
        if not isinstance(initial_messages, list):
            initial_messages = None
        explicit_visible_user_prompt = str(
            context.get("user_surface_validation_prompt")
            or context.get("visible_user_message")
            or context.get("current_user_message")
            or ""
        ).strip()
        # The user's question is supplied, never inferred from the prompt.
        #
        # A chat turn hands this in — response generation passes
        # visible_user_message and user_surface_validation_prompt, and so does
        # every path that produces a visible reply. When nobody supplies one,
        # reading the last user-role message out of the envelope does not
        # recover it: it invents it, out of whatever text the caller happened
        # to send the model.
        #
        # LIVE 2026-08-19: deciding a move on a 2048 board, the deliberation's
        # own prompt became "the question". It carried a screen reading full of
        # numbers, so the reply was required to contain a total, and
        #
        #   "right — the board is mostly open on the right side, sliding right
        #    consolidates the smaller numbers and creates space"
        #
        # was rejected as arithmetic_answer_missing. Three retries, then no
        # text at all, and the pursuit reported she had named no move.
        #
        # The fallback survives where it is meaningful: an origin that really
        # is a person talking. Anywhere else an unsupplied question means
        # there is no user turn here to grade.
        # Read here rather than at its later assignment: this decision is
        # made before the binding, and the binding is the thing being guarded.
        derivable = self._origin_is_user_facing(
            str(context.get("origin", "") or "").lower()
        )
        initial_visible_user_prompt = explicit_visible_user_prompt or (
            self._visible_user_prompt_from_messages(initial_messages, prompt)
            if derivable
            else ""
        )
        surface_prompt = resolve_user_surface_prompt(
            context,
            fallback=initial_visible_user_prompt,
        )
        # The question THIS call is answering wins, always.
        #
        # LIVE DEFECT, 2026-08-10. Asked "look at my screen and tell me what
        # app is in front right now" (128 chars), the worker rejected every
        # draft with:
        #
        #   Rejected live user-surface draft reasons=arithmetic_answer_missing
        #       validation_chars=31 excerpt='50864799'
        #
        # 31 characters is "what's 7919 multiplied by 6421?" — the PREVIOUS
        # turn. A binding already present was treated as authoritative, so the
        # explicit visible message handed in for this call was ignored, and the
        # screen question was graded against the arithmetic question's expected
        # answer. It could not pass: there is no product in a reply about
        # windows. The turn died as arithmetic_answer_missing and the person
        # got a refusal about their screen.
        #
        # A binding survives to protect a turn from losing its own question
        # mid-flight. Carrying one INTO a different turn inverts that: it grades
        # every later answer against an older question, and the mismatch is
        # invisible because the reason names arithmetic while the user asked
        # about a window.
        stale_binding = bool(
            surface_prompt.bound
            and explicit_visible_user_prompt
            and str(surface_prompt.prompt or "").strip() != explicit_visible_user_prompt
        )
        # An internal generation has no user question to be graded against.
        #
        # Binding one anyway is where the harm starts: the prompt of a
        # deliberation becomes "the question", and every later check reads it
        # as something a person asked. A screen reading full of numbers then
        # looks like arithmetic, and a correct one-word move is rejected for
        # not containing a total.
        # Declared if the caller could say so, derived otherwise.
        #
        # The declaration travels through several client shapes and does not
        # survive all of them, so it is not the only evidence. A call that
        # supplied no user question AND whose origin is not a person talking
        # cannot be a reply to anybody — there is nothing it could be
        # answering. That is the same test used below to decide whether to
        # bind a question at all, and it is made from fields that reach here
        # on every path.
        internal_inference_call = bool(
            context.get("internal_inference", False)
            or context.get("_non_chat_inference", False)
            or (
                not explicit_visible_user_prompt
                and not self._origin_is_user_facing(
                    str(context.get("origin", "") or "").lower()
                )
            )
        )
        if internal_inference_call:
            context["internal_inference"] = True
        elif not initial_visible_user_prompt and not surface_prompt.bound:
            # Nothing to grade against, so nothing is bound. Binding an empty
            # prompt is what let the checks downstream fall back to reading
            # the model prompt again.
            pass
        elif not surface_prompt.bound or stale_binding:
            if stale_binding:
                logger.warning(
                    "🔗 Rebinding the user-surface validation prompt: the bound "
                    "one (%d chars) is not this turn's question (%d chars).",
                    len(str(surface_prompt.prompt or "")),
                    len(explicit_visible_user_prompt),
                )
            bind_user_surface_prompt(
                context,
                explicit_visible_user_prompt
                if stale_binding
                else (surface_prompt.prompt or initial_visible_user_prompt),
                source="inference_gate.visible_user_message",
                overwrite=True,
            )
            surface_prompt = resolve_user_surface_prompt(context)
        initial_visible_user_prompt = surface_prompt.prompt or initial_visible_user_prompt
        return initial_visible_user_prompt, internal_inference_call

    def _generate_with_metadata_sink_is_background(self, context, explicit_background, explicit_foreground, origin, purpose, requested_tier):
        is_background = bool(context.get("is_background", False))
        if explicit_foreground:
            is_background = False
        elif not is_background and not explicit_background:
            if origin:
                is_background = not self._origin_is_user_facing(origin)
            elif purpose in {"reply", "expression", "chat", "conversation", "user_response"}:
                is_background = False
            elif not explicit_background:
                # Origin-less requests are internal by default. User-facing turns
                # must carry an explicit origin such as api/user/voice.
                is_background = True
        deep_handoff = bool(context.get("deep_handoff", False))
        deep_reasoning_requested = bool(
            str(context.get("reasoning_mode") or "").strip().lower() == "deep"
            or str(context.get("serving_lane") or "").strip().lower() == "deep_reasoning"
            or deep_handoff
            or requested_tier == "secondary"
        )
        if deep_reasoning_requested:
            # Reasoning depth belongs to the turn contract, not to one model.
            # A specialist handoff may be inadmissible while the resident
            # cortex, RLC, verifiers, tools, and memory still execute the deep
            # systems lane. Preserve that mode across provider fallback.
            context["reasoning_mode"] = "deep"
        return deep_handoff, is_background

    @staticmethod
    def _generate_with_metadata_sink_part_3(context, deep_handoff, explicit_background, health_probe, is_background, live_benchmark_request, origin, proof_evaluation_contract, purpose):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _record_inference_degradation,
            env_str,
            proof_model_tier,
        )

        if deep_handoff and not explicit_background:
            # Explicit deep handoffs are foreground reasoning requests even if
            # the caller forgot to stamp a user-facing origin.
            is_background = False
        strict_primary_proof_lane = False
        try:
            proof_run_enabled = env_str(
                "AURA_PROOF_RUN",
                description="Mark a hermetic proof runtime",
                owner="core.runtime.state_ownership",
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        proof_evaluation_contract
                        or health_probe
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except _INFERENCE_RECOVERABLE_ERRORS as _proof_policy_exc:
            # Fail CLOSED for proof routing: an explicit caller requirement
            # survives a policy-probe failure, and the failure goes on the
            # record — a silently disabled proof lane could later be mistaken
            # for a valid proof-lane result.
            strict_primary_proof_lane = bool(
                context.get("proof_primary_lane_required", False)
            )
            _record_inference_degradation(
                _proof_policy_exc,
                action="kept explicit proof-lane requirement after proof policy probe failed",
                severity="error",
            )
        return is_background, strict_primary_proof_lane

    @staticmethod
    def _generate_with_metadata_sink_part_4(background_deferral, origin):
        from .inference_gate import (
            logger,
        )

        if background_deferral == "memory_pressure":
            logger.info(
                "⏸️ InferenceGate: Deferring background inference for origin=%s due to memory pressure.",
                origin,
            )
        elif background_deferral == "foreground_headroom_reserved":
            logger.info(
                "⏸️ InferenceGate: Foreground headroom reserved. Deferring background inference for origin=%s.",
                origin,
            )
        elif background_deferral == "cortex_startup_quiet":
            logger.info(
                "⏸️ InferenceGate: Cortex quiet window active. Deferring background inference for origin=%s.",
                origin,
            )
        elif background_deferral == "foreground_quiet_window":
            logger.info(
                "⏸️ InferenceGate: Foreground quiet window active. Deferring background inference for origin=%s.",
                origin,
            )
        elif background_deferral == "desktop_background_disabled":
            logger.info(
                "⏸️ InferenceGate: Desktop background local LLM disabled. Deferring background inference for origin=%s.",
                origin,
            )
        else:
            logger.info(
                "⏸️ InferenceGate: Foreground lane reserved. Deferring background inference for origin=%s.",
                origin,
            )

    async def _generate_with_metadata_sink_part_5(self, context, is_background, origin, protected_foreground_lane, requested_tier, strict_primary_proof_lane):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            _recover_the_cortex_before_answering,
            logger,
        )

        if protected_foreground_lane and not is_background:
            # Global resource side effects, and the promotion that reaches them
            # can come from PROMPT TEXT: looks_like_deep_mind_probe is a
            # heuristic over what the person typed, and a match promotes the
            # turn to the protected lane. Unbudgeted, a run of probe-shaped
            # messages sheds background workers and extends the quiet window
            # once per message. Explicit callers — a contract that says
            # protected_foreground_lane — are not rate-limited; a promotion
            # inferred from text is.
            heuristic_promotion = bool(
                context.get("deep_mind_probe", False)
                and not context.get("protected_foreground_lane", False)
            )
            if heuristic_promotion and not self._admit_heuristic_protected_shed():
                logger.info(
                    "🛡️ Text-inferred protected lane: shed budget spent, "
                    "serving on the current lane state."
                )
            else:
                self._extend_startup_quiet_window(180.0)
                if not self._primary_lane_ready():
                    await self._shed_background_workers_for_memory_pressure(
                        force=True,
                        reason="protected_foreground_shed",
                    )

        # ── Morphogenesis routing advice ──────────────────────────────────
        # If the morphogenetic metabolism reports very high system pressure,
        # downgrade non-protected foreground requests from the heavy 32B
        # cortex to the lighter brainstem to avoid OOM/stall under load.
        if not is_background and not protected_foreground_lane and requested_tier != "tertiary":
            try:
                from core.morphogenesis.hooks import get_morphogenesis_routing_advice

                _morph_advice = get_morphogenesis_routing_advice()
                # [RESILIENCE] Only downgrade for genuinely critical pressure,
                # not routine background morphogenetic oscillations.
                if (
                    _morph_advice.get("recommend_downgrade", False)
                    and _morph_advice.get("pressure", 0.0) > 0.85
                ):
                    logger.info(
                        "🧬 Morphogenesis recommends tier downgrade: %s (pressure=%.2f)",
                        _morph_advice.get("reason", "unknown"),
                        _morph_advice.get("pressure", 0.0),
                    )
                    requested_tier = "tertiary"
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                logger.debug("Morphogenesis routing advice unavailable: %s", exc)

        # ── Proactive cortex recovery (laptop sleep / MLX worker death) ───
        _seam_early_response, requested_tier = await _recover_the_cortex_before_answering(
            context=context,
            is_background=is_background,
            origin=origin,
            protected_foreground_lane=protected_foreground_lane,
            requested_tier=requested_tier,
            self=self,
            strict_primary_proof_lane=strict_primary_proof_lane,
        )
        return _seam_early_response, requested_tier

    @staticmethod
    def _generate_with_metadata_sink_source_code_prose(context, operator_evidence_contract, proof_evaluation_contract, strict_answer_contract, strict_proof_answer_request, strict_value_contract, web_interlocutor_contract):
        # Source code is not prose, and the conversational pipeline exists to
        # shape prose for a person: it repairs sentences, normalises
        # whitespace, and enforces a reply contract. Every one of those is
        # wrong for Python. Measured live 2026-07-28, the 2048 rules came back
        # through this path as
        #
        #   import randomdef move(case): board = case['board'] direction = ...
        #
        # — newlines gone, indentation collapsed, and therefore
        # "invalid syntax at line 1" on a generation the model had written
        # correctly. Code generation now declares itself isolated, like every
        # other non-conversational contract here.
        code_generation_contract = bool(context.get("code_generation_contract", False))
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or operator_evidence_contract
            or web_interlocutor_contract
            or code_generation_contract
        )
        # Sealed proof prompts (<answer> envelope) get a micro budget so a
        # one-word answer cannot ramble; a caller-pinned max_tokens always
        # wins. But the contract also reaches structured proof requests
        # (e.g. the repair loop asking for a full replacement file as
        # JSON) — for those, an unconditional 128 default truncated every
        # generation mid-JSON. Unpinned non-envelope requests now keep the
        # budget computed below instead of collapsing to 128.
        strict_max_token_cap: int | None = 128
        if strict_answer_contract:
            try:
                explicit_cap = context.get("max_tokens")
                if explicit_cap:
                    strict_max_token_cap = max(1, int(explicit_cap))
                elif strict_proof_answer_request:
                    strict_max_token_cap = 128
                else:
                    strict_max_token_cap = None
            except (TypeError, ValueError, OverflowError):
                strict_max_token_cap = 128
        return isolated_generation_contract, strict_max_token_cap

    @staticmethod
    def _generate_with_metadata_sink_part_7(caller_declared_completion_floor, context, explicit_max_tokens_cap, initial_visible_user_prompt, max_tokens, surface_completion_floor):
        from .inference_gate import (
            answer_surface_token_floor,
            logger,
        )

        try:
            surface_completion_floor = max(
                1,
                int(
                    context.get("user_surface_completion_floor")
                    or answer_surface_token_floor(initial_visible_user_prompt)
                ),
            )
        except (TypeError, ValueError, OverflowError):
            surface_completion_floor = answer_surface_token_floor(
                initial_visible_user_prompt
            )
        context["user_surface_completion_floor"] = surface_completion_floor
        # A declared ceiling wins, which is what the paragraph above says
        # and what this used to contradict: it raised explicit_max_tokens_cap
        # to the floor, so a caller asking for 384 was dispatched with 1000.
        # The floor is for a turn nobody sized; it is not a licence to
        # overrule a caller who did.
        room = (
            min(surface_completion_floor, explicit_max_tokens_cap)
            if explicit_max_tokens_cap is not None
            and not caller_declared_completion_floor
            else surface_completion_floor
        )
        if max_tokens < room:
            logger.info(
                "🧠 Foreground completion contract raised the decode budget %d→%d.",
                max_tokens,
                room,
            )
            max_tokens = room
        return max_tokens, surface_completion_floor

    @staticmethod
    def _generate_with_metadata_sink_resource_stakes_scale(benchmark_request, health_probe, isolated_generation_contract, max_tokens, strict_answer_contract):
        # ── Resource Stakes: scale token budget by computational survival state ──
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
            record_degradation,
        )

        try:
            from core.consciousness.resource_stakes import get_resource_stakes

            token_mult = get_resource_stakes().get_token_budget_multiplier()
            if (
                token_mult < 0.95
                and not strict_answer_contract
                and not health_probe
                and not isolated_generation_contract
                and not benchmark_request
            ):
                max_tokens = max(384, int(max_tokens * token_mult))
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "inference_gate",
                exc,
                severity="warning",
                action="kept default token budget multiplier",
            )
            logger.debug("Resource-stakes token multiplier unavailable: %s", exc)

        # ── Operational Resource Stakes: persistent viability constrains action ──
        # This newer ledger is stricter than the legacy multiplier above: it can
        # downgrade the large-model lane and hard-cap output when viability drops.
        stakes_token_ceiling: int | None = None
        return max_tokens, stakes_token_ceiling

    @staticmethod
    def _generate_with_metadata_sink_affective_circumplex_let(context):
        # ── Affective Circumplex: let somatic state modulate generation params ──
        # Only applies on user-facing, non-background requests. Background tasks
        # run at fixed params to avoid thermal feedback loops.
        from .inference_gate import (
            logger,
        )

        somatic_temperature: float | None = None
        morpho_kwargs: dict[str, Any] = {}
        caller_temperature = context.get("temperature", context.get("temp"))
        if caller_temperature is not None:
            try:
                _caller_temp = float(caller_temperature)
                # NaN slides through min/max to the 2.0 ceiling — reject
                # non-finite values instead of maxing out sampling entropy.
                somatic_temperature = (
                    max(0.0, min(2.0, _caller_temp))
                    if math.isfinite(_caller_temp)
                    else None
                )
            except (TypeError, ValueError) as exc:
                logger.debug("Caller temperature is not a number, leaving it unset: %s", exc)
                somatic_temperature = None
        for _gen_key in (
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "stop_sequences",
            "schema",
            "benchmark_request",
            "purpose",
            "cognitive_mode",
            # The shape the caller will parse, held by the decoder in the
            # worker (core/brain/llm/a_shape_the_decoder_enforces.py).
            "output_shape",
            "strict_answer_contract",
            "strict_value_contract",
            "proof_evaluation_contract",
            "operator_evidence_contract",
            "web_interlocutor_contract",
            "runtime_fact_status_contract",
            "grounded_runtime_status_contract",
            "clean_user_surface_contract",
            "user_surface_completion_floor",
            "user_surface_validation_prompt",
            "semantic_completion_contract",
            "user_surface_continuation_contract",
            "user_surface_continuation_partial",
            "user_surface_continuation_resume_handle",
            "user_surface_conversation_resume_handle",
            "user_surface_prompt_binding",
            "user_surface_grounding_evidence",
            "clean_user_surface_steering_alpha",
            "clean_user_surface_recurrent_loops",
            "live_mind_controls_bound",
            "live_mind_generation_controls",
            "live_mind_snapshot_ready",
            "live_mind_required_subsystems_ok",
            "disable_prompt_cache",
            "clear_prompt_cache",
            "health_probe",
        ):
            if _gen_key in context:
                morpho_kwargs[_gen_key] = context[_gen_key]
        return morpho_kwargs, somatic_temperature

    @staticmethod
    def _generate_with_metadata_sink_part_10(context, max_tokens, somatic_temperature):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
            record_degradation,
        )

        try:
            from core.affect.affective_circumplex import get_circumplex
            from core.verify import influence_channels
            from core.verify.lesion_registry import apply_channel

            circumplex_params = get_circumplex().get_llm_params()
            # Wrapped so a paired trial can run this exact code with the
            # affect contribution removed. This is the largest direct
            # actuation in the system — the circumplex moves temperature
            # across 0.500..0.858 and the token budget across 472..768 —
            # and it was the only live actuator with no lesion, so the one
            # faculty with a visibly large effect was the one the influence
            # apparatus could not ask about. Neutral is "no affective
            # modulation": the caller's own budget and the default
            # temperature, which is what this block would produce if the
            # circumplex were flat.
            if not context.get("max_tokens"):
                max_tokens = max(
                    384,
                    min(
                        max_tokens,
                        int(
                            apply_channel(
                                influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
                                circumplex_params["max_tokens"],
                                neutral=max_tokens,
                            )
                        ),
                    ),
                )
            somatic_temperature = apply_channel(
                influence_channels.AFFECT_CIRCUMPLEX_SAMPLING,
                circumplex_params["temperature"],
                neutral=None,
            )
            logger.debug(
                "💓 Circumplex: V=%.2f A=%.2f → temp=%s tokens=%d",
                circumplex_params["valence"],
                circumplex_params["arousal"],
                # %s, not %.2f: under a paired trial this channel is
                # lesioned to None and a float format would raise inside
                # the logging call.
                "lesioned"
                if somatic_temperature is None
                else f"{somatic_temperature:.2f}",
                max_tokens,
            )
        except _INFERENCE_RECOVERABLE_ERRORS as _ce:
            record_degradation(
                "inference_gate",
                _ce,
                severity="warning",
                action="kept default sampling parameters without affective circumplex",
            )
            logger.debug("Circumplex unavailable: %s", _ce)

        # ── PNEUMA precision sampler: blend with circumplex temperature ──
        try:
            from core.consciousness.precision_sampler import get_active_inference_sampler

            _ais_params = get_active_inference_sampler().get_sampling_params()
            ais_temp = _ais_params.get("temperature")
            if ais_temp is not None:
                # Blend: 50% circumplex + 50% PNEUMA precision
                base = somatic_temperature if somatic_temperature is not None else 0.72
                somatic_temperature = round(0.5 * base + 0.5 * ais_temp, 3)
                logger.debug("🎯 PNEUMA precision temp blend → %.3f", somatic_temperature)
        except _INFERENCE_RECOVERABLE_ERRORS as _ais_e:
            record_degradation(
                "inference_gate",
                _ais_e,
                severity="warning",
                action="kept existing sampling temperature without active-inference blend",
            )
            logger.debug("ActiveInferenceSampler unavailable: %s", _ais_e)
        return max_tokens, somatic_temperature

    def _generate_with_metadata_sink_part_11(self, _homeostasis, max_tokens, somatic_temperature):
        from .inference_gate import (
            logger,
        )

        if _homeostasis and hasattr(_homeostasis, "get_inference_modifiers"):
            _h_mods = _homeostasis.get_inference_modifiers()
            if somatic_temperature is not None:
                somatic_temperature = round(
                    somatic_temperature
                    + self._modulator_delta(
                        _h_mods["temperature_mod"],
                        source="homeostasis.temperature_mod",
                        limit=0.5,
                    ),
                    3,
                )
                somatic_temperature = max(0.1, min(1.5, somatic_temperature))
            max_tokens = max(
                384,
                int(
                    max_tokens
                    * self._modulator_factor(
                        _h_mods["token_multiplier"],
                        source="homeostasis.token_multiplier",
                        low=0.5,
                        high=2.0,
                    )
                ),
            )
            logger.debug(
                "🫀 Homeostasis: temp_mod=%+.3f token_mult=%.2f caution=%.2f",
                _h_mods["temperature_mod"],
                _h_mods["token_multiplier"],
                _h_mods["caution_level"],
            )
        return max_tokens, somatic_temperature

    @staticmethod
    def _generate_with_metadata_sink_part_12(_plasticity, _substrate, morpho_kwargs, somatic_temperature):
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
            record_degradation,
        )

        if _substrate is not None and hasattr(_substrate, "x"):
            import numpy as _np_plast
            _sub_state = _np_plast.asarray(_substrate.x, dtype=_np_plast.float32)
            _plast_mod = _plasticity.compute_modulation(_sub_state)
            if _plast_mod:
                _p_temp_d = _plast_mod.get("temperature_delta", 0.0)
                _p_topp_d = _plast_mod.get("top_p_delta", 0.0)
                _p_rep_d = _plast_mod.get("repetition_penalty_delta", 0.0)
                if somatic_temperature is not None:
                    somatic_temperature = max(0.1, min(1.5, somatic_temperature + _p_temp_d))
                else:
                    somatic_temperature = max(0.1, min(1.5, 0.72 + _p_temp_d))
                if "top_p" in morpho_kwargs:
                    morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _p_topp_d))
                if "repetition_penalty" in morpho_kwargs:
                    morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _p_rep_d))
                logger.debug(
                    "🧬 SynapticPlasticity: temp_d=%.3f topp_d=%.3f rep_d=%.3f",
                    _p_temp_d, _p_topp_d, _p_rep_d,
                )
            # Pre-inference capture for post-inference learning
            _hedonic = 0.0
            try:
                from core.consciousness.hedonic_gradient import get_hedonic_gradient
                _hedonic = get_hedonic_gradient().score
            except _INFERENCE_RECOVERABLE_ERRORS as _hedonic_exc:
                record_degradation(
                    "inference_gate",
                    _hedonic_exc,
                    severity="warning",
                    action="continued synaptic plasticity capture without hedonic score",
                )
                logger.debug(
                    "SynapticPlasticity hedonic capture unavailable: %s",
                    _hedonic_exc,
                )
            _plasticity.pre_inference_capture(_sub_state, _hedonic)
        return somatic_temperature

    def _generate_with_metadata_sink_part_13(self, _sq, morpho_kwargs, somatic_temperature):
        from .inference_gate import (
            logger,
        )

        if _sq is not None:
            _sq_pert = _sq.compute_perturbation()
            if _sq_pert:
                _sq_temp = self._modulator_delta(
                    _sq_pert.get("temperature_perturbation", 0.0),
                    source="somatic_qualia.temperature",
                    limit=0.5,
                )
                _sq_rep = self._modulator_delta(
                    _sq_pert.get("repetition_penalty_perturbation", 0.0),
                    source="somatic_qualia.repetition_penalty",
                    limit=0.5,
                )
                _sq_topp = self._modulator_delta(
                    _sq_pert.get("top_p_perturbation", 0.0),
                    source="somatic_qualia.top_p",
                    limit=0.5,
                )
                _sq_freq = self._modulator_delta(
                    _sq_pert.get("frequency_penalty_perturbation", 0.0),
                    source="somatic_qualia.frequency_penalty",
                    limit=0.5,
                )
                if somatic_temperature is not None:
                    somatic_temperature = max(0.1, min(1.5, somatic_temperature + _sq_temp))
                if "repetition_penalty" in morpho_kwargs:
                    morpho_kwargs["repetition_penalty"] = max(0.9, min(1.4, morpho_kwargs["repetition_penalty"] + _sq_rep))
                if "top_p" in morpho_kwargs:
                    morpho_kwargs["top_p"] = max(0.3, min(0.98, morpho_kwargs["top_p"] + _sq_topp))
                if _sq_freq:
                    morpho_kwargs["frequency_penalty"] = max(0.0, min(0.5, morpho_kwargs.get("frequency_penalty", 0.0) + _sq_freq))
                logger.debug(
                    "🫀 SomaticQualia: temp=%.4f rep=%.4f topp=%.4f freq=%.4f",
                    _sq_temp, _sq_rep, _sq_topp, _sq_freq,
                )
        return somatic_temperature

    def _generate_with_metadata_sink_block_above_skipped(self, context, deep_probe_request, health_probe, is_background, isolated_generation_contract, max_tokens, origin, requested_tier):
        # The block above is skipped whenever the caller named its own budget —
        # and the desktop chat route always does, so no live desktop turn has
        # ever had a starvation floor. An explicit cap is an upper bound the
        # caller is entitled to; it is not permission to modulate the answer
        # down to a length that cannot finish a sentence.
        #
        # LIVE DEFECT, 2026-07-26: the route asked for 1,536 tokens for
        # "…show the reasoning, then give the exact fraction". Memory-pressure
        # capping took it to 384 and affective/resource scaling to 239, and the
        # answer arrived correct and cut mid-sentence:
        #
        #   "Total marbles: 3 red + 4 blue + 5 green = 12. For both to be the
        #    same colour, we need to consider each case separately: Both Red"
        #
        # A truncated answer is not a cheaper answer. The person re-asks and
        # the whole turn is paid for twice.
        from .inference_gate import (
            logger,
        )

        if (
            not is_background
            and self._origin_is_user_facing(origin)
            and requested_tier in {"primary", "secondary"}
            and not deep_probe_request
            and not isolated_generation_contract
            and not health_probe
            and not bool(context.get("resource_stakes_blocked", False))
        ):
            try:
                requested_budget = int(context.get("max_tokens") or 0)
            except (TypeError, ValueError) as exc:
                logger.debug("Requested budget is not an integer, reading it as none: %s", exc)
                requested_budget = 0
            if requested_budget > 0:
                # A flat floor rescues a conversational reply and still starves
                # a derivation: measured live 2026-07-27, the caller asked 896
                # for "when does the second train catch the first, and how far
                # from the station?", pressure scaling cut it to 459, and the
                # floor lifted it to 512 — enough to reach step 5 of 5 and stop
                # at "- The". The caller now says whether the turn needs room,
                # and the floor answers that instead of a constant.
                needs_room = bool(context.get("reply_needs_room", False))
                starvation_floor = min(
                    requested_budget,
                    self._configured_token_bound(
                        "AURA_FOREGROUND_CHAT_DERIVATION_FLOOR_TOKENS"
                        if needs_room
                        else "AURA_FOREGROUND_CHAT_STARVATION_FLOOR_TOKENS",
                        1024 if needs_room else 512,
                        minimum=256,
                    ),
                )
                if max_tokens < starvation_floor:
                    logger.info(
                        "🧠 Foreground starvation floor raised budget %d→%d "
                        "(caller asked %d, origin=%s).",
                        max_tokens,
                        starvation_floor,
                        requested_budget,
                        origin or "unknown",
                    )
                    max_tokens = starvation_floor
        return max_tokens

    @staticmethod
    def _generate_with_metadata_sink_part_15(context, deep_probe_request, explicit_max_tokens_cap, is_background, max_tokens, operator_evidence_contract, protected_compact_capability_contract, strict_answer_contract, strict_max_token_cap):
        from .inference_gate import (
            _FLAG_DEEP_PROBE_MAX_TOKENS,
            _record_inference_degradation,
        )

        if explicit_max_tokens_cap is not None:
            max_tokens = min(max_tokens, explicit_max_tokens_cap)
            if (
                protected_compact_capability_contract
                and not bool(context.get("resource_stakes_blocked", False))
            ):
                max_tokens = max(max_tokens, min(384, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        # The live primary conversation lane USES the prompt cache. It used to
        # force-set disable_prompt_cache=True here, on the premise that reuse
        # was "approximate" and the cause of clipped or stale Cortex drafts.
        #
        # Reuse is not approximate. A cached entry is KV for a byte-identical
        # token prefix; measured end to end on the same stack, three cached
        # turns produced continuations byte-identical to an uncached control
        # while reuse climbed 0 -> 34/53 -> 57/74 tokens. The clipped and stale
        # drafts had three other causes, all since fixed: the trie stored one
        # mutable cache object under every growing prefix, so old keys aliased
        # later KV; the trim probe checked the wrong module and answered False
        # forever, so a diverging prefix could never be trimmed to fit; and
        # nothing partitioned lanes, so internal generations and the
        # conversation shared entries. Reuse is now scoped to `user_surface`,
        # inserted once after generation, and trimmed when prefixes diverge.
        #
        # Leaving it force-disabled is what the endurance wall is made of: this
        # lane's prompt IS the whole conversation, so re-prefilling from token
        # zero makes time-to-first-token climb until it crosses the turn budget
        # and the conversation stops answering. Callers that genuinely need an
        # exact cold prompt (strict/proof/operator contracts, health probes)
        # still set the flag themselves and are bypassed in the worker.

        if deep_probe_request and not is_background:
            try:
                probe_token_cap = int(_FLAG_DEEP_PROBE_MAX_TOKENS.value())
            except (TypeError, ValueError) as _probe_cap_exc:
                _record_inference_degradation(
                    _probe_cap_exc,
                    action="used default deep-probe token cap after malformed environment value",
                )
                probe_token_cap = 384
            max_tokens = min(max_tokens, max(128, probe_token_cap))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False

        if strict_answer_contract and strict_max_token_cap is not None:
            max_tokens = max(1, min(max_tokens, strict_max_token_cap))
            context["max_tokens"] = max_tokens

        if operator_evidence_contract:
            try:
                requested_operator_cap = int(context.get("max_tokens") or 220)
            except (TypeError, ValueError):
                requested_operator_cap = 220
            max_tokens = max(1, min(max_tokens, requested_operator_cap, 220))
            context["max_tokens"] = max_tokens
            context["allow_tools"] = False
            context["disable_prompt_cache"] = True
            context["clear_prompt_cache"] = True
        return max_tokens

    @staticmethod
    def _generate_with_metadata_sink_part_16(benchmark_request, context, health_probe, initial_visible_user_prompt, max_tokens, morpho_kwargs):
        if health_probe:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 32
            max_tokens = max(1, min(max_tokens, requested_cap_int, 64))
            context["max_tokens"] = max_tokens
            context.setdefault("clean_user_surface_contract", True)
            context.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            context.setdefault("clean_user_surface_recurrent_loops", 1)
            context.setdefault("clean_user_surface_steering_alpha", 0.25)
            morpho_kwargs.setdefault("clean_user_surface_contract", True)
            morpho_kwargs.setdefault("user_surface_validation_prompt", initial_visible_user_prompt)
            morpho_kwargs.setdefault("clean_user_surface_recurrent_loops", 1)
            morpho_kwargs.setdefault("clean_user_surface_steering_alpha", 0.25)

        if benchmark_request:
            requested_cap = context.get("max_tokens", max_tokens)
            try:
                requested_cap_int = max(1, int(requested_cap))
            except (TypeError, ValueError):
                requested_cap_int = 96
            max_tokens = max(1, min(max_tokens, requested_cap_int))
            context["max_tokens"] = max_tokens
        return max_tokens

    @staticmethod
    def _generate_with_metadata_sink_turn_executes_something(context, explicit_max_tokens_cap, max_tokens, morpho_kwargs, output_contract, output_contract_is_user_facing, output_contract_payload, stakes_token_ceiling):
        # On a turn that EXECUTES something, a shape phrase describes the
        # ARTIFACT, not her reply.
        #
        # "Open the Notes app and write a new note with three sentences about
        # humpback whales" parses to sentence_count=3, and that was applied to
        # the chat reply: it forced the compact foreground context and clamped
        # the budget to max_tokens=288 — far too small to emit a multi-step
        # desktop plan. She produced conversational filler instead, nothing
        # executed, and the gate then vetoed the filler for not matching the
        # three-sentence shape it had itself imposed.
        #
        # Measured live twice, and confirmed by removing the phrase: the same
        # request without "three sentences" planned and executed, and the note
        # is on disk. Every demo instruction carries this kind of clause ("3
        # articles", "a coherent summary", "a short note"), so the artifact
        # spec was systematically starving the plan that would have produced it.
        #
        # The executor already owns artifact shape through `document_body`;
        # here it must not also become a ceiling on the report she gives back.
        from .inference_gate import (
            logger,
        )

        if bool(context.get("desktop_execution_contract", False)):
            # Carry the flag to the client so the unified-memory clamp keeps a
            # floor under the PLAN. Without it, pressure (a screen recorder is
            # enough) shrinks the budget below what the steps need and the task
            # cannot be attempted at all.
            morpho_kwargs["desktop_execution_contract"] = True
            # And give the plan room to exist. The origin's conversational
            # default capped this turn at 288 tokens, which cannot hold a
            # multi-step JSON plan, so the model emitted prose, the draft was
            # judged truncated, and nothing executed. Measured live on a
            # DELIBERATE desktop turn that had already been routed to
            # desktop_task. Success up to now depended on which planner ran:
            # the deterministic heuristic needs no tokens, the model one does.
            _plan_floor = 1024
            if int(max_tokens or 0) < _plan_floor:
                logger.info(
                    "🧾 [CONTRACT] Desktop execution turn: raising the reply "
                    "budget %s → %d so the plan can be expressed.",
                    max_tokens,
                    _plan_floor,
                )
                max_tokens = _plan_floor
                context["max_tokens"] = max_tokens
                # NOT morpho_kwargs. Every _generate_with_client call site
                # passes max_tokens= explicitly AND splats **morpho_kwargs, so
                # putting it in both raised
                #   TypeError: _generate_with_client() got multiple values for
                #   keyword argument 'max_tokens'
                # which failed the inference_gate closed and surfaced as
                # user_cycle_no_response — the engine returning nothing at all
                # on every desktop turn, in two seconds, while ordinary
                # conversation through the same engine kept working.
            if output_contract_is_user_facing:
                output_contract_is_user_facing = False
                logger.info(
                    "🧾 [CONTRACT] Output-shape request treated as the ARTIFACT's "
                    "shape on a desktop-execution turn; the reply keeps its full "
                    "budget (would have capped at %s tokens).",
                    getattr(output_contract, "hard_token_ceiling", None),
                )
        if (
            output_contract_is_user_facing
            and output_contract_payload is not None
            and output_contract.hard_token_ceiling is not None
        ):
            planned_tokens = max_tokens
            max_tokens = max(
                1,
                min(max_tokens, int(output_contract.hard_token_ceiling)),
            )
            context["requested_output_contract"] = dict(output_contract_payload)
            context["semantic_output_token_cap"] = output_contract.semantic_token_cap
            context["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            context["max_tokens"] = max_tokens
            morpho_kwargs["requested_output_contract"] = dict(output_contract_payload)
            morpho_kwargs["semantic_output_token_cap"] = output_contract.semantic_token_cap
            morpho_kwargs["hard_output_token_ceiling"] = output_contract.hard_token_ceiling
            if max_tokens < planned_tokens:
                logger.info(
                    "🧠 Explicit output contract capped generation %d→%d "
                    "(kind=%s semantic=%s hard=%s).",
                    planned_tokens,
                    max_tokens,
                    output_contract.kind,
                    output_contract.semantic_token_cap,
                    output_contract.hard_token_ceiling,
                )

        # No policy floor may expand a caller-admitted ceiling, and none may
        # expand a viability ceiling either. Keep both as the final
        # token-budget transformations before prompt construction and every
        # local provider call below.
        if stakes_token_ceiling is not None and max_tokens > stakes_token_ceiling:
            logger.info(
                "🪫 Resource-stakes ceiling re-applied after later modifiers: %d→%d.",
                max_tokens,
                stakes_token_ceiling,
            )
            max_tokens = max(1, min(max_tokens, stakes_token_ceiling))
            context["max_tokens"] = max_tokens
        if explicit_max_tokens_cap is not None:
            max_tokens = max(1, min(max_tokens, explicit_max_tokens_cap))
            context["max_tokens"] = max_tokens

        # Build the prompt only after routing intent is known so we can choose
        # a compact user-facing path instead of always constructing the richest stack.
        brief = context.get("brief", "")
        if hasattr(brief, "to_briefing_text"):
            brief = brief.to_briefing_text()
        elif not isinstance(brief, str):
            brief = str(brief)
        return brief, max_tokens

