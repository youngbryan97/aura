"""The steps a response generation phase is assembled from.

Lifted whole out of `response_generation`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)



class _RunsTheGenerationSteps:
    """Lifted whole out of ResponseGenerationPhase; see response_generation.py."""

    @staticmethod
    def _execute_substrate_voice_compile(
        objective: str | None,
        origin: str,
        state: Any,
    ) -> tuple[Any, Any]:
        # ── SUBSTRATE VOICE: Compile speech profile BEFORE prompt assembly ──
        # The substrate reads all internal systems and decides HOW Aura will speak.
        # This must happen before ContextAssembler builds the prompt so the
        # hard constraint block is available for injection.
        from .response_generation import (
            _record_response_generation_degradation,
            logger,
        )

        _sve = None
        _speech_profile = None
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine

            _sve = get_substrate_voice_engine()
            _speech_profile = _sve.compile_profile(
                state=state,
                user_message=str(objective)[:500],
                origin=origin,
            )
            logger.debug(
                "🗣️ [SubstrateVoice] Profile: budget=%d, tone=%s, multi=%s, fu=%.2f",
                _speech_profile.word_budget,
                _speech_profile.tone_override or "default",
                _speech_profile.multi_message,
                _speech_profile.followup_probability,
            )
        except (ImportError, AttributeError, RuntimeError) as _sve_exc:
            _record_response_generation_degradation(
                _sve_exc,
                action="continued response generation without substrate voice shaping profile",
                severity="error",
            )
            logger.error("SubstrateVoiceEngine compile failed: %s", _sve_exc, exc_info=True)
        return _speech_profile, _sve

    def _execute_derived_merely_read(
        self,
        messages: Any,
        objective: str | None,
        runtime_context: Any,
        state: Any,
    ) -> None:
        # Derived, not merely read. The flag has to survive chat route ->
        # CognitiveEngine -> phase context to arrive here, and when it
        # does not, this block never renders: she is then asked to do a
        # desktop task with no instruction that she can, and answers
        # from her identity text instead. Measured live, verbatim:
        #   "I can't interact with your device or open apps. But here's
        #    a note for you: 'Orcas, also known as killer whales...'"
        # — a false capability denial plus the note's content typed into
        # chat, while Notes stayed untouched. The objective itself is
        # sufficient evidence, and the same detector already governs the
        # route's own decision.
        from .response_generation import (
            desktop_task_action_sentence,
        )

        _desktop_objective = bool(runtime_context.get("desktop_execution_contract"))
        if not _desktop_objective:
            # The ROUTER already decided this. It writes its matched
            # skills onto the state — "Routing: multi-step skill-backed
            # task detected → TASK via ['desktop_task']" — and nothing
            # downstream ever read them, so the one component that knew
            # could not tell the one component that had to act. Reading
            # the decision beats re-deriving it: there is a single
            # source of truth and it cannot disagree with itself.
            _matched = state.response_modifiers.get("matched_skills")
            if isinstance(_matched, (list, tuple, set)):
                _desktop_objective = any(
                    "desktop" in str(skill).lower() or "computer_use" in str(skill).lower()
                    for skill in _matched
                )
        if not _desktop_objective:
            # Last resort: the objective itself. Kept so a lane that
            # never reached the router still plans rather than denying.
            try:
                from core.runtime.desktop_objective_intent import (
                    looks_like_desktop_objective,
                )

                _desktop_objective = bool(looks_like_desktop_objective(objective))
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.debug(
                    "desktop-objective detection is unavailable, so this is not treated as one (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                _desktop_objective = False
        if _desktop_objective:
            desktop_block = (
                "## LIVE DESKTOP EXECUTION PLANNING CONTRACT\n"
                "The user's request is a live desktop/computer objective. Produce a compact "
                "execution draft that can drive the governed desktop_task lane. Prefer a JSON "
                "object with optional `document_body` and a bounded `steps` array. Allowed step "
                f"actions are {desktop_task_action_sentence()}. "
                "Use `{{document_body}}` inside a step target when a long composed body should be "
                "typed, pasted, written, or exported. A later step may reference verified prior "
                "output with `{{steps.1.result.path}}` or `{{last.result.path}}`. Each step needs "
                "a reason, an expected observable effect, and may set `critical` false only when "
                "the objective can still succeed after that step fails. Do not claim completion "
                "inside this draft; completion is only true after downstream desktop_task "
                "receipts verify effects. Keep the plan general to the named apps/surfaces and "
                "requested artifacts; do not use a hardcoded demo shortcut. Do not answer like "
                "a hosted chatbot: this runtime has governed local desktop control, so never say "
                "you cannot interact with apps, open Notes/Docs/Chrome, write text, or control "
                "the desktop when the requested action is inside the desktop_task contract."
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{desktop_block}"
            else:
                messages.insert(0, {"role": "system", "content": desktop_block})
        self._inject_live_runtime_grounding(messages, runtime_context)

    def _execute_causal_world_model(
        self,
        kwargs: Any,
        messages: Any,
        proof_answer_run: bool,
        state: Any,
    ) -> tuple[Any, Any]:
        # Causal World Model Context Injection
        from .response_generation import (
            CognitiveMode,
            logger,
        )

        causal_model = None if proof_answer_run else self.container.get("causal_world_model", default=None)
        if causal_model:
            causal_context = causal_model.get_prompt_context()
            if causal_context:
                messages.insert(1, {"role": "system", "content": causal_context})
                logger.debug(
                    "🧶 ResponseGeneration: Causal world cascades injected into prompt."
                )

        # ISSUE-80: Context Fix (Identity Reinforcement)
        if state.cognition.current_mode == CognitiveMode.DELIBERATE and not proof_answer_run:
            # Ensure the system prompt or first message reinforces identity if buried
            if len(messages) > 10:
                logger.debug(
                    "🛡️ ResponseGeneration: Reinforcing identity anchor for long context."
                )
                identity_reminder = {
                    "role": "system",
                    "content": (
                        "REMEMBER: You are Aura. Speak from live grounded context, memory, "
                        "and evidence. Do not fall back to generic AI-assistant disclaimers."
                    ),
                }
                messages.insert(1, identity_reminder)

        # Skill result narration hint (GodModeToolPhase may have fired a skill this tick)
        last_skill = None if proof_answer_run else state.response_modifiers.get("last_skill_run")
        if last_skill:
            ok = state.response_modifiers.get("last_skill_ok", True)
            status_hint = "completed successfully" if ok else "encountered an issue"
            skill_hint = {
                "role": "system",
                "content": (
                    f"[SKILL EXECUTION] The skill '{last_skill}' just {status_hint}. "
                    f"Its result is in your context as [SKILL RESULT: {last_skill}]. "
                    f"Narrate it naturally — as yourself, not as a tool output log."
                ),
            }
            messages.insert(1, skill_hint)

        # 3. Invoke LLM Router with messages and watchdog
        router = self.container.get("llm_router")

        # Derive context-dependent parameters from state
        runtime_context = kwargs.get("context")
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        return router, runtime_context

    def _execute_turn_completed_capabilities(
        self,
        contract: Any,
        incoming_continuation_contract: bool,
        is_background: bool,
        is_test_run: bool,
        runtime_context: Any,
        state: Any,
    ) -> tuple[Any, Any]:
        from .response_generation import (
            completed_capabilities,
            project_user_surface_resume_capability,
        )

        turn_completed_capabilities = completed_capabilities(
            runtime_context.get("completed_capability_evidence")
        )
        state_tool_hit = self._successful_required_search_payload(state, contract)
        if state_tool_hit is not None:
            turn_completed_capabilities = frozenset(
                (*turn_completed_capabilities, state_tool_hit[0])
            )
        if turn_completed_capabilities:
            state.response_modifiers["completed_capability_turn_owner"] = {
                "capabilities": sorted(turn_completed_capabilities),
                "source": (
                    "runtime_stamped_turn_receipt_and_state"
                    if state_tool_hit is not None
                    and completed_capabilities(
                        runtime_context.get("completed_capability_evidence")
                    )
                    else "state_skill_receipt"
                    if state_tool_hit is not None
                    else "runtime_stamped_turn_receipt"
                ),
            }
        incompatible_conversation_contract = bool(
            contract.reason != "ordinary_dialogue"
            or turn_completed_capabilities
            or runtime_context.get("desktop_execution_contract", False)
            or runtime_context.get("capability_inventory_contract", False)
            or runtime_context.get("memory_state_contract", False)
            or runtime_context.get("runtime_fact_status_contract", False)
            or runtime_context.get("grounded_runtime_status_contract", False)
            or runtime_context.get("self_condition_contract", False)
            or runtime_context.get("strict_answer_contract", False)
            or runtime_context.get("strict_value_contract", False)
            or runtime_context.get("proof_evaluation_contract", False)
            or runtime_context.get("operator_evidence_contract", False)
        )
        resume_capability = project_user_surface_resume_capability(
            runtime_context,
            continuation_contract=incoming_continuation_contract,
            user_surface=not is_background and not is_test_run,
            conversation_contract_compatible=(
                incoming_continuation_contract
                or not incompatible_conversation_contract
            ),
        )
        if resume_capability.rejected:
            state.response_modifiers["exact_resume_transport_refusal"] = (
                resume_capability.to_dict()
            )
        return resume_capability, turn_completed_capabilities

    def _execute_affect_modulated_generation(
        self,
        deep_handoff: bool,
        is_background: bool,
        live_mind_controls_bound: bool,
        live_mind_generation_controls: Any,
        runtime_context: Any,
        state: Any,
    ) -> tuple[Any, Any]:
        # Affect-modulated generation parameters
        from .response_generation import (
            CognitiveMode,
            logger,
        )

        affect = getattr(state, "affect", None)
        curiosity = getattr(affect, "curiosity", 0.5) if affect else 0.5
        temp_mod = 0.8 + (curiosity * 0.4)  # 0.8–1.2 range based on curiosity
        depth_mod = 1.0
        if state.cognition.current_mode == CognitiveMode.DELIBERATE:
            depth_mod = 1.5

        token_budget = (
            int((6144 if deep_handoff else 4096) * depth_mod) if not is_background else 1024
        )
        generation_temperature, token_budget = self._apply_generation_sampling_bias(
            base_temperature=0.7 * temp_mod,
            token_budget=token_budget,
            biases=[
                state.response_modifiers.get("sampling_bias"),
                state.response_modifiers.get("imagination_sampling_bias"),
                state.response_modifiers.get("bicameral_sampling_bias"),
                state.response_modifiers.get("cognitive_situation_sampling_bias"),
            ],
        )
        # The caller owns the user-facing latency envelope.  Sampling and
        # cognitive biases may spend less of that budget, but they must not
        # multiply past it.  This keeps the full phase stack available on
        # live desktop turns without turning an ordinary follow-up into a
        # multi-minute local generation.
        requested_token_cap = runtime_context.get("max_tokens")
        if requested_token_cap is not None:
            try:
                token_budget = min(token_budget, max(1, int(requested_token_cap)))
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "ResponseGeneration ignored invalid caller token cap %r.",
                    requested_token_cap,
                )
        if live_mind_controls_bound:
            generation_temperature = max(
                0.10,
                min(
                    1.15,
                    self._safe_bias_float(
                        live_mind_generation_controls.get("temperature"),
                        generation_temperature,
                    ),
                ),
            )
        return generation_temperature, token_budget

    async def _execute_tenth_clock_same(
        self,
        is_background: bool,
        latent_trace: dict[str, Any],
        ordinary_timeout: Any,
        origin: str,
        router: Any,
        router_generation_metadata_sink: dict[str, Any],
        think_coro: Any,
    ) -> tuple[Any, Any]:
        # The tenth clock, and the same lesson as the other nine.
        #
        # asyncio.wait_for cancels on a stopwatch. LIVE
        # 2026-08-29: "ResponseGeneration Phase TIMEOUT (476s).
        # Logic took too long." on a turn that was running a tool
        # and composing an answer from it, and the person got the
        # canned apology.
        #
        # The helper waits while tokens are arriving, gives up on
        # silence, and stays inside the turn's own ceiling. This
        # phase composes what a person reads, so a person is
        # waiting on it unless the state says otherwise.
        from core.brain.llm_health_router import (
            _await_while_it_is_working,
        )
        from core.runtime.turn_origin import a_person_is_waiting

        from .response_generation import (
            _resolve_request_generation_metadata,
            generation_metadata_of,
        )

        response_text = await _await_while_it_is_working(
            think_coro,
            budget_s=ordinary_timeout + 2.0,
            user_facing=True,
            # The facts this call already has: the origin it is
            # about to stamp on the request, and whether it is
            # background. A background generation has nobody
            # waiting whatever its origin says.
            person_is_waiting=(
                not is_background and a_person_is_waiting(origin)
            ),
        )
        attributed_generation_metadata = generation_metadata_of(
            response_text
        )
        generation_metadata = _resolve_request_generation_metadata(
            sink=router_generation_metadata_sink,
            task_snapshot=self._generation_metadata_snapshot(router),
            attributed=attributed_generation_metadata,
            latent_trace=latent_trace,
        )
        return generation_metadata, response_text

    @staticmethod
    def _execute_amplifier_generation_metadata(
        generation_metadata: dict[str, Any],
        latent_trace: dict[str, Any],
        pre_amplifier_text: Any,
        response_mutation_receipt: dict[str, Any],
        response_text: Any,
    ) -> dict[str, Any]:
        from .response_generation import (
            append_text_mutation,
            generation_metadata_of,
            merge_text_mutations,
            summarize_text_mutation_authorship,
        )

        amplifier_generation_metadata = generation_metadata_of(response_text)
        amplifier_source_answer = str(
            getattr(response_text, "reasoning_source_answer", response_text)
            or ""
        )
        append_text_mutation(
            response_mutation_receipt,
            stage="response_generation.reasoning_amplifier",
            method="verifier_backed_candidate_replacement",
            reasons=["reasoning_amplifier_selected_candidate"],
            before=pre_amplifier_text,
            after=amplifier_source_answer,
            deterministic=False,
            authorship_effect="replaced_by_model",
        )
        amplifier_text_mutations = getattr(
            response_text,
            "reasoning_text_mutations",
            [],
        )
        if amplifier_text_mutations:
            merged_amplifier_mutations = merge_text_mutations(
                response_mutation_receipt.get("text_mutations"),
                amplifier_text_mutations,
            )
            response_mutation_receipt["text_mutations"] = (
                merged_amplifier_mutations
            )
            response_mutation_receipt["text_mutation_count"] = len(
                merged_amplifier_mutations
            )
            response_mutation_receipt["deterministic_repair_applied"] = any(
                bool(item.get("deterministic"))
                for item in merged_amplifier_mutations
            )
            response_mutation_receipt.update(
                summarize_text_mutation_authorship(
                    merged_amplifier_mutations
                )
            )
        if response_text != pre_amplifier_text:
            generation_metadata = {
                **amplifier_generation_metadata,
                **latent_trace,
            }
        return generation_metadata

    @staticmethod
    def _execute_defensive_hardening_json(
        append_only_continuation_pending: Any,
        kwargs: Any,
        objective: str | None,
        response_mutation_receipt: dict[str, Any],
        response_text: Any,
        state: Any,
    ) -> tuple[Any, Any]:
        # 4. Defensive Hardening: JSON Repair & Proactive Extraction
        from .response_generation import (
            CognitiveMode,
            _record_response_generation_degradation,
            append_text_mutation,
            logger,
        )

        content = response_text
        action = None

        # PROACTIVE JSON EXTRACTION:
        # If the response contains a JSON-like structure with "content", extract it
        # regardless of the current mode. This prevents raw "philosophical_insight"
        # JSON from leaking into the UI if the LLM slips into JSON mode accidentally.
        if (
            not append_only_continuation_pending
            and "{" in response_text
            and '"content":' in response_text
        ):
            try:
                import re

                # Find the outermost { ... } block
                match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if match:
                    potential_json = match.group(1)
                    from core.utils.json_utils import extract_json

                    data = extract_json(potential_json)
                    if isinstance(data, dict):
                        # Try both "content" and deeper "response": {"content": ...}
                        ext_content = data.get("content")
                        if (
                            not ext_content
                            and "response" in data
                            and isinstance(data["response"], dict)
                        ):
                            ext_content = data["response"].get("content")
                            if not action:
                                action = data["response"].get("action")

                        if ext_content:
                            logger.info(
                                "🛡️ [HARDENING] Proactively extracted content from accidental JSON block."
                            )
                            content = ext_content
                            if not action:
                                action = data.get("action")
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_response_generation_degradation(
                    e,
                    action="continued with raw response after proactive JSON extraction failed",
                )
                logger.debug("Proactive JSON extraction failed (normal for non-JSON): %s", e)

        # Mode-specific validation for DELIBERATE reasoning
        if (
            not append_only_continuation_pending
            and state.cognition.current_mode == CognitiveMode.DELIBERATE
            and not action
        ):
            from core.llm.llm_guard import validate_json_response

            success, obj, err = validate_json_response(response_text, expected_keys=["content"])
            if success:
                content = obj["content"]
                action = obj.get("action")
            else:
                # Robust fallback: if LLM failed to return valid JSON, or returned plain text instead of JSON
                import json
                import re

                is_json_like = response_text.strip().startswith("{") and response_text.strip().endswith("}")
                if not is_json_like:
                    logger.info("🛡️ [HARDENING] DELIBERATE mode validation failed, but output is not JSON-like. Reverting to plain text.")
                    content = response_text
                else:
                    # It is JSON-like but parsing or validation failed. Let's see if we can extract "content" via regex
                    content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text)
                    if content_match:
                        try:
                            content = json.loads(f'"{content_match.group(1)}"')
                            logger.info("🛡️ [HARDENING] DELIBERATE mode validation recovered content from JSON-like response using regex.")
                        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                            content = response_text
                    else:
                        content = response_text

        append_text_mutation(
            response_mutation_receipt,
            stage="response_generation.structured_output_extraction",
            method="deterministic_structured_content_extraction",
            reasons=["model_wrapper_removed"],
            before=response_text,
            after=content,
            deterministic=True,
            authorship_effect="preserved",
        )

        # Proactive XML Answer Tag formatting guard:
        # If the user prompt or system instruction requires XML answer tagging (e.g. "<answer>"),
        # but the model's generated text doesn't contain a valid "<answer>...</answer>" tag:
        # We use a robust regex parsing cascade to extract the plain-text answer from the model's explanation,
        # and automatically wrap it in a clean "<answer>...</answer>" block at the end of the text.
        lower_objective = objective.lower() if objective else ""
        lower_response = content.lower() if content else ""
        if (
            not append_only_continuation_pending
            and ("<answer>" in lower_objective or "answer_format" in kwargs)
            and content
            and "<answer>" not in lower_response
        ):
            import re
            extracted_ans = None

            # 1. Look for markdown bolded final answer (e.g. **Answer**: 5 or **Final Answer**: Alice)
            match = re.search(r"\*\*(?:final\s+)?answer\*\*:\s*([^\n]+)", content, re.IGNORECASE)
            if match:
                extracted_ans = match.group(1).strip()

            # 2. Look for plain-text answer prefix (e.g. Final Answer: same)
            if not extracted_ans:
                match = re.search(r"(?:final\s+)?answer:\s*([^\n]+)", content, re.IGNORECASE)
                if match:
                    extracted_ans = match.group(1).strip()

            # 3. Look for concluding "therefore, the answer is X"
            if not extracted_ans:
                match = re.search(r"(?:therefore|thus|hence|so),\s*(?:the\s+)?answer\s+(?:is|must\s+be)\s+([^\n.]+)", content, re.IGNORECASE)
                if match:
                    extracted_ans = match.group(1).strip()

            # 4. If the response is short enough (e.g. under 60 chars) and has no explanation, use the whole text
            if not extracted_ans and len(content.strip()) < 60 and not any(k in lower_response for k in ("because", "since", "as we", "therefore")):
                extracted_ans = content.strip()

            if extracted_ans:
                # Clean trailing punctuation
                extracted_ans = extracted_ans.rstrip(".,;:!?* ")
                # Wrap and append
                pre_answer_tag_text = content
                content += f"\n\n<answer>{extracted_ans}</answer>"
                append_text_mutation(
                    response_mutation_receipt,
                    stage="response_generation.answer_tag_formatting",
                    method="deterministic_answer_tag_append",
                    reasons=["required_answer_tag"],
                    before=pre_answer_tag_text,
                    after=content,
                    deterministic=True,
                    authorship_effect="preserved",
                )
                logger.info("🛡️ [HARDENING] Auto-corrected and wrapped extracted answer '%s' in XML tags.", extracted_ans)
        return action, content

    @staticmethod
    def _execute_surface_control_receipt(
        final_latent_quality: dict[str, Any],
        generation_metadata: Any,
        latent_trace: dict[str, Any],
        live_mind_controls_bound: bool,
        live_mind_generation_controls: Any,
        response_mutation_receipt: dict[str, Any],
        state: Any,
    ) -> None:
        from .response_generation import (
            merge_text_mutations,
            normalize_live_mind_surface_control_receipt,
            summarize_text_mutation_authorship,
        )

        surface_control_receipt: dict[str, Any] = {}
        candidate = generation_metadata.get("surface_control_receipt")
        if isinstance(candidate, dict):
            surface_control_receipt = dict(candidate)
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=live_mind_controls_bound,
            generation_controls=live_mind_generation_controls,
            source="response_generation_live_mind_controls",
        )
        merged_mutations = merge_text_mutations(
            surface_control_receipt.get("text_mutations"),
            response_mutation_receipt.get("text_mutations"),
        )
        surface_control_receipt["text_mutations"] = merged_mutations
        surface_control_receipt["text_mutation_count"] = len(merged_mutations)
        surface_control_receipt["deterministic_repair_applied"] = any(
            bool(item.get("deterministic")) for item in merged_mutations
        )
        surface_control_receipt.update(
            summarize_text_mutation_authorship(merged_mutations)
        )
        if final_latent_quality:
            surface_control_receipt["surface_quality_gate_enabled"] = True
            surface_control_receipt["surface_quality_gate_passed"] = bool(
                final_latent_quality.get("passed") is True
            )
            surface_control_receipt["surface_quality_gate_attempts"] = 1
            surface_control_receipt["surface_quality_gate_reasons"] = list(
                final_latent_quality.get("reasons") or []
            )
            surface_control_receipt[
                "latent_final_output_quality"
            ] = dict(final_latent_quality)
        state.response_modifiers["live_mind_surface_control_receipt"] = dict(
            surface_control_receipt
        )
        state.response_modifiers["live_mind_controls_worker_applied"] = bool(
            surface_control_receipt.get("live_mind_controls_bound")
            and surface_control_receipt.get("applied")
        )
        latent_trace["latent_cortex_final_text_transformed"] = bool(
            response_mutation_receipt.get("text_mutations")
        )
        for key, value in latent_trace.items():
            state.response_modifiers[key] = value

    @staticmethod
    def _execute_derive_new_state(
        _shaped_messages: Any,
        _speech_profile: Any,
        _sve: Any,
        action: Any,
        cleaned_response: Any,
        is_background: bool,
        is_test_run: bool,
        objective: str | None,
        state: Any,
    ) -> Any:
        # 7. Derive new state with the response
        from .response_generation import (
            _record_response_generation_degradation,
            logger,
            schedule_conversation_support_updates,
        )

        new_state = state.derive("response_generation")
        new_state.cognition.working_memory.append(
            {
                "role": "assistant",
                "content": str(cleaned_response),
                "timestamp": float(time.time()),
                "mode": str(state.cognition.current_mode.value),
                "objective_ref": "".join(
                    [str(objective)[i] for i in range(min(50, len(str(objective))))]
                ),
                "action": action,
            }
        )
        new_state.cognition.last_thought_at = time.time()
        # Set last_response so RepairPhase can inspect and clean it
        new_state.cognition.last_response = str(cleaned_response)

        # One bounded owner observes conversational model updates and
        # shared-ground callbacks under the same stable partner identity.
        schedule_conversation_support_updates(
            str(objective or ""),
            str(cleaned_response or ""),
            state,
        )

        # ── SUBSTRATE VOICE: Follow-up decision ──────────────────────
        # Ask the substrate if a follow-up is warranted. This is organic,
        # not forced — driven by actual curiosity/engagement/dopamine.
        if _sve and _speech_profile and not is_background and cleaned_response and not is_test_run:
            try:
                history = [
                    {"role": m.get("role", ""), "content": str(m.get("content", ""))}
                    for m in (state.cognition.working_memory or [])[-8:]
                ]
                fu_decision = _sve.decide_followup(
                    user_message=str(objective),
                    aura_response=str(cleaned_response),
                    state=state,
                    conversation_history=history,
                )
                if fu_decision.should_followup:
                    # Store decision in state for the orchestrator to pick up
                    new_state.response_modifiers["pending_followup"] = {
                        "type": fu_decision.followup_type,
                        "delay": fu_decision.delay_seconds,
                        "word_budget": fu_decision.word_budget,
                        "context_hint": fu_decision.context_hint,
                        "reason": fu_decision.reason,
                    }
                    logger.info(
                        "💬 [SubstrateVoice] Follow-up queued: %s in %.1fs",
                        fu_decision.followup_type,
                        fu_decision.delay_seconds,
                    )

                # Queue additional shaped messages (from multi-message split)
                if _shaped_messages:
                    new_state.response_modifiers["queued_messages"] = _shaped_messages
            except (OSError, ConnectionError, TimeoutError) as _fu_exc:
                _record_response_generation_degradation(
                    _fu_exc,
                    action="returned primary response without queuing substrate follow-up",
                )
                logger.debug("Follow-up decision failed: %s", _fu_exc)
        return new_state

