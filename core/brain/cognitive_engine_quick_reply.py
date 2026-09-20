"""The desktop quick reply, and the recovery it falls back to.

Lifted whole out of `cognitive_engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cognitive_engine import (
        ThinkingMode,
        Thought,
    )


class _AnswersTheDesktopDirectly:
    """Lifted whole out of CognitiveEngine; see cognitive_engine.py."""

    async def _direct_user_facing_recovery(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        reason: str,
    ) -> Thought | None:
        from .cognitive_engine import (
            _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
            Thought,
            get_container,
            logger,
            record_degradation,
            uuid,
        )

        if not self._is_user_facing_origin(origin):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)
        if router is None or not hasattr(router, "think"):
            return None

        max_tokens = 384 if len(str(objective or "")) <= 900 else 640
        system_prompt = (
            "You are Aura's live CognitiveEngine recovery path. The main phase loop "
            "timed out or failed, but the user still needs one coherent answer. "
            "Answer the current user request directly and honestly. Do not mention "
            "reactive recovery, fallback, internal errors, hidden gates, or implementation "
            "details unless the user specifically asked for them."
        )
        try:
            content = await asyncio.wait_for(
                router.think(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": objective},
                    ],
                    origin=f"recovery_{origin}",
                    prefer_tier="primary",
                    foreground_request=True,
                    protected_foreground_lane=True,
                    is_background=False,
                    deep_handoff=False,
                    allow_deep_handoff=False,
                    allow_cloud_fallback=False,
                    skip_runtime_payload=False,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    max_tokens=max_tokens,
                    num_predict=max_tokens,
                    timeout=15.0,
                ),
                timeout=17.0,
            )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as rec_err:
            record_degradation(
                "cognitive_engine",
                rec_err,
                severity="degraded",
                action="continued after bounded user-facing direct recovery failed",
            )
            logger.warning("Bounded CognitiveEngine direct recovery failed (%s): %s", reason, rec_err)
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            return None

        thought = Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.65,
            reasoning=[
                f"Bounded user-facing direct recovery succeeded after cognitive failure: {reason}",
                "Recovery used the governed primary router with compact payload and no deep handoff.",
            ],
        )
        self.thoughts.append(thought)
        return thought

    @staticmethod
    def _direct_desktop_quick_reply_prompt_shape(context, objective, self_condition_contract_covers_turn):
        from .cognitive_engine import (
            _turn_wants_a_derivation,
            answer_surface_token_floor,
        )

        prompt_shape = context.get("prompt_shape")
        if not isinstance(prompt_shape, dict):
            prompt_shape = {}
        visible_capacity_request = str(
            context.get("visible_user_message") or objective or ""
        ).strip()
        structural_answer_floor = answer_surface_token_floor(
            visible_capacity_request
        )
        # How much room an answer needs is a property of the question, not of
        # the lane it arrived on. "When does the second train catch the first,
        # and how far from the station?" is a two-part derivation either way;
        # on the quick lane it got the 512 floor, ran out mid-derivation at
        # "The first train has been traveling for 3:00pm + 2.25", and was
        # trimmed back to the last complete sentence — so the answer was never
        # given. Measured live 2026-07-27, along with a recall answer cut at
        # "Probably just read".
        shape_wants_room = bool(
            context.get("bounded_planning_contract", False)
            or prompt_shape.get("prefers_extended_answer", False)
            or prompt_shape.get("requires_single_reply_coverage", False)
            or int(prompt_shape.get("question_parts", 0) or 0) >= 2
            or structural_answer_floor > 256
            or _turn_wants_a_derivation(
                str(context.get("visible_user_message") or objective or "")
            )
        )
        if self_condition_contract_covers_turn:
            # A multi-clause condition check remains one bounded state report.
            # The route has already proved that no planning, execution,
            # retrieval, identity, or memory contract competes for coverage.
            # Treating its evidence distinctions as independent long-form asks
            # previously expanded a 256-token answer into a 1,024-token job.
            shape_wants_room = False
        return shape_wants_room, structural_answer_floor

    @staticmethod
    def _direct_desktop_quick_reply_part_2(advisory_factors, capability_inventory_contract, context, extended_full_mind_reply, max_tokens, memory_state_contract, runtime_fact_status_contract, shape_wants_room, structural_answer_floor):
        from .cognitive_engine import (
            _combine_advisory_token_factors,
        )

        if advisory_factors:
            max_tokens = max(
                128,
                int(max_tokens * _combine_advisory_token_factors(advisory_factors)),
            )
        # A narrow budget requires a narrow turn.
        #
        # These caps are right for what they were written for: "what did I
        # pin?" and "how much RAM?" have bounded, machine-readable answers.
        # A self-condition question is natural conversation: its evidence is
        # structured, but the answer is authored by the resident model and can
        # legitimately need more than a status-line budget. These caps are
        # wrong the moment such a request shares a message with something
        # substantive — the cap then sizes the whole turn by its smallest part.
        # Measured live 2026-07-27:
        # a pin-plus-philosophy message drew 172 tokens, ran out mid-sentence,
        # and the answer was discarded as a truncated tail.
        #
        # memory_state_contract_covers_turn is chat.py reporting what its own
        # parser found; the other two narrow contracts defer to the same
        # question-shape signal the comment above already establishes.
        narrow_state_contract = bool(runtime_fact_status_contract)
        if memory_state_contract and context.get(
            "memory_state_contract_covers_turn", True
        ):
            narrow_state_contract = True
        if narrow_state_contract and not shape_wants_room:
            max_tokens = max(128, min(max_tokens, 256))
        elif capability_inventory_contract and not shape_wants_room:
            max_tokens = max(160, min(max_tokens, 220))
        elif extended_full_mind_reply:
            max_tokens = max(1024, structural_answer_floor, min(max_tokens, 4096))
        elif shape_wants_room:
            # Capacity follows the visible work contract. Natural EOS keeps a
            # short answer short; reducing a five-part answer to a generic
            # middle band only guarantees a later retry.
            max_tokens = max(896, structural_answer_floor, min(max_tokens, 4096))
        else:
            # 512-token floor: a live conversational reply must have room to
            # finish its sentences even after advisory reductions.
            max_tokens = max(512, min(max_tokens, 1024))
        # ``max_tokens`` is already the question-shaped, route-approved budget.
        # Preserve it unless memory pressure is truly critical. The MLX gate
        # keeps 64/32-token critical and emergency caps hard.
        completion_floor = max_tokens
        return completion_floor, max_tokens

    @staticmethod
    def _direct_desktop_quick_reply_authority_head_always(capability_inventory_contract, completion_retry_contract, continuation_contract, memory_state_contract, obligation_contract, runtime_fact_status_contract, self_condition_contract, style_contract, visible_user_message):
        # ONE authority head, always the same bytes.
        #
        # This used to be five hand-written system prompts selected by contract
        # flag, all opening with the same sentence and differing after it. That
        # made the front of the prompt a different token sequence on almost
        # every turn, and the front of the prompt is the only part a KV cache
        # can reuse. Measured live 2026-09-07: two consecutive turns of one
        # conversation produced authority heads of 587 and 459 characters with
        # different digests, the prompt cache matched 0 tokens of 1,844, and
        # prefill was 17.7s of a 22s turn.
        #
        # The principle is the one already written forty lines below about
        # per-turn control state: what governs THIS turn belongs next to the
        # turn, not in the head every turn shares. These directives govern one
        # turn, so they travel with it.
        from .cognitive_engine import (
            _DESKTOP_AUTHORITY_HEAD,
            _record_the_capability_inventory_miss,
        )

        system_prompt = _DESKTOP_AUTHORITY_HEAD
        turn_dynamic_contracts: list[str] = []
        if self_condition_contract:
            turn_dynamic_contracts.append(
                "Answer whether you are okay from the canonical self-condition evidence. "
                "Put the direct condition answer first, then one or two natural grounding "
                "sentences. Affect, welfare, felt coherence, continuity, and agency are the "
                "answer; CPU, RAM, host load, and availability are supporting body context "
                "only. Do not replace an inner-state answer with resource telemetry or a "
                "generic presence reassurance."
            )
        elif memory_state_contract:
            turn_dynamic_contracts.append(
                "Answer the current user message directly in one compact, natural paragraph. "
                "Use canonical memory/state evidence as source of truth. "
                "The current user message has priority over older topics. "
                "Do not mention prompt contracts, internal recovery, or implementation details."
            )
        elif runtime_fact_status_contract:
            turn_dynamic_contracts.append(
                "Answer the current runtime-path question directly and compactly. "
                "Use only the verified runtime status evidence supplied for this turn; "
                "do not infer tool readiness, model identity, fallback state, or recurrent "
                "depth from general knowledge. Do not mention hidden prompt contracts."
            )
        elif capability_inventory_contract:
            turn_dynamic_contracts.append(
                "Answer the current capability question from the supplied capability evidence only. "
                "Write exactly four short complete sentences under 80 words total. Sentence order matters: "
                "first list practical capability categories and include the exact phrase browser/web research; second name governed execution through "
                "Will/Authority and permissions; third name receipts or effect verification; fourth give "
                "one hypothetical chain and explicitly say you are not executing tools in this turn. "
                "Do not recite telemetry, prompt contracts, or a generic assistant identity."
            )
        else:
            turn_dynamic_contracts.append(
                "Answer the user's current message directly and naturally. "
                "Use the current conversation rather than a canned status line. "
                "The current user message has priority over all recalled context. "
                "When recent conversation context is provided, use it only for continuity; do not continue "
                "or answer an older topic unless the current user message explicitly asks you to recall or continue it. "
                "Do not mention hidden fallback paths, internal recovery, prompt contracts, or implementation details "
                "unless the user specifically asks for them."
            )
        if (
            completion_retry_contract
            and not continuation_contract
            and not obligation_contract
        ):
            # Append to the stable ordinary-chat prefix. The prefix can still
            # reuse resident KV, while the suffix gives the replacement its
            # only special instruction. Never include the rejected fragment:
            # a partial answer is a powerful continuation anchor and tended to
            # reproduce the same cutoff.
            turn_dynamic_contracts.append(
                "Regenerate the answer from the beginning. "
                "Cover every requested part, finish every sentence, and end "
                "with the requested conclusion. Prefer a concise complete "
                "answer over an unfinished exhaustive one."
            )
        # Four subsystems used to speak to the model in English here, and all
        # four already actuate for real: the neurodynamics, imagination,
        # bicameral and cognitive-situation frames each publish a sampling bias
        # that moves temperature, top-p and the token budget inside bounds the
        # affective controls respect.
        #
        # The argument against the sentences is written forty lines above
        # `_apply_neurodynamic_sampling_bias`, about the same subsystem: its
        # only actuator was a sentence, and "asking the model nicely is not a
        # mechanism". The mechanism was built. The sentences were left beside
        # it, so every turn carried both — a real bias AND a weaker duplicate
        # of it in prose.
        #
        # LIVE, 2026-08-28: a 192-character question was sent with 8,049
        # characters of system prompt, in three blocks, all of them
        # instruction prose. Cortex produced nothing on half of twelve such
        # turns and the person got a canned apology.
        #
        # What each subsystem does is unchanged. What it stopped doing is
        # asking.
        # The desktop conversation lane builds its own system prompt, so the
        # grounding wired into inference_gate never reached the turns people
        # actually take: after that fix landed and the runtime restarted, "what
        # is it actually like in there right now?" still answered "the sun's up
        # ... clouds gathering in the east" at 00:53 in the morning, word for
        # word. Two prompt builders, one of them ungrounded, and this is the one
        # every real conversation goes through.
        _record_the_capability_inventory_miss(
            capability_inventory_contract=capability_inventory_contract,
            system_prompt=system_prompt,
            visible_user_message=visible_user_message,
        )

        if style_contract and not capability_inventory_contract:
            turn_dynamic_contracts.append(style_contract)
        return system_prompt, turn_dynamic_contracts

    @staticmethod
    def _direct_desktop_quick_reply_task_grounding_blocks(capability_inventory_contract, live_mind_context, live_speech_frame, memory_state_contract, mind_context_contract, self_condition_contract):
        from .cognitive_engine import (
            _note_the_quick_reply_contract,
            get_lesion_registry,
            influence_channels,
        )

        task_grounding_blocks: list[str] = []
        ambient_grounding_blocks: list[str] = []
        # The block below tells the model its own state is "causal grounding for
        # the reply". Whether it is, is a measurement, and this is the switch
        # that lets the measurement happen: lesioned, the whole block is absent
        # and the turn runs without ever being told about the mind behind it.
        mind_context_lesioned = get_lesion_registry().is_lesioned(
            influence_channels.LIVE_MIND_CONTEXT_BLOCK
        )
        _note_the_quick_reply_contract(
            ambient_grounding_blocks=ambient_grounding_blocks,
            capability_inventory_contract=capability_inventory_contract,
            live_mind_context=live_mind_context,
            memory_state_contract=memory_state_contract,
            mind_context_contract=mind_context_contract,
            mind_context_lesioned=mind_context_lesioned,
            self_condition_contract=self_condition_contract,
        )
        if isinstance(live_speech_frame, dict) and live_speech_frame and not capability_inventory_contract:
            compact_frame = {
                key: live_speech_frame.get(key)
                for key in (
                    "attention_focus",
                    "dominant_action",
                    "dominant_emotions",
                    "interests",
                    "mood",
                    "tone",
                    "requires_explicit_live_grounding",
                )
                if live_speech_frame.get(key) not in (None, "", [], {})
            }
            if compact_frame:
                ambient_grounding_blocks.append(
                    "[LIVE SPEECH GROUNDING]\n"
                    f"{compact_frame}\n"
                    "This frame is grounding, not prose to repeat. Convert it into ordinary speech only when it helps answer the user.\n"
                    "[END LIVE SPEECH GROUNDING]"
                )
        return ambient_grounding_blocks, task_grounding_blocks

    @staticmethod
    def _direct_desktop_quick_reply_context_challenge_evidence(canonical_memory_state_evidence, canonical_self_condition_context, context, contract_grounding_blocks, discourse_repair_contract, runtime_fact_status_contract, self_condition_contract, task_grounding_blocks, user_prompt):
        from .cognitive_engine import (
            logger,
        )

        context_challenge_evidence = str(
            context.get("contextual_relevance_evidence") or ""
        ).strip()
        if context_challenge_evidence:
            task_grounding_blocks.append(
                "[CONTEXT CHALLENGE EVIDENCE]\n"
                f"{context_challenge_evidence}\n"
                "Use this to repair context confusion. Do not invent a pitch, project, or prior object "
                "that is not supported by this evidence. Answer in one or two complete sentences under "
                "70 words and end with normal punctuation."
            )
        action_episode_evidence = str(
            context.get("action_episode_evidence") or ""
        ).strip()
        if action_episode_evidence:
            task_grounding_blocks.append(action_episode_evidence)
        from core.conversation.answer_provenance import AnswerProvenance, provenance_grounding_json

        prior_answer_provenance = AnswerProvenance.from_value(
            context.get("prior_answer_provenance")
        )
        if prior_answer_provenance is not None:
            task_grounding_blocks.append(provenance_grounding_json(prior_answer_provenance))
        recall_evidence = str(context.get("conversation_recall_evidence") or "").strip()
        if recall_evidence:
            task_grounding_blocks.append(
                "[CONVERSATION RECALL EVIDENCE]\n"
                f"{recall_evidence}\n"
                "Use this as the source of truth for the current recall question."
            )
        deep_memory = str(context.get("deep_memory_context") or "").strip()
        if deep_memory:
            task_grounding_blocks.append(
                "[DEEP MEMORY RECALL]\n"
                f"{deep_memory}\n"
                "Silent background recall from long-term memory. Draw on it only where "
                "it is genuinely relevant to the user's message; never recite it, and "
                "never present it as something the user just said."
            )
        if canonical_memory_state_evidence:
            contract_grounding_blocks.append(
                "[CANONICAL MEMORY STATE EVIDENCE]\n"
                f"{canonical_memory_state_evidence}\n"
                "Use this canonical memory/state result as the source of truth for this turn. "
                "If it contains an exact remembered phrase, include that phrase visibly. "
                "If the current user also asks for one live-state detail, answer that from the live mind context "
                "without reciting telemetry."
            )
        if self_condition_contract and canonical_self_condition_context:
            contract_grounding_blocks.append(
                "[CANONICAL SELF-CONDITION EVIDENCE]\n"
                f"{canonical_self_condition_context}\n"
                "Answer the condition directly from this projection. Preserve its freshness "
                "and uncertainty boundary. Host resource telemetry may only support, never "
                "replace, the answer."
            )
        declared_interlocutor = context.get("declared_interlocutor")
        if isinstance(declared_interlocutor, dict) and declared_interlocutor:
            # Typed turn data, separate from the utterance. The declaration is
            # retained in the transcript, while the final user role contains
            # only the text Aura must answer.
            contract_grounding_blocks.append(
                "[TURN INTERLOCUTOR]\n"
                + json.dumps(
                    {
                        "display_name": str(
                            declared_interlocutor.get("display_name") or ""
                        )[:80],
                        "speaking_role": "user",
                        "source": str(
                            declared_interlocutor.get("source") or ""
                        )[:80],
                        "authenticated": bool(
                            declared_interlocutor.get("authenticated", False)
                        ),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n[END TURN INTERLOCUTOR]"
            )
        if isinstance(discourse_repair_contract, dict) and discourse_repair_contract:
            # Typed discourse state, not a replacement answer.  The rejected
            # assistant wording was removed from role history above; this
            # preserves only the open semantic dimension the user pursued.
            contract_grounding_blocks.append(
                "[DISCOURSE REPAIR STATE]\n"
                + json.dumps(
                    {
                        "act": "repair_pursuit",
                        "focus_kind": str(
                            discourse_repair_contract.get("focus_kind") or ""
                        ),
                        "focus_terms": list(
                            discourse_repair_contract.get("focus_terms") or ()
                        )[:8],
                        "prior_question": str(
                            discourse_repair_contract.get("prior_question") or ""
                        )[:420],
                        "current_question": str(
                            discourse_repair_contract.get("current_question") or ""
                        )[:420],
                        "prior_answer_status": "rejected_by_user_pursuit",
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n[END DISCOURSE REPAIR STATE]"
            )
        live_capability_condition = str(
            context.get("live_capability_condition") or ""
        ).strip()
        if live_capability_condition:
            # Facts, not a script. She says it however she says things.
            task_grounding_blocks.append(live_capability_condition)
        grounded_runtime_status = str(
            context.get("grounded_runtime_status_context") or ""
        ).strip()
        if runtime_fact_status_contract and grounded_runtime_status:
            contract_grounding_blocks.append(
                "[VERIFIED LIVE RUNTIME STATUS]\n"
                f"{grounded_runtime_status}\n"
                "Use this as the source of truth. Preserve its factual boundaries and do not "
                "invent stronger availability or completion claims."
            )
        capability_evidence = str(
            context.get("grounded_capability_inventory_context") or ""
        ).strip()
        if capability_evidence:
            contract_grounding_blocks.append(
                "[GOVERNED CAPABILITY INVENTORY EVIDENCE]\n"
                f"{capability_evidence}\n"
                "Answer in this exact order: practical categories including the exact phrase browser/web research; governance/Will/Authority/permissions; "
                "receipts or effect verification; one hypothetical chain plus the boundary that you are "
                "not executing tools in this turn. Keep the answer complete and under 120 words."
            )
        bounded_plan_evidence = str(context.get("bounded_planning_reply") or "").strip()
        if bool(context.get("bounded_planning_contract")) and bounded_plan_evidence:
            contract_grounding_blocks.append(
                "[GOVERNED PLANNING OUTLINE]\n"
                f"{bounded_plan_evidence}\n"
                "Treat this as verified workflow structure, not text to copy mechanically. "
                "Answer the current request in one natural paragraph of four to six complete "
                "sentences under 180 words. Cover the goal, authorization boundary, action "
                "sequence, effect verification, and bounded recovery. Do not use a numbered "
                "list unless the user explicitly asks for one."
            )
        self_claim_evidence = str(
            context.get("evidence_bound_self_claim_context") or ""
        ).strip()
        if self_claim_evidence:
            contract_grounding_blocks.append(
                "[EVIDENCE-BOUND SELF-CLAIM EVIDENCE]\n"
                f"{self_claim_evidence}\n"
                "Use this to keep consciousness, sentience, self-awareness, and personhood claims "
                "functional, bounded, and evidence-based."
            )
        try:
            from core.introspection.capability_map import (
                build_capability_map_context,
                is_actionable_request,
            )

            if is_actionable_request(user_prompt):
                # Action requests get the honest lane map so the mind
                # decomposes to granted paths (filesystem → scripting →
                # GUI) instead of declining whole tasks — observed live:
                # a notes+folder+export task declined entirely when only
                # raw GUI control was actually blocked.
                _cap_map = build_capability_map_context()
                if _cap_map:
                    task_grounding_blocks.append("[CAPABILITY MAP]\n" + _cap_map)
        except (ImportError, AttributeError, RuntimeError, OSError) as _cm_exc:
            logger.debug("Capability-map grounding unavailable: %s", _cm_exc)

        try:
            from core.introspection.self_forensics import (
                build_self_forensics_context,
                is_self_forensics_question,
            )

            if is_self_forensics_question(user_prompt):
                # Asked about her own shutdown/crash history, she gets her
                # actual black boxes (grace flag, sentinel log, incident
                # records, faults) — observed live: without this she
                # confabulated electromagnetic interference for a
                # generation-gate wedge, three rejected drafts in a row.
                _forensics = build_self_forensics_context()
                if _forensics:
                    task_grounding_blocks.append(
                        "[SELF-FORENSICS EVIDENCE]\n" + _forensics
                    )
        except (ImportError, AttributeError, RuntimeError, OSError) as _sf_exc:
            logger.debug("Self-forensics grounding unavailable: %s", _sf_exc)
        return action_episode_evidence

    @staticmethod
    async def _direct_desktop_quick_reply_part_6(context, live_mind_generation_controls, obligation_contract, obligation_segment, origin, request_timeout, router, router_generation_metadata, router_kwargs):
        from .cognitive_engine import (
            get_lesion_registry,
            influence_channels,
        )

        if obligation_contract:
            router_kwargs["user_surface_obligation_contract"] = True
            router_kwargs["user_surface_obligation_segment"] = obligation_segment
        # The lesion for this channel is omission, not substitution: a
        # neutral temperature is still a temperature somebody chose, and
        # measuring against one would compare two mind-derived settings
        # instead of comparing the mind's setting against its absence.
        if not get_lesion_registry().is_lesioned(
            influence_channels.LIVE_MIND_GENERATION_CONTROLS
        ):
            if "temperature" in live_mind_generation_controls:
                router_kwargs["temperature"] = live_mind_generation_controls["temperature"]
                router_kwargs["temp"] = live_mind_generation_controls["temperature"]
            if "top_p" in live_mind_generation_controls:
                router_kwargs["top_p"] = live_mind_generation_controls["top_p"]
        router_generation_metadata_sink: dict[str, Any] = {}
        router_kwargs["_generation_metadata_sink"] = (
            router_generation_metadata_sink
        )
        # The ninth clock, and the last hard one on this path.
        #
        # asyncio.wait_for cancels on a stopwatch and cannot tell a
        # generation that is writing from one that has stopped. Every
        # other clock a desktop turn passes through has been taught the
        # difference; this one was still counting.
        #
        # LIVE 2026-08-29: asked what she could work out about herself
        # from what she can measure, the turn ran 185 seconds and ended
        # in "TimeoutError: <no message; raised in
        # asyncio.timeouts:__aexit__>" — the empty message being what a
        # stopwatch has to say about work it did not watch. The person got
        # the canned apology.
        #
        # Same helper as the rest of them: it waits while tokens are
        # arriving, gives up on silence, and is bounded by the turn's own
        # ceiling. This origin is one a person types into, and the caller
        # already said so.
        from core.brain.llm_health_router import _await_while_it_is_working
        from core.runtime.turn_origin import a_person_is_waiting

        content = await _await_while_it_is_working(
            router.think(**router_kwargs),
            budget_s=request_timeout + 3.0,
            user_facing=True,
            # The bare origin, not the decorated one: every
            # "desktop_quick_*" matches the foreground prefix, so asking
            # about the decorated name says yes for an autonomous
            # initiative too. The origin judged as itself is the fact.
            person_is_waiting=a_person_is_waiting(
                origin, stated=context.get("a_person_is_waiting")
            ),
        )
        if router_generation_metadata_sink:
            router_generation_metadata = dict(router_generation_metadata_sink)
        elif hasattr(router, "get_last_generation_metadata"):
            raw_metadata = router.get_last_generation_metadata()
            if isinstance(raw_metadata, dict):
                router_generation_metadata = dict(raw_metadata)
        return content, router_generation_metadata

    @staticmethod
    def _direct_desktop_quick_reply_semantic_completion_incomplete(generation_stop_reason, surface_reasons, surface_receipt, text):
        from .cognitive_engine import (
            _truncation_verdict,
            record_degradation,
        )

        semantic_completion_incomplete = bool(
            surface_receipt.get("semantic_completion_incomplete", False)
        )
        reply_generation_incomplete = bool(
            semantic_completion_incomplete
            or "truncated_tail" in surface_reasons
            or generation_stop_reason
            in {"max_tokens", "deadline_exceeded", "soft_cancelled"}
            or _truncation_verdict(
                text,
                generation_stop_reason=generation_stop_reason,
            )
        )
        if reply_generation_incomplete:
            record_degradation(
                "cognitive_engine",
                RuntimeError("desktop_quick_reply_midsentence_cutoff"),
                severity="info",
                action=(
                    "preserved a clipped draft as incomplete so the chat route can "
                    "replace it with a full answer before surfacing"
                ),
            )
        # 0.8 immediately after a nonempty generation, before user feedback,
        # task outcome, factual verification, or even a correlated quality
        # receipt — so a fluent failure reinforced the components that shaped
        # it. The reply is not yet known to be good; what IS known is whether
        # it came out whole. A reply the budget cut mid-sentence is the one
        # signal available here, and it is negative.
        _quick_reward = 0.4 if reply_generation_incomplete else 0.6
        return _quick_reward, reply_generation_incomplete

    async def _direct_desktop_quick_reply(
        self,
        objective: str,
        mode: ThinkingMode,
        origin: str,
        context: dict[str, Any] | None,
        *,
        timeout_s: float,
    ) -> Thought | None:
        from .cognitive_engine import (
            _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
            _SINGLE_PASS,
            _STEERING_OFF,
            Thought,
            _apply_neurodynamic_sampling_bias,
            _bind_live_mind_generation_contract,
            _desktop_history_messages_from_context,
            _fit_history_to_what_is_left,
            _fit_prompt_to_what_the_turn_can_read,
            _live_mind_controls_bound,
            _restore_sentence_spacing,
            apply_channel,
            continuation_prompt_prefix,
            continuation_state_text,
            get_container,
            influence_channels,
            logger,
            normalize_live_mind_surface_control_receipt,
            project_user_surface_resume_capability,
            record_degradation,
            response_policy,
            uuid,
        )

        if not self._is_user_facing_origin(origin):
            return None
        if not isinstance(context, dict) or not bool(context.get("desktop_quick_reply_contract")):
            return None

        container = get_container()
        router = container.get("llm_router", default=None)

        # int() on caller input, outside the guarded router call below: a
        # string or a NaN raised TypeError/ValueError here and took the turn
        # down before any bounded failure thought could be produced. A bad
        # request is a bad request, not a crash.
        max_tokens = self._bounded_request_int(
            context.get("max_tokens"), default=768, low=1, high=32_768
        )
        advice = context.get("spiking_active_inference")
        imagination_frame = context.get("imagination_workspace")
        bicameral_frame = context.get("bicameral_advisory")
        cognitive_situation_frame = context.get("cognitive_situation_frame")
        sampling_sources: list[Any] = []
        if isinstance(advice, dict):
            sampling_sources.append(advice.get("sampling_bias") or {})
        if isinstance(imagination_frame, dict):
            sampling_sources.append(imagination_frame.get("sampling_bias") or {})
        if isinstance(bicameral_frame, dict):
            sampling_sources.append(bicameral_frame.get("sampling_bias") or {})
        if isinstance(cognitive_situation_frame, dict):
            sampling_sources.append(cognitive_situation_frame.get("sampling_bias") or {})
        memory_state_contract = bool(context.get("memory_state_contract", False))
        runtime_fact_status_contract = bool(
            context.get("runtime_fact_status_contract", False)
            or context.get("grounded_runtime_status_contract", False)
        )
        self_condition_contract = bool(context.get("self_condition_contract", False))
        self_condition_contract_covers_turn = bool(
            context.get(
                "self_condition_contract_covers_turn",
                self_condition_contract,
            )
        )
        capability_inventory_contract = bool(context.get("capability_inventory_contract", False))
        identity_continuity_contract = bool(
            context.get("identity_continuity_contract", False)
            or context.get("grounded_identity_continuity_context")
        )
        completion_retry_contract = bool(
            context.get("user_surface_completion_retry", False)
        )
        continuation_partial = continuation_state_text(
            context.get("user_surface_continuation_partial")
        )
        continuation_prefix = continuation_prompt_prefix(continuation_partial)
        continuation_contract = bool(
            completion_retry_contract
            and context.get("user_surface_continuation_contract", False)
            and continuation_partial
        )
        resume_capability = project_user_surface_resume_capability(
            context,
            continuation_contract=continuation_contract,
            conversation_contract_compatible=not any(
                (
                    memory_state_contract,
                    runtime_fact_status_contract,
                    self_condition_contract,
                    capability_inventory_contract,
                    identity_continuity_contract,
                    bool(context.get("desktop_execution_contract", False)),
                    bool(context.get("strict_answer_contract", False)),
                    bool(context.get("strict_value_contract", False)),
                    bool(context.get("proof_evaluation_contract", False)),
                    bool(context.get("operator_evidence_contract", False)),
                    bool(context.get("completed_capability_evidence")),
                )
            ),
        )
        obligation_segment = str(
            context.get("user_surface_obligation_segment") or ""
        ).strip()
        obligation_parent_request = str(
            context.get("user_surface_obligation_parent_request") or ""
        ).strip()
        obligation_partial = continuation_state_text(
            context.get("user_surface_obligation_partial")
        )
        obligation_contract = bool(
            completion_retry_contract
            and context.get("user_surface_obligation_contract", False)
            and obligation_segment
            and obligation_parent_request
        )
        shape_wants_room, structural_answer_floor = self._direct_desktop_quick_reply_prompt_shape(context, objective, self_condition_contract_covers_turn)
        extended_full_mind_reply = bool(
            context.get("require_full_foreground_mind_reply", False) and shape_wants_room
        )
        canonical_memory_state_evidence = str(
            context.get("canonical_memory_state_evidence") or ""
        ).strip()
        canonical_self_condition_context = str(
            context.get("canonical_self_condition_context") or ""
        ).strip()
        advisory_factors: list[float] = []
        for sampling in sampling_sources:
            if isinstance(sampling, dict):
                try:
                    factor_value = float(sampling.get("max_tokens_factor", 1.0))
                except (TypeError, ValueError):
                    factor_value = 1.0
                if 0.25 <= factor_value <= 1.25:
                    if capability_inventory_contract and factor_value < 1.0:
                        continue
                    advisory_factors.append(factor_value)
        completion_floor, max_tokens = self._direct_desktop_quick_reply_part_2(advisory_factors, capability_inventory_contract, context, extended_full_mind_reply, max_tokens, memory_state_contract, runtime_fact_status_contract, shape_wants_room, structural_answer_floor)
        request_timeout_cap = (
            response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S
            if shape_wants_room
            else 180.0
        )
        request_timeout = max(
            12.0,
            min(
                max(12.0, float(timeout_s or 32.0) - 5.0),
                request_timeout_cap,
            ),
        )
        if memory_state_contract or runtime_fact_status_contract or self_condition_contract:
            request_timeout = min(request_timeout, 90.0)
        if capability_inventory_contract:
            request_timeout = min(request_timeout, 28.0)
        style_contract = self._contract_safe(
            context.get("response_style_contract"), self._STYLE_CONTRACT_LIMIT
        )
        visible_user_message = str(context.get("visible_user_message") or objective or "").strip()
        recent_conversation_context = str(context.get("recent_conversation_context") or "").strip()
        history_messages, history_reach_note = _desktop_history_messages_from_context(
            context,
        )
        discourse_repair_contract = context.get("discourse_repair_contract")
        if isinstance(discourse_repair_contract, dict):
            from core.utils.injected_blocks import is_stamped_runtime_payload

            if is_stamped_runtime_payload(discourse_repair_contract) and bool(
                discourse_repair_contract.get("active")
            ):
                from core.conversation.discourse_repair_pursuit import (
                    apply_repair_pursuit_to_history,
                )

                # The user's pursuit rejects the prior answer, not the earlier
                # context.  Leaving that assistant text in history turns it
                # into the strongest few-shot example for the replacement and
                # reproduces the same evasion nearly verbatim.
                history_messages = apply_repair_pursuit_to_history(
                    history_messages,
                    discourse_repair_contract,
                )
            else:
                discourse_repair_contract = {}
        live_speech_frame = context.get("live_speech_grounding_frame")
        live_mind_context = context.get("live_mind_context")
        live_mind_required = bool(context.get("live_mind_context_required", False))
        # The visible request is bound before the cognitive loop starts. Use
        # that turn-owned control contract here instead of deriving a second
        # one from a later view of the same context. Two derivations produced
        # requested-depth=2 in the parent and applied-depth=1 in the worker on
        # one live turn, so neither receipt described the execution it judged.
        live_mind_generation_controls = _bind_live_mind_generation_contract(context)
        # The spiking model's temperature and top-p deltas reach the sampler
        # here. Before this they were computed every turn and dropped, leaving
        # a prompt sentence as the neurodynamics' only actuator.
        live_mind_generation_controls = _apply_neurodynamic_sampling_bias(
            live_mind_generation_controls, advice
        )
        context["live_mind_generation_controls"] = dict(
            live_mind_generation_controls
        )
        live_mind_controls_bound = _live_mind_controls_bound(
            live_mind_context,
            live_mind_generation_controls,
        )
        # The three flags below gate the structured floors, which return
        # self-condition, planning, capability and identity answers at high
        # confidence with live-mind metadata attached. They used to be
        # satisfiable from the caller's own context booleans — and the last
        # branch re-derived controls_bound as True from them, bypassing
        # _live_mind_controls_bound entirely. A caller could therefore mint a
        # proof-bearing reply by asserting that it was entitled to one.
        #
        # A context fallback is still allowed, but only when the payload
        # carries this runtime's stamp: then the booleans are the runtime's own
        # summary of a snapshot it produced, not a claim about itself.
        from core.utils.injected_blocks import is_stamped_runtime_payload

        _context_attested = is_stamped_runtime_payload(live_mind_context)
        live_mind_snapshot_ready = bool(
            isinstance(live_mind_context, dict)
            and isinstance(live_mind_context.get("mind_snapshot_quality"), dict)
            and live_mind_context["mind_snapshot_quality"].get("ready")
        )
        if not live_mind_snapshot_ready and _context_attested:
            live_mind_snapshot_ready = bool(context.get("live_mind_snapshot_ready"))
        live_mind_required_subsystems_ok = bool(
            isinstance(live_mind_context, dict)
            and live_mind_context.get("required_subsystems_ok")
        )
        if not live_mind_required_subsystems_ok and _context_attested:
            live_mind_required_subsystems_ok = bool(
                context.get("live_mind_required_subsystems_ok")
            )
        # controls_bound comes from _live_mind_controls_bound and nowhere else.
        # It used to be re-derived True here from the flags above, which is the
        # check answering to the thing it was checking.
        if live_mind_controls_bound and not (
            live_mind_generation_controls
            and live_mind_snapshot_ready
            and live_mind_required_subsystems_ok
        ):
            live_mind_controls_bound = False
        # A typed self-condition projection is evidence for Aura's answer, not
        # Aura's answer.  Returning it here bypassed the resident model entirely
        # and made an ordinary "how are you?" turn look like a health endpoint.
        # Keep the projection in the grounded prompt below.  The route may use a
        # visibly bounded projection only after model generation and one
        # same-worker corrective attempt have both failed.
        if bool(context.get("bounded_planning_contract")) and not bool(
            context.get("require_full_foreground_mind_reply", False)
        ):
            bounded_reply = str(context.get("bounded_planning_reply") or "").strip()
            if bounded_reply:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_bounded_planning",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_bounded_planning",
                        "bounded_planning_contract": True,
                        "bounded_planning_floor": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=bounded_reply,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Bounded non-executing desktop planning was answered through the CognitiveEngine floor.",
                        "The reply remained governed, non-executing, and attached to live mind proof metadata.",
                    ],
                    metadata=metadata,
                )
        if capability_inventory_contract:
            grounded_inventory = str(
                context.get("grounded_capability_inventory_context") or ""
            ).strip()
            if grounded_inventory:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_capability_catalog_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_capability_catalog_grounding",
                        "capability_inventory_contract": True,
                        "grounded_capability_inventory": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_inventory,
                    mode=mode,
                    confidence=0.86,
                    reasoning=[
                        "Desktop capability inventory was grounded from the governed live capability catalog.",
                        "No foreground model generation was required for this runtime-fact turn.",
                    ],
                    metadata=metadata,
                )
        if identity_continuity_contract:
            grounded_identity = str(
                context.get("grounded_identity_continuity_context") or ""
            ).strip()
            if grounded_identity:
                metadata = self._live_mind_structured_floor_metadata(
                    context,
                    source="cognitive_engine_identity_continuity_grounding",
                )
                metadata.update(
                    {
                        "response_path": "cognitive_engine_identity_continuity_grounding",
                        "identity_continuity_contract": True,
                        "grounded_identity_continuity": True,
                    }
                )
                return Thought(
                    id=str(uuid.uuid4()),
                    content=grounded_identity,
                    mode=mode,
                    confidence=0.88,
                    reasoning=[
                        "Identity and continuity were answered from canonical live identity grounding inside CognitiveEngine.",
                        "The route had already bound live mind context and generation controls, so no recovery model cycle was needed.",
                    ],
                    metadata=metadata,
                )
        if router is None or not hasattr(router, "think"):
            return None
        live_runtime_required = bool(
            context.get("live_runtime_payload_required", False)
            or (live_mind_required and isinstance(live_mind_context, dict))
        )
        system_prompt, turn_dynamic_contracts = self._direct_desktop_quick_reply_authority_head_always(capability_inventory_contract, completion_retry_contract, continuation_contract, memory_state_contract, obligation_contract, runtime_fact_status_contract, self_condition_contract, style_contract, visible_user_message)
        persona_contract = str(context.get("persona_system_prompt") or "").strip()
        if persona_contract:
            # CP126 ab3abbae: persona conditioning arrives as a structured
            # context field and is applied here, at SYSTEM role. It used to be
            # string-prepended into the user objective, where later objective
            # text could override it and it polluted task semantics, caching,
            # memory and audit attribution.
            system_prompt = f"{system_prompt}\n[PERSONA CONTRACT]\n{persona_contract[:2000]}"
        mind_context_contract = self._contract_safe(
            context.get("mind_context_contract"), self._MIND_CONTRACT_LIMIT
        )
        # Per-turn control state belongs next to the turn it governs. Keeping it
        # out of the stable system head lets the resident model reuse the full
        # identity/persona prefix and prior conversation KV across turns.
        contract_grounding_blocks: list[str] = list(turn_dynamic_contracts)
        ambient_grounding_blocks, task_grounding_blocks = self._direct_desktop_quick_reply_task_grounding_blocks(capability_inventory_contract, live_mind_context, live_speech_frame, memory_state_contract, mind_context_contract, self_condition_contract)
        user_prompt = visible_user_message or objective
        try:
            from core.senses.turn_evidence import sensory_evidence_grounding_block

            turn_sensory_evidence = sensory_evidence_grounding_block(
                context.get("turn_sensory_evidence")
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Turn sensory evidence unavailable: %s", exc)
            turn_sensory_evidence = ""
        if turn_sensory_evidence:
            task_grounding_blocks.append(turn_sensory_evidence)
        action_episode_evidence = self._direct_desktop_quick_reply_context_challenge_evidence(canonical_memory_state_evidence, canonical_self_condition_context, context, contract_grounding_blocks, discourse_repair_contract, runtime_fact_status_contract, self_condition_contract, task_grounding_blocks, user_prompt)

        # What the turn can afford to read, now that both claimants exist.
        # The system prompt is what she cannot answer without, so it is
        # served first and the conversation takes the remainder; both are
        # held to the same measured rate.
        history_messages, history_reach_note = _fit_history_to_what_is_left(
            history_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        if history_reach_note:
            ambient_grounding_blocks.append(history_reach_note)

        if recent_conversation_context and not history_messages:
            ambient_grounding_blocks.append(
                "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]\n"
                f"{recent_conversation_context}\n"
                "[END RECENT COMPLETED CONVERSATION]"
            )

        # Completion still needs the admitted conversation: references in the
        # request or partial can depend on any earlier exchange. Keep that
        # transcript while omitting newly assembled telemetry and advice.
        # Sampler controls remain causal through router kwargs.
        if continuation_contract or obligation_contract:
            contract_grounding_blocks = []
            task_grounding_blocks = []
            ambient_grounding_blocks = []

        router_generation_metadata: dict[str, Any] = {}
        try:
            from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

            # The other prompt builder has a budget table and trims to it.
            # This one is the builder every desktop conversation goes through
            # and it had none: measured live on 2026-08-28, a turn whose
            # question was 213 characters carried a 96,430-character system
            # message — 28,147 tokens, ~46 seconds of reading before a token —
            # to serve an answer of fifty.
            #
            # The budget is not a number chosen here. It is what the turn can
            # afford to read given what its own answer costs to write, from
            # rates the reserve measured on this hardware and kept across
            # restarts. A thousand-token answer buys more prompt than the
            # assembler produces and is not trimmed at all.
            system_prompt = _fit_prompt_to_what_the_turn_can_read(
                system_prompt,
                request=visible_user_message or objective,
                max_tokens=max_tokens,
                room_taken=sum(
                    len(str(message.get("content") or ""))
                    for message in (history_messages or [])
                )
                + len(str(visible_user_message or objective or "")),
            )
            messages = [stamp_grounding({"role": "system", "content": system_prompt})]
            if history_messages:
                messages.extend(history_messages)
            grounding_blocks = [
                *contract_grounding_blocks,
                *task_grounding_blocks,
                *ambient_grounding_blocks,
            ]
            if grounding_blocks:
                messages.append(
                    stamp_grounding(
                        {
                            "role": RUNTIME_EVIDENCE_ROLE,
                            "content": (
                                "[GROUNDING EVIDENCE FOR THIS TURN]\n"
                                + "\n\n".join(grounding_blocks)
                                + "\n[END GROUNDING EVIDENCE FOR THIS TURN]"
                            ),
                            "metadata": {
                                "type": "turn_grounding",
                                "snapshot_owner": "cognitive_engine",
                                "evidence_priority": (
                                    "contract",
                                    "task",
                                    "ambient",
                                ),
                                "live_mind_context_bound": bool(
                                    isinstance(live_mind_context, dict)
                                    and live_mind_context
                                    and _context_attested
                                ),
                            },
                        }
                    )
                )
            validation_prompt = visible_user_message or objective
            if obligation_contract:
                messages.append(
                    {"role": "user", "content": obligation_parent_request}
                )
                if obligation_partial:
                    messages.append(
                        {"role": "assistant", "content": obligation_partial}
                    )
                messages.append({"role": "user", "content": obligation_segment})
                validation_prompt = obligation_segment
            else:
                messages.append({"role": "user", "content": user_prompt})
            if continuation_contract:
                messages.append(
                    {"role": "assistant", "content": continuation_prefix}
                )
            router_kwargs = {
                "messages": messages,
                "origin": f"desktop_quick_{origin}",
                "cognitive_mode": mode.name.lower(),
                "prefer_tier": "primary",
                "foreground_request": True,
                "protected_foreground_lane": True,
                "cognitive_engine_required": bool(
                    context.get("cognitive_engine_required", False)
                ),
                "desktop_cognitive_engine_required": bool(
                    context.get("desktop_cognitive_engine_required", False)
                ),
                "is_background": False,
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
                "allow_mesh_cognition": False,
                "skip_runtime_payload": True,
                "memory_state_contract": memory_state_contract,
                "runtime_fact_status_contract": runtime_fact_status_contract,
                "grounded_runtime_status_contract": runtime_fact_status_contract,
                "self_condition_contract": self_condition_contract,
                "self_condition_contract_covers_turn": (
                    self_condition_contract_covers_turn
                ),
                "capability_inventory_contract": capability_inventory_contract,
                "clean_user_surface_contract": True,
                # Every clean user turn measures semantic completion. Natural
                # EOS remains available; this contract preserves an incomplete
                # deadline/max-token draft and its exact continuation state.
                "semantic_completion_contract": True,
                "user_surface_validation_prompt": validation_prompt,
                "user_surface_sensory_evidence": context.get(
                    "turn_sensory_evidence"
                ),
                # Wrapped so a paired trial can run this exact code with the
                # contribution removed, rather than reconstructing what that
                # would have looked like. Outside a trial this is one dict
                # lookup and the value passes through unchanged.
                "clean_user_surface_recurrent_loops": apply_channel(
                    influence_channels.LIVE_MIND_RECURRENT_LOOPS,
                    live_mind_generation_controls.get(
                        "clean_user_surface_recurrent_loops",
                        _SINGLE_PASS,
                    ),
                    neutral=_SINGLE_PASS,
                ),
                "clean_user_surface_steering_alpha": apply_channel(
                    influence_channels.LIVE_MIND_STEERING_ALPHA,
                    live_mind_generation_controls.get(
                        "clean_user_surface_steering_alpha",
                        _STEERING_OFF,
                    ),
                    neutral=_STEERING_OFF,
                ),
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "live_context_already_grounded": bool(
                    isinstance(live_mind_context, dict)
                    and live_mind_context
                    and _context_attested
                ),
                "recent_actions_already_grounded": bool(action_episode_evidence),
                # No disable_prompt_cache here. This is the lane the desktop UI
                # actually talks through (origin=desktop_quick_*), and it was
                # the FOURTH place independently switching the cache off for the
                # conversation — after the chat contract, the inference gate's
                # foreground force-set, and the worker's own bypass list. Each
                # one made the others invisible: lifting three still produced a
                # turn with no cache lookup logged at all.
                #
                # Reuse is scoped to `user_surface` and is KV for a
                # byte-identical prefix, so within the scope the only shared
                # state is this conversation's own history. `clear_prompt_cache`
                # was worse than the disable: it wiped every other lane's entry
                # on every user turn.
                "max_tokens": max_tokens,
                "num_predict": max_tokens,
                # Why this budget, not just how big. The gate's starvation
                # floor is flat 512, so pressure scaling could cut a 896-token
                # derivation to 459 and the floor would "rescue" it back to
                # 512 — the caller's reason for asking was never carried, so
                # the train problem still ran out of room at "- The".
                "reply_needs_room": shape_wants_room,
                "user_surface_completion_floor": completion_floor,
                "sampling_bias": apply_channel(
                    influence_channels.SPIKING_SAMPLING_BIAS,
                    advice.get("sampling_bias") if isinstance(advice, dict) else None,
                    neutral=None,
                ),
                "imagination_sampling_bias": apply_channel(
                    influence_channels.IMAGINATION_SAMPLING_BIAS,
                    (
                        imagination_frame.get("sampling_bias")
                        if isinstance(imagination_frame, dict)
                        else None
                    ),
                    neutral=None,
                ),
                "bicameral_sampling_bias": apply_channel(
                    influence_channels.BICAMERAL_SAMPLING_BIAS,
                    (
                        bicameral_frame.get("sampling_bias")
                        if isinstance(bicameral_frame, dict)
                        else None
                    ),
                    neutral=None,
                ),
                "cognitive_situation_sampling_bias": (
                    cognitive_situation_frame.get("sampling_bias")
                    if isinstance(cognitive_situation_frame, dict)
                    else None
                ),
                "timeout": request_timeout,
            }
            if continuation_contract:
                # One continuation owns the remaining surface. Keep enough
                # capacity for any unserved obligations rather than splitting
                # completion across a chain of progressively smaller decodes.
                continuation_tokens = max(512, min(structural_answer_floor, 1024))
                router_kwargs["max_tokens"] = continuation_tokens
                router_kwargs["num_predict"] = continuation_tokens
                router_kwargs["user_surface_completion_floor"] = continuation_tokens
                router_kwargs["reply_needs_room"] = True
                router_kwargs["user_surface_continuation_contract"] = True
                router_kwargs["user_surface_continuation_partial"] = continuation_partial
            router_kwargs.update(resume_capability.context)
            content, router_generation_metadata = await self._direct_desktop_quick_reply_part_6(context, live_mind_generation_controls, obligation_contract, obligation_segment, origin, request_timeout, router, router_generation_metadata, router_kwargs)
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="degraded",
                action=(
                    "surfaced bounded desktop inference failure without entering "
                    "a second heavyweight model path"
                ),
                enforce_failure_policy=False,
            )
            logger.warning("Desktop quick CognitiveEngine generation failed: %s", exc)
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                return self._desktop_cognitive_failure_thought(
                    mode,
                    f"compact_desktop_generation_failed:{type(exc).__name__}",
                )
            return None

        text = str(content or "").strip()
        if not text or text == "…" or text.startswith("background_thought_suppressed"):
            if bool(
                context.get("desktop_cognitive_engine_required", False)
                or context.get("cognitive_engine_required", False)
            ):
                generation_failure_class = str(
                    router_generation_metadata.get("error") or ""
                ).strip()
                if generation_failure_class != "surface_quality_rejected":
                    record_degradation(
                        "cognitive_engine",
                        RuntimeError("compact desktop generation returned no usable text"),
                        severity="degraded",
                        action=(
                            "surfaced bounded desktop inference failure without entering "
                            "a second heavyweight model path"
                        ),
                        enforce_failure_policy=False,
                    )
                else:
                    logger.warning(
                        "Desktop quick CognitiveEngine generation was intentionally "
                        "rejected by the worker quality gate."
                    )
                return self._desktop_cognitive_failure_thought(
                    mode,
                    generation_failure_class or "compact_desktop_generation_empty",
                    generation_metadata=router_generation_metadata,
                )
            return None
        text = _restore_sentence_spacing(text)
        surface_receipt = (
            router_generation_metadata.get("surface_control_receipt")
            if isinstance(router_generation_metadata, dict)
            else None
        )
        if not isinstance(surface_receipt, dict):
            surface_receipt = {}
        surface_reasons = tuple(surface_receipt.get("surface_quality_gate_reasons") or ())
        generation_stop_reason = str(
            surface_receipt.get("generation_stop_reason") or ""
        )
        _quick_reward, reply_generation_incomplete = self._direct_desktop_quick_reply_semantic_completion_incomplete(generation_stop_reason, surface_reasons, surface_receipt, text)
        imagination_feedback = self._learn_imagination_workspace_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=_quick_reward,
        )
        bicameral_feedback = self._learn_bicameral_advisory_outcome(
            context,
            outcome="desktop_quick_reply",
            reward=_quick_reward,
        )
        surface_control_receipt = (
            router_generation_metadata.get("surface_control_receipt")
            if isinstance(router_generation_metadata, dict)
            else None
        )
        if not isinstance(surface_control_receipt, dict):
            surface_control_receipt = {}
        surface_control_receipt = normalize_live_mind_surface_control_receipt(
            surface_control_receipt,
            controls_bound=live_mind_controls_bound,
            generation_controls=live_mind_generation_controls,
            source="cognitive_engine_direct_quick_reply_controls",
        )

        return Thought(
            id=str(uuid.uuid4()),
            content=text,
            mode=mode,
            confidence=0.72,
            reasoning=[
                "Desktop quick reply used the governed primary router through CognitiveEngine.",
                (
                    "The compact path disabled deep handoff and prompt-cache reuse; "
                    "live mind context was embedded without duplicating the heavyweight runtime payload."
                    if live_runtime_required
                    else "The compact path disabled deep handoff, runtime payload, and prompt-cache reuse."
                ),
            ],
            metadata={
                "spiking_active_inference": advice
                if isinstance(advice, dict)
                else None,
                "imagination_workspace": imagination_frame
                if isinstance(imagination_frame, dict)
                else None,
                "imagination_workspace_feedback": imagination_feedback,
                "bicameral_advisory": bicameral_frame
                if isinstance(bicameral_frame, dict)
                else None,
                "bicameral_advisory_feedback": bicameral_feedback,
                "cognitive_situation_frame": cognitive_situation_frame
                if isinstance(cognitive_situation_frame, dict)
                else None,
                "live_mind_controls_bound": live_mind_controls_bound,
                "live_mind_generation_controls": dict(live_mind_generation_controls),
                "live_mind_snapshot_ready": live_mind_snapshot_ready,
                "live_mind_required_subsystems_ok": live_mind_required_subsystems_ok,
                "live_mind_context_required": live_mind_required,
                "live_mind_surface_control_receipt": surface_control_receipt,
                "live_mind_controls_worker_applied": bool(
                    surface_control_receipt.get("live_mind_controls_bound")
                    and surface_control_receipt.get("applied")
                ),
                "reply_generation_incomplete": reply_generation_incomplete,
                "reply_generation_stop_reason": generation_stop_reason,
                "reply_generation_failure_reasons": surface_reasons,
                "reply_original_chars": len(text),
                "self_condition_contract": self_condition_contract,
                "self_condition_evidence_id": str(
                    (
                        context.get("canonical_self_condition_projection")
                        if isinstance(
                            context.get("canonical_self_condition_projection"),
                            dict,
                        )
                        else {}
                    ).get("evidence_id")
                    or ""
                ),
                "response_path": (
                    "cognitive_engine_self_condition"
                    if self_condition_contract
                    else ""
                ),
            },
        )

