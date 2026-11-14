"""A reply that nearly worked, and what is done about it before a person sees it.

Stabilising is not rewriting. Each repair here has a named fault it answers —
a context leak, a dangling marker, a reply that drifted off the question, one
that ends mid-sentence, one that is the same answer to a different prompt — and
where none of them applies the draft goes out as written. A repair that cannot
name what it fixed is a repair that will eventually destroy a good answer, which
is what all of these were written after.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi.responses import JSONResponse

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_inference_gate
from interface.routes import chat_conversation_repair as _chat_conversation_repair  # noqa: E402
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_memory_state as _chat_memory_state  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes.chat_common import (  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402  # noqa: E402
    _CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S,  # noqa: F401
    _CHAT_DELIVERY_IDEMPOTENCY_KEY,  # noqa: F401
    _CHAT_PENDING_DELIVERY_CLAIM,  # noqa: F401
    _CHAT_RECOVERABLE_ERRORS,  # noqa: F401
    _CHAT_REQUEST_PRINCIPAL,  # noqa: F401
    _CHAT_REQUEST_SURFACE,  # noqa: F401
    _CHAT_SESSION_ID_MAX_CHARS,  # noqa: F401
    _EXPLICIT_NON_EXECUTION_RE,  # noqa: F401
    _INCOMPLETE_TAIL_WORDS,  # noqa: F401
    _INTERNAL_STATE_PATTERNS,  # noqa: F401
    _INTERNAL_SURFACE_CONTEXT,  # noqa: F401
    _LOCAL_CHOICE_REFERENCE_RE,  # noqa: F401
    _MAX_CONVERSATION_LOG_EXCHANGES,  # noqa: F401
    _MAX_USER_SURFACE_CONTINUATIONS,  # noqa: F401
    _ORGAN_ABSENCE_STREAKS,  # noqa: F401
    _ORGAN_INERT_STREAKS,  # noqa: F401
    _PROMPT_ARTIFACT_PATTERNS,  # noqa: F401
    _SEARCH_SKILL_NAMES,  # noqa: F401
    _TOPIC_STOPWORDS,  # noqa: F401
    _UNSET,  # noqa: F401
    MAX_CHAT_MESSAGE_BYTES,  # noqa: F401
    _conversation_log,  # noqa: F401
    _locks,  # noqa: F401
    logger,  # noqa: F401
)
from interface.routes.chat_quality import (  # noqa: E402
    _reply_assessment_requires_repair,
)
from interface.routes.chat_self_reply import (  # noqa: E402,F401
    _build_architecture_self_reflex,
    _build_self_condition_evidence,
    _build_self_diagnostic_reply,
    _build_subjective_self_reflex,
    _canonical_memory_state_grounding_reply,
    _classify_self_condition_contract,
    _fallback_ladder_identity,
    _humanize_recent_self_process_concern,
    _humanize_self_process_dimensions,
    _is_identity_challenge_request,
    _is_self_claim_boundary_question,
    _same_live_self_reflection_prompt_class,
)
from interface.routes.chat_turn_evidence import (  # noqa: E402,F401
    _benchmark_prompt_requests_fenced_artifact,
    _build_explicit_local_file_artifact,
    _canonical_memory_state_evidence_from_tuple,
    _canonical_memory_state_evidence_missing_from_reply,
    _collect_named_url_evidence,
    _collect_recent_traceability_event_sync,
    _context_challenge_repair_has_evidence,
    _correct_unevidenced_action_claims,
    _emit_chat_output_receipt,
    _evidence_came_from_the_network,
    _extract_canonical_memory_state_evidence_block,
    _prime_requested_output_contract_trace,
    _recent_action_receipts,
    _record_desktop_evidence_on_the_trace,
    _resolve_prior_answer_provenance,
    _save_requested_artifact,
    _serve_built_artifact,
    _worker_receipt_transaction_id,
)

from .chat_lane_bookkeeping import (  # noqa: E402
    _apply_aura_voice_shaping_compat,
    _bound_stabilizer_generation_budget,
    _build_stateful_voice_reflex,
    _flag_unstable_choice_commitment,
    _gather_recent_user_messages_for_relevance,
    _has_first_person_anchor,
    _has_live_aura_grounding,
    _is_simple_affect_check_request,
    _mark_conversation_lane_state,
)
from .chat_reply_assessment import (
    _measure_reply_quality_candidate,  # noqa: F401 - re-exported for interface/routes/chat.py
    _original_reply_is_safe_to_surface,  # noqa: F401 - re-exported for chat.py, chat_own_source, chat_foreground_lane
    _ends_where_it_meant_to,  # noqa: F401
    _evaluate_reply_topicality,
    _is_actionably_stale_response,
    _is_same_answer_different_prompt,
    _looks_semantically_glitched,
    _strip_unexpected_cjk_artifacts,
    _strip_user_visible_context_leaks,
)
from .chat_reply_shaping import (  # noqa: E402
    _append_sensory_claim_correction,
    _append_turn_text_mutation,
    _build_context_challenge_repair_reply,
    _build_evidence_bound_self_claim_reply,
    _build_grounded_self_condition_reply,
    _complete_repairable_truncated_reply,
    _correct_unfulfilled_write_claims,
    _looks_generic_assistantish,
    _remove_self_denials_the_record_refutes,
)

#: What an extracted block returns when it fell through to the code after it.
_FALL_THROUGH = object()


def _repair_missing_followup_delta(user_message: str, reply_text: str) -> str:
    """Add a requested follow-up delta when the draft mostly repeated context.

    This is intentionally narrow and model-free. It handles same-topic turns
    where the model preserved continuity but forgot the user's requested
    incremental move, such as adding a limitation or caveat to the prior
    example. It does not invent task-specific facts or execute tools.
    """
    from .chat import (
        _FOLLOWUP_DELTA_MARKERS,
    )


    normalized_user = _chat_memory_state._normalize_user_message(user_message)
    normalized_reply = _chat_memory_state._normalize_user_message(reply_text)
    if not normalized_user or not normalized_reply:
        return str(reply_text or "").strip()
    if not names_any(normalized_user, ("add", "include", "give", "connect")):
        return str(reply_text or "").strip()

    additions: list[str] = []
    for request_marker, reply_markers, addition in _FOLLOWUP_DELTA_MARKERS:
        if request_marker not in normalized_user:
            continue
        if any(marker in normalized_reply for marker in reply_markers):
            continue
        additions.append(addition)

    if (
        "connect" in normalized_user
        and "example" in normalized_user
        and "example" not in normalized_reply
    ):
        additions.append(
            "Connected back to the example, the new point changes how that example "
            "should be interpreted rather than replacing the original setup."
        )

    if not additions:
        return str(reply_text or "").strip()

    base = str(reply_text or "").strip()
    separator = " " if base.endswith((".", "!", "?")) else ". "
    return f"{base}{separator}{' '.join(additions)}".strip()






def _stabilize_user_facing_reply_architecture_self_assessment(
    reply_text: Any,
    user_message: str,
) -> tuple[Any, Any]:
    architecture_self_assessment = _chat_preflight._is_architecture_self_assessment_request(
        user_message
    )
    text = _apply_aura_voice_shaping_compat(
        _strip_unexpected_cjk_artifacts(user_message, str(reply_text or "").strip() or ""),
        user_message,
    )
    stripped = _strip_user_visible_context_leaks(text)
    if text and not stripped:
        # An ellipsis is not an answer; it is the shape of one.
        #
        # The salvage inside the stripper has already tried and failed by the
        # time this is reached, so the choice here is between an empty reply
        # and something that LOOKS like one. `or "…"` chose the second, and it
        # defeats every `if not reply` guard downstream — the turn then looks
        # answered to everything that asks.
        #
        # LIVE 2026-09-07: "does the file X exist, and what is in it?" was
        # served as a bare "…". The cortex had produced 1,155 characters and
        # quality had scored them confidence=high, off_topic=False. The same
        # shape is recorded in this file for 2026-08-17 at a different call
        # site, where the fix was the salvage rather than the substitution.
        logger.warning(
            "Context-leak strip emptied a %d-char reply and no salvage held; "
            "returning empty so the recovery paths run instead of serving an "
            "ellipsis.",
            len(text),
        )
        record_degradation(
            "chat.context_leak_strip",
            RuntimeError("stripping context leaks emptied a shaped reply"),
            severity="warning",
            action="returned empty rather than an ellipsis",
        )
    text = stripped
    repair_override = _chat_conversation_repair._maybe_build_conversation_repair_override(
        user_message, text
    )
    if repair_override:
        text = _apply_aura_voice_shaping_compat(
            _strip_unexpected_cjk_artifacts(user_message, repair_override),
            user_message,
        )
    return architecture_self_assessment, text

async def _stabilize_user_facing_reply_part_3(
    asked: dict[str, Any],
    desktop_cognitive_engine_required: bool,
    inference_gate: Any,
    memory_block: Any,
    protected_foreground_lane: bool,
    stabilizer_max_tokens: Any,
    user_message: str,
) -> tuple[Any, Any]:
    """Her turn asked again, exactly as her model was first asked it (core/brain/her_turn_as_asked.py): nothing is
    added to it, so what comes back is hers, and the checks that refused the first draft judge this one."""
    if memory_block:
        logger.warning(
            "Skipping stabilizer rewrite under memory pressure: %s",
            memory_block,
        )
        raise RuntimeError(f"stabilizer_rewrite_memory_pressure:{memory_block}")
    from core.brain.llm_health_router_waiting import _await_while_it_is_working

    # Use the same completion owner as the original generation.
    # An estimate is not permission to cancel an active repair.
    strict_desktop_repair = bool(desktop_cognitive_engine_required)
    stabilizer_timeout = 28.0 if strict_desktop_repair else 12.0
    corrected = await _await_while_it_is_working(
        inference_gate.think(
            asked["prompt"],
            system_prompt=asked["system_prompt"] or None,
            messages=asked["messages"],
            prefer_tier="primary",
            origin="api_stabilizer",
            # The person's question, not the draft under repair. Without it
            # the gate took the correction prompt as the question and read
            # its observables off the DRAFT: LIVE 2026-09-16 a draft that
            # named phi_core.py had the file looked up by its bare name,
            # "No file exists at phi_core.py" went into the repair turn, and
            # the repaired reply told the person the file was gone.
            visible_user_message=user_message,
            foreground_request=True,
            is_background=False,
            protected_foreground_lane=bool(
                protected_foreground_lane or strict_desktop_repair
            ),
            cognitive_engine_required=strict_desktop_repair,
            desktop_cognitive_engine_required=strict_desktop_repair,
            deep_handoff=False,
            allow_deep_handoff=False,
            allow_cloud_fallback=False,
            allow_tools=False,
            skip_runtime_payload=True,
            max_tokens=stabilizer_max_tokens,
        ),
        budget_s=stabilizer_timeout,
        user_facing=True,
        person_is_waiting=strict_desktop_repair,
    )
    corrected_text = _apply_aura_voice_shaping_compat(
        str(corrected or "").strip(), user_message
    )
    return corrected_text, stabilizer_timeout

def _through_the_record(user_message: str, reply_text: Any) -> Any:
    """Her reply checked against what the runtime holds: her senses, her machinery, her numbers, her choices, her
    writes, her actions, her receipts. A sentence the record refutes is taken out; what the record says is kept for her
    turn asked again (core/brain/her_turn_as_asked.py), and nothing is written into her reply in her voice."""
    from .chat import _append_past_action_record, _correct_unsourced_self_metrics

    reply_text = _append_sensory_claim_correction(user_message, reply_text)
    reply_text = _remove_self_denials_the_record_refutes(reply_text)
    reply_text = _correct_unsourced_self_metrics(reply_text)
    reply_text = _flag_unstable_choice_commitment(user_message, reply_text)
    reply_text = _correct_unfulfilled_write_claims(reply_text, user_message)
    # After the path check, so a reply that names a missing file is caught by that instrument first; this one catches
    # the completion claims that name no path and reach no lane with receipts.
    reply_text = _correct_unevidenced_action_claims(reply_text, user_message)
    return _append_past_action_record(user_message, reply_text)


async def _asked_again_with_the_record(user_message: str, checked: Any, *, desktop_cognitive_engine_required: bool,
                                       protected_foreground_lane: bool) -> Any:
    """Her turn asked again, once, with what the record said among its readings; her new draft goes through the same
    checks. The new draft when the record finds nothing in it; otherwise the first, with what it refuted taken out.

    LIVE 12 November 2026, Bryan, of "I cut a sentence there — I had written ..., and nothing I sensed this turn showed
    that. The rest stands.": a leak, and he was seeing more like it. Every check here used to append its own paragraph
    in her first person.
    """
    from core.brain.her_turn_as_asked import as_asked, clear_the_record, what_the_record_says

    asked = as_asked()
    gate = resolve_inference_gate()
    if asked is None or gate is None or not hasattr(gate, "think"):
        return checked
    max_tokens, memory_block = _bound_stabilizer_generation_budget(int(asked.get("max_tokens") or 1024))
    record = what_the_record_says()
    try:
        again, _timeout = await _stabilize_user_facing_reply_part_3(
            asked, desktop_cognitive_engine_required, gate, memory_block, protected_foreground_lane, max_tokens,
            user_message)
    except (*_CHAT_RECOVERABLE_ERRORS, TimeoutError) as exc:
        logger.info("Her turn was not asked again with the record (%s); the checked draft stands.", exc)
        return checked
    clear_the_record()
    second = _through_the_record(user_message, again) if str(again or "").strip() else ""
    if str(second or "").strip() and not what_the_record_says():
        logger.info("Her turn asked again with the record (%d fact(s)); her new draft passed it.", len(record))
        return second
    logger.info("Her turn asked again with the record; the record still refuted her new draft, so the first stands "
                "with what it refuted taken out.")
    return checked if str(checked or "").strip() else second


def _assessment_or_none(user_message: str, reply: str, recent_user_messages: Any) -> Any:
    """The reliability verdict on a reply, or None where the assessor refused it."""
    try:
        from core.conversation.response_reliability import assess_user_facing_reply

        return assess_user_facing_reply(
            user_message,
            reply,
            recent_user_messages=recent_user_messages,
        )
    # not a failure: a reply this assessor refuses has no assessment, and the caller
    # treats None as no verdict.
    except _CHAT_RECOVERABLE_ERRORS:
        return None


def _what_to_serve_of_the_cleaned_reply(
    user_message: str,
    cleaned: str,
    *,
    valid_cleaned: bool,
    recent_user_messages: Any,
    needs_self_expression: bool,
    requires_first_person_anchor: bool,
) -> str:
    """The sanitized reply if it passes every check, what she finished of it
    if only its last sentence is cut off, or "" when neither holds."""
    from .chat import _has_unexpected_cjk, _is_objective_parrot_reply

    cleaned_generic, _cleaned_reason = _looks_generic_assistantish(user_message, cleaned)
    cleaned_objective_parrot = _is_objective_parrot_reply(user_message, cleaned)
    cleaned_lacks_self_anchor = (
        needs_self_expression or requires_first_person_anchor
    ) and not _has_first_person_anchor(cleaned)
    cleaned_lacks_live_grounding = needs_self_expression and not _has_live_aura_grounding(
        cleaned
    )
    cleaned_unexpected_cjk = _has_unexpected_cjk(user_message, cleaned)
    cleaned_off_topic, _cleaned_off_topic_reason = _evaluate_reply_topicality(
        user_message,
        cleaned,
        recent_user_messages=recent_user_messages,
    )
    cleaned_stale_repeat = _is_actionably_stale_response(user_message, cleaned)
    cleaned_same_diff = _is_same_answer_different_prompt(user_message, cleaned)
    cleaned_truncated_tail = _chat_desktop_repair._looks_truncated_tail(cleaned)
    cleaned_semantic_glitch, _cleaned_semantic_reason = _looks_semantically_glitched(
        user_message, cleaned
    )
    cleaned_assessment = _assessment_or_none(
        user_message, cleaned, recent_user_messages
    )
    if (
        valid_cleaned
        and not cleaned_generic
        and not cleaned_objective_parrot
        and not cleaned_lacks_self_anchor
        and not cleaned_lacks_live_grounding
        and not cleaned_unexpected_cjk
        and not cleaned_off_topic
        and not cleaned_stale_repeat
        and not cleaned_same_diff
        and not cleaned_truncated_tail
        and not cleaned_semantic_glitch
        and not _reply_assessment_requires_repair(cleaned_assessment)
        and len(cleaned) >= 16
    ):
        return cleaned
    if cleaned_truncated_tail and valid_cleaned and not any((
        cleaned_generic, cleaned_objective_parrot, cleaned_lacks_self_anchor,
        cleaned_lacks_live_grounding, cleaned_unexpected_cjk, cleaned_off_topic,
        cleaned_stale_repeat, cleaned_same_diff,
        cleaned_semantic_glitch and _cleaned_semantic_reason != "truncated_tail",
    )):
        # Only the last sentence is unfinished: serve what she finished,
        # held to the same checks, rather than a regenerated reply.
        finished = _chat_desktop_repair._up_to_its_last_finished_sentence(cleaned)
        if (
            len(finished) >= 16
            and not _chat_desktop_repair._looks_truncated_tail(finished)
            and not _looks_semantically_glitched(user_message, finished)[0]
            and not _reply_assessment_requires_repair(
                _assessment_or_none(user_message, finished, recent_user_messages)
            )
        ):
            logger.info(
                "Served her answer up to its last finished sentence (%d of %d chars).",
                len(finished), len(cleaned),
            )
            return finished
    return ""


async def _stabilize_user_facing_reply(
    user_message: str,
    reply_text: Any,
    *,
    desktop_cognitive_engine_required: bool = False,
    protected_foreground_lane: bool = False,
) -> str:
    """Her reply as she wrote it, less any claim her record refutes.

    Bryan, 13 November 2026: "gates and stabilizers have a tendancy to throw away fine drafts. no canned replies
    generally. certainly shouldnt be throwing away legit responses. doing the job theyre there for doesnt mean again
    defaulting to heuristics or replacing with canned speech ... and ideally they dont slow down the process either."
    This was six hundred lines that judged a draft by its wording (generic, off topic, parroting, glitched, lacking a
    self anchor) and served something else in its place: a closed "cortex evidence" reply, a "bounded greeting", a
    canonical self-condition reply, a "subjective self-reflex", a deterministic ending, a fixed sentence when the
    person's message contained "language model", or a second generation. LIVE that afternoon her answer to "What have
    you been thinking about while I was testing your voice?" went through three generations here and took 93 s.

    What is left does the job by the record: a claim a reading refutes (a sense with no reading, a file not on disk, an
    action with no receipt, a number no channel produced) is taken out, and the record's fact reaches her at the head of
    her next turn in this conversation, where what she makes of it is hers (core/brain/her_turn_as_asked.py). A draft
    that is not speech at all (empty, the runtime's own text, corrupted language;
    core/conversation/surface_disposition.UNSPEAKABLE_REASONS) is asked again once, exactly as it was asked.
    """
    from core.brain.her_turn_as_asked import carry_the_record_to_the_next_turn

    checked = _what_she_finished(str(_through_the_record(user_message, reply_text) or "").strip())
    carry_the_record_to_the_next_turn()
    broken = _what_is_not_speech(user_message, checked)
    if not broken:
        return checked
    logger.info("Her draft is not speech (%s); her turn is asked again as it was asked.", ", ".join(broken))
    again = await _asked_again_as_asked(user_message, desktop_cognitive_engine_required=desktop_cognitive_engine_required,
                                        protected_foreground_lane=protected_foreground_lane)
    second = _what_she_finished(str(_through_the_record(user_message, again) or "").strip()) if again else ""
    carry_the_record_to_the_next_turn()
    if second and not _what_is_not_speech(user_message, second):
        return second
    return checked


def _what_she_finished(text: str) -> str:
    """A reply cut off mid-sentence (its token allowance ran out) up to its last finished sentence: her words, ending
    where she ended one. LIVE 2026-10-03 a 3,862-character answer that ran out mid-sentence was replaced whole by an
    87-character repair (tests/test_a_cut_off_answer_keeps_what_she_finished.py)."""
    if not _chat_desktop_repair._looks_truncated_tail(text):
        return text
    return _chat_desktop_repair._up_to_its_last_finished_sentence(text) or text


def _what_is_not_speech(user_message: str, text: str) -> tuple[str, ...]:
    """The reasons this text is not her speech at all, or none."""
    if not text:
        return ("empty_reply",)
    from core.conversation.surface_disposition import integrity_failures

    assessment = _assessment_or_none(user_message, text, ())
    return integrity_failures(getattr(assessment, "reasons", ()) or ())


async def _asked_again_as_asked(user_message: str, *, desktop_cognitive_engine_required: bool,
                                protected_foreground_lane: bool) -> str:
    """Her turn asked again exactly as her model was asked it, or "" when it was not kept or could not be asked."""
    from core.brain.her_turn_as_asked import as_asked

    asked = as_asked()
    gate = resolve_inference_gate()
    if asked is None or gate is None or not hasattr(gate, "think"):
        return ""
    max_tokens, memory_block = _bound_stabilizer_generation_budget(int(asked.get("max_tokens") or 1024))
    try:
        again, _timeout = await _stabilize_user_facing_reply_part_3(
            asked, desktop_cognitive_engine_required, gate, memory_block, protected_foreground_lane, max_tokens,
            user_message)
    except (*_CHAT_RECOVERABLE_ERRORS, TimeoutError) as exc:
        logger.info("Her turn was not asked again (%s); her draft stands.", exc)
        return ""
    return str(again or "").strip()


async def _repair_final_degraded_reply_assessment_reasons(
    assess_user_facing_reply: Any,
    assessment: Any,
    desktop_cognitive_engine_required: bool,
    recent_user_messages: Any,
    reply_text: str,
    session_id: str,
    user_message: str,
) -> Any:
    from .chat import _build_grounded_self_process_repair_reply
    assessment_reasons = set(getattr(assessment, "reasons", ()) or ())
    if _is_simple_affect_check_request(user_message):
        self_condition_repair = _build_grounded_self_condition_reply(user_message)
        if self_condition_repair:
            condition_assessment = (
                (await asyncio.to_thread(
                    assess_user_facing_reply,
                    user_message,
                    self_condition_repair,
                    recent_user_messages=recent_user_messages,
                ))
                if assess_user_facing_reply
                else None
            )
            if not _reply_assessment_requires_repair(condition_assessment):
                logger.warning(
                    "Final reply quality gate rebound condition question to canonical self-state."
                )
                return self_condition_repair, False, False, False, "", True
    if not desktop_cognitive_engine_required:
        conversation_recall_reply = await _chat_memory_state._build_conversation_recall_reply(
            user_message,
            session_id=session_id,
        )
        if conversation_recall_reply:
            return conversation_recall_reply, False, False, False, "", True

        context_challenge_repair = await _build_context_challenge_repair_reply(
            user_message,
            session_id=session_id,
        )
        if context_challenge_repair:
            context_challenge_repair = _apply_aura_voice_shaping_compat(
                context_challenge_repair,
                user_message,
            )
            context_assessment = (
                (await asyncio.to_thread(
                    assess_user_facing_reply,
                    user_message,
                    context_challenge_repair,
                    recent_user_messages=recent_user_messages,
                ))
                if assess_user_facing_reply
                else None
            )
            if not _reply_assessment_requires_repair(context_assessment):
                return context_challenge_repair, False, False, False, "", True

        if assessment_reasons & {"raw_model_identity_leak", "missing_self_claim_evidence_boundary"}:
            self_claim_repair = _build_evidence_bound_self_claim_reply(
                user_message,
                lane=_chat_preflight._collect_conversation_lane_status(),
            )
            if self_claim_repair:
                self_claim_assessment = (
                    (await asyncio.to_thread(
                        assess_user_facing_reply,
                        user_message,
                        self_claim_repair,
                        recent_user_messages=recent_user_messages,
                    ))
                    if assess_user_facing_reply
                    else None
                )
                if not _reply_assessment_requires_repair(self_claim_assessment):
                    return self_claim_repair, False, False, False, "", True

    if not desktop_cognitive_engine_required and assessment_reasons & {
        "missing_requested_self_process_coverage",
        "off_topic_self_reflection_reply",
        "status_page_self_reflection",
        "pseudo_internal_jargon",
    }:
        self_process_repair = await _build_grounded_self_process_repair_reply(
            user_message,
            reply_text,
            lane=_chat_preflight._collect_conversation_lane_status(),
            session_id=session_id,
        )
        if self_process_repair:
            self_process_repair = _apply_aura_voice_shaping_compat(
                self_process_repair,
                user_message,
            )
            self_process_assessment = (
                (await asyncio.to_thread(
                    assess_user_facing_reply,
                    user_message,
                    self_process_repair,
                    recent_user_messages=recent_user_messages,
                ))
                if assess_user_facing_reply
                else None
            )
            if not _reply_assessment_requires_repair(self_process_assessment):
                logger.warning(
                    "🛡️ Final reply quality gate repaired self-process coverage from live context."
                )
                return self_process_repair, False, False, False, "", True

    if (
        _chat_desktop_repair._is_low_risk_social_continuity_request(user_message)
        and not desktop_cognitive_engine_required
    ):
        social_repair = _chat_desktop_repair._build_social_continuity_repair_reply(user_message)
        return social_repair, False, False, False, "", True
    return _FALL_THROUGH

def _repair_final_degraded_reply_part_2(
    assess_user_facing_reply: Any,
    assessment: Any,
    off_topic: bool,
    recent_user_messages: Any,
    repair_instruction_shape: Any,
    reply_text: str,
    same_diff: bool,
    stale: bool,
    user_message: str,
) -> Any:
    logger.warning(
        "🛡️ Final reply quality gate repairing degraded output "
        "(stale=%s same_diff=%s off_topic=%s assessment=%s).",
        stale,
        same_diff,
        off_topic,
        ",".join(getattr(assessment, "reasons", ()) or ()) if assessment else "",
    )

    completed_tail = _complete_repairable_truncated_reply(user_message, reply_text)
    if completed_tail:
        completed_tail = _apply_aura_voice_shaping_compat(completed_tail, user_message)
        completed_stale = _is_actionably_stale_response(user_message, completed_tail)
        completed_same_diff = _is_same_answer_different_prompt(user_message, completed_tail)
        completed_off_topic, completed_off_topic_reason = _evaluate_reply_topicality(
            user_message,
            completed_tail,
            recent_user_messages=recent_user_messages,
        )
        completed_assessment = (
            assess_user_facing_reply(
                user_message,
                completed_tail,
                recent_user_messages=recent_user_messages,
            )
            if assess_user_facing_reply
            else None
        )
        if not (
            completed_stale
            or completed_same_diff
            or completed_off_topic
            or _reply_assessment_requires_repair(completed_assessment)
        ):
            logger.warning(
                "🛡️ Final reply quality gate completed clipped Cortex draft without a second model call."
            )
            return (
                completed_tail,
                completed_stale,
                completed_same_diff,
                completed_off_topic,
                completed_off_topic_reason,
                True,
            )

    if repair_instruction_shape is not None and assessment is not None:
        shaped = repair_instruction_shape(user_message, reply_text)
        if shaped and shaped != str(reply_text or "").strip():
            shaped_stale = _is_actionably_stale_response(user_message, shaped)
            shaped_same_diff = _is_same_answer_different_prompt(user_message, shaped)
            shaped_off_topic, shaped_off_topic_reason = _evaluate_reply_topicality(
                user_message,
                shaped,
                recent_user_messages=recent_user_messages,
            )
            shaped_assessment = assess_user_facing_reply(
                user_message,
                shaped,
                recent_user_messages=recent_user_messages,
            )
            if not (
                shaped_stale
                or shaped_same_diff
                or shaped_off_topic
                or _reply_assessment_requires_repair(shaped_assessment)
            ):
                logger.warning(
                    "🛡️ Final reply quality gate repaired explicit response shape "
                    "deterministically (%s -> clean, len=%d).",
                    ",".join(getattr(assessment, "reasons", ()) or ()) or "unknown",
                    len(shaped),
                )
                return (
                    shaped,
                    shaped_stale,
                    shaped_same_diff,
                    shaped_off_topic,
                    shaped_off_topic_reason,
                    True,
                )

    if same_diff and not stale and not off_topic:
        delta_repaired = _repair_missing_followup_delta(user_message, reply_text)
        if delta_repaired and delta_repaired != str(reply_text or "").strip():
            delta_repaired = _apply_aura_voice_shaping_compat(delta_repaired, user_message)
            delta_stale = _is_actionably_stale_response(user_message, delta_repaired)
            delta_same_diff = _is_same_answer_different_prompt(user_message, delta_repaired)
            delta_off_topic, delta_off_topic_reason = _evaluate_reply_topicality(
                user_message,
                delta_repaired,
                recent_user_messages=recent_user_messages,
            )
            delta_assessment = (
                assess_user_facing_reply(
                    user_message,
                    delta_repaired,
                    recent_user_messages=recent_user_messages,
                )
                if assess_user_facing_reply
                else None
            )
            if not (
                delta_stale
                or delta_same_diff
                or delta_off_topic
                or _reply_assessment_requires_repair(delta_assessment)
            ):
                logger.warning(
                    "🛡️ Final reply quality gate repaired missing follow-up delta without a second model call."
                )
                return (
                    delta_repaired,
                    delta_stale,
                    delta_same_diff,
                    delta_off_topic,
                    delta_off_topic_reason,
                    True,
                )
    return _FALL_THROUGH

def _repair_final_degraded_reply_repaired_stale(
    assess_user_facing_reply: Any,
    desktop_cognitive_engine_required: bool,
    recent_user_messages: Any,
    reliability_floor_for_user: Any,
    repaired: Any,
    reply_text: str,
    user_message: str,
) -> Any:
    repaired_stale = _is_actionably_stale_response(user_message, repaired)
    repaired_same_diff = _is_same_answer_different_prompt(user_message, repaired)
    repaired_off_topic, repaired_off_topic_reason = _evaluate_reply_topicality(
        user_message,
        repaired,
        recent_user_messages=recent_user_messages,
    )
    repaired_assessment = (
        assess_user_facing_reply(
            user_message,
            repaired,
            recent_user_messages=recent_user_messages,
        )
        if assess_user_facing_reply
        else None
    )
    if not (
        repaired_stale
        or repaired_same_diff
        or repaired_off_topic
        or _reply_assessment_requires_repair(repaired_assessment)
    ):
        return (
            repaired,
            repaired_stale,
            repaired_same_diff,
            repaired_off_topic,
            repaired_off_topic_reason,
            True,
        )

    # Similar-answer detection is useful telemetry, but it is not strong enough
    # by itself to justify discarding an otherwise topical, coherent live reply.
    # The old path could turn a healthy present-turn answer into a canned repair
    # reflex purely because it resembled a prior response shape.
    repaired_same_diff_only = bool(
        repaired_same_diff
        and not repaired_stale
        and not repaired_off_topic
        and not _reply_assessment_requires_repair(repaired_assessment)
    )
    if repaired_same_diff_only:
        return (
            repaired,
            repaired_stale,
            repaired_same_diff,
            repaired_off_topic,
            repaired_off_topic_reason,
            True,
        )

    floor = reliability_floor_for_user(user_message) if reliability_floor_for_user else ""
    if floor:
        floor_stale = _is_actionably_stale_response(user_message, floor)
        floor_same_diff = _is_same_answer_different_prompt(user_message, floor)
        floor_off_topic, floor_off_topic_reason = _evaluate_reply_topicality(
            user_message,
            floor,
            recent_user_messages=recent_user_messages,
        )
        floor_assessment = (
            assess_user_facing_reply(
                user_message,
                floor,
                recent_user_messages=recent_user_messages,
            )
            if assess_user_facing_reply
            else None
        )
        if not (
            floor_stale
            or floor_same_diff
            or floor_off_topic
            or _reply_assessment_requires_repair(floor_assessment)
        ):
            return (
                floor,
                floor_stale,
                floor_same_diff,
                floor_off_topic,
                floor_off_topic_reason,
                True,
            )

    if desktop_cognitive_engine_required:
        logger.warning(
            "🛡️ Final reply quality gate refused freeform reflex fallback for desktop-required CognitiveEngine turn."
        )
        return (
            repaired,
            repaired_stale,
            repaired_same_diff,
            True,
            repaired_off_topic_reason or "desktop_cognitive_engine_repair_failed",
            bool(repaired != str(reply_text or "").strip()),
        )

    frame = _chat_desktop_repair._build_aura_expression_frame(user_message)
    reflex = _build_stateful_voice_reflex(frame, user_message)
    reflex_stale = _is_actionably_stale_response(user_message, reflex)
    reflex_same_diff = _is_same_answer_different_prompt(user_message, reflex)
    reflex_off_topic, reflex_off_topic_reason = _evaluate_reply_topicality(
        user_message,
        reflex,
        recent_user_messages=recent_user_messages,
    )
    reflex_semantic, reflex_semantic_reason = _looks_semantically_glitched(user_message, reflex)
    reflex_assessment = (
        assess_user_facing_reply(
            user_message,
            reflex,
            recent_user_messages=recent_user_messages,
        )
        if assess_user_facing_reply
        else None
    )
    if not (
        reflex_stale
        or reflex_same_diff
        or reflex_off_topic
        or reflex_semantic
        or _reply_assessment_requires_repair(reflex_assessment)
    ):
        return (
            reflex,
            reflex_stale,
            reflex_same_diff,
            reflex_off_topic,
            reflex_off_topic_reason,
            True,
        )

    honest_failure = _chat_conversation_repair._build_degraded_live_reply(
        _chat_desktop_repair._build_aura_expression_frame(user_message),
        user_message,
        reason=reflex_off_topic_reason or reflex_semantic_reason or "unrepaired_degraded_turn",
    )
    return (
        honest_failure,
        False,
        False,
        True,
        reflex_off_topic_reason or reflex_semantic_reason or "unrepaired_degraded_turn",
        True,
    )
    return _FALL_THROUGH

async def _repair_final_degraded_reply(
    user_message: str,
    reply_text: str,
    *,
    stale: bool,
    same_diff: bool,
    off_topic: bool,
    off_topic_reason: str = "",
    desktop_cognitive_engine_required: bool = False,
    protected_foreground_lane: bool = False,
    session_id: str = "",
) -> tuple[str, bool, bool, bool, str, bool]:
    """Final user-facing gate: degraded text must be repaired or replaced."""
    from .chat import _repair_owner_name_drift_reply, _reply_has_owner_name_drift

    try:
        from core.conversation.response_reliability import (
            assess_user_facing_reply,
            reliability_floor_for_user,
            repair_instruction_shape,
        )
    except _CHAT_RECOVERABLE_ERRORS:
        assess_user_facing_reply = None
        reliability_floor_for_user = None
        repair_instruction_shape = None

    recent_user_messages = await _gather_recent_user_messages_for_relevance(user_message)
    assessment = (
        (await asyncio.to_thread(
            assess_user_facing_reply,
            user_message,
            reply_text,
            recent_user_messages=recent_user_messages,
        ))
        if assess_user_facing_reply
        else None
    )
    needs_repair = bool(
        stale or same_diff or off_topic or _reply_assessment_requires_repair(assessment)
    )
    if _reply_has_owner_name_drift(user_message, reply_text):
        return (
            _repair_owner_name_drift_reply(reply_text),
            False,
            False,
            False,
            "",
            True,
        )
    owner_name_reply = _chat_memory_state._build_owner_name_recall_reply(user_message)
    if owner_name_reply and not desktop_cognitive_engine_required:
        normalized_reply = _chat_memory_state._normalize_user_message(reply_text)
        owner_name = _chat_memory_state._resolve_primary_operator_name()
        if (
            len(normalized_reply.split()) <= 4
            or owner_name.lower() not in normalized_reply
            or "verified" not in normalized_reply
        ):
            return owner_name_reply, False, False, False, "", True

    if not needs_repair:
        return reply_text, stale, same_diff, off_topic, off_topic_reason, False

    _left = await _repair_final_degraded_reply_assessment_reasons(assess_user_facing_reply, assessment, desktop_cognitive_engine_required, recent_user_messages, reply_text, session_id, user_message)
    if _left is not _FALL_THROUGH:
        return _left

    _left = _repair_final_degraded_reply_part_2(assess_user_facing_reply, assessment, off_topic, recent_user_messages, repair_instruction_shape, reply_text, same_diff, stale, user_message)
    if _left is not _FALL_THROUGH:
        return _left

    if desktop_cognitive_engine_required or protected_foreground_lane:
        repaired = await _stabilize_user_facing_reply(
            user_message,
            reply_text,
            desktop_cognitive_engine_required=desktop_cognitive_engine_required,
            protected_foreground_lane=protected_foreground_lane,
        )
    else:
        repaired = await _stabilize_user_facing_reply(user_message, reply_text)
    _left = _repair_final_degraded_reply_repaired_stale(assess_user_facing_reply, desktop_cognitive_engine_required, recent_user_messages, reliability_floor_for_user, repaired, reply_text, user_message)
    if _left is not _FALL_THROUGH:
        return _left


async def _repair_final_degraded_reply_with_provenance(
    trace: dict[str, Any],
    *,
    stage: str,
    user_message: str,
    reply_text: str,
    **kwargs: Any,
) -> tuple[str, bool, bool, bool, str, bool]:
    """Run the final gate and atomically record any visible text replacement."""

    result = await _repair_final_degraded_reply(
        user_message,
        reply_text,
        **kwargs,
    )
    repaired, _stale, _same_diff, _off_topic, reason, did_repair = result
    if did_repair and repaired != reply_text:
        _append_turn_text_mutation(
            trace,
            stage=stage,
            method="deterministic_final_reply_repair",
            reasons=[reason or "degraded_reply"],
            before=reply_text,
            after=repaired,
            deterministic=True,
            authorship_effect="replaced_by_runtime",
        )
    return result


async def _recheck_a_degraded_reply(
    *,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    desktop_memory_state_evidence: Any,
    desktop_requires_cognitive_engine: Any,
    hard_final_quality_failed: Any,
    lane: Any,
    reply_text: Any,
    response_confidence: Any,
) -> tuple[Any, Any, Any]:
    """Re-check a degraded reply against the memory evidence the turn gathered.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 7 name(s) from the turn and hands back
    3.
    """
    from .chat import (
        _collect_live_mind_context_payload,
        _memory_state_evidence_is_missing_from_reply,
        _set_conversation_degradation_streak,
    )

    if (
        desktop_requires_cognitive_engine
        and response_confidence == "degraded"
        and desktop_memory_state_evidence
        and bool(_live_turn_trace.get("engine_think_invoked"))
        and bool(_live_turn_trace.get("cognitive_engine_reply_accepted"))
        and not bool(_live_turn_trace.get("cognitive_engine_reply_failed"))
        and not bool(_live_turn_trace.get("bounded_contract_used"))
        and not bool(_live_turn_trace.get("legacy_fallback_used"))
    ):
        canonical_evidence = _canonical_memory_state_evidence_from_tuple(
            desktop_memory_state_evidence
        )
        rebound_live_mind_context = await _collect_live_mind_context_payload(
            user_message=_semantic_user_message,
            lane=lane,
            require_engine=True,
        )
        grounded_memory_reply = _canonical_memory_state_grounding_reply(
            _semantic_user_message,
            canonical_evidence,
            live_mind_context=rebound_live_mind_context,
        )
        if grounded_memory_reply and not _memory_state_evidence_is_missing_from_reply(
            _semantic_user_message,
            grounded_memory_reply,
            desktop_memory_state_evidence,
        ):
            # Replace, or add?
            #
            # This rebound exists so a "what did I pin?" turn cannot drift
            # off the canonical record. It fires whenever the reply does
            # not repeat the pinned phrase — which, on a turn that pins a
            # fact AND asks something real, is simply what a good answer
            # looks like. Live 2026-07-27: "Remember this: my project
            # codename is HELIOTROPE... Separately — do you think a system
            # like you can actually prefer one thing over another?" came
            # back as the pin confirmation alone, degraded with an EMPTY
            # assessment and an EMPTY reason. Nothing was wrong with the
            # answer. It just didn't recite the codename, so it was thrown
            # away and the template took the turn.
            #
            # A person told "remember X, and also what do you think about
            # Y" says both. So: when the turn is only the memory request,
            # the canonical reply stands in. When there is more to it, the
            # confirmation joins the answer instead of erasing it.
            _pre_memory_grounding_reply = reply_text
            if (
                _chat_memory_state._turn_has_substance_beyond_memory_request(
                    _semantic_user_message
                )
                and str(reply_text or "").strip()
            ):
                # The whole canonical reply, not just its first sentence:
                # it is short, it is grounded, and its later sentences
                # carry real content ("right now I am keeping attention
                # on this live desktop thread") that answers the state
                # half of turns like "remember X, and tell me what you
                # are attending to".
                confirmation = str(grounded_memory_reply or "").strip()
                logger.info(
                    "Memory/state evidence joined the CognitiveEngine answer "
                    "instead of replacing it: the turn asked for more than "
                    "the canonical record."
                )
                reply_text = f"{confirmation}\n\n{str(reply_text).strip()}"
            else:
                logger.warning(
                    "Final desktop quality gate rebound reply to canonical "
                    "memory/state evidence after CognitiveEngine invocation."
                )
                reply_text = grounded_memory_reply
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.final_memory_state_grounding",
                method="deterministic_canonical_grounding",
                reasons=["memory_state_evidence_grounding"],
                before=_pre_memory_grounding_reply,
                after=reply_text,
                deterministic=True,
                authorship_effect=(
                    "augmented_by_runtime"
                    if _chat_memory_state._turn_has_substance_beyond_memory_request(
                        _semantic_user_message
                    )
                    and str(_pre_memory_grounding_reply or "").strip()
                    else "replaced_by_runtime"
                ),
            )
            response_confidence = "high"
            hard_final_quality_failed = False
            _set_conversation_degradation_streak(0)
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "response_path": "cognitive_engine_memory_state_grounding",
                }
            )
    return hard_final_quality_failed, reply_text, response_confidence




async def _serve_the_identity_repair(
    *,
    _live_turn_contract: Any,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    identity_repair: Any,
    lane: Any,
    pending_exchange_id: Any,
) -> tuple[Any, Any, Any]:
    """Serve the identity repair when the engine failed on an identity question.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 6 name(s) and hands back
    2.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
    )

    async def _block() -> Any:
        nonlocal lane, pending_exchange_id
        if identity_repair:
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "response_path": "cognitive_engine_identity_continuity_grounding",
                    "canonical_grounding_used": True,
                }
            )
            lane = _mark_conversation_lane_state(
                "cognitive_engine_identity_continuity_grounding",
                state="recovering",
            )
            logger.warning(
                "Desktop CognitiveEngine produced no acceptable reply for an identity "
                "turn; serving canonical identity/continuity grounding instead of legacy fallback."
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    identity_repair,
                    record_experience=True,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                identity_repair,
                cause="chat_response",
                metadata={
                    "response_confidence": "bounded",
                    "path": "cognitive_engine_identity_continuity_grounding",
                    "status": "cognitive_engine_identity_continuity_grounding",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                },
            )
            return JSONResponse(
                {
                    "response": identity_repair,
                    "status": "cognitive_engine_identity_continuity_grounding",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "bounded",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="bounded",
                        status="cognitive_engine_identity_continuity_grounding",
                        reply_source="cognitive_engine_identity_continuity_grounding",
                    ),
                }
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, lane, pending_exchange_id


# Moved out whole when this module crossed the 2,000-line ceiling that a
# module not already in the baseline never gets grandfathered past. Neither
# took anything from here, so it was a move and not a rewrite — and both
# names stay importable from where every caller already imports them.
from interface.routes.chat_reply_repair_about_herself import (  # noqa: E402
    _repair_a_degraded_affect_reply,
    _repair_a_degraded_identity_reply,
    _serve_the_bounded_repair,
)

__all__ = [
    *(globals().get("__all__") or []),
    "_repair_a_degraded_affect_reply",
    "_repair_a_degraded_identity_reply",
    "_serve_the_bounded_repair",
]
