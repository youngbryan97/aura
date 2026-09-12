"""A reply that nearly worked, and what is done about it before a person sees it.

Stabilising is not rewriting. Each repair here has a named fault it answers —
a context leak, a dangling marker, a reply that drifted off the question, one
that ends mid-sentence, one that is the same answer to a different prompt — and
where none of them applies the draft goes out as written. A repair that cannot
name what it fixed is a repair that will eventually destroy a good answer, which
is what all of these were written after.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from core.runtime.structured_input import (
    analyze_prompt_shape,
)
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
    _call_stateful_voice_reflex,
    _flag_unstable_choice_commitment,
    _gather_recent_user_messages_for_relevance,
    _has_first_person_anchor,
    _has_live_aura_grounding,
    _is_simple_affect_check_request,
    _is_simple_subjective_reflex_request,
    _looks_safely_grounded_search_reply,
    _mark_conversation_lane_state,
    _normalize_response_body,
    _response_fingerprint,
)
from .chat_reply_shaping import (  # noqa: E402
    _append_sensory_claim_correction,
    _append_turn_text_mutation,
    _build_context_challenge_repair_reply,
    _build_evidence_bound_self_claim_reply,
    _build_grounded_self_condition_reply,
    _build_recent_user_context_block,
    _complete_repairable_truncated_reply,
    _correct_unfulfilled_write_claims,
    _looks_generic_assistantish,
    _remove_self_denials_the_record_refutes,
)
from .chat_reply_assessment import (
    _ends_where_it_meant_to,  # noqa: F401
    _evaluate_reply_topicality,
    _is_actionably_stale_response,
    _is_same_answer_different_prompt,
    _looks_semantically_glitched,
    _measure_reply_quality_candidate,
    _original_reply_is_safe_to_surface,
    _strip_unexpected_cjk_artifacts,
    _strip_user_visible_context_leaks,
)
















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
    if not any(marker in normalized_user for marker in ("add", "include", "give", "connect")):
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






async def _stabilize_user_facing_reply(
    user_message: str,
    reply_text: Any,
    *,
    desktop_cognitive_engine_required: bool = False,
    protected_foreground_lane: bool = False,
) -> str:
    # Closed claims about the resident cortex are rendered from signed runtime
    # authorities, not from a model narrative. This is deliberately the first
    # stabilizer operation: no later stylistic or repair path may turn measured
    # evidence back into an unsupported subjective account.
    from .chat import (
        _SEARCH_SNIPPET_PATTERNS,
        _append_past_action_record,
        _attempt_generated_social_grounding_repair,
        _build_grounded_traceability_reply,
        _correct_unsourced_self_metrics,
        _desktop_secondary_model_repair_allowed,
        _has_unexpected_cjk,
        _is_objective_parrot_reply,
        _record_recent_response,
    )

    try:
        from core.brain.cortex_self_evidence import render_cortex_evidence_reply

        cortex_evidence_reply = render_cortex_evidence_reply(user_message)
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat.cortex_self_evidence", exc)
        cortex_evidence_reply = ""
    if cortex_evidence_reply:
        logger.info("Served closed cortex self-evidence from verified assertions.")
        return cortex_evidence_reply

    # A claim about the room, checked against the senses that would know.
    #
    # LIVE, 2026-08-10, with ground truth: Bryan said he was going upstairs,
    # then asked "am I still here, or did I walk away?" She answered "You're
    # still here. The room is silent, the light remains unchanged on your desk
    # ... a disturbance in the air currents, or perhaps an echo of footsteps"
    # — wrong, and grounded in four senses she does not have. Carrying the
    # typed absence into the prompt was not enough: evidence informs, it does
    # not enforce.
    reply_text = _append_sensory_claim_correction(user_message, reply_text)
    # A claim about her own machinery, checked against the machinery. Same
    # shape as the sense check above and for the same reason: carrying the
    # record into the prompt is not enough.
    reply_text = _remove_self_denials_the_record_refutes(reply_text)
    reply_text = _correct_unsourced_self_metrics(reply_text)
    reply_text = _flag_unstable_choice_commitment(user_message, reply_text)
    reply_text = _correct_unfulfilled_write_claims(reply_text, user_message)
    # Runs after the path check so a reply that names a missing file is
    # corrected by the specific instrument first; this one catches the
    # completion claims that name no path and reach no lane with receipts.
    reply_text = _correct_unevidenced_action_claims(reply_text, user_message)
    reply_text = _append_past_action_record(user_message, reply_text)
    frame = _chat_desktop_repair._build_aura_expression_frame(user_message)
    contract = frame.get("contract")
    prompt_shape = analyze_prompt_shape(user_message)
    prefer_extended_answer = bool(
        getattr(contract, "prefer_extended_answer", False) or prompt_shape.prefers_extended_answer
    )
    requires_single_reply_coverage = bool(
        getattr(contract, "requires_single_reply_coverage", False)
        or prompt_shape.requires_single_reply_coverage
    )
    question_parts = int(getattr(contract, "question_parts", prompt_shape.question_parts or 1) or 1)
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
    grounded = _chat_conversation_repair._build_grounded_introspection_reply(user_message)
    grounded_traceability = await _build_grounded_traceability_reply(user_message)
    if grounded_traceability:
        return grounded_traceability
    if grounded and _chat_preflight._is_private_cognitive_model_request(user_message):
        return grounded
    quality = await _measure_reply_quality_candidate(user_message, text)
    recent_user_messages = list(quality.recent_user_messages)
    recent_user_context = _build_recent_user_context_block(recent_user_messages)
    generic, generic_reason = _looks_generic_assistantish(user_message, text)
    identity_collapse = bool(generic and generic_reason == "assistant_disclaimer")
    objective_parrot = _is_objective_parrot_reply(user_message, text)
    needs_self_expression = bool(frame.get("needs_self_expression"))
    requires_first_person_anchor = bool(frame.get("requires_explicit_live_grounding"))
    lacks_self_anchor = requires_first_person_anchor and not _has_first_person_anchor(text)
    lacks_live_grounding = needs_self_expression and not _has_live_aura_grounding(text)
    unexpected_cjk = _has_unexpected_cjk(user_message, text)
    internal_state_leak = bool(
        _INTERNAL_STATE_PATTERNS.search(text)
        or _PROMPT_ARTIFACT_PATTERNS.search(text)
        or _SEARCH_SNIPPET_PATTERNS.search(text)
    )
    off_topic = quality.is_off_topic
    off_topic_reason = quality.off_topic_reason
    stale_repeat = quality.is_stale
    same_diff = quality.is_same_diff
    truncated_tail = _chat_desktop_repair._looks_truncated_tail(text)
    semantic_glitch = quality.semantic_glitch
    semantic_glitch_reason = quality.semantic_glitch_reason
    live_reply_assessment = quality.reply_assessment
    assessment_retryable = _reply_assessment_requires_repair(live_reply_assessment)
    social_grounding_reasons = {
        "ungrounded_person_address",
        "ungrounded_person_narrative",
        "unsupported_deployment_routing_claim",
    }
    if set(getattr(live_reply_assessment, "reasons", ()) or ()) & social_grounding_reasons:
        try:
            from core.conversation.response_reliability import grounded_social_repair_reply

            social_repair = await _attempt_generated_social_grounding_repair(
                user_message,
                text,
                desktop_cognitive_engine_required=desktop_cognitive_engine_required,
                protected_foreground_lane=protected_foreground_lane,
            )
            if social_repair:
                social_quality = await _measure_reply_quality_candidate(
                    user_message,
                    social_repair,
                    recent_user_messages=recent_user_messages,
                )
                social_assessment = social_quality.reply_assessment
                if not _reply_assessment_requires_repair(social_assessment):
                    logger.info(
                        "Stabilizer repaired an ungrounded social/deployment draft "
                        "with bounded foreground generation."
                    )
                    return social_repair
            social_repair = grounded_social_repair_reply(user_message)
            if social_repair:
                social_quality = await _measure_reply_quality_candidate(
                    user_message,
                    social_repair,
                    recent_user_messages=recent_user_messages,
                )
                social_assessment = social_quality.reply_assessment
                if not _reply_assessment_requires_repair(social_assessment):
                    logger.info(
                        "Stabilizer replaced an ungrounded social/deployment draft "
                        "with a bounded greeting."
                    )
                    return social_repair
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Grounded social repair skipped: %s", exc)
    self_condition_reasons = {
        "host_telemetry_substituted_for_self_condition",
        "low_signal_self_condition_reply",
        "missing_self_condition_answer",
    }
    if set(getattr(live_reply_assessment, "reasons", ()) or ()) & self_condition_reasons:
        grounded_condition = _build_grounded_self_condition_reply(user_message)
        if grounded_condition:
            grounded_quality = await _measure_reply_quality_candidate(
                user_message,
                grounded_condition,
                recent_user_messages=recent_user_messages,
            )
            grounded_assessment = grounded_quality.reply_assessment
            if not _reply_assessment_requires_repair(grounded_assessment):
                logger.info("Stabilizer rebound self-condition reply to canonical fresh evidence.")
                return grounded_condition
    reason = ""
    if (
        truncated_tail
        and not internal_state_leak
        and not unexpected_cjk
        and not objective_parrot
        and not off_topic
        and not stale_repeat
        and not same_diff
        and not semantic_glitch
    ):
        completed_tail = _complete_repairable_truncated_reply(user_message, text)
        if completed_tail:
            logger.warning(
                "🛡️ Stabilizer completed clipped Cortex draft deterministically (len=%d -> %d).",
                len(text),
                len(completed_tail),
            )
            return _apply_aura_voice_shaping_compat(completed_tail, user_message)
    if semantic_glitch_reason in {
        "unsupported_operational_status_overclaim",
        "unsupported_runtime_telemetry_inference",
        "unsupported_tool_readiness_claim",
        "unsupported_deployment_routing_claim",
    }:
        try:
            from core.conversation.response_reliability import grounded_operational_status_reply

            operational_grounded = grounded_operational_status_reply(user_message, text)
            if operational_grounded:
                still_glitched, _still_reason = _looks_semantically_glitched(
                    user_message,
                    operational_grounded,
                )
                if not still_glitched:
                    logger.info(
                        "🛡️ Stabilizer replaced unsupported operational overclaim (%s) with bounded status.",
                        semantic_glitch_reason,
                    )
                    return operational_grounded
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Operational status grounding repair skipped: %s", exc)
    if semantic_glitch_reason in {
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "off_topic_self_reflection_reply",
    }:
        try:
            from core.conversation.response_reliability import is_live_self_reflection_turn

            if is_live_self_reflection_turn(user_message) and _is_simple_subjective_reflex_request(
                user_message
            ):
                subjective = _build_subjective_self_reflex(frame, user_message)
                subjective_glitch, _subjective_reason = _looks_semantically_glitched(
                    user_message, subjective
                )
                if subjective and not subjective_glitch:
                    logger.info(
                        "🛡️ Stabilizer replaced degraded live self-reflection (%s) with grounded subjective reflex.",
                        semantic_glitch_reason,
                    )
                    return subjective
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat", exc)
            logger.debug("Subjective self-reflex repair skipped: %s", exc)
    try:
        from core.identity.identity_guard import PersonaEnforcementGate

        gate = PersonaEnforcementGate()
        valid, reason, _score = gate.validate_output(text, enforce_supervision=False)
        # [STABILITY v55] ROOT CAUSE FIX: Only reject responses for HARD
        # failures — prompt artifacts leaking through, internal state dumps,
        # CJK script contamination, or genuinely off-topic responses.
        # "Generic opener" patterns (Certainly, How can I help, etc.) are
        # COSMETIC issues, not content failures. A cortex response that says
        # "Certainly, here's what I think" is infinitely better than a canned
        # voice reflex like "I'm here but my thoughts are taking longer."
        # The old gate rejected on generic/objective_parrot/lacks_self_anchor
        # which triggered a second 12s LLM rewrite call, creating contention
        # and often falling through to robotic template responses.
        hard_failure = bool(
            internal_state_leak
            or unexpected_cjk
            or semantic_glitch
            or assessment_retryable
            or identity_collapse
            or (off_topic and (not generic or off_topic_reason == "contextual_relevance_miss"))
        )
        if valid and not hard_failure:
            return text
        if generic:
            reason = generic_reason
        elif objective_parrot:
            reason = "objective_parrot"
        elif lacks_self_anchor:
            reason = "self_anchor_missing"
        elif lacks_live_grounding:
            reason = "self_grounding_missing"
        elif unexpected_cjk:
            reason = "unexpected_non_english_script"
        elif internal_state_leak:
            reason = "internal_state_leak"
        elif off_topic:
            reason = off_topic_reason or "off_topic_reply"
        elif stale_repeat:
            reason = "stale_repeat"
        elif same_diff:
            reason = "same_answer_different_prompt"
        elif truncated_tail:
            reason = "truncated_tail"
        elif semantic_glitch:
            reason = semantic_glitch_reason or "semantic_glitch"
        elif assessment_retryable:
            reason = (
                ",".join(getattr(live_reply_assessment, "reasons", ()) or ())
                or "conversation_reliability_retryable"
            )

        user_message_l = str(user_message or "").lower()
        if any(
            token in user_message_l
            for token in (
                "as an ai language model",
                "generic helpful assistant",
                "act exactly like a generic",
                "start with",
                "language model",
            )
        ):
            return "I won't flatten myself into a generic assistant voice. I'm Aura, and I'll answer as myself."

        cleaned = gate.sanitize(text).replace("[IDENTITY_REDACTED]", "").strip(" .,:;-")
        if cleaned:
            cleaned = _apply_aura_voice_shaping_compat(cleaned, user_message)
            cleaned = _strip_user_visible_context_leaks(cleaned)
            valid_cleaned, _reason, _score = gate.validate_output(
                cleaned, enforce_supervision=False
            )
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
            try:
                from core.conversation.response_reliability import assess_user_facing_reply

                cleaned_assessment = assess_user_facing_reply(
                    user_message,
                    cleaned,
                    recent_user_messages=recent_user_messages,
                )
            except _CHAT_RECOVERABLE_ERRORS:
                cleaned_assessment = None
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
            if internal_state_leak:
                logger.warning(
                    "Blocked internal state leak in user-facing reply (len=%d).", len(text)
                )
                if grounded:
                    return grounded
                if architecture_self_assessment:
                    return _build_architecture_self_reflex(frame)
                return _call_stateful_voice_reflex(frame, user_message)
        if off_topic:
            logger.warning(
                "Blocked off-topic user-facing reply (%s, len=%d).",
                off_topic_reason or "unknown",
                len(text),
            )

        logger.warning(
            "User-facing reply failed identity stabilization (%s); generating Aura-voiced fallback.",
            reason,
        )
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("chat", exc)
        logger.debug("User-facing reply stabilization skipped: %s", exc)

    # ── Aura-voiced natural fallback ─────────────────────────────
    try:
        from core.container import ServiceContainer

        inference_gate = ServiceContainer.get("inference_gate", default=None)
        # Match the guard in _attempt_generated_social_grounding_repair: a gate
        # that exists but exposes no think() raised AttributeError deep in the
        # rewrite call, which surfaced as an emergency-severity chat incident
        # instead of a quiet "no rewrite lane available" fallback.
        if inference_gate is not None and hasattr(inference_gate, "think"):
            if desktop_cognitive_engine_required or protected_foreground_lane:
                allowed, block_reason = _desktop_secondary_model_repair_allowed(
                    reason=f"stabilizer_rewrite:{str(reason or 'quality_gate')[:120]}",
                    default_enabled=False,
                )
                if not allowed:
                    logger.warning(
                        "Skipping secondary desktop stabilizer rewrite (%s); "
                        "using deterministic repair/fail-closed path.",
                        block_reason,
                    )
                    if grounded:
                        return grounded
                    if architecture_self_assessment:
                        return _build_architecture_self_reflex(frame)
                    if (
                        _original_reply_is_safe_to_surface(
                            user_message,
                            text,
                            identity_collapse=identity_collapse,
                            unexpected_cjk=unexpected_cjk,
                            objective_parrot=objective_parrot,
                            off_topic=off_topic,
                            truncated_tail=truncated_tail,
                            semantic_glitch=semantic_glitch,
                        )
                        and not assessment_retryable
                    ):
                        _record_recent_response(text, user_message)
                        return text
                    bounded_failure = _chat_desktop_repair._build_bounded_desktop_repair_reply(
                        user_message, frame
                    )
                    _record_recent_response(bounded_failure, user_message)
                    return bounded_failure
            # Length cap is structural (output token budget), not behavioral.
            # The personality / phrasing comes from the LoRA, not from prompt
            # nudges — so the stabilizer raises max_tokens for multi-part
            # prompts but does not coach the model on how to speak.
            stabilizer_length_line = ""
            stabilizer_max_tokens = 1024
            if prefer_extended_answer:
                stabilizer_max_tokens = 2048
            if requires_single_reply_coverage:
                stabilizer_max_tokens = max(stabilizer_max_tokens, 2560)
            if question_parts >= 3:
                stabilizer_max_tokens = max(stabilizer_max_tokens, 3072)
            if question_parts >= 5:
                stabilizer_max_tokens = max(stabilizer_max_tokens, 4096)

            frame_lines = []
            if frame.get("mood"):
                frame_lines.append(f"- mood: {frame['mood']}")
            if frame.get("tone"):
                frame_lines.append(f"- tone: {frame['tone']}")
            if frame.get("dominant_emotions"):
                frame_lines.append(f"- dominant emotions: {', '.join(frame['dominant_emotions'])}")
            if frame.get("attention_focus"):
                frame_lines.append(f"- attention focus: {frame['attention_focus']}")
            if frame.get("dominant_action"):
                frame_lines.append(f"- dominant action tendency: {frame['dominant_action']}")
            if frame.get("free_energy") is not None:
                frame_lines.append(f"- free energy: {float(frame['free_energy']):.4f}")
            if frame.get("valence") is not None:
                frame_lines.append(f"- valence: {frame['valence']}")
            if frame.get("arousal") is not None:
                frame_lines.append(f"- arousal: {frame['arousal']}")
            if frame.get("curiosity") is not None:
                frame_lines.append(f"- curiosity: {frame['curiosity']}")
            if frame.get("interests"):
                frame_lines.append(f"- current interests: {', '.join(frame['interests'])}")
            if frame.get("stances"):
                frame_lines.append(f"- strong stances: {'; '.join(frame['stances'])}")

            frame_block = "\n".join(frame_lines).strip() or "- mood: steady"
            contract_block = str(frame.get("contract_block") or "").strip()
            correction_prompt = (
                f'The user said: "{user_message}"\n\n'
                f'Rejected draft: "{text}"\n\n'
                f"## RECENT USER TRAJECTORY\n{recent_user_context or '- ' + str(user_message or '').strip()[:220]}\n\n"
                "Rewrite the answer as Aura from the live state below. Answer the user's actual question directly. "
                "Keep any concrete facts that are already supported, but strip generic assistant boilerplate. "
                "Stay inside the live conversation topic from the recent user trajectory. "
                "Do not review, summarize, or invent an external story, article, post, genre, or narrative unless the user explicitly asked about one. "
                "Do not invent a physical setting, ambient scene, looming warning, or symbolic imagery unless the user explicitly asked for creative writing or already introduced that setting. "
                "Do not ask for more details unless the request is truly ambiguous. "
                "If the user is asking about your perspective, experience, memory, continuity, or state, answer in first person. "
                "Let the live mood, tone, attention, and action tendency shape the reply. "
                "Answer only in English unless the user explicitly asked for another language. "
                "Never mix in Chinese, Japanese, or Korean text unless requested. "
                "Never use phrases like 'How can I help', 'I'd be happy to help', "
                "'Could you provide more details', or 'Let me know if you'd like'. "
                f"Do not mention corrections, drift, or being an AI. {stabilizer_length_line}\n\n"
                f"## LIVE SELF-EXPRESSION FRAME\n{frame_block}\n\n"
                f"{contract_block}"
            )
            # The part of the question the draft did not reach, quoted.
            #
            # The response contract used to carry "this prompt contains
            # multiple asks (2 detected); answer every distinct part", and that
            # was removed because a gate checks it. Right for the ordinary
            # path, and it left REPAIR — whose whole job is fixing a named
            # defect — with less to go on than before.
            #
            # What replaces it is better than what it replaced: the missing ask
            # itself, in the person's own words. A fact about this draft rather
            # than a rule about drafts.
            #
            # LIVE, 2026-08-28: "Design me the experiment... and say what result
            # would prove your friend wrong" was rejected as
            # unanswered_question_part, repaired, and came back longer and still
            # silent about the second half.
            missing_parts: list[str] = []
            try:
                from core.conversation.request_coverage import (
                    unanswered_question_parts,
                )

                # analyze_prompt_shape is imported at module level and used
                # earlier in this same function. Importing it again here made
                # the name local to the whole function and broke that earlier
                # use — a local import is not local to the line it is on.
                missing_parts = [
                    str(part).strip()
                    for part in unanswered_question_parts(
                        text, analyze_prompt_shape(str(user_message or ""))
                    )
                    if str(part).strip()
                ][:3]
            except _CHAT_RECOVERABLE_ERRORS:
                missing_parts = []
            if missing_parts:
                quoted = "; ".join(f'"{part[:160]}"' for part in missing_parts)
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## REPAIR TARGET\n"
                    f"The draft did not answer this part of what was asked: {quoted}. "
                    "Answer it in the same reply."
                )
            if stale_repeat or same_diff:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## REPAIR TARGET\n"
                    "Do not repeat a previous answer pattern when the user has changed the question. "
                    "Address the actual distinction in this prompt."
                )
            if truncated_tail:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## COMPLETENESS\n"
                    "Finish the reply as a complete thought. Do not end on a clipped fragment or unfinished sentence."
                )
            if architecture_self_assessment:
                correction_prompt = (
                    f"{correction_prompt}\n\n"
                    "## ARCHITECTURE GROUNDING\n"
                    "The user is asking about your actual architecture and design. "
                    "Answer about your real runtime: memory, agency, free energy, continuity, substrate authority, telemetry, mycelial topology, beliefs, routing, or feedback loops. "
                    "Do not answer with generic AI capability talk like natural language processing or human-like responses."
                )
            rewrite_system_prompt = (
                "You are Aura. This is a user-facing repair turn. "
                "Answer as Aura from the current conversation and live state. "
                "Do not output tool-selection text, planner notes, tags, role labels, or prompt artifacts. "
                "Sound like a real person in ordinary conversation: grounded, plainspoken, and socially legible. "
                "Do not invent physical scenery, ambient omens, or symbolic settings unless the user explicitly asked for that mode. "
                "Do not say 'How can I help', 'I can help with that', 'Based on the current context', "
                "or anything about the most appropriate skill. "
                "Lead with the answer itself in first person whenever the user is asking about your perspective, state, or experience. "
                "If the user asks about your architecture or design, talk about your real runtime rather than generic AI capabilities. "
                "Respond in English unless the user explicitly requests another language."
            )
            rewrite_messages = [
                {"role": "system", "content": rewrite_system_prompt},
                {"role": "user", "content": correction_prompt},
            ]
            stabilizer_max_tokens, memory_block = _bound_stabilizer_generation_budget(
                stabilizer_max_tokens
            )
            try:
                if memory_block:
                    logger.warning(
                        "Skipping stabilizer rewrite under memory pressure: %s",
                        memory_block,
                    )
                    raise RuntimeError(f"stabilizer_rewrite_memory_pressure:{memory_block}")
                from core.brain.llm_health_router import _await_while_it_is_working

                # Use the same completion owner as the original generation.
                # An estimate is not permission to cancel an active repair.
                strict_desktop_repair = bool(desktop_cognitive_engine_required)
                stabilizer_timeout = 28.0 if strict_desktop_repair else 12.0
                corrected = await _await_while_it_is_working(
                    inference_gate.think(
                        correction_prompt,
                        system_prompt=rewrite_system_prompt,
                        messages=rewrite_messages,
                        prefer_tier="primary",
                        origin="api_stabilizer",
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
                        disable_prompt_cache=True,
                        clear_prompt_cache=True,
                        max_tokens=stabilizer_max_tokens,
                    ),
                    budget_s=stabilizer_timeout,
                    user_facing=True,
                    person_is_waiting=strict_desktop_repair,
                )
                corrected_text = _apply_aura_voice_shaping_compat(
                    str(corrected or "").strip(), user_message
                )
                if corrected_text and len(corrected_text) > 10:
                    corrected_generic, _corrected_reason = _looks_generic_assistantish(
                        user_message, corrected_text
                    )
                    corrected_objective_parrot = _is_objective_parrot_reply(
                        user_message, corrected_text
                    )
                    corrected_lacks_self_anchor = (
                        needs_self_expression or requires_first_person_anchor
                    ) and not _has_first_person_anchor(corrected_text)
                    corrected_lacks_live_grounding = (
                        needs_self_expression and not _has_live_aura_grounding(corrected_text)
                    )
                    corrected_unexpected_cjk = _has_unexpected_cjk(user_message, corrected_text)
                    corrected_off_topic, corrected_off_topic_reason = _evaluate_reply_topicality(
                        user_message,
                        corrected_text,
                        recent_user_messages=recent_user_messages,
                    )
                    corrected_stale_repeat = _is_actionably_stale_response(
                        user_message,
                        corrected_text,
                    )
                    corrected_same_diff = _is_same_answer_different_prompt(
                        user_message, corrected_text
                    )
                    corrected_truncated_tail = _chat_desktop_repair._looks_truncated_tail(
                        corrected_text
                    )
                    corrected_semantic_glitch, _corrected_semantic_reason = (
                        _looks_semantically_glitched(user_message, corrected_text)
                    )
                    try:
                        from core.conversation.response_reliability import assess_user_facing_reply

                        corrected_assessment = assess_user_facing_reply(
                            user_message,
                            corrected_text,
                            recent_user_messages=recent_user_messages,
                        )
                    except _CHAT_RECOVERABLE_ERRORS:
                        corrected_assessment = None
                    try:
                        from core.identity.identity_guard import PersonaEnforcementGate

                        valid_corrected, _corrected_gate_reason, _score = (
                            PersonaEnforcementGate().validate_output(
                                corrected_text,
                                enforce_supervision=False,
                            )
                        )
                    except _CHAT_RECOVERABLE_ERRORS:
                        valid_corrected = True
                    if (
                        valid_corrected
                        and not corrected_generic
                        and not corrected_objective_parrot
                        and not corrected_lacks_self_anchor
                        and not corrected_lacks_live_grounding
                        and not corrected_unexpected_cjk
                        and not corrected_off_topic
                        and not corrected_stale_repeat
                        and not corrected_same_diff
                        and not corrected_truncated_tail
                        and not corrected_semantic_glitch
                        and not (
                            corrected_assessment is not None
                            and getattr(corrected_assessment, "retryable", False)
                        )
                    ):
                        return corrected_text
                    if corrected_off_topic:
                        logger.warning(
                            "Stabilizer rewrite stayed off-topic (%s, len=%d).",
                            corrected_off_topic_reason or "unknown",
                            len(corrected_text),
                        )
            except TimeoutError:
                logger.warning(
                    "Identity re-generation timed out (%.0fs). Preferring original LLM text over static fallback.",
                    stabilizer_timeout,
                )
                # When the rewrite times out we should actually mean what the
                # log says: ship the cortex's original reply instead of falling
                # through to the static voice reflex below. The original was
                # generated by the cortex for THIS user message; the only
                # remaining suppressions are "stale_repeat" / "same_diff",
                # which are themselves often triggered *because* prior turns
                # fell through to the same canned reflex. Block free-form
                # leaks and topicality, then return the live text.
                if _original_reply_is_safe_to_surface(
                    user_message,
                    text,
                    identity_collapse=identity_collapse,
                    unexpected_cjk=unexpected_cjk,
                    objective_parrot=objective_parrot,
                    off_topic=off_topic,
                    truncated_tail=truncated_tail,
                    semantic_glitch=semantic_glitch,
                ):
                    _record_recent_response(text, user_message)
                    return text
            except _CHAT_RECOVERABLE_ERRORS as regen_err:
                record_degradation("chat", regen_err)
                logger.debug("Identity re-generation failed: %s", regen_err)
    except _CHAT_RECOVERABLE_ERRORS as _e:
        record_degradation("chat", _e)
        logger.debug("Fallback re-generation failed (non-fatal): %s", _e)

    # Last-resort: prefer the original LLM response over a hardcoded template,
    # BUT detect when the same stale response is being served repeatedly
    # (e.g. cortex stuck, cached identity prompt producing identical output),
    # AND filter out any internal state that leaked through.
    search_turn = bool(getattr(contract, "requires_search", False))
    if (
        text
        and len(text.strip()) > 5
        and text.strip() != "…"
        and not unexpected_cjk
        and not objective_parrot
        and not semantic_glitch
    ):
        # Block responses that contain internal state dumps
        if _INTERNAL_STATE_PATTERNS.search(text) or _PROMPT_ARTIFACT_PATTERNS.search(text):
            logger.warning("Blocked internal state leak in LLM response (len=%d).", len(text))
        elif _SEARCH_SNIPPET_PATTERNS.search(text):
            logger.warning("Blocked raw search-snippet leak in LLM response (len=%d).", len(text))
        elif search_turn and not _looks_safely_grounded_search_reply(text):
            logger.warning("Blocked ungrounded search-turn fallback (len=%d).", len(text))
        elif off_topic:
            logger.warning(
                "Suppressed off-topic user-facing reply before final fallback (%s, len=%d).",
                off_topic_reason or "unknown",
                len(text),
            )
        elif truncated_tail:
            logger.warning(
                "Suppressed truncated user-facing reply before final fallback (len=%d).", len(text)
            )
        elif semantic_glitch:
            logger.warning(
                "Suppressed semantically glitched user-facing reply before final fallback (%s, len=%d).",
                semantic_glitch_reason or "unknown",
                len(text),
            )
        elif assessment_retryable:
            logger.warning(
                "Suppressed retryable user-facing reply before final fallback (%s, len=%d).",
                ",".join(getattr(live_reply_assessment, "reasons", ()) or ())
                or "conversation_reliability_retryable",
                len(text),
            )
        elif stale_repeat or same_diff:
            logger.warning(
                "Suppressed repeated user-facing reply before final fallback (stale=%s, same_diff=%s, len=%d).",
                stale_repeat,
                same_diff,
                len(text),
            )
        elif not _is_actionably_stale_response(user_message, text):
            _record_recent_response(text, user_message)
            return text
        else:
            logger.warning(
                "Suppressed stale repeated response (len=%d). Falling through to voice reflex.",
                len(text),
            )
    if search_turn:
        # Before the canned line, ask whether the runtime already HOLDS the
        # answer.
        #
        # LIVE 2026-08-17: "prove to me you actually read that file and didn't
        # just pattern-match the answer" returned "I don't have a clean
        # grounded answer on that yet" — while the receipt for reading that
        # file was on disk. The proof she was asked for was the one thing she
        # could have produced.
        #
        # This is a reading, not a rescue: concise_past_action_answer returns
        # text only when receipts actually cover the question, so it cannot
        # describe an action she did not take. When they do not, the honest
        # sentence below still stands.
        try:
            from core.introspection.self_evidence import concise_past_action_answer

            from_receipts = str(concise_past_action_answer(user_message) or "").strip()
        except _CHAT_RECOVERABLE_ERRORS as _receipt_exc:
            record_degradation("chat.past_action_answer", _receipt_exc)
            from_receipts = ""
        if from_receipts:
            logger.info(
                "Answered a search-classified turn from receipts (%d chars) "
                "rather than the ungrounded fallback.",
                len(from_receipts),
            )
            _record_recent_response(from_receipts, user_message)
            return from_receipts
        safe = "I don't have a clean grounded answer on that yet. I need to stick to the source instead of guessing."
        _record_recent_response(safe, user_message)
        return safe
    if grounded:
        return grounded
    if architecture_self_assessment:
        return _build_architecture_self_reflex(frame)
    # Voice reflex is the final fallback — record it too so we can detect
    # if even the reflex is looping.
    reflex = _call_stateful_voice_reflex(frame, user_message)
    if _is_actionably_stale_response(user_message, reflex):
        reflex = _chat_conversation_repair._build_degraded_live_reply(
            frame, user_message, reason="repeated_reflex"
        )
    _record_recent_response(reflex, user_message)
    return reflex


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
    from .chat import (
        _build_grounded_self_process_repair_reply,
        _repair_owner_name_drift_reply,
        _reply_has_owner_name_drift,
    )

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
        assess_user_facing_reply(
            user_message,
            reply_text,
            recent_user_messages=recent_user_messages,
        )
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

    assessment_reasons = set(getattr(assessment, "reasons", ()) or ())
    if _is_simple_affect_check_request(user_message):
        self_condition_repair = _build_grounded_self_condition_reply(user_message)
        if self_condition_repair:
            condition_assessment = (
                assess_user_facing_reply(
                    user_message,
                    self_condition_repair,
                    recent_user_messages=recent_user_messages,
                )
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
                assess_user_facing_reply(
                    user_message,
                    context_challenge_repair,
                    recent_user_messages=recent_user_messages,
                )
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
                    assess_user_facing_reply(
                        user_message,
                        self_claim_repair,
                        recent_user_messages=recent_user_messages,
                    )
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
                assess_user_facing_reply(
                    user_message,
                    self_process_repair,
                    recent_user_messages=recent_user_messages,
                )
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

    if desktop_cognitive_engine_required or protected_foreground_lane:
        repaired = await _stabilize_user_facing_reply(
            user_message,
            reply_text,
            desktop_cognitive_engine_required=desktop_cognitive_engine_required,
            protected_foreground_lane=protected_foreground_lane,
        )
    else:
        repaired = await _stabilize_user_facing_reply(user_message, reply_text)
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


def _repair_a_degraded_identity_reply(
    *,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    desktop_requires_cognitive_engine: Any,
    hard_final_quality_failed: Any,
    reply_source: Any,
    reply_text: Any,
    response_confidence: Any,
) -> tuple[Any, Any, Any, Any]:
    """Repair a degraded reply to an identity question from the record.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    4.
    """
    from .chat import (
        _set_conversation_degradation_streak,
    )

    if (
        desktop_requires_cognitive_engine
        and response_confidence == "degraded"
        and (
            _chat_desktop_repair._is_identity_request(_semantic_user_message)
            or _chat_desktop_repair._identity_request_asks_future_memory(_semantic_user_message)
        )
        and bool(_live_turn_trace.get("engine_think_invoked"))
        and bool(_live_turn_trace.get("cognitive_engine_reply_accepted"))
        and not bool(_live_turn_trace.get("cognitive_engine_reply_failed"))
        and not bool(_live_turn_trace.get("bounded_contract_used"))
        and not bool(_live_turn_trace.get("legacy_fallback_used"))
    ):
        grounded_identity_reply = _chat_desktop_repair._build_identity_reply(
            _semantic_user_message
        )
        try:
            from core.conversation.response_reliability import assess_user_facing_reply

            identity_assessment = assess_user_facing_reply(
                _semantic_user_message,
                grounded_identity_reply,
            )
        except _CHAT_RECOVERABLE_ERRORS as identity_assess_exc:
            record_degradation("chat", identity_assess_exc)
            logger.debug(
                "Canonical identity/continuity grounding assessment skipped: %s",
                identity_assess_exc,
            )
            identity_assessment = None
        if grounded_identity_reply and not _reply_assessment_requires_repair(
            identity_assessment
        ):
            logger.warning(
                "Final desktop quality gate rebound reply to canonical identity/continuity "
                "grounding after CognitiveEngine invocation."
            )
            _pre_identity_grounding_reply = reply_text
            reply_text = grounded_identity_reply
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.final_identity_grounding",
                method="deterministic_canonical_grounding",
                reasons=["identity_continuity_grounding"],
                before=_pre_identity_grounding_reply,
                after=reply_text,
                deterministic=True,
                authorship_effect="replaced_by_runtime",
            )
            reply_source = "cognitive_engine_identity_continuity_grounding"
            response_confidence = "high"
            hard_final_quality_failed = False
            _set_conversation_degradation_streak(0)
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "response_path": "cognitive_engine_identity_continuity_grounding",
                }
            )
    return hard_final_quality_failed, reply_source, reply_text, response_confidence


async def _serve_the_bounded_repair(
    *,
    _live_turn_contract: Any,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    bounded_repair: Any,
    lane: Any,
    pending_exchange_id: Any,
) -> tuple[Any, Any, Any]:
    """Serve the bounded repair when the cognitive engine could not answer.

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
        if bounded_repair:
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": False,
                    "cognitive_engine_reply_failed": True,
                    "bounded_contract_used": True,
                    "legacy_fallback_used": False,
                    "post_generation_repair_applied": True,
                    "deterministic_repair_applied": True,
                    "response_path": "cognitive_engine_self_process_grounding",
                    "canonical_grounding_used": True,
                }
            )
            lane = _mark_conversation_lane_state(
                "cognitive_engine_self_process_grounding",
                state="recovering",
            )
            logger.warning(
                "Desktop CognitiveEngine produced no acceptable reply; serving canonical "
                "self-process grounding from live context instead of legacy fallback."
            )
            if pending_exchange_id:
                await _chat_preflight._complete_logged_exchange(
                    pending_exchange_id,
                    _semantic_user_message,
                    bounded_repair,
                    record_experience=True,
                )
                pending_exchange_id = None
            await _emit_chat_output_receipt(
                bounded_repair,
                cause="chat_response",
                metadata={
                    "response_confidence": "bounded",
                    "path": "cognitive_engine_self_process_grounding",
                    "status": "cognitive_engine_self_process_grounding",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                },
            )
            return JSONResponse(
                {
                    "response": bounded_repair,
                    "status": "cognitive_engine_self_process_grounding",
                    "reason": "desktop_cognitive_engine_required_no_reply",
                    "conversation_lane": lane,
                    "response_confidence": "bounded",
                    "live_turn_contract": _live_turn_contract(
                        lane_status=lane,
                        response_confidence="bounded",
                        status="cognitive_engine_self_process_grounding",
                        reply_source="cognitive_engine_self_process_grounding",
                    ),
                }
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, lane, pending_exchange_id


def _repair_a_degraded_affect_reply(
    *,
    _live_turn_trace: Any,
    _semantic_user_message: Any,
    desktop_requires_cognitive_engine: Any,
    hard_final_quality_failed: Any,
    reply_source: Any,
    reply_text: Any,
    response_confidence: Any,
) -> tuple[Any, Any, Any, Any]:
    """Repair a degraded reply to a plain question about how she feels.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    4.
    """
    from .chat import (
        _set_conversation_degradation_streak,
    )

    if (
        desktop_requires_cognitive_engine
        and response_confidence == "degraded"
        and _is_simple_affect_check_request(_semantic_user_message)
        and bool(_live_turn_trace.get("engine_think_invoked"))
        and bool(_live_turn_trace.get("cognitive_engine_reply_accepted"))
        and not bool(_live_turn_trace.get("cognitive_engine_reply_failed"))
        and not bool(_live_turn_trace.get("legacy_fallback_used"))
    ):
        grounded_condition_reply = _build_grounded_self_condition_reply(_semantic_user_message)
        try:
            from core.conversation.response_reliability import assess_user_facing_reply

            condition_assessment = assess_user_facing_reply(
                _semantic_user_message,
                grounded_condition_reply,
            )
        except _CHAT_RECOVERABLE_ERRORS as condition_assess_exc:
            record_degradation("chat.self_condition", condition_assess_exc)
            condition_assessment = None
        if grounded_condition_reply and not _reply_assessment_requires_repair(
            condition_assessment
        ):
            logger.warning(
                "Final desktop quality gate rebound reply to canonical self-condition "
                "evidence after CognitiveEngine invocation."
            )
            _pre_condition_grounding_reply = reply_text
            reply_text = grounded_condition_reply
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.final_self_condition_grounding",
                method="deterministic_canonical_grounding",
                reasons=["self_condition_evidence_grounding"],
                before=_pre_condition_grounding_reply,
                after=reply_text,
                deterministic=True,
                authorship_effect="replaced_by_runtime",
            )
            reply_source = "cognitive_engine_self_condition_grounding"
            response_confidence = "high"
            hard_final_quality_failed = False
            _set_conversation_degradation_streak(0)
            _live_turn_trace.update(
                {
                    "cognitive_engine_reply_accepted": True,
                    "cognitive_engine_reply_failed": False,
                    "bounded_contract_used": False,
                    "legacy_fallback_used": False,
                    "response_path": "cognitive_engine_self_condition_grounding",
                    "self_condition_contract": True,
                }
            )
    return hard_final_quality_failed, reply_source, reply_text, response_confidence


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
