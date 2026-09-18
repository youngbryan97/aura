"""Repairing a degraded reply to a question about herself.

Two kinds, and both read the record rather than asking the model again:
what she is, and how she feels. They were the last four hundred lines of
``chat_reply_repair``, which crossed the 2,000-line ceiling that
``tests/test_a_god_object_only_shrinks.py`` says a module NOT already in
the baseline never gets grandfathered past.

They came out whole. Neither takes anything from the module they left —
no shared state, no helper defined beside them — which is what made this
a move rather than a rewrite: every caller keeps the name it had, because
the parent re-exports both.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import (  # noqa: E402
    JSONResponse,
)

from core.runtime.errors import (  # noqa: E402
    record_degradation,
)
from interface.routes import chat_desktop_repair as _chat_desktop_repair  # noqa: E402
from interface.routes import chat_preflight as _chat_preflight  # noqa: E402
from interface.routes.chat_common import (  # noqa: E402
    _CHAT_RECOVERABLE_ERRORS,
    logger,  # noqa: F401
)
from interface.routes.chat_lane_bookkeeping import (  # noqa: E402
    _is_simple_affect_check_request,
    _mark_conversation_lane_state,
)
from interface.routes.chat_quality import (  # noqa: E402
    _reply_assessment_requires_repair,
)
from interface.routes.chat_reply_shaping import (  # noqa: E402
    _append_turn_text_mutation,
    _build_grounded_self_condition_reply,
)
from interface.routes.chat_turn_evidence import (  # noqa: E402
    _emit_chat_output_receipt,
)


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
