"""How a chat turn ends when no reply of hers formed.

Every exit of the chat route that had nothing of hers to send used to write
a sentence for her: the giving-up line, the lane's status lines with a mood
put in front ("Ugh, ..."), the error floor's line. This module is the one
place those exits go now. The turn returns an empty reply and the runtime's
notice (core/conversation/runtime_notice.py), and the exchange keeps the
person's words with no reply of hers.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi.responses import JSONResponse

from core.conversation.runtime_notice import NO_REPLY_FORMED, no_reply_fields
from core.runtime.errors import record_degradation
from interface.routes import chat_memory_state as _chat_memory_state
from interface.routes import chat_preflight as _chat_preflight
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

logger = logging.getLogger("Aura.Server.Chat")


async def close_unanswered(exchange_id: str | None, user_message: str, *, notice: str) -> str:
    """Keep the person's message as an exchange with no reply of hers.

    Under the delivery boundary the exchange is held until the turn is
    sealed, and the boundary calls this again with the sealed notice. The
    person's turn was written durably when the exchange opened; it is written
    here only if that write had failed.
    """
    capture = _chat_preflight._TERMINAL_EXCHANGES.get()
    if capture is not None and exchange_id in capture.exchanges:
        capture.exchanges[exchange_id]["record_experience"] = False
        return "pending_terminal_delivery"
    async with _chat_memory_state._get_convo_lock():
        target = next(
            (entry for entry in reversed(_chat_preflight._conversation_log)
             if exchange_id and str(entry.get("id") or "") == str(exchange_id)),
            None,
        )
        if target is None:
            principal_id, principal_surface = _chat_memory_state._chat_memory_identity()
            target = {
                "id": exchange_id or _chat_preflight._new_exchange_id(),
                "timestamp": _chat_preflight._utc_now_iso(),
                "user": str(user_message or ""),
                "principal_id": principal_id,
                "principal_surface": principal_surface,
                "session_id": str(_chat_preflight._CHAT_REQUEST_SESSION.get() or "")[:64],
                "user_persisted": False,
            }
            _chat_preflight._conversation_log.append(target)
        metadata = dict(target.get("metadata") or {}) if isinstance(target.get("metadata"), dict) else {}
        metadata["runtime_notice"] = str(notice or "")
        target.update({"aura": "", "status": "complete", "unanswered": True,
                       "completed_at": _chat_preflight._utc_now_iso(), "metadata": metadata})
        _chat_preflight._trim_conversation_log_locked()
        persisted = bool(target.get("user_persisted"))
        recorded = (str(target.get("id") or ""), str(target.get("user") or ""), str(target.get("session_id") or ""))
    if persisted:
        return "committed"
    state = await _chat_preflight._persist_pending_conversation_user(
        exchange_id=recorded[0], user_message=recorded[1], session_id=recorded[2],
    )
    async with _chat_memory_state._get_convo_lock():
        target["user_persisted"] = state == "committed"
        target["user_persistence_state"] = state
    return state


async def close_unanswered_exchanges(capture: Any, payload: dict[str, Any]) -> bool:
    """The delivery boundary's close of a sealed turn that carried a notice and no reply."""
    notice = str(payload.get("notice") or "")
    token = _chat_preflight._TERMINAL_EXCHANGES.set(None)
    committed = True
    try:
        for exchange_id, entry in capture.exchanges.items():
            state = await close_unanswered(exchange_id, str(entry.get("user") or ""), notice=notice)
            committed = committed and state == "committed"
    finally:
        _chat_preflight._TERMINAL_EXCHANGES.reset(token)
    return committed


async def no_reply_response(
    *,
    why: str = NO_REPLY_FORMED,
    held: str = "",
    status: str,
    user_message: str = "",
    pending_exchange_id: str | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    path: str = "",
    **fields: Any,
) -> JSONResponse:
    """The response of a turn that ends with no reply of hers, and its exchange closed as unanswered."""
    from .chat import _emit_chat_output_receipt

    notice = no_reply_fields(why, held=held)
    if pending_exchange_id or user_message:
        try:
            await close_unanswered(pending_exchange_id, user_message, notice=notice["notice"])
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation("chat.unanswered_exchange", exc, action="served the notice without closing its exchange")
    response = no_reply_payload(why=why, held=held, status=status, status_code=status_code, headers=headers, **fields)
    await _emit_chat_output_receipt(
        notice["notice"], cause="runtime_notice", origin="runtime", target="notice",
        metadata={"status": status, "reason": str(fields.get("reason") or ""), "path": path or status,
                  "response_confidence": "failed"},
    )
    return response


def no_reply_payload(
    *,
    why: str = NO_REPLY_FORMED,
    held: str = "",
    status: str,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    **fields: Any,
) -> JSONResponse:
    """The same response for an exit that cannot wait; the delivery boundary closes its exchange."""
    notice = no_reply_fields(why, held=held)
    logger.warning("No reply of hers formed (%s): %s", status, notice["notice"][:240])
    return JSONResponse({**fields, "status": status, **notice}, status_code=status_code, headers=headers)


def what_the_runtime_held(message: str, *, session_id: Any = None) -> str:
    """What the runtime already holds for this message without her model, or "".

    In order: an exact computed answer, her condition when that is what was
    asked, her live readings or a record of what she did, and what this
    turn's tools returned. Each is a reading; none is written as her reply.
    """
    from . import chat as _chat

    held = _chat._known_answer_for_this_turn()
    if not held and _chat._is_simple_affect_check_request(message):
        held = _chat._build_grounded_self_condition_reply(message, session_id=session_id)
    if not held:
        held = _chat._chat_conversation_repair._self_health_answer_or_empty(message)
    return held or _chat._what_the_tools_found()


async def what_a_turn_with_no_reply_returns(
    *,
    semantic_message: str,
    original_message: str,
    session_id: Any,
    pending_exchange_id: str | None,
    status: str,
    reason: str,
    lane: Any,
    reply_source: str,
    is_benchmark: bool,
    budget_s: float | None,
    live_turn_contract: Callable[..., Any],
    receipt_path: str = "",
) -> JSONResponse:
    """The end of a turn her cognitive engine could not serve.

    What the runtime holds goes in the notice. With nothing held the smaller
    model is asked, and what it writes is served as an answer marked
    "fallback", which the page tags as the smaller model's. With neither, the
    notice says no reply formed.
    """
    from . import chat as _chat

    held = what_the_runtime_held(semantic_message, session_id=session_id)
    said = "" if held else await _chat._anything_better_than_giving_up(
        semantic_message, reason="the cognitive engine could not serve this turn", already="", budget_s=budget_s,
    )
    status_code = 503 if is_benchmark else 200
    if not said:
        return await no_reply_response(
            held=held, status=status, user_message=semantic_message if pending_exchange_id else original_message,
            pending_exchange_id=pending_exchange_id, status_code=status_code, reason=reason, path=receipt_path or reply_source,
            conversation_lane=lane, live_turn_contract=live_turn_contract(
                lane_status=lane, response_confidence="failed", status=status, reply_source=reply_source,
            ),
        )
    if pending_exchange_id:
        await _chat_preflight._complete_logged_exchange(
            pending_exchange_id, semantic_message, said, record_experience=False,
        )
    else:
        await _chat_preflight._log_exchange(
            original_message, said, record_experience=False, session_id=session_id,
        )
    await _chat._emit_chat_output_receipt(
        said, cause="chat_response",
        metadata={"response_confidence": "fallback", "path": receipt_path or reply_source, "status": status,
                  "reason": reason},
    )
    return JSONResponse(
        {
            "response": said, "status": status, "reason": reason, "conversation_lane": lane,
            "response_confidence": "fallback",
            "live_turn_contract": live_turn_contract(
                lane_status=lane, response_confidence="fallback", status=status, reply_source=reply_source,
            ),
        },
        status_code=status_code,
    )


__all__ = [
    "close_unanswered",
    "close_unanswered_exchanges",
    "no_reply_payload",
    "no_reply_response",
    "what_a_turn_with_no_reply_returns",
    "what_the_runtime_held",
]
