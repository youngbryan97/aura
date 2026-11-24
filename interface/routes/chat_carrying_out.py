"""A turn that asks for something to be brought about: the request opened or found, the work begun, and what it comes to
told in the conversation when it is known.

Her reply to the turn is hers. What reaches her is her own call and what it came back with, in the form her template
gives a call of hers (core/conversation/surface_disposition.py): that the work has begun, or how it stands, from the
request itself. A turn she has begun work for offers no second way to do the same work.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.what_this_turn_asks import Asked, hold, this_turn
from core.understanding.requests_of_her import Request, get_requests_of_her
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

logger = logging.getLogger("Aura.Chat")

__all__ = ["ORIGIN", "take_up_what_is_asked", "tell_what_it_came_to", "the_message"]

#: What names this work to her lane, her log and her conversation.
ORIGIN = "work_asked_for"
#: The least time her words on what a request came to may take to form.
LEAST_S = 60.0


def _recorded(said: str, returned: dict[str, Any], *, ok: bool = True) -> None:
    """Her call, and what it came back with, where her turn reads its calls."""
    from core.conversation.surface_disposition import record_tool_receipt
    from core.conversation.turn_evidence_custody import record_turn_grounding

    text = json.dumps(returned, ensure_ascii=False)
    record_tool_receipt(ORIGIN, ok=ok, evidence=text[:1000], observed_content=text,
                        as_made={"arguments": {"request": said}, "returned": text})
    record_turn_grounding(text)


async def _the_words_before(session_id: str, said: str) -> list[str]:
    from interface.routes import chat_memory_state

    recent = await chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=said, session_id=session_id, limit=1, allow_cross_session=False)
    return [str(exchange.get("user") or "") for exchange in recent]


async def take_up_what_is_asked(said: str, *, session_id: str, attachments: Sequence[str] = ()) -> bool:
    """Take up what this turn was read as asking for; whether her turn is now about work of hers."""
    asked = this_turn()
    if asked is None:
        return False
    try:
        return await _take_up(asked, said, session_id=session_id, attachments=[str(a) for a in attachments])
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("carrying_out", exc, severity="warning", action="what this turn asked for was not begun")
        return False


async def _take_up(asked: Asked, said: str, *, session_id: str, attachments: list[str]) -> bool:
    from core.understanding import carrying_out
    from core.understanding.taking_up import settle

    ledger = get_requests_of_her()
    before = await _the_words_before(session_id, said)
    ledger.gave(session_id, said, attachments)
    settled = settle(asked, ledger, conversation=session_id, before=before)
    asked = settled.asked
    hold(asked)
    if asked.decision == "ask_first" and asked.effect in carrying_out.carriers():
        _recorded(said, {"begun": False, "could_be_asked_for": asked.effect, "sure_it_was_asked_for": round(asked.asking, 2)})
        return False
    if not asked.acts:
        return False
    if settled.what == "how_it_stands" and settled.request is not None:
        hold(dataclasses.replace(asked, request_id=settled.request.id))
        _recorded(said, {"how_it_stands": settled.request.account()})
        return True
    if asked.effect not in carrying_out.carriers():
        return False
    request = settled.request
    # What they gave with it; for a document, the one last given, and for the rest, what came with the turn before.
    materials = attachments or ledger.given(session_id, with_words=None if asked.effect == "document_read" else before)
    if asked.effect == "document_read" and request is None and not await _is_long(materials, said):
        return False  # it is whole in the turn already, or there is none: the turn answers
    if request is None:
        request = ledger.asked(said, effect=asked.effect, conversation=session_id, materials=materials)
    else:
        request.added.append(said)
        request.materials = list(dict.fromkeys([*request.materials, *materials]))
        if asked.effect == "program_changed":
            request.effect = "program_changed"
    hold(dataclasses.replace(asked, request_id=request.id))
    await ledger.keep()
    if carrying_out.is_under_way(request):
        _recorded(said, {"under_way": request.account(), "given_more": said})
        return True
    from core.skills.screen_pursuit import _tell

    begun = carrying_out.begin(request, tell=_tell, told=tell_what_it_came_to)
    _recorded(said, {"begun": begun, "request": request.said, "kind": request.effect,
                     "it_goes_on_after_this_reply": True, "they_will_be_told_what_it_came_to": True,
                     **({"how_it_stood": request.account()} if request.tries else {})}, ok=begun)
    logger.info("🧰 Request %s (%s) %s: %s", request.id, request.effect, "begun" if begun else "not begun", said[:120])
    return True


async def _is_long(materials: Sequence[str], said: str) -> bool:
    """Whether what was given is more than the turn itself can hold whole (interface/routes/chat_preflight.py)."""
    from core.conversation.attachments import kept
    from core.conversation.chat_preflight import MAX_COMPOSED_PREFLIGHT_CHARS

    held = [item for item in await asyncio.gather(*(asyncio.to_thread(kept, m) for m in materials)) if item]
    return sum(len(text) for _note, text in held) > MAX_COMPOSED_PREFLIGHT_CHARS - len(said) - 4096


async def tell_what_it_came_to(request: Request) -> None:
    """Say in the conversation what a request came to: her words on the facts of it, and what was found or made."""
    words = await _her_words(request)
    message = the_message(request, words)
    _shown(message, request)
    from core.conversation.said_unasked import she_said_unasked

    await she_said_unasked(message, origin=ORIGIN)


def the_message(request: Request, words: str) -> str:
    """Her words where they hold to the facts, the facts where they do not; and what was asked for, whole."""
    account = request.account()
    body = words if words and _holds_to(words, account, request) else "\n".join(account)
    deliverable = str(request.made.get("told") or "").strip()
    return f"{body}\n\n{deliverable}" if deliverable else body


def _holds_to(words: str, account: list[str], request: Request) -> bool:
    """Whether her words name no place or page that is not among what was made, found or done."""
    import re

    known = "\n".join([*account, str(request.made), request.whole])
    named = re.findall(r"https?://[^\s)\]>\"']+|(?:~|/Users|/Applications)/[^\s)\]>\"'`]+", words)
    strange = [name for name in named if name.rstrip(".,;:!?") not in known]
    if strange:
        logger.info("🧰 Her words on request %s name what was neither made nor found: %s", request.id, strange[:3])
    return not strange


async def _her_words(request: Request) -> str:
    from core.brain.her_unprompted_turn import (
        her_conversation_now_with,
        on_her_cortex,
        what_she_finished,
    )
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model

    facts = json.dumps({"what_it_came_to": request.account(), **({"found": request.made["told"]} if request.made.get("told") else {})},
                       ensure_ascii=False)
    try:
        while a_person_is_waiting_for_her_model():
            await asyncio.sleep(1.0)
        messages = her_conversation_now_with("", facts, looked_with=(ORIGIN, {"request": request.said}))
        return what_she_finished(await on_her_cortex(messages, origin=ORIGIN, least_s=LEAST_S))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
        record_degradation(ORIGIN, exc, severity="info", action="what a request came to was told as its facts")
        return ""


def _shown(message: str, request: Request) -> None:
    """Put the message in the conversation window."""
    from core.runtime.service_access import resolve_orchestrator

    orchestrator = resolve_orchestrator()
    publish = getattr(orchestrator, "_publish_telemetry", None) if orchestrator else None
    if publish is None:
        logger.warning("🧰 What request %s came to could not be shown: no window to show it in.", request.id)
        return
    publish({"type": "aura_message", "message": message,
             "metadata": {"work_asked_for": request.id, "stands": request.stands}})
