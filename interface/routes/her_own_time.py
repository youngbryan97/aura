"""Her own time, joined to what does each thing she could do with it, and seen.

What she chooses among and what came of it is core/agency/her_own_time.py and core/agency/what_she_does_next.py. Here each
thing is given what does it: the engine that carries out a request, her search, and her word in a conversation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation
from interface.auth import _require_internal

logger = logging.getLogger("Aura.HerOwnTime")
router = APIRouter(tags=["her own time"])

__all__ = ["begin_what_she_does_unasked", "router", "somebody_spoke"]

#: What names what she says and does on her own time.
ORIGIN = "her_own_time"
#: How long since somebody last spoke counts as half as near, in seconds: the length of a pause in a conversation.
HALF_AS_NEAR_S = 600.0
#: How long her request of her own work is waited for, before what it has come to so far is what it came to, in seconds.
CARRIED_S = 1700.0


async def _carry_on(request: Any) -> None:
    """Go on with a request, until it has come to what it comes to."""
    from core.skills.screen_pursuit import _tell
    from core.understanding import carrying_out
    from interface.routes.chat_carrying_out import ask_them, tell_what_it_came_to

    if not carrying_out.begin(request, tell=_tell, told=tell_what_it_came_to, asks=ask_them):
        return
    ends = time.monotonic() + CARRIED_S
    while carrying_out.is_under_way(request) and time.monotonic() < ends:
        await asyncio.sleep(2.0)


async def _search(words: str) -> dict[str, Any]:
    from core.capability_engine import execute_tool

    result = await execute_tool("web_search", {"query": words}, context={"origin": "curiosity", "source": ORIGIN, "objective": f"find out about {words}", "autonomous": True})
    return result if isinstance(result, dict) else {"ok": False, "error": "no answer"}


def _here() -> float:
    """How near somebody is to talking with her: one as they speak, half after a pause, and less as it goes on."""
    from core.runtime.background_policy import _last_user_interaction_time

    last = _last_user_interaction_time(None)
    return 0.0 if last <= 0.0 else 0.5 ** (max(0.0, time.time() - last) / HALF_AS_NEAR_S)


async def _tell(find: dict[str, Any]) -> bool:
    """Say something she found to whoever is here, in her words, where her Will says it may be said; whether it was said."""
    from core.brain.her_unprompted_turn import (
        her_conversation_now_with,
        on_her_cortex,
        what_she_finished,
    )
    from core.conversation.said_unasked import she_said_unasked
    from core.runtime.service_access import resolve_orchestrator
    from core.understanding.her_model_between_turns import nobody_waiting

    orchestrator = resolve_orchestrator()
    if orchestrator is None:
        return False
    reading = json.dumps({"what_she_found": find["what"], "about": find["about"], "where": find.get("where") or ""}, ensure_ascii=False)
    await nobody_waiting()
    messages = her_conversation_now_with("", reading, looked_with=("web_search", {"query": find["about"]}))
    words = what_she_finished(await on_her_cortex(messages, origin=ORIGIN, least_s=60.0))
    if not words:
        return False
    decision = await orchestrator.emit_spontaneous_message(words, origin=ORIGIN, urgency=0.6)
    if not isinstance(decision, dict) or not decision.get("ok"):
        logger.info("🧭 What she found was not said: %s", (decision or {}).get("reason") if isinstance(decision, dict) else decision)
        return False
    await she_said_unasked(words, origin=ORIGIN)
    return True


async def _her_will_says(option: Any) -> bool:
    """Whether what she chose may be done, as every act she begins herself is asked."""
    from core.constitution import get_constitutional_core
    from core.runtime.service_access import resolve_orchestrator

    try:
        allowed, reason, _decision = await get_constitutional_core(resolve_orchestrator()).approve_initiative(str(option.what)[:240], source=ORIGIN, urgency=0.6)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(ORIGIN, exc, severity="warning", action="what she chose was not done, for it could not be asked whether it may be")
        return False
    if not allowed:
        logger.info("🧭 Not let: %s (%s)", str(option.what)[:120], reason)
    return bool(allowed)


def somebody_spoke() -> None:
    """Somebody spoke to her: what that says of what she told them is learned."""
    from core.agency.her_own_time import get_her_own_time

    try:
        get_her_own_time().somebody_spoke()
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(ORIGIN, exc, severity="debug", action="what somebody's answer says of what she told them was not learned")


def begin_what_she_does_unasked() -> None:
    """Join her own time to what does each thing, and go on with what she was in the middle of for somebody when she was closed."""
    from core.agency.her_own_time import get_her_own_time
    from core.agency.what_she_could_do import begin_her_own_time
    from core.skills.screen_pursuit import _tell as tell_them
    from core.understanding.carrying_out import take_up_what_was_left
    from core.utils.task_tracker import get_task_tracker
    from interface.routes.chat_carrying_out import ask_them, tell_what_it_came_to
    from interface.routes.chat_what_is_asked import her_client

    from core.container import ServiceContainer

    hers = begin_her_own_time(get_her_own_time(), carry_on=_carry_on, search=_search, tell=_tell, here=_here)
    hers.may = _her_will_says
    # Where the runtime's own loop, which is under her agency and may not reach up to it, finds it.
    ServiceContainer.register_instance("her_own_time", hers)

    async def go_on() -> None:
        taken = await take_up_what_was_left(tell=tell_them, told=tell_what_it_came_to, asks=ask_them, ready=lambda: her_client() is not None)
        if taken:
            logger.info("🧰 Taken up again after a restart: %s", "; ".join(r.said[:80] for r in taken))

    get_task_tracker().create_task(go_on(), name="requests.left_off")


@router.get("/her_own_time")
async def her_own_time(_: None = Depends(_require_internal)) -> JSONResponse:
    """What she is doing on her own time, what she chose it over, and what she has learned of what is worth doing."""
    from core.agency.her_own_time import get_her_own_time

    hers = get_her_own_time()
    learned = await asyncio.to_thread(hers.chooser.how_she_chooses)
    choices = await asyncio.to_thread(hers.chooser.choices, 30)
    return JSONResponse({
        "doing": hers.doing, "before": hers.before, "found": len(hers.found), "learned": learned,
        "choices": [{"at": c.at, "what": c.what, "kind": c.kind, "why": c.says(), "over": c.over[:4], "came": c.came.get("what", ""),
                     "reward": None if c.reward is None or math.isnan(c.reward) else round(c.reward, 3)} for c in reversed(choices)]})
