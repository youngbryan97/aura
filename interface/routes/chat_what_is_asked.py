"""Reading what a turn asks for, once, before any door acts on it.

The reading (core/understanding/what_is_asked.py) is taken at the start of the turn and held where every door sees it
(core/runtime/what_this_turn_asks.py). Where it cannot be taken, because the model is still warming or the fit is not
ready, nothing is held and the doors decide as they did before.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.what_this_turn_asks import Asked, clear, hold
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

logger = logging.getLogger("Aura.Chat")

__all__ = ["begin_reading_what_is_asked", "files_meant", "her_client", "names_of", "read_what_this_turn_asks"]

#: The longest a turn waits to be read. One pass over the exchange takes well under a second on a warm lane; past this
#: the lane is busy with something else and the turn goes on unread.
READ_WITHIN_S = 12.0
#: How many exchanges before the turn it is read beside.
BESIDE = 1


def her_client() -> Any:
    """The client of the model she thinks with, as her turns hold it."""
    gate = ServiceContainer.get("inference_gate", default=None)
    return getattr(gate, "_mlx_client", None)


def begin_reading_what_is_asked() -> None:
    """Fit the reading on the model that is running, while nobody is waiting; asked for again while it is not ready."""
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model
    from core.understanding.what_is_asked import get_what_is_asked
    from core.understanding.what_she_says_she_does import get_what_she_says_she_does
    from core.utils.task_tracker import get_task_tracker

    async def learn() -> None:
        client = her_client()
        if client is not None and await get_what_is_asked().learn(client, waiting=a_person_is_waiting_for_her_model):
            await get_what_she_says_she_does().learn(client, waiting=a_person_is_waiting_for_her_model)

    get_task_tracker().create_task(learn(), name="what_is_asked.learn")


def names_of(attachment_ids: Sequence[str]) -> list[str]:
    """What the files that came with a turn are called."""
    from core.conversation.attachments import kept

    names = []
    for attachment_id in list(attachment_ids)[:8]:
        held = kept(str(attachment_id))
        if held is not None:
            names.append(str(held[0].get("name") or "a file"))
    return names


async def _beside(said: str, session_id: str) -> list[Any]:
    """What was said before this turn, as turns."""
    from core.understanding.examples import Turn
    from interface.routes import chat_memory_state

    recent = await chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message=said, session_id=session_id, limit=BESIDE, allow_cross_session=False,
    )
    turns: list[Turn] = []
    for exchange in recent[-BESIDE:]:
        for role, key in (("user", "user"), ("assistant", "aura")):
            words = str(exchange.get(key) or "").strip()
            if words:
                turns.append(Turn(role, words))
    return turns


def files_meant(asked: Asked | None, session_id: str, attachments: Sequence[str]) -> list[str]:
    """The files a turn is about: what came with it, and for a turn read as about a document that brought none, the ones
    last given in the conversation. LIVE 23 November 2026 a paper was attached to one message and asked about in the
    next, and the next was answered without it."""
    given = [str(item) for item in attachments]
    if given or asked is None or not asked.acts or asked.effect != "document_read":
        return given
    from core.understanding.requests_of_her import get_requests_of_her

    return get_requests_of_her().given(session_id)


async def read_what_this_turn_asks(said: str, *, session_id: str, attachments: Sequence[str] = (),
                                   skip: bool = False) -> Asked | None:
    """What this turn asks for, held for every door; None where it was not read."""
    clear()
    if skip or not str(said or "").strip():
        return None
    try:
        from core.understanding.examples import attached
        from core.understanding.what_is_asked import get_what_is_asked

        client, reads = her_client(), get_what_is_asked()
        if client is None:
            return None
        if not reads.ready(client):
            begin_reading_what_is_asked()
            return None
        names = await asyncio.to_thread(names_of, attachments) if attachments else []
        asked = await asyncio.wait_for(
            reads.read(attached(said, names), await _beside(said, session_id), client=client), timeout=READ_WITHIN_S,
        )
    except TimeoutError:
        logger.info("🧭 This turn was not read for what it asks within %.0f s.", READ_WITHIN_S)
        return None
    except _CHAT_RECOVERABLE_ERRORS as exc:
        record_degradation("what_is_asked", exc, severity="warning", action="this turn was not read for what it asks")
        return None
    if asked is None:
        return None
    # Held by the person's own words, which is what the doors ask by.
    asked = dataclasses.replace(asked, said=said)
    hold(asked)
    return asked
