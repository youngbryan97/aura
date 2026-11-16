"""What she says without being asked, kept with the exchange it follows: in memory, on disk, and read back.

Lifted out of interface/routes/chat_memory_state.py, which is over its size ceiling, when the keeper learned to fall
back to the last exchange on disk after a restart (16 November 2026). chat_memory_state imports these back and
registers the keeper with core/conversation/said_unasked.py as before; the conversation they read stays there.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from interface.routes.chat_common import _CHAT_RECOVERABLE_ERRORS

__all__ = ["_keep_what_she_said_unasked", "_last_said_at", "_said_after"]


async def _keep_what_she_said_unasked(words: str, origin: str, at: float) -> bool:
    """Keep a remark of hers with the exchange it follows: the last one complete, as the chat window shows it.

    Kept in memory as that exchange's ``said_after``, and on disk as her turn with the cid
    ``<exchange id>:after:<n>``, so a restart reads it back where it was said (core/conversation/said_unasked.py).
    """
    from interface.routes import chat_memory_state as state

    async with state._get_convo_lock():
        target = next(
            (
                entry
                for entry in reversed(state._conversation_log)
                if str(entry.get("status") or "complete").strip().lower() == "complete"
            ),
            None,
        )
        if target is not None:
            said_after = target.setdefault("said_after", [])
            said_after.append({"words": words, "at": float(at), "origin": origin})
            nth = len(said_after)
            exchange_id = str(target.get("id") or "")[:64]
            session_id = str(target.get("session_id") or "")[:64]
            principal_id = str(target.get("principal_id") or "")
            principal_surface = str(target.get("principal_surface") or "")
    if target is None:
        # Nothing is in memory after a restart until someone takes a turn; the last exchange on disk is where her
        # window shows what follows it. LIVE 16 November 2026, 08:32 local: "said unasked, and no conversation kept
        # it", and the companion panel opened on nothing. Read as the owner, whose desk it is.
        surface_token = state._CHAT_REQUEST_SURFACE.set("owner")
        try:
            durable = await state._load_durable_conversation_exchanges(limit=1)
        finally:
            state._CHAT_REQUEST_SURFACE.reset(surface_token)
        if not durable:
            return False
        last = durable[-1]
        nth = len(list(last.get("said_after") or [])) + 1
        exchange_id = str(last.get("exchange_id") or "")[:64]
        session_id = str(last.get("session_id") or "")[:64]
        principal_id = principal_surface = ""
    persistence = ServiceContainer.get("persistence", default=None)
    record_turn = getattr(persistence, "record_turn", None)
    if exchange_id and callable(record_turn):
        scope = (
            {"principal_id": principal_id, "principal_surface": principal_surface}
            if principal_id and principal_surface
            else {}
        )
        try:
            await asyncio.to_thread(
                record_turn,
                "aura",
                words,
                origin=origin,
                cid=f"{exchange_id}:after:{nth}",
                session_id=session_id or None,
                metadata={"unasked": True},
                **scope,
            )
        except _CHAT_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "chat.conversation_persistence",
                exc,
                severity="warning",
                action="a remark of hers was kept in memory and not on disk",
            )
    return True


def _last_said_at(entry: dict[str, Any]) -> float:
    """When she last said something unasked after an exchange: where it last had anybody's attention."""
    return max(
        (float(item.get("at") or 0.0) for item in list(entry.get("said_after") or []) if isinstance(item, dict)),
        default=0.0,
    )


def _said_after(entry: dict[str, Any]) -> list[str]:
    """What she said unasked after an exchange, in the order she said it."""
    return [
        str(item.get("words") or "").strip()
        for item in list(entry.get("said_after") or [])
        if isinstance(item, dict) and str(item.get("words") or "").strip()
    ]
