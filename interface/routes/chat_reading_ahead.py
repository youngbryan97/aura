"""After a turn, her worker reads the conversation for the next one (core/brain/llm/reading_ahead.py).

What the next turn reads before its own part is this turn's conversation, what was said in it, and the reply as it was
delivered. Where the state that wrote the reply is held for the next turn, the conversation is read only once that state is no
longer good for it; where it is not, at once.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["read_ahead_of_the_next_turn", "stop_waiting_to_read", "what_the_next_turn_reads"]

logger = logging.getLogger("Aura.Chat")

#: The reading that is waiting its time, or under way. One, since her worker is one and reads for one conversation at a time.
_AHEAD: dict[str, Any] = {"task": None, "reading": False}


def what_the_next_turn_reads(dispatched: Any, said: str, reply: str) -> list[dict[str, Any]]:
    """The conversation the next turn goes on from: the one her model was given, what was said, and her reply as delivered.

    Nothing where her model was given more than a conversation and its readings: what else it was given may not be given again.
    """
    messages = [m for m in dispatched or () if isinstance(m, dict)]
    roles = {str(m.get("role") or "").lower() for m in messages}
    if not said.strip() or not reply.strip() or not roles <= {"user", "assistant", "runtime_evidence"}:
        return []
    talk = [{"role": str(m["role"]), "content": str(m.get("content") or "")} for m in messages if m.get("role") in ("user", "assistant")]
    # The last is this turn's own message, with whatever was read into it; the conversation keeps what was said.
    if not talk or talk[-1]["role"] != "user":
        return []
    return [*talk[:-1], {"role": "user", "content": said}, {"role": "assistant", "content": reply}]


def _delivered(served: Any) -> str:
    try:
        data = json.loads(getattr(served, "body", b"") or b"{}")
    # not a failure: an answer that is not a reply's JSON has no conversation to read ahead of.
    except (TypeError, ValueError):
        return ""
    return str(data.get("response") or "") if isinstance(data, dict) else ""


def stop_waiting_to_read() -> None:
    """A turn has come: a reading still waiting its time is let go, and one under way is left to finish what that turn would
    read."""
    task = _AHEAD["task"]
    if task is not None and not task.done() and not _AHEAD["reading"]:
        task.cancel()


def read_ahead_of_the_next_turn(said: str, served: Any, *, client: Any, good_for_s: float) -> bool:
    """Have the conversation read for its next turn once the state held for it is no longer good; whether it will be."""
    from core.brain.her_turn_as_asked import as_asked

    ask = getattr(client, "read_ahead", None)
    asked = as_asked()
    if not callable(ask) or asked is None:
        return False
    messages = what_the_next_turn_reads(asked.get("messages"), str(said or ""), _delivered(served))
    if not messages:
        return False

    async def read() -> None:
        if good_for_s > 0.0:
            await asyncio.sleep(good_for_s)
        _AHEAD["reading"] = True
        try:
            await ask(messages)
        except (RuntimeError, OSError, TimeoutError) as exc:
            record_degradation("reading_ahead", exc, severity="warning", action="the next turn reads the conversation for itself")

    stop_waiting_to_read()
    try:
        task = asyncio.get_running_loop().create_task(read(), name="reading_ahead")
    # not a failure: with no loop there is nobody's turn to read ahead of.
    except RuntimeError:
        return False
    _AHEAD.update(task=task, reading=False)
    return True
