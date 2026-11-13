"""Her turn as her model was asked it, kept so that a draft her checks refused is asked again exactly as it was.

Her stabilizer used to write a repair turn of its own: "You are Aura. This is a user-facing repair turn." with how
to sound and what not to say, put to her model in place of the turn. That is instruction on the path to her words
(F06; Bryan's rule, no prompt engineering, ever). LIVE 11 November 2026, on a typed follow-up, it ran a 30 s chain of
rewrites and its draft was withheld anyway. Asked again, her model gets the messages her gate dispatched for the
person's turn and nothing else; the checks that refused the first draft judge the second.

The chat route binds one holder per turn (`a_turn_is_asked`); her inference gate fills it once, at the dispatch of
the person's turn, so the dispatch of a turn asked again never overwrites it.

The checks her reply passes through also keep here what the record says that her draft contradicted, as readings
(`the_record_says`): a file that is not on disk, a sense with no reading, a number no channel produced. They used to
append that to her reply in her first person ("I cut a sentence there — I had written ...", "Correction on one thing I
just said: ..."), runtime text she never said; LIVE 12 November 2026 Bryan saw it as the leak it is. Asked again, her
model reads the record among the turn's readings, and what she says of it is hers.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_HOLDER: ContextVar[dict[str, Any] | None] = ContextVar("aura_her_turn_as_asked", default=None)


@contextmanager
def a_turn_is_asked() -> Iterator[None]:
    """One person's turn: what her model is asked for it can be kept."""
    token = _HOLDER.set({})
    try:
        yield
    finally:
        _HOLDER.reset(token)


def keep(*, prompt: Any, messages: Any, system_prompt: Any, max_tokens: Any) -> None:
    """The person's turn as her gate dispatched it. Only the first dispatch of a turn is kept."""
    holder = _HOLDER.get()
    if holder is None or "messages" in holder or not isinstance(messages, list) or not messages:
        return
    holder.update(prompt=str(prompt or ""), messages=copy.deepcopy(messages),
                  system_prompt=str(system_prompt or ""), max_tokens=max_tokens)


def as_asked() -> dict[str, Any] | None:
    """The person's turn as her model was asked it, or None when none was kept. With anything the record said since,
    as a reading at the head of the person's message."""
    holder = _HOLDER.get()
    if not holder or "messages" not in holder:
        return None
    messages = copy.deepcopy(holder["messages"])
    record = holder.get("record") or []
    if record:
        reading = "[WHAT THE RECORD SAYS]\n" + "\n".join(f"- {fact}" for fact in record) + "\n[END WHAT THE RECORD SAYS]\n\n"
        for message in reversed(messages):
            if isinstance(message, dict) and str(message.get("role") or "").lower() == "user":
                message["content"] = reading + str(message.get("content") or "")
                break
    return {key: value for key, value in holder.items() if key != "record"} | {"messages": messages}


def the_record_says(fact: str) -> None:
    """A fact the runtime holds that her draft contradicted, for her turn asked again. Written as a reading."""
    holder = _HOLDER.get()
    fact = " ".join(str(fact or "").split())
    if holder is None or not fact:
        return
    record = holder.setdefault("record", [])
    if fact not in record:
        record.append(fact)


def what_the_record_says() -> list[str]:
    """What the record said against her draft on this turn, so far."""
    holder = _HOLDER.get()
    return list(holder.get("record") or []) if holder else []


def clear_the_record() -> None:
    """A new draft is checked from nothing."""
    holder = _HOLDER.get()
    if holder is not None:
        holder.pop("record", None)
