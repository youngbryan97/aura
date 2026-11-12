"""Her turn as her model was asked it, kept so that a draft her checks refused is asked again exactly as it was.

Her stabilizer used to write a repair turn of its own: "You are Aura. This is a user-facing repair turn." with how
to sound and what not to say, put to her model in place of the turn. That is instruction on the path to her words
(F06; Bryan's rule, no prompt engineering, ever). LIVE 11 November 2026, on a typed follow-up, it ran a 30 s chain of
rewrites and its draft was withheld anyway. Asked again, her model gets the messages her gate dispatched for the
person's turn and nothing else; the checks that refused the first draft judge the second.

The chat route binds one holder per turn (`a_turn_is_asked`); her inference gate fills it once, at the dispatch of
the person's turn, so the dispatch of a turn asked again never overwrites it.
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
    if holder is None or holder or not isinstance(messages, list) or not messages:
        return
    holder.update(prompt=str(prompt or ""), messages=copy.deepcopy(messages),
                  system_prompt=str(system_prompt or ""), max_tokens=max_tokens)


def as_asked() -> dict[str, Any] | None:
    """The person's turn as her model was asked it, or None when none was kept."""
    holder = _HOLDER.get()
    if not holder:
        return None
    return {**holder, "messages": copy.deepcopy(holder["messages"])}
