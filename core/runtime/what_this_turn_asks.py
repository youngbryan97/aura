"""What the turn being answered was read to ask for, where every door that could act on it can see it.

About thirty places used to decide for themselves whether a sentence asked for something, from the words in it, and each
acted when it thought so. The reading of a turn is taken once (core/understanding) and held here, below everything that
acts. A door that asks about this turn's words is told what was read; a door that asks about other words, or in a turn
nothing was read for, is told nothing and does what it did before.

Held twice: in the context of the turn, and by the words, since some doors run on threads of their own and some are
handed the words alone.
"""
from __future__ import annotations

import contextvars
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["Asked", "asked_by", "clear", "hold", "licensed", "this_turn"]

#: How long a reading answers for its words outside the context of its turn: as long as a turn can run.
HELD_FOR_S = 3600.0
#: How many are held by their words.
HELD = 64


@dataclass(frozen=True)
class Asked:
    """What one turn asks for."""

    said: str
    #: The kind of thing to be brought about (core/understanding/effects.py), or ``talk``.
    effect: str
    #: ``act``, ``ask_first`` or ``talk``: what is to be done about it.
    decision: str
    #: How sure, of the kind and of its asking for anything at all.
    sure: float = 0.0
    asking: float = 0.0
    by_effect: dict[str, float] = field(default_factory=dict, hash=False, compare=False)
    at: float = 0.0
    #: The request this turn belongs to, once one is open for it.
    request_id: str = ""

    @property
    def acts(self) -> bool:
        return self.decision == "act"


_THIS_TURN: contextvars.ContextVar[Asked | None] = contextvars.ContextVar("aura_what_this_turn_asks", default=None)
_BY_WORDS: OrderedDict[str, Asked] = OrderedDict()
_LOCK = checked_lock("runtime.what_this_turn_asks")


def _words(text: Any) -> str:
    return " ".join(str(text or "").split())


def hold(asked: Asked) -> None:
    """Hold this reading for its turn and for its words."""
    _THIS_TURN.set(asked)
    key = _words(asked.said)
    if not key:
        return
    with _LOCK:
        _BY_WORDS.pop(key, None)
        _BY_WORDS[key] = asked
        while len(_BY_WORDS) > HELD:
            _BY_WORDS.popitem(last=False)


def clear() -> None:
    """This turn was read for nothing: doors decide as they did before."""
    _THIS_TURN.set(None)


def this_turn() -> Asked | None:
    return _THIS_TURN.get()


def asked_by(text: Any) -> Asked | None:
    """What was read of these words, where they are a turn's own words or a part of them."""
    key = _words(text)
    if not key:
        return None
    current = _THIS_TURN.get()
    if current is not None:
        said = _words(current.said)
        # A part of what was said is what was said, and so is a composed prompt that holds it. A turn of a word or two
        # is found by accident in anything, so it counts only at either end.
        if key in said or key.startswith(said + " ") or key.endswith(" " + said):
            return current
        if len(said.split()) >= 4 and said in key:
            return current
    with _LOCK:
        held = _BY_WORDS.get(key)
    if held is not None and time.time() - held.at <= HELD_FOR_S:
        return held
    return None


def licensed(text: Any, *effects: str) -> bool | None:
    """Whether these words were read as asking for one of these effects now; None where they were not read."""
    asked = asked_by(text)
    if asked is None:
        return None
    return asked.acts and asked.effect in effects
