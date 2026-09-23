"""Whether the work running here is still getting somewhere.

A timeout answers "has this taken too long". The question worth asking about
long work is different: has it stopped. LIVE 2026-09-20, at 09:27: a game she
had been asked to play until the 2048 tile was killed an hour in —
"Task computer_use timed out" — while it was landing a move every two
seconds and had climbed 32, 64, 128, 256. Nothing was wrong with it except
the clock.

So the work says when it gets somewhere, and whoever is holding a deadline
over it can ask. A slot in the context rather than a global: the reporter and
the waiter are the same asyncio task, the waiter makes the slot before it
awaits, and work that reports into no slot says so by returning False rather
than by failing.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

__all__ = [
    "a_place_to_report_it",
    "it_got_somewhere",
    "when_it_last_got_somewhere",
]

_GETTING_SOMEWHERE: ContextVar[dict[str, Any] | None] = ContextVar(
    "aura_still_getting_somewhere", default=None
)


def a_place_to_report_it() -> dict[str, Any]:
    """Give the work about to start here somewhere to say it is moving."""
    slot: dict[str, Any] = {"at": time.monotonic(), "note": "", "times": 0}
    _GETTING_SOMEWHERE.set(slot)
    return slot


def it_got_somewhere(note: str = "") -> bool:
    """Say the work running here just moved. False where nobody is listening."""
    slot = _GETTING_SOMEWHERE.get()
    if slot is None:
        return False
    slot["at"] = time.monotonic()
    slot["times"] = int(slot.get("times", 0)) + 1
    if note:
        slot["note"] = " ".join(str(note).split())[:160]
    return True


def when_it_last_got_somewhere(slot: dict[str, Any] | None) -> float:
    """How long ago, in seconds, that slot last heard anything. Infinite for never."""
    if not slot or not slot.get("times"):
        return float("inf")
    return max(0.0, time.monotonic() - float(slot.get("at") or 0.0))
