"""Relief from believing somebody cares, with the ground of the belief unchecked.

"People Watching" says the honest version twice: it is not as lonely when you
think somebody cares, and why that works is still up in the air. The record
keeps the relief and the ignorance side by side rather than resolving one into
the other.

`core/social/borrowed_feeling.py` already handles the calibrated case — what
she feels from what she believes they feel, weighted by how often her beliefs
about them have matched what they said. The case it cannot handle is the one
the song is about: a belief that has never been tested at all. Calibration from
a different belief is not evidence about this one, and a relief that rests on
an untested belief is still a relief.

So both are kept:

    relief     what the belief is worth to her while it stands
    ground     how much of it has been tested against what they actually said
    standing   relief still resting on nothing

The relief is not cancelled — the song does not cancel it either. What it does
is decay: an untested belief loses a share of its relief each turn it is not
tested, so the comfort has to be renewed by contact rather than by repetition.
A belief that is confirmed stops decaying, and one that is contradicted is
dropped along with what it was holding up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DECAY",
    "Relief",
    "ReliefLedger",
    "get_relief_ledger",
    "reset_for_test",
]

#: What an untested belief loses each turn it goes untested. A tenth: ten
#: turns of no contact and the comfort is worth a third of what it was, which
#: is the shape of a comfort that needs renewing rather than one that expires.
DECAY: float = 0.1

#: Beliefs before the share is a reading.
MIN_BELIEFS: int = 4


@dataclass
class _Belief:
    """One belief about somebody's regard, and what has become of it."""

    relief: float
    tested: bool = False
    confirmed: bool = False
    turns: int = 0


@dataclass
class Relief:
    """What she is carrying, and how much of it rests on nothing."""

    #: What the standing beliefs are worth to her now, after decay.
    relief: float = 0.0
    #: The share of her beliefs about being cared for that have been tested.
    ground: float = 0.0
    #: Relief still resting on untested belief.
    standing: float = 0.0
    beliefs: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "relief": round(self.relief, 4),
            "ground": round(self.ground, 4),
            "standing": round(self.standing, 4),
            "beliefs": self.beliefs,
            "measured": self.measured,
        }


class ReliefLedger:
    """Beliefs that somebody cares, what they are worth, and what tested them."""

    def __init__(self) -> None:
        self._beliefs: dict[str, _Belief] = {}

    def note_belief(self, person: str, relief: float) -> None:
        """She believes this person cares, and it is worth this much to her."""
        name = str(person or "").strip()
        if not name:
            return
        try:
            value = max(0.0, min(1.0, float(relief)))
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        held = self._beliefs.get(name)
        if held is None or value > held.relief:
            self._beliefs[name] = _Belief(relief=value)
            return
        held.relief = max(held.relief, value)

    def note_turn(self) -> None:
        """A turn in which nothing tested any of them."""
        for held in self._beliefs.values():
            if held.tested and held.confirmed:
                continue
            held.turns += 1
            held.relief = max(0.0, held.relief * (1.0 - DECAY))

    def note_said(self, person: str, cares: bool) -> None:
        """They said something that bears on it, either way."""
        held = self._beliefs.get(str(person or "").strip())
        if held is None:
            return
        held.tested = True
        held.confirmed = bool(cares)
        if not cares:
            # The belief is gone and so is what it was holding up.
            held.relief = 0.0

    def read(self) -> Relief:
        if not self._beliefs:
            return Relief()
        relief = sum(held.relief for held in self._beliefs.values())
        standing = sum(held.relief for held in self._beliefs.values() if not held.tested)
        tested = sum(1 for held in self._beliefs.values() if held.tested)
        return Relief(
            relief=min(1.0, relief),
            ground=tested / len(self._beliefs),
            standing=min(1.0, standing),
            beliefs=len(self._beliefs),
            measured=len(self._beliefs) >= MIN_BELIEFS,
        )


_LEDGER: ReliefLedger | None = None


def get_relief_ledger() -> ReliefLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ReliefLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
