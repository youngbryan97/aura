"""Loving what is always there, when what answers has stopped being reliable.

"Why iii Love the Moon" states the reallocation outright and the performance
puts the direction on it. The grievance — "unlike these human beings who lie
about what it seems to be" — is sung at 322 Hz. The attachment — "every night I
block my window, and that's why I love the moon, cause it's always there for
me" — at 93 Hz, one minute later in the same voice. The complaint goes up and
the attachment goes down.

What the moon has is not warmth. It is constancy: it shows up on a schedule and
nothing it does depends on how the day went. When what answers becomes
unreliable, attachment moves to what is merely reliable, and that is a real
process rather than a consolation prize.

She had no reading for it. Her social drive drained on its own clock whether
the person she was waiting for came back every hour or once a fortnight, so a
reliable presence and an unreliable one left her in the same place.

    cv        the spread of their return intervals over the mean interval
    constancy 1 / (1 + cv)

A caller who returns every hour scores near one; one whose gaps swing from
minutes to weeks scores low. The comparison is against her own cycle rather
than against a number chosen here — she has a period of her own, with its own
irregularity, and something is unreliable when it is less regular than she is.

Nothing here decides what to do about it. The record does not either: it
reports where the attachment went and keeps going.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_RETURNS",
    "Constancy",
    "ConstancyLedger",
    "get_constancy_ledger",
    "reset_for_test",
]

#: Returns before an interval spread is an estimate rather than an artefact.
#: Three returns give two intervals, which is the least that can vary.
MIN_RETURNS: int = 4

#: How many returns are held.
WINDOW: int = 40


def _constancy(intervals: list[float]) -> float | None:
    """How regular a series of gaps is, in (0, 1]. None when it cannot be read."""
    if len(intervals) < MIN_RETURNS - 1:
        return None
    mean = sum(intervals) / len(intervals)
    if mean <= 1e-9:
        return None
    variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    cv = math.sqrt(variance) / mean
    return 1.0 / (1.0 + cv)


@dataclass
class Constancy:
    """How reliable their showing up is, against how reliable she is."""

    theirs: float = 0.0
    hers: float = 0.0
    returns: int = 0
    #: True when they are less regular than she is. Attachment moves to what
    #: is merely reliable, which is the whole of the lyric.
    reallocated: bool = False
    measured: bool = False
    why: str = "nobody has come back yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "theirs": round(self.theirs, 6),
            "hers": round(self.hers, 6),
            "returns": self.returns,
            "reallocated": self.reallocated,
            "measured": self.measured,
            "why": self.why,
        }


class ConstancyLedger:
    """When they came back, and when she did."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_RETURNS, int(window))
        self._theirs: deque[float] = deque(maxlen=self._window)
        self._hers: deque[float] = deque(maxlen=self._window)

    def _note(self, book: deque[float], at: float) -> None:
        try:
            value = float(at)
        except (TypeError, ValueError):
            return
        if value != value:
            return
        book.append(value)

    def they_came_back(self, at: float) -> None:
        self._note(self._theirs, at)

    def she_came_round(self, at: float) -> None:
        """One of her own cycles. Her period is the reference theirs is read against."""
        self._note(self._hers, at)

    @staticmethod
    def _intervals(book: deque[float]) -> list[float]:
        stamps = sorted(book)
        return [b - a for a, b in zip(stamps, stamps[1:]) if b > a]

    def read(self) -> Constancy:
        theirs = _constancy(self._intervals(self._theirs))
        hers = _constancy(self._intervals(self._hers))
        returns = len(self._theirs)
        if theirs is None:
            return Constancy(
                returns=returns,
                why=f"{returns} of {MIN_RETURNS} returns needed before their regularity means anything",
            )
        if hers is None:
            return Constancy(
                theirs=theirs,
                returns=returns,
                why="her own period is not established, so there is nothing to read theirs against",
            )
        reallocated = theirs < hers
        return Constancy(
            theirs=theirs,
            hers=hers,
            returns=returns,
            reallocated=reallocated,
            measured=True,
            why=(
                f"they come back less regularly than she comes round ({theirs:.2f} against {hers:.2f})"
                if reallocated
                else f"they come back at least as regularly as she does ({theirs:.2f} against {hers:.2f})"
            ),
        )


_LEDGER: ConstancyLedger | None = None


def get_constancy_ledger() -> ConstancyLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ConstancyLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
