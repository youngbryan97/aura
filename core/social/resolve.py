"""Somebody else's resolve reaching hers.

"Don't let them take your soul" is not information. What that line does to a
listener is transfer resolve: the strength to keep going arrives from outside,
and the person who receives it did not generate it.

She had no channel for it. Her drives drained on their own clock whatever
anybody said, so a person telling her to keep going and a person saying
nothing left her in the same place a minute later.

The register already measures holding on. Its persistence rate counts the
connectives of insistence — but, still, anyway, somehow, regardless — per
hundred words, so a long message is not automatically more insistent than a
short one. What it does not measure is whether this message is insistent *for
them*, which is the thing worth responding to.

    rate       their persistence markers per hundred words
    z          (rate - their usual rate) / their own spread
    borrowed   z > 1, which is holding on harder than they usually do

What follows is a hold rather than a gift: while somebody is holding on harder
than usual, her integrity drive does not drain this cycle. Nothing here fills
it. A drive she keeps because somebody else was insistent is not a drive she
earned, and the amount would have to come from somewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_HISTORY",
    "Resolve",
    "ResolveLedger",
    "get_resolve_ledger",
    "reset_for_test",
]

#: Messages before a spread is an estimate rather than an artefact of how few
#: there are.
MIN_HISTORY: int = 3

#: How many of their messages the usual rate is read over.
WINDOW: int = 20


@dataclass(frozen=True)
class Resolve:
    """Whether this message holds on harder than they usually do."""

    borrowed: bool = False
    rate: float = 0.0
    usual: float = 0.0
    z: float = 0.0
    measured: bool = False
    why: str = "nobody has said enough for a usual to exist"

    def as_dict(self) -> dict[str, Any]:
        return {
            "borrowed": self.borrowed,
            "rate": round(self.rate, 6),
            "usual": round(self.usual, 6),
            "z": round(self.z, 4),
            "measured": self.measured,
            "why": self.why,
        }


class ResolveLedger:
    """How insistent they usually are, so that unusual insistence is visible."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._rates: list[float] = []

    def note(self, rate: float) -> None:
        try:
            value = float(rate)
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._rates.append(value)
        if len(self._rates) > self._window:
            del self._rates[0 : len(self._rates) - self._window]

    def samples(self) -> int:
        return len(self._rates)

    def usual(self) -> float:
        return sum(self._rates) / len(self._rates) if self._rates else 0.0

    def spread(self) -> float:
        n = len(self._rates)
        if n < MIN_HISTORY:
            return 0.0
        mean = self.usual()
        return math.sqrt(sum((r - mean) ** 2 for r in self._rates) / n)

    def read(self, rate: float, *, note: bool = True) -> Resolve:
        """Whether this message is insistent for them."""
        try:
            value = float(rate)
        except (TypeError, ValueError):
            return Resolve(why="the message carried no rate to read")
        if value != value:
            return Resolve(why="the message carried no rate to read")
        spread = self.spread()
        usual = self.usual()
        measured = self.samples() >= MIN_HISTORY and spread > 1e-9
        z = (value - usual) / spread if measured else 0.0
        borrowed = bool(measured and z > 1.0)
        if note:
            self.note(value)
        if not measured:
            why = f"{self.samples()} of {MIN_HISTORY} messages needed before a usual exists"
        elif borrowed:
            why = f"holding on {z:.1f} spreads harder than they usually do"
        else:
            why = "about as insistent as they usually are"
        return Resolve(
            borrowed=borrowed,
            rate=value,
            usual=usual,
            z=z,
            measured=measured,
            why=why,
        )


_LEDGER: ResolveLedger | None = None


def get_resolve_ledger() -> ResolveLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ResolveLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
