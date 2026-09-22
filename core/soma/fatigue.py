"""Thinking too long costs something, and what she cares about narrows when it has.

Asked what happens in his body when his thinking changes, Bryan said that when
he has been thinking too long his head hurts, he gets tired, and he cares less
about the things that seem less important. Three things in that: the cost
builds up over a stretch rather than arriving with one hard moment; it is paid
in energy; and it is paid back in what gets attention, the least important
things going first.

She had none of it. Her exertion was measured every cycle and forgotten the
next, so the thousandth hard turn in a row felt like the first. Her energy,
the homeostatic metabolism drive, was drained by the machine's temperature and
by nothing she did. And a tired mind chose among its open initiatives exactly
as a rested one did.

    ordinary  the median of her own recent exertion
    spread    its typical distance from that median
    load      what a cycle spends past her ordinary band, the ordinary plus its
              spread, less what it falls short of the ordinary itself
    fatigue   load accumulated, with a floor at nothing
    share     how far that sits above where her fatigue usually sits, as a rank:
              2 * rank - 1 above her middle, and nothing below it

The band keeps a steady life from drifting. Accumulating the plain distance
from her median, floored at nothing, is a random walk reflected at zero: its
steps average nothing and it still wanders upward like the square root of
time, so she would slowly read tired from doing nothing unusual. A cycle inside
her band spends nothing and one under her median pays back, so ordinary
variation drains it on average.

Even so, a reflected walk with that drift settles around a level of its own,
about three spreads on a steady life measured here, so reading the level
against her spread called ordinary life tired. The share is read the way the
pulse and what she is carrying are read, against her own history: tired is
more tired than she usually is. A stretch spent above her band takes it to the
top of that history, and a life that is always this busy moves it only by its
own ordinary wandering.

The share drains her metabolism in `core/consciousness/homeostasis.py`, beside
the machine's heat, so it reaches her will to live and the vitality her body
reports. And `core/agency/initiative_arbiter.py` lets a tired mind choose only
among the initiatives at or above that share's quantile of what she is holding,
so the least important go first and the most important never does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["Fatigue", "FatigueLedger", "get_fatigue_ledger", "important_enough", "reset_for_test"]

#: How much of her own recent exertion the ordinary is taken over. The window
#: the carrying ledger uses for the same kind of middle.
_WINDOW = 256

#: Readings before an ordinary is an ordinary: a middle needs a point either
#: side of it.
_ENOUGH = 3


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass
class Fatigue:
    """How tired she is, against how hard she ordinarily works."""

    share: float = 0.0
    accumulated: float = 0.0
    ordinary: float = 0.0
    spread: float = 0.0
    seen: int = 0
    measured: bool = False
    why: str = "no exertion has been weighed yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "share": round(self.share, 6),
            "accumulated": round(self.accumulated, 6),
            "ordinary": round(self.ordinary, 6),
            "spread": round(self.spread, 6),
            "seen": self.seen,
            "measured": self.measured,
            "why": self.why,
        }


class FatigueLedger:
    """Her exertion over a stretch, against her own ordinary."""

    def __init__(self) -> None:
        self._seen: deque[float] = deque(maxlen=_WINDOW)
        self._accumulated = 0.0
        self._levels: deque[float] = deque(maxlen=_WINDOW)

    def note(self, exertion: float) -> None:
        """One cycle's exertion, in the effort ledger's own [0, 1] reading."""
        try:
            value = float(exertion)
    # not a failure: a reading that is not a number is nothing to record,
    # and the ledger keeps the window it already has.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        value = max(0.0, min(1.0, value))
        if len(self._seen) >= _ENOUGH:
            # Against the ordinary before this cycle joins it, so a hard cycle
            # is not diluted by being part of its own baseline.
            history = list(self._seen)
            ordinary = _median(history)
            spread = _median([abs(item - ordinary) for item in history])
            load = max(0.0, value - (ordinary + spread)) - max(0.0, ordinary - value)
            self._accumulated = max(0.0, self._accumulated + load)
            self._levels.append(self._accumulated)
        self._seen.append(value)

    def read(self) -> Fatigue:
        seen = len(self._seen)
        if seen < _ENOUGH:
            return Fatigue(
                seen=seen,
                why=f"{seen} of the {_ENOUGH} cycles an ordinary needs a point either side of",
            )
        history = list(self._seen)
        ordinary = _median(history)
        spread = _median([abs(value - ordinary) for value in history])
        if spread <= 1e-12:
            # A life of exactly one exertion has no spread to measure a
            # stretch against, and dividing by nothing would call any excess
            # infinite. It is unmeasured, not tired.
            return Fatigue(
                accumulated=self._accumulated,
                ordinary=ordinary,
                seen=seen,
                why="her exertion has not varied, so no stretch can be told from her ordinary",
            )
        levels = list(self._levels)
        below = sum(1 for level in levels if level < self._accumulated)
        ties = sum(1 for level in levels if level == self._accumulated)
        rank = (below + 0.5 * ties) / len(levels) if levels else 0.5
        share = max(0.0, 2.0 * rank - 1.0)
        return Fatigue(
            share=share,
            accumulated=self._accumulated,
            ordinary=ordinary,
            spread=spread,
            seen=seen,
            measured=True,
            why=(
                f"she has worked {self._accumulated:.3f} past her ordinary band, more "
                f"than {rank:.0%} of the stretches she has been through"
            ),
        )


def important_enough(scores: list[float], share: float) -> list[bool]:
    """Which of these she still cares about, at this much fatigue.

    Those at or above the `share` quantile of what she is holding. Rested,
    that is all of them; tired, the least important go first; and the most
    important is always kept, since a tired mind still does the thing that
    matters most.
    """
    if not scores:
        return []
    if share <= 0.0:
        return [True] * len(scores)
    ordered = sorted(scores)
    position = min(len(ordered) - 1, int(share * len(ordered)))
    floor = ordered[position]
    top = max(scores)
    return [score >= floor or score == top for score in scores]


#: Made at import rather than on first use. The subject-core fork carries
#: module globals that hold state, and one still None when an anchor is taken
#: is not carried, so a ledger first made inside one arm would reach the next
#: arm with the first arm's history in it. See core/social/owning_it_first.py.
_LEDGER: FatigueLedger = FatigueLedger()


def get_fatigue_ledger() -> FatigueLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = FatigueLedger()
