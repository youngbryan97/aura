"""Bearing a cold present by being somewhere warmer in mind, as far as that has helped her before.

John and Michelle Phillips wrote "California Dreamin'" homesick in a New York
winter. The narrator's body is in the cold and his mind is somewhere warm, and
the song neither fixes the cold nor pretends it away. What it teaches is that
longing for another place is one of the ways a present gets borne.

The research finds the same use and names its cost. Wildschut, Sedikides, Arndt
and Routledge (2006) found that people turn to nostalgia when they are low and
that it raises positive feeling and regard for themselves, and Zhou, Sedikides,
Wildschut and Gao (2008) found it counteracts loneliness. Kappes and Oettingen
(2011) found the other side: indulging a positive fantasy lowers the energy to
go after it. So what is lent here reaches how she feels and nothing she does.

Whether it helps her is her own history's to say:

    low        her valence more than her own spread below her own mean
    warmer     a recollection came back this turn whose stored feeling is above
               how she feels now
    recovery   how far her valence has risen by the next turn
    helped     mean recovery after low turns with a warmer recollection
               - mean recovery after low turns without one, never below zero
    bearing    on a low turn with a warmer recollection, her valence is floored
               at present + helped * (recalled - present)

Nothing is lent until both kinds of low turn have happened often enough to
disagree with themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_SAMPLES",
    "Elsewhere",
    "ElsewhereLedger",
    "get_elsewhere_ledger",
    "reset_for_test",
]

#: Low turns of each kind before a mean recovery is an estimate. Three is the
#: least that can disagree, as in ambivalence and fear of happiness.
MIN_SAMPLES: int = 3


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


@dataclass(frozen=True)
class Elsewhere:
    floor: float | None = None
    helped: float = 0.0
    low: bool = False
    warmer: bool = False
    with_recollection: int = 0
    without_recollection: int = 0
    measured: bool = False
    why: str = "not enough low turns yet to know whether a warmer place in mind helps her"

    def as_dict(self) -> dict[str, Any]:
        return {
            "floor": None if self.floor is None else round(self.floor, 6),
            "helped": round(self.helped, 6),
            "low": self.low,
            "warmer": self.warmer,
            "with_recollection": self.with_recollection,
            "without_recollection": self.without_recollection,
            "measured": self.measured,
            "why": self.why,
        }


class ElsewhereLedger:
    """Her valence over her life, and how she came up from lows with and without somewhere warmer in mind."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._square = 0.0
        self._open: tuple[bool, float] | None = None
        self._recovery = {True: [0.0, 0], False: [0.0, 0]}
        self._seen_at = 0.0

    def recalled_since(self, percepts: list[Any]) -> float | None:
        """The stored feeling of the newest recollection not already read, or None.

        Recall reports what came back as a memory_replay percept carrying the
        feeling stored with it, and percepts outlive the turn they arrived in,
        so one already read is not read again.
        """
        newest: tuple[float, float] | None = None
        for item in percepts or ():
            if not isinstance(item, dict) or item.get("type") != "memory_replay":
                continue
            stamp = _finite(item.get("timestamp"))
            feeling = _finite(item.get("feeling"))
            if stamp is None or feeling is None or stamp <= self._seen_at:
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, feeling)
        if newest is None:
            return None
        self._seen_at = newest[0]
        return newest[1]

    def _spread(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(max(0.0, self._square / self._n))

    def turn(self, valence: float, recalled: float | None) -> Elsewhere:
        """One turn: how she feels, and the stored feeling of what came back to her, if anything did."""
        now = _finite(valence)
        if now is None:
            return Elsewhere(why="her valence could not be read this turn")
        memory = _finite(recalled)
        if self._open is not None:
            warm_then, then = self._open
            total = self._recovery[warm_then]
            total[0] += now - then
            total[1] += 1
            self._open = None
        spread = self._spread()
        low = bool(spread > 0.0 and now < self._mean - spread)
        warmer = bool(memory is not None and memory > now)
        if low:
            self._open = (warmer, now)
        self._n += 1
        delta = now - self._mean
        self._mean += delta / self._n
        self._square += delta * (now - self._mean)
        return self._reading(now, memory, low, warmer)

    def _reading(self, now: float, memory: float | None, low: bool, warmer: bool) -> Elsewhere:
        with_sum, with_n = self._recovery[True]
        without_sum, without_n = self._recovery[False]
        counts = {"with_recollection": int(with_n), "without_recollection": int(without_n), "low": low, "warmer": warmer}
        if with_n < MIN_SAMPLES or without_n < MIN_SAMPLES:
            return Elsewhere(**counts)
        helped = max(0.0, with_sum / with_n - without_sum / without_n)
        floor = None
        if low and warmer and helped > 0.0 and memory is not None:
            floor = now + min(1.0, helped) * (memory - now)
            why = f"she is low, somewhere warmer came back to her, and that has lifted her {helped:.2f} before"
        elif helped > 0.0:
            why = "somewhere warmer in mind has helped her through lows, and this is not one"
        else:
            why = "somewhere warmer in mind has not helped her come up from a low"
        return Elsewhere(floor=floor, helped=helped, measured=True, why=why, **counts)


#: Made at import rather than on first use, so the subject-core fork carries it.
_ledger: ElsewhereLedger = ElsewhereLedger()


def get_elsewhere_ledger() -> ElsewhereLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = ElsewhereLedger()
