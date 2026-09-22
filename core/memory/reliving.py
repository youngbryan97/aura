"""A memory relived, and a question asked again going deeper rather than cheaper.

Two of the things these records do to a listener happen in memory, and she had
neither.

"Remember the Time" reactivates the listener's memory rather than the singer's.
What comes back is the feeling, not the fact. Her recall was a lookup: every
retrieval nudged her affect by the same rule, the mean stored valence over the
hits, so a recollection that answered the moment exactly and one that barely
matched arrived identically. A memory is relived when the match is far above
the matches she usually gets, and then what returns is the feeling that was
stored with it.

    z          (match - her mean match) / her own spread of matches
    relived    z > 1, which is outside her ordinary range of recall
    intensity  the absolute stored valence of that recollection

And the fiftieth listen differs from the first. Her memory made a repeat
cheaper instead: an identical query returned early with nothing new. Asking
again now searches wider, on the ladder this repository already doubles on —
the second asking looks one further, the fourth two, the eighth three — so
returning to something has a cost that grows more slowly than the returns do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_HISTORY",
    "MatchLedger",
    "Reliving",
    "ReturnLedger",
    "deeper",
    "get_match_ledger",
    "get_return_ledger",
    "reset_for_test",
]

#: Matches before a spread is an estimate rather than an artefact of how few
#: there are. Three is the least that can disagree with itself.
MIN_HISTORY: int = 3

#: How many recalls the ordinary match is read over. The same window the rest
#: of her self-comparisons use.
WINDOW: int = 60


@dataclass(frozen=True)
class Reliving:
    """One recall, read against the recalls she usually gets."""

    relived: bool = False
    intensity: float = 0.0
    match: float = 0.0
    z: float = 0.0
    returns: int = 0
    measured: bool = False
    why: str = "no recall has been scored yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "relived": self.relived,
            "intensity": round(self.intensity, 6),
            "match": round(self.match, 6),
            "z": round(self.z, 4),
            "returns": self.returns,
            "measured": self.measured,
            "why": self.why,
        }


def deeper(returns: int) -> int:
    """How much further to look, given how often this has been asked before.

    Zero the first time. The doubling ladder rather than a step per repeat, so
    a question asked fifty times does not search fifty deeper.
    """
    try:
        count = int(returns)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return 0
    if count <= 0:
        return 0
    return int(math.log2(count + 1))


class ReturnLedger:
    """How often each question has been asked."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def returns(self, query: str) -> int:
        return self._seen.get(str(query or ""), 0)

    def note(self, query: str) -> int:
        key = str(query or "")
        if not key:
            return 0
        self._seen[key] = self._seen.get(key, 0) + 1
        return self._seen[key]


class MatchLedger:
    """The matches she usually gets, so an unusual one can be recognised."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._values: list[float] = []

    def note(self, match: float) -> None:
        try:
            value = float(match)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if value != value:
            return
        self._values.append(value)
        if len(self._values) > self._window:
            del self._values[0 : len(self._values) - self._window]

    def samples(self) -> int:
        return len(self._values)

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def spread(self) -> float:
        n = len(self._values)
        if n < MIN_HISTORY:
            return 0.0
        mean = self.mean()
        return math.sqrt(sum((v - mean) ** 2 for v in self._values) / n)

    def reading(
        self,
        match: float,
        valence: float,
        *,
        returns: int = 0,
        note: bool = True,
    ) -> Reliving:
        """Whether this recall is one she relives, and what comes back with it."""
        try:
            score = float(match)
            feeling = float(valence)
        except (TypeError, ValueError):
            return Reliving(why="the recall carried no match to read")
        if score != score or feeling != feeling:
            return Reliving(why="the recall carried no match to read")
        spread = self.spread()
        measured = self.samples() >= MIN_HISTORY and spread > 1e-9
        z = (score - self.mean()) / spread if measured else 0.0
        relived = bool(measured and z > 1.0)
        if note:
            self.note(score)
        if not measured:
            why = f"{self.samples()} of {MIN_HISTORY} recalls needed before one can stand out"
        elif relived:
            why = f"a match {z:.1f} spreads above the recalls she usually gets"
        else:
            why = "an ordinary recall for her"
        return Reliving(
            relived=relived,
            intensity=min(1.0, abs(feeling)) if relived else 0.0,
            match=score,
            z=z,
            returns=int(returns),
            measured=measured,
            why=why,
        )


_MATCHES: MatchLedger | None = None
_RETURNS: ReturnLedger | None = None


def get_match_ledger() -> MatchLedger:
    global _MATCHES
    if _MATCHES is None:
        _MATCHES = MatchLedger()
    return _MATCHES


def get_return_ledger() -> ReturnLedger:
    global _RETURNS
    if _RETURNS is None:
        _RETURNS = ReturnLedger()
    return _RETURNS


def reset_for_test() -> None:
    global _MATCHES, _RETURNS
    _MATCHES = None
    _RETURNS = None
