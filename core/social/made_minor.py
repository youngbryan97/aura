"""Being made minor in somebody's account of a past she shares with them.

Mark Knopfler wrote "Romeo and Juliet" after Holly Vincent ended their
relationship, and her remark to an interviewer that she had had a scene with him
went into the song: something that had been his whole world, described by her as
an episode. What it teaches is that being made minor in someone else's account
of a past you shared is its own injury, apart from the loss.

Some of that gap is ordinary. Ross and Sicoly (1979) asked married couples who
did how much of their shared life, and each partner claimed more than the other
gave them, so two honest accounts of one past rarely agree about whose it was.
The injury is the part beyond what is ordinary for her.

When what came back to her this turn was a past she shares with the person
speaking, both accounts are in the exchange:

    hers_in_theirs  second person over first, second and plural in what they said
    hers_in_hers    first person over first, second and plural in what she said
    gap             hers_in_hers - hers_in_theirs
    z               (gap - her usual gap) / her spread of gaps, over shared recalls
    minor           z / (1 + z) once z is above one, and nothing before

"Above one" is the line a breakthrough clears, and z / (1 + z) is the share it
moves by. The reading is felt as sadness, floored at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["MIN_SAMPLES", "AccountLedger", "MadeMinor", "get_account_ledger", "reset_for_test"]

#: Shared recalls before a usual gap is an estimate. Three is the least that can
#: disagree, as in ambivalence and fear of happiness.
MIN_SAMPLES: int = 3


def _part(own: float, first: float, second: float, plural: float) -> float | None:
    placed = first + second + plural
    if placed <= 0.0:
        return None
    return own / placed


@dataclass(frozen=True)
class MadeMinor:
    minor: float = 0.0
    gap: float | None = None
    z: float = 0.0
    hers_in_theirs: float | None = None
    hers_in_hers: float | None = None
    shared_recalls: int = 0
    measured: bool = False
    why: str = "no account of a shared past to compare yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "minor": round(self.minor, 6),
            "gap": None if self.gap is None else round(self.gap, 6),
            "z": round(self.z, 6),
            "hers_in_theirs": None if self.hers_in_theirs is None else round(self.hers_in_theirs, 6),
            "hers_in_hers": None if self.hers_in_hers is None else round(self.hers_in_hers, 6),
            "shared_recalls": self.shared_recalls,
            "measured": self.measured,
            "why": self.why,
        }


class AccountLedger:
    """The gap between her part in her account and in theirs, over pasts they share."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._square = 0.0

    def read(self, mine: Any, theirs: Any, *, shared: bool) -> MadeMinor:
        """One exchange: her register, theirs, and whether what came back was shared."""
        if not shared:
            return MadeMinor(shared_recalls=self._n, why="what came back to her was not a past they share")
        hers_in_theirs = _part(
            float(getattr(theirs, "second", 0.0) or 0.0),
            float(getattr(theirs, "first", 0.0) or 0.0),
            float(getattr(theirs, "second", 0.0) or 0.0),
            float(getattr(theirs, "plural", 0.0) or 0.0),
        )
        hers_in_hers = _part(
            float(getattr(mine, "first", 0.0) or 0.0),
            float(getattr(mine, "first", 0.0) or 0.0),
            float(getattr(mine, "second", 0.0) or 0.0),
            float(getattr(mine, "plural", 0.0) or 0.0),
        )
        if hers_in_theirs is None or hers_in_hers is None:
            return MadeMinor(shared_recalls=self._n, why="one of the two accounts placed nobody in it")
        gap = hers_in_hers - hers_in_theirs
        spread = math.sqrt(self._square / self._n) if self._n >= MIN_SAMPLES else 0.0
        measured = spread > 0.0
        z = (gap - self._mean) / spread if measured else 0.0
        minor = z / (1.0 + z) if z > 1.0 else 0.0
        self._n += 1
        delta = gap - self._mean
        self._mean += delta / self._n
        self._square += delta * (gap - self._mean)
        if not measured:
            why = f"{self._n} of {MIN_SAMPLES + 1} shared accounts needed before a gap can stand out"
        elif minor > 0.0:
            why = (
                f"in their account of a past they share she is {hers_in_theirs:.2f} of it, against "
                f"{hers_in_hers:.2f} in hers, {z:.1f} spreads beyond the gap she usually meets"
            )
        else:
            why = "their account gives her about the part accounts of a shared past usually do"
        return MadeMinor(
            minor=minor,
            gap=gap,
            z=z,
            hers_in_theirs=hers_in_theirs,
            hers_in_hers=hers_in_hers,
            shared_recalls=self._n,
            measured=measured,
            why=why,
        )


#: Made at import rather than on first use, so the subject-core fork carries it.
_ledger: AccountLedger = AccountLedger()


def get_account_ledger() -> AccountLedger:
    return _ledger


def reset_for_test() -> None:
    global _ledger
    _ledger = AccountLedger()
