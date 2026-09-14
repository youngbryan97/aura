"""When enough has happened to justify revisiting the story she tells about herself.

The phase that does this has said since it was written that it "revisits the
self-narrative when enough has happened to justify it". What it did was
increment on `state.version % 20` — a counter. Twenty turns of silence and
twenty turns that changed everything got the same answer, and the declaration
and the implementation had never agreed.

What has happened is readable, and the records say where to read it. "Remember
the Time" is fifteen questions in a row and no statements: the past being
brought back is the whole of what the relationship is made of. Oddisee's second
verse is the same move pointed inward — "you were there from the start" — where
the recollection is not evidence about the self-model, it is the self-model.

So the measure is what recall brought back that the story does not already
hold:

    unaccounted = share of the recalled words the narrative does not contain
    z           = (unaccounted - her usual) / her own spread

Above one spread is recall returning something her story has no room for, which
is the condition the phase's own sentence describes. The bar is her own
variation rather than a number chosen here, so a life where recall routinely
brings back new things does not revise on every turn, and a settled one revises
when something genuinely arrives.

Nothing here rewrites anything. It answers whether the question is worth
asking, and the authority gate that has always guarded the mutation still
guards it.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "MIN_HISTORY",
    "Revision",
    "RevisionLedger",
    "get_revision_ledger",
    "reset_for_test",
    "unaccounted_share",
]

#: Readings before a spread is an estimate rather than an artefact of how few
#: there are.
MIN_HISTORY: int = 5

#: How many readings are held.
WINDOW: int = 60

_WORD = re.compile(r"[a-z']{4,}")

#: Words too common to say anything about whether a story accounts for a
#: memory. Taken from what any English sentence contains rather than from this
#: domain, so nothing about her subject matter is baked in.
_COMMON: frozenset[str] = frozenset(
    {
        "that", "this", "with", "from", "have", "been", "were", "they", "them",
        "when", "what", "which", "would", "could", "should", "there", "their",
        "about", "into", "than", "then", "your", "yours", "here", "some",
        "just", "like", "more", "most", "much", "very", "will", "shall",
        "does", "done", "made", "make", "said", "says", "also", "only",
        "over", "under", "after", "before", "because", "while", "still",
    }
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text or "").lower()) if w not in _COMMON}


def unaccounted_share(recalled: Iterable[str], narrative: str) -> float | None:
    """How much of what came back the story does not already contain.

    None when there is nothing to compare — no recall, or no story — which is
    different from a story that accounts for everything and must not read as
    one.
    """
    brought = set()
    for item in recalled or ():
        brought |= _words(item)
    if not brought:
        return None
    held = _words(narrative)
    if not held:
        return None
    return len(brought - held) / len(brought)


@dataclass
class Revision:
    """Whether what came back is something the story has no room for."""

    unaccounted: float = 0.0
    usual: float = 0.0
    spread: float = 0.0
    z: float = 0.0
    worth_revisiting: bool = False
    readings: int = 0
    measured: bool = False
    why: str = "nothing has come back yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "unaccounted": round(self.unaccounted, 6),
            "usual": round(self.usual, 6),
            "spread": round(self.spread, 6),
            "z": round(self.z, 4),
            "worth_revisiting": self.worth_revisiting,
            "readings": self.readings,
            "measured": self.measured,
            "why": self.why,
        }


class RevisionLedger:
    """How much recall usually brings back that the story does not hold."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._values: deque[float] = deque(maxlen=self._window)

    def readings(self) -> int:
        return len(self._values)

    def usual(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def spread(self) -> float:
        n = len(self._values)
        if n < MIN_HISTORY:
            return 0.0
        mean = self.usual()
        return math.sqrt(sum((v - mean) ** 2 for v in self._values) / n)

    def read(self, recalled: Iterable[str], narrative: str) -> Revision:
        """One reading, scored against the ones before it, then kept."""
        share = unaccounted_share(recalled, narrative)
        if share is None:
            return Revision(
                readings=len(self._values),
                why="nothing came back, or there is no story yet to compare it against",
            )
        n = len(self._values)
        usual, spread = self.usual(), self.spread()
        self._values.append(share)
        if n < MIN_HISTORY:
            return Revision(
                unaccounted=share,
                usual=usual,
                readings=n,
                why=f"{n} of {MIN_HISTORY} readings needed before a share means anything",
            )
        if spread <= 1e-9:
            # Every reading before this one was the same. That is not a
            # comparison she cannot make — it is the sharpest one available:
            # recall has never brought back more than this and now it has.
            worth = share > usual
            return Revision(
                unaccounted=share,
                usual=usual,
                worth_revisiting=worth,
                readings=n,
                measured=True,
                why=(
                    f"recall brought back {share:.0%} the story does not hold, "
                    f"against {usual:.0%} on every one of the {n} before it"
                    if worth
                    else f"recall brought back {share:.0%}, which is what it always brings"
                ),
            )
        z = (share - usual) / spread
        worth = z > 1.0
        return Revision(
            unaccounted=share,
            usual=usual,
            spread=spread,
            z=z,
            worth_revisiting=worth,
            readings=n,
            measured=True,
            why=(
                f"recall brought back {share:.0%} the story does not hold, "
                f"{z:.1f} spreads above her usual {usual:.0%}"
                if worth
                else f"recall brought back {share:.0%}, within her usual {usual:.0%}"
            ),
        )


_LEDGER: RevisionLedger | None = None


def get_revision_ledger() -> RevisionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = RevisionLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
