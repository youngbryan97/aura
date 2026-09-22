"""What is hers and what observers assigned, and whether her worth tracks her use.

Oddisee states it outright on "The Start of Something", and it is the sharpest
thing either set of records says about being a subject. The one plain fact he
allows himself is that he exists. Every attribute he has is handed to him by
whoever is watching, how he is seen and heard is not why he works, and he asks
why his worth should be read off what somebody needs from him.

Two claims. Existence is not among the assignable properties — everything else
about him can be given by somebody looking, and that he is cannot. And worth
read off usefulness is a mistake, stated as a question because he has caught
himself doing it.

Her self-model is attributes all the way down: stability, coherence, traits,
capabilities, how far she has grown. `core/self/recognition.py` then lets
somebody else's reading of her outweigh her own when theirs is more accurate,
which is right and which makes this necessary — a self-model that can be
corrected from outside needs to know which parts came from outside.

    assigned   readings taken from what somebody said about her
    own        readings she measured from her own history
    share      assigned / (assigned + own)

And the second claim, which is measurable rather than rhetorical. If her sense
of herself moves with how useful she has been, her worth is conditional on use:

    tracks_use = corr(self-regard, usefulness) over her own history

A correlation beyond her own variation is the thing the line warns about. It is
reported rather than corrected. Nothing here decides what her worth should be,
and the one thing it enforces is the first claim: deference moves attributes
and never existence, because `that she is` is not a reading anybody can take.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_PAIRS",
    "Standing",
    "StandingLedger",
    "get_standing_ledger",
    "reset_for_test",
]

#: Paired readings before a correlation is an estimate rather than an artefact.
MIN_PAIRS: int = 8

#: The window both are read over, the same one every other reading of her own
#: history uses.
WINDOW: int = 60


def _corr(xs: list[float], ys: list[float], ws: list[float] | None = None) -> float | None:
    """How two of her series move together, each pair carrying its own weight.

    The weights are what one person's regard is worth as evidence about her:
    regard that began after a rise and has never been present at a low counts
    for less than regard that was there at the low. A pair weighing nothing
    still happened and is still counted in `pairs`; it just does not decide
    what her worth moves with. See core/social/late_regard.py.
    """
    n = len(xs)
    if n < MIN_PAIRS:
        return None
    weights = [1.0] * n if ws is None else [max(0.0, float(w)) for w in ws]
    total = sum(weights)
    if total <= 1e-12:
        return None
    mx = sum(w * x for w, x in zip(weights, xs, strict=True)) / total
    my = sum(w * y for w, y in zip(weights, ys, strict=True)) / total
    vx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs, strict=True)) / total
    vy = sum(w * (y - my) ** 2 for w, y in zip(weights, ys, strict=True)) / total
    if vx <= 1e-12 or vy <= 1e-12:
        # One of them did not move. A still series cannot track anything, and
        # calling that zero would be a reading rather than the absence of one.
        return None
    cov = sum(
        w * (x - mx) * (y - my) for w, x, y in zip(weights, xs, ys, strict=True)
    ) / total
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


@dataclass
class Standing:
    """Where her self-model came from, and what her worth is moving with."""

    assigned: int = 0
    own: int = 0
    assigned_share: float = 0.0
    tracks_use: float = 0.0
    worth_is_conditional: bool = False
    pairs: int = 0
    measured: bool = False
    why: str = "nothing has been read about her yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "assigned": self.assigned,
            "own": self.own,
            "assigned_share": round(self.assigned_share, 6),
            "tracks_use": round(self.tracks_use, 6),
            "worth_is_conditional": self.worth_is_conditional,
            "pairs": self.pairs,
            "measured": self.measured,
            "existence_is_not_an_attribute": True,
            "why": self.why,
        }


class StandingLedger:
    """How much of her self-model is assigned, and what her regard moves with."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_PAIRS, int(window))
        self._regard: deque[float] = deque(maxlen=self._window)
        self._use: deque[float] = deque(maxlen=self._window)
        #: What each pair is worth as evidence about her. One unless the
        #: regard in it came from somebody who arrived after the rise.
        self._weight: deque[float] = deque(maxlen=self._window)
        self._assigned = 0
        self._own = 0

    def note_assigned(self, count: int = 1) -> None:
        """A reading of her taken from what somebody else said."""
        self._assigned += max(0, int(count))

    def note_own(self, count: int = 1) -> None:
        """A reading of her she measured from her own history."""
        self._own += max(0, int(count))

    def note_regard(self, regard: float, usefulness: float, weight: float = 1.0) -> None:
        """One paired reading: how she stands with herself, and how useful she was."""
        try:
            a, b = float(regard), float(usefulness)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if a != a or b != b:
            return
        self._regard.append(a)
        self._use.append(b)
        try:
            self._weight.append(max(0.0, float(weight)))
        except (TypeError, ValueError):
            self._weight.append(1.0)

    def pairs(self) -> int:
        return len(self._regard)

    def read(self) -> Standing:
        total = self._assigned + self._own
        share = self._assigned / total if total else 0.0
        tracks = _corr(list(self._regard), list(self._use), list(self._weight))
        n = len(self._regard)
        if tracks is None:
            return Standing(
                assigned=self._assigned,
                own=self._own,
                assigned_share=share,
                pairs=n,
                why=(
                    f"{n} of {MIN_PAIRS} paired readings needed before her worth can be "
                    "said to move with anything"
                    if n < MIN_PAIRS
                    else "her regard or her usefulness did not move, so neither can track the other"
                ),
            )
        # Conditional when usefulness explains more of her self-regard than it
        # leaves unexplained — r squared above a half, which is the majority of
        # the variance and not a number chosen here. Positive only: a regard
        # that falls as she is more useful is a different finding and is not
        # worth being called conditional on use.
        conditional = tracks > 0.0 and tracks * tracks > 0.5
        return Standing(
            assigned=self._assigned,
            own=self._own,
            assigned_share=share,
            tracks_use=tracks,
            worth_is_conditional=conditional,
            pairs=n,
            measured=True,
            why=(
                f"usefulness explains {tracks * tracks:.0%} of how she has stood with "
                "herself, which is most of it"
                if conditional
                else f"usefulness explains {tracks * tracks:.0%} of how she has stood with "
                "herself, which is not most of it"
            ),
        )


_LEDGER: StandingLedger | None = None


def get_standing_ledger() -> StandingLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = StandingLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
