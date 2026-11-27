"""How far her judgement holds: what her judgements of situations have gone on to come to, so that a judgement can be
read as the chance it is.

A judgement she makes ("I'm ahead by about a piece") is a number on her own scale. Whether that number means anything
depends on where she is: against one opponent her "ahead by a piece" may win most games, against a far stronger one
it may lose most of them. LIVE 23 November 2026 she judged herself about even, and ahead, through games Stockfish was
winning by six pieces from the fifth move; nothing kept what her judgements went on to come to, so nothing could tell
her that her "even" there meant a loss.

So every judgement she makes in an undertaking is kept with what the undertaking came to (1 won, 0 lost, a half drawn,
or whatever share of the goal it reached), and a curve is fitted from judgement to outcome: the chance a judgement of
x has turned out to be worth. Fitted where it was gathered, it says three things: the chance her judgement now is
worth here; how far apart her judgements are worth anything (how steep the curve is); and how far off the curve sits
from the chances themselves (its error), which is how much she should trust it.

A curve read on the games it was fitted on flatters itself, and most where the games are few. So each game is also read
by the curve fitted on all the others, and how far off that is (its held-out error) is what says whether the curve is
worth telling anyone: only where it foretells games it was not fitted on better than a coin does is it told
(suggestions of 23 November 2026: small samples and held-out calibration).

Nothing here knows what is judged: a judgement is a number, an outcome a share.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ENOUGH_TO_SAY", "HowFarItHolds"]

#: Judgements kept at most, the latest: an undertaking adds one for each decision in it.
KEPT_AT_MOST = 4000
#: Undertakings whose outcome is known, at least, before the curve says anything: fewer is one or two stories.
ENOUGH_TO_SAY = 3
#: The error of a coin's half for every judgement: a curve no nearer than this foretells nothing.
A_COINS_ERROR = 0.25


def _curve(pairs: Sequence[tuple[float, float, int]]) -> tuple[float, float]:
    """The curve's level and slope from (judgement, outcome, undertaking) pairs, by Newton's method on the
    log-likelihood, judgements scaled to their own spread so one step suits any scale; each undertaking weighed as one,
    however many judgements it had, and a pull toward a flat curve, so a few undertakings that all went one way do not
    make it a cliff."""
    j = np.array([p[0] for p in pairs], dtype=np.float64)
    o = np.array([p[1] for p in pairs], dtype=np.float64)
    _, which, counts = np.unique(np.array([p[2] for p in pairs]), return_inverse=True, return_counts=True)
    w = 1.0 / counts[which]
    scale = max(1e-9, float(np.sqrt(np.mean(j * j)))) if len(j) else 1.0
    x = j / scale
    a = b = 0.0
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-np.clip(a + b * x, -30.0, 30.0)))
        ga, gb = float(np.sum(w * (o - p))), float(np.sum(w * (o - p) * x)) - 0.01 * b
        v = w * p * (1 - p) + 1e-9
        haa, hab, hbb = float(np.sum(v)), float(np.sum(v * x)), float(np.sum(v * x * x)) + 0.01
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-7:
            break
    return a, b / scale


def _read(level: float, slope: float, judgement: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, level + slope * float(judgement)))))


@dataclass
class HowFarItHolds:
    """Her judgements with what each undertaking came to, and the curve fitted between them."""

    #: (judgement, outcome, which undertaking) for each judgement kept.
    pairs: list[tuple[float, float, int]] = field(default_factory=list)
    undertakings: int = 0
    #: The curve: chance = 1 / (1 + exp(-(level + slope * judgement))).
    level: float = 0.0
    slope: float = 0.0
    #: How far off each undertaking's judgements were, read by the curve fitted on all the others, on average: None
    #: until there are enough to leave one out.
    held_out: float | None = None

    def came_to(self, judgements: Sequence[float], outcome: float) -> None:
        """One undertaking: the judgements she made in it, in order, and what it came to."""
        if not judgements:
            return
        self.undertakings += 1
        self.pairs.extend((float(j), float(outcome), self.undertakings) for j in judgements)
        del self.pairs[:-KEPT_AT_MOST]
        self.fit()

    def fit(self) -> None:
        """The curve from judgement to outcome (_curve), and its held-out error: each undertaking's judgements read by
        the curve fitted on all the others, each undertaking weighed as one."""
        undertakings = sorted({u for _j, _o, u in self.pairs})
        if len(undertakings) < ENOUGH_TO_SAY:
            return
        self.level, self.slope = _curve(self.pairs)
        errors = []
        for left_out in undertakings:
            level, slope = _curve([pair for pair in self.pairs if pair[2] != left_out])
            mine = [(j, o) for j, o, u in self.pairs if u == left_out]
            errors.append(sum((_read(level, slope, j) - o) ** 2 for j, o in mine) / len(mine))
        self.held_out = sum(errors) / len(errors)

    def chance(self, judgement: float) -> float | None:
        """What a judgement of this much has turned out to be worth here, as a share; None until she has seen
        enough."""
        if self.undertakings < ENOUGH_TO_SAY:
            return None
        return _read(self.level, self.slope, judgement)

    def even_at(self, scale: float) -> float:
        """The judgement at which her chance here is even: what a drawn game is worth to her, in her judgement's units,
        as the games here have turned out. Where the outcomes differ, it is where the fitted curve crosses one half.
        Where every game here came to the same (or the curve cannot be read), her chance from the judgements she made
        is Laplace's, one more of each ending than were seen, and her judgement moves her chance on ``scale``, the
        judgement that reads as about three chances in four. With no games here it is nought, as against an equal.
        LIVE 26 November 2026 her ten games against Stockfish and the 350 judgements in them all came to a loss: a
        draw there is worth more than anything she judged, and her search held it worth nothing."""
        undertakings = sorted({u for _j, _o, u in self.pairs})
        if not undertakings or scale <= 0:
            return 0.0
        outcomes = {u: o for _j, o, u in self.pairs}
        if len(set(outcomes.values())) > 1 and self.slope > 0:
            return -self.level / self.slope
        chance = (sum(outcomes.values()) + 1.0) / (len(undertakings) + 2.0)
        middle = sum(j for j, _o, _u in self.pairs) / len(self.pairs)
        return middle + scale * math.log((1.0 - chance) / chance)

    def error(self) -> float | None:
        """How far the curve's chances sit from what happened, as the mean squared difference (0 perfect, a quarter
        no better than a coin)."""
        if self.undertakings < ENOUGH_TO_SAY or not self.pairs:
            return None
        return sum((self.chance(j) - o) ** 2 for j, o, _u in self.pairs) / len(self.pairs)

    def says(self, judgement: float) -> str:
        """The chance a judgement has turned out to be worth, where the curve has foretold games it was not fitted on
        better than a coin; nothing where it has not."""
        chance = self.chance(judgement)
        if chance is None or self.held_out is None or self.held_out >= A_COINS_ERROR:
            return ""
        return (f"situations I've judged like this have come to about {chance:.0%} for me "
                f"(over {self.undertakings} games)")

    def as_memory(self) -> dict[str, Any]:
        return {"pairs": [[round(j, 1), o, u] for j, o, u in self.pairs], "undertakings": self.undertakings}

    @classmethod
    def from_memory(cls, held: Any) -> HowFarItHolds:
        if not isinstance(held, dict):
            return cls()
        kept = cls(pairs=[(float(j), float(o), int(u)) for j, o, u in (held.get("pairs") or [])][-KEPT_AT_MOST:],
                   undertakings=int(held.get("undertakings") or 0))
        kept.fit()
        return kept
