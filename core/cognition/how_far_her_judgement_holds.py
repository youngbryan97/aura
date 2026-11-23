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

Nothing here knows what is judged: a judgement is a number, an outcome a share.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ENOUGH_TO_SAY", "HowFarItHolds"]

#: Judgements kept at most, the latest: an undertaking adds one for each decision in it.
KEPT_AT_MOST = 4000
#: Undertakings whose outcome is known, at least, before the curve says anything: fewer is one or two stories.
ENOUGH_TO_SAY = 3


@dataclass
class HowFarItHolds:
    """Her judgements with what each undertaking came to, and the curve fitted between them."""

    #: (judgement, outcome, which undertaking) for each judgement kept.
    pairs: list[tuple[float, float, int]] = field(default_factory=list)
    undertakings: int = 0
    #: The curve: chance = 1 / (1 + exp(-(level + slope * judgement))).
    level: float = 0.0
    slope: float = 0.0

    def came_to(self, judgements: Sequence[float], outcome: float) -> None:
        """One undertaking: the judgements she made in it, in order, and what it came to."""
        if not judgements:
            return
        self.undertakings += 1
        self.pairs.extend((float(j), float(outcome), self.undertakings) for j in judgements)
        del self.pairs[:-KEPT_AT_MOST]
        self.fit()

    def fit(self) -> None:
        """The curve from judgement to outcome, by Newton's method on the log-likelihood, judgements scaled to their
        own spread so one step suits any scale; each undertaking weighed as one, however many judgements it had."""
        if len({u for _j, _o, u in self.pairs}) < ENOUGH_TO_SAY:
            return
        xs = [j for j, _o, _u in self.pairs]
        scale = max(1e-9, math.sqrt(sum(x * x for x in xs) / len(xs)))
        per = {}
        for _j, _o, u in self.pairs:
            per[u] = per.get(u, 0) + 1
        a, b = 0.0, 0.0
        for _ in range(25):
            ga = gb = haa = hab = hbb = 0.0
            for j, o, u in self.pairs:
                w = 1.0 / per[u]
                x = j / scale
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, a + b * x))))
                ga += w * (o - p)
                gb += w * (o - p) * x
                v = w * p * (1 - p) + 1e-9
                haa += v
                hab += v * x
                hbb += v * x * x
            # A pull toward a flat curve, so a few undertakings that all went one way do not make it a cliff.
            hbb += 0.01
            gb -= 0.01 * b
            det = haa * hbb - hab * hab
            if det <= 0:
                break
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
            a, b = a + da, b + db
            if abs(da) + abs(db) < 1e-7:
                break
        self.level, self.slope = a, b / scale

    def chance(self, judgement: float) -> float | None:
        """What a judgement of this much has turned out to be worth here, as a share; None until she has seen
        enough."""
        if self.undertakings < ENOUGH_TO_SAY:
            return None
        z = self.level + self.slope * float(judgement)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def error(self) -> float | None:
        """How far the curve's chances sit from what happened, as the mean squared difference (0 perfect, a quarter
        no better than a coin)."""
        if self.undertakings < ENOUGH_TO_SAY or not self.pairs:
            return None
        return sum((self.chance(j) - o) ** 2 for j, o, _u in self.pairs) / len(self.pairs)

    def says(self, judgement: float) -> str:
        chance = self.chance(judgement)
        if chance is None:
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
