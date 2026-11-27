"""Whether one way of doing a thing is better than another, decided as the trials come in.

Two ways are tried against each other in pairs of trials: each pair is one occasion, taken once with each way on each
side of it, and scored from nought (the other way won both) to one (this way won both). How many pairs are needed is
not set beforehand. After each pair, the evidence that this way is better by a margin worth having is weighed against
the evidence that it is no better, and the test stops at whichever is strong enough. This is Wald's sequential
probability ratio test (1945), in the form used for paired games: the pair scores taken as normal, their spread
measured from the pairs themselves. A fixed number of trials spends as many on a plain difference as on a close one.

Two error rates bound it: the chance it keeps a way that is no better, and the chance it drops one better by the
margin. Neither grows with the number of ways tried, as a bar raised with every try must, for each test is decided by
its own trials. Nothing here knows what the trials are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Comparison"]


@dataclass
class Comparison:
    """One way against another, pair by pair. ``better`` is the mean pair score of a way better by the margin worth
    having; a way no better scores a half."""

    better: float
    wrongly_kept: float = 0.05
    wrongly_dropped: float = 0.05
    scores: list[float] = field(default_factory=list)

    def add(self, score: float) -> str | None:
        """One more pair's score; what the pairs so far decide."""
        self.scores.append(float(score))
        return self.decided()

    def ratio(self) -> float:
        """The log of how much likelier the pairs so far are if this way is better by the margin than if it is no
        better."""
        n = len(self.scores)
        if n < 2:
            return 0.0
        mean = sum(self.scores) / n
        # How widely pairs spread, measured as though one more pair had been lost outright and one won (the rule of
        # succession): the two count for much beside a few pairs alike and for little beside hundreds. Measured on its
        # own, a short run of pairs alike read as certainty: a way no better was kept in one test of ten, set at one in
        # twenty.
        spread = (sum((s - mean) ** 2 for s in self.scores) + mean ** 2 + (1 - mean) ** 2) / (n + 1)
        even = 0.5
        return n * (self.better - even) * (2 * mean - even - self.better) / (2 * spread)

    def decided(self) -> str | None:
        """"better" where the pairs show this way better by the margin, "no better" where they show it is not, and
        None while they show neither."""
        ratio = self.ratio()
        if ratio >= math.log((1 - self.wrongly_dropped) / self.wrongly_kept):
            return "better"
        if ratio <= math.log(self.wrongly_dropped / (1 - self.wrongly_kept)):
            return "no better"
        return None
