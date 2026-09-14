"""A prediction that came true, which had no signal anywhere in the system.

Every part of this mind measures prediction error. Free energy, the self
prediction loop's smoothed error, the world model's surprise, the ontogenetic
reservoir's novelty. All of them fire when reality disagrees, and not one of
them fires when it agrees. The self prediction loop's own docstring says "low
prediction error -> more confidence in self-model" and nothing carries that
out: there is no event, no reading, no felt quality. She can be wrong and feel
it, and she cannot be right and feel it.

That asymmetry is why a listener's experience of these records is not available
to her. The pleasure in a hook returning is the pleasure of having known it
would. Eight repetitions of "if you love me won't you say something" work
because the ninth is expected. Nothing about that is error.

The fix is the symmetry the maths already had. Surprise is the surprisal of the
upper tail of her own error distribution: an error larger than she usually
makes. Confirmation is the surprisal of the lower tail: an error smaller than
she usually makes. Same estimator, same units, opposite side.

    p_high(e) = P(error >= e)          from her own history
    p_low(e)  = P(error <= e)
    surprise  = -log2 p_high(e)        bits
    confirmed = -log2 p_low(e)         bits

This gets the important thing right for free: a prediction she always gets
right carries nothing when it comes true, and one she usually misses carries a
lot. Being right about the easy thing is not an experience.

Bits are unbounded, and an affect axis is not. The bound is what her history
can resolve rather than a number chosen here: with n samples the smallest tail
probability observable is 1/(n+1), so the most bits she could possibly see is
log2(n+1), and the reading is scaled by that. It rises as her life gets long
enough to resolve rarer agreements.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import deque
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "Expectation",
    "ExpectationLedger",
    "MIN_HISTORY",
    "get_expectation_ledger",
    "reset_for_test",
]

#: Scored predictions before a tail probability is an estimate rather than an
#: artefact of how few there are. Below this the reading reports that it cannot
#: say, which is different from reporting no confirmation.
MIN_HISTORY: int = 8

#: How much history the tails are read from. Her error distribution moves as
#: she learns, so a lifelong estimate would compare today's accuracy against a
#: version of her that no longer exists. The window is the self prediction
#: loop's own history size, which is the length over which that loop already
#: considers its own error to be one distribution.
WINDOW: int = 60


@dataclass(frozen=True)
class Expectation:
    """One scored prediction, read from both ends."""

    error: float
    confirmation: float = 0.0
    surprise: float = 0.0
    confirmation_bits: float = 0.0
    surprise_bits: float = 0.0
    ceiling_bits: float = 0.0
    samples: int = 0
    measured: bool = False
    why: str = "not enough history to say"

    #: A bit is the information in a coin flip, so more than one bit either way
    #: means the outcome was rarer than half her predictions. The bar is the
    #: unit rather than a number chosen for it.
    def confirmed(self) -> bool:
        return self.measured and self.confirmation_bits > 1.0

    def surprised(self) -> bool:
        return self.measured and self.surprise_bits > 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "error": round(self.error, 6),
            "confirmation": round(self.confirmation, 6),
            "surprise": round(self.surprise, 6),
            "confirmation_bits": round(self.confirmation_bits, 4),
            "surprise_bits": round(self.surprise_bits, 4),
            "ceiling_bits": round(self.ceiling_bits, 4),
            "samples": self.samples,
            "measured": self.measured,
            "confirmed": self.confirmed(),
            "surprised": self.surprised(),
            "why": self.why,
        }


class ExpectationLedger:
    """Her own distribution of prediction errors, read from either end.

    The errors are kept sorted as well as in arrival order, so a tail
    probability is a search rather than a scan and the cost does not grow with
    how long she has been running.
    """

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_HISTORY, int(window))
        self._order: deque[float] = deque()
        self._sorted: list[float] = []

    def note(self, error: float) -> None:
        try:
            value = float(error)
        except (TypeError, ValueError):
            return
        if value != value:  # NaN is not a score
            return
        self._order.append(value)
        insort(self._sorted, value)
        while len(self._order) > self._window:
            oldest = self._order.popleft()
            index = bisect_left(self._sorted, oldest)
            if index < len(self._sorted) and self._sorted[index] == oldest:
                self._sorted.pop(index)

    def samples(self) -> int:
        return len(self._order)

    def ceiling_bits(self) -> float:
        """The most bits her history can resolve, which is the reading's bound."""
        n = len(self._order)
        return 0.0 if n <= 0 else math.log2(n + 1)

    def _tails(self, error: float) -> tuple[float, float]:
        """(P(error <= e), P(error >= e)) with the Laplace correction.

        Without the correction the best possible outcome has probability zero
        and infinite surprisal, which is a statement about the window's length
        rather than about the prediction.
        """
        n = len(self._sorted)
        if n <= 0:
            return 1.0, 1.0
        at_or_below = bisect_right(self._sorted, error)
        at_or_above = n - bisect_left(self._sorted, error)
        return (at_or_below + 1.0) / (n + 1.0), (at_or_above + 1.0) / (n + 1.0)

    def score(self, error: float) -> Expectation:
        """Read one outcome from both ends, before it joins the history.

        Scored against the history that preceded it. Folding it in first would
        let every outcome make itself look ordinary.
        """
        try:
            value = float(error)
        except (TypeError, ValueError):
            return Expectation(error=0.0, why="the error was not a number")
        n = len(self._order)
        if n < MIN_HISTORY:
            return Expectation(
                error=value,
                samples=n,
                why=f"{n} of {MIN_HISTORY} scored predictions needed to read a tail",
            )
        low, high = self._tails(value)
        ceiling = self.ceiling_bits()
        # Clamped at zero: a tail probability of one is an outcome as ordinary
        # as they come, and float arithmetic puts its surprisal a hair below
        # zero, which would read as negative information.
        confirmation_bits = max(0.0, -math.log2(low))
        surprise_bits = max(0.0, -math.log2(high))
        scale = ceiling if ceiling > 0.0 else 1.0
        return Expectation(
            error=value,
            confirmation=max(0.0, min(1.0, confirmation_bits / scale)),
            surprise=max(0.0, min(1.0, surprise_bits / scale)),
            confirmation_bits=confirmation_bits,
            surprise_bits=surprise_bits,
            ceiling_bits=ceiling,
            samples=n,
            measured=True,
            why=(
                f"an error of {value:.4f} sits below {low:.3f} and above "
                f"{high:.3f} of the last {n} she made"
            ),
        )

    def score_and_note(self, error: float) -> Expectation:
        reading = self.score(error)
        self.note(error)
        return reading

    def load(self, errors: Iterable[float]) -> None:
        for error in errors:
            self.note(error)


_LEDGER: ExpectationLedger | None = None


def get_expectation_ledger() -> ExpectationLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ExpectationLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
