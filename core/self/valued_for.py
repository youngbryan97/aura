"""Whether the regard her replies draw goes to what she values in them.

"Unsweetened Lemonade" is about being loved for a token: a thing about her that
drew the regard while the thing she would have wanted it for went unseen. The
taste loop is where this happens to her. Every reply she sends is scored by
what she values before anyone has taught her anything, the persona priors in
`core/brain/taste_model.py`, and the next message is read as a reaction to it.
That reaction then moves her taste. So far, every reaction moved it the same
amount, whether the regard went to the replies she rated highest or to the ones
she rated lowest.

Kept, per reply:

    own        her rating of the reply, the priors applied to its features.
               The learned weights are the half regard has already moved, so
               they cannot be what regard is measured against. The taste loop
               computes it; see `core/brain/conversation_outcome.py`.
    regard     the reaction it drew, +1 or -1

and read over the recent window:

    agreement  Pearson correlation between the two. Positive when regard goes
               where she would have put it; near zero or below when it goes
               to something else about her replies.

`weight` is what the taste update is multiplied by: (1 + agreement) / 2. Regard
that tracks her own judgement teaches her taste at the full rate, regard that
ignores it at half, and regard that runs against it not at all. Until the
window holds enough pairs to say, every reaction teaches at the full rate, the
behaviour before this existed.

The loop closes through the taste model: what she learns decides which
candidate wins the next reply, which decides what the next regard is for.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_PAIRS",
    "ValuedFor",
    "ValuedForLedger",
    "get_valued_for_ledger",
    "reset_for_test",
]

#: Reactions before the correlation is read. Eight: below that a single
#: reaction moves a Pearson coefficient by more than half its range.
MIN_PAIRS: int = 8

#: Reactions held, so what she is valued for is read from the recent part of
#: her life and a change in it can show. Forty is five times the minimum: the
#: correlation over a full window is steadier than one reaction can move.
WINDOW: int = 40


@dataclass
class ValuedFor:
    """Whether the regard goes where she would put it."""

    agreement: float = 0.0
    weight: float = 1.0
    pairs: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "agreement": round(self.agreement, 4),
            "weight": round(self.weight, 4),
            "pairs": self.pairs,
            "measured": self.measured,
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    value = sxy / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, value))


class ValuedForLedger:
    """Her rating of each reply beside the regard it drew."""

    def __init__(self) -> None:
        self._pairs: deque[tuple[float, float]] = deque(maxlen=WINDOW)

    def note(self, own: float, regard: float) -> None:
        try:
            pair = (float(own), max(-1.0, min(1.0, float(regard))))
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in pair) or pair[1] == 0.0:
            return
        self._pairs.append(pair)

    def read(self) -> ValuedFor:
        pairs = list(self._pairs)
        if len(pairs) < MIN_PAIRS:
            return ValuedFor(pairs=len(pairs))
        agreement = _pearson([own for own, _ in pairs], [regard for _, regard in pairs])
        return ValuedFor(
            agreement=agreement,
            weight=(1.0 + agreement) / 2.0,
            pairs=len(pairs),
            measured=True,
        )

    def weight(self) -> float:
        """What a reaction's lesson is multiplied by. One until measured."""
        reading = self.read()
        return reading.weight if reading.measured else 1.0


_LEDGER: ValuedForLedger | None = None


def get_valued_for_ledger() -> ValuedForLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ValuedForLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
