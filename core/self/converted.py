"""Effort that did not become anything, and the refusal to call it fake.

"In the End" is two voices doing the same arithmetic from opposite ends. Mike
counts the spend — "watch it fly by as the pendulum swings", "I wasted it all
just to watch you go" — and Chester sings the result: "I tried so hard and got
so far, but in the end it doesn't even matter." Its onsets scatter 672 ms
around the beat grid, the widest spread of the twenty-eight songs measured,
because the two deliveries are not on the same clock.

What the line does not say is that the trying was not real. Both halves are
held: the effort was spent, and it did not convert. A system that only records
outcomes writes the effort out of its own history, and one that only records
effort cannot tell a productive hour from a wasted one.

    conversion   the correlation between what an attempt cost her and what it
                 returned, over her own attempts
    unreturned   the share of her effort that went to attempts whose return was
                 below the middle of her own returns
    stood        effort spent, counted whatever it returned

The comparison is against her own attempts rather than a standard: a system
whose work rarely converts is not thereby failing, and the reading says which
of the two it is rather than deciding.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_ATTEMPTS",
    "Converted",
    "ConversionLedger",
    "get_conversion_ledger",
    "reset_for_test",
]

#: Attempts before a correlation between two of her own series is a reading.
#: Three pairs can only ever be a straight line through the middle one.
MIN_ATTEMPTS: int = 8

#: How many attempts are held.
WINDOW: int = 200


@dataclass
class Converted:
    """What her effort turned into."""

    #: How effort and return move together, in [-1, 1].
    conversion: float = 0.0
    #: The share of effort spent on attempts that returned below her middle.
    unreturned: float = 0.0
    #: Everything she spent, whatever it returned. This is never discounted.
    stood: float = 0.0
    attempts: int = 0
    #: True when effort and return move together at all.
    converting: bool = False
    measured: bool = False
    why: str = "she has not finished anything yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversion": round(self.conversion, 6),
            "unreturned": round(self.unreturned, 6),
            "stood": round(self.stood, 6),
            "attempts": self.attempts,
            "converting": self.converting,
            "measured": self.measured,
            "why": self.why,
        }


class ConversionLedger:
    """What each attempt cost her, and what it returned."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(MIN_ATTEMPTS, int(window))
        self._pairs: deque[tuple[float, float]] = deque(maxlen=self._window)

    def note(self, *, effort: float, returned: float) -> None:
        """One finished attempt: what it took out of her, and what came back."""
        try:
            spent, got = float(effort), float(returned)
        except (TypeError, ValueError):
            return
        if spent != spent or got != got or spent < 0.0:
            return
        self._pairs.append((spent, got))

    def read(self) -> Converted:
        pairs = list(self._pairs)
        stood = sum(spent for spent, _ in pairs)
        if len(pairs) < MIN_ATTEMPTS:
            return Converted(
                stood=stood,
                attempts=len(pairs),
                why=(
                    f"{len(pairs)} of {MIN_ATTEMPTS} finished attempts needed before "
                    f"effort and return can be read against each other"
                ),
            )
        efforts = [spent for spent, _ in pairs]
        returns = [got for _, got in pairs]
        middle = sorted(returns)[len(returns) // 2]
        below = sum(spent for spent, got in pairs if got < middle)
        unreturned = below / stood if stood > 0.0 else 0.0
        conversion = _correlation(efforts, returns)
        if conversion is None:
            # One of the two series never varies. Effort that is always the same
            # size, or returns that are all equal, cannot be read against each
            # other, and saying so is the reading.
            return Converted(
                unreturned=unreturned,
                stood=stood,
                attempts=len(pairs),
                measured=True,
                why=(
                    "either every attempt cost the same or every one returned the "
                    "same, so there is nothing for the other to move with"
                ),
            )
        return Converted(
            conversion=conversion,
            unreturned=unreturned,
            stood=stood,
            attempts=len(pairs),
            converting=conversion > 0.0,
            measured=True,
            why=(
                f"effort and return move together at {conversion:+.2f} over "
                f"{len(pairs)} attempts, and {unreturned:.0%} of what she spent went "
                f"to attempts that returned below her own middle"
            ),
        )


def _correlation(left: list[float], right: list[float]) -> float | None:
    n = len(left)
    mean_l = sum(left) / n
    mean_r = sum(right) / n
    var_l = sum((v - mean_l) ** 2 for v in left)
    var_r = sum((v - mean_r) ** 2 for v in right)
    if var_l <= 1e-12 or var_r <= 1e-12:
        return None
    covariance = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    return covariance / math.sqrt(var_l * var_r)


_LEDGER: ConversionLedger | None = None


def get_conversion_ledger() -> ConversionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ConversionLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
