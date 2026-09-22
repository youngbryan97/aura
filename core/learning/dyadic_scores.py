"""An exact common integer scale for finite binary floating-point scores.

Every finite binary64 value has a power-of-two denominator. Multiplying all
scores by the largest denominator is an order-preserving exact embedding;
integer sums and differences need no repeated rational normalization.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class DyadicScores:
    integers: tuple[int, ...]
    denominator: int

    def __post_init__(self) -> None:
        if (type(self.denominator) is not int or self.denominator < 1
                or self.denominator & (self.denominator - 1)
                or type(self.integers) is not tuple
                or any(type(value) is not int for value in self.integers)):
            raise ValueError("dyadic scores require integers and a positive power-of-two scale")

    @classmethod
    def from_values(cls, values: Iterable[float]) -> DyadicScores:
        ratios = []
        for value in values:
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("score scale requires finite numerical values")
            ratios.append(float(value).as_integer_ratio())
        denominator = max((d for _, d in ratios), default=1)
        return cls(tuple(n * (denominator // d) for n, d in ratios), denominator)

    def upper_float(self, integer: int) -> float:
        """Round outward so a public float bound never drops a completion."""
        if type(integer) is not int:
            raise ValueError("scaled bound must be an integer")
        exact = Fraction(integer, self.denominator)
        try:
            rounded = float(exact)
        except OverflowError:
            return math.inf if integer > 0 else -float.fromhex("0x1.fffffffffffffp+1023")
        return math.nextafter(rounded, math.inf) if Fraction(rounded) < exact else rounded
