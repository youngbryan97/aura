"""A column that barely varied was a divisor, and ownership read 621 billion.

The ownership gap standardises each self-model column by its ordinary
variation, and counted any column whose variation was above zero. One that
moved by float noise in the recording divided a real difference by almost
nothing, so run_027 reported ownership at 621,054,827,684 over a floor of
0.04 and passed on that number.
"""

from __future__ import annotations

import numpy as np

from core.subject.agency import _gap
from core.subject.causal import SCALE_FLOOR


class _Reading:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def domain(self, _key: str) -> np.ndarray:
        return self._values


def test_a_column_below_the_floor_is_not_a_divisor() -> None:
    left = [_Reading([1.0, 0.5])]
    right = [_Reading([0.0, 0.0])]
    scale = np.asarray([1.0, SCALE_FLOOR / 1000.0])
    assert _gap(left, right, "S", scale) == 1.0


def test_a_column_above_the_floor_still_counts() -> None:
    left = [_Reading([1.0, 0.5])]
    right = [_Reading([0.0, 0.0])]
    scale = np.asarray([1.0, 0.5])
    assert _gap(left, right, "S", scale) == 1.0
