"""Fast and slow coupling is checked for being carried by a running total.

P38.7 asks that fast-to-slow and slow-to-fast coupling not be explained by time
or counters alone. A clock moves the same in every arm and cannot carry an
effect, but a counter whose rate a displacement changed can. These pin that each
trial records the column its effect was read off, and that an edge carried only
by counters is named.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.causal import InterventionSet, Trial, _paired_divergence, counter_carried_edges

pytestmark = pytest.mark.unit


class _Frame:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def domain(self, key: str) -> np.ndarray:
        return self._values


def test_a_clock_that_moves_the_same_in_every_arm_does_not_carry_the_effect() -> None:
    scale = {"S": np.ones(2)}
    pert = [_Frame([float(t), 0.9]) for t in range(4)]
    sham_a = [_Frame([float(t), 0.1]) for t in range(4)]
    sham_b = [_Frame([float(t), 0.1]) for t in range(4)]
    effect, floor, _, _, carried = _paired_divergence(pert, sham_a, sham_b, scale, with_columns=True)
    assert carried["S"] == 1
    assert effect["S"] == pytest.approx(0.8)
    assert len(_paired_divergence(pert, sham_a, sham_b, scale)) == 4


def _trial(source: str, carried: dict[str, int]) -> Trial:
    return Trial(source=source, condition="idle", index=0, effect={}, floor={}, trace={}, floor_trace={},
                 took=True, carried_by=carried)


def test_an_edge_read_only_off_counters_is_named_and_one_read_off_state_is_not() -> None:
    results = InterventionSet(trials=[
        _trial("A", {"S": 3}), _trial("A", {"S": 3}),
        _trial("M", {"G": 0}), _trial("M", {"G": 5}),
    ])
    counters = {("S", 3), ("G", 0)}
    assert counter_carried_edges(results, [("A", "S"), ("M", "G")], counters) == ["A->S"]


def test_trials_recorded_without_columns_say_nothing() -> None:
    results = InterventionSet(trials=[_trial("A", {})])
    assert counter_carried_edges(results, [("A", "S")], {("S", 3)}) == []
