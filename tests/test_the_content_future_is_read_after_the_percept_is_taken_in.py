"""The content run reads each class's future after the percept has been taken in.

A turn records one frame before its phases run and one after each phase. The
content run read frame `lag`, and frame one is the open frame: captured after
the class's percept arrived in the stream and before any phase had processed
it. Every class forked from one anchor had the same future there, and the
internal geometry came out as one distance for every pair of classes. These pin
that the future is the end of the turn `lag` counts, whatever the number of
frames a turn records.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.subject.content_runtime import grid, sample_classes
from core.subject.driver import Condition

pytestmark = pytest.mark.unit


class _Frame:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def vector(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float64)


class _Runtime:
    """Three frames a turn: open, then two phases that have read the newest percept."""

    def __init__(self) -> None:
        self.state = AuraState.default()
        self.organs = None
        self.ontogeny = None
        self.turn = 0

    def restore(self, snapshot: object) -> None:
        self.state = AuraState.default()
        self.turn = 0

    async def turn_once(self, condition: Condition) -> list[_Frame]:
        if condition.prepare is not None:
            condition.prepare(self.state, random.Random(0))
        self.turn += 1
        newest = self.state.world.recent_percepts[-1]
        taken = float(len(newest["content"]))
        return [
            _Frame([0.0, 0.0]),
            _Frame([taken, 0.0]),
            _Frame([taken, float(self.turn)]),
        ]


def _futures(*, turns: int, lag: int) -> dict[str, np.ndarray]:
    runtime = _Runtime()
    anchors = [SimpleNamespace(snapshot=None, current=np.zeros(2))]
    classes = grid()[:3]
    samples = asyncio.run(
        sample_classes(runtime, anchors, [Condition("idle", "", origin="system")], classes, turns=turns, lag=lag)
    )
    return {cls.content: samples[cls.name].futures[0] for cls in classes}


def test_each_class_s_future_is_read_after_its_percept_was_taken_in() -> None:
    futures = _futures(turns=1, lag=1)
    for content, future in futures.items():
        assert future[0] == pytest.approx(len(content))
    assert len({tuple(future) for future in futures.values()}) == len(futures)


def test_lag_counts_turns_and_reads_the_end_of_that_turn() -> None:
    first = _futures(turns=2, lag=1)
    second = _futures(turns=2, lag=2)
    assert all(future[1] == pytest.approx(1.0) for future in first.values())
    assert all(future[1] == pytest.approx(2.0) for future in second.values())
