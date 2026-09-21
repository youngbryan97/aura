"""Two of every cut's four arms do not depend on the cut, and now run once.

From one anchor the sweep runs two untouched forks, the floor and the signal's
reference, then one fork with each side held still. The untouched pair is the
same two runs from the same snapshot whichever of the 511 cuts is being scored,
and it was run again for every one of them. These pin that the pair is run once
per anchor and reused, and that an anchor collected in a later round keeps the
condition it was paired with when the bank was dealt out.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import v25_runtime
from core.subject.v25_runtime import collect_partition_samples

pytestmark = pytest.mark.unit


class _Frame:
    def __init__(self, value: float) -> None:
        self._value = value

    def vector(self) -> np.ndarray:
        return np.full(3, self._value)


class _Runtime:
    """Counts the turns it is asked for; every fork records two frames."""

    def __init__(self) -> None:
        self.turns = 0
        self.conditions: list[str] = []

    def restore(self, snapshot) -> None:
        pass

    async def turn_once(self, condition):
        self.turns += 1
        self.conditions.append(condition.name)
        return [_Frame(1.0), _Frame(2.0)]


@pytest.fixture
def runtime(monkeypatch) -> _Runtime:
    @contextmanager
    def clamped(_runtime, _domains):
        yield None

    monkeypatch.setattr(v25_runtime, "clamped", clamped)
    monkeypatch.setattr(v25_runtime, "compose", lambda left_rows, right_rows, left: list(left_rows))
    return _Runtime()


CONDITIONS = [SimpleNamespace(name=name) for name in ("rest", "talk", "work")]


def _anchors(count: int) -> list:
    return [SimpleNamespace(snapshot=index, current=np.zeros(2)) for index in range(count)]


@pytest.mark.asyncio
async def test_the_untouched_pair_is_run_once_per_anchor(runtime) -> None:
    untouched: dict = {}
    anchors = _anchors(4)
    for left, right in ((("A",), ("B", "C")), (("A", "B"), ("C",)), (("A", "C"), ("B",))):
        await collect_partition_samples(
            runtime, anchors, CONDITIONS, left=left, right=right, turns=1, lags=(1, 2),
            untouched=untouched,
        )
    # Four arms for each anchor on the first cut, two on each cut after it.
    assert runtime.turns == 4 * 4 + 2 * 4 * 2


@pytest.mark.asyncio
async def test_without_the_cache_every_cut_runs_all_four(runtime) -> None:
    anchors = _anchors(4)
    for _ in range(3):
        await collect_partition_samples(
            runtime, anchors, CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1,),
        )
    assert runtime.turns == 4 * 4 * 3


@pytest.mark.asyncio
async def test_a_later_slice_keeps_the_conditions_it_was_dealt(runtime) -> None:
    anchors = _anchors(5)
    whole = _Runtime()
    await collect_partition_samples(
        whole, anchors, CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1,),
    )
    await collect_partition_samples(
        runtime, anchors[:2], CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1,),
    )
    await collect_partition_samples(
        runtime, anchors[2:], CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1,), offset=2,
    )
    assert runtime.conditions == whole.conditions


@pytest.mark.asyncio
async def test_the_samples_are_the_same_with_and_without_the_cache(runtime) -> None:
    anchors = _anchors(3)
    plain = await collect_partition_samples(
        runtime, anchors, CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1, 2),
    )
    cached = await collect_partition_samples(
        runtime, anchors, CONDITIONS, left=("A",), right=("B", "C"), turns=1, lags=(1, 2),
        untouched={},
    )
    for lag in (1, 2):
        for key in ("context", "intact", "cut", "sham_a", "sham_b", "reached"):
            assert np.array_equal(plain[lag][key], cached[lag][key])
