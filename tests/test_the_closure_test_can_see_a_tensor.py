"""Every array the machine was carrying was invisible to the closure test.

The periphery walk reads numbers off the machine minus the core and asks
whether any of them predict the core's next state. It handled booleans,
numbers, and the length of lists, tuples, dicts and sets, and then recursed
into anything with a `__dict__`.

A numpy array is none of those. It is not a sequence that branch recognised and
it has no instance dictionary, so it fell through every case and was never
read. A broker keeping its state in a tensor — which is where a broker would
keep it — could not have been found by the test that exists to find one.

An array is summarised into a fixed number of columns now: its size, its four
moments, and a seeded Gaussian projection that preserves distances in
expectation. Fixed, so a thousand-unit reservoir costs exactly what a
three-element vector costs and a wide tensor cannot eat the column budget the
rest of the periphery needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.closure import SKETCH_SEED, SKETCH_WIDTH, _is_array, _numbers

pytestmark = pytest.mark.unit


class Holder:
    def __init__(self, value):
        self.hidden = value
        self.count = 3


def _walk(value) -> dict[str, float]:
    out: dict[str, float] = {}
    _numbers(Holder(value), "broker", out)
    return out


def test_an_array_is_read_at_all() -> None:
    """The defect: it fell through every branch and was never seen."""
    out = _walk(np.arange(1000, dtype=float))
    assert any(key.startswith("broker.hidden") for key in out)


def test_two_different_arrays_sketch_differently() -> None:
    """Otherwise a leak through a tensor still could not be found."""
    assert _walk(np.zeros(1000)) != _walk(np.ones(1000))


def test_the_sketch_is_a_fixed_width() -> None:
    """A wide tensor must not eat the budget the rest of the periphery needs."""
    wide = _walk(np.arange(100_000, dtype=float))
    narrow = _walk(np.arange(3, dtype=float))
    assert len(wide) == len(narrow)
    # size, four moments, the projections, and the holder's own counter.
    assert len(wide) == 1 + 1 + 4 + SKETCH_WIDTH


def test_the_moments_and_the_projection_are_both_there() -> None:
    """A projection of a constant array is a constant; the moments say it was one."""
    out = _walk(np.full(64, 2.5))
    assert out["broker.hidden.mean"] == pytest.approx(2.5)
    assert out["broker.hidden.sd"] == pytest.approx(0.0)
    assert out["broker.hidden#"] == 64.0
    assert all(f"broker.hidden.p{i}" in out for i in range(SKETCH_WIDTH))


def test_the_directions_are_the_same_between_runs() -> None:
    """Two runs have to sketch the same directions or nothing is comparable."""
    assert _walk(np.linspace(0, 1, 50)) == _walk(np.linspace(0, 1, 50))
    assert isinstance(SKETCH_SEED, int)


def test_an_empty_array_says_it_was_empty_rather_than_vanishing() -> None:
    out = _walk(np.zeros(0))
    assert out["broker.hidden#"] == 0.0


def test_a_non_finite_entry_does_not_take_the_whole_sketch_with_it() -> None:
    out = _walk(np.array([1.0, np.nan, 3.0, np.inf]))
    assert out["broker.hidden#"] == 2.0
    assert out["broker.hidden.mean"] == pytest.approx(2.0)


def test_the_array_test_does_not_catch_ordinary_objects() -> None:
    assert _is_array(np.zeros(3))
    assert not _is_array([1, 2, 3])
    assert not _is_array({"a": 1})
    assert not _is_array(3.0)
    assert not _is_array(Holder(1))


def test_something_with_a_shape_and_a_dtype_is_treated_as_one() -> None:
    """Torch and MLX tensors answer to both and neither imports cleanly here."""

    class Tensorish:
        shape = (4,)
        dtype = "float32"

        def __array__(self, dtype=None):
            return np.arange(4, dtype=float)

    assert _is_array(Tensorish())
    out = _walk(Tensorish())
    assert out["broker.hidden#"] == 4.0
