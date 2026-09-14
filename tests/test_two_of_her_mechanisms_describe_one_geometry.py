"""What the content geometry says yes to, and what it must refuse.

A correlation between two distance matrices is easy to get by accident and
easy to get from nothing. These are the cases that decide whether the measure
is worth reading: a structure genuinely shared, a structure that is not there,
a measure that did not vary, and a displacement that moved one geometry and
not the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.content import (
    FLAT,
    PerceptClass,
    agreement,
    design_recovery,
    gauge,
    moves_together,
    pairs,
    spread,
)

SIZE = 8
KEYS = pairs([object()] * SIZE)


def _from_points(points: np.ndarray) -> dict[tuple[int, int], float]:
    return {(i, j): float(np.linalg.norm(points[i] - points[j])) for i, j in KEYS}


def _points(seed: int, dimensions: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(SIZE, dimensions))


def test_one_shared_geometry_is_found() -> None:
    """Two readings of the same layout agree, and the null says it is not luck."""
    points = _points(1)
    internal = _from_points(points)
    # The behavioural reading is the same layout seen through a monotone
    # distortion and noise, which is what a different mechanism looks like.
    rng = np.random.default_rng(2)
    behavioural = {
        k: float(np.sqrt(v) + rng.normal(scale=0.02)) for k, v in internal.items()
    }
    result = agreement(internal, behavioural, size=SIZE, draws=199, seed=3)
    assert result.measured
    assert result.rho > 0.8, result
    assert result.p_value < 0.01, result
    assert result.holds(bar=0.3)


def test_two_unrelated_geometries_are_not_one_structure() -> None:
    """Independent layouts do not pass, and the p-value is not small."""
    internal = _from_points(_points(11))
    behavioural = _from_points(_points(12))
    result = agreement(internal, behavioural, size=SIZE, draws=199, seed=13)
    assert result.measured
    assert not result.holds(bar=0.3), result


def test_a_flat_behavioural_measure_is_not_measured() -> None:
    """Constant distances correlate perfectly and mean nothing, so they refuse."""
    internal = _from_points(_points(21))
    behavioural = {k: 0.5 for k in KEYS}
    result = agreement(internal, behavioural, size=SIZE, draws=99, seed=23)
    assert not result.measured
    assert "behavioural" in result.why
    assert not result.holds(bar=0.0)


def test_a_flat_internal_measure_is_not_measured() -> None:
    internal = {k: 1.25 for k in KEYS}
    behavioural = _from_points(_points(31))
    result = agreement(internal, behavioural, size=SIZE, draws=99, seed=33)
    assert not result.measured
    assert "internal" in result.why


def test_spread_is_zero_when_nothing_varied() -> None:
    assert spread([1.0, 1.0, 1.0]) == 0.0
    assert spread([]) == 0.0
    assert spread([0.0, 1.0, 2.0]) > FLAT


def test_the_design_is_recovered_when_the_state_read_the_percepts() -> None:
    """An internal geometry that follows the grid says the classes arrived."""
    grid = _points(41, dimensions=2)
    classes = [
        PerceptClass(f"c{i}", "kind", "world", 0.5, "x", tuple(grid[i]))
        for i in range(SIZE)
    ]
    internal = _from_points(grid)
    recovery = design_recovery(internal, classes, draws=199, seed=43)
    assert recovery.measured
    assert recovery.rho > 0.95, recovery
    assert recovery.p_value < 0.01


def test_a_flat_internal_geometry_against_a_varying_design_says_so() -> None:
    """The percepts did not reach the state, and that is the finding."""
    grid = _points(51, dimensions=2)
    classes = [
        PerceptClass(f"c{i}", "kind", "world", 0.5, "x", tuple(grid[i]))
        for i in range(SIZE)
    ]
    recovery = design_recovery({k: 0.7 for k in KEYS}, classes, draws=49, seed=53)
    assert recovery.rho == 0.0
    assert "flat" in recovery.why


def test_a_displacement_that_moves_both_geometries_together_is_found() -> None:
    before = _points(61)
    after = before + np.random.default_rng(62).normal(scale=0.6, size=before.shape)
    internal_before = _from_points(before)
    internal_after = _from_points(after)
    rng = np.random.default_rng(63)
    behavioural_before = {
        k: float(np.sqrt(v) + rng.normal(scale=0.01)) for k, v in internal_before.items()
    }
    behavioural_after = {
        k: float(np.sqrt(v) + rng.normal(scale=0.01)) for k, v in internal_after.items()
    }
    sham_internal = {k: v + rng.normal(scale=0.01) for k, v in internal_before.items()}
    sham_behavioural = {
        k: v + rng.normal(scale=0.01) for k, v in behavioural_before.items()
    }
    result = moves_together(
        internal_before, internal_after, behavioural_before, behavioural_after,
        size=SIZE, sham_internal_after=sham_internal,
        sham_behavioural_after=sham_behavioural, draws=199, seed=64,
    )
    assert result.measured, result
    assert result.rho > 0.5, result
    assert result.p_value < 0.01
    assert result.holds(bar=0.3)


def test_a_displacement_the_behaviour_ignored_does_not_pass() -> None:
    """The internal geometry moved and the behavioural one moved elsewhere."""
    before = _points(71)
    after = before + np.random.default_rng(72).normal(scale=0.6, size=before.shape)
    internal_before = _from_points(before)
    internal_after = _from_points(after)
    behavioural_before = _from_points(_points(73))
    behavioural_after = _from_points(_points(74))
    result = moves_together(
        internal_before, internal_after, behavioural_before, behavioural_after,
        size=SIZE, draws=199, seed=75,
    )
    assert not result.holds(bar=0.3), result


def test_the_gauge_limit_is_carried_in_every_report() -> None:
    """A reader with the correlation and not this will read structure as quality."""
    limits = gauge()
    assert "identified_up_to_isomorphism" in limits
    assert "structure_is_not_character" in limits
    assert all(isinstance(v, str) and v for v in limits.values())


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_the_null_does_not_pass_on_noise(seed: int) -> None:
    """Five independent draws of nothing. The measure must not find structure."""
    internal = _from_points(_points(seed))
    behavioural = _from_points(_points(seed + 1))
    result = agreement(internal, behavioural, size=SIZE, draws=199, seed=seed + 2)
    assert not result.holds(bar=0.3), (seed, result)
