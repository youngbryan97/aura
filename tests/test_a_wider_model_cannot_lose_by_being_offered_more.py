"""A model offered more information must not do worse than one offered less.

That is a fact about nested hypothesis classes. `_penalties` already puts the
narrow model exactly inside the wide model's grid, so the fallback exists —
and run_031 still read a negative gain on both sides of its cheapest cut, each
half predicting itself better than the whole system predicted it (own 0.2633
against intact 0.2731, and own 0.1734 against intact 0.1855).

The fallback existing is not the same as it being chosen. The wide model is
given twice as many candidates to score on the same validation rows, so a
finite addition that wins there by chance is taken and then loses on the test
rows. The asymmetry runs one way, against the model offered more.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.estimate import fit_predict, split_rows

ROWS = 1500
WIDTH = 4
DRAWS = 8


def _gain(mine: np.ndarray, theirs: np.ndarray, target: np.ndarray) -> float:
    train, validate, test = split_rows(mine.shape[0])
    own = fit_predict(mine, target, train=train, validate=validate, test=test)
    whole = fit_predict(
        np.hstack([mine, theirs]),
        target,
        train=train,
        validate=validate,
        test=test,
        own_width=mine.shape[1],
    )
    if own.base <= 0.0:
        return 0.0
    return (own.sse - whole.sse) / own.base


def _noise_addition(rng: np.random.Generator):
    mine = rng.normal(size=(ROWS, WIDTH))
    target = mine @ rng.normal(size=(WIDTH, WIDTH)) + 0.5 * rng.normal(size=(ROWS, WIDTH))
    return mine, rng.normal(size=(ROWS, WIDTH * 5)), target


def _informative_addition(rng: np.random.Generator):
    mine = rng.normal(size=(ROWS, WIDTH))
    theirs = rng.normal(size=(ROWS, WIDTH * 5))
    target = (
        mine @ rng.normal(size=(WIDTH, WIDTH))
        + theirs @ rng.normal(size=(WIDTH * 5, WIDTH))
        + 0.5 * rng.normal(size=(ROWS, WIDTH))
    )
    return mine, theirs, target


def test_an_addition_that_carries_nothing_costs_nothing():
    """The null that matters. Under argmin this read a negative mean with five
    draws of twelve below zero; the addition is now switched off exactly."""
    gains = [_gain(*_noise_addition(np.random.default_rng(seed))) for seed in range(DRAWS)]
    assert min(gains) >= 0.0, gains
    assert max(gains) == pytest.approx(0.0, abs=1e-9), gains


def test_an_addition_that_carries_something_is_still_taken():
    """A rule that only ever switched the addition off would pass the test
    above and measure nothing, which is the failure mode to guard."""
    gains = [_gain(*_informative_addition(np.random.default_rng(seed))) for seed in range(DRAWS)]
    assert min(gains) > 0.5, gains


def test_destroying_the_structure_and_keeping_the_variance_reads_near_zero():
    gains = []
    for seed in range(DRAWS):
        rng = np.random.default_rng(seed)
        mine, theirs, target = _informative_addition(rng)
        gains.append(_gain(mine, theirs[rng.permutation(ROWS)], target))
    assert max(gains) < 0.02, gains
    assert min(gains) > -0.02, gains


def test_the_rule_prefers_the_addition_switched_off_among_equals():
    """Ties go to the more heavily penalised candidate, which for the wide
    grid is the narrow model — not to whichever index came first."""
    from core.subject.estimate import ALPHAS, _penalties, _within_one_standard_error

    grid = _penalties(WIDTH * 2, WIDTH)
    scores = np.full((len(grid), 1), 1.0)
    spread = np.full((len(grid), 1), 0.1)
    chosen = int(_within_one_standard_error(scores, spread, grid)[0])
    assert grid[chosen][1] == float("inf")
    assert grid[chosen][0] == max(ALPHAS)
