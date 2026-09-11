"""Shift every unit against every other one and see what survives.

A run of non-silent bins is a cascade only if the units in it are related. Roll
each unit's own spike train by its own random amount and every firing rate is
untouched while every coincidence between units is gone, so whatever
distribution comes through unchanged was the recording's occupancy rather than
its cascades. Beggs and Plenz ran this control against their own arrays.

What gets compared is how much the number of units active in a bin varies.
Independent units make that the sum of their own variances, which is what the
shuffled ensemble measures, and units that recruit each other push it above.
Comparing the avalanche exponent instead was tried first and it called
independent Poisson structured and real recruitment flat: at high occupancy the
avalanches are separated by whichever bins happen to be silent, so rolling the
units moves the largest cascade for reasons that are not recruitment.
"""

from __future__ import annotations

import numpy as np

from core.connectome.criticality import cascades_beyond_rate


def _independent(ticks: int, units: int, rate: float, seed: int) -> np.ndarray:
    """Units that fire on their own, with no cascade anywhere in them."""
    rng = np.random.default_rng(seed)
    return (rng.random((ticks, units)) < rate).astype(np.float64)


def _cascading(ticks: int, units: int, seed: int) -> np.ndarray:
    """Bursts where one unit's firing recruits others in the next bin."""
    rng = np.random.default_rng(seed)
    raster = np.zeros((ticks, units), dtype=np.float64)
    tick = 0
    while tick < ticks - 6:
        recruited = rng.choice(units, size=2, replace=False)
        step = 0
        while len(recruited) and tick + step < ticks - 1 and step < 6:
            raster[tick + step, recruited] = 1.0
            keep = max(0, int(rng.poisson(len(recruited) * 1.1)))
            recruited = rng.choice(units, size=min(keep, units), replace=False)
            step += 1
        tick += step + int(rng.integers(3, 9))
    return raster


def test_a_busy_recording_with_no_cascades_in_it_is_named() -> None:
    """Independent units at 92% occupancy, which is the case that fooled the
    first version of this control."""
    report = cascades_beyond_rate(_independent(4000, 40, 0.06, seed=3))

    assert report["active_fraction"] > 0.9
    assert report["sigmas_above_independence"] < 3.0
    assert "not a cascade measurement" in report["verdict"]


def test_a_sparse_recording_with_no_cascades_in_it_is_named() -> None:
    report = cascades_beyond_rate(_independent(4000, 40, 0.01, seed=3))

    assert report["sigmas_above_independence"] < 3.0
    assert "not a cascade measurement" in report["verdict"]


def test_units_that_recruit_each_other_are_named() -> None:
    report = cascades_beyond_rate(_cascading(6000, 60, seed=4))

    assert report["sigmas_above_independence"] > 3.0
    assert report["population_variance"] > report["shuffled_variance"]
    assert "cascades" in report["verdict"]


def test_the_shuffle_leaves_every_firing_rate_alone() -> None:
    """The control is only a control if it changes nothing but the coincidences."""
    raster = _cascading(3000, 30, seed=5)
    rng = np.random.default_rng(11)
    rolled = np.empty_like(raster)
    for unit in range(raster.shape[1]):
        rolled[:, unit] = np.roll(raster[:, unit], int(rng.integers(0, raster.shape[0])))

    assert np.array_equal(raster.sum(axis=0), rolled.sum(axis=0))
    assert raster.sum() == rolled.sum()


def test_an_empty_recording_is_refused() -> None:
    assert cascades_beyond_rate(np.zeros((0, 0)))["verdict"] == "no recording"
