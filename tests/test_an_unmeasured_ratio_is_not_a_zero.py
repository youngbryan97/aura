"""A sensor that fails toward zero drives the controller into the failure.

The regulator steers gain, noise and excitation from a branching ratio. When
the multistep regression is rejected it used to fall back to the per-tick mean,
whose failure mode is zero: a saturated mesh has no newly active units, every
sample comes back 0, and the controller reads 0.0000. That is not "maximally
subcritical" — it is no reading at all, and the PID answers it with a constant
maximal demand for more gain and more noise, which is what saturated the mesh.

A 6,000-tick run walked into exactly that: branching 0.0000, gain and noise and
excitation all at their rails, 97% of bins carrying a spike. An unmeasured
quantity is not zero, so the controller holds instead.
"""

from __future__ import annotations

import asyncio

import numpy as np

from core.consciousness.criticality_regulator import (
    CriticalityConfig,
    CriticalityRegulator,
)


def _regulator(columns: int = 64) -> CriticalityRegulator:
    return CriticalityRegulator(CriticalityConfig(num_columns=columns))


def _drive(regulator: CriticalityRegulator, activations, weights, ticks: int) -> None:
    loop = asyncio.new_event_loop()
    try:
        for _ in range(ticks):
            loop.run_until_complete(regulator.tick(activations, weights))
    finally:
        loop.close()


def test_a_flat_recording_leaves_the_outputs_where_they_were() -> None:
    """Nothing moves. Nothing can be measured, so nothing is decided."""
    columns = 64
    regulator = _regulator(columns)
    flat = np.zeros(columns, dtype=np.float64)
    weights = np.zeros((columns, columns), dtype=np.float64)

    before = dict(regulator.get_adjustments())
    _drive(regulator, flat, weights, 300)
    after = dict(regulator.get_adjustments())

    assert after == before, (
        f"a regulator that cannot measure anything moved its outputs from "
        f"{before} to {after}"
    )


def test_a_flat_recording_is_reported_as_unmeasured_not_as_zero() -> None:
    columns = 64
    regulator = _regulator(columns)
    flat = np.zeros(columns, dtype=np.float64)
    weights = np.zeros((columns, columns), dtype=np.float64)

    _drive(regulator, flat, weights, 300)

    assert regulator._branching_measured is False
    assert regulator._branching_estimator == "unmeasured"
    # The held value, not the per-tick mean's zero.
    assert regulator._branching_ratio > 0.0


def test_the_per_tick_mean_never_drives_the_controller() -> None:
    """It is reported and it is biased toward zero; those cannot both be safe."""
    columns = 64
    regulator = _regulator(columns)
    flat = np.zeros(columns, dtype=np.float64)
    weights = np.zeros((columns, columns), dtype=np.float64)

    _drive(regulator, flat, weights, 300)
    state = regulator.get_state()

    assert state.branching_estimator == "unmeasured"
    assert state.branching_ratio != 0.0, (
        "the controller is reading the per-tick mean's zero as a measurement"
    )


def test_a_measurable_recording_does_move_the_outputs() -> None:
    """Holding still on a blind sensor must not mean holding still always.

    The activity here is an actual branching process — each tick's active count
    is Poisson around m times the last one, plus a trickle of spontaneous
    drive — because that is the model the multistep regression fits. A sine
    plus noise is not one, and the estimator is right to refuse it.
    """
    columns = 64
    regulator = _regulator(columns)
    rng = np.random.default_rng(5)
    weights = rng.normal(0.0, 0.1, (columns, columns))

    loop = asyncio.new_event_loop()
    try:
        moved = False
        before = dict(regulator.get_adjustments())
        # Sparse and large, not dense and moderate. A column's threshold is
        # three sigma of its OWN running variance, so excursions that arrive
        # often enough inflate the variance that is supposed to detect them:
        # the first version of this test drove a tenth of the columns every
        # tick and recorded a mean activity of 0.01 columns over six hundred
        # ticks. Roughly three of sixty-four keeps each column's baseline
        # spread set by its noise, where three sigma can still be crossed.
        count = 3
        for _ in range(800):
            count = min(columns - 1, int(rng.poisson(0.9 * count)) + int(rng.poisson(0.6)))
            activations = rng.normal(0.0, 0.05, columns)
            if count:
                activations[rng.choice(columns, size=count, replace=False)] += 1.0
            loop.run_until_complete(regulator.tick(activations, weights))
            if dict(regulator.get_adjustments()) != before:
                moved = True
    finally:
        loop.close()

    assert regulator._branching_measured, "the estimator never fitted this recording"
    assert moved, "a regulator with a working sensor never moved an output"
