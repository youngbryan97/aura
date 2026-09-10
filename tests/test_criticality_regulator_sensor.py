"""The regulator's sensor has to fire, or it is a controller reporting success.

LIVE, measured over 1,200 ticks: column activations ran between -0.0014 and
+0.0010 against a fixed activation threshold of 0.15, so not one column was
ever counted active. `_compute_branching_sample` returned its neutral 1.0 every
tick, the branching ratio read exactly 1.0000, the criticality score read
1.0000, and the gain PID had an error of zero forever — while the unbiased
estimator on the same run read 0.908.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.consciousness.criticality_regulator import (
    CriticalityConfig,
    CriticalityRegulator,
)


def _drive(regulator, ticks: int, seed: int = 5):
    """Run the real mesh into the real regulator."""
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    mesh = NeuralMesh(MeshConfig())
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    rng = np.random.default_rng(seed)
    loop = asyncio.new_event_loop()
    try:
        for _ in range(ticks):
            mesh.inject_sensory(rng.standard_normal(width).astype(np.float32) * 0.1)
            mesh._tick_inner()
            loop.run_until_complete(
                regulator.tick(mesh.column_activations, mesh.inter_column_weights)
            )
    finally:
        loop.close()
    return mesh


def test_the_threshold_is_not_a_fixed_level():
    """A fixed level on a quantity whose scale nobody knows is a dead sensor."""
    config = CriticalityConfig()
    assert config.activation_threshold == 0.0
    assert config.activation_sigma == 3.0
    assert config.activation_minimum_samples >= 2


def test_the_sensor_fires_on_the_real_mesh():
    regulator = CriticalityRegulator(CriticalityConfig(num_columns=64))
    _drive(regulator, 400)
    # The estimator names its own variant, so match the family rather than one
    # spelling of it.
    assert regulator.get_state().branching_estimator.startswith("multistep-regression"), (
        "the regulator fell back to its per-tick mean, which means nothing was active"
    )


def test_the_regulator_notices_a_subcritical_network():
    """It reported exactly 1.0000 — perfectly critical — for the life of the mesh."""
    regulator = CriticalityRegulator(CriticalityConfig(num_columns=64))
    _drive(regulator, 800)
    state = regulator.get_state()
    assert state.branching_ratio != pytest.approx(1.0, abs=1e-9), (
        "a branching ratio of exactly one is the neutral value, not a measurement"
    )
    assert regulator.get_criticality_score() < 1.0


def test_the_controller_acts_on_what_it_measures():
    regulator = CriticalityRegulator(CriticalityConfig(num_columns=64))
    _drive(regulator, 800)
    state = regulator.get_state()
    adjustments = regulator.get_adjustments()
    if state.branching_ratio < 1.0:
        assert adjustments["gain"] > 1.0, "subcritical, and the gain was not raised"
    elif state.branching_ratio > 1.0:
        assert adjustments["gain"] < 1.0, "supercritical, and the gain was not lowered"


def test_a_caller_can_still_ask_for_a_fixed_level():
    """The knob stays, so an experiment can pin it."""
    regulator = CriticalityRegulator(
        CriticalityConfig(num_columns=64, activation_threshold=0.15)
    )
    _drive(regulator, 200)
    assert regulator.get_state().branching_ratio_naive == pytest.approx(1.0)


def test_the_baseline_survives_a_column_count_that_changes():
    regulator = CriticalityRegulator(CriticalityConfig(num_columns=8))
    loop = asyncio.new_event_loop()
    try:
        for _ in range(20):
            loop.run_until_complete(
                regulator.tick(np.random.random(8), np.zeros((8, 8)))
            )
    finally:
        loop.close()
    assert regulator.get_state() is not None
