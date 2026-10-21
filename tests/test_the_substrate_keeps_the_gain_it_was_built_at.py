"""Her substrate's weights hold the gain they were built at, and learn every tick.

On every run until 21 October 2026 the substrate's weights went round an exact
five-turn cycle: their spread read 0.0195, 0.0213, 0.0231, 0.0249, 0.0266 and
then 0.0195 again, turn after turn, whatever she was doing. The homeostatic
scaling aimed the mean absolute weight at 0.3, a 64-neuron engine's number; on
512 neurons that target has a largest singular value near seventeen, so each
turn's regulation pushed every weight up 8.7% until the spectral cap of 3.0
pulled the matrix back. That cycle was 91% of recurrent cognition's turn-to-turn
change in the battery's recording, and nothing outside it could predict it or
read it, so C was the cheapest cut of all 511.

The Hebbian step was the cycle's other half: it read the coactivity of one tick
in a hundred and added a hundred ticks' learning at once, and on the experiment
clock a hundred ticks is five turns.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.liquid_substrate_learning import PLASTICITY_EVERY
from core.consciousness.stdp_learning import STDPLearningEngine, gain_one_mean_abs


def _built(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal((n, n)) / np.sqrt(n)
    np.fill_diagonal(weights, 0.0)
    return weights


@pytest.mark.parametrize("n", [16, 64, 512])
def test_the_set_point_is_the_mean_of_a_gain_one_draw(n):
    draws = np.abs(_built(n, seed=3))
    off_diagonal = draws[~np.eye(n, dtype=bool)]
    assert gain_one_mean_abs(n) == pytest.approx(float(off_diagonal.mean()), rel=0.15)


def test_a_network_at_the_gain_it_was_built_at_is_left_where_it_is():
    """Twenty-five regulations of an untouched 512-neuron matrix move nothing.

    Before the fix each one multiplied the matrix by about 1.087, and the
    spread climbed until the spectral cap cut it back.
    """
    engine = STDPLearningEngine(n_neurons=512)
    weights = _built(512)
    spreads = []
    for _ in range(25):
        weights = engine.regulate(weights)
        spreads.append(float(weights.std()))
    assert max(spreads) / min(spreads) < 1.0 + 1e-9
    assert np.linalg.norm(weights, ord=2) < 3.0


def test_a_network_that_has_drifted_is_brought_back_without_a_cycle():
    """Shrunk to the spread her runs ended at, the scaling walks it back up and stops.

    The walk is monotone: no step goes down, which is what a cap cutting the
    matrix back would look like.
    """
    engine = STDPLearningEngine(n_neurons=512)
    weights = _built(512) * (0.0195 / 0.0442)
    spreads = [float(weights.std())]
    for _ in range(400):
        weights = engine.regulate(weights)
        spreads.append(float(weights.std()))
    steps = np.diff(spreads)
    assert (steps >= -1e-12).all()
    target = gain_one_mean_abs(512)
    assert float(np.mean(np.abs(weights))) == pytest.approx(target, rel=0.11)


def test_hebbian_learning_runs_every_tick_at_a_ticks_share():
    """A hundred ticks of one coactivity learn what one old plasticity step did."""
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=16))
    rng = np.random.default_rng(11)
    substrate.x = rng.uniform(-0.5, 0.5, 16).astype(substrate.x.dtype)
    substrate._current_phi = 0.0
    before = np.array(substrate.W, copy=True)
    for _ in range(PLASTICITY_EVERY):
        substrate._learn_from_this_tick_sync()
    expected = before + substrate.config.hebbian_rate * np.tanh(np.outer(substrate.x, substrate.x))
    assert np.allclose(substrate.W, expected, atol=1e-9)


def test_one_tick_learns_from_that_tick_and_not_from_another():
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=16))
    substrate._current_phi = 0.0
    substrate.x = np.zeros(16, dtype=substrate.x.dtype)
    still = np.array(substrate.W, copy=True)
    substrate._learn_from_this_tick_sync()
    assert np.array_equal(substrate.W, still)
    substrate.x = np.full(16, 0.5, dtype=substrate.x.dtype)
    substrate._learn_from_this_tick_sync()
    assert not np.array_equal(substrate.W, still)


def test_the_plasticity_step_no_longer_adds_a_hundred_ticks_at_once():
    """It records spike timing and holds the bounds; the coactivity is the tick's job."""
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=16))
    substrate.W = _built(16, seed=8).astype(substrate.W.dtype)
    substrate.x = np.full(16, 0.9, dtype=substrate.x.dtype)
    substrate._current_phi = 0.0
    before = np.array(substrate.W, copy=True)
    substrate._apply_plasticity_sync()
    assert np.allclose(substrate.W, before, atol=1e-12)


def test_the_counted_body_learns_every_tick_as_the_loop_does():
    from core.subject.snapshot import SUBSTRATE_BODY

    every = {name: count for name, count, _ in SUBSTRATE_BODY}
    assert every["_learn_from_this_tick"] == 1
    assert every["_apply_plasticity"] == PLASTICITY_EVERY
    names = [name for name, _, _ in SUBSTRATE_BODY]
    assert names.index("_learn_from_this_tick") < names.index("_apply_plasticity")


def _with_a_learned_mode(n: int = 512, seed: int = 4) -> np.ndarray:
    """A bulk shrunk to 0.02 under a rank-one mode at the cap: her persisted weights' shape."""
    rng = np.random.default_rng(seed)
    bulk = rng.standard_normal((n, n)) * 0.02
    direction = rng.standard_normal(n)
    direction /= np.linalg.norm(direction)
    weights = bulk + 2.9 * np.outer(direction, direction)
    np.fill_diagonal(weights, 0.0)
    return weights * (3.0 / np.linalg.norm(weights, ord=2))


def test_a_network_with_a_mode_at_the_cap_is_not_scaled_into_it():
    """The look of 21 October: the set point pushed 0.6% a turn and the cap cut back every fifth."""
    engine = STDPLearningEngine(n_neurons=512)
    weights = _with_a_learned_mode()
    assert float(np.mean(np.abs(weights))) < gain_one_mean_abs(512) / 1.1
    spreads = []
    for _ in range(25):
        weights = engine.regulate(weights)
        spreads.append(float(weights.std()))
    assert max(spreads) / min(spreads) < 1.0 + 1e-6
    assert np.linalg.norm(weights, ord=2) <= 3.0 + 1e-9


def test_the_gain_is_held_at_the_cap_tick_by_tick():
    """Learning that pushes the mode over the cap is held there as it arrives, with no lump."""
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=128))
    weights = _with_a_learned_mode(128, seed=9)
    substrate.W = weights.astype(substrate.W.dtype)
    substrate._current_phi = 0.0
    rng = np.random.default_rng(2)
    pattern = rng.uniform(-0.6, 0.6, 128)
    norms = []
    for tick in range(400):
        substrate.x = (pattern + 0.05 * rng.standard_normal(128)).astype(substrate.x.dtype)
        substrate._learn_from_this_tick_sync()
        if tick % PLASTICITY_EVERY == 0:
            substrate._apply_plasticity_sync()
        norms.append(float(np.linalg.norm(substrate.W, ord=2)))
    held = np.asarray(norms[20:])
    assert held.max() <= 3.0 * 1.01
    # No step down larger than a tick's learning: the cap is not arriving in lumps.
    assert np.diff(held).min() > -0.01


def test_the_plasticity_step_leaves_a_network_at_its_built_gain_alone():
    """A flat Frobenius clip of 10 cut her 512-neuron network to a spread of 10 / 512."""
    from core.consciousness.stdp_learning import frobenius_ceiling

    assert frobenius_ceiling(64) > 8.0 and frobenius_ceiling(512) > np.sqrt(512)
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=512))
    substrate.W = _built(512).astype(substrate.W.dtype)
    substrate.x = np.zeros(512, dtype=substrate.x.dtype)
    before = float(np.linalg.norm(substrate.W))
    substrate._apply_plasticity_sync()
    assert float(np.linalg.norm(substrate.W)) == pytest.approx(before, rel=1e-6)


def test_her_campaign_schedule_has_no_cycle_left_in_it():
    """Twenty ticks a turn, a regulation each turn, a plasticity step every hundred ticks."""
    n = 512
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=n))
    substrate.W = _with_a_learned_mode(n, seed=4).astype(substrate.W.dtype)
    substrate._current_phi = 0.0
    engine = STDPLearningEngine(n_neurons=n)
    rng = np.random.default_rng(3)
    mode = np.linalg.svd(substrate.W)[2][0]
    spreads = []
    for turn in range(30):
        for step in range(20):
            substrate.x = np.tanh(0.3 * rng.standard_normal(n) + 0.5 * mode).astype(substrate.x.dtype)
            substrate._learn_from_this_tick_sync()
            if (turn * 20 + step) % PLASTICITY_EVERY == 0:
                substrate._apply_plasticity_sync()
        substrate.W = engine.regulate(substrate.W)
        spreads.append(float(substrate.W.std()))
    ratios = np.asarray(spreads[1:]) / np.asarray(spreads[:-1])
    assert ratios.min() > 0.999 and ratios.max() < 1.001
