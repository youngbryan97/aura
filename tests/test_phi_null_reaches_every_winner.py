"""The runtime's phi core measures whichever estimator wins against its sampling null, and unmeasured is not a complex.

Chen, Zhang, Xu and Zhang ran the runtime's own pre-committed criteria from outside ("Real Computation or Merely
Naming?", arXiv:2610.41604, section 8, 2026). On CPU-only runs the 8-node affective estimator won the live
selection, the hook that attaches the null knew only the two residual estimators, and ``is_complex`` fell back to
``phi_s > 0``: memoryless noise scored phi_s 0.42 to 0.46 and was called a complex (their P1), and coupled dynamics
read no higher than noise (their P2). The null itself, run alone, was right: 0.000 for noise, 0.84 to 0.91 coupled.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.phi_core import PhiCore

pytestmark = pytest.mark.unit

STEPS = 4096


def _memoryless(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, 256)) for _ in range(STEPS)]


def _coupled_ring(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    state, states = 0b10110101, []
    for _ in range(STEPS):
        bits = [(state >> i) & 1 for i in range(8)]
        nxt = [bits[(i - 1) % 8] ^ bits[(i + 1) % 8] for i in range(8)]
        if rng.random() < 0.15:
            nxt[int(rng.integers(0, 8))] ^= 1
        state = sum(bit << i for i, bit in enumerate(nxt))
        states.append(state)
    return states


def _live_selection(states: list[int], *, null_due: bool = True):
    """The live selection with only the affective history filled, as on the audit's CPU-only runs."""
    core = PhiCore()
    core._null_interval_s = 0.0 if null_due else 1e9
    core._last_null_measured_at = 0.0 if null_due else 1e18
    core._affective_state_history.extend(states)
    selection = core.compute_full_kernel_selection()
    assert selection.winner_name == "affective"
    return selection.winner


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noise_that_wins_the_live_selection_meets_its_null_and_is_not_a_complex(seed):
    winner = _live_selection(_memoryless(seed))
    assert winner.null_surrogates > 0 and winner.phi_null_mean is not None
    assert winner.integration_fraction < 0.05
    assert winner.is_complex is False


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_coupled_dynamics_that_win_it_read_above_noise(seed):
    coupled = _live_selection(_coupled_ring(seed))
    noise = _live_selection(_memoryless(seed))
    assert coupled.integration_fraction > noise.integration_fraction + 0.5
    assert coupled.is_complex is True


def test_a_winner_not_yet_measured_is_not_a_complex():
    winner = _live_selection(_memoryless(4), null_due=False)
    assert winner.phi_s > 0.0
    assert winner.phi_null_mean is None and winner.null_surrogates == 0
    assert winner.is_complex is False


def test_every_estimator_the_selection_can_pick_has_a_null():
    core = PhiCore()
    for name in ("residual_stream_grassmann", "residual_stream", "affective", "mesh"):
        history, bipartitions, bit_tables = (getattr(core, attribute) for attribute in PhiCore._NULL_INPUTS[name])
        assert history is not None and bipartitions and bit_tables
