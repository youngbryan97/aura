"""Held-out sufficiency separates a grain that is right from one that is missing a dimension.

The check the v25 runner made subtracted one shuffled gain from the gain raw
history added. On a system whose futures run through a two-dimensional
macrostate, the correct two-dimensional grain and a one-dimensional grain
both came out with a margin near one, because the extra features overfit, the
shuffled gain went negative, and history that determines the state exactly
improves on any noisy estimate of it. A sufficiency check that cannot tell a
missing dimension from estimation noise refuses every grain and so decides
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _macro(seed: int, *, anchors: int = 80, micro: int = 12, macro: int = 2, columns: int = 48, noise: float = 0.05):
    rng = np.random.default_rng(seed)
    micro_state = rng.normal(size=(anchors, micro))
    macrostate = micro_state @ rng.normal(size=(macro, micro)).T
    signatures = macrostate @ rng.normal(size=(columns, macro)).T + noise * rng.normal(size=(anchors, columns))
    future = macrostate @ rng.normal(size=(columns, macro)).T + noise * rng.normal(size=(anchors, columns))
    return micro_state, signatures, future


def _verdict(history: np.ndarray, state: np.ndarray, future: np.ndarray, *, seed: int) -> tuple[float, float]:
    from core.subject.intrinsic_v25 import heldout_new_direction_gain

    gain = heldout_new_direction_gain(history, state, future, seed=seed)
    rng = np.random.default_rng(seed)
    shuffled = [
        heldout_new_direction_gain(history[rng.permutation(len(history))], state, future, seed=seed)
        for _ in range(64)
    ]
    return gain, float(np.quantile(shuffled, 0.99))


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_a_correct_grain_leaves_history_no_new_direction(seed: int) -> None:
    from core.subject.intrinsic_v25 import fit_predictive_grain

    history, signatures, future = _macro(seed)
    grain = fit_predictive_grain(signatures)
    assert grain.rank == 2
    gain, _ = _verdict(history, grain.transform(signatures), future, seed=seed)
    assert gain <= 0.05, gain


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_a_grain_missing_a_dimension_is_caught(seed: int) -> None:
    from core.subject.intrinsic_v25 import fit_predictive_grain

    history, signatures, future = _macro(seed)
    state = fit_predictive_grain(signatures).transform(signatures)[:, :1]
    gain, quantile = _verdict(history, state, future, seed=seed)
    assert gain > 0.05 and gain > quantile, (gain, quantile)


def test_the_fractional_gain_could_not_tell_them_apart() -> None:
    """Kept as the record of why the statistic changed."""
    from core.subject.intrinsic_v25 import fit_predictive_grain, heldout_sufficiency_gain

    history, signatures, future = _macro(21)
    state = fit_predictive_grain(signatures).transform(signatures)
    rng = np.random.default_rng(121)
    shuffled = history[rng.permutation(len(history))]
    margin_right = heldout_sufficiency_gain(history, state, future) - heldout_sufficiency_gain(shuffled, state, future)
    margin_missing = heldout_sufficiency_gain(history, state[:, :1], future) - heldout_sufficiency_gain(shuffled, state[:, :1], future)
    assert margin_right > 0.05 and margin_missing > 0.05
