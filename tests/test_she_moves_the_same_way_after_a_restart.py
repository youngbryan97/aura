"""Continuity across a break is read as the same dynamics before and after it."""
from __future__ import annotations

import numpy as np

from core.subject.continuity import transition_match


def _life(matrix: np.ndarray, turns: int, seed: int, start: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((turns, matrix.shape[0]))
    x[0] = start if start is not None else rng.normal(size=matrix.shape[0])
    for t in range(1, turns):
        x[t] = matrix @ x[t - 1] + rng.normal(0, 0.3, matrix.shape[0])
    return x


def _dynamics(seed: int, size: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(size, size))
    return 0.8 * m / np.max(np.abs(np.linalg.eigvals(m)))


def test_the_same_mind_resumed_moves_the_way_it_did():
    m = _dynamics(1)
    before = _life(m, 400, seed=2)
    after = _life(m, 400, seed=3, start=before[-1])
    assert transition_match(before, after)["ratio"] < 1.2


def test_the_same_state_in_different_machinery_does_not():
    before = _life(_dynamics(1), 400, seed=2)
    after = _life(_dynamics(9), 400, seed=3, start=before[-1])
    assert transition_match(before, after)["ratio"] > 1.5
