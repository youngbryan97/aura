"""Ridge with more columns than rows is solved in its dual, and gives the primal's weights."""
from __future__ import annotations

import numpy as np
import pytest

from core.subject.estimate import _ridge

pytestmark = pytest.mark.unit


def _primal(x: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    return np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)


@pytest.mark.parametrize("columns", [80, 300])
def test_the_dual_gives_the_primal_weights(columns: int) -> None:
    rng = np.random.default_rng(columns)
    x = rng.normal(size=(60, columns))
    y = rng.normal(size=(60, 3))
    penalty = np.concatenate([np.full(20, 0.01), np.full(columns - 20, 100.0)])
    assert np.allclose(_ridge(x, y, penalty), _primal(x, y, penalty), atol=1e-8)
    assert np.allclose(_ridge(x, y, 1.0), _primal(x, y, np.full(columns, 1.0)), atol=1e-8)


def test_a_switched_off_block_still_reads_exact_zeros_when_wide() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 120))
    y = rng.normal(size=(40, 2))
    penalty = np.concatenate([np.full(30, 1.0), np.full(90, np.inf)])
    weights = _ridge(x, y, penalty)
    assert np.all(weights[30:] == 0.0)
    assert np.allclose(weights[:30], _primal(x[:, :30], y, np.full(30, 1.0)))
