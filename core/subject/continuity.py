"""Whether she carries on across a restart or a move: the same dynamics before and after.

A lineage says which stage came from which. It cannot say the process that
resumed works the way the one that stopped did: a state restored into
different machinery is the same record in a different mind. So the transition
itself is compared. A model of how her core goes from one turn to the next,
fitted on the turns before the break, predicts the turns after it; so does one
fitted after the break, on its own earlier turns. If the first does as well as
the second, the way she moves carried over. If it does much worse, something
other than her state changed at the break.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["transition_match"]


def _ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)


def _loss(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean((x @ weights - y) ** 2))


def transition_match(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """How well a transition fitted before a break predicts after it, against one fitted after.

    `before` and `after` are rows of her core state, one per turn, in the same
    columns. Both sides are standardised on the turns before the break, so a
    shift in scale at the break counts against the match. The penalty is
    chosen on the second half of the turns before the break.
    """
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    centre = before.mean(axis=0)
    spread = before.std(axis=0)
    keep = spread > 1e-12
    z_before = (before[:, keep] - centre[keep]) / spread[keep]
    z_after = (after[:, keep] - centre[keep]) / spread[keep]
    x_b, y_b = z_before[:-1], z_before[1:]
    x_a, y_a = z_after[:-1], z_after[1:]
    half_b, half_a = x_b.shape[0] // 2, x_a.shape[0] // 2
    width = x_b.shape[1]
    penalties = [width * 10.0 ** k for k in range(-3, 3)]
    chosen = min(penalties, key=lambda p: _loss(x_b[half_b:], y_b[half_b:], _ridge(x_b[:half_b], y_b[:half_b], p)))
    carried = _ridge(x_b, y_b, chosen)
    own = _ridge(x_a[:half_a], y_a[:half_a], chosen)
    test_x, test_y = x_a[half_a:], y_a[half_a:]
    from_before = _loss(test_x, test_y, carried)
    from_after = _loss(test_x, test_y, own)
    return {
        "loss_with_the_dynamics_from_before": round(from_before, 6),
        "loss_with_the_dynamics_from_after": round(from_after, 6),
        "ratio": round(from_before / from_after, 6) if from_after > 0 else None,
        "turns_before": int(before.shape[0]),
        "turns_after": int(after.shape[0]),
    }
