"""Prospective nonlinear closure predictor; never used by the official verdict.

The linear estimator can call a nonlinear copy of K an outside predictor, and
its marginal screen can discard an interacting cause. This candidate expands
both sides before fitting, using only training rows for centering and scaling.
It retains the narrow model as a restriction of the wider model. Finite fits
can still lose on test rows, so their signed gains are reported without clipping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.subject.estimate import ALPHAS, Fit, _standardise, fit_predict


@dataclass(frozen=True)
class Comparison:
    narrow: Fit
    wide: Fit
    own_width: int
    added_width: int

    @property
    def gain(self) -> float | None:
        if self.narrow.degenerate or self.wide.degenerate or self.narrow.sse <= 1e-12:
            return None
        return (self.narrow.sse - self.wide.sse) / self.narrow.sse


def _quadratic(block: np.ndarray, *, seed: int, budget: int = 64) -> np.ndarray:
    """Squares and pair interactions, with a bounded random basis for wide inputs."""
    width = block.shape[1]
    if not width:
        return block
    if width <= 12:
        return np.column_stack(
            [block[:, i] * block[:, j] for i in range(width) for j in range(i, width)]
        )
    # Keep every coordinate's square. A projection alone dilutes a sparse
    # nonlinear cause as the number of unrelated coordinates grows.
    rng = np.random.default_rng(seed)
    left = block @ (rng.choice([-1., 1.], size=(width, budget)) / np.sqrt(width))
    right = block @ (rng.choice([-1., 1.], size=(width, budget)) / np.sqrt(width))
    return np.hstack([block * block, left * right])


def feature_blocks(
    own: np.ndarray, outside: np.ndarray, *, train: slice, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A fixed own-state basis and added terms; no test labels enter either."""
    own, _ = _standardise(own, own[train])
    outside, _ = _standardise(outside, outside[train])
    baseline = np.hstack([own, _quadratic(own, seed=seed)])
    added = np.hstack([outside, _quadratic(outside, seed=seed + 1)])
    if own.shape[1] and outside.shape[1]:
        if own.shape[1] * outside.shape[1] <= 144:
            cross = np.column_stack([
                own[:, i] * outside[:, j]
                for i in range(own.shape[1]) for j in range(outside.shape[1])
            ])
        else:
            rng = np.random.default_rng(seed + 2)
            a = own @ (rng.choice([-1., 1.], size=(own.shape[1], 64)) / np.sqrt(own.shape[1]))
            b = outside @ (rng.choice([-1., 1.], size=(outside.shape[1], 64)) / np.sqrt(outside.shape[1]))
            cross = a * b
        added = np.hstack([added, cross])
    return baseline, added


def compare(
    own: np.ndarray, outside: np.ndarray, target: np.ndarray, *,
    train: slice, validate: slice, test: slice, seed: int = 0,
) -> Comparison:
    """Fit the two declared bases on identical rows without marginal screening."""
    baseline, added = feature_blocks(own, outside, train=train, seed=seed)
    narrow = fit_predict(baseline, target, train=train, validate=validate, test=test, paired_selection=True)
    # Screen the expanded basis, so a product can enter even when neither
    # factor has marginal correlation. Screening raw P would miss that cause.
    from core.subject.closure import _added_variable_scores

    if added.shape[1]:
        alpha = max(narrow.alpha, 1e-3)
        score = _added_variable_scores(baseline, target, added, train, alpha)
        shifted = added.copy()
        count = train.stop - (train.start or 0)
        shifted[train] = np.roll(added[train], max(1, count // 3), axis=0)
        threshold = _added_variable_scores(baseline, target, shifted, train, alpha).max()
        added = added[:, score > threshold]
    # A narrow fit can need weak shrinkage while the outside needs strong
    # shrinkage. The official diagonal grid cannot express that combination.
    width = baseline.shape[1] + added.shape[1]
    grid = []
    for own_alpha in ALPHAS:
        for extra_alpha in (float("inf"), *ALPHAS):
            penalty = np.full(width, extra_alpha)
            penalty[:baseline.shape[1]] = own_alpha
            grid.append((own_alpha, extra_alpha, penalty))
    wide = fit_predict(
        np.hstack([baseline, added]), target, train=train, validate=validate, test=test,
        own_width=baseline.shape[1],
        penalty_grid=grid,
        paired_selection=True,
    )
    return Comparison(narrow, wide, baseline.shape[1], added.shape[1])


def gaussian_world(
    *, seed: int, rows: int, effect: float, outside_width: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known conditional partial R-squared: effect, independent of the estimator.

    Y = K squared + sqrt(effect) P0 + sqrt(1-effect) noise.
    Given K, the residual variance is one; given K and P it is 1-effect.
    The oracle risk reduction is therefore exactly effect.
    """
    if not 0.0 <= effect < 1.0 or outside_width < 1:
        raise ValueError("effect must be in [0,1), with at least one outside column")
    rng = np.random.default_rng(seed)
    own = rng.normal(size=(rows, 1))
    outside = rng.normal(size=(rows, outside_width))
    target = own**2 + np.sqrt(effect) * outside[:, :1] + np.sqrt(1-effect) * rng.normal(size=(rows, 1))
    return own, outside, target
