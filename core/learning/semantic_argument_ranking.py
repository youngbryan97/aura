"""Fit argument-choice margins from source-labelled mention groups."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pairwise_loss(
    weight: np.ndarray,
    features: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """Weighted logistic ranking loss without materializing a doubled matrix."""
    from scipy.special import expit

    loss = 0.5 * regularization * float(weight @ weight)
    gradient = regularization * weight
    normalization = float(np.sum(sample_weight))
    for start in range(0, negative.size, batch_size):
        stop = start + batch_size
        differences = (
            features[positive[start:stop]].astype(np.float64)
            - features[negative[start:stop]].astype(np.float64)
        )
        weights = sample_weight[start:stop] / normalization
        margins = differences @ weight
        loss += float(weights @ np.logaddexp(0.0, -margins))
        gradient -= (weights * expit(-margins)) @ differences
    return loss, gradient


def fit_pairwise_argument_weight(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    initial_weight: np.ndarray,
    max_iter: int = 400,
    inverse_regularization: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank each group's positive mention above that group's alternatives.

    Each group starts with one positive row followed by its negative rows.
    The input comes from the existing source-supervised proposal builder.
    A per-slot bias cancels in pairwise margins and is therefore not fitted.
    """
    from scipy.optimize import minimize

    features = np.asarray(features)
    labels = np.asarray(labels)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    initial = np.asarray(initial_weight, dtype=np.float64)
    if (
        features.ndim != 2 or features.shape[0] < 2
        or labels.shape != (features.shape[0],)
        or sample_weight.shape != labels.shape
        or initial.shape != (features.shape[1],)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(sample_weight))
        or np.any(sample_weight <= 0)
        or set(np.unique(labels).tolist()) != {0, 1}
        or labels[0] != 1
        or type(max_iter) is not int or max_iter < 1
        or not np.isfinite(inverse_regularization) or inverse_regularization <= 0
    ):
        raise ValueError("invalid grouped argument ranking supervision")
    indices = np.arange(labels.size)
    group_positive = np.maximum.accumulate(np.where(labels == 1, indices, -1))
    negative = indices[labels == 0]
    positive = group_positive[negative]
    if set(positive.tolist()) != set(indices[labels == 1].tolist()):
        raise ValueError("argument ranking group has no alternative")
    weights = sample_weight[negative]
    regularization = 1.0 / (inverse_regularization * float(np.sum(weights)))

    def objective(weight):
        return _pairwise_loss(
            weight, features, positive, negative, weights,
            regularization=regularization,
        )

    initial_loss, _ = objective(initial)
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"argument ranking fit incomplete: {result.status}: {result.message}")
    fitted = np.asarray(result.x, dtype=np.float32)
    exported_loss, _ = objective(fitted.astype(np.float64))
    return fitted, {
        "objective": "source_grouped_pairwise_logistic_v1",
        "groups": int(np.count_nonzero(labels == 1)),
        "pairs": int(negative.size),
        "initial_loss": initial_loss,
        "exported_loss": exported_loss,
        "regularization": regularization,
        "iterations": int(result.nit),
        "converged": True,
        "bias_fitted": False,
        "test_examples_used": 0,
    }
