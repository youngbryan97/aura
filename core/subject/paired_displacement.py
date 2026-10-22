"""How far a cut moved her from where the same moment went untouched, against how far two untouched runs part.

The v5 question, as the preregistration puts it: from one snapshot, hold one
side of a cut and let the other run, then the reverse; compose the two free
halves; and ask whether the composed run differs from the untouched one by more
than two untouched runs differ from each other. The forks are deterministic now
(the sham reads 0.0), so each anchor gives the answer directly: one displacement
under the cut, one between two untouched forks, both from the same state.

The Fisher-Rao estimator it replaces for v5 asked something else. It trained a
neighbour classifier to tell the arms apart across anchors, with an anchor's
rows kept in one fold, so it could only use what the cut does the same way at
every anchor. On seed 7, holding C for two turns moved 199 of her columns beyond
their sham at 127 of 128 anchors, up to 0.97 of a column's spread, and the
classifier read a rate of 0.0067 with a lower bound below zero.

The displacement is measured in her own units, which here means the covariance
of the untouched futures, so a cut is large when it moves her along a direction
she varies little in and small when it moves her along one she always varies
in. Three choices keep it free of the coordinates the state is written in:

- the arms are projected onto the subspace they move in, found by SVD of all
  four arms pooled at numpy's own rank tolerance, which uses no label;
- the covariance there is Ledoit and Wolf's shrinkage estimate ("A
  well-conditioned estimator for large-dimensional covariance matrices",
  Journal of Multivariate Analysis 88, 2004), whose intensity is estimated from
  the data, so nothing is chosen, and which turns with the data;
- the distance is Mahalanobis under that estimate.

An orthogonal recoding of the state then changes nothing, and neither does
writing every column twice. `tests/test_a_cut_is_read_by_where_it_moved_her.py`
holds both, and that identical arms read zero, that a real move is decided, and
that a cut arm that is only another untouched fork is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DisplacementEstimate", "DisplacementMetric", "decide", "displacement_metric", "per_anchor"]


@dataclass(frozen=True)
class DisplacementEstimate:
    """The cut's and the sham's mean displacement per second, as a cut verdict reads rates."""

    raw_rate: float
    sham_rate: float

    @property
    def excess_rate(self) -> float:
        return self.raw_rate - self.sham_rate


def per_anchor(
    intact: np.ndarray, cut: np.ndarray, sham_a: np.ndarray, sham_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each anchor's squared displacement under the cut, and between its two untouched forks."""
    from sklearn.covariance import LedoitWolf

    arms = [np.asarray(a, dtype=np.float64) for a in (intact, cut, sham_a, sham_b)]
    n = min(len(a) for a in arms)
    arms = [a[:n] for a in arms]
    pool = np.vstack(arms)
    centre = pool.mean(axis=0)
    _, spectrum, basis = np.linalg.svd(pool - centre, full_matrices=False)
    if spectrum.size == 0 or spectrum[0] <= 0.0:
        return np.zeros(n), np.zeros(n)
    tolerance = spectrum[0] * max(pool.shape) * np.finfo(np.float64).eps
    moving = basis[: int(np.sum(spectrum > tolerance))].T
    untouched, composed, first, second = ((a - centre) @ moving for a in arms)
    precision = LedoitWolf().fit(untouched).precision_
    moved = composed - untouched
    parted = second - first
    return (
        np.einsum("ij,jk,ik->i", moved, precision, moved),
        np.einsum("ij,jk,ik->i", parted, precision, parted),
    )


@dataclass(frozen=True)
class DisplacementMetric:
    """The distance `per_anchor` reads, fitted on a baseline: moving subspace, shrunk precision.

    Dividing each column by its own spread lets a column that barely moved in
    the baseline turn any later change into thousands of spreads: the hot-stove
    probe's first run read 30,042 against 30,047 that way, every other column
    lost in it.
    """

    moving: np.ndarray
    root: np.ndarray

    def whitened(self, delta: np.ndarray) -> np.ndarray:
        """A difference of states in coordinates where this distance is the plain one."""
        return np.atleast_2d(np.asarray(delta, dtype=np.float64)) @ self.moving @ self.root

    def squared(self, delta: np.ndarray) -> np.ndarray:
        """The squared distance of each row of a difference of states."""
        return np.sum(self.whitened(delta) ** 2, axis=1)


def displacement_metric(baseline: np.ndarray) -> DisplacementMetric:
    """Fit the metric on baseline states, one per row."""
    from sklearn.covariance import LedoitWolf

    pool = np.asarray(baseline, dtype=np.float64)
    centre = pool.mean(axis=0)
    _, spectrum, basis = np.linalg.svd(pool - centre, full_matrices=False)
    if spectrum.size == 0 or spectrum[0] <= 0.0:
        return DisplacementMetric(np.zeros((pool.shape[1], 0)), np.zeros((0, 0)))
    tolerance = spectrum[0] * max(pool.shape) * np.finfo(np.float64).eps
    moving = basis[: int(np.sum(spectrum > tolerance))].T
    precision = LedoitWolf().fit((pool - centre) @ moving).precision_
    return DisplacementMetric(moving, np.linalg.cholesky(precision))


def decide(
    samples: dict[str, np.ndarray],
    *,
    tau_seconds: float,
    seed: int,
    alpha: float,
    draws: int,
    permutation_draws: int = 199,
) -> tuple[DisplacementEstimate, float, float, float]:
    """The excess rate, its lower bound at `alpha`, and a sign-flip p-value, as `decide_cut` returns them.

    The lower bound is the `alpha` quantile of the mean excess over `draws`
    bootstrap resamples of the anchors. Under the null the cut arm is one more
    untouched fork, so each anchor's cut and sham displacements are exchangeable
    and their difference is symmetric about zero; the p-value flips its sign.
    """
    moved, parted = per_anchor(samples["intact"], samples["cut"], samples["sham_a"], samples["sham_b"])
    excess = moved - parted
    tau = float(tau_seconds)
    estimate = DisplacementEstimate(raw_rate=float(moved.mean()) / tau, sham_rate=float(parted.mean()) / tau)
    rng = np.random.default_rng(seed + 1)
    n = len(excess)
    means = excess[rng.integers(0, n, size=(int(draws), n))].mean(axis=1)
    lower = float(np.quantile(means, alpha)) / tau
    flips = np.random.default_rng(seed + 2).choice((-1.0, 1.0), size=(int(permutation_draws), n))
    observed = float(excess.mean())
    exceed = int(np.sum((flips * excess).mean(axis=1) >= observed - 1e-15))
    p_value = (exceed + 1.0) / (int(permutation_draws) + 1.0)
    return estimate, observed / tau, lower, p_value
