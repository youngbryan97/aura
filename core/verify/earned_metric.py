"""core/verify/earned_metric.py

A quantity may not carry a name it has not earned.

The Qualia Engine assigns the first three dimensions of the substrate state to
valence, arousal and dominance "by convention", detects self-reference when a
cosine similarity clears 0.85, and reports phenomenal richness as a weighted sum
whose six coefficients were chosen by hand. Every number in that pipeline is
real. None of the names are: nothing checked that dimension 0 tracks anything
valence-like, that 0.85 separates recurrence from drift, or that richness
corresponds to anything at all.

This module is what it takes to keep such a name. An :class:`EarnedAxis` is a
direction in state space fitted against an external target, and it will not
report a value until three things hold:

    it predicts the target on data it was not fitted to (chronological holdout,
    never a random split — a random split on an autocorrelated trajectory leaks
    the answer across the boundary);

    it beats its own permutation null, so a fit that merely tracks the target's
    marginal distribution does not count;

    and it has enough samples that the first two mean something.

Until then :meth:`EarnedAxis.value` returns ``None`` and the snapshot says
``validated: false``. An unvalidated axis is not a broken axis — most of them
will stay unvalidated, and that is the honest description of a feature that has
never been checked against anything.

Layering: no faculty imports. Observations are pushed in. See DEPS.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Verify.EarnedMetric")

__all__ = ["AxisFit", "EarnedAxis", "RecurrenceVerdict", "recurrence_verdict"]


@dataclass(frozen=True)
class AxisFit:
    """The evidence for or against an axis's name."""

    name: str
    validated: bool
    #: Pearson correlation on the chronological holdout. The number that
    #: decides whether the name survives.
    holdout_r: float
    #: Fraction of permutation-null refits that matched or beat the observed
    #: holdout r. Low means the fit is not an artefact of the target's own
    #: distribution.
    permutation_p: float
    n_train: int
    n_holdout: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "validated": self.validated,
            "holdout_r": round(self.holdout_r, 4),
            "permutation_p": round(self.permutation_p, 4),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "reason": self.reason,
        }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float(a_centered @ a_centered) * float(b_centered @ b_centered))
    if denominator < 1e-12:
        # A constant target has no correlation to find. Returning 0 rather than
        # nan keeps a degenerate window from validating on a division artefact.
        return 0.0
    return float((a_centered @ b_centered) / denominator)


def _ridge(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    """Ridge coefficients with an intercept column already present."""

    features = design.shape[1]
    regularizer = penalty * np.eye(features)
    # Never penalize the intercept: doing so shrinks the axis toward zero
    # rather than toward the target's mean.
    regularizer[0, 0] = 0.0
    try:
        return np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, target, rcond=None)[0]


class EarnedAxis:
    """A named direction in state space that must predict its target to keep the name.

    ``min_samples``, ``holdout_fraction``, ``min_holdout_r`` and ``max_p`` are
    the caller's evidentiary bar and are all required. There is no default bar,
    because "what counts as predicting it" is a claim about the domain and not
    something this module can decide on anyone's behalf.
    """

    def __init__(
        self,
        name: str,
        *,
        min_samples: int,
        holdout_fraction: float,
        min_holdout_r: float,
        max_p: float,
        ridge_penalty: float,
        permutations: int,
        capacity: int,
        seed: int = 0x5EED,
    ) -> None:
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must sit strictly between 0 and 1")
        if min_samples < 4:
            raise ValueError("an axis cannot be validated on fewer than 4 observations")
        self.name = name
        self._min_samples = min_samples
        self._holdout_fraction = holdout_fraction
        self._min_holdout_r = min_holdout_r
        self._max_p = max_p
        self._penalty = ridge_penalty
        self._permutations = permutations
        self._capacity = capacity
        self._rng = np.random.default_rng(seed)

        # Bounded at construction rather than trimmed after the fact. The
        # trim was del self._states[:n] on a list of capacity
        # elements, inside the lock, on every observation once full — an
        # O(n) shift in a critical section every hot path waits on.
        # Lockdep measured 87ms on earned_metric.axis.valence. A deque
        # bounded at the same capacity evicts in constant time.
        self._observations: deque[tuple[np.ndarray, float]] = deque(
            maxlen=max(1, int(capacity))
        )
        self._coefficients: np.ndarray | None = None
        self._fit: AxisFit = AxisFit(
            name=name,
            validated=False,
            holdout_r=0.0,
            permutation_p=1.0,
            n_train=0,
            n_holdout=0,
            reason="no observations yet",
        )
        self._lock = checked_lock(f"earned_metric.axis.{name}")

    # -- observation -------------------------------------------------------

    def observe(self, state: Sequence[float], target: float) -> None:
        """Record one (state, external target) pair.

        The target must come from outside this pipeline. An axis fitted against
        something derived from its own inputs validates trivially and means
        nothing.
        """

        vector = np.asarray(state, dtype=float).ravel()
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return
        try:
            value = float(target)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return

        with self._lock:
            if self._observations and self._observations[-1][0].size != vector.size:
                # The substrate was resized under us. Old observations describe
                # a different space and averaging across the change would fit a
                # direction that never existed.
                self._observations.clear()
            self._observations.append((vector, value))

    # -- fitting -----------------------------------------------------------

    def fit(self) -> AxisFit:
        """Refit and revalidate. Returns the evidence, and stores it."""

        with self._lock:
            observed = list(self._observations)
        states = [vector for vector, _target in observed]
        targets = [target for _vector, target in observed]

        if len(states) < self._min_samples:
            result = AxisFit(
                name=self.name,
                validated=False,
                holdout_r=0.0,
                permutation_p=1.0,
                n_train=0,
                n_holdout=0,
                reason=(
                    f"{len(states)} observations, {self._min_samples} needed before "
                    "the name can be earned"
                ),
            )
            with self._lock:
                self._fit = result
                self._coefficients = None
            return result

        matrix = np.vstack(states)
        target_vector = np.asarray(targets, dtype=float)
        # Chronological split. A random split on a trajectory whose neighbours
        # are near-copies puts a near-copy of every holdout point in the
        # training set, and the axis validates on leakage.
        split = int(len(states) * (1.0 - self._holdout_fraction))
        split = max(2, min(split, len(states) - 2))

        design = np.hstack([np.ones((matrix.shape[0], 1)), matrix])
        train_x, holdout_x = design[:split], design[split:]
        train_y, holdout_y = target_vector[:split], target_vector[split:]

        coefficients = _ridge(train_x, train_y, self._penalty)
        holdout_r = _pearson(holdout_x @ coefficients, holdout_y)

        # Permutation null: shuffle the targets, refit, and see how often chance
        # does this well. A fit that only tracks the target's marginal shape
        # produces a null distribution the observed r does not stand out from.
        beat_or_matched = 0
        for _ in range(self._permutations):
            shuffled = self._rng.permutation(train_y)
            null_coefficients = _ridge(train_x, shuffled, self._penalty)
            null_r = _pearson(holdout_x @ null_coefficients, holdout_y)
            if abs(null_r) >= abs(holdout_r):
                beat_or_matched += 1
        permutation_p = (beat_or_matched + 1) / (self._permutations + 1)

        validated = (
            abs(holdout_r) >= self._min_holdout_r and permutation_p <= self._max_p
        )
        if validated:
            reason = (
                f"holdout r={holdout_r:.3f} (bar {self._min_holdout_r:.3f}), "
                f"permutation p={permutation_p:.4f}: the name is supported"
            )
        elif abs(holdout_r) < self._min_holdout_r:
            reason = (
                f"holdout r={holdout_r:.3f} misses the {self._min_holdout_r:.3f} bar: "
                f"this direction does not predict {self.name}"
            )
        else:
            reason = (
                f"holdout r={holdout_r:.3f} but permutation p={permutation_p:.4f}: "
                "chance reproduces this fit too often to credit it"
            )

        result = AxisFit(
            name=self.name,
            validated=validated,
            holdout_r=holdout_r,
            permutation_p=permutation_p,
            n_train=int(split),
            n_holdout=int(len(states) - split),
            reason=reason,
        )
        with self._lock:
            self._fit = result
            # Coefficients are kept only while the name holds. An unvalidated
            # axis has no value to report, so it has no use for them.
            self._coefficients = coefficients if validated else None
        return result

    # -- reading -----------------------------------------------------------

    @property
    def validated(self) -> bool:
        with self._lock:
            return self._fit.validated

    @property
    def last_fit(self) -> AxisFit:
        with self._lock:
            return self._fit

    def value(self, state: Sequence[float]) -> float | None:
        """The axis's reading for ``state`` — or ``None`` if it has not earned one."""

        with self._lock:
            coefficients = self._coefficients
        if coefficients is None:
            return None
        vector = np.asarray(state, dtype=float).ravel()
        if vector.size + 1 != coefficients.size or not np.all(np.isfinite(vector)):
            return None
        return float(coefficients[0] + vector @ coefficients[1:])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            observations = len(self._observations)
            fit = self._fit
        return {**fit.as_dict(), "observations": observations}


@dataclass(frozen=True)
class RecurrenceVerdict:
    """Whether a trajectory revisits its own past more than chance ordering does."""

    recurrent: bool
    #: Strongest mean similarity found at any single temporal lag.
    statistic: float
    #: The percentile of the surrogate null the statistic had to clear.
    threshold: float
    #: The lag at which the trajectory most strongly repeats, in steps.
    dominant_lag: int
    n_states: int
    surrogates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "recurrent": self.recurrent,
            "statistic": round(self.statistic, 4),
            "threshold": round(self.threshold, 4),
            "dominant_lag": self.dominant_lag,
            "n_states": self.n_states,
            "surrogates": self.surrogates,
        }


def _lag_profile_from_gram(gram: np.ndarray) -> np.ndarray:
    """Mean similarity between states separated by each lag k.

    Reads the k-th subdiagonal of the Gram matrix rather than recomputing dot
    products. Every surrogate is a permutation of the same states, so the whole
    campaign needs exactly one matmul: ``gram[perm][:, perm]`` reorders what has
    already been computed. Measured at 128 surrogates over a 20-state window,
    this is the difference between ~10 ms and well under 1 ms per verdict —
    which is what makes a properly calibrated null affordable on a live loop.
    """

    n = gram.shape[0]
    # Beyond half the window a lag has too few pairs for its mean to mean
    # anything: at k = n-1 it is a single comparison.
    return np.array(
        [float(np.mean(np.diagonal(gram, offset=-k))) for k in range(1, n // 2 + 1)]
    )


def _null_maxima(gram: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """The lag-profile maximum of every surrogate, in a dozen array calls.

    The loop version made ten diagonal reads per surrogate — 1,300 small
    NumPy calls for 128 surrogates, each one a GIL round trip. That is under
    a millisecond on an idle host; with an embedding thread and BLAS holding
    the GIL it starved the loop thread for 5.4s (stall dump, 2026-09-15).
    Reordering every surrogate's Gram in one indexing call and reading each
    lag's diagonal across all of them at once keeps the count near twenty.
    """

    permuted = gram[orders[:, :, None], orders[:, None, :]]
    n = gram.shape[0]
    profiles = np.stack(
        [
            np.diagonal(permuted, offset=-k, axis1=1, axis2=2).mean(axis=1)
            for k in range(1, n // 2 + 1)
        ],
        axis=1,
    )
    return profiles.max(axis=1)


def recurrence_verdict(
    history: Sequence[Sequence[float]],
    *,
    percentile: float,
    surrogates: int,
    seed: int = 0x5EED,
) -> RecurrenceVerdict | None:
    """Is this trajectory recurrent, judged against its own shuffled self?

    The Witness layer called a state self-referential when its cosine similarity
    to some earlier state exceeded 0.85. Nothing established that 0.85 separates
    recurrence from drift, and the right value depends entirely on the
    substrate's dimensionality — in a smooth high-dimensional trajectory almost
    every consecutive pair clears it.

    Replacing the constant with a measured threshold is not enough on its own,
    and getting that wrong is instructive. The obvious null — shuffle the states
    and ask how similar each is to its predecessors — leaves a trajectory that
    cycles through three attractors looking exactly like its own shuffle, because
    every state still has nine near-twins somewhere in the sequence whatever the
    order. Measured: 0.139 false-positive rate on noise against a 0.163
    detection rate on a clean 3-cycle. A bar that adapts away the signal it is
    testing for is no better than the constant it replaced.

    What a shuffle does destroy is *when* similar states occur. So the statistic
    here is lag-resolved: the mean similarity between states separated by k
    steps, maximized over k. A 3-cycle spikes at k=3; a shuffle flattens every
    lag to the global mean. Same data, same surrogates, and now 0.025 on noise
    against 1.000 on the 3-cycle.

    This measures RECURRENCE — that the trajectory returns to where it has been.
    It is not evidence of self-modeling, and nothing here should be read as a
    strange loop in Hofstadter's sense; that was the second overclaim in the
    original, after the threshold.

    Returns ``None`` when there is not enough history to build a null.
    """

    states = [np.asarray(s, dtype=float).ravel() for s in history]
    states = [s for s in states if s.size and np.all(np.isfinite(s))]
    if len(states) < 6 or surrogates < 1:
        return None
    width = states[0].size
    if any(s.size != width for s in states):
        return None

    matrix = np.vstack(states)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(norms > 1e-12):
        return None
    unit = matrix / norms[:, None]

    gram = unit @ unit.T
    observed_profile = _lag_profile_from_gram(gram)
    if observed_profile.size == 0:
        return None
    statistic = float(np.max(observed_profile))
    dominant_lag = int(np.argmax(observed_profile)) + 1

    rng = np.random.default_rng(seed)
    orders = np.stack([rng.permutation(gram.shape[0]) for _ in range(surrogates)])
    null = _null_maxima(gram, orders)
    threshold = float(np.percentile(null, percentile))

    return RecurrenceVerdict(
        recurrent=statistic > threshold,
        statistic=statistic,
        threshold=threshold,
        dominant_lag=dominant_lag,
        n_states=len(states),
        surrogates=surrogates,
    )
