"""Reference mathematics for Subject Core v25.

This module is intentionally independent of Aura's runtime.  The runtime-facing
collector should produce:

1. interventional future samples used to learn the predictive/causal grain;
2. paired intact / cut / sham future samples at a common set of anchor states;
3. closure and causal-graph admissibility results.

The functions here then compute the representation and Fisher--Rao quantities
without knowing anything about prompts, model names, or subsystem labels.

Nothing in this module equates the resulting quantity with phenomenology.
It estimates the physical-side quantity F_intrinsic from the v25 specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt
from typing import Mapping, Sequence

import numpy as np

_EPS = 1e-12


def fisher_rao_categorical(p: Sequence[float], q: Sequence[float]) -> float:
    """Fisher--Rao geodesic distance on a categorical probability simplex.

    d_FR(p,q) = 2 arccos sum_i sqrt(p_i q_i)

    The inputs are normalized defensively.  This metric is invariant under a
    common permutation of outcome labels.
    """
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("p and q must be one-dimensional arrays of equal shape")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("probabilities cannot be negative")
    ls = float(left.sum())
    rs = float(right.sum())
    if ls <= 0.0 or rs <= 0.0:
        raise ValueError("probability vectors must have positive mass")
    left = left / ls
    right = right / rs
    bc = float(np.sqrt(left * right).sum())
    bc = float(np.clip(bc, 0.0, 1.0))
    return 2.0 * acos(bc)


def bhattacharyya_from_equal_prior_posterior(eta: Sequence[float]) -> float:
    """Estimate BC(P,Q) from P-vs-Q posterior probabilities.

    With equal priors and M=(P+Q)/2,

        eta(x) = p(x)/(p(x)+q(x))

    implies

        BC(P,Q) = E_M[2 sqrt(eta(1-eta))].

    Cross-fitted probabilities should be supplied so the classifier cannot
    manufacture separation by memorizing its training samples.
    """
    posterior = np.asarray(eta, dtype=np.float64)
    if posterior.ndim != 1:
        raise ValueError("eta must be one-dimensional")
    posterior = np.clip(posterior, _EPS, 1.0 - _EPS)
    return float(np.mean(2.0 * np.sqrt(posterior * (1.0 - posterior))))


@dataclass(frozen=True)
class FisherRaoEstimate:
    bhattacharyya: float
    distance: float
    distance_sq: float


def crossfit_fisher_rao(
    intact: np.ndarray,
    cut: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 0,
) -> FisherRaoEstimate:
    """Estimate Fisher--Rao distance between two continuous sample laws.

    A universally consistent k-NN posterior estimator is used as a practical,
    model-light estimator.  The exact theory is estimator-independent; this is
    a finite-data instrument and must always be calibrated against sham-vs-sham
    arms and alternative estimators in confirmatory work.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = np.asarray(intact, dtype=np.float64)
    q = np.asarray(cut, dtype=np.float64)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1]:
        raise ValueError("intact and cut must be 2D with the same feature width")
    if len(p) < folds or len(q) < folds:
        raise ValueError("not enough samples for requested cross-fitting")
    # Equal priors are required by bhattacharyya_from_equal_prior_posterior.
    n = min(len(p), len(q))
    p = p[:n]
    q = q[:n]
    x = np.vstack([p, q])
    y = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])

    posterior = np.zeros(2 * n, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y):
        # k -> infinity and k/n -> 0 is the standard consistency regime.
        k = max(3, int(sqrt(len(train))))
        if k % 2 == 0:
            k += 1
        k = min(k, max(1, len(train) - 1))
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k, weights="uniform"),
        )
        model.fit(x[train], y[train])
        posterior[test] = model.predict_proba(x[test])[:, 1]

    bc = float(np.clip(bhattacharyya_from_equal_prior_posterior(posterior), 0.0, 1.0))
    distance = 2.0 * acos(bc)
    return FisherRaoEstimate(bc, distance, distance * distance)


@dataclass(frozen=True)
class IntrinsicRateEstimate:
    tau_seconds: float
    raw: FisherRaoEstimate
    sham: FisherRaoEstimate
    raw_rate: float
    sham_rate: float
    excess_rate: float


def intrinsic_rate_from_samples(
    intact_future: np.ndarray,
    cut_future: np.ndarray,
    sham_a_future: np.ndarray,
    sham_b_future: np.ndarray,
    *,
    tau_seconds: float,
    context: np.ndarray | None = None,
    folds: int = 5,
    seed: int = 0,
) -> IntrinsicRateEstimate:
    """Finite-data estimate of F_intrinsic for one partition and one lag.

    Context is the shared pre-transition causal state and environment.  Appending
    it to both arms converts the joint-distribution comparison into a comparison
    of conditional futures while preserving identical context marginals.
    """
    if tau_seconds <= 0:
        raise ValueError("tau_seconds must be positive")

    arrays = [
        np.asarray(intact_future, dtype=np.float64),
        np.asarray(cut_future, dtype=np.float64),
        np.asarray(sham_a_future, dtype=np.float64),
        np.asarray(sham_b_future, dtype=np.float64),
    ]
    n = min(len(a) for a in arrays)
    arrays = [a[:n] for a in arrays]
    if context is not None:
        ctx = np.asarray(context, dtype=np.float64)[:n]
        arrays = [np.hstack([ctx, a]) for a in arrays]

    raw = crossfit_fisher_rao(arrays[0], arrays[1], folds=folds, seed=seed)
    sham = crossfit_fisher_rao(arrays[2], arrays[3], folds=folds, seed=seed + 1)
    raw_rate = raw.distance_sq / tau_seconds
    sham_rate = sham.distance_sq / tau_seconds
    return IntrinsicRateEstimate(
        tau_seconds=tau_seconds,
        raw=raw,
        sham=sham,
        raw_rate=raw_rate,
        sham_rate=sham_rate,
        excess_rate=max(0.0, raw_rate - sham_rate),
    )


def characteristic_signature(
    sample_sets: Sequence[np.ndarray],
    frequencies: Sequence[np.ndarray],
) -> np.ndarray:
    """Finite approximation to an interventional future-distribution signature.

    For each test distribution P and each pre-registered frequency omega this
    records Re E exp(i omega.X) and Im E exp(i omega.X).  The complete
    characteristic function uniquely determines a probability distribution;
    finitely many frequencies are an experimental approximation whose adequacy
    must be attacked with held-out frequencies/interventions.
    """
    if len(sample_sets) != len(frequencies):
        raise ValueError("one frequency bank is required per sample set")
    out: list[float] = []
    for samples, omega in zip(sample_sets, frequencies, strict=True):
        x = np.asarray(samples, dtype=np.float64)
        w = np.asarray(omega, dtype=np.float64)
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("sample and frequency dimensions do not agree")
        phases = x @ w.T
        values = np.exp(1j * phases).mean(axis=0)
        out.extend(values.real.tolist())
        out.extend(values.imag.tolist())
    return np.asarray(out, dtype=np.float64)


def deterministic_frequency_bank(
    widths: Sequence[int],
    *,
    frequencies_per_test: int = 16,
    seed: int = 2501,
) -> tuple[np.ndarray, ...]:
    """Pre-register a reproducible characteristic-function probe bank."""
    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal(scale=1.0 / max(1.0, sqrt(int(width))), size=(frequencies_per_test, int(width))) for width in widths
    )


@dataclass(frozen=True)
class PredictiveGrain:
    """Finite-data approximation to the minimal interventionally sufficient grain."""

    centre: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    singular_values: np.ndarray
    null_q: np.ndarray
    rank: int

    def transform(self, signatures: np.ndarray) -> np.ndarray:
        x = np.asarray(signatures, dtype=np.float64)
        z = (x - self.centre) / self.scale
        if self.rank == 0:
            return np.zeros((len(z), 0), dtype=np.float64)
        return z @ self.projection[:, : self.rank]


def fit_predictive_grain(
    signatures: np.ndarray,
    *,
    null_draws: int = 256,
    quantile: float = 0.99,
    seed: int = 2502,
) -> PredictiveGrain:
    """Parallel-analysis estimate of predictive-state rank.

    Rows are history/anchor signatures.  Each column is a fixed interventional
    future statistic.  The null independently permutes each column over
    histories, preserving every marginal while deleting coherent row structure.

    The retained rank is the contiguous leading set of singular values above
    the pre-registered null quantile.  This is a *linear finite-data estimator*
    of the causal-state dimension, not a theorem that the real system is linear.
    Held-out intervention sufficiency is required before the grain is accepted.
    """
    x = np.asarray(signatures, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("signatures must be a nontrivial 2D matrix")
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - centre) / scale

    _, s, vt = np.linalg.svd(z, full_matrices=False)
    rng = np.random.default_rng(seed)
    null = np.zeros((null_draws, len(s)), dtype=np.float64)
    for draw in range(null_draws):
        shuffled = np.empty_like(z)
        for col in range(z.shape[1]):
            shuffled[:, col] = z[rng.permutation(z.shape[0]), col]
        null[draw] = np.linalg.svd(shuffled, compute_uv=False)[: len(s)]
    q = np.quantile(null, quantile, axis=0)

    rank = 0
    for observed, floor in zip(s, q, strict=True):
        if observed > floor:
            rank += 1
        else:
            break

    return PredictiveGrain(
        centre=centre,
        scale=scale,
        projection=vt.T,
        singular_values=s,
        null_q=q,
        rank=rank,
    )


def heldout_sufficiency_gain(
    raw_history: np.ndarray,
    grain_state: np.ndarray,
    heldout_future: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 2503,
) -> float:
    """Does raw history predict held-out futures beyond the proposed grain?

    Returns the fractional held-out MSE reduction from adding raw history to
    the grain.  The experiment compares this against a shuffled-history floor.
    A true sufficient grain should leave no reproducible gain.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    h = np.asarray(raw_history, dtype=np.float64)
    z = np.asarray(grain_state, dtype=np.float64)
    y = np.asarray(heldout_future, dtype=np.float64)
    if not (len(h) == len(z) == len(y)):
        raise ValueError("history, grain and future must have the same rows")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    base_sse = 0.0
    wide_sse = 0.0
    for train, test in splitter.split(z):
        base = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        wide = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        base.fit(z[train], y[train])
        wide.fit(np.hstack([z[train], h[train]]), y[train])
        base_sse += float(np.square(y[test] - base.predict(z[test])).sum())
        wide_sse += float(
            np.square(y[test] - wide.predict(np.hstack([z[test], h[test]]))).sum()
        )
    if base_sse <= _EPS:
        return 0.0
    return (base_sse - wide_sse) / base_sse


def choose_temporal_scale(
    estimates: Mapping[float, IntrinsicRateEstimate],
) -> float:
    """Choose tau maximizing measured excess intrinsic causal rate."""
    if not estimates:
        raise ValueError("at least one temporal scale is required")
    return max(estimates, key=lambda tau: estimates[tau].excess_rate)


def search_horizon_is_binding(
    estimates: Mapping[float, IntrinsicRateEstimate],
    *,
    tail: int = 2,
) -> bool:
    """True when the maximum sits in the unresolved tail of the lag search."""
    ordered = sorted(estimates)
    winner = choose_temporal_scale(estimates)
    return winner in set(ordered[-max(1, tail) :])


def overlap_local_maxima(
    scores: Mapping[frozenset[str], float],
    *,
    tolerance: float = 1e-12,
) -> tuple[frozenset[str], ...]:
    """Select exclusion maxima among overlapping candidate supports.

    Exact ties are retained as a symmetry/equivalence class rather than broken
    by lexicographic naming.
    """
    winners: list[frozenset[str]] = []
    for support, score in scores.items():
        beaten = False
        for other, other_score in scores.items():
            if other == support or not (support & other):
                continue
            if other_score > score + tolerance:
                beaten = True
                break
        if not beaten and score > 0.0:
            winners.append(support)
    return tuple(sorted(winners, key=lambda s: (len(s), sorted(s))))


def bootstrap_rate_difference(
    intact_future: np.ndarray,
    cut_future: np.ndarray,
    sham_a_future: np.ndarray,
    sham_b_future: np.ndarray,
    *,
    tau_seconds: float,
    context: np.ndarray | None = None,
    draws: int = 200,
    seed: int = 2504,
) -> np.ndarray:
    """Paired bootstrap distribution of cut-minus-sham intrinsic rate."""
    p = np.asarray(intact_future)
    q = np.asarray(cut_future)
    a = np.asarray(sham_a_future)
    b = np.asarray(sham_b_future)
    n = min(len(p), len(q), len(a), len(b))
    ctx = None if context is None else np.asarray(context)[:n]
    rng = np.random.default_rng(seed)
    values = np.zeros(draws, dtype=np.float64)
    for draw in range(draws):
        idx = rng.integers(0, n, size=n)
        estimate = intrinsic_rate_from_samples(
            p[idx],
            q[idx],
            a[idx],
            b[idx],
            tau_seconds=tau_seconds,
            context=None if ctx is None else ctx[idx],
            folds=min(5, max(2, n // 4)),
            seed=seed + draw + 1,
        )
        values[draw] = estimate.raw_rate - estimate.sham_rate
    return values


def paired_permutation_pvalue(
    intact_future: np.ndarray,
    cut_future: np.ndarray,
    sham_a_future: np.ndarray,
    sham_b_future: np.ndarray,
    *,
    tau_seconds: float,
    context: np.ndarray | None = None,
    draws: int = 199,
    folds: int = 5,
    seed: int = 2505,
) -> tuple[float, float]:
    """One-sided paired randomization test for cut damage above sham.

    H0: for each common fork, the intact and cut future are exchangeable.
    Statistic: raw Fisher--Rao rate minus sham Fisher--Rao rate.

    Returns (observed_excess, p_value).  Used as one component of the
    intersection-union test that requires *every* bipartition to reject H0.
    """
    p = np.asarray(intact_future, dtype=np.float64)
    q = np.asarray(cut_future, dtype=np.float64)
    a = np.asarray(sham_a_future, dtype=np.float64)
    b = np.asarray(sham_b_future, dtype=np.float64)
    n = min(len(p), len(q), len(a), len(b))
    if n < max(8, folds):
        raise ValueError("too few paired contexts for permutation test")
    p, q, a, b = p[:n], q[:n], a[:n], b[:n]
    ctx = None if context is None else np.asarray(context, dtype=np.float64)[:n]

    observed = intrinsic_rate_from_samples(
        p, q, a, b,
        tau_seconds=tau_seconds,
        context=ctx,
        folds=folds,
        seed=seed,
    )
    observed_stat = observed.raw_rate - observed.sham_rate

    rng = np.random.default_rng(seed + 1)
    exceed = 0
    for draw in range(draws):
        swap = rng.random(n) < 0.5
        pp = p.copy()
        qq = q.copy()
        pp[swap], qq[swap] = q[swap], p[swap]
        perm = intrinsic_rate_from_samples(
            pp, qq, a, b,
            tau_seconds=tau_seconds,
            context=ctx,
            folds=folds,
            seed=seed + 10 + draw,
        )
        stat = perm.raw_rate - perm.sham_rate
        if stat >= observed_stat - 1e-15:
            exceed += 1
    p_value = (exceed + 1.0) / (draws + 1.0)
    return float(observed_stat), float(p_value)


def save_predictive_grain(path: str, grain: PredictiveGrain, **extra: np.ndarray) -> None:
    """Serialize a frozen grain and experiment arrays for confirmation."""
    payload = {
        "centre": grain.centre,
        "scale": grain.scale,
        "projection": grain.projection,
        "singular_values": grain.singular_values,
        "null_q": grain.null_q,
        "rank": np.asarray([grain.rank], dtype=np.int64),
        **{key: np.asarray(value) for key, value in extra.items()},
    }
    np.savez_compressed(path, **payload)


def load_predictive_grain(path: str) -> tuple[PredictiveGrain, dict[str, np.ndarray]]:
    """Load a grain written by save_predictive_grain."""
    blob = np.load(path)
    grain = PredictiveGrain(
        centre=blob["centre"],
        scale=blob["scale"],
        projection=blob["projection"],
        singular_values=blob["singular_values"],
        null_q=blob["null_q"],
        rank=int(blob["rank"][0]),
    )
    reserved = {"centre", "scale", "projection", "singular_values", "null_q", "rank"}
    extra = {key: blob[key] for key in blob.files if key not in reserved}
    return grain, extra
