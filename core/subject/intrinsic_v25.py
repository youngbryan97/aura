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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, sqrt
from typing import Any

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
    whiten: bool = False,
    groups: np.ndarray | None = None,
    one_signal_once: bool | None = None,
) -> FisherRaoEstimate:
    """Estimate Fisher--Rao distance between two continuous sample laws.

    A universally consistent k-NN posterior estimator is used as a practical,
    model-light estimator.  The exact theory is estimator-independent; this is
    a finite-data instrument and must always be calibrated against sham-vs-sham
    arms and alternative estimators in confirmatory work.

    `whiten` decides the metric the neighbours are counted in. Standardising
    each column and taking Euclidean distance weights the coordinates the state
    happens to be written in, so the estimate moves when the same information is
    written in a different basis: on the v25 run of 26 September the rate read
    0.019956 raw and 0.010103 under an invertible rotation, a drift of 0.49, and
    the carrier was reported unresolved for it. Whitening makes Euclidean
    distance equal to Mahalanobis distance, which is the same number under every
    invertible linear recoding, so the estimate is a property of the samples
    rather than of the coordinates.

    It is not the default: changing an estimator changes what every earlier run
    measured, so a campaign asks for it and the preregistration says which.

    `groups` names the anchor each row came from, when row i of `intact` and
    row i of `cut` are two arms forked from one anchor. Folds then keep an
    anchor's rows together. Without it, a test row's twin from the other arm
    sat in the training fold as its nearest neighbour with the other label, so
    the classifier leaned away from each row's own class and read separation
    where there was none: on two identical samples of 128 anchors the distance
    squared came out 0.056, and 0.103 at 64, which was the whole sham floor of
    the v25 looks (0.062 to 0.107). Grouped cross-fitting is the standard remedy
    for paired rows, and it leaves the quantity estimated as it was.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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
    if _one_signal_once(one_signal_once):
        x = x[:, one_column_per_signal(x)]

    posterior = np.zeros(2 * n, dtype=np.float64)
    if groups is not None:
        anchor = np.asarray(groups)[:n]
        if len(np.unique(anchor)) < folds:
            raise ValueError("fewer distinct anchors than folds for grouped cross-fitting")
        grouped = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = grouped.split(x, y, groups=np.concatenate([anchor, anchor]))
    else:
        splits = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y)
    for train, test in splits:
        # k -> infinity and k/n -> 0 is the standard consistency regime.
        k = max(3, int(sqrt(len(train))))
        if groups is None and k % 2 == 0:
            k += 1
        elif groups is not None and k % 2 == 1:
            # An odd k breaks a vote. A posterior takes no vote, and paired rows
            # come as twins: an odd k split one twin pair at every point and left
            # each posterior one neighbour off a half, 4/k^2 of distance squared on
            # two identical samples (0.0178 at 128 anchors, k = 15).
            k += 1
        k = min(k, max(1, len(train) - 1))
        if whiten:
            # Fitted on the training fold only, like every other step here.
            # PCA whitening rather than an inverse covariance: with more
            # columns than rows the covariance is singular, and the components
            # a fold can support are exactly the ones it can whiten.
            keep = max(1, min(len(train) - 1, x.shape[1]))
            metric: Any = PCA(n_components=keep, whiten=True, random_state=seed)
        else:
            metric = StandardScaler()
        if groups is not None:
            fitted = metric.fit(x[train])
            posterior[test] = _tie_inclusive_posterior(
                fitted.transform(x[train]), y[train], fitted.transform(x[test]), k
            )
            continue
        model = make_pipeline(
            metric,
            KNeighborsClassifier(n_neighbors=k, weights="uniform"),
        )
        model.fit(x[train], y[train])
        posterior[test] = model.predict_proba(x[test])[:, 1]

    bc = float(np.clip(bhattacharyya_from_equal_prior_posterior(posterior), 0.0, 1.0))
    distance = 2.0 * acos(bc)
    return FisherRaoEstimate(bc, distance, distance * distance)


def _one_signal_once(asked: bool | None) -> bool:
    """Whether to count each signal once: as asked, or as the run's design set it."""
    if asked is not None:
        return bool(asked)
    from core.runtime.flags import env_str

    design = env_str(
        "AURA_ESTIMATOR_ONE_SIGNAL",
        description="1 keeps one column per signal in the estimator's neighbour distance.",
        owner="core.subject.intrinsic_v25",
    )
    return design.strip() == "1"


def one_column_per_signal(x: np.ndarray) -> np.ndarray:
    """The columns of `x` to keep so that no signal is carried by two of them.

    Two columns carry one signal when one is an affine copy of the other, which
    is how a schema that reads one field twice, or a test that duplicates the
    state, writes them. Counted twice, a signal has twice the say in the
    neighbour distance, and on 28 September duplicating the future block took
    the weakest cut's rate from 0.0032 to 0.0083: the carrier's duplication
    control failed on the metric, not on her. Columns are compared standardised
    over every row of both arms, so nothing about which arm a row is in is used,
    and a constant column (which moves no distance) collapses to one.
    """
    x = np.asarray(x, dtype=np.float64)
    spread = x.std(axis=0)
    unit = np.where(spread > 0.0, spread, 1.0)
    standard = (x - x.mean(axis=0)) / unit
    # Sign as well as scale: a column and its negation are one signal.
    lead = np.argmax(np.abs(standard) > 1e-12, axis=0)
    sign = np.sign(standard[lead, np.arange(standard.shape[1])])
    sign[sign == 0] = 1.0
    _, first = np.unique(np.round(standard * sign, 10), axis=1, return_index=True)
    return np.sort(first)


def _tie_inclusive_posterior(train: np.ndarray, labels: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """The share of the first class among each test row's k nearest, counting every tie at the k-th.

    Paired rows come in blocks at one distance: an anchor's two arms, and every
    copy of an anchor a bootstrap drew twice. A neighbour count that cuts
    through such a block takes whichever rows come first, and the first class
    is stacked first, so the posterior leaned one way whenever a bootstrap
    drew an anchor twice. Counting the whole block keeps it balanced. With no
    ties this is the ordinary k-nearest-neighbour posterior.
    """
    squared = (
        np.einsum("ij,ij->i", test, test)[:, None]
        - 2.0 * test @ train.T
        + np.einsum("ij,ij->i", train, train)[None, :]
    )
    np.maximum(squared, 0.0, out=squared)
    kth = np.partition(squared, k - 1, axis=1)[:, k - 1][:, None]
    # A relative tolerance, because two copies of one row reach a test row by
    # arithmetic that need not agree in the last bits.
    near = squared <= kth + 1e-9 * (1.0 + kth)
    counts = near.sum(axis=1)
    return (near * (labels[None, :] == 1)).sum(axis=1) / np.maximum(counts, 1)


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
    whiten: bool = False,
    groups: np.ndarray | None = None,
    one_signal_once: bool | None = None,
) -> IntrinsicRateEstimate:
    """Finite-data estimate of F_intrinsic for one partition and one lag.

    Context is the shared pre-transition causal state and environment.  Appending
    it to both arms converts the joint-distribution comparison into a comparison
    of conditional futures while preserving identical context marginals.

    `whiten` is passed through to `crossfit_fisher_rao` and decides whether the
    rate is a property of the samples or of the coordinates they are written in.
    `groups` names the anchor of each row, when row i of every arm was forked
    from the same anchor; see `crossfit_fisher_rao`.
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

    if _one_signal_once(one_signal_once):
        # Chosen once over all four arms, so the cut and its sham are read in
        # one metric.
        keep = one_column_per_signal(np.vstack(arrays))
        arrays = [a[:, keep] for a in arrays]

    anchor = None if groups is None else np.asarray(groups)[:n]
    raw = crossfit_fisher_rao(
        arrays[0], arrays[1], folds=folds, seed=seed, whiten=whiten, groups=anchor, one_signal_once=False
    )
    sham = crossfit_fisher_rao(
        arrays[2], arrays[3], folds=folds, seed=seed + 1, whiten=whiten, groups=anchor, one_signal_once=False
    )
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

    The folds are contiguous blocks of rows, in the order given. A carrier's
    anchors are consecutive turns, so with shuffled folds every held-out anchor
    had its neighbours in training, and an anchor's older turns are exactly its
    neighbour's recent ones: extra history placed a held-out anchor among
    neighbours with nearly its own future, which shuffled history cannot do.
    On the carrier at f78644bbd that read as history still needed at the
    longest rung (0.139 against a shuffled 0.022); with blocked folds older
    turns add no more than shuffled ones at any rung (22 October). Rows that
    depend on their neighbours are cross-validated in blocks (Roberts et al.,
    Ecography 40, 2017). ``seed`` no longer orders the rows.
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
    splitter = KFold(n_splits=folds)
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
    paired: bool = False,
) -> np.ndarray:
    """Paired bootstrap distribution of cut-minus-sham intrinsic rate.

    `paired` keeps each resampled anchor's rows in one fold, so a row drawn
    twice is not its own neighbour across folds either.
    """
    p = np.asarray(intact_future)
    q = np.asarray(cut_future)
    a = np.asarray(sham_a_future)
    b = np.asarray(sham_b_future)
    n = min(len(p), len(q), len(a), len(b))
    ctx = None if context is None else np.asarray(context)[:n]
    rng = np.random.default_rng(seed)
    # Every index drawn here, in order, before any draw is evaluated, so the
    # draws can be spread over processes and come back as one process has them.
    resampled = [(rng.integers(0, n, size=n), seed + draw + 1) for draw in range(draws)]
    from core.subject.draw_pool import evaluate_in_order

    shared = (p, q, a, b, ctx, float(tau_seconds), min(5, max(2, n // 4)), bool(paired))
    return np.asarray(evaluate_in_order(_bootstrap_chunk, shared, resampled), dtype=np.float64)


def _bootstrap_chunk(shared: tuple, draws: Sequence[tuple[np.ndarray, int]]) -> list[float]:
    """Cut-minus-sham rate for each resampled draw, in the order given."""
    p, q, a, b, ctx, tau_seconds, folds, paired = shared
    values: list[float] = []
    for idx, draw_seed in draws:
        estimate = intrinsic_rate_from_samples(
            p[idx],
            q[idx],
            a[idx],
            b[idx],
            tau_seconds=tau_seconds,
            context=None if ctx is None else ctx[idx],
            folds=folds,
            seed=draw_seed,
            groups=idx if paired else None,
        )
        values.append(float(estimate.raw_rate - estimate.sham_rate))
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
    paired: bool = False,
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

    anchor = np.arange(n) if paired else None
    observed = intrinsic_rate_from_samples(
        p, q, a, b,
        tau_seconds=tau_seconds,
        context=ctx,
        folds=folds,
        seed=seed,
        groups=anchor,
    )
    observed_stat = observed.raw_rate - observed.sham_rate

    rng = np.random.default_rng(seed + 1)
    swaps = [(rng.random(n) < 0.5, seed + 10 + draw) for draw in range(draws)]
    from core.subject.draw_pool import evaluate_in_order

    shared = (p, q, a, b, ctx, float(tau_seconds), int(folds), anchor)
    stats = evaluate_in_order(_permutation_chunk, shared, swaps)
    exceed = sum(1 for stat in stats if stat >= observed_stat - 1e-15)
    p_value = (exceed + 1.0) / (draws + 1.0)
    return float(observed_stat), float(p_value)


def _permutation_chunk(shared: tuple, draws: Sequence[tuple[np.ndarray, int]]) -> list[float]:
    """Cut-minus-sham rate with each draw's arms swapped where its mask says, in order."""
    p, q, a, b, ctx, tau_seconds, folds, anchor = shared
    stats: list[float] = []
    for swap, draw_seed in draws:
        pp = p.copy()
        qq = q.copy()
        pp[swap], qq[swap] = q[swap], p[swap]
        perm = intrinsic_rate_from_samples(
            pp, qq, a, b,
            tau_seconds=tau_seconds,
            context=ctx,
            folds=folds,
            seed=draw_seed,
            groups=anchor,
        )
        stats.append(float(perm.raw_rate - perm.sham_rate))
    return stats


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


def bicross_validated_rank(
    signatures: np.ndarray,
    *,
    row_folds: int = 2,
    column_folds: int = 2,
    seed: int = 2504,
) -> int:
    """The rank that best predicts held-out blocks of the matrix.

    Bi-cross-validation of the SVD (Owen and Perry, 2009, Annals of Applied
    Statistics 3(2):564-594). Rows and columns are split into folds. For each
    held-out block A, with B its rows over the kept columns, C the kept rows
    over its columns and D the rest, a rank-k fit predicts A as B D_k^+ C, rank
    zero predicting every entry of A as its column mean. The estimate is the
    smallest rank whose total error is within one standard error of the best,
    the standard error read across the blocks. The (2, 2) holdout is the paper's
    default. Columns are standardised as the parallel analysis
    standardises them, so the two estimators read the same matrix.
    """
    x = np.asarray(signatures, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 * max(row_folds, column_folds):
        raise ValueError("signatures must be a 2D matrix with at least two entries per fold")
    scale = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(scale > 1e-12, scale, 1.0)
    rng = np.random.default_rng(seed)
    row_groups = np.array_split(rng.permutation(z.shape[0]), row_folds)
    column_groups = np.array_split(rng.permutation(z.shape[1]), column_folds)
    largest = min(
        z.shape[0] - max(len(group) for group in row_groups),
        z.shape[1] - max(len(group) for group in column_groups),
    )
    blocks: list[np.ndarray] = []
    everything_rows = np.arange(z.shape[0])
    everything_columns = np.arange(z.shape[1])
    for held_rows in row_groups:
        kept_rows = np.setdiff1d(everything_rows, held_rows)
        for held_columns in column_groups:
            kept_columns = np.setdiff1d(everything_columns, held_columns)
            a = z[np.ix_(held_rows, held_columns)]
            b = z[np.ix_(held_rows, kept_columns)]
            c = z[np.ix_(kept_rows, held_columns)]
            d = z[np.ix_(kept_rows, kept_columns)]
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            # numpy's own matrix-rank tolerance, so a direction the SVD cannot
            # tell from rounding is never inverted.
            floor = np.finfo(np.float64).eps * max(d.shape) * (s[0] if s.size else 0.0)
            usable = int(np.sum(s > floor))
            row = np.empty(largest + 1, dtype=np.float64)
            row[0] = float(np.sum(a * a))
            for k in range(1, largest + 1):
                kept = min(k, usable)
                if kept == 0:
                    row[k] = row[0]
                    continue
                pseudo_inverse = (vt[:kept].T / s[:kept]) @ u[:, :kept].T
                residual = a - b @ pseudo_inverse @ c
                row[k] = float(np.sum(residual * residual))
            blocks.append(row)
    per_block = np.vstack(blocks)
    total = per_block.sum(axis=0)
    best = int(np.argmin(total))
    # The one-standard-error rule (Hastie, Tibshirani and Friedman, The Elements
    # of Statistical Learning, 2nd ed., section 7.10). The minimum alone picks up
    # a noise direction in some splits: on a rank-two matrix it found two in
    # twelve seeds of twenty, and this rule found it in all twenty at every noise
    # level tried, with ranks zero, one and three unchanged.
    spread = per_block.std(axis=0, ddof=1) * np.sqrt(len(per_block))
    return int(np.flatnonzero(total <= total[best] + spread[best])[0])


def _out_of_fold(features: np.ndarray, target: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Contiguous blocks, for the reason `heldout_sufficiency_gain` gives.
    prediction = np.zeros_like(target)
    for train, test in KFold(n_splits=folds).split(features):
        model = make_pipeline(StandardScaler(), Ridge()).fit(features[train], target[train])
        prediction[test] = model.predict(features[test])
    return prediction


def heldout_new_direction_gain(
    raw_history: np.ndarray,
    grain_state: np.ndarray,
    heldout_future: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 2505,
) -> float:
    """How much of the held-out future the grain cannot reach, raw history reaches.

    A grain of rank k predicts futures along at most k output directions. The
    question sufficiency asks is whether history predicts the future along any
    other direction, so both are fitted out of fold, the part of the future
    orthogonal to the grain's predicted directions is taken, and the return is
    the share of that part history's extra prediction removes.

    The fractional gain beside this cannot ask that. A grain estimated from
    noisy signatures is a noisy copy of the true state, and history that
    determines the state exactly improves on the copy inside the grain's own
    directions: on a twelve-variable system with a two-dimensional macrostate
    the correct grain and a grain missing a whole dimension both scored a
    margin near one over their shuffled floors. On the same systems this reads
    about -0.2 for the correct grain and above 0.9 for the truncated one.
    """
    h = np.asarray(raw_history, dtype=np.float64)
    z = np.asarray(grain_state, dtype=np.float64)
    y = np.asarray(heldout_future, dtype=np.float64)
    if not (len(h) == len(z) == len(y)):
        raise ValueError("history, grain and future must have the same rows")
    if z.ndim != 2 or z.shape[1] == 0:
        raise ValueError("the grain must have at least one dimension")
    centred = y - y.mean(axis=0)
    from_grain = _out_of_fold(z, centred, folds=folds, seed=seed)
    from_both = _out_of_fold(np.column_stack([z, h]), centred, folds=folds, seed=seed)
    _, _, directions = np.linalg.svd(from_grain - from_grain.mean(axis=0), full_matrices=False)
    reached = directions[: z.shape[1]].T
    elsewhere = np.eye(centred.shape[1]) - reached @ reached.T
    missing = (centred - from_grain) @ elsewhere
    added = (from_both - from_grain) @ elsewhere
    total = float(np.sum(missing * missing))
    if total <= 0.0:
        return 0.0
    return 1.0 - float(np.sum((missing - added) ** 2)) / total

