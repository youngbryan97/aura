"""core/consciousness/integrated_information.py — whole-system Φ, honestly.
==============================================================================
Integrated-information estimation over the ACTUAL runtime, built to answer
the July-2026 review critique head-on:

    "The IIT calculation is a reductionist hack... φ here is a telemetry
     signal, not a consciousness meter."

Exact micro-level IIT-4.0 Φ is off the table for ANY nontrivial system —
its definition quantifies over every partition of every mechanism over the
full counterfactual state space.  That is intrinsic, not an engineering
gap.  This module implements the strongest honest position available:

  RAIL A — WHOLE-SYSTEM CONTINUOUS Φ.  Gaussian/VAR integrated information
      ("stochastic interaction", Ay 2003; Barrett & Seth 2011; Oizumi &
      Kitazono practical-Φ) over a bounded quotient that covers every retained
      runtime channel — no hand-picked cognitive node list and no toy TPM.
      The minimum-information
      partition is found by exhaustive search at tractable dimensions and by
      a labelled Queyranne candidate plus deterministic local refinement at
      larger dimensions.  Stochastic interaction is symmetric but is not
      generally submodular, so the polynomial search is never misreported as
      an exact MIP.

  RAIL B — GRAIN DISCOVERY (causal emergence; Hoel/Albantakis/Tononi).
      The complex is DERIVED, not assumed: hierarchical coarse-grainings
      of the channel set are searched and each grain is scored against its
      own surrogate null; the emergent grain maximizes held-out integrated
      information per element, with a family-wise surrogate test over the
      complete grain search.  Macro may legitimately beat micro.

  RAIL C — EXACT DISCRETE Φ AT THE DERIVED GRAIN.  For k ≤ 12 macro
      elements, system integrated information is computed by EXHAUSTIVE
      bipartition search over the empirical (and optionally interventional)
      transition distribution — exact at the grain where exactness is
      meaningful, state-averaged over visited states.

  RAIL D — statistical honesty as architecture: circular-shift surrogate
      nulls, block-bootstrap confidence intervals, stationarity and
      Gaussianity diagnostics, and a named estimator on every report.
      (The interventional probe lives in perturbational_probe.py and
      feeds Rail C extra transition rows.)

Every PhiEstimate carries a bounded claim: this is an estimate of the
integrated information of the system's macro-dynamics at an empirically
selected grain — evidence about integration structure, NOT a consciousness
meter.  No number this module emits is presented as one.

Pure NumPy, deterministic under a seeded RNG, no LLM, no I/O.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.IntegratedInformation")

SCHEMA_VERSION = 2
ESTIMATOR_NAME = "gaussian_stochastic_interaction.bounded_mip.v2"
DISCRETE_ESTIMATOR_NAME = "empirical_state_phi.exact_bipartition.v2"
EXACT_GAUSSIAN_MIP_MAX_ELEMENTS = 12

_LOG2PIE = math.log(2.0 * math.pi * math.e)
_EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian machinery
# ─────────────────────────────────────────────────────────────────────────────

def _shrunk_cov(Z: np.ndarray, shrinkage: float) -> np.ndarray:
    """Covariance with diagonal shrinkage: (1-λ)·S + λ·diag(S).

    Keeps every principal submatrix well-conditioned so log-dets and Schur
    complements stay finite on short windows."""
    S = np.cov(Z, rowvar=False, bias=False)
    S = np.atleast_2d(S)
    lam = float(min(max(shrinkage, 0.0), 1.0))
    out = (1.0 - lam) * S + lam * np.diag(np.diag(S))
    out += _EPS * np.eye(out.shape[0])
    return out


def _logdet(M: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        # Numerically defective submatrix — regularize harder and retry once.
        val = np.linalg.slogdet(M + 1e-8 * np.eye(M.shape[0]))[1]
    return float(val)


@dataclass
class LaggedGaussian:
    """Joint Gaussian over (X_{t-1}, X_t) — the single object every
    entropy/oracle query is answered from, via Schur complements."""

    sigma: np.ndarray          # (2N × 2N); indices 0..N-1 = past, N..2N-1 = present
    n_channels: int
    n_samples: int

    @classmethod
    def fit(cls, X: np.ndarray, *, shrinkage: float = 0.05) -> LaggedGaussian:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] < 2:
            raise ValueError("channel matrix must be T×N with T ≥ 8, N ≥ 2")
        # Standardize per channel (estimation only — diagnostics see raw data).
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        Z = np.hstack([Xs[:-1], Xs[1:]])            # T-1 × 2N
        return cls(sigma=_shrunk_cov(Z, shrinkage),
                   n_channels=X.shape[1], n_samples=X.shape[0])

    def _cond_entropy(self, present_idx: np.ndarray, past_idx: np.ndarray) -> float:
        """h(present | past) in nats via the Schur complement of sigma."""
        pp = self.sigma[np.ix_(present_idx, present_idx)]
        if past_idx.size == 0:
            return 0.5 * (_logdet(pp) + pp.shape[0] * _LOG2PIE)
        qq = self.sigma[np.ix_(past_idx, past_idx)]
        pq = self.sigma[np.ix_(present_idx, past_idx)]
        cond = pp - pq @ np.linalg.solve(qq, pq.T)
        return 0.5 * (_logdet(cond) + pp.shape[0] * _LOG2PIE)

    def h_part(self, subset: tuple[int, ...]) -> float:
        """h(S_t | S_{t-1}) — the part conditioned on ITS OWN past only."""
        s = np.asarray(subset, dtype=int)
        return self._cond_entropy(s + self.n_channels, s)

    def h_whole(self) -> float:
        n = self.n_channels
        idx = np.arange(n)
        return self._cond_entropy(idx + n, idx)


@dataclass(frozen=True)
class MIPSearchResult:
    """A minimum-partition result with an explicit proof boundary."""

    part: tuple[int, ...]
    complement: tuple[int, ...]
    phi: float
    method: str
    certified_exact: bool
    evaluated_cuts: int


@dataclass(frozen=True)
class InterventionalTransition:
    """One named before/after row produced by a real intervention."""

    channel_names: tuple[str, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]


def stochastic_interaction(model: LaggedGaussian,
                           parts: list[tuple[int, ...]]) -> float:
    """SI(P) = Σ_k h(M_k,t | M_k,t-1) − h(X_t | X_{t-1})  (nats, ≥ 0 up to
    estimation noise).  The information the whole's dynamics carries beyond
    the sum of its parts' self-dynamics — integrated information under the
    Gaussian model (Ay 2003; Barrett & Seth 2011 Φ_AR family)."""
    return float(sum(model.h_part(p) for p in parts) - model.h_whole())


# ─────────────────────────────────────────────────────────────────────────────
# Queyranne: EXACT minimum of a symmetric submodular set function
# ─────────────────────────────────────────────────────────────────────────────

def queyranne_min_bipartition(
    n: int, f: Callable[[tuple[int, ...]], float]
) -> tuple[tuple[int, ...], float]:
    """Exact minimizer of a symmetric submodular f over nontrivial subsets.

    Queyranne (1998): repeatedly find a 'pendent pair' (t, u) — u's
    singleton cut value is optimal among all sets separating u from t —
    record {u} (as the current merged group it represents), then contract
    t,u and recurse.  The best recorded candidate is the global minimum
    cut in O(n³) oracle calls.  For the Gaussian stochastic-interaction
    bipartition function this yields the true MIP without enumerating the
    2^(n-1) cuts.
    """
    if n < 2:
        raise ValueError("need at least two elements")
    groups: list[tuple[int, ...]] = [(i,) for i in range(n)]
    best_set: tuple[int, ...] | None = None
    best_val = math.inf

    while len(groups) > 1:
        # pendent-pair sweep over current contracted universe
        order = [0]
        remaining = list(range(1, len(groups)))
        vals: dict[int, float] = {}
        while remaining:
            w_best, v_best = None, math.inf
            merged_prefix: tuple[int, ...] = tuple(
                x for g in (groups[i] for i in order) for x in g
            )
            for cand in remaining:
                # key(cand) = f(prefix ∪ cand) − f({cand})
                cand_set = groups[cand]
                key = f(tuple(sorted(merged_prefix + cand_set))) - f(cand_set)
                if key < v_best:
                    v_best, w_best = key, cand
            order.append(w_best)
            remaining.remove(w_best)
            vals[w_best] = v_best
        t_idx, u_idx = order[-2], order[-1]
        candidate = groups[u_idx]
        cand_val = f(candidate)
        if cand_val < best_val:
            best_val = cand_val
            best_set = candidate
        # contract t and u
        merged = tuple(sorted(groups[t_idx] + groups[u_idx]))
        keep = [g for i, g in enumerate(groups) if i not in (t_idx, u_idx)]
        groups = keep + [merged]

    assert best_set is not None
    return best_set, float(best_val)


def search_minimum_information_bipartition(model: LaggedGaussian) -> MIPSearchResult:
    """Find the best bounded bipartition and state whether it is certified.

    Stochastic interaction is symmetric, but unlike mutual information it is
    not generally submodular.  Queyranne is therefore a strong polynomial
    candidate search, not an exactness proof.  At tractable dimensions this
    function exhausts every unique bipartition; above that boundary it labels
    the result heuristic and adds deterministic one-move local refinement.
    """
    n = model.n_channels
    universe = tuple(range(n))
    h_whole = model.h_whole()
    part_entropy: dict[tuple[int, ...], float] = {}
    cut_values: dict[tuple[int, ...], float] = {}

    def canonical(subset: tuple[int, ...]) -> tuple[int, ...]:
        s = tuple(sorted(set(subset)))
        comp = tuple(i for i in universe if i not in set(s))
        return min(s, comp, key=lambda item: (len(item), item))

    def h_part(subset: tuple[int, ...]) -> float:
        key = tuple(sorted(subset))
        if key not in part_entropy:
            part_entropy[key] = model.h_part(key)
        return part_entropy[key]

    def f(subset: tuple[int, ...]) -> float:
        s = tuple(sorted(set(subset)))
        comp = tuple(i for i in universe if i not in set(s))
        if not s or not comp:
            return 0.0
        key = canonical(s)
        if key not in cut_values:
            key_comp = tuple(i for i in universe if i not in set(key))
            cut_values[key] = h_part(key) + h_part(key_comp) - h_whole
        return cut_values[key]

    if n <= EXACT_GAUSSIAN_MIP_MAX_ELEMENTS:
        best_part: tuple[int, ...] | None = None
        best_val = math.inf
        for mask in range(1, 2 ** (n - 1)):
            candidate = tuple(i for i in universe if (mask >> i) & 1)
            value = f(candidate)
            if value < best_val:
                best_part, best_val = candidate, value
        assert best_part is not None
        part = canonical(best_part)
        method = "exhaustive_bipartition"
        certified_exact = True
    else:
        part, best_val = queyranne_min_bipartition(n, f)
        part = canonical(part)
        # Queyranne is empirically strong for stochastic interaction.  The
        # local pass catches cheap one-node corrections without pretending to
        # turn a non-submodular objective into a polynomial exact search.
        refinement_converged = False
        for _ in range(n * n):
            current = set(part)
            candidates = {
                canonical(tuple(sorted(current ^ {node})))
                for node in universe
                if 0 < len(current ^ {node}) < n
            }
            improved = min(candidates, key=f, default=part)
            improved_val = f(improved)
            if improved_val >= best_val - 1e-12:
                refinement_converged = True
                break
            part, best_val = improved, improved_val
        method = (
            "queyranne_candidate_plus_local_refinement"
            if refinement_converged
            else "queyranne_candidate_plus_capped_local_refinement"
        )
        certified_exact = False

    comp = tuple(i for i in universe if i not in set(part))
    return MIPSearchResult(
        part=part,
        complement=comp,
        phi=max(0.0, float(best_val)),
        method=method,
        certified_exact=certified_exact,
        evaluated_cuts=len(cut_values),
    )


def minimum_information_bipartition(
    model: LaggedGaussian,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Compatibility tuple for the bounded MIP search."""
    result = search_minimum_information_bipartition(model)
    return result.part, result.complement, result.phi


# ─────────────────────────────────────────────────────────────────────────────
# Surrogates, bootstrap, diagnostics
# ─────────────────────────────────────────────────────────────────────────────

# Significance threshold for the family-wise grain-selection test.
PHI_ALPHA = 0.10

# How far below alpha the test's p-floor must sit before a positive result is
# allowed to count. An empirical p from N surrogates cannot go below 1/(N+1);
# a "significant" p that IS that floor is a statement about the test's
# resolution, not about the data. Requiring an order of magnitude of headroom
# means a result that clears alpha did so with room to spare.
PHI_RESOLUTION_MARGIN = 10.0

# Minimum surrogates for a claim: 1/(N+1) ≤ PHI_ALPHA / PHI_RESOLUTION_MARGIN.
PHI_MIN_CLAIM_SURROGATES = int(round(PHI_RESOLUTION_MARGIN / PHI_ALPHA)) - 1  # 99

# Default for a *claim-grade* run. The live service deliberately runs cheaper
# (it is telemetry, and reports integration_established=False rather than a
# false positive); the measurement tool that produces citable artifacts should
# use this or more.
PHI_CLAIM_SURROGATES = 500


def _circular_shift_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently rotate each channel in time: preserves every marginal
    and autocorrelation, destroys cross-channel coupling — the null for
    'integration', not for 'activity'."""
    T = X.shape[0]
    out = np.empty_like(X)
    min_shift = max(2, T // 10)
    if 2 * min_shift >= T:
        min_shift = 1
    for j in range(X.shape[1]):
        shift = int(rng.integers(min_shift, T - min_shift + 1))
        out[:, j] = np.roll(X[:, j], shift)
    return out


def surrogate_null(
    X: np.ndarray, *, n_surrogates: int, rng: np.random.Generator,
    shrinkage: float,
) -> tuple[float, float, list[float]]:
    values: list[float] = []
    for _ in range(n_surrogates):
        Xs = _circular_shift_surrogate(X, rng)
        try:
            m = LaggedGaussian.fit(Xs, shrinkage=shrinkage)
            _, _, val = minimum_information_bipartition(m)
            values.append(val)
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation("integrated_information", exc, severity="debug",
                               action="dropped one degenerate surrogate")
    if not values:
        return 0.0, 0.0, []
    return float(np.mean(values)), float(np.std(values) + 1e-12), values


def block_bootstrap_ci(
    X: np.ndarray, *, n_boot: int, block: int, rng: np.random.Generator,
    shrinkage: float,
) -> tuple[float, float]:
    """(lo, hi) 5–95% interval for Φ via moving-block bootstrap."""
    T = X.shape[0]
    block = max(4, min(block, T // 4))
    starts_max = T - block
    values: list[float] = []
    n_blocks = int(math.ceil(T / block))
    for _ in range(n_boot):
        idx = np.concatenate([
            np.arange(s, s + block)
            for s in rng.integers(0, starts_max, size=n_blocks)
        ])[:T]
        try:
            m = LaggedGaussian.fit(X[idx], shrinkage=shrinkage)
            _, _, val = minimum_information_bipartition(m)
            values.append(val)
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation("integrated_information", exc, severity="debug",
                               action="dropped one degenerate bootstrap draw")
    if not values:
        return 0.0, 0.0
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def data_diagnostics(X: np.ndarray) -> dict[str, float]:
    """Assumption health, reported not hidden: stationarity proxy (half-vs-
    half mean drift in pooled-σ units) and Gaussianity proxies."""
    T = X.shape[0]
    a, b = X[: T // 2], X[T // 2:]
    pooled = X.std(axis=0)
    pooled[pooled < 1e-9] = 1.0
    drift = float(np.median(np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled))
    Xc = (X - X.mean(axis=0)) / pooled
    skew = float(np.median(np.abs((Xc ** 3).mean(axis=0))))
    kurt = float(np.median(np.abs((Xc ** 4).mean(axis=0) - 3.0)))
    return {
        "stationarity_drift_sigma": round(drift, 4),
        "median_abs_skew": round(skew, 4),
        "median_excess_kurtosis": round(kurt, 4),
        "n_samples": float(T),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rail B: grain discovery (coarse-graining search / causal emergence)
# ─────────────────────────────────────────────────────────────────────────────

def _correlation_linkage_groups(X: np.ndarray, k: int) -> list[list[int]]:
    """Average-linkage agglomerative clustering on 1−|corr| distance.
    Deterministic; NumPy only."""
    n = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - np.abs(C)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best = (0, 1, math.inf)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(np.mean([D[a, b] for a in clusters[i] for b in clusters[j]]))
                if d < best[2]:
                    best = (i, j, d)
        i, j, _ = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    return clusters


def coarse_grain(X: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Macro channels = standardized group means (the black-box mapping)."""
    cols = []
    for g in groups:
        v = X[:, g].mean(axis=1)
        sd = v.std()
        cols.append((v - v.mean()) / (sd if sd > 1e-9 else 1.0))
    return np.stack(cols, axis=1)


def bounded_channel_quotient(
    X: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    max_elements: int,
    discovery_X: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...], list[list[int]]]:
    """Preserve every observed channel through a bounded macro quotient."""
    n = X.shape[1]
    limit = max(2, int(max_elements))
    if n <= limit:
        return X, channel_names, [[index] for index in range(n)]
    groups = _correlation_linkage_groups(
        discovery_X if discovery_X is not None else X,
        limit,
    )
    names = tuple(
        f"macro_{index}[{len(group)}]"
        for index, group in enumerate(groups)
    )
    return coarse_grain(X, groups), names, groups


@dataclass(frozen=True)
class GrainResult:
    k: int
    groups: tuple[tuple[int, ...], ...]
    phi_raw: float
    null_mean: float
    null_std: float
    z: float
    mip: tuple[tuple[int, ...], tuple[int, ...]]
    mip_method: str = ""
    mip_certified_exact: bool = False
    selection_score: float = 0.0


@dataclass(frozen=True)
class GrainSearchSummary:
    results: list[GrainResult]
    selection_p: float
    n_surrogates: int
    used_holdout: bool


def grain_search(
    X: np.ndarray,
    *,
    grains: list[int],
    n_surrogates: int,
    rng: np.random.Generator,
    shrinkage: float,
) -> list[GrainResult]:
    """Compatibility grain search without a separate discovery holdout."""
    return _grain_search_summary(
        discovery_X=X,
        evaluation_X=X,
        grains=grains,
        n_surrogates=n_surrogates,
        rng=rng,
        shrinkage=shrinkage,
        used_holdout=False,
    ).results


def _grain_search_summary(
    *,
    discovery_X: np.ndarray,
    evaluation_X: np.ndarray,
    grains: list[int],
    n_surrogates: int,
    rng: np.random.Generator,
    shrinkage: float,
    used_holdout: bool,
) -> GrainSearchSummary:
    """Discover groups once, evaluate them on held-out data, and use a
    shared surrogate family to adjust the best-grain selection."""
    n = evaluation_X.shape[1]
    specs: list[tuple[int, list[list[int]], np.ndarray, MIPSearchResult]] = []
    for k in sorted(set(grains), reverse=True):
        if k < 2 or k > n:
            continue
        groups = ([[i] for i in range(n)] if k == n
                  else _correlation_linkage_groups(discovery_X, k))
        Xk = evaluation_X if k == n else coarse_grain(evaluation_X, groups)
        try:
            observed = search_minimum_information_bipartition(
                LaggedGaussian.fit(Xk, shrinkage=shrinkage)
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation(
                "integrated_information",
                exc,
                severity="debug",
                action=f"skipped degenerate grain k={k}",
            )
            continue
        specs.append((k, groups, Xk, observed))

    nulls: dict[int, list[float]] = {k: [] for k, *_ in specs}
    for _ in range(max(0, int(n_surrogates))):
        surrogate = _circular_shift_surrogate(evaluation_X, rng)
        for k, groups, _, _observed in specs:
            Xk = surrogate if k == n else coarse_grain(surrogate, groups)
            try:
                value = search_minimum_information_bipartition(
                    LaggedGaussian.fit(Xk, shrinkage=shrinkage)
                ).phi
                nulls[k].append(value)
            except (np.linalg.LinAlgError, ValueError) as exc:
                record_degradation(
                    "integrated_information",
                    exc,
                    severity="debug",
                    action=f"dropped degenerate grain surrogate k={k}",
                )

    results: list[GrainResult] = []
    stats: dict[int, tuple[float, float]] = {}
    for k, groups, _Xk, observed in specs:
        values = nulls[k]
        mu = float(np.mean(values)) if values else 0.0
        sd = float(np.std(values) + 1e-12) if values else 0.0
        z = (observed.phi - mu) / sd if sd > 0 else 0.0
        stats[k] = (mu, sd)
        results.append(GrainResult(
            k=k,
            groups=tuple(tuple(g) for g in groups),
            phi_raw=round(observed.phi, 6),
            null_mean=round(mu, 6),
            null_std=round(sd, 6),
            z=round(float(z), 3),
            mip=(observed.part, observed.complement),
            mip_method=observed.method,
            mip_certified_exact=observed.certified_exact,
            selection_score=round(observed.phi / k, 9),
        ))

    selected_score = max((result.selection_score for result in results), default=0.0)
    complete_draws = min((len(values) for values in nulls.values()), default=0)
    max_null_score: list[float] = []
    for index in range(complete_draws):
        max_null_score.append(
            max(
                (values[index] / k for k, values in nulls.items()),
                default=0.0,
            )
        )
    selection_p = (
        (1.0 + sum(value >= selected_score for value in max_null_score))
        / (len(max_null_score) + 1.0)
        if max_null_score else 1.0
    )
    return GrainSearchSummary(
        results=results,
        selection_p=round(float(selection_p), 6),
        n_surrogates=len(max_null_score),
        used_holdout=used_holdout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rail C: exact discrete Φ at the derived grain (k ≤ 12)
# ─────────────────────────────────────────────────────────────────────────────

MAX_EXACT_ELEMENTS = 12


def exact_state_phi(
    Xk: np.ndarray,
    *,
    extra_transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Exact system-level integrated information of the binarized macro
    dynamics, by exhaustive bipartition search.

    φ_s = min over bipartitions (A,B) of
          E_t[ KL( P(S_t | s_{t-1}) ‖ P(A_t | a_{t-1}) ⊗ P(B_t | b_{t-1}) ) ]

    averaged over the EMPIRICALLY VISITED states (plus any interventional
    transitions supplied by the perturbational probe — counterfactual rows
    sampled by intervention rather than enumerated).  Laplace-smoothed
    (α).  Exact in the search, empirical in the distribution: both facts
    are part of the estimator's name and claim.
    """
    if alpha < 0:
        raise ValueError("Dirichlet smoothing alpha must be non-negative")
    k = Xk.shape[1]
    if k > MAX_EXACT_ELEMENTS:
        raise ValueError(f"exact search capped at {MAX_EXACT_ELEMENTS} elements")
    med = np.median(Xk, axis=0)
    B = (Xk > med).astype(np.int8)
    trans: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    observed_transitions = max(0, B.shape[0] - 1)
    for t in range(observed_transitions):
        key = (tuple(int(v) for v in B[t]), tuple(int(v) for v in B[t + 1]))
        trans[key] = trans.get(key, 0) + 1
    accepted_interventions = 0
    rejected_interventions = 0
    for pair in (extra_transitions or []):
        s0 = tuple(int(v) for v in pair[0])
        s1 = tuple(int(v) for v in pair[1])
        if len(s0) == k and len(s1) == k:
            trans[(s0, s1)] = trans.get((s0, s1), 0) + 1
            accepted_interventions += 1
        else:
            rejected_interventions += 1

    # group transitions by source state
    by_src: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
    for (s0, s1), c in trans.items():
        by_src.setdefault(s0, {})[s1] = by_src.get(s0, {}).get(s1, 0.0) + c

    def _marginal(dist: dict[tuple[int, ...], float], idx: tuple[int, ...]
                  ) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for s1, p in dist.items():
            sub = tuple(s1[i] for i in idx)
            out[sub] = out.get(sub, 0.0) + p
        return out

    def n_states(m: int) -> float:
        return float(2 ** m)

    best = {"phi": math.inf, "partition": None}
    total_weight = sum(sum(d.values()) for d in by_src.values())
    for mask in range(1, 2 ** (k - 1)):
        A = tuple(i for i in range(k) if (mask >> i) & 1)
        Bp = tuple(i for i in range(k) if not (mask >> i) & 1)
        # conditional marginal tables for parts, keyed by part-source
        partA: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
        partB: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
        srcA_count: dict[tuple[int, ...], float] = {}
        srcB_count: dict[tuple[int, ...], float] = {}
        for s0, dist in by_src.items():
            a0 = tuple(s0[i] for i in A)
            b0 = tuple(s0[i] for i in Bp)
            w = sum(dist.values())
            srcA_count[a0] = srcA_count.get(a0, 0.0) + w
            srcB_count[b0] = srcB_count.get(b0, 0.0) + w
            mA = _marginal(dist, A)
            mB = _marginal(dist, Bp)
            dA = partA.setdefault(a0, {})
            for s, p in mA.items():
                dA[s] = dA.get(s, 0.0) + p
            dB = partB.setdefault(b0, {})
            for s, p in mB.items():
                dB[s] = dB.get(s, 0.0) + p

        kl_sum = 0.0
        for s0, dist in by_src.items():
            a0 = tuple(s0[i] for i in A)
            b0 = tuple(s0[i] for i in Bp)
            w = sum(dist.values())
            denomW = w + alpha * n_states(k)
            denomA = srcA_count[a0] + alpha * n_states(len(A))
            denomB = srcB_count[b0] + alpha * n_states(len(Bp))
            # Coherent Dirichlet-smoothed KL over the complete binary support.
            # The entropy term handles unseen joint outcomes analytically; the
            # two cross-entropy terms do the same for each part's support.
            entropy = 0.0
            for c in dist.values():
                p = (c + alpha) / denomW
                entropy -= p * math.log(max(p, _EPS))
            unseen_joint = int(n_states(k)) - len(dist)
            if alpha > 0 and unseen_joint > 0:
                p_unseen = alpha / denomW
                entropy -= unseen_joint * p_unseen * math.log(max(p_unseen, _EPS))

            def _cross_entropy(
                current_counts: dict[tuple[int, ...], float],
                pooled_counts: dict[tuple[int, ...], float],
                *,
                p_multiplicity: int,
                p_denom: float,
                q_denom: float,
                support_size: int,
            ) -> float:
                keys = set(current_counts) | set(pooled_counts)
                value = 0.0
                for state in keys:
                    p_num = current_counts.get(state, 0.0) + alpha * p_multiplicity
                    if p_num <= 0:
                        continue
                    q_num = pooled_counts.get(state, 0.0) + alpha
                    value -= (p_num / p_denom) * math.log(
                        max(q_num / q_denom, _EPS)
                    )
                missing = support_size - len(keys)
                if alpha > 0 and missing > 0:
                    p_num = alpha * p_multiplicity
                    value -= missing * (p_num / p_denom) * math.log(
                        max(alpha / q_denom, _EPS)
                    )
                return value

            currentA = _marginal(dist, A)
            currentB = _marginal(dist, Bp)
            crossA = _cross_entropy(
                currentA,
                partA[a0],
                p_multiplicity=2 ** len(Bp),
                p_denom=denomW,
                q_denom=denomA,
                support_size=2 ** len(A),
            )
            crossB = _cross_entropy(
                currentB,
                partB[b0],
                p_multiplicity=2 ** len(A),
                p_denom=denomW,
                q_denom=denomB,
                support_size=2 ** len(Bp),
            )
            kl = -entropy + crossA + crossB
            kl_sum += (w / total_weight) * kl
        if kl_sum < best["phi"]:
            best = {"phi": kl_sum, "partition": (A, Bp)}

    return {
        "estimator": DISCRETE_ESTIMATOR_NAME,
        "phi_s": round(max(0.0, float(best["phi"])), 6),
        "mip": best["partition"],
        "n_elements": k,
        "n_observed_transitions": observed_transitions,
        "n_interventional_transitions": accepted_interventions,
        "n_rejected_interventional_transitions": rejected_interventions,
        "n_total_transitions": int(total_weight),
        "smoothing_alpha": float(alpha),
        "cuts_searched": 2 ** (k - 1) - 1,
    }


def project_interventional_transitions(
    transitions: list[
        InterventionalTransition
        | tuple[tuple[int, ...], tuple[int, ...]]
    ] | None,
    *,
    channel_names: tuple[str, ...],
    groups: list[list[int]],
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], int]:
    """Align named micro interventions and project them to a macro grain."""
    projected: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    rejected = 0
    n_micro = len(channel_names)
    n_macro = len(groups)

    def valid_bits(values: tuple[int, ...]) -> bool:
        return all(value in (0, 1) for value in values)

    for transition in transitions or []:
        direct_macro = False
        if isinstance(transition, InterventionalTransition):
            if (
                len(transition.channel_names) != len(transition.before)
                or len(transition.before) != len(transition.after)
                or len(set(transition.channel_names)) != len(transition.channel_names)
            ):
                rejected += 1
                continue
            indexed = {
                name: index for index, name in enumerate(transition.channel_names)
            }
            if any(name not in indexed for name in channel_names):
                rejected += 1
                continue
            before = tuple(int(transition.before[indexed[name]]) for name in channel_names)
            after = tuple(int(transition.after[indexed[name]]) for name in channel_names)
        else:
            before = tuple(int(value) for value in transition[0])
            after = tuple(int(value) for value in transition[1])
            if len(before) == n_macro and len(after) == n_macro and n_macro != n_micro:
                direct_macro = True
            elif len(before) != n_micro or len(after) != n_micro:
                rejected += 1
                continue

        if not valid_bits(before) or not valid_bits(after):
            rejected += 1
            continue
        if direct_macro:
            projected.append((before, after))
            continue

        def collapse(values: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(
                int(sum(values[index] for index in group) * 2 >= len(group))
                for group in groups
            )

        projected.append((collapse(before), collapse(after)))
    return projected, rejected


# ─────────────────────────────────────────────────────────────────────────────
# The full estimate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhiEstimate:
    """One provenance-complete whole-system Φ report."""

    schema_version: int
    estimator: str
    computed_at: float
    n_channels: int
    n_samples: int
    channel_names: tuple[str, ...]
    # Rail A — full-grain
    phi_raw: float
    mip: tuple[tuple[int, ...], tuple[int, ...]]
    null_mean: float
    null_std: float
    z: float
    ci_5: float
    ci_95: float
    # Rail B — grains
    grains: list[GrainResult] = field(default_factory=list)
    emergent_grain_k: int = 0
    emergence_delta_z: float = 0.0
    # Rail C — exact discrete at the derived grain
    exact_macro: dict[str, Any] = field(default_factory=dict)
    # Rail D honesty
    diagnostics: dict[str, Any] = field(default_factory=dict)
    claim: str = ""
    random_seed: int = 0
    mip_method: str = ""
    mip_certified_exact: bool = False
    mip_evaluated_cuts: int = 0
    emergent_phi_raw: float = 0.0
    emergent_z: float = 0.0
    emergent_ci_5: float = 0.0
    emergent_ci_95: float = 0.0
    grain_selection_p: float = 1.0
    grain_selection_surrogates: int = 0
    grain_selection_used_holdout: bool = False
    n_observed_channels: int = 0
    observed_channel_names: tuple[str, ...] = ()
    quotient_groups: tuple[tuple[int, ...], ...] = ()

    def p_floor(self) -> float:
        """The smallest p this test could possibly report.

        An empirical p from N surrogates is (1 + #{null ≥ observed}) / (N + 1),
        so its minimum is 1/(N+1). With 20 surrogates that floor is 0.047619 —
        and a reported p of exactly 0.047619 means only "no null draw beat the
        observed score". It does not distinguish a true p of 0.04 from 0.004,
        nor from a result that would not replicate at all.
        """
        n = int(self.grain_selection_surrogates or 0)
        return 1.0 / (n + 1) if n > 0 else 1.0

    def surrogate_resolution_sufficient(self) -> bool:
        """Can the family-wise null resolve a claim at ``PHI_ALPHA``?

        A significance claim made at the resolution limit of its own test is not
        a claim. We require the floor to sit an order of magnitude below the
        threshold, so a p that clears alpha does so with room to spare rather
        than because it could not have been any smaller.

        Scoped to the grain-selection test on purpose. This asks about
        ``grain_selection_p``; an estimate that never selected a grain does not
        use that p as a criterion, so the question does not apply to it — see
        :meth:`integration_established`.
        """
        if not self.emergent_grain_k:
            return True
        return self.p_floor() <= PHI_ALPHA / PHI_RESOLUTION_MARGIN

    def integration_established(self) -> bool:
        """Family-wise evidence at the selected grain, not a raw maximum.

        Two different criteria, and the resolution gate belongs to only one of
        them:

        No grain selected — the criterion is the z-score and CI against the main
        circular-shift null. No p-value is consulted, so the grain-selection
        p-floor is not a question that can be asked about this estimate.

        Grain selected — the criterion includes ``grain_selection_p``, and that
        is where the floor bites: the checked-in campaign reported family-wise
        p = 0.047619 from 20 surrogates, which is exactly 1/(20+1). It cleared
        p ≤ 0.10 only because the test had no finer resolution to offer, so the
        null must be able to resolve the threshold before the verdict counts.
        """
        if not self.emergent_grain_k:
            return self.z >= 3.0 and self.ci_5 > 0.0
        if not self.surrogate_resolution_sufficient():
            return False
        return (
            self.emergent_z >= 3.0
            and self.emergent_ci_5 > 0.0
            and self.grain_selection_p <= PHI_ALPHA
            and self.grain_selection_used_holdout
        )

    def claim_limits(self) -> list[str]:
        """Everything that stops this estimate being a publishable claim.

        Returned on the artifact so a reader does not have to reconstruct the
        caveats from the raw numbers.
        """
        limits: list[str] = []
        if not self.surrogate_resolution_sufficient():
            limits.append(
                f"surrogate_resolution: {self.grain_selection_surrogates} surrogates "
                f"give a p-floor of {self.p_floor():.6f}; at least "
                f"{PHI_MIN_CLAIM_SURROGATES} are needed to resolve p ≤ {PHI_ALPHA} "
                f"with a {PHI_RESOLUTION_MARGIN}x margin"
            )
        if self.grain_selection_p <= self.p_floor() + 1e-12 and self.grain_selection_surrogates:
            limits.append(
                f"p_at_floor: family-wise p={self.grain_selection_p:.6f} is the "
                f"minimum this test can report; the true p is unresolved"
            )
        if not self.grain_selection_used_holdout and self.emergent_grain_k:
            limits.append("grain_selection_not_held_out: selection and evaluation share data")
        if self.n_samples < 1000:
            limits.append(f"short_window: {self.n_samples} samples")
        return limits

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "estimator": self.estimator,
            "computed_at": self.computed_at,
            "n_channels": self.n_channels,
            "n_observed_channels": self.n_observed_channels or self.n_channels,
            "n_samples": self.n_samples,
            "channel_names": list(self.channel_names),
            "observed_channel_names": list(
                self.observed_channel_names or self.channel_names
            ),
            "quotient_groups": [list(group) for group in self.quotient_groups],
            "phi_raw": self.phi_raw,
            "mip": [list(self.mip[0]), list(self.mip[1])],
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z": self.z,
            "ci_5": self.ci_5,
            "ci_95": self.ci_95,
            "random_seed": self.random_seed,
            "mip_method": self.mip_method,
            "mip_certified_exact": self.mip_certified_exact,
            "mip_evaluated_cuts": self.mip_evaluated_cuts,
            "grains": [
                {
                    "k": g.k, "phi_raw": g.phi_raw, "z": g.z,
                    "null_mean": g.null_mean, "null_std": g.null_std,
                    "groups": [list(x) for x in g.groups],
                    "mip_method": g.mip_method,
                    "mip_certified_exact": g.mip_certified_exact,
                    "selection_score": g.selection_score,
                }
                for g in self.grains
            ],
            "emergent_grain_k": self.emergent_grain_k,
            "emergence_delta_z": self.emergence_delta_z,
            "emergent_phi_raw": self.emergent_phi_raw,
            "emergent_z": self.emergent_z,
            "emergent_ci_5": self.emergent_ci_5,
            "emergent_ci_95": self.emergent_ci_95,
            "grain_selection_p": self.grain_selection_p,
            "grain_selection_surrogates": self.grain_selection_surrogates,
            "grain_selection_used_holdout": self.grain_selection_used_holdout,
            "exact_macro": dict(self.exact_macro),
            "diagnostics": dict(self.diagnostics),
            "integration_established": self.integration_established(),
            # A reader must not have to derive 1/(N+1) by hand to discover that
            # a reported p sits at the test's resolution limit.
            "p_floor": round(self.p_floor(), 8),
            "p_at_floor": bool(
                self.grain_selection_surrogates
                and self.grain_selection_p <= self.p_floor() + 1e-12
            ),
            "surrogate_resolution_sufficient": self.surrogate_resolution_sufficient(),
            "claim_limits": self.claim_limits(),
            "claim": self.claim,
        }


def _bounded_claim(est: PhiEstimate) -> str:
    verdict = ("integration beats chance" if est.integration_established()
               else "integration NOT established against the null")
    mip_proof = (
        "certified exhaustive MIP"
        if est.mip_certified_exact
        else "bounded heuristic MIP candidate"
    )
    p_note = f"family-wise p={est.grain_selection_p:.3f}"
    if est.grain_selection_surrogates and est.grain_selection_p <= est.p_floor() + 1e-12:
        # Report the floor inline. "p=0.048" reads as a finding; "p=0.048, the
        # floor for 20 surrogates" reads as what it is.
        p_note += (
            f" (= the 1/(N+1) floor for {est.grain_selection_surrogates} "
            f"surrogates; true p unresolved)"
        )
    limits = est.claim_limits()
    limits_note = f" Claim limits: {'; '.join(limits)}." if limits else ""
    return (
        f"Micro Φ̂={est.phi_raw:.4f} nats over {est.n_channels} estimator "
        f"elements covering {est.n_observed_channels or est.n_channels} live channels "
        f"({est.n_samples} samples; {mip_proof}; z={est.z:.1f}). Selected "
        f"held-out grain k={est.emergent_grain_k}: Φ̂={est.emergent_phi_raw:.4f}, "
        f"z={est.emergent_z:.1f}, {p_note}, "
        f"90% CI [{est.emergent_ci_5:.4f}, {est.emergent_ci_95:.4f}]. "
        f"{verdict}.{limits_note} This is an "
        "estimate of integrated information of the system's macro-dynamics "
        "under a Gaussian model of its OWN measured channels — evidence "
        "about integration structure, not a consciousness meter, and not "
        "formal IIT 4.0 Φ."
    )


def estimate_whole_system_phi(
    X: np.ndarray,
    *,
    channel_names: tuple[str, ...] | None = None,
    n_surrogates: int = 20,
    n_boot: int = 20,
    grains: list[int] | None = None,
    seed: int = 0,
    shrinkage: float = 0.05,
    max_effective_elements: int = 16,
    extra_transitions: list[
        InterventionalTransition
        | tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
) -> PhiEstimate:
    """The full four-rail estimate on a T×N channel matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 16 or X.shape[1] < 2:
        raise ValueError("channel matrix must be T×N with T ≥ 16, N ≥ 2")
    if channel_names is None:
        channel_names = tuple(f"ch{i}" for i in range(X.shape[1]))
    if len(channel_names) != X.shape[1]:
        raise ValueError("channel_names must match the channel matrix width")
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Drop dead channels (zero variance) — reported, not hidden.
    live = X.std(axis=0) > 1e-9
    dropped = [channel_names[i] for i in range(len(channel_names)) if not live[i]]
    X = X[:, live]
    channel_names = tuple(
        name for name, keep in zip(channel_names, live, strict=True) if keep
    )
    observed_X = X
    observed_channel_names = channel_names
    quotient_discovery = observed_X[: observed_X.shape[0] // 2]
    X, channel_names, quotient_groups = bounded_channel_quotient(
        observed_X,
        observed_channel_names,
        max_elements=max_effective_elements,
        discovery_X=quotient_discovery,
    )
    n = X.shape[1]

    model = LaggedGaussian.fit(X, shrinkage=shrinkage)
    mip = search_minimum_information_bipartition(model)
    part, comp, phi = mip.part, mip.complement, mip.phi
    mu, sd, _ = surrogate_null(X, n_surrogates=n_surrogates, rng=rng,
                               shrinkage=shrinkage)
    z = (phi - mu) / sd if sd > 0 else 0.0
    lo, hi = block_bootstrap_ci(X, n_boot=n_boot, block=max(8, X.shape[0] // 10),
                                rng=rng, shrinkage=shrinkage)

    if grains is None:
        grains = sorted({n, max(2, n // 2), max(2, n // 4), 8, 4})
    split = X.shape[0] // 2
    discovery_X = X[:split]
    evaluation_X = X[split:]
    grain_summary = _grain_search_summary(
        discovery_X=discovery_X,
        evaluation_X=evaluation_X,
        grains=[g for g in grains if 2 <= g <= n],
        n_surrogates=max(10, n_surrogates),
        rng=rng,
        shrinkage=shrinkage,
        used_holdout=True,
    )
    grain_results = grain_summary.results

    emergent = max(grain_results, key=lambda g: g.selection_score, default=None)
    micro = next((g for g in grain_results if g.k == n), None)
    emergent_lo, emergent_hi = 0.0, 0.0
    emergent_X = None
    if emergent is not None:
        emergent_groups = [list(group) for group in emergent.groups]
        emergent_X = (
            evaluation_X
            if emergent.k == n
            else coarse_grain(evaluation_X, emergent_groups)
        )
        emergent_lo, emergent_hi = block_bootstrap_ci(
            emergent_X,
            n_boot=n_boot,
            block=max(8, emergent_X.shape[0] // 10),
            rng=rng,
            shrinkage=shrinkage,
        )

    exact_macro: dict[str, Any] = {}
    if emergent is not None and emergent.k <= MAX_EXACT_ELEMENTS:
        groups = [list(g) for g in emergent.groups]
        Xk = coarse_grain(X, groups)
        composed_groups = [
            [
                observed_index
                for quotient_index in group
                for observed_index in quotient_groups[quotient_index]
            ]
            for group in groups
        ]
        projected, projection_rejected = project_interventional_transitions(
            extra_transitions,
            channel_names=observed_channel_names,
            groups=composed_groups,
        )
        try:
            exact_macro = exact_state_phi(Xk, extra_transitions=projected)
            exact_macro["n_projection_rejected_transitions"] = projection_rejected
        except (ValueError, ArithmeticError) as exc:
            record_degradation("integrated_information", exc, severity="warning",
                               action="exact macro phi skipped")

    diagnostics = data_diagnostics(X)
    diagnostics["dropped_dead_channels"] = float(len(dropped))
    diagnostics["quotient_applied"] = float(
        len(observed_channel_names) != len(channel_names)
    )
    diagnostics["compute_seconds"] = round(time.time() - t0, 3)
    diagnostics["grain_discovery_samples"] = float(discovery_X.shape[0])
    diagnostics["grain_evaluation_samples"] = float(evaluation_X.shape[0])

    est = PhiEstimate(
        schema_version=SCHEMA_VERSION,
        estimator=ESTIMATOR_NAME,
        computed_at=time.time(),
        n_channels=n,
        n_samples=X.shape[0],
        channel_names=channel_names,
        phi_raw=round(phi, 6),
        mip=(part, comp),
        null_mean=round(mu, 6),
        null_std=round(sd, 6),
        z=round(float(z), 3),
        ci_5=round(lo, 6),
        ci_95=round(hi, 6),
        grains=grain_results,
        emergent_grain_k=emergent.k if emergent else 0,
        emergence_delta_z=round((emergent.z - micro.z), 3) if emergent and micro else 0.0,
        exact_macro=exact_macro,
        diagnostics=diagnostics,
        random_seed=int(seed),
        mip_method=mip.method,
        mip_certified_exact=mip.certified_exact,
        mip_evaluated_cuts=mip.evaluated_cuts,
        emergent_phi_raw=emergent.phi_raw if emergent else 0.0,
        emergent_z=emergent.z if emergent else 0.0,
        emergent_ci_5=round(emergent_lo, 6),
        emergent_ci_95=round(emergent_hi, 6),
        grain_selection_p=grain_summary.selection_p,
        grain_selection_surrogates=grain_summary.n_surrogates,
        grain_selection_used_holdout=grain_summary.used_holdout,
        n_observed_channels=len(observed_channel_names),
        observed_channel_names=observed_channel_names,
        quotient_groups=tuple(tuple(group) for group in quotient_groups),
    )
    est.claim = _bounded_claim(est)
    return est
