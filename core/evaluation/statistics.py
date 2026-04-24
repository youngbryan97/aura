"""Statistical utilities for adversarial Aura evaluations.

These helpers are deliberately dependency-light so the decisive verification
bundle can run on a fresh checkout. They provide the basics missing from older
claim snapshots: bootstrap confidence intervals, permutation tests, effect
sizes, and bias-aware mutual-information checks with permutation baselines.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


def _as_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("at least one value is required")
    return arr


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval."""
    arr = _as_1d(values)
    stat = statistic or (lambda sample: float(np.mean(sample)))
    rng = np.random.default_rng(seed)
    n = arr.size
    reps = np.empty(int(n_resamples), dtype=np.float64)
    for i in range(int(n_resamples)):
        sample = arr[rng.integers(0, n, size=n)]
        reps[i] = stat(sample)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def cohens_d(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Pooled Cohen's d for two independent samples."""
    x = _as_1d(a)
    y = _as_1d(b)
    if x.size < 2 or y.size < 2:
        return 0.0
    pooled_var = ((x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1)) / max(1, x.size + y.size - 2)
    if pooled_var <= 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled_var))


def permutation_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 2000,
    alternative: str = "two-sided",
    seed: int = 0,
) -> tuple[float, float]:
    """Permutation test for the difference between two samples.

    Returns ``(observed_statistic, p_value)``.
    """
    x = _as_1d(a)
    y = _as_1d(b)
    stat = statistic or (lambda left, right: float(np.mean(left) - np.mean(right)))
    observed = stat(x, y)
    combined = np.concatenate([x, y])
    n_x = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(combined)
        perm_stat = stat(perm[:n_x], perm[n_x:])
        if alternative == "greater":
            count += perm_stat >= observed
        elif alternative == "less":
            count += perm_stat <= observed
        else:
            count += abs(perm_stat) >= abs(observed)
    p = (count + 1.0) / (int(n_permutations) + 1.0)
    return float(observed), float(p)


def mutual_information_discrete(x: Iterable[object], y: Iterable[object]) -> float:
    """Finite-sample discrete mutual information in bits."""
    xs = list(x)
    ys = list(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if not xs:
        return 0.0

    x_vals = {value: idx for idx, value in enumerate(sorted(set(xs), key=repr))}
    y_vals = {value: idx for idx, value in enumerate(sorted(set(ys), key=repr))}
    joint = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    for x_val, y_val in zip(xs, ys):
        joint[x_vals[x_val], y_vals[y_val]] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (joint > 0.0) & (denom > 0.0)
    return float(np.sum(joint[mask] * np.log2(joint[mask] / denom[mask])))


def mutual_information_permutation_baseline(
    x: Iterable[object],
    y: Iterable[object],
    *,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare observed MI to shuffled-target baselines."""
    xs = list(x)
    ys = list(y)
    observed = mutual_information_discrete(xs, ys)
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_permutations), dtype=np.float64)
    y_arr = np.asarray(ys, dtype=object)
    for i in range(int(n_permutations)):
        null[i] = mutual_information_discrete(xs, rng.permutation(y_arr).tolist())
    p = (float(np.sum(null >= observed)) + 1.0) / (int(n_permutations) + 1.0)
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float(p),
    }


_WORD_RE = re.compile(r"[a-z0-9']+")


def word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(str(text).lower()))


def jaccard_distance(a: str, b: str) -> float:
    wa = word_set(a)
    wb = word_set(b)
    if not wa and not wb:
        return 0.0
    return 1.0 - (len(wa & wb) / max(1, len(wa | wb)))


@dataclass(frozen=True)
class ABComparison:
    observed_delta: float
    p_value: float
    ci_low: float
    ci_high: float
    effect_size_d: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.01 and not (self.ci_low <= 0.0 <= self.ci_high)


def paired_distance_comparison(
    treatment_outputs: Sequence[str],
    control_outputs: Sequence[str],
    baseline_outputs: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ABComparison:
    """Compare treatment-vs-control text divergence against baseline.

    The score for each trial is:
        distance(treatment, control) - distance(treatment, baseline)

    Positive values mean the control is farther from the treatment than the
    baseline is. That is a conservative check used by the steering A/B harness.
    """
    if not (len(treatment_outputs) == len(control_outputs) == len(baseline_outputs)):
        raise ValueError("all output lists must have the same length")
    deltas = np.array(
        [
            jaccard_distance(a, b) - jaccard_distance(a, c)
            for a, b, c in zip(treatment_outputs, control_outputs, baseline_outputs)
        ],
        dtype=np.float64,
    )
    observed, p = permutation_test(
        deltas,
        np.zeros_like(deltas),
        n_permutations=n_resamples,
        alternative="greater",
        seed=seed,
    )
    ci_low, ci_high = bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
    d = cohens_d(deltas, np.zeros_like(deltas))
    return ABComparison(observed, p, ci_low, ci_high, d)
