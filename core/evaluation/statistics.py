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
    mean_delta = float(np.mean(x) - np.mean(y))
    pooled_var = ((x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1)) / max(1, x.size + y.size - 2)
    if pooled_var <= 1e-12:
        if abs(mean_delta) <= 1e-12:
            return 0.0
        return float(math.copysign(min(abs(mean_delta) / 1e-6, 1_000_000.0), mean_delta))
    return float(mean_delta / math.sqrt(pooled_var))


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
    #: Mean of the term this statistic subtracts — what the score reads when
    #: the intervention did nothing. Recorded so a reader can see the null
    #: rather than take it on faith.
    null_reference_mean: float = 0.0
    #: Mean of the term measuring the intervention itself.
    treatment_mean: float = 0.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.01 and not (self.ci_low <= 0.0 <= self.ci_high)


def paired_effect_over_null_reference(
    treatment_outputs: Sequence[str],
    baseline_outputs: Sequence[str],
    null_reference_outputs: Sequence[str],
    *,
    distance: Callable[[str, str], float] | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ABComparison:
    """How far an intervention moved the output, minus how far it moves anyway.

    ::

        score_i = distance(treatment_i, baseline_i)
                - distance(baseline_i, null_reference_i)

    ``null_reference_i`` is an independent REDRAW of the baseline condition:
    same prompt, no intervention, a different sampling seed. It measures the
    system's own run-to-run variation.

    That is what puts the null at zero. Under "the intervention did nothing"
    the treated output is just another such redraw, so both terms estimate the
    same quantity and ``E[score] = 0``. A positive, significant score means the
    intervention moved the output further than the system moves on its own —
    which is the only thing a divergence measurement can establish.

    It cannot establish that the movement is in an intended direction. Use a
    scored target behaviour (``paired_score_shift``) for that; divergence and
    direction are different claims and one does not imply the other.

    Why this function exists in this shape
    --------------------------------------
    It replaces ``paired_distance_comparison``, which scored::

        distance(treatment, control) - distance(treatment, baseline)

    Its callers ran the treatment and the baseline from the same prompt under
    the same seed, toggling only the intervention. So an intervention with NO
    effect made ``treatment == baseline``, the subtracted term exactly zero,
    and the score equal to ``distance(baseline, control)`` — which is positive
    by construction, because the control deliberately uses a different prompt.
    The null hypothesis produced a decisive pass, and did: the checked-in
    steering artifact reported d = 2.50, p = 0.0002 with steered and baseline
    samples that are word-for-word identical.

    ``null_effect_probe`` below turns that into a check any effect statistic
    can be held to.
    """
    if not (
        len(treatment_outputs) == len(baseline_outputs) == len(null_reference_outputs)
    ):
        raise ValueError("all output lists must have the same length")
    metric = distance or jaccard_distance
    treated = np.array(
        [metric(t, b) for t, b in zip(treatment_outputs, baseline_outputs)],
        dtype=np.float64,
    )
    reference = np.array(
        [metric(b, r) for b, r in zip(baseline_outputs, null_reference_outputs)],
        dtype=np.float64,
    )
    deltas = treated - reference
    observed, p = permutation_test(
        deltas,
        np.zeros_like(deltas),
        n_permutations=n_resamples,
        alternative="greater",
        seed=seed,
    )
    ci_low, ci_high = bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
    d = cohens_d(deltas, np.zeros_like(deltas))
    return ABComparison(
        observed,
        p,
        ci_low,
        ci_high,
        d,
        null_reference_mean=float(np.mean(reference)),
        treatment_mean=float(np.mean(treated)),
    )


def paired_score_shift(
    treatment_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ABComparison:
    """Did the intervention move a SCORED target behaviour, and which way?

    Divergence says an output changed. This says it changed toward the thing
    the intervention was supposed to produce, which is a separate claim and the
    one an affect-steering result actually needs. ``significant`` here is a
    two-sided question turned one-sided by the caller's choice of sign: pass
    the scores so that "more of the intended behaviour" is larger.
    """
    if len(treatment_scores) != len(baseline_scores):
        raise ValueError("all score lists must have the same length")
    deltas = _as_1d(treatment_scores) - _as_1d(baseline_scores)
    observed, p = permutation_test(
        deltas,
        np.zeros_like(deltas),
        n_permutations=n_resamples,
        alternative="greater",
        seed=seed,
    )
    ci_low, ci_high = bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed)
    d = cohens_d(deltas, np.zeros_like(deltas))
    return ABComparison(
        observed,
        p,
        ci_low,
        ci_high,
        d,
        null_reference_mean=float(np.mean(_as_1d(baseline_scores))),
        treatment_mean=float(np.mean(_as_1d(treatment_scores))),
    )


def null_effect_probe(
    build_comparison: Callable[[list[str], list[str], list[str]], ABComparison],
    *,
    n_trials: int = 40,
    seed: int = 0,
) -> ABComparison:
    """Run an effect statistic on data where the intervention did NOTHING.

    Generic by design: hand it anything that turns
    ``(treatment, baseline, null_reference)`` into an ``ABComparison`` and it
    supplies a world in which the treatment is byte-identical to the baseline
    and the null reference is an ordinary redraw. Whatever comes back is what
    the statistic reports when there is nothing to report.

    A statistic is fit to publish only if this comes back NOT significant. The
    steering A/B shipped for months without anyone running its equivalent, and
    it would have returned d ≈ 2.5, p ≈ 0.0002 — the same numbers the live
    campaign produced, from data containing no effect whatsoever.

    This is a probe, not an assertion: callers decide what to do with the
    verdict, and evaluation suites should assert ``not result.significant``.
    """
    rng = np.random.default_rng(seed)
    vocabulary = [f"w{i}" for i in range(60)]

    def _draw() -> str:
        return " ".join(rng.choice(vocabulary, size=18, replace=True).tolist())

    baseline = [_draw() for _ in range(int(n_trials))]
    # The intervention changed nothing: the treated run IS the baseline run.
    treatment = list(baseline)
    # An honest null reference: the same condition sampled again.
    null_reference = [_draw() for _ in range(int(n_trials))]
    return build_comparison(treatment, baseline, null_reference)
