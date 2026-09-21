"""Exact paired-test power, conditional on discordance or planned over tasks."""

from __future__ import annotations

import math

from core.verify.invariants import invariant
from typing import Any


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a probability")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability")
    return value


def _count(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("pair count must be a nonnegative integer")
    return value


def binomial_tail(k: int, n: int, probability: float) -> float:
    """P(X >= k), without overflowing integer combinations at campaign scale."""
    from scipy.stats import binom

    _count(n)
    if type(k) is not int:
        raise ValueError("tail boundary must be an integer")
    probability = _probability(probability, "probability")
    return float(binom.sf(k - 1, n, probability))


def _critical_counts(counts: Any, alpha: float) -> Any:
    import numpy as np
    from scipy.stats import binom

    # isf returns the last accepted integer. Check both discrete boundaries
    # instead of relying on inverse-CDF rounding near an attainable alpha.
    critical = np.asarray(binom.isf(alpha, counts, 0.5), dtype=np.int64) + 1
    lower = (critical > 0) & (binom.sf(critical - 2, counts, 0.5) <= alpha)
    while np.any(lower):
        critical = np.where(lower, critical - 1, critical)
        lower = (critical > 0) & (binom.sf(critical - 2, counts, 0.5) <= alpha)
    higher = (critical <= counts) & (binom.sf(critical - 1, counts, 0.5) > alpha)
    while np.any(higher):
        critical = np.where(higher, critical + 1, critical)
        higher = (critical <= counts) & (binom.sf(critical - 1, counts, 0.5) > alpha)
    return critical


def conditional_mcnemar_power(discordant: int, alpha: float, win_share: float) -> float:
    """Rejection probability conditional on an explicit discordant-pair count."""
    import numpy as np
    from scipy.stats import binom

    _count(discordant)
    alpha = _probability(alpha, "alpha")
    win_share = _probability(win_share, "win_share")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    critical = _critical_counts(np.asarray([discordant]), alpha)[0]
    return float(binom.sf(critical - 1, discordant, win_share))


def prospective_mcnemar_power(
    tasks: int, *, alpha: float, discordance: float, win_share: float,
) -> float:
    """Integrate over D~Binomial(tasks, discordance), then W|D~Binomial(D, share).

    Tasks must be independent sampling units. Correlated repeats and random
    seeds for the same task cannot be counted as independent extra tasks.
    Inputs are design assumptions, never an estimate of an observed gain.
    """
    import numpy as np
    from scipy.stats import binom

    _count(tasks)
    alpha = _probability(alpha, "alpha")
    discordance = _probability(discordance, "discordance")
    win_share = _probability(win_share, "win_share")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    counts = np.arange(tasks + 1, dtype=np.int64)
    critical = _critical_counts(counts, alpha)
    mass = binom.pmf(counts, tasks, discordance)
    power = float(np.dot(mass, binom.sf(critical - 1, counts, win_share)))
    return min(1.0, max(0.0, power))


@invariant("evaluation.tasks_are_not_discordant_pairs", scope="evaluation",
           owner="core/evaluation/paired_power.py", observational=False)
def _tasks_are_not_discordant_pairs() -> tuple:
    assert prospective_mcnemar_power(100, alpha=0.05, discordance=0, win_share=1) == 0
    assert conditional_mcnemar_power(4, 0.05, 1) == 0
    assert conditional_mcnemar_power(5, 0.05, 1) == 1
    return ()
