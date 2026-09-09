"""Poke one domain and see how far, how long and how differently it carries.

The clinical measure this borrows from perturbs cortex and asks whether the
response is at once widespread and differentiated, because the two failure
modes are opposite and both are common: a response that stays where it started,
and a response in which everything moves together. One is a disconnected
system, the other is a broadcast, and neither is a mind.

The binary matrix is built against a floor rather than a fixed number. A
domain counts as reached at a lag when the displaced arm has diverged from its
sham further than two untouched arms diverged from each other at that same lag
in the same condition. That floor is measured, not assumed, so a domain that is
simply noisy does not light up.

Complexity is Lempel-Ziv over the matrix, normalised the way the clinical index
normalises it, by the entropy of the ones it contains. A matrix of all ones and
a matrix of one row both compress to almost nothing; only a response that is
spread out and structured scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.causal import InterventionSet
from core.subject.state import DOMAINS

__all__ = ["PerturbationReport", "lempel_ziv", "perturbational_complexity"]

#: A domain counts as reached when it moves past this quantile of the floor.
FLOOR_QUANTILE: float = 0.95

#: And by at least this much in units of ordinary variation, so a domain whose
#: sham floor is essentially zero cannot be lit up by float noise.
FLOOR_FLOOR: float = 0.02


def lempel_ziv(bits: np.ndarray) -> int:
    """LZ76 distinct-substring count over a binary sequence."""
    sequence = "".join("1" if int(b) else "0" for b in bits.reshape(-1))
    n = len(sequence)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    complexity, k_max = 1, 1
    while True:
        if i + k > n or l + k > n:
            break
        if sequence[i + k - 1] == sequence[l + k - 1]:
            k += 1
            if l + k > n:
                complexity += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return complexity


def normalised_lz(matrix: np.ndarray) -> float:
    """LZ divided by the value a matrix of that size and density can reach."""
    bits = matrix.reshape(-1)
    n = bits.size
    if n < 8:
        return 0.0
    p = float(bits.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    ceiling = n * entropy / math.log2(n)
    if ceiling <= 0.0:
        return 0.0
    return float(lempel_ziv(bits) / ceiling)


@dataclass
class PerturbationReport:
    source: str
    spread: float
    depth: float
    reached: tuple[str, ...]
    pci: float
    matrix: np.ndarray
    lags: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "spread": round(self.spread, 4),
            "depth": round(self.depth, 4),
            "reached": list(self.reached),
            "pci": round(self.pci, 4),
            "lags": self.lags,
            "rows": ["".join(str(int(v)) for v in row) for row in self.matrix],
        }


def _threshold(floor: list[list[float]], lag: int) -> float:
    column = [row[lag] for row in floor if lag < len(row)]
    if not column:
        return FLOOR_FLOOR
    return max(FLOOR_FLOOR, float(np.quantile(np.asarray(column), FLOOR_QUANTILE)))


def perturbational_complexity(
    results: InterventionSet,
    *,
    source: str,
    condition: str | None = None,
    arm: str = "effect",
) -> PerturbationReport:
    """The binary response matrix for one displaced domain, and its complexity.

    ``arm="floor"`` builds the same matrix from the two untouched runs instead
    of the displaced one, against the same thresholds. That is the null this
    measure needs: identical shape, identical bar, no intervention. Whatever it
    scores is what the procedure returns when nothing happened.
    """
    trials = [
        trial
        for trial in results.trials
        if trial.source == source and (condition is None or trial.condition == condition)
    ]
    targets = [key for key in DOMAINS if key != source]
    if not trials:
        return PerturbationReport(source, 0.0, 0.0, (), 0.0, np.zeros((0, 0)), 0)

    signal_of = (lambda trial: trial.trace) if arm == "effect" else (lambda trial: trial.floor_trace)
    lags = min(
        (len(series) for trial in trials for series in trial.trace.values()),
        default=0,
    )
    if lags == 0:
        return PerturbationReport(source, 0.0, 0.0, (), 0.0, np.zeros((0, 0)), 0)

    matrix = np.zeros((len(targets), lags), dtype=np.int8)
    for row, target in enumerate(targets):
        floors = [trial.floor_trace.get(target, []) for trial in trials]
        for lag in range(lags):
            bar = _threshold(floors, lag)
            hits = [
                signal_of(trial).get(target, [])[lag]
                for trial in trials
                if lag < len(signal_of(trial).get(target, []))
            ]
            if hits and float(np.median(hits)) > bar:
                matrix[row, lag] = 1

    reached = tuple(
        target for row, target in enumerate(targets) if matrix[row].any()
    )
    # Out of the domains that could be reached. A displaced domain cannot
    # reach itself, so dividing by all ten made a perfect result read as 0.9
    # and put a tenth of the scale out of reach of every source.
    spread = len(reached) / float(len(targets)) if targets else 0.0
    touched_lags = int((matrix.sum(axis=0) > 0).sum())
    return PerturbationReport(
        source=source,
        spread=spread,
        depth=touched_lags / float(lags),
        reached=reached,
        pci=normalised_lz(matrix),
        matrix=matrix,
        lags=lags,
    )
