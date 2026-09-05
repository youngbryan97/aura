"""tools/agi_gauntlet/scoring.py — what the numbers have to survive.

Four things, and each is a way a gauntlet gets a result it did not earn.

**Two scores, not one.** P₀ is what she can do on arrival; P_L is what she can
do after the learning the task allows. A benchmark that gives thirty seconds
and marks the answer measures zero-shot capability, and a system that starts
worse and learns faster may be the more generally intelligent one. Both are
reported, and so is the difference.

**Accuracy is half of it.** Solving a trivial problem with a million
environment interactions is not the same competence as solving it in twenty.
Every gate that acts in a world reports what it spent beside what it got.

**A difference has to beat its noise.** Thirty trajectories, not one lucky
run, and a difference of means carries the spread that says whether it is a
difference. A bootstrap rather than a t-test, because the distributions here
are small, bounded and nothing like normal.

**Transfer needs its negative controls.** T_i = P(B_i|A_i) − P(B_i|∅) is a
number that goes up whenever two things look alike. The controls are pairs
built to look alike and to differ underneath, and a system that transfers
there is matching surfaces. A transfer result without them is a statement
about the evaluator's choice of alphabet.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Comparison",
    "Learning",
    "Transfer",
    "compare",
    "efficiency",
    "learning_curve",
    "transfer_gain",
]


#: Resamples for the bootstrap. Enough that the interval stops moving in the
#: third decimal, which is where these comparisons are decided.
HOW_MANY_RESAMPLES = 2000


@dataclass(frozen=True)
class Comparison:
    """Two sets of trajectories, and whether they differ."""

    name: str
    here: tuple[float, ...]
    there: tuple[float, ...]
    difference: float
    low: float
    high: float
    effect: float

    @property
    def real(self) -> bool:
        """Whether the interval excludes no difference."""
        return (self.low > 0.0) or (self.high < 0.0)

    @property
    def enough_trajectories(self) -> bool:
        """Thirty each. One lucky run is not a result."""
        return len(self.here) >= 30 and len(self.there) >= 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_here": len(self.here),
            "n_there": len(self.there),
            "mean_here": round(_mean(self.here), 4),
            "mean_there": round(_mean(self.there), 4),
            "difference": round(self.difference, 4),
            "interval": [round(self.low, 4), round(self.high, 4)],
            "effect": round(self.effect, 4),
            "real": self.real,
            "enough_trajectories": self.enough_trajectories,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((one - mean) ** 2 for one in values) / (len(values) - 1))


def compare(
    name: str,
    here: Sequence[float],
    there: Sequence[float],
    *,
    seed: int = 0,
    resamples: int = HOW_MANY_RESAMPLES,
) -> Comparison:
    """The difference of means, with a bootstrap interval and an effect size.

    Seeded, because a confidence interval that moves between processes is not
    a confidence interval.
    """

    here, there = tuple(here), tuple(there)
    if not here or not there:
        return Comparison(name, here, there, 0.0, 0.0, 0.0, 0.0)
    observed = _mean(here) - _mean(there)
    rng = random.Random(seed)
    drawn: list[float] = []
    for _ in range(resamples):
        a = [here[rng.randrange(len(here))] for _ in here]
        b = [there[rng.randrange(len(there))] for _ in there]
        drawn.append(_mean(a) - _mean(b))
    drawn.sort()
    low = drawn[int(0.025 * len(drawn))]
    high = drawn[min(len(drawn) - 1, int(0.975 * len(drawn)))]
    pooled = math.sqrt((_sd(here) ** 2 + _sd(there) ** 2) / 2) or 1e-9
    return Comparison(name, here, there, observed, low, high, observed / pooled)


@dataclass(frozen=True)
class Learning:
    """What a curve did, rather than what its last point was."""

    name: str
    points: tuple[float, ...]
    first: float
    last: float

    @property
    def gain(self) -> float:
        """P_L − P₀. For some systems this matters more than P₀."""
        return self.last - self.first

    @property
    def rose(self) -> bool:
        return self.gain > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": [round(one, 4) for one in self.points],
            "P0": round(self.first, 4),
            "PL": round(self.last, 4),
            "gain": round(self.gain, 4),
            "rose": self.rose,
        }


def learning_curve(name: str, points: Sequence[float], *, window: int = 3) -> Learning:
    """A curve read at its ends, with the ends averaged over a window.

    A single first point and a single last point make a curve out of two
    coin flips. The window is small because these runs are short and the
    first thing an over-wide window does is hide the learning it is measuring.
    """

    points = tuple(float(one) for one in points)
    if not points:
        return Learning(name, (), 0.0, 0.0)
    take = max(1, min(window, len(points) // 2 or 1))
    return Learning(
        name=name,
        points=points,
        first=_mean(points[:take]),
        last=_mean(points[-take:]),
    )


@dataclass(frozen=True)
class Transfer:
    """T̄ over the pairs, and what the controls did."""

    gains: tuple[float, ...] = ()
    control_gains: tuple[float, ...] = ()
    comparison: Comparison | None = None

    @property
    def mean_gain(self) -> float:
        return _mean(self.gains)

    @property
    def control_gain(self) -> float:
        return _mean(self.control_gains)

    @property
    def transferred(self) -> bool:
        """Real transfer, and none where there should be none.

        Both halves. A system that transfers everywhere has learned that
        two problems in a row are related, which is a fact about the harness.
        """

        return (
            self.mean_gain > 0.0
            and self.control_gain <= 0.0 + 1e-9
            and bool(self.comparison and self.comparison.real)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": len(self.gains),
            "controls": len(self.control_gains),
            "mean_gain": round(self.mean_gain, 4),
            "control_gain": round(self.control_gain, 4),
            "transferred": self.transferred,
            "comparison": self.comparison.to_dict() if self.comparison else None,
        }


def transfer_gain(
    after_learning: Sequence[float],
    from_scratch: Sequence[float],
    *,
    control_after: Sequence[float] = (),
    control_scratch: Sequence[float] = (),
    seed: int = 0,
) -> Transfer:
    """T_i = P(B_i | A_i) − P(B_i | ∅), over the pairs and over the controls."""

    gains = tuple(a - b for a, b in zip(after_learning, from_scratch))
    controls = tuple(a - b for a, b in zip(control_after, control_scratch))
    return Transfer(
        gains=gains,
        control_gains=controls,
        comparison=compare("transfer against its controls", gains, controls or (0.0,), seed=seed),
    )


def efficiency(spent: Sequence[float], fewest: Sequence[float]) -> dict[str, Any]:
    """What it cost against what it could have cost.

    Reported beside accuracy everywhere, because a human finishing in twenty
    actions and a system finishing in five hundred are not the same
    competence even when both finish.
    """

    ratios = [
        (float(one) / float(least)) if least else float("inf")
        for one, least in zip(spent, fewest)
        if one is not None
    ]
    usable = [one for one in ratios if one != float("inf")]
    return {
        "median_ratio": round(sorted(usable)[len(usable) // 2], 3) if usable else None,
        "mean_ratio": round(_mean(usable), 3) if usable else None,
        "worst_ratio": round(max(usable), 3) if usable else None,
        "measured": len(usable),
    }
