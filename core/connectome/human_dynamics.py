"""core/connectome/human_dynamics.py — her activity against a human recording.

Operationally: holds the published statistics of human cortical activity, takes
a trace of hers, and says for each one whether it holds, by how much it misses,
and what would have refuted it. It fits nothing and tunes nothing.

What this is and is not
-----------------------
It is a comparison against the summary statistics of human recordings, which is
what can be done offline. It is not a comparison against raw MEG or single-unit
data, because none is in this repository and downloading some would not make the
comparison better — the statistics below are what those datasets are held to
mean, and they are what a model has to reproduce.

So a passing line here says her trace has the same scaling as a human resting
recording. It does not say her trace resembles any particular person's.

Why these statistics
--------------------
Cortical activity near criticality has scaling that does not depend on what the
cortex is made of, which is exactly why it can be compared across a brain and a
mesh. Three numbers carry it:

* the avalanche SIZE distribution's exponent, 1.5
* the avalanche DURATION distribution's exponent, 2.0
* the relation between them, (alpha - 1) / (tau - 1), which has to come out at
  2.0 if both of the above hold and the system is critical

The third is the one that matters. Either exponent alone can be produced by
things that are not criticality — a subsampled Poisson process gives a power
law — and the scaling relation between them cannot. It is the reason this
module reports the relation as the test and the two exponents as its inputs.

The branching parameter is here as a fourth, and it is the one to be careful
with: a critical system has m = 1, and cortex recorded in a living animal comes
out just under it. A model that reports exactly 1.0 has probably been tuned to.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Connectome.HumanDynamics")

__all__ = [
    "HUMAN_STATISTICS",
    "HumanStatistic",
    "Verdict",
    "compare_to_human_cortex",
]


@dataclass(frozen=True, slots=True)
class HumanStatistic:
    """One published number a recording of cortex is held to."""

    name: str
    value: float
    tolerance: float
    source: str
    recorded_in: str
    what_it_is: str
    #: What a trace would have to show for this to be refused. Written before
    #: the trace is taken.
    falsified_by: str


#: What cortex does, as published. Nothing here was measured in this repository.
HUMAN_STATISTICS: tuple[HumanStatistic, ...] = (
    HumanStatistic(
        name="avalanche_size_exponent",
        value=1.5,
        tolerance=0.3,
        source="Beggs & Plenz 2003, J Neurosci 23(35):11167; Shriki et al. 2013, J Neurosci 33(16):7079",
        recorded_in="cortical slice, and resting human MEG",
        what_it_is="how the number of cascades falls off with how big they are",
        falsified_by="an exponent outside 1.2 to 1.8, or a distribution that is not a power law",
    ),
    HumanStatistic(
        name="avalanche_duration_exponent",
        value=2.0,
        tolerance=0.4,
        source="Beggs & Plenz 2003, J Neurosci 23(35):11167",
        recorded_in="cortical slice",
        what_it_is="how the number of cascades falls off with how long they last",
        falsified_by="an exponent outside 1.6 to 2.4",
    ),
    HumanStatistic(
        name="crackling_relation",
        value=2.0,
        tolerance=0.2,
        source="Sethna, Dahmen & Myers 2001, Nature 410:242; Friedman et al. 2012, PRL 108:208102",
        recorded_in="cortical culture, and every critical system in the class",
        what_it_is=(
            "how a cascade's size grows with its duration, which has to equal "
            "(duration exponent - 1) / (size exponent - 1) if the system is critical"
        ),
        falsified_by=(
            "a measured growth that disagrees with what the two exponents predict "
            "by more than 0.2, which is a system with power laws and no criticality"
        ),
    ),
    HumanStatistic(
        name="branching_parameter",
        value=0.98,
        tolerance=0.05,
        source="Wilting & Priesemann 2018, Nat Commun 9:2325",
        recorded_in="in vivo mammalian cortex, by multistep regression",
        what_it_is="how much activity one unit of activity begets",
        falsified_by=(
            "a ratio above 1.03, which is a network on its way to saturation, or "
            "below 0.93, which is one that forgets its input"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Verdict:
    """One statistic, what her trace gave, and whether that holds."""

    statistic: HumanStatistic
    observed: float
    holds: bool
    reason: str

    @property
    def miss(self) -> float:
        return abs(self.observed - self.statistic.value)

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.statistic.name,
            "human": self.statistic.value,
            "hers": round(self.observed, 4),
            "miss": round(self.miss, 4),
            "tolerance": self.statistic.tolerance,
            "holds": self.holds,
            "reason": self.reason,
            "source": self.statistic.source,
            "recorded_in": self.statistic.recorded_in,
        }


def _statistic(name: str) -> HumanStatistic:
    for entry in HUMAN_STATISTICS:
        if entry.name == name:
            return entry
    raise KeyError(name)


def compare_to_human_cortex(
    activity: Sequence[float],
    *,
    percentile: float = 25.0,
    per_unit: Any = None,
    per_unit_percentile: float = 75.0,
) -> dict[str, Any]:
    """Hold a trace of hers against every published statistic at once.

    A trace passes by satisfying several of these together, not by matching the
    friendliest one. Two systems can share an exponent and differ everywhere
    else, which is why the scaling relation is here and why the report says how
    many held rather than whether any did.
    """
    from .criticality import assess

    report = assess(
        activity,
        percentile=percentile,
        per_unit=per_unit,
        per_unit_percentile=per_unit_percentile,
    )
    observed = {
        "avalanche_size_exponent": float(report.size_exponent.get("alpha", 0.0)),
        "avalanche_duration_exponent": float(report.duration_exponent.get("alpha", 0.0)),
        "crackling_relation": float(report.measured_gamma),
        "branching_parameter": float(report.branching.m),
    }

    usable = (
        float(report.size_exponent.get("ks", 1.0)) < 0.2
        and float(report.duration_exponent.get("ks", 1.0)) < 0.2
        and int(report.size_exponent.get("tail_n", 0)) >= 32
    )

    verdicts: list[Verdict] = []
    for name, value in observed.items():
        statistic = _statistic(name)
        if value <= 0.0:
            verdicts.append(
                Verdict(statistic, value, False, "the trace gave no usable estimate")
            )
            continue
        if name != "branching_parameter" and not usable:
            verdicts.append(
                Verdict(
                    statistic,
                    value,
                    False,
                    "the avalanche fits are too poor to compare; too few cascades or "
                    "a distribution that is not a power law",
                )
            )
            continue
        miss = abs(value - statistic.value)
        holds = miss <= statistic.tolerance
        verdicts.append(
            Verdict(
                statistic,
                value,
                holds,
                ""
                if holds
                else f"{miss:.3f} away from {statistic.value}, past the {statistic.tolerance} allowed",
            )
        )

    held = [verdict for verdict in verdicts if verdict.holds]
    return {
        "held": len(held),
        "of": len(verdicts),
        "usable_fits": usable,
        "avalanches": report.avalanches,
        # Why the fits are or are not usable, because "too poor to compare" is
        # three different situations and they need different answers: too few
        # large cascades wants a longer recording, a bad KS wants a different
        # model, and a missing tail wants neither.
        "size_fit": {key: round(float(value), 4) for key, value in report.size_exponent.items()},
        "duration_fit": {
            key: round(float(value), 4) for key, value in report.duration_exponent.items()
        },
        "predicted_crackling": round(float(report.predicted_gamma), 4),
        "statistics": [verdict.as_json() for verdict in verdicts],
        "verdict": (
            f"{len(held)} of {len(verdicts)} published statistics of cortical activity "
            f"hold on this trace"
        ),
    }
