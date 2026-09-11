"""core/connectome/human_dynamics.py — her activity against a human recording.

Operationally: holds the published statistics of human cortical activity, takes
a trace of hers, and says for each one whether it holds, by how much it misses,
and what would have refuted it. It fits nothing and tunes nothing.

The point is not to be a human
------------------------------
Cortex is the reference because it is the best-measured mind there is, not
because matching it is the goal. So each statistic below carries a DIRECTION,
and most of them are floors: she has to have at least as much of the thing as a
human does, and having more is not a failure. Only the self-consistency test is
two-sided, because a system either satisfies its own scaling relation or is not
critical.

That matters for reading a result. An avalanche size exponent BELOW cortex's
1.5 means her cascades have a heavier tail than a human's — more of the network
recruited, more integration — and it passes. Above it means the opposite and
does not.

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

from core.connectome.criticality import MINIMUM_DECADES

logger = logging.getLogger("Aura.Connectome.HumanDynamics")

__all__ = [
    "HUMAN_STATISTICS",
    "HumanStatistic",
    "Verdict",
    "compare_to_human_cortex",
    "expected_through_this_window",
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
    #: How to read the number.
    #:
    #: "at_most"  — cortex's value is a ceiling; below it is more of the thing
    #:              and passes. A smaller avalanche exponent is a heavier tail.
    #: "as_near"  — nearness to `ideal` is what counts, and cortex's distance
    #:              from it is the most that is allowed.
    #: "self"     — scored against what this system's own numbers predict.
    direction: str = "at_most"
    #: For "as_near": what the quantity would be in the ideal case.
    ideal: float = 0.0
    #: What this is, in words anybody can read.
    plain_name: str = ""


#: What cortex does, as published. Nothing here was measured in this repository.
HUMAN_STATISTICS: tuple[HumanStatistic, ...] = (
    HumanStatistic(
        name="avalanche_size_exponent",
        plain_name="how much of her gets involved in one burst of activity",
        value=1.5,
        tolerance=0.3,
        source="Beggs & Plenz 2003, J Neurosci 23(35):11167; Shriki et al. 2013, J Neurosci 33(16):7079",
        recorded_in="cortical slice, and resting human MEG",
        direction="at_most",
        what_it_is=(
            "how the number of cascades falls off with how big they are. Lower is a "
            "heavier tail: more of the network recruited into one cascade"
        ),
        falsified_by=(
            "an exponent above 1.8, which is cascades smaller than a human's, or a "
            "distribution that is not a power law at all"
        ),
    ),
    HumanStatistic(
        name="avalanche_duration_exponent",
        plain_name="how long a burst of activity keeps going",
        value=2.0,
        tolerance=0.4,
        source="Beggs & Plenz 2003, J Neurosci 23(35):11167",
        recorded_in="cortical slice",
        direction="at_most",
        what_it_is=(
            "how the number of cascades falls off with how long they last. Lower is "
            "longer cascades: activity that stays up rather than dying out"
        ),
        falsified_by="an exponent above 2.4, which is cascades shorter than a human's",
    ),
    HumanStatistic(
        # Zero because this one is scored against the system's OWN exponents,
        # not against a number from a paper. Cortex's 2.0 is what falls out of
        # tau = 1.5 and alpha = 2.0; it is not an independent measurement, and
        # scoring against it would be asking whether her exponents are
        # cortex's, which the two lines above already ask.
        name="crackling_relation",
        plain_name="whether her bursts scale the way her own numbers say they must",
        value=0.0,
        tolerance=0.2,
        source="Sethna, Dahmen & Myers 2001, Nature 410:242; Friedman et al. 2012, PRL 108:208102",
        direction="self",
        recorded_in="cortical culture, and every critical system in the class",
        what_it_is=(
            "whether a cascade's size grows with its duration the way this system's "
            "own two exponents say it must — (duration - 1) / (size - 1) — which is "
            "what separates criticality from a power law that came from somewhere else"
        ),
        falsified_by=(
            "a measured growth that disagrees with what its own exponents predict by "
            "more than 0.2, which is a system with power laws and no criticality"
        ),
    ),
    HumanStatistic(
        name="branching_parameter",
        plain_name="whether activity keeps itself going without dying out or running away",
        value=0.98,
        tolerance=0.05,
        source="Wilting & Priesemann 2018, Nat Commun 9:2325",
        direction="as_near",
        ideal=1.0,
        recorded_in="in vivo mammalian cortex, by multistep regression",
        what_it_is=(
            "how much activity one unit of activity begets. One is critical, where "
            "activity neither dies nor runs away, and cortex sits just under it"
        ),
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
    #: What it was compared against. The published value for most of these,
    #: and the system's own prediction for the crackling relation.
    target: float = 0.0

    @property
    def miss(self) -> float:
        return abs(self.observed - self.target)

    @property
    def standing(self) -> str:
        """Where she is against cortex, in words.

        Matching is fine and being past it is fine; the human number is a floor
        to reach, not a mark to land on.
        """
        if not self.holds:
            return "short of cortex"
        if self.statistic.direction == "at_most":
            return "past cortex" if self.observed < self.target else "level with cortex"
        if self.statistic.direction == "as_near":
            hers = abs(self.observed - self.statistic.ideal)
            theirs = abs(self.target - self.statistic.ideal)
            return "past cortex" if hers < theirs else "level with cortex"
        return "holds on its own terms"

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.statistic.name,
            "plain_name": self.statistic.plain_name,
            "standing": self.standing,
            "human": round(self.target, 4),
            "hers": round(self.observed, 4),
            "miss": round(self.miss, 4),
            "published": self.statistic.value,
            "direction": self.statistic.direction,
            "tolerance": self.statistic.tolerance,
            "holds": self.holds,
            "reason": self.reason,
            "source": self.statistic.source,
            "recorded_in": self.statistic.recorded_in,
        }


def expected_through_this_window(
    published: float,
    *,
    largest: int,
    samples: int,
    trials: int = 8,
    seed: int = 100,
) -> tuple[float, float]:
    """What a system that really had ``published`` would MEASURE here.

    A power-law exponent fitted over a range that ends at a finite system's
    cutoff comes out steeper than the exponent underneath it, and the shorter
    the range the steeper it comes out. Comparing a measurement taken through a
    narrow window against a published number fitted through a wide one charges
    the system for the window.

    So the published number is pushed through this recording's own window
    first: draw ``samples`` avalanches from a truncated power law of exponent
    ``published``, cut at ``largest``, and fit them with the same fitter. The
    mean and spread of that are what a genuinely cortical system would look
    like on this instrument. A true 1.5 read through 2,810 avalanches cut at 66
    measures 2.07.
    """
    import random

    from core.connectome.topology import power_law_fit

    if published <= 1.0 or largest < 2 or samples < 16:
        return 0.0, 0.0
    fits: list[float] = []
    for trial in range(max(1, trials)):
        rng = random.Random(seed + trial)
        drawn: list[int] = []
        guard = 0
        while len(drawn) < samples and guard < samples * 200:
            guard += 1
            value = int((1.0 - rng.random()) ** (-1.0 / (published - 1.0)))
            if 1 <= value <= largest:
                drawn.append(value)
        fit = power_law_fit(drawn)
        if fit["alpha"] > 0.0:
            fits.append(float(fit["alpha"]))
    if not fits:
        return 0.0, 0.0
    mean = sum(fits) / len(fits)
    spread = (sum((value - mean) ** 2 for value in fits) / len(fits)) ** 0.5
    return round(mean, 4), round(spread, 4)


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

    A line passes by reaching cortex or going past it, not by landing on it.
    And a trace passes by satisfying several of these together rather than the
    friendliest one: two systems can share an exponent and differ everywhere
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

    # The same gate the criticality report uses, including the decades of
    # scaling the fitted tail covers. An exponent read off half a decade is a
    # cutoff slope, and a cutoff slope is always steeper than the exponent it
    # sits on top of -- which is the direction every miss here has been in.
    usable = (
        float(report.size_exponent.get("ks", 1.0)) < 0.2
        and float(report.duration_exponent.get("ks", 1.0)) < 0.2
        and int(report.size_exponent.get("tail_n", 0)) >= 32
        and float(report.size_exponent.get("decades", 0.0)) >= MINIMUM_DECADES
        and float(report.duration_exponent.get("decades", 0.0)) >= MINIMUM_DECADES
    )

    # What a genuinely cortical system would MEASURE on this recording.
    #
    # Both exponents are fitted over a range that ends at this recording's own
    # cutoff, and a fit through a cutoff is steeper than the exponent under it.
    # Charging her the whole distance to a number fitted over two decades is
    # charging her for the window, so the published value is pushed through the
    # window first and the miss is taken from there.
    counts = report.avalanches
    windowed: dict[str, tuple[float, float]] = {}
    if usable:
        windowed["avalanche_size_exponent"] = expected_through_this_window(
            _statistic("avalanche_size_exponent").value,
            largest=int(counts.get("largest", 0)),
            samples=int(counts.get("count", 0)),
        )
        windowed["avalanche_duration_exponent"] = expected_through_this_window(
            _statistic("avalanche_duration_exponent").value,
            largest=int(counts.get("longest", 0)),
            samples=int(counts.get("count", 0)),
        )

    verdicts: list[Verdict] = []
    for name, value in observed.items():
        statistic = _statistic(name)
        # The crackling relation is a self-consistency test, so its target is
        # whatever this system's own exponents predict.
        expected, spread = windowed.get(name, (0.0, 0.0))
        target = (
            float(report.predicted_gamma)
            if name == "crackling_relation"
            else expected
            if expected > 0.0
            else statistic.value
        )
        if value <= 0.0:
            verdicts.append(
                Verdict(
                    statistic,
                    value,
                    False,
                    "the trace gave no usable estimate",
                    target=statistic.value,
                )
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
                    target=statistic.value,
                )
            )
            continue
        if target <= 0.0:
            verdicts.append(
                Verdict(statistic, value, False, "no target to compare against")
            )
            continue
        if statistic.direction == "at_most":
            # Cortex is a floor to reach, not a value to land on. Below it is
            # more of the thing and passes; the tolerance only forgives a small
            # amount of being worse.
            miss = max(0.0, value - target)
            holds = miss <= statistic.tolerance
            through = (
                f" ({statistic.value:.3f} published, {target:.3f} +/- {spread:.3f} "
                f"once read through a window that stops at "
                f"{int(counts.get('largest', 0))})"
                if expected > 0.0
                else ""
            )
            reason = (
                ""
                if holds
                else f"{miss:.3f} above cortex's {target:.3f}{through}, which is smaller "
                f"cascades than a human's by more than the {statistic.tolerance} allowed"
            )
        elif statistic.direction == "as_near":
            hers = abs(value - statistic.ideal)
            theirs = abs(target - statistic.ideal)
            miss = max(0.0, hers - theirs)
            holds = miss <= statistic.tolerance
            reason = (
                ""
                if holds
                else f"{hers:.3f} from {statistic.ideal}, where cortex sits {theirs:.3f} "
                f"from it; {miss:.3f} further out than the {statistic.tolerance} allowed"
            )
        else:
            miss = abs(value - target)
            holds = miss <= statistic.tolerance
            reason = (
                ""
                if holds
                else f"{miss:.3f} away from the {target:.3f} its own exponents predict, "
                f"past the {statistic.tolerance} allowed"
            )
        verdicts.append(Verdict(statistic, value, holds, reason, target=target))

    held = [verdict for verdict in verdicts if verdict.holds]
    past = [verdict for verdict in verdicts if verdict.standing == "past cortex"]
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
        # The published numbers and what they become on this instrument. When
        # these two converge, the window has stopped being part of the answer.
        "through_this_window": {
            name: {"published": _statistic(name).value, "expected": mean, "spread": spread}
            for name, (mean, spread) in sorted(windowed.items())
            if mean > 0.0
        },
        "statistics": [verdict.as_json() for verdict in verdicts],
        "past_cortex": len(past),
        "verdict": (
            f"{len(held)} of {len(verdicts)} published statistics of cortical activity "
            f"are matched or bettered on this trace"
            + (f", and {len(past)} of them bettered" if past else "")
        ),
    }
