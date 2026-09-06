"""core/connectome/neuromodulation.py — a modulator is a field, not a dial.

Aura already has a neurochemical system with chemicals, receptor subtypes,
decay and events. What it does not have is the thing every human receptor atlas
shows first: a neuromodulator does not arrive everywhere equally, and the same
level means different things in different places. Hansen and colleagues mapped
nineteen receptors and transporters across nine transmitter systems from more
than 1,200 people, and the headline is spatial. Dopamine is not a number that
the brain has more or less of.

This module adds the missing dimension without inventing any of it.

**The roles are borrowed and cited.** Doya's assignment gives each modulator a
computational parameter rather than a mood: dopamine reports the prediction
error, acetylcholine sets the learning rate, noradrenaline sets the inverse
temperature that trades exploration against exploitation, and serotonin sets
the discount factor — how far ahead the system is willing to care about. Each
one names the parameter it controls and the consumer that reads it, so a
modulator with no consumer is visible as decoration.

**The spatial map is estimated, never asserted.** There is no PET scan of Aura,
and writing per-region receptor densities by hand would be inventing data to
match a conclusion. The sensitivity of each region to each modulator is fitted
from recordings, and the fit carries the grade of the evidence behind it.
Observational fitting — regress region activity on modulator level over normal
operation — gives a correlation and is labelled as one. Interventional fitting
— set the level, record, set it again — gives a causal estimate, and only that
grade is allowed to license a claim that the modulator drives the region.

The distinction is enforced in code. :meth:`ReceptorField.claim` refuses to
return a causal statement from observational data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Connectome.Neuromodulation")

__all__ = [
    "Modulator",
    "ModulatorRole",
    "MODULATOR_ROLES",
    "Evidence",
    "RegionSensitivity",
    "ReceptorField",
    "fit_observational",
    "fit_interventional",
]


class Modulator(StrEnum):
    """The four with a settled computational assignment."""

    DOPAMINE = "dopamine"
    ACETYLCHOLINE = "acetylcholine"
    NORADRENALINE = "noradrenaline"
    SEROTONIN = "serotonin"


class Evidence(StrEnum):
    """How a sensitivity was arrived at. The grade caps what may be claimed."""

    NONE = "none"
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"


@dataclass(frozen=True)
class ModulatorRole:
    """What a modulator is claimed to do, and who is supposed to read it."""

    modulator: Modulator
    parameter: str
    description: str
    citation: str
    consumers: tuple[str, ...]
    #: The direction a rise in the modulator is claimed to push the parameter.
    sign: int = 1

    def as_json(self) -> dict[str, Any]:
        return {
            "modulator": str(self.modulator),
            "parameter": self.parameter,
            "description": self.description,
            "citation": self.citation,
            "consumers": list(self.consumers),
            "sign": self.sign,
        }


#: Doya, "Metalearning and neuromodulation", Neural Networks 15:495 (2002),
#: with the uncertainty split from Yu & Dayan, Neuron 46:681 (2005).
MODULATOR_ROLES: dict[Modulator, ModulatorRole] = {
    Modulator.DOPAMINE: ModulatorRole(
        modulator=Modulator.DOPAMINE,
        parameter="prediction_error",
        description=(
            "Reports the difference between what was expected and what happened. "
            "A rise means the outcome beat the prediction."
        ),
        citation="Doya 2002; Schultz, Dayan & Montague, Science 275:1593 (1997)",
        consumers=("core.learning", "core.memory.reconsolidation"),
    ),
    Modulator.ACETYLCHOLINE: ModulatorRole(
        modulator=Modulator.ACETYLCHOLINE,
        parameter="learning_rate",
        description=(
            "Sets how far a single outcome moves an estimate. Yu and Dayan tie it "
            "to expected uncertainty: known noise argues for slower updates."
        ),
        citation="Doya 2002; Yu & Dayan, Neuron 46:681 (2005)",
        consumers=("core.learning", "core.adaptation"),
    ),
    Modulator.NORADRENALINE: ModulatorRole(
        modulator=Modulator.NORADRENALINE,
        parameter="inverse_temperature",
        description=(
            "Trades exploration against exploitation. Yu and Dayan tie it to "
            "unexpected uncertainty, where the right response is to abandon the "
            "current model rather than update it."
        ),
        citation="Doya 2002; Yu & Dayan, Neuron 46:681 (2005)",
        consumers=("core.curiosity_engine", "core.agency"),
        sign=-1,
    ),
    Modulator.SEROTONIN: ModulatorRole(
        modulator=Modulator.SEROTONIN,
        parameter="discount_factor",
        description=(
            "How far ahead the system is willing to care about. A rise makes a "
            "distant outcome weigh more against an immediate one."
        ),
        citation="Doya 2002; Miyazaki et al., Curr Biol 24:2033 (2014)",
        consumers=("core.planning", "core.conation"),
    ),
}


@dataclass
class RegionSensitivity:
    """How much one region's activity moves with one modulator."""

    region: str
    modulator: Modulator
    slope: float
    intercept: float
    r_squared: float
    samples: int
    evidence: Evidence

    def as_json(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "modulator": str(self.modulator),
            "slope": round(self.slope, 5),
            "r_squared": round(self.r_squared, 4),
            "samples": self.samples,
            "evidence": str(self.evidence),
        }


def _regress(levels: Sequence[float], responses: Sequence[float]) -> tuple[float, float, float]:
    """Ordinary least squares with an R-squared, in plain arithmetic."""
    n = len(levels)
    if n < 3:
        return 0.0, 0.0, 0.0
    mean_x = sum(levels) / n
    mean_y = sum(responses) / n
    sxx = sum((x - mean_x) ** 2 for x in levels)
    if sxx <= 0:
        return 0.0, mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(levels, responses, strict=True))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residual = sum(
        (y - (slope * x + intercept)) ** 2 for x, y in zip(levels, responses, strict=True)
    )
    total = sum((y - mean_y) ** 2 for y in responses)
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    return slope, intercept, r_squared


def fit_observational(
    region: str,
    modulator: Modulator,
    levels: Sequence[float],
    responses: Sequence[float],
) -> RegionSensitivity:
    """Fit a sensitivity from ordinary operation. This is a correlation.

    Anything that moves both the modulator and the region will show up here as
    a slope, which is why the grade is fixed at observational and no amount of
    fit quality raises it.
    """
    slope, intercept, r_squared = _regress(levels, responses)
    return RegionSensitivity(
        region=region,
        modulator=modulator,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        samples=len(levels),
        evidence=Evidence.OBSERVATIONAL,
    )


def fit_interventional(
    region: str,
    modulator: Modulator,
    trials: Sequence[tuple[float, float]],
    *,
    min_levels: int = 3,
) -> RegionSensitivity:
    """Fit a sensitivity from trials where the level was set rather than watched.

    ``trials`` are ``(level, response)`` pairs from runs where the level was
    assigned. Fewer than ``min_levels`` distinct assigned levels cannot separate
    a dose response from a trend, so the grade drops back to observational and
    the caller is told rather than left with a causal-looking number.
    """
    levels = [level for level, _ in trials]
    responses = [response for _, response in trials]
    slope, intercept, r_squared = _regress(levels, responses)
    distinct = len({round(level, 6) for level in levels})
    grade = Evidence.INTERVENTIONAL if distinct >= min_levels else Evidence.OBSERVATIONAL
    if grade is Evidence.OBSERVATIONAL and trials:
        logger.info(
            "%s x %s: %d assigned level(s) is not a dose response; graded observational",
            region,
            modulator,
            distinct,
        )
    return RegionSensitivity(
        region=region,
        modulator=modulator,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        samples=len(trials),
        evidence=grade,
    )


@dataclass
class ReceptorField:
    """Per-region sensitivity to each modulator, with the grade of each entry.

    A field with nothing fitted returns a uniform response, which is what the
    system had before this module existed. Uniform is the honest default: it
    says the spatial structure has not been measured rather than guessing at it.
    """

    sensitivities: dict[tuple[str, Modulator], RegionSensitivity] = field(
        default_factory=dict
    )
    baseline: float = 0.5

    def set(self, sensitivity: RegionSensitivity) -> None:
        self.sensitivities[(sensitivity.region, sensitivity.modulator)] = sensitivity

    def gain(self, region: str, modulator: Modulator, level: float) -> float:
        """Multiplier on this modulator's effect in this region.

        One is no effect. The slope is applied against the level's departure
        from baseline, and the result is clamped so a badly conditioned fit
        cannot silently switch a consumer off or drive it to a large multiple.
        """
        entry = self.sensitivities.get((region, modulator))
        if entry is None or entry.r_squared <= 0.05:
            return 1.0
        raw = 1.0 + entry.slope * (level - self.baseline)
        return max(0.1, min(4.0, raw))

    def claim(self, region: str, modulator: Modulator) -> str:
        """The strongest sentence the evidence for this pair supports."""
        entry = self.sensitivities.get((region, modulator))
        if entry is None or entry.evidence is Evidence.NONE:
            return f"{modulator} in {region}: not measured"
        if entry.evidence is Evidence.OBSERVATIONAL:
            return (
                f"{modulator} level and {region} activity move together "
                f"(slope {entry.slope:.3f}, R^2 {entry.r_squared:.2f}); "
                "no intervention, so no causal claim"
            )
        return (
            f"setting {modulator} changes {region} activity "
            f"(slope {entry.slope:.3f}, R^2 {entry.r_squared:.2f}, "
            f"{entry.samples} assigned trials)"
        )

    def coverage(self) -> dict[str, Any]:
        graded: dict[str, int] = {str(grade): 0 for grade in Evidence}
        for entry in self.sensitivities.values():
            graded[str(entry.evidence)] += 1
        regions = {region for region, _ in self.sensitivities}
        return {
            "pairs": len(self.sensitivities),
            "regions": len(regions),
            "by_evidence": graded,
            "roles": {str(m): role.parameter for m, role in MODULATOR_ROLES.items()},
        }

    def parameter_multipliers(
        self,
        levels: Mapping[Modulator, float],
        region: str,
    ) -> dict[str, float]:
        """What each controlled parameter should be scaled by, in this region."""
        out: dict[str, float] = {}
        for modulator, role in MODULATOR_ROLES.items():
            level = float(levels.get(modulator, self.baseline))
            gain = self.gain(region, modulator, level)
            out[role.parameter] = gain if role.sign > 0 else 1.0 / max(1e-3, gain)
        return out
