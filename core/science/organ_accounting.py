"""core/science/organ_accounting.py — what each organ is actually worth.

Aura has a strong lesion registry and an influence campaign, and what they
establish is that an organ has an effect. That is the CAUSAL rung. The rung
above asks something the lesion cannot answer on its own: **was the effect
worth what it cost, and would something cheaper have done the same?**

Three arms per organ, and the third is the one usually skipped:

* ``full`` — the system as it runs.
* ``lesioned`` — the organ removed.
* ``compute_matched`` — the organ replaced by something that costs the same
  and does nothing useful: a random policy, a delay, a no-op that burns the
  same tokens. Without it, every expensive organ looks load-bearing, because
  removing it removes its compute too.
* ``noise_control`` — the organ replaced by a version whose internal state is
  scrambled. This separates "the organ's computation matters" from "the
  organ's presence in the loop matters".

Synergy
-------
:func:`synergy` is CogPrime's cognitive-synergy idea made into a number:

    S(A,B) = perf(A+B) - max(perf(A), perf(B))

Positive means the pair does something neither does alone. Zero means one is
carrying the other. Negative means they interfere, which happens more than
anyone expects and is invisible unless somebody computes this.

Multiple testing
----------------
Testing thirty organs at p<0.05 finds one and a half spurious effects by
construction. :func:`hochberg` applies the step-up correction, and the report
says how many effects survive it. A single p of 0.04 found by looking at thirty
organs does not survive; thirty organs all at 0.04 do, because that joint result
is unlikely and the step-up rule is right to reject the family. The correction
is named in the report rather than applied silently, so a reader can see which
of those two situations they are in.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ArmKind",
    "OrganMeasurement",
    "OrganVerdict",
    "synergy",
    "hochberg",
    "OrganAccounting",
    "get_organ_accounting",
    "reset_organ_accounting_for_test",
]


class ArmKind(StrEnum):
    FULL = "full"
    LESIONED = "lesioned"
    #: Same cost, no useful computation.
    COMPUTE_MATCHED = "compute_matched"
    #: Same organ, scrambled internal state.
    NOISE_CONTROL = "noise_control"


@dataclass(frozen=True, slots=True)
class OrganMeasurement:
    """One organ, measured on every arm that was run."""

    organ: str
    scores: Mapping[str, float]
    n: int
    p_value: float | None = None
    cost_seconds: Mapping[str, float] = field(default_factory=dict)

    def arm(self, kind: ArmKind) -> float | None:
        return self.scores.get(kind.value)

    @property
    def lesion_effect(self) -> float | None:
        full, lesioned = self.arm(ArmKind.FULL), self.arm(ArmKind.LESIONED)
        return None if full is None or lesioned is None else full - lesioned

    @property
    def effect_over_matched_compute(self) -> float | None:
        """How much of the lesion effect survives giving the compute back."""
        full, matched = self.arm(ArmKind.FULL), self.arm(ArmKind.COMPUTE_MATCHED)
        return None if full is None or matched is None else full - matched

    @property
    def effect_over_noise(self) -> float | None:
        """How much survives scrambling the organ's own state."""
        full, noise = self.arm(ArmKind.FULL), self.arm(ArmKind.NOISE_CONTROL)
        return None if full is None or noise is None else full - noise


@dataclass(frozen=True, slots=True)
class OrganVerdict:
    """What this organ has earned."""

    organ: str
    classification: str
    reason: str
    lesion_effect: float | None
    matched_effect: float | None
    noise_effect: float | None
    survives_correction: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "classification": self.classification,
            "reason": self.reason,
            "lesion_effect": self.lesion_effect,
            "effect_over_matched_compute": self.matched_effect,
            "effect_over_noise": self.noise_effect,
            "survives_multiple_testing": self.survives_correction,
        }


def synergy(pair_score: float, a_alone: float, b_alone: float) -> float:
    """What the pair does that neither does alone. Negative means interference."""
    return pair_score - max(a_alone, b_alone)


def hochberg(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Hochberg step-up. Returns, per input, whether it survives correction.

    Step-up rather than Bonferroni because thirty organs at Bonferroni rejects
    almost everything real; step-up controls the same family-wise error rate
    and keeps more of it.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1], reverse=True)
    survives = [False] * n
    for rank, (index, p) in enumerate(indexed):
        if p <= alpha / (rank + 1):
            for later_index, _ in indexed[rank:]:
                survives[later_index] = True
            break
    return survives


#: An effect smaller than this is not worth an organ, whatever its p-value.
#: Chosen to be visible in the metric a caller is using rather than derived:
#: a caller working in a different unit passes its own.
DEFAULT_MINIMUM_EFFECT = 0.01


class OrganAccounting:
    """Every organ, its arms, and what it is entitled to be kept for."""

    def __init__(self, *, minimum_effect: float = DEFAULT_MINIMUM_EFFECT) -> None:
        self._lock = checked_lock("core.science.organ_accounting.OrganAccounting", reentrant=True)
        self._measurements: dict[str, OrganMeasurement] = {}
        self._synergies: dict[tuple[str, str], float] = {}
        self._minimum = float(minimum_effect)

    def measure(self, measurement: OrganMeasurement) -> OrganMeasurement:
        with self._lock:
            self._measurements[measurement.organ] = measurement
            return measurement

    def record_synergy(self, a: str, b: str, *, pair: float, a_alone: float, b_alone: float) -> float:
        value = synergy(pair, a_alone, b_alone)
        with self._lock:
            self._synergies[tuple(sorted((a, b)))] = value
        return value

    def verdicts(self) -> list[OrganVerdict]:
        """Classify every organ. The classification is the deliverable."""
        with self._lock:
            measurements = list(self._measurements.values())
        with_p = [m for m in measurements if m.p_value is not None]
        survival = dict(
            zip(
                (m.organ for m in with_p),
                hochberg([m.p_value for m in with_p]),
                strict=True,
            )
        )

        out: list[OrganVerdict] = []
        for m in measurements:
            survives = survival.get(m.organ)
            lesion, matched, noise = m.lesion_effect, m.effect_over_matched_compute, m.effect_over_noise
            if lesion is None:
                classification, reason = "unmeasured", "no lesion arm was run"
            elif lesion < self._minimum:
                classification, reason = (
                    "non_load_bearing",
                    f"removing it costs {lesion:+.4g}, below the {self._minimum} floor",
                )
            elif matched is None:
                classification, reason = (
                    "causal_unpriced",
                    "the lesion has an effect and no compute-matched arm was run, so the "
                    "effect may be the compute",
                )
            elif matched < self._minimum:
                classification, reason = (
                    "compute_not_computation",
                    f"the effect vanishes ({matched:+.4g}) when the compute is given back",
                )
            elif noise is not None and noise < self._minimum:
                classification, reason = (
                    "presence_not_content",
                    f"scrambling its state costs {noise:+.4g}; being in the loop is what matters",
                )
            elif survives is False:
                classification, reason = (
                    "not_after_correction",
                    "the effect does not survive multiple-testing correction",
                )
            else:
                classification, reason = (
                    "load_bearing",
                    f"survives lesion ({lesion:+.4g}), matched compute ({matched:+.4g})"
                    + (f" and noise ({noise:+.4g})" if noise is not None else ""),
                )
            out.append(
                OrganVerdict(
                    organ=m.organ, classification=classification, reason=reason,
                    lesion_effect=lesion, matched_effect=matched, noise_effect=noise,
                    survives_correction=survives,
                )
            )
        return sorted(out, key=lambda v: v.organ)

    def report(self) -> dict[str, Any]:
        verdicts = self.verdicts()
        by_class: dict[str, list[str]] = {}
        for verdict in verdicts:
            by_class.setdefault(verdict.classification, []).append(verdict.organ)
        with self._lock:
            synergies = dict(self._synergies)
        return {
            "organs": len(verdicts),
            "by_classification": {k: sorted(v) for k, v in sorted(by_class.items())},
            "load_bearing": sorted(by_class.get("load_bearing", [])),
            "candidates_for_removal": sorted(
                by_class.get("non_load_bearing", []) + by_class.get("compute_not_computation", [])
            ),
            "synergies": {f"{a}+{b}": v for (a, b), v in sorted(synergies.items())},
            "interfering_pairs": sorted(
                f"{a}+{b}" for (a, b), v in synergies.items() if v < 0
            ),
            "verdicts": [v.to_dict() for v in verdicts],
        }


_lock = checked_lock("core.science.organ_accounting.singleton")
_accounting: OrganAccounting | None = None


def get_organ_accounting() -> OrganAccounting:
    global _accounting
    with _lock:
        if _accounting is None:
            _accounting = OrganAccounting()
        return _accounting


def reset_organ_accounting_for_test(**kwargs: Any) -> OrganAccounting:
    global _accounting
    with _lock:
        _accounting = OrganAccounting(**kwargs)
        return _accounting
