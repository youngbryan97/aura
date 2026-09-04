"""Necessity is the bottom rung, and most claims stop there.

A lesion result — turn the organ off and the capability drops — is the
evidence almost every consciousness architecture offers, and on its own it is
weak. It shows the component was being used. It does not show the component
produces the state, that only this component does, that more of it gives more
of the state, or that the state comes back.

So a causal claim in this package is graded on five rungs, and the grade is
the LOWEST rung that has been climbed rather than the highest:

    necessity     do(M = 0) and the effect falls
    sufficiency   do(M = m*) with the ordinary cause ABSENT and the effect
                  appears anyway. This is the rung that turns C -> M into
                  M -> C, and it is the one nobody runs
    specificity   lesioning a matched OTHER component does not do it. Without
                  this, "the system degrades when you break part of it" is
                  the whole finding
    dose_response the effect tracks the size of the intervention. A switch is
                  weaker evidence than a dial, because a dial is much harder
                  to produce by accident
    reversibility restore the component and the effect returns. Distinguishes
                  a mechanism from damage

The five are separate because each kills a different alternative explanation,
and a claim that reports "causally validated" while having climbed only the
first has picked the flattering summary of its own evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Rung", "Arm", "CausalClaim", "grade"]


class Rung(StrEnum):
    """How far up the ladder a claim has actually climbed."""

    NONE = "none"
    NECESSITY = "necessity"
    SUFFICIENCY = "sufficiency"
    SPECIFICITY = "specificity"
    DOSE_RESPONSE = "dose_response"
    REVERSIBILITY = "reversibility"


#: The order matters: the grade is the highest rung such that every rung
#: below it also held. A claim with sufficiency and reversibility but no
#: specificity grades at SUFFICIENCY, because the gap is what a reader needs
#: to know.
LADDER: tuple[Rung, ...] = (
    Rung.NECESSITY,
    Rung.SUFFICIENCY,
    Rung.SPECIFICITY,
    Rung.DOSE_RESPONSE,
    Rung.REVERSIBILITY,
)

RUNG_MEANING: dict[Rung, str] = {
    Rung.NONE: "nothing has been established about this mechanism",
    Rung.NECESSITY: (
        "the component was being used. Not that it produces the state, not "
        "that only it does, and not that more of it gives more"
    ),
    Rung.SUFFICIENCY: (
        "inducing the pattern produces the effect with the ordinary cause "
        "absent, so the arrow runs from the mechanism to the state"
    ),
    Rung.SPECIFICITY: (
        "a matched other component does not do it, so this is not the general "
        "consequence of breaking something"
    ),
    Rung.DOSE_RESPONSE: (
        "the effect tracks the size of the intervention, which is much harder "
        "to get by accident than a switch"
    ),
    Rung.REVERSIBILITY: (
        "restoring the component restores the effect, so this is a mechanism "
        "rather than damage"
    ),
}


@dataclass(frozen=True)
class Arm:
    """One measured condition of a causal experiment."""

    name: str
    #: What the intervention was, in do() terms.
    intervention: str
    #: The pre-registered measure, same in every arm.
    measure: str
    value: float
    #: Values of the same measure under the matched null for THIS arm. An arm
    #: with no null cannot say whether its number is unusual.
    nulls: tuple[float, ...] = ()

    @property
    def has_null(self) -> bool:
        return len(self.nulls) >= 3

    @property
    def null_band(self) -> tuple[float, float]:
        """The range the null occupies, widened for a deterministic one.

        A null with no spread is not a broken null: a deterministic system
        measured under the same conditions gives the same number every time,
        and that number is the truth about what "no effect" looks like. But a
        strict comparison against a point mass can never be satisfied in one
        direction and is always satisfied in the other, so the band carries a
        floor.
        """
        low, high = min(self.nulls), max(self.nulls)
        spread = high - low
        pad = max(spread * 0.5, 1e-6)
        return low - pad, high + pad

    def exceeds_null(self, *, direction: str = "below") -> bool:
        """Whether this arm's value sits outside its own null band."""
        if not self.has_null:
            return False
        low, high = self.null_band
        return self.value < low if direction == "below" else self.value > high

    def inside_null(self) -> bool:
        """Whether this arm is indistinguishable from no effect.

        What a successful lesion looks like. The effect does not go NEGATIVE
        when the mechanism is removed; it goes away.
        """
        if not self.has_null:
            return False
        low, high = self.null_band
        return low <= self.value <= high


@dataclass
class CausalClaim:
    """A mechanism, the arms run against it, and the rung that earns."""

    mechanism: str
    effect: str
    baseline: Arm | None = None
    #: do(M = 0)
    lesion: Arm | None = None
    #: do(M = m*) with the ordinary cause absent
    induction: Arm | None = None
    #: do(other = 0), a matched component
    matched_control: Arm | None = None
    #: do(M = m) at several m, ordered by m
    dose: tuple[Arm, ...] = ()
    #: M restored after lesion
    restored: Arm | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        rung, unmet = grade(self)
        return {
            "mechanism": self.mechanism,
            "effect": self.effect,
            "rung": str(rung),
            "means": RUNG_MEANING[rung],
            "first_unmet": str(unmet) if unmet else "",
            "arms_run": [
                arm.name
                for arm in (
                    self.baseline,
                    self.lesion,
                    self.induction,
                    self.matched_control,
                    self.restored,
                    *self.dose,
                )
                if arm is not None
            ],
            "notes": list(self.notes),
        }


def _necessity_holds(claim: CausalClaim) -> bool:
    """There was an effect, and lesioning the mechanism removed it.

    The lesioned arm must land back INSIDE the null, not below it. Requiring
    it to go below was a bug: a lesion that takes the effect to exactly the
    no-effect level is the strongest possible result and it graded as a
    failure, because "below the null" cannot be satisfied against a
    deterministic null sitting at zero.
    """
    if claim.baseline is None or claim.lesion is None:
        return False
    if not claim.lesion.has_null or not claim.baseline.has_null:
        return False
    effect_existed = claim.baseline.exceeds_null(direction="above")
    effect_removed = claim.lesion.inside_null()
    return effect_existed and effect_removed and claim.lesion.value < claim.baseline.value


def _sufficiency_holds(claim: CausalClaim) -> bool:
    """Induced pattern, ordinary cause absent, effect appears anyway."""
    if claim.induction is None or not claim.induction.has_null:
        return False
    return claim.induction.exceeds_null(direction="above")


def _specificity_holds(claim: CausalClaim) -> bool:
    """Breaking a matched OTHER component must not reproduce the effect."""
    if claim.matched_control is None or claim.lesion is None:
        return False
    if claim.baseline is None:
        return False
    control_drop = claim.baseline.value - claim.matched_control.value
    lesion_drop = claim.baseline.value - claim.lesion.value
    if lesion_drop <= 0.0:
        return False
    # The matched control must produce a MUCH smaller drop. Half is the line:
    # a control that does most of what the lesion does has shown the effect is
    # general damage.
    return control_drop < 0.5 * lesion_drop


def _dose_response_holds(claim: CausalClaim) -> bool:
    """The effect must track the size of the intervention monotonically."""
    if len(claim.dose) < 3:
        return False
    values = [arm.value for arm in claim.dose]
    steps = list(zip(values, values[1:], strict=False))
    rising = all(b >= a for a, b in steps)
    falling = all(b <= a for a, b in steps)
    if not (rising or falling):
        return False
    # A flat line is monotone by both definitions and shows nothing.
    return abs(values[-1] - values[0]) > 1e-9


def _reversibility_holds(claim: CausalClaim) -> bool:
    if claim.restored is None or claim.baseline is None or claim.lesion is None:
        return False
    recovered = claim.restored.value - claim.lesion.value
    lost = claim.baseline.value - claim.lesion.value
    if lost <= 0.0:
        return False
    return recovered >= 0.5 * lost


_CHECKS = {
    Rung.NECESSITY: _necessity_holds,
    Rung.SUFFICIENCY: _sufficiency_holds,
    Rung.SPECIFICITY: _specificity_holds,
    Rung.DOSE_RESPONSE: _dose_response_holds,
    Rung.REVERSIBILITY: _reversibility_holds,
}


def grade(claim: CausalClaim) -> tuple[Rung, Rung | None]:
    """The highest rung reached with every rung below it also met.

    Returns the grade and the first rung that failed, because the gap is the
    actionable part: a claim graded NECESSITY whose next unmet rung is
    SUFFICIENCY needs an induction arm, and saying so is more useful than a
    score.
    """
    reached = Rung.NONE
    for rung in LADDER:
        if _CHECKS[rung](claim):
            reached = rung
            continue
        return reached, rung
    return reached, None


def log_likelihood_ratio(claim: CausalClaim) -> float:
    """How far this claim moves the odds between costume and load-bearing.

    Each rung is worth a factor, and they are not equal. Necessity is cheap:
    almost any component of a working system produces it. Sufficiency is
    expensive, because a costume has no reason to generate the effect when the
    ordinary cause is absent.
    """
    weights = {
        Rung.NONE: 0.0,
        Rung.NECESSITY: math.log(2.0),
        Rung.SUFFICIENCY: math.log(8.0),
        Rung.SPECIFICITY: math.log(16.0),
        Rung.DOSE_RESPONSE: math.log(32.0),
        Rung.REVERSIBILITY: math.log(64.0),
    }
    rung, _unmet = grade(claim)
    return weights[rung]
