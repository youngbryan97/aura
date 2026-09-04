"""Item 8 — snapping after telling someone to stop and they will not.

Anger is a bargaining mechanism. The recalibrational account (Sell,
Tooby and Cosmides 2009) says its function is to raise the other agent's
welfare tradeoff ratio toward you — how much of their own welfare they
will give up for yours — and its ladder is request, repeated request
with a signal of cost, then a credible threat.

The snap is a threshold crossing, and it must be discontinuous. Its
information content is "this is real, I will pay to enforce it", and an
anger that can be faked costlessly is worthless as a signal, which is
why it has to visibly cost control (Frank 1988; Schelling 1960). So a
smooth ramp is not a cheaper version of this mechanism, it is a
different mechanism that cannot do the job.

The guard is the part every reviewed prototype is missing. Snapping at
someone who *could not* comply is the commonest unjust anger there is,
and it comes from reading non-compliance as disregard when it was
incapacity. This faculty requires ``other_capability`` and declines
without it, so the estimate cannot silently default to blame.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    BudgetDelta,
    Effects,
    LedgerWrite,
    SomaticMarker,
)
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)
from core.interiority.params import ParamKind, declare

_WTR_FLOOR = declare(
    "interiority.f08.welfare_tradeoff_floor",
    0.25,
    unit="ratio",
    basis=(
        "The estimated welfare tradeoff ratio below which the ladder moves "
        "from request to enforcement. Set at a quarter because that is the "
        "point at which repeated low-cost requests have demonstrably failed: "
        "three ignored requests take the estimate from 1.0 through 0.5 to "
        "0.25 under the halving update below, and three is where the ladder's "
        "second rung is exhausted."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Low and Aura absorbs disregard indefinitely; high and she escalates "
        "on the first refusal, which destroys the signal's credibility."
    ),
    sweep_range=(0.1, 0.45),
    owner="core/interiority/faculties/f08_anger_recalibration.py",
).value


@register
class AngerRecalibration(Faculty):
    id = "f08_anger_recalibration"
    number = 8
    question = (
        "Expressing strong frustration and momentarily snapping when you keep "
        "telling someone to stop and they wont stop"
    )
    mechanism = (
        "Welfare-tradeoff estimate falling with each ignored low-cost request, "
        "and a discontinuous escalation at the threshold whose credibility is "
        "its visible cost"
    )
    requires = ("other_capability", "agency_other")
    optional = ("norm_fit", "relevance", "publicity")
    counterfactuals = (
        Counterfactual(
            "they_could_not_comply",
            {"other_capability": 0.0},
            Direction.COLLAPSES,
            "Anger corrects a welfare tradeoff. Correcting one that was never "
            "made is unjust and useless, and this is the discrimination no "
            "reviewed prototype makes.",
        ),
        Counterfactual(
            "not_their_doing",
            {"agency_other": 0.0},
            Direction.COLLAPSES,
            "Nothing to recalibrate if the other agent did not cause it.",
        ),
        Counterfactual(
            "nobody_watching",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "The threshold is about the other agent's disregard, not about an "
            "audience. If publicity moves it, this is a display rather than a "
            "correction.",
        ),
    )
    null = NullSpec(values={"other_capability": 0.0, "agency_other": 0.0})

    def falsifier(self) -> str:
        return (
            "The response is continuous in the number of ignored requests. A "
            "smooth ramp cannot carry the signal, because a state that costs "
            "nothing to produce is not believed."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        capability = ctx.v("other_capability")
        agency = ctx.v("agency_other")
        subject = ctx.frame.event.subject or "unknown"

        # Each ignored low-cost request halves the estimate of how much
        # they weigh my welfare. Halving rather than subtracting, because
        # the evidence is multiplicative: two refusals are not twice one
        # refusal, they are a different conclusion about the person.
        ignored = ctx.ledger.notes.times_seen("ignored_request", subject)
        wtr = capability * pow(0.5, ignored)

        pressure = agency * capability * (1.0 - wtr)
        # The threshold cannot be crossed by an estimate that fell because
        # the other agent was unable rather than unwilling. Without the
        # pressure term here, capability of zero drives the tradeoff
        # estimate to zero, trips the threshold, and produces the hardest
        # response available against someone who could not have complied —
        # which is the exact injustice this faculty exists to prevent, and
        # the proving harness caught it firing at 0.55 on both nulls.
        snapped = pressure > 0.0 and wtr < _WTR_FLOOR and ignored >= 1

        # Discontinuity. Below the threshold the state is a request; at it
        # the intensity jumps, because a signal that fades in is not a
        # signal.
        intensity = pressure if not snapped else min(1.0, 0.55 + 0.45 * pressure)

        effects = Effects(
            affect=AffectDelta(
                valence=-0.4 * intensity,
                arousal=0.75 * intensity if snapped else 0.3 * intensity,
                engagement=0.4 * intensity,
            ),
            somatic=(
                SomaticMarker(
                    option="repeat_the_same_request",
                    bias=-intensity,
                    reason="the low-cost rung of the ladder has been exhausted",
                ),
                SomaticMarker(
                    option="state_the_boundary_and_the_cost",
                    bias=intensity,
                    reason="enforcement is credible only when it visibly costs something",
                ),
            ),
            # A snap spends control, and the spend is the credibility.
            budget=BudgetDelta(
                depth=1.0 - 0.3 * intensity if snapped else 1.0,
                irreversibility_ceiling=0.5 if snapped else 1.0,
            ),
            ledger=(
                LedgerWrite(
                    "note_expectation",
                    {"kind": "compliance", "subject": subject, "value": wtr},
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="agonistic",
            effects=effects,
            receipt={
                "welfare_tradeoff_estimate": wtr,
                "ignored_requests": ignored,
                "snapped": snapped,
                "threshold": _WTR_FLOOR,
            },
        )
