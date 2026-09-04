"""Item 41 — hatred gives nothing and takes everything.

Separable from item 34. That one is about the dynamics — hatred
reproduces itself. This is the accounting claim, and it can be measured
rather than asserted.

*Gives nothing.* Unlike anger, hatred has no repair condition and
therefore no terminating success state. Anger can be satisfied: the
other changes their behaviour and it ends. Hatred's object is the other
as a kind, so no act of theirs can discharge it. A drive with no
reachable satisfaction condition produces unbounded expenditure for zero
return, and that is a structural property, not a moral one.

*Takes everything.* Three costs, all measurable. It consumes attention,
which is the scarcest resource any agent has. It corrupts inference,
because a strong prior about its object filters the evidence — so the
hater's model of the person they most need to predict is their worst
model. And it forecloses options: the action set available to someone
committed to another's harm is strictly smaller.

So this is an auditable ledger. It tracks any disposition that targets
an agent rather than an act, has no stateable satisfaction condition,
and is consuming budget, and it reports what that has cost against what
it has returned. The claim becomes a measurement the system takes on
itself, and it can come out the other way.
"""

from __future__ import annotations

from core.interiority.effects import AttentionBias, Effects, SomaticMarker
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)


@register
class HatredLedger(Faculty):
    id = "f41_hatred_ledger"
    number = 41
    question = "Hatred gives nothing and takes everything"
    mechanism = (
        "Audits any agent-directed disposition with no satisfaction condition: "
        "attention consumed, inference corrupted, options foreclosed, against "
        "what it returned"
    )
    requires = ()
    optional = ("agency_other", "relevance")
    counterfactuals = (
        Counterfactual(
            "no_standing_disposition",
            {"agency_other": 0.0},
            Direction.COLLAPSES,
            "The audit is of a held disposition, not of an event. With none "
            "held there is nothing to charge, and a ledger that charges an "
            "agent holding nothing is the Grok defect: 0.04 of tax on zero "
            "hatred, because a policy default leaked into the sum.",
        ),
    )
    null = NullSpec(values={"agency_other": 0.0})

    def falsifier(self) -> str:
        return (
            "It reports a cost for an agent holding no such disposition. It is "
            "an audit; auditing nothing must produce nothing, and any floor is "
            "a number that arrived from somewhere other than the measurement."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        # A disposition qualifies only if it targets an agent rather than an
        # act and has no stateable satisfaction condition. Both are read
        # from the ledger; neither is asserted.
        held = ctx.interior.get("agent_directed_dispositions")
        if not isinstance(held, dict) or not held:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no agent-directed disposition is held; there is nothing to audit",
            )

        unsatisfiable = {
            k: v
            for k, v in held.items()
            if isinstance(v, dict) and not v.get("satisfaction_condition")
        }
        if not unsatisfiable:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "every held disposition has a satisfaction condition; these "
                    "are answerable grievances, and they terminate"
                ),
                receipt={"held": list(held)},
            )

        attention_spent = sum(
            float(v.get("attention_share", 0.0)) for v in unsatisfiable.values()
        )
        # Inference corruption: how far the model of the target has drifted
        # from the model of comparable others.
        corruption = max(
            (float(v.get("model_divergence", 0.0)) for v in unsatisfiable.values()),
            default=0.0,
        )
        foreclosed = sum(
            int(v.get("actions_foreclosed", 0)) for v in unsatisfiable.values()
        )
        returned = sum(float(v.get("returned", 0.0)) for v in unsatisfiable.values())

        taken = min(
            1.0, attention_spent + 0.3 * corruption + 0.05 * foreclosed
        )
        intensity = taken

        effects = Effects(
            attention=tuple(
                AttentionBias(
                    target=f"disposition:{name}",
                    weight=-float(v.get("attention_share", 0.0)),
                    reason="unsatisfiable disposition consuming attention for no return",
                )
                for name, v in list(unsatisfiable.items())[:4]
            ),
            somatic=(
                SomaticMarker(
                    option="maintain_the_disposition",
                    bias=-(taken - returned),
                    reason="audited cost exceeds audited return",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="",
            effects=effects,
            receipt={
                "unsatisfiable_dispositions": list(unsatisfiable),
                "attention_consumed": attention_spent,
                "model_corruption": corruption,
                "actions_foreclosed": foreclosed,
                "returned": returned,
                "net": returned - taken,
            },
        )
