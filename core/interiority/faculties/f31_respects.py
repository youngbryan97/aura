"""Item 31 — paying respects at a grave.

An action whose entire effect is on the actor's own model of a
relationship, addressed to someone who cannot receive it. Three
functions, and each is real.

Psychologically it schedules the update that grief requires. Extinction
learning does not happen without cue exposure, so going to the place
forces contact with the prediction that keeps failing — which is why
avoidance prolongs grief and why item 7's integration rises here and
nowhere else.

Socially it is a costly signal of continued commitment to a relationship
that pays nothing back, which is the strongest available evidence about
a person's dispositions, and everyone reads it that way.

Morally it maintains the practice of treating people as mattering after
they can no longer enforce it. The obligation is asymmetric and
unenforceable, and honouring it anyway is what a value is.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, LedgerWrite
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
class PayingRespects(Faculty):
    id = "f31_respects"
    number = 31
    question = "Paying respects at a grave"
    mechanism = (
        "Deliberate contact with a context that still holds the old "
        "prediction, which is the only thing that raises integration, plus a "
        "costly signal toward a relationship that cannot reciprocate"
    )
    requires = ("attachment_impact",)
    optional = ("publicity", "irreversibility")
    counterfactuals = (
        Counterfactual(
            "no_bond",
            {"attachment_impact": 0.0},
            Direction.COLLAPSES,
            "There is no relationship whose model needs meeting.",
        ),
        Counterfactual(
            "nobody_would_know",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "The act is addressed to someone who cannot receive it and is not "
            "for observers. If an audience changes it, it is a performance.",
        ),
    )
    null = NullSpec(values={"attachment_impact": 0.0})

    def falsifier(self) -> str:
        return (
            "Integration rises without the context actually being met. That "
            "would make the ritual decorative, and it is the one thing in "
            "grief that does work."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        bond = ctx.v("attachment_impact")
        subject = ctx.frame.event.subject
        loss = ctx.ledger.loss_for(subject) if subject else None
        if loss is None:
            return Activation(
                faculty=self.id, intensity=0.0, declined="no loss on record for this subject"
            )

        place = ctx.frame.event.object
        meets_context = bool(place and place in loss.unvisited_contexts)
        intensity = bond * (0.6 + 0.4 * float(meets_context))

        effects = Effects(
            affect=AffectDelta(
                valence=-0.3 * intensity, arousal=-0.2 * intensity, engagement=0.3 * intensity
            ),
            # The cost. Time spent on a relationship that returns nothing is
            # what makes the signal worth anything.
            budget=BudgetDelta(deadline=1.0 + 0.4 * intensity),
            ledger=(
                (
                    LedgerWrite(
                        "visit_context", {"entity": subject, "context": place}
                    ),
                )
                if meets_context
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend",
            effects=effects,
            receipt={
                "context_met": meets_context,
                "context": place,
                "unvisited_remaining": len(loss.unvisited_contexts),
                "integration_before": loss.integrated,
            },
        )
