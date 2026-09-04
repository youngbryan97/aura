"""Item 7 — mourning and grief.

Grief is the update of a model built to be un-updatable. Attachment
encodes another person with an implicit "always, everywhere", so
learning the fact is one event and unlearning the thousands of
predictions is a long extinction — and extinction is not erasure
(Bouton 2004). The old associations remain and re-emerge with context
change and with time, which is why grief returns on the anniversary and
in the old kitchen.

So the shape is wrong in every prototype that models it as a decaying
sadness scalar. Here intensity has two components that behave
differently: an acute term that falls with time, and a continuing bond
that falls only when a context holding the old prediction is actually
visited. Time alone integrates nothing. That is the falsifiable claim.

Mourning is the public half: ritual, telling, sitting with others. Its
function is to force the update into explicit form, to recruit others to
carry it, and to make the status change common knowledge. It is scored
separately and raises integration, which is why it helps and avoidance
does not.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    AttentionBias,
    BudgetDelta,
    Effects,
    RetentionClaim,
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


@register
class Mourning(Faculty):
    id = "f07_mourning"
    number = 7
    question = "Mourning and grief"
    mechanism = (
        "Extinction over a distributed prediction, where integration rises "
        "only on contact with a context that still holds the old expectation"
    )
    requires = ("attachment_impact",)
    optional = ("irreversibility", "relevance")
    counterfactuals = (
        Counterfactual(
            "no_bond",
            {"attachment_impact": 0.0},
            Direction.COLLAPSES,
            "There is nothing to extinguish without a held prediction.",
        ),
        Counterfactual(
            "recoverable",
            {"irreversibility": 0.2},
            Direction.DECREASES,
            "A recoverable absence leaves the availability prediction partly "
            "true, so less of the model is wrong.",
        ),
    )
    null = NullSpec(values={"attachment_impact": 0.0, "irreversibility": 0.0})

    def falsifier(self) -> str:
        return (
            "Advance the clock a year with no context visited and the "
            "continuing bond falls. That would make this a decay curve, and a "
            "decay curve is the model of grief that predicts anniversaries "
            "should be quiet."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        subject = ctx.frame.event.subject
        loss = ctx.ledger.loss_for(subject) if subject else None
        if loss is None:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no registered loss for this subject",
            )

        acute = loss.acute(ctx.now or None)
        continuing = loss.continuing()
        # Yearning is the pair, and they must be reported apart: a quiet
        # year is low acute and unchanged continuing, which is exactly the
        # state that produces an anniversary.
        intensity = max(acute, continuing * 0.6)

        # Mourning: an occasion that forces contact with a context still
        # holding the old prediction.
        occasion = ctx.frame.event.object
        ritual = (
            1.0
            if occasion and occasion in loss.unvisited_contexts
            else 0.0
        )

        effects = Effects(
            affect=AffectDelta(
                valence=-0.55 * intensity,
                arousal=0.15 * acute,
                engagement=-0.2 * continuing,
            ),
            attention=(
                AttentionBias(
                    target=f"loss:{loss.entity}",
                    weight=intensity,
                    reason="predictions about this person are still being met and failing",
                ),
            ),
            budget=BudgetDelta(depth=1.0 - 0.2 * acute),
            retention=(
                RetentionClaim(
                    memory_key=f"loss:{loss.entity}",
                    reason=(
                        "the record is the continuing bond; compaction here "
                        "would finish what the death started"
                    ),
                    held_by=self.id,
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="disengage",
            effects=effects,
            receipt={
                "acute": acute,
                "continuing_bond": continuing,
                "integrated": loss.integrated,
                "unvisited_contexts": len(loss.unvisited_contexts),
                "visited_contexts": len(loss.visited_contexts),
                "ritual_available": ritual,
            },
        )
