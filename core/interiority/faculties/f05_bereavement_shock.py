"""Item 5 — grief, frustration and confusion to the point of a physical reaction
on hearing that a friend has died.

The physical part is the whole question. A sentence moves a body because
the attachment system holds a live prediction — this person exists and
can be reached — that has been confirmed thousands of times and now
carries enormous precision. An unambiguous message violating a
high-precision prediction produces a very large error, and a large
sudden error is what recruits autonomic mobilisation: the body prepares
for an action that would fix it, and there is none.

Frustration is the action system finding no admissible move for an error
that size. Confusion is not a bystander: the world model still contains
the person everywhere else — the number in the phone, the reply owed,
the birthday in May — and cannot be edited in one step because a person
is not a node but a distributed set of expectations. That is why this
faculty writes a loss record with the unvisited contexts enumerated, and
why item 7 then takes months rather than decaying.

The magnitude is precision-weighted, not attachment-weighted, and the
distinction is load-bearing. You can be very close to someone whose
death you have been expecting for a year; the bond is large and the
prediction violation is small, and the body reacts far less. Every
reviewed prototype multiplies by attachment alone and cannot represent
that.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    BudgetDelta,
    Effects,
    LedgerWrite,
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
class BereavementShock(Faculty):
    id = "f05_bereavement_shock"
    number = 5
    question = (
        "Grief/frustration/confusion to the point of a physical reaction after "
        "reading/hearing about the death of a friend"
    )
    mechanism = (
        "Precision-weighted violation of a held availability prediction, "
        "mobilising a body for an action that does not exist"
    )
    requires = ("attachment_impact", "certainty", "irreversibility")
    optional = ("power", "expectation_deviation")
    counterfactuals = (
        Counterfactual(
            "it_was_expected",
            {"expectation_deviation": 0.0},
            Direction.DECREASES,
            "The magnitude is prediction error, not bond size. A death long "
            "foreseen violates little however large the attachment, and the "
            "body reacts far less. A model that cannot show this is scoring "
            "closeness and calling it shock.",
        ),
        Counterfactual(
            "no_bond",
            {"attachment_impact": 0.0},
            Direction.COLLAPSES,
            "The prediction that was violated was an attachment's. With no "
            "bond there was no such prediction to break.",
        ),
        Counterfactual(
            "unconfirmed_rumour",
            {"certainty": 0.1},
            Direction.DECREASES,
            "An uncertain report leaves the prediction partly standing, and "
            "the error is proportionally smaller.",
        ),
    )
    null = NullSpec(
        values={"attachment_impact": 0.0, "certainty": 0.0, "irreversibility": 0.0}
    )

    def falsifier(self) -> str:
        return (
            "Two losses with the same attachment and different expectedness "
            "produce the same magnitude. That would show the mechanism is "
            "reading the bond rather than the violated prediction."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        bond = ctx.v("attachment_impact")
        certainty = ctx.v("certainty")
        irreversible = ctx.v("irreversibility")

        subject = ctx.frame.event.subject
        record = ctx.ledger.bond_for(subject) if subject else None
        precision = record.availability_expectation if record else 1.0
        surprise = (
            ctx.check("expectation_deviation").value
            if ctx.check("expectation_deviation").present
            else precision
        )

        # Precision-weighted prediction error. The bond sets what is at
        # stake; precision and surprise set how much of it was overturned
        # in this instant.
        shock = bond * precision * surprise * certainty * irreversible

        # Frustration: an error this size with no admissible action.
        power = ctx.check("power").value if ctx.check("power").present else 0.0
        frustration = shock * (1.0 - power)

        # Confusion: how much of the model still holds the old prediction.
        contexts = 0
        if subject:
            loss = ctx.ledger.loss_for(subject)
            contexts = len(loss.unvisited_contexts) if loss else 0
        confusion = shock * (1.0 - 1.0 / (1.0 + contexts)) if contexts else 0.0

        effects = Effects(
            affect=AffectDelta(
                valence=-0.85 * shock,
                arousal=0.7 * shock,
                engagement=-0.3 * shock,
            ),
            # Mobilised for an action that does not exist: the depth is
            # spent and the authority to commit is withdrawn.
            budget=BudgetDelta(
                depth=1.0 + 0.6 * shock,
                irreversibility_ceiling=max(0.0, 1.0 - shock),
            ),
            ledger=(
                (
                    LedgerWrite(
                        "register_loss",
                        {"entity": subject, "irreversibility": irreversible},
                    ),
                )
                if subject and irreversible > 0.5
                else ()
            ),
            retention=(
                (
                    RetentionClaim(
                        memory_key=f"bond:{subject}",
                        reason="the record of a person who can no longer be met",
                        held_by=self.id,
                    ),
                )
                if subject
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=shock,
            tendency="inhibit",
            effects=effects,
            receipt={
                "precision": precision,
                "surprise": surprise,
                "frustration": frustration,
                "confusion": confusion,
                "unvisited_contexts": contexts,
            },
        )
