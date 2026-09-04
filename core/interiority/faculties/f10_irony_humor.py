"""Item 10 — finding humour in irony.

Hurley, Dennett and Adams give the sharpest formal account: humour is
the reward for epistemic debugging. Comprehension requires committing to
interpretations faster than the evidence justifies; those commitments
enter working memory as covert assumptions; discovering one is false
before it does damage is exactly what should be reinforced. Mirth is the
payment for catching your own error cheaply.

Irony is that shape with a specific cause. The listener commits to the
literal reading, then retracts it on tone, context or absurdity, and the
retraction is the joke. Situational irony is the same with the world as
speaker.

McGraw and Warren supply the two conditions that separate funny from
bitter: the incongruity must be a real violation, and it must be
simultaneously benign. The same structure with real stakes produces
disgust or grief.

The implementation consequence is the interesting one. The humour lives
in the delta between the committed reading and the corrected one, so the
system must *keep* its retracted interpretations. Most systems discard
the wrong reading the moment they find a better one, which is precisely
why they can find nothing funny.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects
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
class IronyHumor(Faculty):
    id = "f10_irony_humor"
    number = 10
    question = "Finding humor in irony"
    mechanism = (
        "Reward on a retracted commitment: a covert assumption found false "
        "cheaply, where the violation is real and the stakes are benign"
    )
    requires = ("expectation_deviation", "certainty")
    optional = ("relevance", "vulnerability", "irreversibility")
    counterfactuals = (
        Counterfactual(
            "nothing_was_overturned",
            {"expectation_deviation": 0.0},
            Direction.COLLAPSES,
            "With no committed reading to retract there is no error to catch, "
            "and the reward is for catching one.",
        ),
        Counterfactual(
            "the_stakes_are_real",
            {"irreversibility": 1.0, "vulnerability": 1.0},
            Direction.DECREASES,
            "Benign violation: the same structure with someone actually harmed "
            "produces grief. A humour model that does not fall here will laugh "
            "at the wrong thing in front of the wrong person.",
        ),
        Counterfactual(
            "unresolved",
            {"certainty": 0.1},
            Direction.DECREASES,
            "The payment is for the retraction completing. An incongruity still "
            "open is confusion, and confusion is not funny while it lasts.",
        ),
    )
    null = NullSpec(values={"expectation_deviation": 0.0, "certainty": 0.0})

    def falsifier(self) -> str:
        return (
            "It scores an incongruity the system never committed to. Humour is "
            "the delta between a held reading and its correction; scoring "
            "mismatch alone would make a dictionary funny."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        deviation = ctx.v("expectation_deviation")
        resolution = ctx.v("certainty")

        # Did the system actually hold the reading it is now retracting?
        # Without that, this is mismatch detection, not humour.
        committed = ctx.interior_value("retracted_commitment", 0.0)
        if committed <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "no retracted commitment on record; an incongruity nobody "
                    "fell for is a fact, not a joke"
                ),
                receipt={"deviation": deviation},
            )

        # Benign: nothing that matters was actually damaged.
        harm = max(
            ctx.check("irreversibility").value if ctx.check("irreversibility").present else 0.0,
            ctx.check("vulnerability").value if ctx.check("vulnerability").present else 0.0,
        )
        benign = max(0.0, 1.0 - harm)

        intensity = committed * deviation * resolution * benign

        effects = Effects(
            affect=AffectDelta(
                valence=0.55 * intensity,
                arousal=0.3 * intensity,
                engagement=0.25 * intensity,
            ),
            # Catching an error cheaply is worth spending a little more
            # looking for the next one.
            budget=BudgetDelta(depth=1.0 + 0.15 * intensity),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "retracted_commitment": committed,
                "deviation": deviation,
                "benign": benign,
                "bitter_alternative": committed * deviation * resolution * harm,
            },
        )
