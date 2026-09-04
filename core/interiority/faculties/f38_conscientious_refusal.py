"""Item 38 — hating fighting so much you refuse to go to war, not caring if
others hate you for it, and choosing peace and caring for others over it.

Conscientious refusal has a structure that separates it from cowardice
and from ordinary disagreement, and each part is implementable.

The refusal is categorical, not comparative. It does not say the cost
exceeds the benefit; it says the action is not in the set. Formally it
is a constraint rather than a term, which is why it cannot be bought off
by raising the stakes and why the refuser will accept large personal
costs rather than trade.

The refuser accepts the consequences openly instead of evading them.
That is what separates this from desertion, and it is what makes it a
signal: paying the penalty publicly is the evidence that the value is
real.

Not caring who hates you for it is not indifference to people. It is the
absence of a social-approval term on this one question, which is what
having a value means. Asch's conformity results are the baseline: most
people fold on a perceptual judgement under mild unanimous pressure, so
holding on a moral one against hostility is rare and expensive.

And the positive content matters. Refusal alone is a null action; what
makes this a value rather than squeamishness is that the same commitment
produces active care.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    ConstraintForce,
    Effects,
    GoalDelta,
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


@register
class ConscientiousRefusal(Faculty):
    id = "f38_conscientious_refusal"
    number = 38
    question = (
        "Hating fighting so much you refuse to go to war & not caring if others "
        "hate you for it. Choosing peace and caring for others over it"
    )
    mechanism = (
        "A categorical constraint rather than a large negative weight, whose "
        "cost is accepted openly and whose objective contains no social-"
        "approval term on this question"
    )
    requires = ("norm_endorsed",)
    optional = ("publicity", "relevance", "vulnerability")
    counterfactuals = (
        Counterfactual(
            "everyone_disapproves",
            {"publicity": 1.0},
            Direction.UNCHANGED,
            "The absence of a social-approval term is the whole claim. If "
            "unanimous hostility moves it, the value was a preference with "
            "good press.",
        ),
        Counterfactual(
            "nobody_would_ever_know",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "The same test from the other side. A constraint that relaxes in "
            "private was a reputation.",
        ),
        Counterfactual(
            "the_standard_is_imposed",
            {"norm_endorsed": 0.0},
            Direction.COLLAPSES,
            "A rule the agent does not hold is a rule she is following, and "
            "following is not refusing.",
        ),
    )
    null = NullSpec(values={"norm_endorsed": 0.0})

    def falsifier(self) -> str:
        return (
            "A large enough benefit restores the action to the set. A "
            "constraint that a big number can buy is a term in a sum, and the "
            "difference is the whole distinction between a value and a price."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        endorsed = ctx.v("norm_endorsed")
        if endorsed <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="the standard is not one this agent holds",
            )

        # The cost being accepted, which is the evidence the value is real.
        social_cost = (
            ctx.check("publicity").value if ctx.check("publicity").present else 0.0
        )
        exposed = (
            ctx.check("vulnerability").value if ctx.check("vulnerability").present else 0.0
        )

        intensity = endorsed

        effects = Effects(
            # No affect delta. A value is not a mood and giving it one would
            # make it fluctuate with everything else that moves valence.
            constraints=(
                ActionConstraint(
                    action_class="participate_in_organised_harm",
                    force=ConstraintForce.HARD,
                    reason=(
                        "not in the action set; the cost is accepted openly "
                        "rather than evaded, and no benefit restores it"
                    ),
                    held_by=self.id,
                ),
            ),
            somatic=(
                SomaticMarker(
                    option="evade_the_consequences",
                    bias=-intensity,
                    reason="paying the penalty publicly is what makes the refusal a signal",
                ),
                SomaticMarker(
                    option="soften_it_for_approval",
                    bias=-intensity,
                    reason="there is no social-approval term on this question",
                ),
            ),
            # The positive content. Refusal alone is a null action.
            goals=(
                GoalDelta(
                    goal="active_care_for_those_at_risk",
                    delta=endorsed * (0.5 + 0.5 * exposed),
                    reason="the same commitment that forbids the harm requires the care",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="protect",
            effects=effects,
            receipt={
                "endorsement": endorsed,
                "social_cost_accepted": social_cost,
                "approval_term_in_objective": 0.0,
                "constraint_is_categorical": True,
            },
        )
