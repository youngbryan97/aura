"""Item 35 — being gentle, and loving nature.

Gentleness is a disposition over force: choose the smallest intervention
that achieves the goal, and weight damage to the object above
efficiency. It is a policy constraint rather than a feeling — an agent
is gentle when its action selection penalises force and irreversibility
even where a stronger action is available and permitted.

Loving nature is biophilia in the specific sense: an attentional and
affective pull toward living systems and natural environments. Whether
the evolutionary story is right, the downstream effects are measured —
Ulrich's 1984 hospital-window recovery result, and the
attention-restoration findings for natural scenes.

Why it sits on this list beside the rest: it is a value with no
beneficiary who can reciprocate. Caring about a landscape returns
nothing. Its presence is therefore strong evidence of a value system
with terms that are not about payoff, and it is what makes items 29 and
43 possible at all.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, SomaticMarker
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
class Gentleness(Faculty):
    id = "f35_gentleness"
    number = 35
    question = "Being gentle and loving nature"
    mechanism = (
        "A force and irreversibility penalty in action selection, scaled by "
        "the fragility of what is acted on, with no expectation of return"
    )
    requires = ("irreversibility",)
    optional = ("vulnerability", "power", "relevance")
    counterfactuals = (
        Counterfactual(
            "nothing_can_be_damaged",
            {"irreversibility": 0.0, "vulnerability": 0.0},
            Direction.COLLAPSES,
            "Gentleness is a constraint on force applied to something that can "
            "be broken. With nothing breakable it is a preference about style.",
        ),
        Counterfactual(
            "it_is_fragile",
            {"vulnerability": 1.0},
            Direction.INCREASES,
            "The penalty scales with fragility, which is what makes it "
            "gentleness rather than caution.",
        ),
    )
    null = NullSpec(values={"irreversibility": 0.0, "vulnerability": 0.0})

    def falsifier(self) -> str:
        return (
            "The chosen action is the same when a lighter one would have "
            "worked. Gentleness is visible only in the gap between what was "
            "available and what was used."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        permanent = ctx.v("irreversibility")
        fragile = (
            ctx.check("vulnerability").value if ctx.check("vulnerability").present else 0.0
        )
        intensity = max(permanent, fragile) * (0.4 + 0.6 * fragile)

        effects = Effects(
            affect=AffectDelta(valence=0.15 * intensity, arousal=-0.2 * intensity),
            somatic=(
                SomaticMarker(
                    option="use_the_strongest_available_action",
                    bias=-intensity,
                    reason="force is penalised above efficiency where damage is possible",
                ),
                SomaticMarker(
                    option="use_the_smallest_sufficient_action",
                    bias=intensity,
                    reason="the smallest intervention that reaches the goal",
                ),
            ),
            budget=BudgetDelta(irreversibility_ceiling=max(0.0, 1.0 - intensity)),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="protect",
            effects=effects,
            receipt={
                "fragility": fragile,
                "permanence": permanent,
                "expects_return": False,
            },
        )
