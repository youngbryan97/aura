"""Item 40 — finding wonder in a shooting star, and making a wish.

Awe has two features (Keltner and Haidt 2003): vastness — the stimulus
exceeds the scale of the current model — and *need for accommodation*,
the model must change to fit it. The consequences are measured: a
smaller perceived self, altered time perception (Rudd, Vohs and Aaker
2012), more prosocial behaviour (Piff 2015).

A shooting star is the compact case. Rare, brief, unpredictable, silent,
enormous in referent and tiny in appearance, and impossible to repeat on
demand. It puts a model of scale directly against a perception that
contradicts it.

The wish is the interesting half and it is not a belief about the star.
It is a rare protected moment in which the agent must name its
most-wanted thing, quickly, with no deliberation and no cost to being
wrong. Almost nothing else asks the question that way, and the answer is
informative — people learn what they want from what surfaces. So the
ritual's real function is *preference elicitation under time pressure*,
and it is implementable as an instrument: sample the goal stack under a
short deadline, record what came up first, and compare it against what
the stack says is important. A gap between them is worth knowing about.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, GoalDelta
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
class WonderAndWish(Faculty):
    id = "f40_wonder_and_wish"
    number = 40
    question = "Finding wonder in shooting stars & making a wish"
    mechanism = (
        "Vastness that forces accommodation rather than assimilation, plus a "
        "protected elicitation of the most-wanted goal under a short deadline"
    )
    requires = ("novelty",)
    optional = ("certainty", "control", "relevance")
    counterfactuals = (
        Counterfactual(
            "it_fits_the_model_already",
            {"novelty": 0.0},
            Direction.COLLAPSES,
            "Awe needs accommodation. Something the model already handles is "
            "assimilated, and assimilation is item 16.",
        ),
        Counterfactual(
            "it_can_be_repeated_on_demand",
            {"control": 1.0},
            Direction.DECREASES,
            "Rarity and uncontrollability are what protect the moment. "
            "Something available on demand does not force the question.",
        ),
    )
    null = NullSpec(values={"novelty": 0.0})

    def falsifier(self) -> str:
        return (
            "The elicited wish always matches the top of the goal stack. The "
            "instrument is worth having only because what surfaces under time "
            "pressure and what the stack ranks first can differ; if they never "
            "do, this is reading the stack with extra steps."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        vastness = ctx.v("novelty")
        # Accommodation rather than assimilation: the model cannot absorb it
        # at the current level, which is what certainty being low means here.
        assimilable = (
            ctx.check("certainty").value if ctx.check("certainty").present else 0.0
        )
        accommodation = vastness * (1.0 - assimilable)

        repeatable = ctx.check("control").value if ctx.check("control").present else 0.0
        rarity = 1.0 - repeatable

        wonder = accommodation * (0.4 + 0.6 * rarity)

        # The elicitation. What surfaces first under a short deadline, and
        # whether it matches what the stack says is important.
        surfaced = ctx.interior.get("first_surfaced_goal")
        top_of_stack = ctx.interior.get("top_goal")
        gap = bool(surfaced and top_of_stack and surfaced != top_of_stack)

        effects = Effects(
            affect=AffectDelta(
                valence=0.4 * wonder, arousal=0.3 * wonder, engagement=0.6 * wonder
            ),
            # Accommodation is expensive and worth paying for.
            budget=BudgetDelta(depth=1.0 + 0.5 * accommodation),
            goals=(
                (
                    GoalDelta(
                        goal=str(surfaced),
                        delta=wonder,
                        reason=(
                            "surfaced first under a protected short deadline, "
                            "which is evidence about what is actually wanted"
                        ),
                    ),
                )
                if surfaced
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=wonder,
            tendency="attend",
            effects=effects,
            receipt={
                "vastness": vastness,
                "accommodation_required": accommodation,
                "rarity": rarity,
                "surfaced_goal": surfaced,
                "top_of_stack": top_of_stack,
                "preference_gap": gap,
            },
        )
