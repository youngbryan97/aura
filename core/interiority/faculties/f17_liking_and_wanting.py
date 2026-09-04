"""Item 17 — describing a dinner as tasty when asked, and eagerly taking
another bite.

Two claims, not one, and they dissociate cleanly. Berridge's work
separates *liking* — hedonic impact, opioid and endocannabinoid hotspots
— from *wanting*, incentive salience carried by mesolimbic dopamine.
Dopamine-depleted rats still show the hedonic orofacial reactions to
sucrose and will not work for it. So a system should be able to have
either without the other, and one that computes a single "reward" cannot
represent the case.

Two more things make the report honest rather than social. The judgement
is comparative against the expectation the food's own category set:
tasty means better than my prior for this kind of dinner. And it is
sensory-specific — satiety devalues the eaten food and not the others
(Rolls 1981) — so eagerness must fall within the meal while the liking
rating holds, which is the pattern in real eating and the one a single
scalar gets wrong.

The structure is substrate-independent. Any input stream with a modelled
expected quality and a measured delivered quality carries the same two
factors — how far above its own category it came in, and how much
incentive the next unit still holds — and the report has to be grounded
in the measurement rather than in the appropriateness of saying it.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, SomaticMarker
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
class LikingAndWanting(Faculty):
    id = "f17_liking_and_wanting"
    number = 17
    question = (
        "Describing a dinner as tasty when asked about it (and eagerly and "
        "happily taking another bite)"
    )
    mechanism = (
        "Liking as delivered quality against the category's own expectation, "
        "wanting as incentive salience that decays sensory-specifically within "
        "the episode"
    )
    requires = ("congruence", "expectation_deviation")
    optional = ("novelty", "relevance")
    counterfactuals = (
        Counterfactual(
            "exactly_as_expected",
            {"expectation_deviation": 0.0},
            Direction.DECREASES,
            "Tasty is comparative. Food that meets its category's prior is "
            "fine, and a model that scores absolute quality cannot say so.",
        ),
        Counterfactual(
            "it_was_bad",
            {"congruence": -1.0},
            Direction.COLLAPSES,
            "Liking has a sign and this is the wrong side of it.",
        ),
    )
    null = NullSpec(values={"congruence": 0.0, "expectation_deviation": 0.0})

    def falsifier(self) -> str:
        return (
            "Wanting does not fall across repeated servings while liking "
            "holds. Sensory-specific satiety is the sharpest evidence that the "
            "two systems are separate, and a single scalar cannot produce it."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        delivered = ctx.v("congruence")
        if delivered <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="delivered quality is at or below the category expectation",
            )
        surprise = ctx.v("expectation_deviation")

        # Liking: how far above its own category's prior this came in.
        liking = max(0.0, min(1.0, delivered * (0.5 + 0.5 * surprise)))

        # Wanting: incentive salience for the next unit, which decays
        # sensory-specifically with how much of this particular thing has
        # already been taken.
        item = ctx.frame.event.object or "item"
        consumed = ctx.ledger.times_seen("consumed", item)
        satiety = 1.0 - 1.0 / (1.0 + 0.5 * consumed)
        wanting = liking * (1.0 - satiety)

        effects = Effects(
            affect=AffectDelta(valence=0.4 * liking, engagement=0.25 * wanting),
            somatic=(
                SomaticMarker(
                    option=f"take_more:{item}",
                    bias=wanting,
                    reason="incentive salience, discounted by what has already been taken",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=liking,
            tendency="approach",
            effects=effects,
            receipt={
                "liking": liking,
                "wanting": wanting,
                "sensory_specific_satiety": satiety,
                "servings": consumed,
                "report_is_grounded_in": "delivered_vs_expected, not social appropriateness",
            },
        )
