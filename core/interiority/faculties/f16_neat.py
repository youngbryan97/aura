"""Item 16 — thinking something is neat after reading about it.

The smallest item on the list, which is why it is worth being exact. It
is the low-intensity, non-self-relevant form of epistemic pleasure: a
fact that fits a model you already had, in a place you did not know had
a gap, at almost no cost.

Formally it is a belief update with a small divergence and a high ratio
of update to effort — efficient surprise. It does not need
accommodation, which would be awe; it does not threaten anything, which
would be alarm; it demands no action. What it actually does is value the
source, and that is the function: it steers where to read next.

It is kept separate from curiosity on purpose. Curiosity is a drive
toward an unfilled gap; neat is the receipt after one is filled cheaply.
Merging them produces a system that reports interest in things it has
already learned.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, AttentionBias, Effects
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
class Neat(Faculty):
    id = "f16_neat"
    number = 16
    question = "Thinking something is neat after reading about it"
    mechanism = (
        "Efficient surprise: a small model update at a high update-to-effort "
        "ratio, whose output is a valuation of the source"
    )
    requires = ("novelty", "certainty")
    optional = ("relevance", "control")
    counterfactuals = (
        Counterfactual(
            "already_knew_it",
            {"novelty": 0.0},
            Direction.COLLAPSES,
            "The receipt is for an update. Nothing updated, nothing to receipt, "
            "and a system that fires here reports interest in what it knows.",
        ),
        Counterfactual(
            "it_cost_a_lot_to_get",
            {"control": 0.1},
            Direction.DECREASES,
            "It is the ratio that makes it neat rather than merely learned. "
            "Hard-won knowledge produces satisfaction, which is a different "
            "state with a different follow-on.",
        ),
        Counterfactual(
            "it_changes_everything",
            {"relevance": 1.0},
            Direction.DECREASES,
            "Neat is non-self-relevant by construction. A fact that touches "
            "what the agent is holding produces alarm or relief instead.",
        ),
    )
    null = NullSpec(values={"novelty": 0.0, "certainty": 0.0})

    def falsifier(self) -> str:
        return (
            "It fires on a fact that demands the model be rebuilt. That is awe "
            "or alarm, and reporting it as neat would show the accommodation "
            "term is not in the mechanism."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        novelty = ctx.v("novelty")
        assimilable = ctx.v("certainty")
        ease = ctx.check("control").value if ctx.check("control").present else 0.5
        stake = ctx.check("relevance").value if ctx.check("relevance").present else 0.0

        # Efficient surprise: it updated, it fitted, it was cheap, and it
        # did not touch anything at stake.
        intensity = novelty * assimilable * ease * (1.0 - stake)

        source = ctx.frame.event.source or "source"
        effects = Effects(
            affect=AffectDelta(valence=0.25 * intensity, engagement=0.3 * intensity),
            # The actual function: read more of this.
            attention=(
                AttentionBias(
                    target=f"source:{source}",
                    weight=intensity,
                    reason="this source paid well per unit of effort",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend",
            effects=effects,
            receipt={
                "update": novelty,
                "assimilability": assimilable,
                "effort_ratio": ease,
                "self_relevance": stake,
            },
        )
