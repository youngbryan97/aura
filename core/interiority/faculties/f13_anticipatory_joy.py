"""Item 13 — happiness after a long wait to adopt a child, and envisioning a
life with them.

Anticipatory reward is mechanistically different from consummatory
reward. Dopaminergic prediction error fires on the cue and on the
improvement in expectation, not on an outcome once it is predicted
(Schultz 1998). The joy at approval is not the joy of having a child —
nothing has happened yet. It is a long-held uncertainty collapsing in
the favourable direction.

Two more components make it what it is. The wait: anticipation utility
means a resolution after long uncertainty is worth more than the same
outcome delivered immediately (Loewenstein 1987). And the envisioning:
episodic future thinking runs on the same machinery as episodic memory
(Schacter and Addis 2007), which is what turns an abstract good into a
felt one and what makes the commitment durable, because a vividly
simulated future is defended like a possessed one.

The state is prospective and committing, so its effects are goal-stack
changes rather than a happiness scalar. A version that only raises
valence has missed everything about it.
"""

from __future__ import annotations

import math

from core.interiority.effects import AffectDelta, Effects, GoalDelta, LedgerWrite
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
class AnticipatoryJoy(Faculty):
    id = "f13_anticipatory_joy"
    number = 13
    question = (
        "Being happy after waiting a long time to adopt a child and "
        "envisioning a life with them"
    )
    mechanism = (
        "Collapse of a long-held uncertainty in the favourable direction, plus "
        "a simulated future that runs without hitting an obstacle"
    )
    requires = ("congruence", "certainty")
    optional = ("relevance", "attachment_impact", "novelty", "expectation_deviation")
    counterfactuals = (
        Counterfactual(
            "it_was_never_in_doubt",
            {"expectation_deviation": 0.0, "novelty": 0.0},
            Direction.DECREASES,
            "The reward is on the improvement in expectation, which is what "
            "expectation deviation measures. An outcome already predicted "
            "pays nothing when it arrives, which is why a formality does not "
            "feel like this however certain and however welcome.",
        ),
        Counterfactual(
            "it_went_the_other_way",
            {"congruence": -1.0},
            Direction.COLLAPSES,
            "Resolution alone is not the reward; the direction is half of it.",
        ),
    )
    # A resolution that went her way. The default world is a bad event,
    # and anticipatory joy correctly declines there.
    activation = {"congruence": 0.8}
    null = NullSpec(values={"congruence": 0.0, "certainty": 0.0})

    def falsifier(self) -> str:
        return (
            "The same outcome delivered immediately produces the same "
            "intensity as one waited years for. That would show the wait is "
            "decoration rather than a term."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        congruence = ctx.v("congruence")
        if congruence <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="the resolution went the other way",
            )
        certainty = ctx.v("certainty")

        # How long the uncertainty was held, from the goal's own record.
        goal_name = ctx.frame.event.object or ""
        held_since = ctx.ledger.notes.expectation("goal_opened", goal_name)
        # Anticipation utility rises with the log of the wait: the first
        # year of waiting adds far more than the fifth, which is the shape
        # reported rather than a linear one.
        wait_years = max(0.0, (held_since or 0.0))
        wait_term = math.log1p(wait_years) / math.log1p(10.0)

        # The envisioning: a future scene constructed far enough to be
        # defended. Measured as how many downstream goals the resolution
        # opens, not as a vividness the caller asserts.
        opened = ctx.ledger.substitutes_for(goal_name, None) or 0
        envisioned = 1.0 - 1.0 / (1.0 + opened)

        # Prediction error, not outcome value. How far the world moved from
        # what was expected is the term that pays; certainty scales it
        # because an unconfirmed resolution moves less. Without the
        # deviation term this rose when the outcome became more certain,
        # which is the opposite of what the mechanism claims.
        improvement = (
            ctx.check("expectation_deviation").value
            if ctx.check("expectation_deviation").present
            else 1.0
        )
        intensity = (
            congruence
            * certainty
            * improvement
            * (0.4 + 0.35 * wait_term + 0.25 * envisioned)
        )
        intensity = max(0.0, min(1.0, intensity))

        effects = Effects(
            affect=AffectDelta(
                valence=0.8 * intensity, arousal=0.35 * intensity, engagement=0.5 * intensity
            ),
            # The committing part. A prospective state that changes no goal
            # is a feeling about a future rather than a commitment to one.
            goals=(
                GoalDelta(
                    goal=f"prepare:{goal_name}",
                    delta=intensity,
                    reason="the future is now near enough to act toward",
                ),
            ),
            ledger=(
                (
                    LedgerWrite(
                        "bond",
                        {
                            "entity": ctx.frame.event.subject,
                            "strength": intensity,
                            "availability": 1.0,
                        },
                    ),
                )
                if ctx.frame.event.subject
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "wait_term": wait_term,
                "envisioned_branches": opened,
                "uncertainty_collapsed": certainty * congruence,
            },
        )
