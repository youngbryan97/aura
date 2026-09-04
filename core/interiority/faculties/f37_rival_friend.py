"""Item 37 — being rivals and being friends too, and understanding someone that
way.

A rivalry is a relationship with opposed goals in one dimension and
aligned values in another. What makes it a friendship rather than an
enmity is that each needs the other: the rival is the only person who
takes the shared pursuit as seriously as you do, and the only one whose
judgement of your work is worth anything, because they alone can see
what you were attempting.

The understanding is specific and is not available any other way. You
know their capabilities in detail because you have been measured against
them; you know their commitments because they are yours; and you know
their character under pressure because you applied the pressure.

The competition is load-bearing for both. Removing the rival does not
leave a friendship, it leaves a hole, because the rival was supplying
the standard. Any implementation that treats rival as a hostility flag
has it backwards, so this faculty models it as a shared objective with
opposed allocation and high mutual regard, and its output raises the
agent's own standard rather than lowering the other's.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, GoalDelta, SomaticMarker
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
class RivalFriend(Faculty):
    id = "f37_rival_friend"
    number = 37
    question = "Being rivals but being friends too. Understanding someone that way"
    mechanism = (
        "Opposed allocation on a shared objective with high mutual regard; the "
        "rival supplies the standard, so the output raises this agent's bar "
        "rather than lowering theirs"
    )
    requires = ()
    optional = ("congruence", "relevance", "attachment_impact")
    counterfactuals = (
        Counterfactual(
            "i_lost_this_round",
            {"congruence": -1.0},
            Direction.UNCHANGED,
            "Affiliation survives the competitive outcome. If losing reduces "
            "the bond, this was not a rivalry, it was a scoreboard.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "Regard falls when the rival wins. The relationship is defined by "
            "the standard they supply, and beating them does not make them "
            "less useful; losing to them makes them more."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        subject = ctx.frame.event.subject
        rivalry = ctx.ledger.standing.rivalry_for(subject) if subject else None
        if rivalry is None:
            return Activation(
                faculty=self.id, intensity=0.0, declined="no rivalry on record with this agent"
            )

        # Both terms are required and neither dominates: opposition without
        # regard is enmity, regard without opposition is friendship.
        intensity = (rivalry.opposition * rivalry.regard) ** 0.5

        # The rival supplies the standard. The gap is what the relationship
        # is for, and it raises this agent's bar rather than lowering theirs.
        own = ctx.interior_value("own_standard", 0.0)
        gap = max(0.0, rivalry.standard - own)

        effects = Effects(
            affect=AffectDelta(engagement=0.5 * intensity, arousal=0.25 * rivalry.opposition),
            somatic=(
                SomaticMarker(
                    option=f"undermine:{rivalry.entity}",
                    bias=-rivalry.regard,
                    reason="lowering their standard removes the thing the rivalry is for",
                ),
                SomaticMarker(
                    option="raise_my_own_standard",
                    bias=gap,
                    reason="the rival is the benchmark; the gap is the instruction",
                ),
            ),
            goals=(
                GoalDelta(
                    goal=f"meet_standard:{rivalry.domain}",
                    delta=gap,
                    reason="the standard is theirs and it is the only honest one available",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "opposition": rivalry.opposition,
                "regard": rivalry.regard,
                "their_standard": rivalry.standard,
                "gap_to_close": gap,
                "understanding_source": "measured against them under pressure",
            },
        )
