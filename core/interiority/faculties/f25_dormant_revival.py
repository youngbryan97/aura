"""Item 25 — surfing again after forty years, because you are in shape, you met
someone who does it, and it is simply time.

Three conditions, and none alone would do it.

The skill is intact. Motor skills consolidated to proceduralisation
decay far more slowly than declarative memory — the savings effect, and
the continuous-skill retention literature since. So the forty-year gap
is smaller than it looks, and the person knows this in their body.

The body permits it. Being in shape is a live measurement, not a memory,
and it is what makes this a decision rather than a fantasy. The
self-model has to be current.

A social affordance appeared. Meeting someone supplies a partner, a
place, a schedule, and permission. A dormant capability sits below
threshold until something supplies the missing term.

Then "it is simply just time", which names the *absence of remaining
obstacles* being noticed all at once. The person did not decide; they
noticed the reasons had expired. That is a different mental event from
deliberation and it has a signature — no argument, immediate certainty —
so this faculty fires on the ledger's blocker list emptying rather than
on a weighted sum crossing a line.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, GoalDelta
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
class DormantRevival(Faculty):
    id = "f25_dormant_revival"
    number = 25
    question = (
        "Having a hobby (lets say surfing), not doing it for 40 years, meeting "
        "someone who does it, and then taking it up again because youre in "
        "shape, you met someone, and it's simply just time"
    )
    mechanism = (
        "A retained capability below threshold, revived when its blocker list "
        "empties rather than when a score crosses a line"
    )
    requires = ()
    optional = ("relevance", "novelty", "power")
    counterfactuals = (
        Counterfactual(
            "no_affordance",
            {"power": 0.0},
            Direction.DECREASES,
            "A dormant capability needs an occasion. Without one the skill is "
            "retained and unreachable, which is the state it was already in.",
            withhold=("practice",),
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "It fires while a blocker is still on the list. Recognition that "
            "the reasons have expired is the mental event; a weighted sum that "
            "outvotes a live obstacle is deliberation, and it feels different "
            "because it is different."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        name = ctx.frame.event.object
        practice = ctx.ledger.making.practice_for(name) if name else None
        if practice is None:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no dormant practice on record under this name",
            )

        residual = practice.residual(ctx.now or None)
        blockers = practice.blockers

        if blockers:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    f"{len(blockers)} blocker(s) still standing: "
                    f"{', '.join(blockers[:4])}. Revival is the list emptying, "
                    "not a score outvoting an obstacle"
                ),
                receipt={"residual_skill": residual, "blockers": list(blockers)},
            )

        # Everything that was in the way has gone, and the skill is still
        # there. The intensity is the retained capability, because that is
        # what is now reachable.
        intensity = residual

        effects = Effects(
            affect=AffectDelta(valence=0.45 * intensity, engagement=0.6 * intensity),
            goals=(
                GoalDelta(
                    goal=f"resume:{practice.name}",
                    delta=intensity,
                    reason="the capability is retained and nothing is in the way any more",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "residual_skill": residual,
                "peak_skill": practice.peak_skill,
                "blockers_cleared": True,
                "recognition_not_deliberation": True,
            },
        )
