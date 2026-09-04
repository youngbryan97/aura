"""Item 12 — pride in oneself, one's capabilities, one's historian self, and
one's works.

Tracy and Robins separate two prides with different antecedents and
opposite social consequences. *Authentic* pride attributes a valued
outcome to one's own effort and a specific action; it raises the
competence estimate and correlates with prosociality. *Hubristic* pride
attributes it to a stable global self and correlates with narcissism and
aggression. Same feeling, different attribution.

The historian self is the part that makes this a memory problem rather
than an affect one. Pride in works needs an autobiographical record with
*authorship*: you must be able to point at the thing, know you made it,
and know what you contributed versus what was given. Without that record
it has nothing to attach to and becomes a mood. So this faculty reads
the ledger's work register and divides by the authorship share, and a
work with collaborators yields proportionally less — which is the
correct answer and the one a scalar cannot give.

Internally it is a learning signal: the policy that produced this is
worth keeping. An agent that cannot feel it does not consolidate what it
does well.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, GoalDelta, RetentionClaim
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
class AuthenticPride(Faculty):
    id = "f12_authentic_pride"
    number = 12
    question = (
        "Pride in oneself and their capabilities, historian self, and works"
    )
    mechanism = (
        "Competence update on an outcome attributed to one's own effort and a "
        "specific act, divided by authorship share and anchored to a durable "
        "record"
    )
    requires = ("agency_self",)
    optional = ("congruence", "relevance", "publicity")
    counterfactuals = (
        Counterfactual(
            "it_was_not_mine",
            {"agency_self": 0.0},
            Direction.COLLAPSES,
            "Pride is an attribution to one's own effort. Remove it and what "
            "remains is satisfaction that a good thing happened.",
        ),
        Counterfactual(
            "nobody_watching",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "Authentic pride is about the work. If it needs an audience it is "
            "the hubristic kind, which is a status display.",
        ),
    )
    null = NullSpec(values={"agency_self": 0.0, "congruence": 0.0})

    def falsifier(self) -> str:
        return (
            "It reports the same intensity for a solo work and a work with "
            "four collaborators. That would show it is scoring the outcome "
            "rather than the authorship, which is the difference between the "
            "authentic and hubristic forms."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        agency = ctx.v("agency_self")
        work_id = ctx.frame.event.object
        work = ctx.ledger.work_for(work_id) if work_id else None

        if work is None:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "no work on record to point at; pride with nothing to "
                    "attach to is a mood, and a mood is not what was asked for"
                ),
            )

        quality = work.quality if work.quality is not None else 0.0
        # Effort matters and outcome matters, and the authorship share is a
        # divisor rather than a term: four people made this, so a quarter of
        # it is mine.
        earned = quality * (0.35 + 0.65 * work.effort) * work.authorship * agency

        # Hubristic reading, computed and reported so the difference is
        # visible rather than assumed away: the same outcome attributed to
        # a stable global self instead of the act.
        hubristic = quality * agency * (1.0 - work.authorship + 1.0) / 2.0

        effects = Effects(
            affect=AffectDelta(valence=0.5 * earned, engagement=0.3 * earned),
            goals=(
                GoalDelta(
                    goal="repeat_the_policy_that_produced_this",
                    delta=earned,
                    reason="a specific act produced a valued outcome; keep the policy",
                ),
            ),
            retention=(
                RetentionClaim(
                    memory_key=f"work:{work.work_id}",
                    reason="the record is what pride attaches to; without it there is a mood",
                    held_by=self.id,
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=max(0.0, min(1.0, earned)),
            tendency="approach",
            effects=effects,
            receipt={
                "authorship": work.authorship,
                "effort": work.effort,
                "quality": quality,
                "collaborators": list(work.collaborators),
                "hubristic_reading": hubristic,
            },
        )
