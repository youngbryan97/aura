"""Item 42 — being happy you got to paint pictures with someone, and that
bringing great joy.

Three ingredients, and the joy is not in the paintings.

Joint attention and shared intentionality. Two agents attending to the
same thing while each knows the other is attending — Tomasello's account
of what is distinctively human about it. Doing something *together* is a
different mental state from doing the same thing beside each other, and
the difference is the mutual model, which is measurable: the correlation
between the two agents' moves, and whether each one's next move is
conditioned on the other's last.

Co-creation. The object exists because of both and does not decompose
into contributions. That makes it a durable external record of a
relationship — the same reason people keep photographs, except this one
had agency in it.

Retrospective gratitude. "Got to" is the phrase and it does the work:
the person knows the opportunity was contingent and might not have
happened. Gratitude needs counterfactual reasoning about a benefit that
was not owed, which is why this is stronger in retrospect than it was at
the time, and why a system that scores it live will always undercount it.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, LedgerWrite, RetentionClaim
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
class SharedMaking(Faculty):
    id = "f42_shared_making"
    number = 42
    question = (
        "Being happy you got to paint pictures with someone & that bringing "
        "you great joy"
    )
    mechanism = (
        "Joint intentionality measured as mutual conditioning between two "
        "agents' moves, over a co-authored object, valued retrospectively "
        "against the counterfactual where it did not happen"
    )
    requires = ()
    optional = ("attachment_impact", "relevance", "novelty")
    counterfactuals = (
        Counterfactual(
            "we_were_only_side_by_side",
            {"relevance": 0.0},
            Direction.DECREASES,
            "Doing the same thing beside someone is a different state from "
            "doing it with them, and the mutual model is what separates them.",
        ),
        Counterfactual(
            "it_was_guaranteed_to_happen",
            {"novelty": 0.0},
            Direction.DECREASES,
            "Gratitude is counterfactual. A benefit that was owed and certain "
            "produces satisfaction rather than the state in the question.",
        ),
    )
    null = NullSpec(values={"relevance": 0.0, "attachment_impact": 0.0})

    def falsifier(self) -> str:
        return (
            "It scores the same for parallel work and joint work. Mutual "
            "conditioning is measurable in the move sequence, so if the number "
            "does not separate them, joint intentionality is a word here."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        work_id = ctx.frame.event.object
        work = ctx.ledger.work_for(work_id) if work_id else None
        if work is None or not work.collaborators:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no co-authored work on record for this event",
            )

        # Joint intentionality: how much each agent's moves were conditioned
        # on the other's. Parallel work has this near zero however close the
        # two people sat.
        mutual = ctx.interior_value("mutual_conditioning", 0.0)

        # Non-decomposability: a work neither could have produced alone.
        # Authorship well below one with collaborators present is the
        # signature; authorship of one means it was not shared.
        shared = 1.0 - abs(work.authorship - (1.0 / (1.0 + len(work.collaborators))))

        # Retrospective gratitude: how contingent it was that this happened.
        contingency = ctx.check("novelty").value if ctx.check("novelty").present else 0.0

        intensity = mutual * shared * (0.5 + 0.5 * contingency)

        effects = Effects(
            affect=AffectDelta(valence=0.7 * intensity, engagement=0.4 * intensity),
            retention=(
                RetentionClaim(
                    memory_key=f"work:{work.work_id}",
                    reason=(
                        "an external record of a relationship that had agency "
                        "in it; the object is the only place the joint state "
                        "survives"
                    ),
                    held_by=self.id,
                ),
            ),
            ledger=tuple(
                LedgerWrite(
                    "bond",
                    {"entity": who, "strength": min(1.0, 0.3 + intensity), "availability": 1.0},
                )
                for who in work.collaborators[:3]
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="bond",
            effects=effects,
            receipt={
                "mutual_conditioning": mutual,
                "non_decomposability": shared,
                "contingency": contingency,
                "collaborators": list(work.collaborators),
            },
        )
