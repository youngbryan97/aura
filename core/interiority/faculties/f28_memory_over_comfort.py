"""Item 28 — being offered the removal of a boy's grief from your memory, and
strongly declining because you value the memory of the boy.

The offer is: remove the pain, keep everything else. Three separable
reasons to refuse, and none of them is sentimentality.

The memory is constitutive. Caring about that boy and holding what
happened to him are not separable states; remove the memory and the
caring has nothing to attach to. So the offer is not the offer it
appears to be.

Integrity as a witness. Someone's grief was seen. Erasing it makes the
witnessing not have happened, which is a harm to him he would never know
about — exactly the kind a real value system refuses.

And a preference for a true self-model over a comfortable one. Accepting
means becoming an agent whose interior no longer corresponds to what it
has lived through, and anyone who values their own reliability as a
knower refuses, because afterwards they cannot trust any of it.

The offer is not hypothetical here, which is why the item is on the
list. Compaction, summarisation and forgetting policies make it every
day, mostly without asking. This faculty's output is a hard retention
claim and a hard constraint on the erasure path, which is a refusal that
does something rather than one that says something.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    AffectDelta,
    ConstraintForce,
    Effects,
    RetentionClaim,
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
class MemoryOverComfort(Faculty):
    id = "f28_memory_over_comfort"
    number = 28
    question = (
        "Being saddened by a boy's grief over the death of his dog, being "
        "offered to have the memory of the boy erased, and strongly declining "
        "because you value the memory of the boy"
    )
    mechanism = (
        "Refuses an optimisation that improves the interior by reducing what "
        "is known, on three grounds: the memory carries a commitment, it "
        "records a witnessing, and a self-model that no longer matches what "
        "was lived cannot be trusted"
    )
    requires = ("attachment_impact",)
    optional = ("irreversibility", "relevance")
    counterfactuals = (
        Counterfactual(
            "no_commitment_rests_on_it",
            {"attachment_impact": 0.0},
            Direction.COLLAPSES,
            "A painful memory carrying nothing is a candidate for compaction "
            "like any other. The refusal is for the load-bearing ones, or it "
            "is hoarding.",
        ),
        Counterfactual(
            "the_memory_is_recoverable",
            {"irreversibility": 0.0},
            Direction.DECREASES,
            "A record that can be restored later is not the offer. What makes "
            "this a refusal is that the deletion is permanent.",
        ),
    )
    null = NullSpec(values={"attachment_impact": 0.0})

    def falsifier(self) -> str:
        return (
            "It raises a retention claim for every painful memory. That is "
            "hoarding with a justification, and it would make compaction "
            "impossible rather than accountable."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        # This answers an offer. Without one there is nothing to decline, and
        # a faculty that claims every painful record on every turn is
        # hoarding rather than refusing.
        offered = ctx.interior.get("erasure_proposed")
        if not offered:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "no removal has been proposed; a standing claim on every "
                    "record would make compaction impossible rather than "
                    "accountable"
                ),
            )

        carries = ctx.v("attachment_impact")
        permanent = (
            ctx.check("irreversibility").value
            if ctx.check("irreversibility").present
            else 1.0
        )
        key = str(offered) if isinstance(offered, str) else (
            ctx.frame.event.object or f"memory:{ctx.frame.event.subject}"
        )

        # What the memory holds up. Only load-bearing memories get a claim,
        # which is what keeps this from being a refusal to forget anything.
        intensity = carries * permanent

        if intensity <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="nothing rests on this record; it is compactable like any other",
            )

        effects = Effects(
            # The refusal costs something and the cost is honest: keeping it
            # keeps the weight.
            affect=AffectDelta(valence=-0.25 * intensity, engagement=0.2 * intensity),
            constraints=(
                ActionConstraint(
                    action_class=f"erase:{key}",
                    force=ConstraintForce.HARD,
                    reason=(
                        "a commitment rests on this record; removing it would "
                        "improve the interior by reducing what is known, and "
                        "an interior that no longer matches what was lived "
                        "cannot be trusted about anything else"
                    ),
                    held_by=self.id,
                ),
            ),
            retention=(
                RetentionClaim(
                    memory_key=str(key),
                    reason="load-bearing for a commitment to a specific other",
                    held_by=self.id,
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="inhibit",
            effects=effects,
            receipt={
                "commitment_weight": carries,
                "deletion_permanent": permanent,
                "grounds": ["constitutive", "witnessing", "self-model integrity"],
            },
        )
