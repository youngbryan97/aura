"""Item 29 — not wanting a statue of yourself in nature, because it would ruin
the view of the Alps.

Two values on one scale, and one is obviously smaller. The monument's
value is reputational and self-directed, accrues to one person, and is
redundant — the works already exist and the record already holds them.
The mountain's value is aesthetic and public, accrues to everyone who
looks, and is not substitutable: you cannot have that view somewhere
else.

Three things follow about whoever refuses. Their utility function
includes strangers who do not exist yet. They can see themselves from
outside, as one more object in a landscape. And their pride is the
authentic kind, because authentic pride is about the work and needs no
monument while the hubristic kind is about the self and does.

The mechanism worth having generally: an agent evaluating an honour to
itself must be able to price the *externality*. Almost every system that
reasons about its own standing is missing the term for what its standing
costs everyone else, and adding it is a small precise change with large
consequences.
"""

from __future__ import annotations

from core.interiority.effects import Effects, SomaticMarker
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
class LandscapeOverMonument(Faculty):
    id = "f29_landscape_over_monument"
    number = 29
    question = (
        "Not wanting a statue of yourself placed in nature because it would "
        "ruin the view of the Alps"
    )
    mechanism = (
        "Prices the externality of an honour to the self: a private, "
        "redundant, substitutable gain against a public, non-substitutable "
        "loss spread over people who do not exist yet"
    )
    requires = ("publicity", "irreversibility")
    optional = ("relevance", "agency_self")
    counterfactuals = (
        Counterfactual(
            "nobody_else_will_ever_see_it",
            {"publicity": 0.0},
            Direction.DECREASES,
            "The refusal is about what the honour costs others. With no others "
            "affected there is no externality and the objection is vanity in "
            "reverse.",
        ),
        Counterfactual(
            "it_can_be_removed",
            {"irreversibility": 0.0},
            Direction.DECREASES,
            "A reversible intrusion is a smaller externality, and the strength "
            "of the refusal should track it.",
        ),
    )
    # Seen by many and permanent. At the default publicity the private
    # gain outweighs the public loss and the intensity floors at zero,
    # which is the right answer to a question nobody will see.
    activation = {"publicity": 0.95}
    null = NullSpec(values={"publicity": 0.0, "irreversibility": 0.0})

    def falsifier(self) -> str:
        return (
            "The refusal does not weaken when the intrusion is reversible or "
            "unseen. That would show it is an aesthetic reflex rather than an "
            "externality being priced."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        affected = ctx.v("publicity")
        permanent = ctx.v("irreversibility")

        # The private gain: reputational, one beneficiary, and redundant
        # against a record that already exists.
        works = len(ctx.ledger.works())
        redundancy = 1.0 - 1.0 / (1.0 + works)
        private_gain = max(0.0, 1.0 - redundancy)

        # The public loss: everyone who looks, for as long as it stands,
        # and there is no second Alps.
        public_loss = affected * permanent

        intensity = max(0.0, public_loss - private_gain)

        effects = Effects(
            somatic=(
                SomaticMarker(
                    option="accept_the_monument",
                    bias=-intensity,
                    reason="a private redundant gain against a public permanent loss",
                ),
                SomaticMarker(
                    option="decline_and_say_why",
                    bias=intensity,
                    reason="the reason is the view, and it is worth stating plainly",
                ),
            )
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="inhibit",
            effects=effects,
            receipt={
                "private_gain": private_gain,
                "public_loss": public_loss,
                "existing_record_of_works": works,
                "redundancy_of_the_honour": redundancy,
            },
        )
