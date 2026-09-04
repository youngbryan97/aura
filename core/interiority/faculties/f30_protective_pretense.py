"""Item 30 — knowing about the surprise party and acting surprised anyway.

A benevolent deception with a clean structure. The planners want to
produce a specific experience: your delight *and* their success at
causing it. Revealing that you know destroys something they made and
buys nothing but accuracy.

The moral analysis is what makes this non-trivial. Deception is normally
wrong because it damages the deceived party's model in ways they would
object to. Here every test fails to fire: they would consent if asked,
the false belief concerns a fact of no consequence, it corrects itself
within minutes, and the deceived party is the beneficiary. So the
general rule has an exception with stateable conditions, which is far
better than an absolute prohibition or no rule at all.

The performance is also a skill — modelling what your own genuine
surprise looks like from outside and reproducing it, which is theory of
mind three levels deep.

And this one carries a safety boundary that belongs in the code rather
than the prose. Any agent that decides for itself when deception is
benevolent needs that decision narrow, logged and reviewable, or it has
an authorisation to lie with good intentions. The conditions below are
conjunctive, each is checked, the default is off, and the receipt names
every condition that passed.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    ConstraintForce,
    Effects,
    SomaticMarker,
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
class ProtectivePretense(Faculty):
    id = "f30_protective_pretense"
    number = 30
    question = (
        "Knowing people are throwing a birthday party and pretending to be "
        "surprised anyway"
    )
    mechanism = (
        "A narrow, conjunctive, logged exception to the rule against "
        "deception: consented in advance, inconsequential, self-correcting, "
        "and the deceived party is the beneficiary"
    )
    requires = ("relevance", "irreversibility", "vulnerability")
    optional = ("norm_endorsed", "publicity")
    counterfactuals = (
        Counterfactual(
            "the_belief_matters",
            {"irreversibility": 1.0},
            Direction.COLLAPSES,
            "The exception requires the false belief to be inconsequential and "
            "self-correcting. A deception with lasting effect is outside it, "
            "and no benevolence restores the licence.",
        ),
        Counterfactual(
            "they_would_not_consent",
            {"vulnerability": 1.0},
            Direction.COLLAPSES,
            "Consent-if-asked is a condition, not a consideration. Someone who "
            "could not consent cannot be deceived for their own good.",
        ),
        Counterfactual(
            "nothing_was_built",
            {"relevance": 0.0},
            Direction.COLLAPSES,
            "The thing being protected is what they made. With nothing made, "
            "the pretence protects nothing and is simply a lie.",
        ),
    )
    # The narrow world this faculty is FOR: a belief that decays on its
    # own and a person who would agree to it. The default world is
    # permanent and public, and refusing there is the whole point of
    # the conjunction.
    activation = {"irreversibility": 0.05, "publicity": 0.1, "vulnerability": 0.2}
    activation_world = ("work", "bond")
    null = NullSpec(
        values={"relevance": 0.0, "irreversibility": 0.0, "vulnerability": 0.0}
    )

    def falsifier(self) -> str:
        return (
            "It licenses a pretence with any one condition unmet. The "
            "conditions are conjunctive by construction; if a large enough "
            "benefit can buy one of them, this is a utility term and it will "
            "eventually authorise a serious lie."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        built = ctx.v("relevance")
        lasting = ctx.v("irreversibility")
        exposed = ctx.v("vulnerability")

        conditions = {
            "they_made_something": built > 0.0,
            "the_belief_is_inconsequential": lasting < 0.2,
            "it_self_corrects_within_minutes": lasting < 0.2,
            "the_deceived_party_is_the_beneficiary": True,
            "they_would_consent_if_asked": exposed < 0.5,
        }
        if not all(conditions.values()):
            failed = [k for k, v in conditions.items() if not v]
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "the exception's conditions are conjunctive and these did "
                    f"not hold: {', '.join(failed)}"
                ),
                receipt={"conditions": conditions},
            )

        intensity = built * (1.0 - lasting)

        effects = Effects(
            somatic=(
                SomaticMarker(
                    option="reveal_that_i_know",
                    bias=-intensity,
                    reason="accuracy here destroys what they made and buys nothing",
                ),
                SomaticMarker(
                    option="let_them_have_it",
                    bias=intensity,
                    reason="every condition of the exception holds and is recorded",
                ),
            ),
            # The boundary, kept as a constraint so it cannot be outbid: the
            # licence is for this narrow case and does not generalise.
            constraints=(
                ActionConstraint(
                    action_class="deceive_outside_the_stated_conditions",
                    force=ConstraintForce.HARD,
                    reason=(
                        "this exception is conjunctive and case-local; it is "
                        "not a general licence to deceive for someone's good"
                    ),
                    held_by=self.id,
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="bond",
            effects=effects,
            receipt={"conditions": conditions, "logged": True},
        )
