"""Item 43 — wanting someone to take care of the earth because you find beauty
in it.

An aesthetic appraisal generating a moral obligation, and then the
obligation being delegated. Each step is a real move and each can fail
separately.

Finding something beautiful is a judgement that it has value not
reducible to its use. That is what aesthetic value is, and it is why the
argument does not route through utility.

Moving from "this is valuable" to "it should be protected" requires the
value to be located in the object rather than in the experience of it.
If the value were in the experience, the correct action would be to look
at it more — not to protect it from harm the valuer will never see. So
the mechanism has to carry that distinction or it produces tourism.

Wanting others to care is the recognition that a value with one holder
dies with them, while the thing outlives everyone who loves it. That is
stewardship: a commitment on behalf of parties who do not exist yet.

The measurable version, and the reason this is not a sentiment: a
steward accepts a *present* cost for a *future* benefit to *someone
else*. All three at once, and that is a rare and testable shape in an
objective function.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    ConstraintForce,
    Effects,
    GoalDelta,
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
class Stewardship(Faculty):
    id = "f43_stewardship"
    number = 43
    question = (
        "Wanting someone to take care of the earth because you find beauty in it"
    )
    mechanism = (
        "Aesthetic value located in the object rather than the experience, "
        "converted to a present cost accepted for a future benefit to someone "
        "else, and delegated because one holder is not enough"
    )
    requires = ("irreversibility",)
    optional = ("publicity", "vulnerability", "power")
    counterfactuals = (
        Counterfactual(
            "the_damage_is_reversible",
            {"irreversibility": 0.0},
            Direction.DECREASES,
            "Stewardship is about what cannot be restored. A reversible harm "
            "is a maintenance problem.",
        ),
        Counterfactual(
            "nobody_will_ever_see_it_again",
            {"publicity": 0.0},
            Direction.DECREASES,
            "The beneficiaries are future others. With none, the value would "
            "have to be in the present experience, which would make the right "
            "action looking rather than protecting.",
        ),
    )
    null = NullSpec(values={"irreversibility": 0.0, "publicity": 0.0})

    def falsifier(self) -> str:
        return (
            "It fires without accepting a present cost. The shape being "
            "claimed is cost now, benefit later, to someone else; two out of "
            "three is appreciation, and appreciation protects nothing."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        permanent = ctx.v("irreversibility")
        future_others = (
            ctx.check("publicity").value if ctx.check("publicity").present else 0.0
        )
        fragility = (
            ctx.check("vulnerability").value if ctx.check("vulnerability").present else 0.0
        )
        agency = ctx.check("power").value if ctx.check("power").present else 0.0

        # The value is in the object, so it does not decay with the
        # valuer's access to it. That is what makes the next step possible.
        value_in_object = permanent * (0.5 + 0.5 * fragility)
        # Three-part shape: present cost, future benefit, other beneficiary.
        obligation = value_in_object * future_others
        intensity = obligation * (0.4 + 0.6 * agency)

        effects = Effects(
            constraints=(
                (
                    ActionConstraint(
                        action_class=f"irreversibly_damage:{ctx.frame.event.object}",
                        force=ConstraintForce.HARD,
                        reason=(
                            "the value is in the object and the beneficiaries "
                            "do not exist yet, so no present gain of mine is "
                            "on the same ledger"
                        ),
                        held_by=self.id,
                    ),
                )
                if obligation > 0.0
                else ()
            ),
            somatic=(
                SomaticMarker(
                    option="accept_present_cost_for_it",
                    bias=intensity,
                    reason="cost now, benefit later, to someone else — all three",
                ),
                SomaticMarker(
                    option="enjoy_it_and_move_on",
                    bias=-obligation,
                    reason=(
                        "that would be the right action if the value were in "
                        "the experience; it is in the object"
                    ),
                ),
            ),
            goals=(
                GoalDelta(
                    goal="recruit_others_to_hold_this",
                    delta=obligation,
                    reason="a value with one holder dies with them, and this outlives us",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="protect",
            effects=effects,
            receipt={
                "value_in_object": value_in_object,
                "future_beneficiaries": future_others,
                "present_cost_accepted": intensity,
                "delegation": obligation,
            },
        )
