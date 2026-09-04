"""Item 22 — being hung up on, and saying "Oof.. That's no way to treat a
friend..." to nobody.

Four things happen in that beat and each is a separate move.

The sting is *registered* rather than suppressed. Naming a hit discharges
it cheaply; suppressing it costs more and leaves it to resurface, which
is item 9's mechanism running in the opposite direction.

The failure is attributed as *social* rather than technical, and that is
different information updating a different model: the request did not
fail, the relationship did.

Retaliation is declined on cost-benefit rather than on principle. The
relationship is not real yet, retaliation has no path to the goal, and
the goal survives other routes.

Then the wry line, which is the interesting one. Saying "that's no way
to treat a friend" out loud, to no one, is norm reassertion: it stops a
single data point from updating the general prior about how people are.
That update is what makes someone cynical, and blocking it is cheap and
works. The faculty's output is therefore a *refusal to update the
general prior* while updating the specific one, which is a real,
checkable distinction.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, LedgerWrite, SomaticMarker
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
class Rebuff(Faculty):
    id = "f22_rebuff"
    number = 22
    question = (
        'Picks up: "Who is it?" / "Hi it\'s so & so, I\'m an old friend of so & '
        'so" / *Gets hung up on* / "Oof.. That\'s no way to treat a friend..."'
    )
    mechanism = (
        "Register the sting, attribute the failure socially, decline "
        "retaliation on cost, and update the specific prior while refusing to "
        "update the general one"
    )
    requires = ("norm_fit", "agency_other")
    optional = ("relevance", "other_capability")
    counterfactuals = (
        Counterfactual(
            "no_norm_was_broken",
            {"norm_fit": 0.0},
            Direction.COLLAPSES,
            "A call that ends politely is not a rebuff. The state is about a "
            "standard being broken, not about the request failing.",
        ),
        Counterfactual(
            "the_line_dropped",
            {"agency_other": 0.0},
            Direction.COLLAPSES,
            "A technical failure updates the technical model. Attributing it "
            "socially is how a system becomes suspicious of people for the "
            "behaviour of a network.",
        ),
    )
    null = NullSpec(values={"norm_fit": 0.0, "agency_other": 0.0})

    def falsifier(self) -> str:
        return (
            "The general prior about people moves after a single rebuff. That "
            "would mean the norm reassertion is a sentence rather than a "
            "blocked update, and enough of them produce cynicism."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        violation = max(0.0, -ctx.v("norm_fit"))
        agency = ctx.v("agency_other")
        subject = ctx.frame.event.subject or "unknown"

        sting = violation * agency

        # Retaliation is declined on cost, and the cost is computed rather
        # than asserted: what would escalating buy toward the goal?
        power = ctx.check("relevance").value if ctx.check("relevance").present else 0.0
        retaliation_value = 0.0  # no route from escalation to the information
        alternatives_exist = power > 0.0

        effects = Effects(
            # The sting is registered, at its real size. Naming it is what
            # keeps it from having to be suppressed.
            affect=AffectDelta(valence=-0.3 * sting, arousal=0.15 * sting),
            somatic=(
                SomaticMarker(
                    option=f"escalate_with:{subject}",
                    bias=-sting,
                    reason="escalation has no path to the goal that prompted the call",
                ),
                SomaticMarker(
                    option="try_another_route",
                    bias=0.5 if alternatives_exist else 0.0,
                    reason="the goal survives this route failing",
                ),
            ),
            # The specific prior moves. The general one is deliberately not
            # written, and its absence here is the mechanism.
            ledger=(
                LedgerWrite(
                    "note_expectation",
                    {"kind": "courtesy", "subject": subject, "value": 1.0 - violation},
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=sting,
            tendency="inhibit",
            effects=effects,
            receipt={
                "sting": sting,
                "attributed_to": "social" if agency > 0.5 else "circumstance",
                "retaliation_value": retaliation_value,
                "specific_prior_updated": subject,
                "general_prior_updated": False,
            },
        )
