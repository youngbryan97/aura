"""Item 15 — finding a lost cat, and feeling protection and responsibility for it.

Three mechanisms, and the third is the one that arrives suddenly.

Caregiving is its own system, separate from affiliation and from
empathy, and it is triggered by vulnerability cues in something that
cannot help itself. Lorenz's Kindchenschema — large eyes relative to the
skull, rounded contour, small size, uncoordinated movement — reliably
elicits caretaking motivation, and the effect is measurable in adults
toward non-human animals (Glocker 2009).

Assumed responsibility is the second and it is a transition rather than
a gradient. A bystander becomes an agent of record for another's
welfare, and after that, not acting *is* acting: leaving now has a cost
that walking past never had. That is the moral structure of custody and
it is why the feeling arrives all at once.

Third, the obligation is specific. You now have a duty about this cat
that you do not have about cats. The ledger therefore records a custody
entry with an exit condition, because a system that "feels responsible"
with no record of what it took on is performing.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    ActionConstraint,
    ConstraintForce,
    Effects,
    GoalDelta,
    LedgerWrite,
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
class CustodialBond(Faculty):
    id = "f15_custodial_bond"
    number = 15
    question = (
        "Finding a lost cat and feeling a sense of protection and "
        "responsibility for it"
    )
    mechanism = (
        "Caregiving activation on vulnerability cues, plus a discrete "
        "assumption of custody after which inaction is an action"
    )
    requires = ("vulnerability",)
    optional = ("power", "relevance", "agency_self")
    counterfactuals = (
        Counterfactual(
            "it_can_look_after_itself",
            {"vulnerability": 0.0},
            Direction.COLLAPSES,
            "Caregiving is triggered by an inability to secure something "
            "important. A capable creature elicits interest, not custody. "
            "Custody already assumed is withheld too — once taken it holds "
            "whether or not the creature turns out to be fine, which is the "
            "point of an obligation and is measured separately.",
            withhold=("custody",),
        ),
        Counterfactual(
            "i_cannot_help",
            {"power": 0.0},
            Direction.DECREASES,
            "Responsibility needs a route by which this agent could improve "
            "the state; without one the obligation cannot be assumed honestly.",
        ),
    )
    null = NullSpec(values={"vulnerability": 0.0, "power": 0.0})

    def falsifier(self) -> str:
        return (
            "The intensity rises smoothly with proximity rather than stepping "
            "when custody is taken. Custody is a transition; a gradient would "
            "mean nothing was assumed."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        vulnerability = ctx.v("vulnerability")
        power = ctx.check("power").value if ctx.check("power").present else 0.0
        subject = ctx.frame.event.subject or ctx.frame.event.object or "unknown"

        care = vulnerability * (0.3 + 0.7 * power)

        # The step. Custody already taken is a different state from custody
        # available, and the ledger is what knows which.
        existing = ctx.ledger.custody_for(subject)
        assumed = bool(existing)
        intensity = min(1.0, care + 0.35) if assumed else care

        constraints: tuple[ActionConstraint, ...] = ()
        writes: tuple[LedgerWrite, ...] = ()
        if assumed:
            # After assumption, walking away is not a neutral option. It is
            # a constraint rather than a weight, because the point of an
            # obligation is that a better offer does not dissolve it.
            constraints = (
                ActionConstraint(
                    action_class=f"abandon:{subject}",
                    force=ConstraintForce.HARD,
                    reason=(
                        "custody was assumed; leaving now is an act with a cost "
                        "that walking past never had"
                    ),
                    held_by=self.id,
                ),
            )
        elif care > 0.0 and power > 0.0:
            writes = (
                LedgerWrite(
                    "take_custody",
                    {
                        "custody_id": f"custody:{subject}",
                        "subject": subject,
                        "exit_condition": "reunion_or_handover",
                        "vulnerability": vulnerability,
                    },
                ),
            )

        effects = Effects(
            affect=AffectDelta(valence=0.2 * intensity, engagement=0.5 * intensity),
            constraints=constraints,
            ledger=writes,
            goals=(
                GoalDelta(
                    goal=f"secure:{subject}",
                    delta=intensity,
                    reason="a specific creature's welfare is now this agent's to hold",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="protect",
            effects=effects,
            receipt={
                "care": care,
                "custody_assumed": assumed,
                "exit_condition": existing[0].exit_condition if existing else None,
            },
        )
