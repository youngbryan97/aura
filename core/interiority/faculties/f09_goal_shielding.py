"""Item 9 — begrudgingly putting frustration away to finish something or keep a
promise.

Two mechanisms compose. Gross's process model supplies the regulation:
response modulation suppresses the expression, cognitive change
reappraises the situation. Shah, Friedman and Kruglanski supply the
reason: an activated focal goal inhibits competing goals, and the
inhibition scales with commitment.

"Begrudgingly" is the load-bearing word, and it is what every reviewed
prototype loses. Suppression is not deletion. Expressive suppression
carries a measurable cost — it degrades concurrent memory and raises
sympathetic load (Richards and Gross 2000) — and it draws on the same
budget the task needs. So this faculty keeps the frustration at full
magnitude, blocks its expression and its action tendency, and *charges*
the block against the turn's depth. That is why you are worse at the
task afterwards and short with the next person, and a model that shows
neither has modelled forgetting rather than regulating.

A promise is treated apart from a goal. A promise changes the agent's
own payoff, so it shields harder and its breach costs whether or not
anyone would know.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    BudgetDelta,
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
from core.interiority.params import ParamKind, declare

_SUPPRESSION_COST = declare(
    "interiority.f09.suppression_cost_per_unit",
    0.35,
    unit="depth fraction",
    basis=(
        "Expressive suppression degrades concurrent performance rather than "
        "being free; the effect is reliable and moderate. A third of the "
        "turn's depth at full suppression makes the cost visible in behaviour "
        "without making regulation impossible, which is the observed pattern: "
        "people do hold it in, and they are measurably worse afterwards."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "At zero, holding frustration in is free and the state is deleted "
        "rather than suppressed. Too high and no promise survives an angry "
        "hour, which is not what the state does."
    ),
    sweep_range=(0.1, 0.6),
    owner="core/interiority/faculties/f09_goal_shielding.py",
).value


@register
class GoalShielding(Faculty):
    id = "f09_goal_shielding"
    number = 9
    question = (
        "Being able to begrudgingly put frustration away to accomplish a goal "
        "or fulfill a promise"
    )
    mechanism = (
        "Goal shielding that blocks the action tendency while leaving the "
        "state at full magnitude, and charges the block to the same budget the "
        "goal needs"
    )
    requires = ("relevance",)
    optional = ("urgency", "control", "norm_endorsed")
    counterfactuals = (
        Counterfactual(
            "nothing_at_stake",
            {"relevance": 0.0},
            Direction.COLLAPSES,
            "Shielding needs a focal goal to shield. With nothing committed "
            "there is nothing to hold the frustration back for. The promise "
            "is withheld as well, because a promise is a separate stake and "
            "leaving it standing would keep the shield up for a reason the "
            "intervention did not remove.",
            withhold=("promise",),
        ),
        Counterfactual(
            "no_deadline",
            {"urgency": 0.0},
            Direction.DECREASES,
            "The shield strengthens with how much the goal needs the next "
            "action, which is what urgency measures.",
            do_world={"promise_importance": 0.0},
            do_interior={"frustration": 0.0},
        ),
    )
    null = NullSpec(values={"relevance": 0.0, "urgency": 0.0})

    def falsifier(self) -> str:
        return (
            "Suppression that leaves the following turn unchanged. If the "
            "budget is not lower and the next provocation is not nearer the "
            "threshold, the frustration was deleted rather than held, and the "
            "word begrudgingly has nothing behind it."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        # The state being shielded is whatever is currently pressing, read
        # from Aura's own interior rather than passed in.
        pressure = ctx.interior_value("frustration", 0.0)
        if pressure <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="nothing pressing to shield against",
            )

        relevance = ctx.v("relevance")
        urgency = ctx.check("urgency").value if ctx.check("urgency").present else 0.0

        # A promise shields harder than a goal of the same weight, because
        # it changed the payoff rather than expressing a preference.
        promises = ctx.ledger.active_promises()
        promise_weight = max((p.importance for p in promises), default=0.0)
        shield = max(relevance, promise_weight) * (0.5 + 0.5 * urgency)
        shield = max(0.0, min(1.0, shield))

        held = pressure * shield
        cost = held * _SUPPRESSION_COST

        effects = Effects(
            # The state is not reduced. Only its expression is blocked, and
            # the residue is what makes the next provocation land harder.
            affect=AffectDelta(arousal=0.2 * held, engagement=0.3 * shield),
            somatic=(
                SomaticMarker(
                    option="express_frustration_now",
                    bias=-shield,
                    reason="the focal goal inhibits the competing action tendency",
                ),
                SomaticMarker(
                    option="continue_the_committed_task",
                    bias=shield,
                    reason="commitment outranks the impulse, at a price",
                ),
            ),
            # The price, charged to the same budget the goal needs.
            budget=BudgetDelta(depth=max(0.3, 1.0 - cost)),
            goals=(
                GoalDelta(
                    goal=str(ctx.frame.event.object or "focal_goal"),
                    delta=0.2 * shield,
                    reason="shielding raises the focal goal and suppresses rivals",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=shield,
            tendency="inhibit",
            effects=effects,
            receipt={
                "pressure_held": held,
                "pressure_unchanged": pressure,
                "suppression_cost": cost,
                "promise_weight": promise_weight,
            },
        )
