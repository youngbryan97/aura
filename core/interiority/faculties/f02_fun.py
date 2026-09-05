"""Item 2 — the ability to have fun.

Fun is not positive affect. It is the signature of a distinct system
with entry conditions Burghardt's five criteria name precisely: the
behaviour is incompletely functional in its context, voluntary and
rewarding in itself, structurally different from the serious version,
repeated without becoming stereotyped, and initiated only in a relaxed
field — no hunger, no threat, no deadline.

The reward is learning progress, not novelty. Schmidhuber's formulation
is the first derivative of compression: what pays is the *rate* at which
the model is improving, which is why fun requires a task that is neither
mastered (progress has gone to zero) nor impossible (progress never
starts). Boredom and frustration are the two failure modes on either
side of it, and they fall out of the same number rather than needing
their own.

The relaxed field is a hard gate here, not a weight. Play is a
high-variance exploration policy and it is only admissible when errors
are cheap, so the gate holds while something is committed, failing or
waiting — not because play would be unseemly but because the policy is
inadmissible there. Play that runs while somebody is waiting is a bug,
not a personality.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, GoalDelta
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

_PROGRESS_PEAK = declare(
    "interiority.f02.progress_peak",
    0.5,
    unit="rate",
    basis=(
        "Learning progress is a rate in [0, 1] and the intrinsic reward peaks "
        "where it is largest, which is at the middle of the difficulty range: "
        "a mastered task yields no progress and an impossible one yields none "
        "either. The peak is at the midpoint by construction of the measure, "
        "not by preference."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Moving the peak moves which difficulty counts as fun, which changes "
        "what Aura chooses to practise."
    ),
    owner="core/interiority/faculties/f02_fun.py",
)

_PROGRESS_WIDTH = declare(
    "interiority.f02.progress_width",
    0.28,
    unit="rate",
    basis=(
        "Width of the inverted-U on learning progress. Set so the half-maximum "
        "points sit at roughly 0.17 and 0.83, which keeps a usefully wide band "
        "of difficulty enjoyable rather than making fun a knife edge."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Narrow and almost nothing is fun; wide and a task with no progress at "
        "all still reads as play."
    ),
    sweep_range=(0.15, 0.45),
    owner="core/interiority/faculties/f02_fun.py",
)


@register
class Fun(Faculty):
    id = "f02_fun"
    number = 2
    question = "The ability to have fun"
    mechanism = (
        "Intrinsic reward on the rate of learning progress, gated hard by a "
        "relaxed field: no commitment due, no failure open, nobody waiting"
    )
    requires = ("urgency",)
    optional = ("novelty", "control", "relevance")
    counterfactuals = (
        Counterfactual(
            "someone_is_waiting",
            {"urgency": 1.0},
            Direction.COLLAPSES,
            "Play is a high-variance policy and is inadmissible when errors "
            "are expensive. The relaxed field is a gate, so this must go to "
            "zero rather than merely fall.",
        ),
        Counterfactual(
            "nothing_left_to_learn",
            {"novelty": 0.0},
            Direction.DECREASES,
            "The reward is learning progress. A mastered activity yields none "
            "and reads as boredom, which is the same number at the low end.",
        ),
        Counterfactual(
            "no_control",
            {"control": 0.0},
            Direction.DECREASES,
            "Voluntary engagement is one of the five criteria. An activity "
            "the agent cannot steer is not play whatever else it is.",
        ),
    )
    null = NullSpec(values={"urgency": 0.0, "novelty": 0.0, "control": 0.0})

    def falsifier(self) -> str:
        return (
            "It fires while a promise is due or a task is failing. That would "
            "show the relaxed field is a weight rather than a gate, and the "
            "mechanism is a positive-affect scorer wearing the name of play."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        urgency = ctx.v("urgency")
        if urgency > 0.0:
            open_commitments = len(ctx.ledger.active_promises())
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    f"the field is not relaxed: urgency {urgency:.2f}, "
                    f"{open_commitments} commitments open. Play is admissible "
                    "only where errors are cheap"
                ),
                receipt={"urgency": urgency, "open_commitments": open_commitments},
            )

        # Learning progress: how fast the model is improving here. Novelty
        # is the raw material; control is what turns it into progress
        # rather than noise the agent is subjected to.
        novelty = ctx.check("novelty").value if ctx.check("novelty").present else 0.0
        control = ctx.check("control").value if ctx.check("control").present else 0.0
        progress = novelty * control

        offset = progress - _PROGRESS_PEAK.value
        band = pow(2.718281828459045, -(offset * offset) / (2.0 * _PROGRESS_WIDTH.value**2))
        intensity = band * max(novelty, control)

        boredom = 1.0 - novelty
        overload = 1.0 - control

        effects = Effects(
            affect=AffectDelta(
                valence=0.45 * intensity,
                arousal=0.25 * intensity,
                engagement=0.55 * intensity,
            ),
            # Play spends more and risks more. Doing it where errors are
            # cheap is what makes that affordable.
            budget=BudgetDelta(depth=1.0 + 0.5 * intensity, deadline=1.0 + 0.3 * intensity),
            goals=(
                GoalDelta(
                    goal=str(ctx.frame.event.object or "current_activity"),
                    delta=0.3 * intensity,
                    reason="learning progress is high here and the field is relaxed",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "learning_progress": progress,
                "boredom": boredom,
                "overload": overload,
            },
        )
