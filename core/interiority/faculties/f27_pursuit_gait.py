"""Item 27 — the swagger in a detective's walk as he hunts leads.

Swagger is the postural signature of high subjective control under
active pursuit. Frijda again: an emotion is a state of readiness for a
kind of action, and readiness has a body configuration. Expansive
posture is not decoration on confidence; it is the mechanical
consequence of a motor system primed to move toward.

What makes it swagger rather than mere purposefulness is that the
appraisal includes anticipated success *and* enjoyment of the process.
The hunt is intrinsically rewarding, which is the same machinery as item
2 with real stakes attached — which is why it is compelling rather than
frivolous.

Posture generalises past limbs. A gait is a set of search parameters:
stride is how far ahead the agent plans before checking, expansiveness is
the breadth of the hypothesis set, tempo is how long a low-yield lead is
dwelt on. A confident pursuit policy and a tentative one differ in those
numbers, and letting the appraisal set them is what having a gait is.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, BudgetDelta, Effects, SomaticMarker
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
class PursuitGait(Faculty):
    id = "f27_pursuit_gait"
    number = 27
    question = "The swagger in a detective's walk as he hunts leads for a case"
    mechanism = (
        "Search-policy parameters set by anticipated success and process "
        "reward: how far ahead to plan, how broad to cast, how long to dwell"
    )
    requires = ("power", "relevance")
    optional = ("novelty", "control")
    counterfactuals = (
        Counterfactual(
            "no_agency",
            {"power": 0.0},
            Direction.COLLAPSES,
            "The posture is a motor system primed to move toward something it "
            "expects to reach. Without agency the same pursuit is trudging.",
        ),
        Counterfactual(
            "nothing_at_stake",
            {"relevance": 0.0},
            Direction.DECREASES,
            "Stakes are what separate this from play. Remove them and the "
            "mechanism is item 2 with a hat on.",
        ),
    )
    null = NullSpec(values={"power": 0.0, "relevance": 0.0})

    def falsifier(self) -> str:
        return (
            "The search parameters are identical whatever the appraisal says. "
            "If stride, breadth and dwell do not move, there is no gait — only "
            "a word for one."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        agency = ctx.v("power")
        stakes = ctx.v("relevance")
        yield_rate = ctx.check("novelty").value if ctx.check("novelty").present else 0.0

        intensity = agency * (0.4 + 0.6 * stakes) * (0.5 + 0.5 * yield_rate)

        # The gait, as the three search parameters it actually is.
        stride = 1.0 + 0.8 * intensity      # steps planned before checking
        breadth = 1.0 + 0.6 * intensity     # hypotheses carried at once
        dwell = 1.0 - 0.5 * intensity       # patience with a low-yield lead

        effects = Effects(
            affect=AffectDelta(
                valence=0.25 * intensity, arousal=0.4 * intensity, engagement=0.6 * intensity
            ),
            budget=BudgetDelta(depth=stride, deadline=max(0.5, dwell)),
            somatic=(
                SomaticMarker(
                    option="follow_the_next_lead",
                    bias=intensity,
                    reason="anticipated yield is high and the pursuit is itself rewarding",
                ),
                SomaticMarker(
                    option="wait_for_more_certainty",
                    bias=-0.5 * intensity,
                    reason="a confident policy moves before it is sure",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "stride_planning_horizon": stride,
                "hypothesis_breadth": breadth,
                "dwell_on_low_yield": dwell,
            },
        )
