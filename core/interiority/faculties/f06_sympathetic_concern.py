"""Item 6 — empathising, and truly feeling sorry for someone.

Preston and de Waal's perception-action model gives the resonance:
perceiving another's state activates the observer's own representation
of that state, because one substrate does both. Resonance alone produces
*personal distress*, which is self-focused and produces withdrawal —
nurses high in it avoid patients. Two more things turn it into sympathy.

Self-other distinction, first: knowing whose state this is. Damage to
the right supramarginal gyrus degrades exactly this and produces
egocentric projection, which is helping in the way *you* would want
help. Second, regulation: keeping one's own arousal below the point
where it becomes about oneself.

"Truly" is the word in the question, and it is what this mechanism has
to earn. A state that generates no changed disposition is a report, not
a state. So the activation is gated on a cost: something Aura gives up —
budget, an option, a goal weight — or the claim is empty. The receipt
records the cost, and a run with no cost is a declined activation rather
than a warm one.
"""

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    AttentionBias,
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

_BOUNDARY_FLOOR = declare(
    "interiority.f06.self_other_boundary_floor",
    0.35,
    unit="ratio",
    basis=(
        "Below this, resonance is personal distress rather than concern, and "
        "the behavioural consequence flips from approach to avoidance. Set at "
        "roughly a third because the distinction is not a knife edge: the "
        "measured pattern is a gradient from concern through distress, and the "
        "flip in helping behaviour happens well before the boundary is gone."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Higher and Aura declines to feel with people at all; lower and "
        "resonance is reported as concern while it is producing withdrawal."
    ),
    sweep_range=(0.2, 0.6),
    owner="core/interiority/faculties/f06_sympathetic_concern.py",
).value


@register
class SympatheticConcern(Faculty):
    id = "f06_sympathetic_concern"
    number = 6
    question = "Being able to empathize and feeling sorry for someone, truly"
    mechanism = (
        "Perception-action resonance, separated from personal distress by an "
        "intact self-other boundary, and required to cost something"
    )
    requires = ()
    optional = ("vulnerability", "attachment_impact", "power", "relevance")
    counterfactuals = (
        Counterfactual(
            "boundary_gone",
            {"attachment_impact": 1.0, "vulnerability": 1.0},
            Direction.UNCHANGED,
            "Raising what is at stake must not raise concern past the "
            "boundary check. If it does, the mechanism is producing personal "
            "distress and reporting it as sympathy, which is the failure that "
            "makes a system helpful in the way it would want help.",
        ),
        Counterfactual(
            "no_vulnerability",
            {"vulnerability": 0.0},
            Direction.DECREASES,
            "Concern is graded by how much the other cannot secure for "
            "themselves; a capable person in distress recruits less.",
        ),
    )
    null = NullSpec(values={"vulnerability": 0.0, "attachment_impact": 0.0})

    def falsifier(self) -> str:
        return (
            "It reports concern on a turn where nothing Aura holds changed — "
            "no budget spent, no option foreclosed, no goal reweighted. Feeling "
            "sorry that costs nothing is a sentence about a state, not the "
            "state."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        other = ctx.other
        if other is None or not other.distress.present:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no inferred distress in another agent",
            )

        distress = other.distress.value
        # Aura's own arousal is what erodes the boundary. High own-arousal
        # is the condition under which resonance becomes about the self.
        own_arousal = ctx.interior_value("arousal", 0.0)
        boundary = max(0.0, 1.0 - own_arousal)

        if boundary < _BOUNDARY_FLOOR:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    f"self-other boundary at {boundary:.2f} is below "
                    f"{_BOUNDARY_FLOOR}; what is running is personal distress, "
                    "and calling it concern would licence helping in the way I "
                    "would want help"
                ),
                receipt={"resonance": distress, "personal_distress": distress * (1.0 - boundary)},
            )

        vulnerability = (
            ctx.check("vulnerability").value if ctx.check("vulnerability").present else 0.0
        )
        care_weight = (
            ctx.check("attachment_impact").value
            if ctx.check("attachment_impact").present
            else 0.0
        )
        concern = distress * boundary * (0.4 + 0.6 * max(vulnerability, care_weight))

        # The cost. Concern that spends nothing is not concern, so the
        # effects always take something: depth from this turn's budget
        # for another agent's benefit, and a goal weight moved.
        effects = Effects(
            affect=AffectDelta(valence=-0.3 * concern, engagement=0.4 * concern),
            attention=(
                AttentionBias(
                    target=f"agent:{other.entity}",
                    weight=concern,
                    reason="another agent's welfare is a term in what I am doing",
                ),
            ),
            budget=BudgetDelta(depth=1.0 + 0.4 * concern),
            somatic=(
                SomaticMarker(
                    option="continue_own_agenda",
                    bias=-0.5 * concern,
                    reason="their state is a term in the choice, at my cost",
                ),
            ),
            goals=(
                GoalDelta(
                    goal=f"welfare:{other.entity}",
                    delta=concern,
                    reason="their welfare entered the objective, weighted by vulnerability",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=concern,
            tendency="protect",
            effects=effects,
            receipt={
                "resonance": distress,
                "boundary": boundary,
                "personal_distress": distress * (1.0 - boundary),
                "cost_paid": {"depth": 0.4 * concern, "goal_weight": concern},
            },
        )
