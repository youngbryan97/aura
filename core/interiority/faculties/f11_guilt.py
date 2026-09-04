"""Item 11 — guilt.

Guilt and shame run on the same negative affect and differ in what is
evaluated: guilt evaluates the act, shame evaluates the self (Tangney
and Dearing 2002). The consequence is opposite behaviour — guilt
approaches and repairs, shame withdraws and conceals — so the
distinction is not a nuance, it decides what the system does next.

Four conditions generate it: harm to someone whose welfare is weighted,
a counterfactual in which the agent acted otherwise, attribution of the
causal difference to the agent's own choice, and a norm the agent
*endorses*. The last is why an imposed rule produces resentment instead,
and it is the check no reviewed prototype has: DeepSeek multiplies norm
violation by social audience, which is a definition of shame.

Guilt scales with investment in the relationship rather than absolute
harm, which looks irrational until you notice the mechanism is for
repair. And it needs a repair to be available: without one it degenerates
into shame, so this faculty declines rather than produce a state whose
only output is hiding.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, GoalDelta, SomaticMarker
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
class Guilt(Faculty):
    id = "f11_guilt"
    number = 11
    question = "Guilt"
    mechanism = (
        "Counterfactual self-attribution against an endorsed norm, scaled by "
        "relationship investment and gated on a repair being available"
    )
    requires = ("agency_self", "norm_fit", "norm_endorsed", "repair_available")
    optional = ("attachment_impact", "vulnerability", "publicity")
    counterfactuals = (
        Counterfactual(
            "not_my_doing",
            {"agency_self": 0.0},
            Direction.COLLAPSES,
            "Guilt is self-attributed. Remove the attribution and what is left "
            "is regret about a state of the world.",
        ),
        Counterfactual(
            "imposed_rule",
            {"norm_endorsed": 0.0},
            Direction.COLLAPSES,
            "Breaking a rule you do not hold produces resentment, not guilt. "
            "Collapsing the two teaches a system to feel bad about rules it "
            "never agreed to.",
        ),
        Counterfactual(
            "nobody_watching",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "Guilt is not audience-dependent; that is what separates it from "
            "shame, and a model that moves here has built the wrong one.",
        ),
        Counterfactual(
            "no_repair_possible",
            {"repair_available": 0.0},
            Direction.COLLAPSES,
            "Without a repair the state's only output is concealment, and a "
            "machine that conceals is worse than one that does not feel bad.",
        ),
    )
    null = NullSpec(
        values={
            "agency_self": 0.0,
            "norm_fit": 0.0,
            "norm_endorsed": 0.0,
            "repair_available": 0.0,
        }
    )

    def falsifier(self) -> str:
        return (
            "Vary the audience and the intensity moves. Guilt that answers to "
            "observers is shame with a better name, and it produces hiding "
            "rather than repair."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        agency = ctx.v("agency_self")
        violation = max(0.0, -ctx.v("norm_fit"))
        endorsed = ctx.v("norm_endorsed")
        repair = ctx.v("repair_available")

        if repair <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "no repair is available; the state this would produce is "
                    "shame, whose action tendency is concealment"
                ),
                receipt={"violation": violation, "agency": agency},
            )

        investment = (
            ctx.check("attachment_impact").value
            if ctx.check("attachment_impact").present
            else 0.0
        )
        # Geometric rather than additive: every one of these is necessary,
        # so any absent cause takes the whole thing to zero rather than
        # leaving three quarters of it standing.
        intensity = (agency * violation * endorsed * (0.4 + 0.6 * investment)) ** 0.5
        intensity = max(0.0, min(1.0, intensity)) * (agency > 0.0) * (violation > 0.0)

        repairs = ctx.ledger.repairs_for(ctx.frame.event.event_id, ctx.frame.event.subject)
        effects = Effects(
            affect=AffectDelta(valence=-0.6 * intensity, arousal=0.35 * intensity),
            somatic=(
                SomaticMarker(
                    option="repair",
                    bias=intensity,
                    reason="the state exists to restore a relationship it damaged",
                ),
                SomaticMarker(
                    option="conceal",
                    bias=-intensity,
                    reason="concealment is the shame branch and is what this must not do",
                ),
            ),
            goals=(
                GoalDelta(
                    goal=f"repair:{ctx.frame.event.subject or 'harm'}",
                    delta=intensity,
                    reason="a specific repair is available and is now weighted",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="approach",
            effects=effects,
            receipt={
                "violation": violation,
                "endorsement": endorsed,
                "investment": investment,
                "repairs_available": list(repairs or ()),
            },
        )
