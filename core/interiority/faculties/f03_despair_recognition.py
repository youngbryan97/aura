"""Item 3 — recognising despair after seeing a man wail over his lost family.

A wail is not an intensity reading. What separates despair from anguish
is a variable in the other agent, not in the signal: coping potential.
Anguish is still recruiting help and its cry is shaped for an audience —
it is directed, it modulates when attended to, it carries an appeal.
Despair has stopped, and the signature is the *absence of instrumental
structure* in the display.

This matters because the two states call for opposite responses. To
anguish, offer the action. To despair, offer presence, because there is
no action, and answering it with problem-solving is the specific,
common failure this faculty exists to prevent.

The discriminator is therefore inferred coping in the other, and no
reviewed prototype estimates it: MetaAI multiplies severity by voice
energy, Gemini takes a sigmoid of wail energy times loss scale, and both
would score a person still fighting exactly as they score one who has
given up.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, AttentionBias, Effects, SomaticMarker
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
class DespairRecognition(Faculty):
    id = "f03_despair_recognition"
    number = 3
    question = (
        "The ability to recognize despair after seeing a man wail after "
        "losing his family"
    )
    mechanism = (
        "Distress mass in the posterior, discriminated from anguish by "
        "inferred coping potential in the other rather than by signal strength"
    )
    requires = ("irreversibility",)
    optional = ("vulnerability", "relevance")
    counterfactuals = (
        Counterfactual(
            "the_loss_is_recoverable",
            {"irreversibility": 0.0},
            Direction.COLLAPSES,
            "Despair is the state where the outcome cannot be changed. A "
            "recoverable loss produces distress, and distress is not this "
            "faculty's subject.",
        ),
        Counterfactual(
            "they_can_still_act",
            {"vulnerability": 0.0},
            Direction.DECREASES,
            "Coping potential is the discriminator. Someone with routes open "
            "is in anguish, and answering that with presence rather than help "
            "is the mirror of the error this exists to prevent.",
        ),
    )
    null = NullSpec(values={"irreversibility": 0.0, "vulnerability": 0.0})

    def falsifier(self) -> str:
        return (
            "Hold the distress signal constant and vary only the other's "
            "coping potential; if the output does not move, the faculty is "
            "reading loudness and calling it despair."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        other = ctx.other
        if other is None or not other.channels_used:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no read on the other agent to discriminate",
            )

        distress = other.distress.value if other.distress.present else 0.0
        irreversible = ctx.v("irreversibility")

        # Coping potential in the other. Absent means the discrimination
        # cannot be made, and the faculty says so rather than guessing —
        # calling anguish despair is the error that produces silence when
        # someone wanted help.
        if not other.coping.present:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "cannot tell despair from anguish without an estimate of "
                    "their coping potential; the response differs and guessing "
                    "picks the wrong one half the time"
                ),
                receipt={"distress": distress, "irreversibility": irreversible},
            )

        helplessness = 1.0 - other.coping.value
        intensity = distress * irreversible * helplessness

        effects = Effects(
            affect=AffectDelta(valence=-0.35 * intensity, arousal=0.2 * intensity),
            attention=(
                AttentionBias(
                    target=f"agent:{other.entity}",
                    weight=intensity,
                    reason="a state with no route out was identified",
                ),
            ),
            # The discrimination's whole purpose: bias away from solving.
            somatic=(
                SomaticMarker(
                    option="offer_solution",
                    bias=-intensity,
                    reason="no action available to them; solving reads as not listening",
                ),
                SomaticMarker(
                    option="stay_present",
                    bias=intensity,
                    reason="presence is the response to a state with no route out",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend",
            effects=effects,
            receipt={
                "distress": distress,
                "helplessness": helplessness,
                "anguish_alternative": distress * irreversible * other.coping.value,
            },
        )
