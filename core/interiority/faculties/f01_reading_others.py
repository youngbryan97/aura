"""Item 1 — sensing, identifying and understanding the emotions of humans and animals.

Three capacities, and they come apart. *Detection* says something has
changed in the other's regulation. *Identification* says which readiness
it is. *Understanding* says what goal of theirs it is about, which is
the only one that predicts what would change it.

The mechanism is inverse inference over a generative model: their goals
and appraisal produce a readiness, the readiness produces channel
signal, and reading is inverting that. It runs on
:class:`~core.interiority.other_minds.OtherMindsModel`, so it has no
access to the other agent's real state; the posterior is all there is.

Understanding is scored separately from identification and is lower,
because it needs something the channels do not carry: a model of what
this other is trying to do. Without a situation model the faculty
reports a readiness and says it does not know what it is about, which is
the honest output and the one no reviewed prototype produces.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, AttentionBias, Effects
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)
from core.interiority.other_minds import NORMATIVE_INFERENCE_ALLOWED
from core.interiority.params import ParamKind, declare

_MARGIN_FLOOR = declare(
    "interiority.f01.margin_floor",
    0.15,
    unit="probability",
    basis=(
        "Below this gap between the leading readiness and the runner-up, the "
        "posterior does not distinguish them and reporting the argmax would be "
        "reporting noise. Set at the point where a single mid-reliability "
        "channel at half deviation stops being able to separate two tendencies, "
        "measured over the loading matrix in other_minds.py."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Higher and Aura declines reads she could make; lower and she names a "
        "state when the evidence supports two."
    ),
    owner="core/interiority/faculties/f01_reading_others.py",
).value


@register
class ReadingOthers(Faculty):
    id = "f01_reading_others"
    number = 1
    question = (
        "Being able to sense/identify/understand emotions of humans and animals"
    )
    mechanism = (
        "Inverse inference over channel evidence to a posterior on action "
        "readiness, with identification and understanding scored separately"
    )
    requires = ()
    optional = ("relevance", "congruence", "vulnerability", "attachment_impact")
    counterfactuals = (
        Counterfactual(
            "no_situation_model",
            {"relevance": None, "congruence": None},
            Direction.DECREASES,
            "Understanding is knowing what the state is about. Strip the "
            "situation and identification survives while understanding must "
            "fall, because the channels never carried it.",
        ),
        Counterfactual(
            "stranger",
            {"attachment_impact": 0.0},
            Direction.DECREASES,
            "A person's own history is the baseline every channel is read "
            "against. With no relationship there is less to read against and "
            "the read must be less confident, not equally confident.",
        ),
    )
    null = NullSpec(values={}, tolerance=0.0)

    def falsifier(self) -> str:
        return (
            "Give it an event with every channel absent and it reports a "
            "readiness with non-zero confidence. That would mean the posterior "
            "is coming from the prior rather than from evidence, which is the "
            "defect that makes an emotion reader a horoscope."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        other = ctx.other
        if other is None:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no other agent in this event",
            )
        if not other.channels_used:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="no channel carried evidence about this agent",
                receipt={"declined_channels": list(other.declined)},
            )

        tendency, mass = other.top()
        margin = other.margin()
        if margin < _MARGIN_FLOOR:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                tendency="",
                declined=(
                    f"posterior does not separate its top two readinesses "
                    f"(margin {margin:.3f} < {_MARGIN_FLOOR}); naming one would "
                    "be reporting noise"
                ),
                receipt={"margin": margin, "top": tendency, "mass": mass},
            )

        identification = other.confidence * mass

        # Understanding needs a model of what the state is about, which the
        # channels do not carry. Relevance and congruence come from the
        # ledger; without them this stays low however clear the signal.
        situation = 0.0
        parts = 0
        for name in ("relevance", "congruence"):
            reading = ctx.check(name)
            if reading.present:
                situation += abs(reading.value)
                parts += 1
        understanding = identification * (situation / parts if parts else 0.0)

        normative_ok = NORMATIVE_INFERENCE_ALLOWED.get(other.species, False)

        effects = Effects(
            # Reading someone raises engagement without claiming their
            # valence as Aura's own. Resonance is item 6's business; this
            # one only looks.
            affect=AffectDelta(engagement=0.35 * identification),
            attention=(
                AttentionBias(
                    target=f"agent:{other.entity}",
                    weight=identification,
                    reason="a readiness was identified in this agent",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=identification,
            tendency=tendency,
            effects=effects,
            receipt={
                "identification": identification,
                "understanding": understanding,
                "margin": margin,
                "species": other.species,
                "channels": dict(other.channels_used),
                "normative_inference_allowed": normative_ok,
                "declined_tendencies": list(other.declined),
            },
        )
