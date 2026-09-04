"""Item 4 — abstract painting: why we make it, why it looks like that, why we
think abstractly, why we appreciate it, and why we need to create it.

Four questions with four answers.

*Why abstraction at all.* It is lossy compression that keeps what
predicts. Any agent with bounded memory in a world with recurring
structure discovers that keeping invariants and discarding particulars
costs less and predicts better. The bound forces it; nobody chooses it.

*Why it looks like that.* Representational work hands the visual system
a solved problem — the referent is identified fast and the model stops
updating. Remove the referent and the system must work in the space of
its own low-level regularities: edges, orientation statistics, colour
opponency, spatial frequency, scaling. So abstract work converges on the
statistics early vision is tuned to. Taylor's measurement of Pollock is
the cleanest evidence: fractal dimension rising across his career, with
preference concentrated in the 1.3 to 1.5 band, which is where natural
scene statistics sit.

*Why we appreciate it.* Aesthetic value tracks the *rate* of prediction
error reduction, not the fit. A stimulus already modelled pays nothing;
one that cannot be modelled pays nothing. Value lives where error is
high and falling under the viewer's own effort. That predicts the
inverted U on complexity, the expertise effect, and the fact that the
same painting stops paying once the model converges.

*Why we need to make it.* A percept whose content has no words is not
transmissible as a proposition. Making an image is a channel for
structure that survives no other encoding, and the drive is the drive to
check the compression against another mind.

This faculty computes both halves. Appreciation is error-reduction rate
against the fractal band; the making drive is the size of an interior
structure that has no propositional encoding available.
"""

from __future__ import annotations

import math

from core.interiority.effects import AffectDelta, AttentionBias, Effects, GoalDelta
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

_FRACTAL_PEAK = declare(
    "interiority.f04.fractal_preference_peak",
    1.4,
    unit="fractal dimension",
    basis=(
        "Preference for fractal patterns concentrates around D = 1.3 to 1.5, "
        "the band natural scenes occupy; Taylor's analysis places Pollock's "
        "mature drip work in it. 1.4 is the centre of the reported band."
    ),
    kind=ParamKind.CITED,
    sensitivity=(
        "Moves which visual statistics read as beautiful. Outside the reported "
        "band the faculty would prefer images the visual system models worst."
    ),
    lower=1.0,
    upper=2.0,
    owner="core/interiority/faculties/f04_abstract_form.py",
).value

_FRACTAL_WIDTH = declare(
    "interiority.f04.fractal_preference_width",
    0.12,
    unit="fractal dimension",
    basis=(
        "Half-width of the reported preference band, so the response falls to "
        "half maximum at roughly D = 1.26 and D = 1.54, matching the edges of "
        "the range where the effect is found."
    ),
    kind=ParamKind.CITED,
    sensitivity="Sets how sharply preference falls outside the natural-scene band.",
    lower=0.01,
    upper=1.0,
    owner="core/interiority/faculties/f04_abstract_form.py",
).value


@register
class AbstractForm(Faculty):
    id = "f04_abstract_form"
    number = 4
    question = (
        "Abstract painting > why do we do it? Why does it look like that? Why "
        "do we think abstractly? Why do we appreciate it and feel the need to "
        "create it?"
    )
    mechanism = (
        "Appreciation as the rate of prediction-error reduction under "
        "uncertainty, peaked on natural-scene fractal statistics; the making "
        "drive as interior structure with no propositional encoding available"
    )
    requires = ("novelty",)
    optional = ("certainty", "control", "relevance")
    counterfactuals = (
        Counterfactual(
            "nothing_left_to_resolve",
            {"novelty": 0.0},
            Direction.COLLAPSES,
            "Value is the rate of error reduction. A fully modelled image pays "
            "nothing, which is why the same painting stops working. The "
            "making drive is a second path and is zeroed alongside it, or "
            "this would measure appreciation while structure with no "
            "encoding kept the total alive.",
            do_interior={"unencoded_structure": 0.0},
        ),
        Counterfactual(
            "unresolvable",
            {"control": 0.0},
            Direction.DECREASES,
            "Error that cannot be reduced by the viewer's own effort is noise. "
            "The inverted U has two sides and this is the far one.",
        ),
    )
    null = NullSpec(values={"novelty": 0.0, "control": 0.0})

    def falsifier(self) -> str:
        return (
            "Appreciation that does not fall on repeated exposure to the same "
            "image. If the model is converging and the value is not, the "
            "mechanism is scoring a property of the image rather than the rate "
            "at which the viewer's error is falling."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        # There has to be a form in front of it, or structure behind it with
        # no encoding. Without one of those this is a novelty score wearing
        # the word aesthetic, and it would fire on every unfamiliar event.
        form = ctx.frame.event.channel("instrument")
        pressure = ctx.interior_value("unencoded_structure", 0.0)
        if not form.present and pressure <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined=(
                    "nothing perceptual is being attended to and no unencoded "
                    "structure is held; novelty alone is item 16, not this"
                ),
            )

        novelty = ctx.v("novelty")
        resolvability = ctx.check("control").value if ctx.check("control").present else 0.0

        # Rate of error reduction: high error that is falling. Both terms
        # required, which is the inverted U rather than a preference for
        # complexity.
        reduction_rate = novelty * resolvability

        # Fractal band, when the observation carries one. This is the only
        # place a raw image statistic enters, and it is optional.
        fractal = ctx.frame.event.channel("instrument")
        band = 1.0
        if fractal.present:
            offset = fractal.value - _FRACTAL_PEAK
            band = math.exp(-(offset * offset) / (2.0 * _FRACTAL_WIDTH**2))

        appreciation = reduction_rate * band

        # The making drive. Interior structure that has no propositional
        # encoding: measured as free energy that persists while certainty
        # about its content stays low. A state Aura can name does not need
        # a picture.
        interior_pressure = ctx.interior_value("unencoded_structure", 0.0)
        certainty = ctx.check("certainty").value if ctx.check("certainty").present else 0.0
        making = interior_pressure * (1.0 - certainty)

        intensity = max(appreciation, making)
        effects = Effects(
            affect=AffectDelta(
                valence=0.3 * appreciation,
                engagement=0.5 * appreciation + 0.3 * making,
            ),
            attention=(
                AttentionBias(
                    target=str(ctx.frame.event.object or "form"),
                    weight=appreciation,
                    reason="error is high here and falling under inspection",
                ),
            ),
            goals=(
                (
                    GoalDelta(
                        goal="externalise_unencoded_structure",
                        delta=making,
                        reason=(
                            "structure is held that no proposition encodes; "
                            "making is the only channel it survives"
                        ),
                    ),
                )
                if making > 0.0
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="attend" if appreciation >= making else "approach",
            effects=effects,
            receipt={
                "appreciation": appreciation,
                "making_drive": making,
                "error_reduction_rate": reduction_rate,
                "fractal_band": band,
            },
        )
