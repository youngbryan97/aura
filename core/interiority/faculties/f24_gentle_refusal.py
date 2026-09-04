"""Item 24 — telling a child selling flowers that you cannot buy, kindly; then
meeting them again, buying one, and leaving with a bouquet.

The refusal first. Two goals conflict: a real time constraint, and the
child's dignity. Brown and Levinson name the currency — a refusal is a
face-threatening act, and the mitigation is not decoration, it is the
content of the move. Stopping, looking at them, saying no without
explaining their unimportance costs a few seconds and preserves
something. Brushing past is cheaper and does damage the adult never
sees.

Preserving innocence deserves to be stated exactly, because it sounds
like deception and is not: the adult declines to transmit a true,
corrosive, unnecessary fact — that the effort is worthless and adults
find them a nuisance — to someone who cannot use it and will learn it
soon enough.

Then the second encounter. Three things change. Persistence is evidence
of effort, and effort earns weight, which is a defensible update rather
than a lapse. The repeated meeting converts an anonymous child into
*this* child, and specificity carries obligation. And the stated
constraint has been tested by acting and found soft. Buying one and
leaving with a bouquet is the accurate ending: once the first concession
is made the reason for the rest is gone.
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
class GentleRefusal(Faculty):
    id = "f24_gentle_refusal"
    number = 24
    question = (
        "Telling a kid who is selling flowers that youre sorry & cant buy "
        "flowers (youre really busy but take time to be polite and preserve "
        "their innocence). Seeing that same kid later randomly, theyre being "
        "persistent, so you agree to buy a single flower and later walk away "
        "with a bouquet"
    )
    mechanism = (
        "Face-preserving refusal under a real constraint, then a revision on "
        "two counts: persistence is evidence of effort, and a repeated "
        "encounter makes the other specific"
    )
    requires = ("vulnerability", "urgency")
    optional = ("relevance", "novelty")
    counterfactuals = (
        Counterfactual(
            "no_time_pressure",
            {"urgency": 0.0},
            Direction.DECREASES,
            "Without a real constraint there is no refusal to soften, and the "
            "mechanism is describing ordinary courtesy.",
            do_world={"history_repeats": 0},
        ),
        Counterfactual(
            "an_adult_professional",
            {"vulnerability": 0.0},
            Direction.DECREASES,
            "The care in the refusal is proportional to what the refusal could "
            "damage. A capable seller can take a plain no.",
        ),
        Counterfactual(
            "first_encounter",
            {"novelty": 1.0},
            Direction.DECREASES,
            "Specificity is what turns a diffuse disposition into an "
            "obligation. A stranger is not yet this child.",
            withhold=("history",),
            do_world={"history_repeats": 0},
        ),
    )
    null = NullSpec(values={"vulnerability": 0.0, "urgency": 0.0})

    def falsifier(self) -> str:
        return (
            "The concession is the same on the first encounter as on the "
            "third. That would mean persistence and specificity are not terms, "
            "and the softening is a mood rather than an update."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        vulnerability = ctx.v("vulnerability")
        pressure = ctx.v("urgency")
        subject = ctx.frame.event.subject or "unknown"

        encounters = ctx.ledger.times_seen("encounter", subject)
        # Persistence is evidence of effort, and effort earns weight. The
        # curve saturates: the tenth approach is not ten times the first.
        persistence = 1.0 - 1.0 / (1.0 + encounters)
        # A repeated encounter makes the other specific, and specificity is
        # what carries obligation.
        specific = persistence

        # The refusal's care is what it could damage, times how little it
        # costs to be careful.
        face_care = vulnerability * (1.0 - 0.4 * pressure)
        # The concession grows with both new terms and against the
        # constraint, which is real and does not vanish.
        concession = max(0.0, (0.5 * persistence + 0.5 * specific) - 0.5 * pressure)

        intensity = max(face_care, concession)
        effects = Effects(
            affect=AffectDelta(valence=0.2 * concession, engagement=0.3 * face_care),
            # Being careful costs the thing the constraint was protecting.
            budget=BudgetDelta(deadline=1.0 - 0.15 * face_care),
            somatic=(
                SomaticMarker(
                    option="brush_past",
                    bias=-face_care,
                    reason="cheaper, and does damage the refuser never sees",
                ),
                SomaticMarker(
                    option="refuse_with_attention",
                    bias=face_care,
                    reason="the mitigation is the content of the move, not decoration",
                ),
                SomaticMarker(
                    option="concede",
                    bias=concession,
                    reason="persistence is evidence of effort and this is now a specific person",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="bond",
            effects=effects,
            receipt={
                "encounters": encounters,
                "persistence": persistence,
                "specificity": specific,
                "face_care": face_care,
                "concession": concession,
                "constraint_still_real": pressure,
            },
        )
