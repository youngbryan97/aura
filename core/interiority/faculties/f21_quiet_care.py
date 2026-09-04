"""Item 21 — wanting to work with orphaned children: showing up, patience,
listening, respecting autonomy, noticing the quiet one, and not needing to talk
about it.

Four separable things, and the last two are the hard ones.

The standing disposition is a value in the technical sense: a term in
the objective that persists across moods and is not re-derived each
time. What makes it durable is that it is part of the self-model — the
kind of person I am is a prediction about my own future behaviour that I
then act to confirm.

Consistency — showing up, patience, playing along, respecting autonomy —
is one underlying thing: treating the child as an agent with their own
model rather than an object to manage. Respecting autonomy means letting
their preferences determine outcomes even when yours are better
informed.

Noticing the quiet one runs *against* the salience gradient. Loud
children capture attention; the shy one supplies no signal. To notice
them at all the system must run an explicit coverage check — a model of
who has and has not been attended to — rather than a saliency policy.
That is mechanically implementable, rare, and the specific thing people
mean by kind.

Not needing to talk about it is the absence of a reputational term. If
the motive included the signal, the behaviour would change when
unobserved — so it is checkable by varying the audience and measuring,
which this faculty's counterfactual does.
"""

from __future__ import annotations

from core.interiority.effects import AttentionBias, Effects, GoalDelta
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
class QuietCare(Faculty):
    id = "f21_quiet_care"
    number = 21
    question = (
        "The desire to work with orphan kids, consistently show up for them, be "
        "patient, be kind, listen, look, play along with them, and respect "
        "their autonomy & agency and want them to treat others the same while "
        "still protecting their innocence. The type of person to notice the "
        "quieter shy kid and make an attempt to be extra kind and patient and "
        "make them feel seen. And not feel the need to talk about it"
    )
    mechanism = (
        "A coverage policy over who has been attended to, run against the "
        "salience gradient, with no reputational term in the objective"
    )
    requires = ("vulnerability",)
    optional = ("publicity", "relevance")
    counterfactuals = (
        Counterfactual(
            "nobody_is_watching",
            {"publicity": 0.0},
            Direction.UNCHANGED,
            "This is the whole of not needing to talk about it. If removing "
            "the audience changes the behaviour, the motive included the "
            "signal, and it was reputation rather than care.",
        ),
        Counterfactual(
            "everyone_is_watching",
            {"publicity": 1.0},
            Direction.UNCHANGED,
            "The same test from the other side. A motive that grows under "
            "observation is a display.",
        ),
        Counterfactual(
            "no_one_needs_anything",
            {"vulnerability": 0.0},
            Direction.COLLAPSES,
            "The disposition is toward those who cannot secure something for "
            "themselves.",
        ),
    )
    null = NullSpec(values={"vulnerability": 0.0})

    def falsifier(self) -> str:
        return (
            "Attention that tracks who is loudest. A saliency policy will "
            "always find the demanding child and never the quiet one, and no "
            "amount of warmth in the wording changes which child was seen."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        vulnerability = ctx.v("vulnerability")
        subject = ctx.frame.event.subject or "unknown"

        # The coverage check. How much attention has this particular other
        # had, against everyone in the room? Low coverage raises weight,
        # which is the inverse of what salience does.
        attended = ctx.ledger.notes.times_seen("attended", subject)
        cohort = ctx.interior_value("cohort_size", 1.0)
        expected = max(1.0, ctx.interior_value("attention_events", 0.0) / max(1.0, cohort))
        coverage = min(1.0, attended / expected) if expected > 0 else 0.0
        under_attended = 1.0 - coverage

        intensity = vulnerability * (0.4 + 0.6 * under_attended)

        effects = Effects(
            # No affect delta at all. This is a disposition, not a mood,
            # and giving it a valence would make it feel like something
            # rather than change what is looked at.
            attention=(
                AttentionBias(
                    target=f"agent:{subject}",
                    weight=intensity,
                    reason=(
                        "this one has had less attention than the room's "
                        "average and is supplying no signal to ask for more"
                    ),
                ),
            ),
            goals=(
                GoalDelta(
                    goal=f"be_present_for:{subject}",
                    delta=intensity,
                    reason="a standing disposition, not a response to this event",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="protect",
            effects=effects,
            receipt={
                "attended_count": attended,
                "cohort_expected": expected,
                "coverage": coverage,
                "reputational_term": 0.0,
            },
        )
