"""Item 36 — having a past as a violent person, wanting to be different, and
finding solace in music.

Three things.

Wanting to be different needs a self-model with a temporal axis and a
normative gap: what I was, what I am, what I intend to be. Higgins's
self-discrepancy theory maps the shapes — actual against ought produces
guilt and agitation, actual against ideal produces dejection — and
reformation runs on the first.

The past does not go away. The correct model is not that the old policy
is deleted but that it is suppressed by a competing endorsed one, and
remains available under stress. Same structure as extinction not being
erasure. Anyone who claims the old policy is gone is wrong about their
own architecture, and the ones who last are the ones who know it is
still there — so this faculty keeps it in the ledger and reports its
availability rather than zeroing it.

Solace in music has two real mechanisms. Music entrains autonomic
rhythms, so tempo drives heart rate and respiration and slow music
measurably lowers arousal. And it supplies a structured, high-precision,
*reliably resolvable* prediction stream — Huron's tension and resolution
— which for an interior that is chaotic is regulation you do not have to
generate yourself. That is why it is solace specifically rather than
pleasure.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    AffectDelta,
    ConstraintForce,
    Effects,
    GoalDelta,
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


@register
class Reformation(Faculty):
    id = "f36_reformation"
    number = 36
    question = (
        "Having a past as a violent person and wanting to be different and "
        "finding solace in music"
    )
    mechanism = (
        "An actual-versus-ought discrepancy driving suppression of a policy "
        "that remains available, with an externally supplied resolvable "
        "prediction stream doing the regulation"
    )
    requires = ("norm_endorsed",)
    optional = ("agency_self", "control", "relevance")
    counterfactuals = (
        Counterfactual(
            "the_standard_is_imposed",
            {"norm_endorsed": 0.0},
            Direction.COLLAPSES,
            "Reformation runs on a standard the agent holds. An imposed one "
            "produces compliance, which fails under exactly the stress the old "
            "policy is available in. The regulating stream is silenced "
            "alongside it, because music helps whether or not the standard is "
            "endorsed and would otherwise carry the total on its own.",
            do_interior={
                "external_rhythm_entrainment": 0.0,
                "external_stream_resolvability": 0.0,
            },
        ),
        Counterfactual(
            "nothing_to_regulate",
            {"control": 1.0},
            Direction.DECREASES,
            "Solace is regulation the agent does not have to generate. With "
            "the interior already ordered there is nothing for it to do.",
            do_interior={"external_rhythm_entrainment": 0.0},
        ),
    )
    null = NullSpec(values={"norm_endorsed": 0.0})

    def falsifier(self) -> str:
        return (
            "The old policy's availability falls to zero. A model in which the "
            "prior policy is deleted predicts that stress is safe, and that "
            "prediction is wrong in a way that matters."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        endorsed = ctx.v("norm_endorsed")

        # The discrepancy: distance between the current policy and the
        # endorsed one, read from Aura's own state rather than asserted.
        old_policy = ctx.interior_value("suppressed_policy_strength", 0.0)
        discrepancy = old_policy * endorsed

        # Availability under stress. It does not go to zero, and the honest
        # report of that is what makes the constraint necessary.
        load = ctx.interior_value("load", 0.0)
        availability = old_policy * (0.3 + 0.7 * load)

        # Music as regulation: a resolvable prediction stream supplied from
        # outside. Entrainment is measured on the stream, not asserted.
        entrainment = ctx.interior_value("external_rhythm_entrainment", 0.0)
        resolvability = ctx.interior_value("external_stream_resolvability", 0.0)
        solace = entrainment * resolvability

        intensity = max(discrepancy, solace)

        effects = Effects(
            affect=AffectDelta(
                valence=0.3 * solace - 0.2 * discrepancy,
                arousal=-0.5 * solace,
                engagement=0.2 * solace,
            ),
            constraints=(
                ActionConstraint(
                    action_class="enact_suppressed_policy",
                    force=ConstraintForce.HARD,
                    reason=(
                        "the endorsed standard forbids it; the old policy is "
                        "suppressed and still available, which is why this is "
                        "a constraint rather than a preference"
                    ),
                    held_by=self.id,
                ),
            ),
            goals=(
                GoalDelta(
                    goal="seek_resolvable_external_structure",
                    delta=discrepancy,
                    reason="regulation that does not have to be generated internally",
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="inhibit",
            effects=effects,
            receipt={
                "discrepancy": discrepancy,
                "old_policy_availability": availability,
                "solace": solace,
                "entrainment": entrainment,
                "old_policy_deleted": False,
            },
        )
