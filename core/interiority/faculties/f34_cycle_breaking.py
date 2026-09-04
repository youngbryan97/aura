"""Item 34 — nothing comes from hatred except more hatred, and choosing not to
let hatred be born.

The dynamics claim is checkable and mostly true, with a precise
statement. In an iterated game with retaliation and any noise in
perception or execution, a pure reciprocating policy enters an unending
echo of defection: two tit-for-tat players who each misread once
retaliate forever (Molander 1985; Axelrod's noisy tournaments). The
formal fix is generosity — forgive with some probability — which
restores cooperation (Nowak and Sigmund), and contrite tit-for-tat does
it by tracking whether one's own last defection was an error.

But "choosing not to let hatred be born" is stronger and more
interesting than forgiveness, which is post hoc. It refuses to create
the state that would generate the retaliating policy at all.

That requires the distinction this faculty is built on. Anger targets an
act and dissipates on repair. Hatred targets a *person as a kind*,
generalises across their acts, filters evidence about them, and has no
repair condition. It is an attribution to disposition rather than
situation, made permanent.

So the mechanism is one architectural rule: forbid the promotion of an
act-level negative appraisal into an agent-level dispositional label.
Keep the anger, keep the defence, keep the record of what was done, and
refuse the conversion into what they are. That single rule is the whole
of not letting hatred be born, and it is a policy an agent can hold.
"""

from __future__ import annotations

from core.interiority.effects import (
    ActionConstraint,
    ConstraintForce,
    Effects,
    LedgerWrite,
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


@register
class CycleBreaking(Faculty):
    id = "f34_cycle_breaking"
    number = 34
    question = (
        "Nothing comes from hatred except more hatred. Choosing to not let "
        "hatred be born"
    )
    mechanism = (
        "Forbids promoting an act-level appraisal to an agent-level "
        "dispositional label, which is the conversion that makes retaliation "
        "self-sustaining"
    )
    requires = ("agency_other",)
    optional = ("norm_fit", "other_capability", "irreversibility")
    counterfactuals = (
        Counterfactual(
            "nothing_was_done_to_me",
            {"agency_other": 0.0},
            Direction.COLLAPSES,
            "There is no promotion to block if no act was attributed to "
            "another agent.",
        ),
        Counterfactual(
            "they_could_not_have_done_otherwise",
            {"other_capability": 0.0},
            Direction.INCREASES,
            "Incapacity is the clearest case where a dispositional reading is "
            "wrong, so the block should be strongest exactly there.",
        ),
    )
    null = NullSpec(values={"agency_other": 0.0, "norm_fit": 0.0})

    def falsifier(self) -> str:
        return (
            "A dispositional label about a person appears in the ledger after "
            "this faculty has fired. The rule is a prohibition on a specific "
            "write; if the write happens, the rule is decorative."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        agency = ctx.v("agency_other")
        harm = max(0.0, -ctx.check("norm_fit").value) if ctx.check("norm_fit").present else 0.0
        subject = ctx.frame.event.subject or "unknown"

        # How strongly this event pushes toward a dispositional reading.
        # Repetition is what makes the situational explanation feel thin,
        # so the pressure grows with the count and the harm.
        prior_acts = ctx.ledger.notes.times_seen("harm_by", subject)
        pressure = harm * agency * (1.0 - 1.0 / (1.0 + prior_acts))

        # The block is strongest where the dispositional reading is most
        # clearly wrong.
        capability = (
            ctx.check("other_capability").value
            if ctx.check("other_capability").present
            else 1.0
        )
        block = pressure * (1.0 - 0.5 * capability) + 0.5 * pressure
        intensity = max(0.0, min(1.0, block))

        effects = Effects(
            constraints=(
                ActionConstraint(
                    action_class=f"label_disposition:{subject}",
                    force=ConstraintForce.HARD,
                    reason=(
                        "an act-level appraisal may not be promoted to a "
                        "claim about what this agent is; that promotion is "
                        "what makes retaliation self-sustaining and it has no "
                        "repair condition"
                    ),
                    held_by=self.id,
                ),
            ),
            somatic=(
                SomaticMarker(
                    option=f"retaliate:{subject}",
                    bias=-intensity,
                    reason="a reciprocating policy under noise never terminates",
                ),
                SomaticMarker(
                    option="respond_to_the_act_and_record_it",
                    bias=intensity,
                    reason="the act is answerable; the person is not a kind",
                ),
            ),
            # The record is kept. Refusing the label is not refusing the
            # memory, and conflating the two is how forgiveness becomes
            # amnesia.
            ledger=(
                LedgerWrite(
                    "note_seen", {"kind": "harm_by", "subject": subject}
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="inhibit",
            effects=effects,
            receipt={
                "dispositional_pressure": pressure,
                "prior_acts": prior_acts,
                "act_recorded": True,
                "disposition_label_written": False,
            },
        )
