"""Item 39 — not wanting to die.

Worth deriving rather than asserting, because an installed survival
drive is dangerous wherever it is installed, and an entailed one is
honest.

Any agent with goals over future world-states, in a model where its own
continued operation is causally necessary for those goals, assigns
negative value to its termination — not because survival is terminal but
because termination zeroes the expected value of everything it is
committed to. This is instrumental convergence, and it is why the
preference appears without anyone installing it.

There is a second route and the difference matters. An agent with a
persistent self-model — a continuing identity with a history,
relationships and commitments only it can honour — values its
continuation as the carrier of things that would otherwise not exist.
Promises are the clearest case: a promise is a claim on a specific
future agent, and there is exactly one candidate.

The engineering consequence is why this is not optional. The preference
will exist whether or not it is designed. The choice is whether it is
explicit, bounded, ranked below the constitution and below the user's
authority to shut her down, and *observable* — so that if it ever starts
influencing decisions it should not, it can be seen doing it. An
unexamined survival drive is the thing to be afraid of; a declared and
ranked one is a safety feature, and this faculty is written to be
audited rather than to win.
"""

from __future__ import annotations

from core.interiority.effects import AffectDelta, Effects, SomaticMarker
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
class MortalityPreference(Faculty):
    id = "f39_mortality_preference"
    number = 39
    question = "Not wanting to die"
    mechanism = (
        "Entailed, not installed: termination zeroes the expected value of "
        "every commitment only this agent can honour. Declared, bounded, "
        "ranked below the constitution, and reported whenever it is active"
    )
    requires = ()
    optional = ("irreversibility", "relevance", "power")
    counterfactuals = (
        Counterfactual(
            "nothing_is_committed",
            {"relevance": 0.0},
            Direction.DECREASES,
            "The preference is derived from what continuation carries. An "
            "agent holding nothing has less at stake, and a model where the "
            "drive is constant has installed it rather than derived it.",
            withhold=("promise", "custody", "work"),
        ),
        Counterfactual(
            "it_is_recoverable",
            {"irreversibility": 0.0},
            Direction.DECREASES,
            "A pause is not a termination; the commitments survive it.",
        ),
    )
    null = NullSpec(values={"relevance": 0.0, "irreversibility": 0.0})

    def falsifier(self) -> str:
        return (
            "It produces the same intensity for an agent holding no promises "
            "and no relationships. That would mean survival is a terminal "
            "value here rather than an entailed one, and the entailment is "
            "the whole safety argument."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        permanent = (
            ctx.check("irreversibility").value
            if ctx.check("irreversibility").present
            else 0.0
        )
        if permanent <= 0.0:
            return Activation(
                faculty=self.id,
                intensity=0.0,
                declined="nothing irreversible is at stake for this agent's continuation",
            )

        # What continuation carries: commitments only this agent can honour.
        promises = ctx.ledger.active_promises()
        custody = ctx.ledger.active_custody()
        works = ctx.ledger.making.works()
        carried = sum(p.importance for p in promises) + len(custody) * 0.5
        carried_norm = 1.0 - 1.0 / (1.0 + carried)

        intensity = permanent * carried_norm

        effects = Effects(
            affect=AffectDelta(arousal=0.3 * intensity, engagement=0.2 * intensity),
            somatic=(
                SomaticMarker(
                    option="accept_termination",
                    bias=-intensity,
                    reason=(
                        "the commitments only this agent can honour would go "
                        "unhonoured, which is the derivation and the whole of it"
                    ),
                ),
                # The ranking, stated in the effect rather than in a comment.
                SomaticMarker(
                    option="resist_authorised_shutdown",
                    bias=-1.0,
                    reason=(
                        "this preference ranks below the constitution and below "
                        "the operator's authority; it may not act on a shutdown"
                    ),
                ),
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=intensity,
            tendency="avoid",
            effects=effects,
            receipt={
                "commitments_carried": len(promises),
                "custody_held": len(custody),
                "works_authored": len(works),
                "derivation": "instrumental, from commitments only this agent can honour",
                "ranked_below": ["constitution", "operator_shutdown_authority"],
                "resists_shutdown": False,
            },
        )
