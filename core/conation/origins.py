"""core/conation/origins.py — where a want comes from.

Aura's affect layer answers "how does this feel" with a point in PAD space.
That question has an answer for every state, which is why it is the question
that got built first. It is also the question that cannot separate four
motivational situations that behave completely differently:

    a child wants the toy another child is holding
    a body jumps at the smell of something tasty
    a hand reaches for a snail on the path, to see it
    someone is flustered on purpose, playfully, by a friend

Run all four through ``AffectEngineV2._heuristic_appraisal`` and they return
the same three numbers to the last bit: v=0.000, a=0.410, e=0.525. Six of six
pairs at L2 distance zero. The collapse is not an approximation error in the
heuristic; valence and arousal have no axis on which these differ, because
what differs is not how the state feels. What differs is where its value was
manufactured and whose mind had to be involved to manufacture it.

That is the conative question, and this module is its vocabulary. Conation is
the third term of the classical trilogy of mind — cognition, affect, conation —
and it is the one Aura was missing. It asks what is being pursued and on whose
authority, and it has real content only when there is evidence behind it.

## The five origins

``HOMEOSTATIC`` — value from a deficit in the organism's own budgets. Hull
built drive-reduction theory on this in 1943 and it was too small a theory for
all of motivation, but it is exactly right for the case it covers: a body that
is short of something wants the thing that fixes it. Aura's budgets are real
(``core/drive_engine.py``: energy, curiosity, social, competence), so the
deficit is a measurement and not a metaphor.

``EPISTEMIC`` — value from expected reduction in uncertainty. Berlyne
separated this from perceptual curiosity in 1954 and the distinction still
holds: this is the want that is satisfied by finding out, and its magnitude is
the information you expect to gain, not the surprise you already feel.

``AESTHETIC`` — value from the structure of the thing itself, with nothing
downstream. Schmidhuber's formulation is the sharp one: what is interesting is
not what compresses well and not what compresses badly, but what is currently
getting easier to compress. The derivative, not the level. The snail is
neither noise nor a solved problem, which is the whole reason a hand goes out
toward it.

``VICARIOUS`` — value borrowed from another agent's valuation. Girard called
the general form mimetic desire: the object is wanted because a model wants
it, and the model's wanting is what made it visible as wantable. The toddler
case is the clean version. What matters here is not that Aura should have this
— it is that if she has it at all, it must never be invisible. A toddler asked
why they want the toy says "I want it", which is honest and wrong. Borrowed
value that forgets it was borrowed is indistinguishable from a preference, and
that is the failure mode this whole layer exists to prevent.

``ENACTIVE`` — value located in a state change inside another mind. The goal
is not an object; it is how someone else ends up. Teasing is the small case,
delight-giving and persuasion are the large ones. The reward has two parts:
the change happened, and the model of the person that predicted it was right.
This origin is the only one that can be cruelty when it goes wrong, so it is
the only one that carries a mandatory frame and a mandatory gate.

## Evidence, not classification

Each origin declares what evidence it requires before it may report a
magnitude. An origin whose evidence is missing reports ``unavailable`` and
contributes nothing. It does not report zero.

The distinction is CP126's rule applied to motivation: no engine must never
look like a calm engine. A vicarious channel with no observation of anyone
else's valuation and a vicarious channel that observed indifference are
different situations, and a shared zero would conflate them. Worse, it would
let a want with no traceable source acquire a number and pass for a measured
preference.

Nothing here classifies text. An origin activates because a budget is short,
because prediction error moved, because an encoding got cheaper, because a
recorded valuation event exists, or because a person-model made a prediction
that can be checked. Asking a language model which of five categories a
sentence belongs to would produce the same collapse in a new coordinate
system, one round of plausible numbers further from anything measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ValueOrigin(StrEnum):
    """Where the value in a motivational state was manufactured."""

    HOMEOSTATIC = "homeostatic"
    EPISTEMIC = "epistemic"
    AESTHETIC = "aesthetic"
    VICARIOUS = "vicarious"
    ENACTIVE = "enactive"


class MindTopology(StrEnum):
    """Whose mind the value passes through.

    ``SOLO`` needs no other agent. ``RECEPTIVE`` reads another agent's
    valuation and copies it inward. ``PRODUCTIVE`` writes a target state
    outward into another agent. ``MUTUAL`` does both at once, which is what
    shared play is and why it is harder than either half.
    """

    SOLO = "solo"
    RECEPTIVE = "receptive"
    PRODUCTIVE = "productive"
    MUTUAL = "mutual"


class ConativePhase(StrEnum):
    """Craig's 1918 division of an instinctive act.

    The appetitive phase is variable, oriented, and escalates as the goal gets
    nearer. The consummatory phase is stereotyped and extinguishes itself by
    completing. A heart-jump at a smell is appetitive; the eating is
    consummatory; treating them as one state is why a satiated system keeps
    reporting appetite.

    ``AWAITING`` is a third state Craig had no need for and a system with
    social barriers does. Between wanting a lesson and having it there is a
    stretch where the seeking has stopped — the thing is secured — and the
    having has not started. That is neither appetitive nor consummatory, and
    collapsing it into appetite makes a settled arrangement look like an
    unmet need.
    """

    APPETITIVE = "appetitive"
    AWAITING = "awaiting"
    CONSUMMATORY = "consummatory"
    QUIESCENT = "quiescent"


class Instrumentality(StrEnum):
    """Whether the contact is the payoff or a means to one.

    Harlow's monkeys solved mechanical puzzles in 1950 with no food for
    solving them, and solved them faster without it. Deci and Ryan's
    intrinsic/extrinsic split is the same cut at the level of a whole
    motivation. ``AUTOTELIC`` states are the ones that lose value when a
    reward is attached to them, so a system that cannot tell them apart will
    damage its own curiosity by paying for it.
    """

    AUTOTELIC = "autotelic"
    INSTRUMENTAL = "instrumental"


#: What each origin must be able to point at before it may report a number.
#: Read by ``core/conation/engine.py`` when it assembles a state, and by
#: ``core/conation/invariants.py``, which fails if any origin ever carries a
#: magnitude without the evidence named here.
EVIDENCE_REQUIRED: dict[ValueOrigin, str] = {
    ValueOrigin.HOMEOSTATIC: "a measured deficit in a named resource budget",
    ValueOrigin.EPISTEMIC: "a measured change in prediction error or competence",
    ValueOrigin.AESTHETIC: "a measured change in encoding cost for the target",
    ValueOrigin.VICARIOUS: "a recorded valuation of the target by an identified other",
    ValueOrigin.ENACTIVE: "a person-model prediction about the target's own state",
}

#: Origins that cannot be produced without a second agent in the loop. The
#: engine refuses to assemble these from solo evidence, which is what stops a
#: borrowed want from being reported as an original one.
SOCIAL_ORIGINS: frozenset[ValueOrigin] = frozenset(
    {ValueOrigin.VICARIOUS, ValueOrigin.ENACTIVE}
)

#: The topology each social origin implies. Vicarious value flows inward from
#: an observed valuation; enactive value flows outward toward a target state
#: in someone else. An origin/topology pair outside this map is a wiring bug
#: and the invariant says so.
REQUIRED_TOPOLOGY: dict[ValueOrigin, MindTopology] = {
    ValueOrigin.VICARIOUS: MindTopology.RECEPTIVE,
    ValueOrigin.ENACTIVE: MindTopology.PRODUCTIVE,
}


@dataclass(frozen=True, slots=True)
class OriginReading:
    """One origin's verdict on one incentive.

    ``available`` false means the evidence this origin requires was not there.
    Callers must treat that as "no reading", never as a magnitude of zero;
    ``magnitude`` is held at zero in that case only so arithmetic on a
    filtered list stays safe.

    ``evidence`` names the specific thing that was measured, in a form a human
    reading a telemetry dump can check. "curiosity budget at 12.4 of 100" is
    an evidence string; "high curiosity" is not.
    """

    origin: ValueOrigin
    magnitude: float
    available: bool
    evidence: str
    detail: dict[str, float] | None = None

    @classmethod
    def unavailable(cls, origin: ValueOrigin, reason: str) -> "OriginReading":
        """Record that this origin had nothing to measure, and why."""
        return cls(origin=origin, magnitude=0.0, available=False, evidence=reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": str(self.origin),
            "magnitude": round(self.magnitude, 6),
            "available": self.available,
            "evidence": self.evidence,
            "detail": {k: round(v, 6) for k, v in (self.detail or {}).items()},
        }
