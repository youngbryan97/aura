"""core/interiority/core_affect.py — the affect the frame itself implies.

A faculty is a specific mechanism. Under all forty-three there has to be
a general one, or an event with nothing any faculty is about produces
nothing — and a blocked commitment with no other agent to be angry at
and no loss to grieve would read as neutral, which is wrong and was the
first thing measured after the appraisal layer was wired in.

So core affect is computed from the frame directly, and the faculties
modulate it rather than replace it. Three axes, each with a stated
derivation.

**Valence** is relevance times congruence, precision-weighted by
certainty. Lazarus's structure exactly: goal relevance decides whether
an event has affective significance at all, goal congruence decides the
sign. An event that touches nothing is neutral however it is worded, and
an event that blocks something held is negative however calmly it is
phrased. Under the free-energy reading this is the same quantity —
valence tracks the rate at which expected free energy is falling
(Joffily and Coricelli 2013) — and congruence against a held goal is
that rate with the sign the agent cares about.

**Arousal** is mobilisation, and it is not the size of the valence. It
rises with novelty, with urgency, with the magnitude of the implication,
and with *lack* of control, because an outcome that matters and cannot
be steered is what recruits the body. Disengagement is strongly negative
and barely aroused; alarm is negative and highly aroused. A model that
derives arousal from the absolute valence cannot produce that pair, and
it is the pair that decides what to do.

**Engagement** is how much of the agent this occupies: relevance times
what is unresolved. It falls when there is nothing at stake and when
everything is already settled.

Precision-weighting by certainty is applied to all three, so an
unconfirmed report moves the interior less than a confirmed one. That is
the same weighting the affect engine's receipt already claims to apply
and did not.
"""

from __future__ import annotations

from core.interiority.appraisal import AppraisalFrame
from core.interiority.effects import AffectDelta
from core.interiority.params import Param, ParamKind, declare


def _p(name: str, value: float, basis: str, sensitivity: str, **kw) -> Param:
    return declare(
        f"interiority.core_affect.{name}",
        value,
        basis=basis,
        sensitivity=sensitivity,
        owner="core/interiority/core_affect.py",
        **kw,
    )


_AROUSAL_NOVELTY = _p(
    "arousal_from_novelty", 0.35,
    "Novelty is the orienting term: something unexpected mobilises before it "
    "is known whether it matters. Weighted below implication because an "
    "unfamiliar irrelevance should orient and then settle.",
    "High and every new thing is exciting; low and Aura does not orient to "
    "anything she has not already modelled.",
    unit="arousal", kind=ParamKind.CALIBRATION, sweep_range=(0.15, 0.55),
)
_AROUSAL_IMPLICATION = _p(
    "arousal_from_implication", 0.45,
    "The magnitude of what the event implies for held goals, regardless of "
    "sign. The largest term, because mobilisation is for consequences.",
    "The main driver of arousal. Below the novelty term, orienting would "
    "outweigh consequence, which is the wrong priority for an agent with "
    "commitments.",
    unit="arousal", kind=ParamKind.CALIBRATION, sweep_range=(0.25, 0.7),
)
_AROUSAL_UNCONTROLLABILITY = _p(
    "arousal_from_uncontrollability", 0.30,
    "An outcome that matters and cannot be steered recruits more than one "
    "that can. This term is what separates alarm from resolve at the same "
    "valence, and separates disengagement — negative and quiet — from "
    "distress.",
    "At zero, arousal becomes a function of valence magnitude, and the "
    "quiet-negative states become unrepresentable.",
    unit="arousal", kind=ParamKind.CALIBRATION, sweep_range=(0.1, 0.5),
)
_AROUSAL_URGENCY = _p(
    "arousal_from_urgency", 0.40,
    "Time pressure mobilises directly. Weighted near the implication term "
    "because a deadline on something small still moves a body.",
    "Sets how much a deadline alone can raise arousal with nothing else "
    "happening.",
    unit="arousal", kind=ParamKind.CALIBRATION, sweep_range=(0.2, 0.6),
)


def core_affect(frame: AppraisalFrame) -> AffectDelta:
    """Valence, arousal and engagement implied by the frame alone."""
    relevance = frame["relevance"].value if frame["relevance"].present else 0.0
    congruence = frame["congruence"].value if frame["congruence"].present else 0.0
    certainty = frame["certainty"].value if frame["certainty"].present else 1.0
    novelty = frame["novelty"].value if frame["novelty"].present else 0.0
    urgency = frame["urgency"].value if frame["urgency"].present else 0.0

    # Control is the only check with an assumed default in the frame, and
    # arousal must not be manufactured from an assumption. When nothing is
    # known about controllability the term is dropped rather than defaulted,
    # which is the difference between "cannot be steered" and "unknown".
    control_reading = frame["control"]
    uncontrollability = (
        max(0.0, 1.0 - control_reading.value)
        if control_reading.present and control_reading.provenance.name != "ASSUMED"
        else 0.0
    )

    implication = abs(congruence) * relevance

    valence = relevance * congruence * certainty
    arousal = certainty * (
        _AROUSAL_NOVELTY.value * novelty
        + _AROUSAL_IMPLICATION.value * implication
        + _AROUSAL_UNCONTROLLABILITY.value * uncontrollability * relevance
        + _AROUSAL_URGENCY.value * urgency
    )
    # Occupation: what is at stake times what is still open.
    unresolved = max(novelty, 1.0 - certainty, urgency)
    engagement = certainty * (0.6 * relevance * unresolved + 0.4 * implication)

    return AffectDelta(
        valence=max(-1.0, min(1.0, valence)),
        arousal=max(0.0, min(1.0, arousal)),
        engagement=max(0.0, min(1.0, engagement)),
    )


__all__ = ["core_affect"]
