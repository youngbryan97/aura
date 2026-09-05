"""core/affect/affective_resonance.py

Affective Resonance — reading the person in front of Aura.

This used to score a message against three word lists: fifteen negative
terms, twelve positive, seven "high arousal", counted and divided. The
result was a valence, an arousal and a tone string, and the tone string
was written into the context that reaches the model on every turn. A
lookup table cannot be wrong in an interesting way, carries no
uncertainty, cannot improve, and produced an instruction about how to
sound rather than information about the person.

It now runs on :mod:`core.interiority.other_minds`, which infers a
posterior over action readiness from channel evidence, read against that
person's own baseline, with per-channel reliability that moves when an
outcome is recorded. Text supplies two channels — distributional
statistics the writer is not managing, and the message's shape — and any
other channel a caller has (timing, prosody, behaviour, context) is
passed straight through and weighted higher, because those are the ones
people do not curate.

Two properties the previous version could not have. The read carries its
own confidence, so a two-word message and a paragraph do not produce
equally sure answers. And it declines: below the margin where the top
two readinesses separate, it reports that it does not know, which is the
correct output far more often than a lookup table admits.

The honesty note is kept and sharpened: attunement is modelling, not
performed affection, and the object below reports what was inferred and
how sure it is rather than telling anything how to sound.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.AffectiveResonance")

#: How the inferred readiness maps onto the valence axis the rest of the
#: runtime reads. Approach and bonding are appetitive; avoidance,
#: disengagement and confrontation are aversive; attending and inhibiting
#: are neither, which is why they are absent rather than zero-weighted.
_VALENCE_LOADING: Mapping[str, float] = {
    "approach": 0.8,
    "bond": 0.9,
    "protect": 0.4,
    "attend": 0.1,
    "avoid": -0.8,
    "disengage": -0.9,
    "agonistic": -0.5,
    "reject": -0.7,
    "submit": -0.3,
}

#: Which readinesses carry mobilisation. Disengagement is the strongest
#: negative valence and among the lowest arousal, which is the pattern a
#: single scalar cannot represent and the reason the two axes are read
#: separately.
_AROUSAL_LOADING: Mapping[str, float] = {
    "agonistic": 0.9,
    "avoid": 0.85,
    "approach": 0.6,
    "reject": 0.6,
    "attend": 0.4,
    "protect": 0.5,
    "bond": 0.35,
    "submit": 0.3,
    "inhibit": 0.25,
    "disengage": 0.1,
}


@dataclass
class Resonance:
    valence: float             # -1 (aversive) .. 1 (appetitive)
    arousal: float             # 0 (quiescent) .. 1 (mobilised)
    resonance: float           # 0 .. 1 — confidence in the read, not its strength
    recommended_tone: str      # what was inferred and how sure, not how to sound
    #: The leading action readiness, or empty when the read did not separate.
    readiness: str = ""
    #: Gap between the leading readiness and the runner-up.
    margin: float = 0.0
    #: Channels that carried evidence, with the weight each had.
    channels: dict[str, float] = field(default_factory=dict)
    #: Non-empty when the read was declined, with the reason.
    declined: str = ""
    timestamp: float = field(default_factory=time.time)


class AffectiveResonance:
    def __init__(self):
        self._reads = 0
        self._declines = 0

    def attune(
        self,
        message: str,
        *,
        user_valence: float | None = None,
        user_arousal: float | None = None,
        subject: str | None = None,
        species: str = "human",
        channels: Mapping[str, Any] | None = None,
    ) -> Resonance:
        """Infer the person's readiness from every channel available.

        ``user_valence`` and ``user_arousal`` remain honoured when a
        caller has a direct measurement, because a measurement beats an
        inference. Everything else is inferred rather than looked up.
        """
        self._reads += 1

        if user_valence is not None or user_arousal is not None:
            valence = 0.0 if user_valence is None else max(-1.0, min(1.0, user_valence))
            arousal = 0.0 if user_arousal is None else max(0.0, min(1.0, user_arousal))
            return Resonance(
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                resonance=1.0,
                recommended_tone="supplied by the caller as a direct measurement",
                readiness="",
                margin=0.0,
                channels={"instrument": 1.0},
            )

        try:
            from core.interiority.event import CHANNELS, EventKind, InteriorEvent
            from core.interiority.evidence import Reading, measured
            from core.interiority.other_minds import get_other_minds_model
            from core.interiority.text_features import channels as text_channels
        except ImportError as exc:  # interiority unavailable in this build
            from core.runtime.errors import record_degradation

            record_degradation(
                "affective_resonance", exc, action="attunement declined; no inference layer"
            )
            self._declines += 1
            return Resonance(
                valence=0.0,
                arousal=0.0,
                resonance=0.0,
                recommended_tone="no read: the inference layer is unavailable",
                declined="interiority not importable",
            )

        observations: dict[str, Reading] = dict(text_channels(message))
        for name, value in (channels or {}).items():
            if name not in CHANNELS:
                continue
            if isinstance(value, Reading):
                observations[name] = value
            elif isinstance(value, (int, float)):
                observations[name] = measured(
                    max(0.0, min(1.0, float(value))), source=f"caller:{name}"
                )

        event = InteriorEvent(
            kind=EventKind.SOCIAL,
            summary=str(message)[:200],
            subject=subject or "unknown",
            observations=observations,
            source="affective_resonance",
        )
        estimate = get_other_minds_model().estimate(event, species=species)

        if not estimate.channels_used:
            self._declines += 1
            return Resonance(
                valence=0.0,
                arousal=0.0,
                resonance=0.0,
                recommended_tone=(
                    "no read: nothing in this message carried evidence about "
                    "them beyond their own baseline"
                ),
                channels={},
                declined="no channel carried evidence",
            )

        posterior = estimate.tendencies
        valence = sum(
            posterior.get(name, 0.0) * loading for name, loading in _VALENCE_LOADING.items()
        )
        arousal = sum(
            posterior.get(name, 0.0) * loading for name, loading in _AROUSAL_LOADING.items()
        )
        readiness, _mass = estimate.top()
        margin = estimate.margin()

        declined = ""
        if margin < 0.05:
            self._declines += 1
            tone = (
                f"no read: the top two readinesses ({readiness} and the next) "
                f"are within {margin:.2f} of each other"
            )
            declined = "the posterior does not separate its top two readinesses"
            readiness = ""
        else:
            tone = (
                f"inferred readiness {readiness}, margin {margin:.2f}, "
                f"confidence {estimate.confidence:.2f} from "
                f"{len(estimate.channels_used)} channel(s)"
            )

        return Resonance(
            valence=round(max(-1.0, min(1.0, valence)), 3),
            arousal=round(max(0.0, min(1.0, arousal)), 3),
            resonance=round(estimate.confidence, 3),
            recommended_tone=tone,
            readiness=readiness,
            margin=round(margin, 3),
            channels={k: round(v, 3) for k, v in estimate.channels_used.items()},
            declined=declined,
        )

    async def deep_attune(self, message: str, *, timeout: float = 8.0) -> Resonance:
        """Kept for callers that ask for a deeper read.

        The previous version asked a model "how the listener should sound
        in reply" and wrote the answer into the tone field, which then
        reached the context of every turn. That is an instruction about
        style produced by a system with no evidence about the person, and
        it is removed rather than improved. A deeper read means more
        channels, so this returns the same inference and says what would
        actually deepen it.
        """
        base = self.attune(message)
        if not base.declined and base.channels:
            missing = [c for c in ("timing", "prosody", "behaviour", "context")
                       if c not in base.channels]
            if missing:
                base.recommended_tone += (
                    f"; a stronger read needs {', '.join(missing)}, which this "
                    "call had no access to"
                )
        return base

    def record_outcome(self, resonance: Resonance, *, acted: str) -> dict[str, float]:
        """Tell the model what the person actually did.

        The only thing that moves channel reliability. Without it the
        weights are frozen and the read cannot improve, which is the
        state the word lists were permanently in.
        """
        try:
            from core.interiority.other_minds import get_other_minds_model

            model = get_other_minds_model()
            # Rebuild the minimum estimate shape the learner needs.
            from core.interiority.other_minds import OtherEstimate
            from core.interiority.evidence import absent

            estimate = OtherEstimate(
                entity="unknown",
                species="human",
                tendencies={},
                declined=(),
                distress=absent(),
                vulnerability=absent(),
                coping=absent(),
                capability=absent(),
                channels_used=dict(resonance.channels),
                confidence=resonance.resonance,
            )
            return model.record_outcome(estimate, actual_tendency=acted)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation("affective_resonance", exc, action="outcome not recorded")
            return {}

    def get_status(self) -> dict[str, Any]:
        return {
            "reads": self._reads,
            "declines": self._declines,
            "healthy": True,
            "inference": "core.interiority.other_minds",
        }


_INSTANCE: AffectiveResonance | None = None


def get_affective_resonance() -> AffectiveResonance:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AffectiveResonance()
    return _INSTANCE


def register_affective_resonance(orchestrator: Any = None) -> AffectiveResonance:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.SAMANTHA, default=None) or get_affective_resonance()
    register_runtime_service(ServiceNames.SAMANTHA, inst)
    return inst
