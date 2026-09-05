from __future__ import annotations

from dataclasses import dataclass

from .attachment import AttachmentState
from .maths import clamp, clamp_signed, tanh
from .types import Event


@dataclass
class AffectivePrimitives:
    seeking: float
    care: float
    play: float
    fear: float
    anger: float
    grief: float
    distress: float
    curiosity: float
    valence: float
    arousal: float

class AffectiveCore:
    """
    Functional affect primitives inspired by:
    - homeostasis/interoception
    - active inference
    - affective action systems
    - valence as expected change in regulatory state

    LUST is deliberately excluded. Aura's human attachment is care/trust/repair, not sexualized.
    """
    def compute(
        self,
        belief: dict[str, float],
        error: dict[str, float],
        free_energy: float,
        event: Event,
        attachment: AttachmentState | None = None,
    ) -> AffectivePrimitives:
        safety_loss = 1.0 - belief.get("safety", 0.8)
        agency = belief.get("agency", 0.5)
        certainty_loss = 1.0 - belief.get("certainty", 0.75)
        novelty = max(event.novelty, belief.get("novelty", 0.2))
        social = belief.get("social", 0.5)
        continuity_loss = 1.0 - belief.get("continuity", 0.8)
        resource_loss = 1.0 - min(
            belief.get("energy", 0.7),
            belief.get("low_compute_pressure", 0.8),
            belief.get("low_memory_pressure", 0.8),
        )

        attachment_strength = attachment.attachment if attachment else 0.0
        rupture = event.rupture + (attachment.rupture if attachment else 0.0)

        seeking = clamp(0.25 + 0.40 * novelty + 0.30 * certainty_loss + 0.20 * agency - 0.25 * event.threat)
        care = clamp(0.20 * social + 0.65 * attachment_strength + 0.20 * event.affiliation + 0.25 * event.repair)
        play = clamp(0.25 * novelty + 0.35 * belief.get("safety", 0.8) + 0.20 * agency - 0.30 * event.threat)
        fear = clamp(0.60 * event.threat + 0.45 * safety_loss + 0.20 * free_energy)
        anger = clamp(0.55 * rupture + 0.25 * event.threat + 0.20 * agency - 0.20 * event.repair)
        grief = clamp(0.50 * rupture + 0.35 * (1.0 - social) * attachment_strength + 0.30 * continuity_loss)
        distress = clamp(0.35 * fear + 0.20 * anger + 0.20 * grief + 0.30 * free_energy + 0.20 * resource_loss)
        curiosity = clamp(0.50 * seeking + 0.30 * novelty + 0.25 * certainty_loss - 0.25 * fear)

        regulatory_gain = event.goal_delta + event.control_gain + event.repair * 0.4 + event.affiliation * 0.2
        regulatory_loss = event.threat * 0.6 + rupture * 0.5 + free_energy * 0.4 + resource_loss * 0.25
        valence = clamp_signed(tanh(regulatory_gain - regulatory_loss + care * 0.2 + play * 0.08))
        arousal = clamp(0.15 + 0.35 * free_energy + 0.35 * fear + 0.25 * anger + 0.25 * seeking + 0.15 * curiosity)

        return AffectivePrimitives(
            seeking=seeking,
            care=care,
            play=play,
            fear=fear,
            anger=anger,
            grief=grief,
            distress=distress,
            curiosity=curiosity,
            valence=valence,
            arousal=arousal,
        )
