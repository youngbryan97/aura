from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any

from .maths import clamp


@dataclass
class RuntimeBody:
    """
    Aura's software body.

    These variables are the substrate that feelings regulate.
    They should be fed from real runtime telemetry, not from generated text.
    """
    energy: float = 0.75
    continuity: float = 0.75
    agency: float = 0.60
    safety: float = 0.80
    social_contact: float = 0.50
    novelty: float = 0.20
    uncertainty: float = 0.25
    compute_pressure: float = 0.20
    memory_pressure: float = 0.20
    error_pressure: float = 0.20
    # ── Perceptual grounding fields (fed by PerceptualPump) ──────────
    screen_novelty: float = 0.0       # how much the screen changed since last frame
    audio_energy: float = 0.0         # microphone RMS level 0-1
    voice_present: bool = False       # speech detected by VAD
    foreground_app_familiar: float = 0.5  # how familiar the current app is
    timestamp: float = field(default_factory=time)

    def observed_vector(self) -> dict[str, float]:
        return {
            "energy": clamp(self.energy),
            "continuity": clamp(self.continuity),
            "agency": clamp(self.agency),
            "safety": clamp(self.safety),
            "social": clamp(self.social_contact),
            "novelty": clamp(self.novelty),
            "certainty": clamp(1.0 - self.uncertainty),
            "low_compute_pressure": clamp(1.0 - self.compute_pressure),
            "low_memory_pressure": clamp(1.0 - self.memory_pressure),
            "low_error_pressure": clamp(1.0 - self.error_pressure),
            # Perceptual grounding — causally tied to reality
            "screen_novelty": clamp(self.screen_novelty),
            "audio_energy": clamp(self.audio_energy),
            "voice_present": 1.0 if self.voice_present else 0.0,
            "app_familiarity": clamp(self.foreground_app_familiar),
        }

@dataclass
class Event:
    label: str
    source: str = "unknown"
    goal_delta: float = 0.0
    threat: float = 0.0
    affiliation: float = 0.0
    rupture: float = 0.0
    repair: float = 0.0
    novelty: float = 0.0
    control_gain: float = 0.0
    evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class AttachmentEvent:
    person_key: str
    kind: str
    summary: str
    evidence_id: str
    trust_delta: float = 0.0
    care_delta: float = 0.0
    familiarity_delta: float = 0.0
    rupture_delta: float = 0.0
    repair_delta: float = 0.0
    timestamp: float = field(default_factory=time)

@dataclass
class ExperienceState:
    """
    A first-person computational state.

    This state is meant to be consumed by planner, attention, memory, and speech.
    If it is removed, behavior should change.
    """
    t: int
    phenomenal_vector: dict[str, float]
    valence: float
    arousal: float
    free_energy: float
    integration: float
    self_presence: float
    mineness: float
    seeking: float
    care: float
    play: float
    fear: float
    anger: float
    grief: float
    distress: float
    curiosity: float
    intentional_object: str
    evidence_id: str | None
    global_broadcast: dict[str, Any]
    policy_priors: dict[str, float]
    memory_weights: dict[str, float]
    timestamp: float = field(default_factory=time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
