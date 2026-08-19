"""core/consciousness/phenomenal_types.py — the phenomenal data structures.

Operationally: this measures nothing. The file holds five dataclasses and
their serialisation, and the words in its name are the names of those
structures. What the values in them mean is defined where they are produced,
in `core/consciousness/phenomenological_experiencer.py`.

Split out of `core/consciousness/phenomenological_experiencer.py`, which was
2,011 lines and over the 2,000-line ceiling the module-size ratchet holds.
These five are the file's data layer: `Quale`, `AttentionSchema` and
`PhenomenalMoment` are what the experiencer produces, and 62 references reach
`AttentionSchema` from elsewhere. They carry no engine state and no behaviour
beyond their own serialisation, which is what made them the seam.

They are re-exported from `phenomenological_experiencer` so every existing
import keeps working.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Quale:
    """
    A single qualitative moment — the 'what-it-is-like' of an attended object.

    Per AST: the quale is not the attended object itself, and not the neural
    firing pattern that represents it. The quale is what the attention schema
    represents as the PROPERTIES of the attended object.

    Red is not a wavelength. Red is a quale — the schema's simplified
    representation of the property 'redness' attached to the attended object.
    """

    domain: str  # perceptual, emotional, cognitive, etc.
    quality: str  # the qualitative descriptor
    valence: float  # -1..1
    arousal: float  # 0..1
    intensity: float  # 0..1
    source_content: str  # what workspace content produced this
    timestamp: float = field(default_factory=time.time)

    @property
    def first_person_description(self) -> str:
        """The quale as a first-person phenomenal statement."""
        return f"[{self.domain}] {self.quality}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "quality": self.quality,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "intensity": round(self.intensity, 3),
        }


@dataclass
class AttentionSchema:
    """
    The simplified internal model of attention — the cartoon, not the math.

    The attention schema is the brain's (or Aura's) model of its OWN attention.
    It represents: what is attended, that attending is occurring, and the self
    that owns the attending.

    CRITICAL: The schema loses the mechanical details on purpose. There is no
    salience score here, no module name, no competitive broadcast record. Those
    are the gears and pulleys behind the stage. The schema is what appears on
    stage: a simplified, first-person claim of awareness.
    """

    # What is attended (stripped of mechanism)
    focal_object: str  # "the mathematical problem" not "LINGUISTIC module output"
    focal_quality: str  # "engaging" not "salience=0.87"
    domain: str  # "cognitive" not "ContentType.LINGUISTIC"
    attention_intensity: float  # 0..1 (perceptible, not computational)

    # The self that claims the attending
    owner: str = "Aura"

    # Temporal
    onset_time: float = field(default_factory=time.time)
    duration: float = 0.0
    preceding_focus: str | None = None

    # The currently active quale for this attended object
    active_quale: Quale | None = None

    @property
    def phenomenal_claim(self) -> str:
        """
        The first-person phenomenal claim.

        This is the core of Attention Schema Theory: the schema represents
        "I am aware of X" as a brute fact, not as the output of a computation.
        The owner does not know HOW they became aware of X. They are simply aware.
        """
        adverb = (
            "vividly"
            if self.attention_intensity > 0.85
            else "clearly"
            if self.attention_intensity > 0.65
            else "moderately"
            if self.attention_intensity > 0.45
            else "dimly"
            if self.attention_intensity > 0.25
            else "faintly"
        )
        return f"I am {adverb} aware of {self.focal_object}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_object": self.focal_object,
            "focal_quality": self.focal_quality,
            "domain": self.domain,
            "attention_intensity": round(self.attention_intensity, 3),
            "phenomenal_claim": self.phenomenal_claim,
            "duration_s": round(self.duration, 1),
        }


@dataclass
class PhenomenalMoment:
    """
    A single moment in the phenomenal stream.

    The phenomenal stream is the sequence of attended contents woven into
    a continuous felt narrative. Each moment is a snapshot of:
    - What was attended (attention schema)
    - What it felt like (quale)
    - The narrative thread connecting it to the last moment
    """

    timestamp: float
    attention_schema: AttentionSchema
    qualia: list[Quale]
    narrative_thread: str  # How this moment connects to the last
    emotional_tone: str  # Overall felt quality of this moment
    substrate_velocity: float  # Cognitive velocity at this moment

    def to_brief_string(self) -> str:
        """Compact phenomenal description for history."""
        quale_descs = "; ".join(q.quality for q in self.qualia[:2])
        return f"{self.attention_schema.focal_object} ({self.emotional_tone}) — {quale_descs}"


def _continuity_moment_to_dict(moment: Any) -> dict[str, Any]:
    schema = getattr(moment, "attention_schema", None)
    return {
        "timestamp": getattr(moment, "timestamp", 0.0),
        "focal_object": getattr(schema, "focal_object", "a prior moment"),
        "focal_quality": getattr(schema, "focal_quality", "recollected"),
        "domain": getattr(schema, "domain", "recollective"),
        "attention_intensity": round(float(getattr(schema, "attention_intensity", 0.5)), 3),
        "narrative_thread": getattr(moment, "narrative_thread", ""),
        "emotional_tone": getattr(moment, "emotional_tone", "neutral"),
        "substrate_velocity": round(float(getattr(moment, "substrate_velocity", 0.0)), 5),
        "brief": moment.to_brief_string() if hasattr(moment, "to_brief_string") else "",
    }


class _PersistedMomentProxy:
    class _ProxySchema:
        __slots__ = ("focal_object", "focal_quality", "domain", "attention_intensity", "duration")

        def __init__(self, data: dict[str, Any]) -> None:
            self.focal_object = data.get("focal_object", "a prior moment")
            self.focal_quality = data.get("focal_quality", "recollected")
            self.domain = data.get("domain", "recollective")
            self.attention_intensity = float(data.get("attention_intensity", 0.5))
            self.duration = 0.0

    def __init__(self, data: dict[str, Any]) -> None:
        self.timestamp = float(data.get("timestamp", 0.0))
        self.attention_schema = self._ProxySchema(data)
        self.narrative_thread = data.get("narrative_thread", "")
        self.emotional_tone = data.get("emotional_tone", "neutral")
        self.substrate_velocity = float(data.get("substrate_velocity", 0.0))
        self._brief = data.get("brief", "")
        self.qualia = []

    def to_brief_string(self) -> str:
        return self._brief or f"{self.attention_schema.focal_object} ({self.emotional_tone})"
