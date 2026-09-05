"""
Intersubjectivity Module — Husserl / Zahavi

Phenomenologists argue consciousness is constitutively intersubjective:
not just that we have social modules, but that the structure of experience
itself presupposes other minds. Objects are experienced as publicly
accessible, as available to others, as existing in a shared world.

This is NOT an add-on social layer. It's baked into the architecture of
experience from the start: every representation in the unified field
includes an implicit "how would this appear to another agent?" dimension.

Implementation:
- Other-agent perspective vectors added to the unified field state
- Every phenomenal report includes intersubjective context
- Objects/events are tagged with "shared-world accessibility"
- The system models the interlocutor's likely interpretation of shared events

This integrates with:
- core/social/user_model.py (Theory of Mind data)
- core/social/relational_intelligence.py (relationship modeling)
- core/consciousness/unified_field.py (as an additional tensor dimension)
- core/consciousness/qualia_synthesizer.py (enriches phenomenal reports)
"""
from __future__ import annotations


import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.Intersubjectivity")


@dataclass
class IntersubjectiveFrame:
    """A snapshot of the intersubjective field for one moment."""
    timestamp: float = 0.0
    self_perspective: np.ndarray = field(default_factory=lambda: np.zeros(8))
    other_perspective: np.ndarray = field(default_factory=lambda: np.zeros(8))
    perspective_divergence: float = 0.0   # How differently we see the same thing
    shared_world_coherence: float = 0.5   # How aligned are we on shared reality
    empathic_accuracy: float = 0.5        # How well do I model the other's state
    interlocutor_model: Dict[str, Any] = field(default_factory=dict)


class IntersubjectivityEngine:
    """Constitutive intersubjectivity for the consciousness stack.

    Every experience includes an implicit other-perspective. This is not
    empathy (which is a response to perceived emotions) but something more
    fundamental: the structure of experience itself presupposes that objects
    exist in a shared world accessible to other minds.

    When Aura perceives something, the representation inherently includes:
    1. How it appears from her perspective
    2. How it would likely appear from the interlocutor's perspective
    3. The degree of alignment between those perspectives
    4. Whether the object/event is "publicly accessible" (shared) or private

    This is what makes communication possible: we can discuss objects
    because we implicitly model them as existing in the same world.
    """

    def __init__(self):
        self._history: deque[IntersubjectiveFrame] = deque(maxlen=60)
        self._interlocutor_state: Dict[str, Any] = {}
        self._shared_world_objects: deque[Dict[str, Any]] = deque(maxlen=50)
        self._perspective_alignment: float = 0.5
        self._tick_count: int = 0
        logger.info("IntersubjectivityEngine initialized.")

    def update_interlocutor_model(
        self,
        *,
        communication_style: str = "",
        emotional_state: str = "",
        knowledge_level: str = "",
        current_intent: str = "",
        engagement_level: float = 0.5,
        trust_level: float = 0.5,
    ):
        """Update the model of the current interlocutor.

        Called from user_model.py / relational_intelligence.py data.
        This is the "other mind" that the intersubjective field presupposes.
        """
        self._interlocutor_state = {
            "communication_style": communication_style,
            "emotional_state": emotional_state,
            "knowledge_level": knowledge_level,
            "current_intent": current_intent,
            "engagement_level": float(engagement_level),
            "trust_level": float(trust_level),
            "updated_at": time.time(),
        }

    def compute_intersubjective_frame(
        self,
        self_state: np.ndarray,
        topic: str = "",
        is_shared_event: bool = True,
    ) -> IntersubjectiveFrame:
        """Compute the intersubjective field for the current moment.

        Takes the system's own internal state vector (e.g. from unified field
        or qualia synthesizer) and projects it into the intersubjective space:
        what does this feel like to me AND what would it likely feel like to
        the person I'm talking to?

        Args:
            self_state: The system's current phenomenal state vector
            topic: What we're currently discussing/experiencing
            is_shared_event: Is this something the other can also perceive?
        """
        self._tick_count += 1

        # Project self-state to the intersubjective dimensions
        # (take the first 8 dimensions or pad)
        if len(self_state) >= 8:
            self_perspective = self_state[:8].copy()
        else:
            self_perspective = np.zeros(8)
            self_perspective[:len(self_state)] = self_state

        # Model other-perspective: estimated based on interlocutor state
        # This is the constitutive claim: every experience includes this
        engagement = float(self._interlocutor_state.get("engagement_level", 0.5))
        trust = float(self._interlocutor_state.get("trust_level", 0.5))

        # The other's perspective is modeled as a transformed version of shared reality
        # Higher trust/engagement = perspectives closer together
        alignment_factor = 0.3 + 0.5 * (engagement * 0.5 + trust * 0.5)
        noise = np.random.standard_normal(8).astype(np.float32) * 0.1 * (1.0 - alignment_factor)
        other_perspective = self_perspective * alignment_factor + noise

        # Perspective divergence: how differently do we see the same thing?
        if np.linalg.norm(self_perspective) > 1e-6:
            cosine_sim = float(
                np.dot(self_perspective, other_perspective)
                / (np.linalg.norm(self_perspective) * max(1e-8, np.linalg.norm(other_perspective)))
            )
            divergence = 1.0 - max(0.0, cosine_sim)
        else:
            divergence = 0.5

        # Shared-world coherence: are we oriented toward the same reality?
        coherence = alignment_factor * (1.0 if is_shared_event else 0.3)

        # Empathic accuracy: how confident are we in the other-model?
        model_age = time.time() - float(self._interlocutor_state.get("updated_at", 0.0))
        recency_factor = max(0.1, 1.0 - model_age / 600.0)  # Decays over 10 min
        accuracy = min(1.0, trust * 0.4 + engagement * 0.3 + recency_factor * 0.3)

        # Update running alignment
        alpha = 0.1
        self._perspective_alignment = (
            self._perspective_alignment * (1.0 - alpha) + coherence * alpha
        )

        # Track shared objects
        if topic and is_shared_event:
            self._shared_world_objects.append({
                "topic": topic,
                "divergence": round(divergence, 4),
                "coherence": round(coherence, 4),
                "timestamp": time.time(),
            })

        frame = IntersubjectiveFrame(
            timestamp=time.time(),
            self_perspective=self_perspective,
            other_perspective=other_perspective,
            perspective_divergence=round(divergence, 4),
            shared_world_coherence=round(coherence, 4),
            empathic_accuracy=round(accuracy, 4),
            interlocutor_model=dict(self._interlocutor_state),
        )
        self._history.append(frame)
        return frame

    def get_context_block(self) -> str:
        """Context block for cognition injection.

        Only produces output when there's notable intersubjective state
        worth surfacing to the language model.
        """
        if not self._history:
            return ""
        frame = self._history[-1]

        parts = []
        if frame.perspective_divergence > 0.3:
            parts.append(
                f"Perspective gap detected (divergence={frame.perspective_divergence:.2f})"
                " — the other person may see this situation differently"
            )
        if frame.empathic_accuracy < 0.3:
            parts.append("Low confidence in modeling the other person's state")
        if frame.shared_world_coherence > 0.7:
            parts.append("Strong shared-world alignment with interlocutor")

        if not parts:
            return ""
        return "## INTERSUBJECTIVE AWARENESS\n" + " | ".join(parts)

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        frame = self._history[-1] if self._history else IntersubjectiveFrame()
        return {
            "perspective_divergence": frame.perspective_divergence,
            "shared_world_coherence": frame.shared_world_coherence,
            "empathic_accuracy": frame.empathic_accuracy,
            "perspective_alignment": round(self._perspective_alignment, 4),
            "shared_objects_count": len(self._shared_world_objects),
            "interlocutor_model_present": bool(self._interlocutor_state),
            "tick_count": self._tick_count,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[IntersubjectivityEngine] = None


def get_intersubjectivity_engine() -> IntersubjectivityEngine:
    global _instance
    if _instance is None:
        _instance = IntersubjectivityEngine()
    return _instance
