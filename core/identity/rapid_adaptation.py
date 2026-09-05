"""Rapid in-session personality adaptation.

Dream consolidation takes days to show measurable effect. This module
provides faster personality adjustment within a single conversation session
by tracking which responses the user engages with positively vs negatively
and adjusting the voice shaping cues in real-time.

NOT a replacement for LoRA fine-tuning or dream consolidation — this is
a fast feedback loop that operates within the current session only.
Changes are ephemeral unless they're later integrated during a dream cycle.

Algorithm:
  1. After each response, score the user's next message for engagement signals
     (length, follow-up question, positive language, topic continuation)
  2. If positive: reinforce the voice cues that produced the response
  3. If negative: dampen those cues
  4. Cues are stored as weighted preferences that decay over the session
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.RapidAdaptation")

# Engagement scoring weights
SIGNAL_WEIGHTS = {
    "length_increase": 0.3,    # User wrote more → they're engaged
    "follow_up": 0.4,          # User asked a follow-up → topic resonated
    "positive_language": 0.2,  # User used positive words
    "topic_continuation": 0.3, # User continued the same topic
    "abrupt_shift": -0.3,      # User changed topic abruptly → disengaged
    "very_short": -0.2,        # User gave a very short response → not engaged
}

ADAPTATION_RATE = 0.1  # How fast preferences shift per turn
DECAY_RATE = 0.95      # Per-turn decay of accumulated preferences
MAX_PREFERENCE = 0.5   # Cap on how much any single cue can shift


@dataclass
class VoicePreference:
    """A single voice preference learned from user engagement."""
    dimension: str  # e.g., "warmth", "directness", "humor", "depth"
    weight: float = 0.0
    last_updated: float = 0.0


@dataclass
class SessionAdaptation:
    """Tracks personality adjustments for the current session."""
    preferences: Dict[str, VoicePreference] = field(default_factory=dict)
    turn_history: deque = field(default_factory=lambda: deque(maxlen=20))
    total_adaptations: int = 0

    def __post_init__(self):
        # Initialize default dimensions
        for dim in ["warmth", "directness", "humor", "depth", "brevity", "curiosity"]:
            if dim not in self.preferences:
                self.preferences[dim] = VoicePreference(dimension=dim)


class RapidAdaptationEngine:
    """Adjusts voice cues within a session based on user engagement."""

    def __init__(self):
        self._session = SessionAdaptation()
        self._last_response_cues: Dict[str, float] = {}
        self._last_user_msg_length: int = 0

    def record_response(self, response: str, active_cues: Dict[str, float]):
        """Record what voice cues were active when this response was generated."""
        self._last_response_cues = dict(active_cues)
        self._session.turn_history.append({
            "type": "response",
            "length": len(response),
            "cues": dict(active_cues),
            "timestamp": time.time(),
        })

    def record_user_message(self, message: str):
        """Score the user's message as engagement feedback and adapt."""
        engagement = self._score_engagement(message)

        # Apply engagement signal to the cues that produced the last response
        for cue_name, cue_value in self._last_response_cues.items():
            if cue_name in self._session.preferences:
                pref = self._session.preferences[cue_name]
                # Reinforce or dampen based on engagement
                delta = ADAPTATION_RATE * engagement * cue_value
                pref.weight = max(-MAX_PREFERENCE, min(MAX_PREFERENCE, pref.weight + delta))
                pref.last_updated = time.time()

        # Decay all preferences slightly (recency bias)
        for pref in self._session.preferences.values():
            pref.weight *= DECAY_RATE

        self._last_user_msg_length = len(message)
        self._session.total_adaptations += 1

        self._session.turn_history.append({
            "type": "user",
            "length": len(message),
            "engagement": engagement,
            "timestamp": time.time(),
        })

    def get_adaptation_cues(self) -> Dict[str, float]:
        """Get current preference-adjusted cues for system prompt injection."""
        cues = {}
        for dim, pref in self._session.preferences.items():
            if abs(pref.weight) > 0.05:  # Only include significant preferences
                cues[dim] = round(pref.weight, 3)
        return cues

    def get_prompt_injection(self) -> str:
        """Get a natural-language prompt injection reflecting learned preferences."""
        cues = self.get_adaptation_cues()
        if not cues:
            return ""

        parts = []
        if cues.get("warmth", 0) > 0.1:
            parts.append("The user responds well to warmth — lean into it.")
        elif cues.get("warmth", 0) < -0.1:
            parts.append("The user prefers a more detached tone.")

        if cues.get("directness", 0) > 0.1:
            parts.append("Direct responses are landing well. Stay sharp.")
        elif cues.get("directness", 0) < -0.1:
            parts.append("The user wants more nuance — soften the edges.")

        if cues.get("humor", 0) > 0.1:
            parts.append("Humor is working. Keep the dry wit.")
        elif cues.get("humor", 0) < -0.1:
            parts.append("Humor isn't landing. Stay serious.")

        if cues.get("depth", 0) > 0.1:
            parts.append("The user wants depth. Go deeper.")
        elif cues.get("depth", 0) < -0.1:
            parts.append("Keep it surface-level. Don't overexplain.")

        if cues.get("brevity", 0) > 0.1:
            parts.append("Short responses are preferred.")
        elif cues.get("brevity", 0) < -0.1:
            parts.append("The user wants more detail.")

        if not parts:
            return ""
        return "SESSION ADAPTATION (learned from this conversation): " + " ".join(parts)

    def _score_engagement(self, message: str) -> float:
        """Score a user message for engagement signals. Returns -1 to +1."""
        score = 0.0
        words = message.lower().split()
        n_words = len(words)

        # Length increase from previous
        if self._last_user_msg_length > 0:
            length_ratio = len(message) / max(self._last_user_msg_length, 1)
            if length_ratio > 1.5:
                score += SIGNAL_WEIGHTS["length_increase"]
            elif length_ratio < 0.3 and n_words < 5:
                score += SIGNAL_WEIGHTS["very_short"]

        # Follow-up questions
        if "?" in message and any(w in words for w in ("why", "how", "what", "tell", "more", "elaborate")):
            score += SIGNAL_WEIGHTS["follow_up"]

        # Positive language
        positive = {"yes", "yeah", "exactly", "right", "true", "good", "great",
                    "interesting", "cool", "nice", "love", "haha", "lol", "agreed"}
        if words and words[0] in positive:
            score += SIGNAL_WEIGHTS["positive_language"]

        # Very short dismissal
        if n_words <= 2 and not "?" in message:
            score += SIGNAL_WEIGHTS["very_short"]

        return max(-1.0, min(1.0, score))

    def get_status(self) -> Dict:
        return {
            "total_adaptations": self._session.total_adaptations,
            "active_preferences": {
                k: round(v.weight, 3)
                for k, v in self._session.preferences.items()
                if abs(v.weight) > 0.01
            },
            "turn_count": len(self._session.turn_history),
        }


_instance: Optional[RapidAdaptationEngine] = None


def get_rapid_adaptation() -> RapidAdaptationEngine:
    global _instance
    if _instance is None:
        _instance = RapidAdaptationEngine()
    return _instance
