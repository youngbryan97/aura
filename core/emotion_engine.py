import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from core.runtime.errors import record_degradation

from .affect import AffectEngine, AffectState

logger = logging.getLogger("Aura.EmotionShim")


_RECOVERABLE_EMOTION_ERRORS = (RuntimeError, TypeError, ValueError, AttributeError)


class EmotionType(Enum):
    # Mapping legacy enum to new PAD state is approximate
    JOY = auto()
    TRUST = auto()
    FEAR = auto()
    SURPRISE = auto()
    SADNESS = auto()
    DISGUST = auto()
    ANGER = auto()
    ANTICIPATION = auto()
    NEUTRAL = auto()


@dataclass(frozen=True)
class LegacyEmotionState:
    """Read-only view expected by older callers of the emotion engine."""

    primary: str
    intensity: float
    mood: str
    last_update: float

    @classmethod
    def from_affect(cls, pad: AffectState) -> "LegacyEmotionState":
        mood = pad.dominant_emotion or "neutral"
        return cls(
            primary=mood.upper(),
            intensity=pad.arousal,
            mood=mood,
            last_update=pad.last_update,
        )


class EmotionEngine:
    """Compatibility layer from the legacy emotion API to AffectEngine PAD state."""

    def __init__(self):
        self.engine = AffectEngine()

    @property
    def state(self) -> LegacyEmotionState:
        return LegacyEmotionState.from_affect(self.engine.state)

    def get_state(self) -> dict[str, Any]:
        s = self.engine.state
        return {
            "primary": s.dominant_emotion,
            "intensity": round(s.arousal, 2),
            "mood": s.dominant_emotion,
            "valence": round(s.valence, 2),
            "engagement": round(s.engagement, 2),
        }

    def react(self, trigger: str, context: dict[str, Any] | None = None) -> None:
        """Forward react call."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.engine.react(trigger, context))
                return

            if loop.is_running():
                loop.create_task(self.engine.react(trigger, context))

        except _RECOVERABLE_EMOTION_ERRORS as exc:
            record_degradation("emotion_engine", exc)
            logger.error("Failed to forward reaction: %s", exc)


# Global Instance (Legacy)
emotion_engine = EmotionEngine()
