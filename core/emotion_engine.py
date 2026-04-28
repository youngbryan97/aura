from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Dict, Optional

from .affect import AffectEngine, AffectState

logger = logging.getLogger("Aura.EmotionShim")

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

class EmotionEngine:
    """Legacy Shim for EmotionEngine -> AffectEngine (PAD).
    """

    def __init__(self):
        self.engine = AffectEngine()
        
    @property
    def state(self):
        # Mock the legacy state object structure if possible,
        # or just return something safe.
        # Legacy accessed: state.primary, state.intensity, state.mood
        # We construct a dummy object on the fly.
        class LegacyState:
            def __init__(self, pad: AffectState):
                self.primary = pad.dominant_emotion.upper() if pad.dominant_emotion else "NEUTRAL"
                self.intensity = pad.arousal
                self.mood = pad.dominant_emotion
                self.last_update = pad.last_update
        
        # We need async access to get real state, but property is sync.
        # This is the tricky part of shimming async back to sync.
        # We'll just read the internal state directly (it's a dataclass, fairly safe to read).
        return LegacyState(self.engine.state)

    def get_state(self) -> Dict[str, Any]:
        s = self.engine.state
        return {
            "primary": s.dominant_emotion, # Return string instead of Enum for API
            "intensity": round(s.arousal, 2),
            "mood": s.dominant_emotion,
            "valence": round(s.valence, 2),
            "engagement": round(s.engagement, 2)
        }

    def react(self, trigger: str, context: Dict = None):
        """Forward react call."""
        # Fire-and-forget the async reaction
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            if loop.is_running():
                loop.create_task(self.engine.react(trigger, context))
            else:
                 loop.run_until_complete(self.engine.react(trigger, context))

        except Exception as e:
            record_degradation('emotion_engine', e)
            logger.error("Failed to forward reaction: %s", e)

# Global Instance (Legacy)
emotion_engine = EmotionEngine()
