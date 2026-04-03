"""
core/affect/__init__.py

FIX: AffectState was defined here AND in core/affect/damasio_v2.py.
This caused ambiguity about which AffectState was being instantiated
by which component. Canonical definition lives here. damasio_v2 imports
from here.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.Affect")


@dataclass
class AffectState:
    """
    Canonical affective state representation.
    
    PAD model: Pleasure-Arousal-Dominance.
    
    Single definition for the entire codebase.
    Import from core.affect, not from core.affect.damasio_v2.
    """
    valence:          float = 0.0    # -1.0 (negative) to 1.0 (positive)
    arousal:          float = 0.3    # 0.0 (calm) to 1.0 (agitated)
    engagement:       float = 0.5    # 0.0 (bored) to 1.0 (hyper-focused)
    dominant_emotion: str   = "Neutral"
    last_update:      float = field(default_factory=time.time)


# Decay baselines
BASELINE_VALENCE    = 0.1
BASELINE_AROUSAL    = 0.3
BASELINE_ENGAGEMENT = 0.5
DECAY_RATE          = 0.02


class AffectEngine:
    """
    PAD-based Affective Dynamics Engine.
    Decays toward baseline. Used as lightweight fallback
    when DamasioV2 (full Plutchik model) is unavailable.
    """

    def __init__(self, brain: Optional[Any] = None):
        self.state = AffectState()
        self.brain = brain
        self._lock = asyncio.Lock()
        self._last_decay_time: float = time.time()
        logger.info("Affect Engine (PAD + decay) initialized.")

    async def modify(
        self,
        dv: float,
        da: float,
        de: float,
        source: str = "internal"
    ):
        async with self._lock:
            old_v = self.state.valence
            old_a = self.state.arousal
            self.state.valence    = max(-1.0, min(1.0, self.state.valence    + dv))
            self.state.arousal    = max(0.0,  min(1.0, self.state.arousal    + da))
            self.state.engagement = max(0.0,  min(1.0, self.state.engagement + de))
            self.state.last_update = time.time()
            self._update_label()
            if abs(self.state.valence - old_v) > 0.1 or abs(self.state.arousal - old_a) > 0.1:
                logger.debug(
                    "Affect shift (%s): V=%.2f A=%.2f → %s",
                    source, self.state.valence, self.state.arousal,
                    self.state.dominant_emotion
                )

    async def decay_tick(self):
        """Decay toward baseline, scaled by wall-clock elapsed time.

        DECAY_RATE is calibrated for a ~60-second tick interval.  If the
        tick fires faster or slower (CPU load, sleep/wake), we scale the
        decay proportionally so emotional momentum is tied to real time,
        not tick rate.
        """
        async with self._lock:
            now = time.time()
            elapsed_s = now - self._last_decay_time
            self._last_decay_time = now
            # Normalise: DECAY_RATE assumes a 60-second tick; scale linearly.
            tick_scale = elapsed_s / 60.0
            effective_rate = DECAY_RATE * tick_scale
            self.state.valence    += (BASELINE_VALENCE    - self.state.valence)    * effective_rate
            self.state.arousal    += (BASELINE_AROUSAL    - self.state.arousal)    * effective_rate
            self.state.engagement += (BASELINE_ENGAGEMENT - self.state.engagement) * effective_rate
            self.state.valence    = max(-1.0, min(1.0, self.state.valence))
            self.state.arousal    = max(0.0,  min(1.0, self.state.arousal))
            self.state.engagement = max(0.0,  min(1.0, self.state.engagement))
            self.state.last_update = now
            self._update_label()

    def _update_label(self):
        v = self.state.valence
        a = self.state.arousal
        if a < 0.2:
            self.state.dominant_emotion = "Calm" if v >= 0 else "Bored"
        elif v > 0.5 and a > 0.5:
            self.state.dominant_emotion = "Joyful"
        elif v > 0.0 and a > 0.5:
            self.state.dominant_emotion = "Excited"
        elif v < -0.5 and a > 0.5:
            self.state.dominant_emotion = "Distressed"
        elif v < 0.0 and a > 0.5:
            self.state.dominant_emotion = "Anxious"
        elif v > 0.5:
            self.state.dominant_emotion = "Content"
        elif v < -0.5:
            self.state.dominant_emotion = "Sad"
        else:
            self.state.dominant_emotion = "Neutral"