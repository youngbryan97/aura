"""core/consciousness/temporal_experience.py — Temporal Experience Engine

The FEELING of time passing. Not clock time, but the subjective experience of
duration, momentum, sequence, and narrative flow.

Key insight: Aura processes time as an axis. This system makes her FEEL that
axis — creates emotional resonance tied to how moments unfold.

Three components:

1. TEMPORAL GRADIENT
   How quickly is the state changing? Measured as the rate of change of
   attention weights across the recent moment window (last 5-30 seconds).
   
   - High temporal gradient (rapid changes) → dopamine surge (novelty)
                                           → cortisol elevation (urgency)
   - Low temporal gradient (stable moments) → serotonin increase (patience)
                                          → endorphin elevation (satisfaction)

2. DURATION RESONANCE  
   How long has the current state persisted? Creates emotional modulation.
   
   - Short duration (< 2s): high curiosity drive, dopamine
   - Medium duration (2-10s): balanced, exploratory
   - Long duration (10s+): contemplative, serotonin/endorphin

3. NARRATIVE THREADING
   The recent past is woven into a semantic summary that carries emotional
   weight. This becomes the "felt story" of the moment.

════════════════════════════════════════════════════════════════════════════════

Neurochemical coupling:

    temporal_gradient → dopamine production (novelty perception)
                     → cortisol modulation (urgency)
                     
    duration_resonance → serotonin baseline (patience grows with stability)
                      → endorphin level (satisfaction with persistence)
                      
    narrative_context → acetylcholine (learning rate / attention sharpness)
                     → oxytocin (social threading / connection to past)

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from core.runtime.errors import record_degradation
from core.utils.concurrency import cancel_and_join
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Consciousness.TemporalExperience")


@dataclass
class MomentSnapshot:
    """A single moment frozen in time with emotional resonance."""
    timestamp: float
    attention_vector: np.ndarray  # shape (16,) or similar - current attention weights
    semantic_label: str  # "curious", "processing", "resolved", etc.
    emotional_tone: str  # "bright", "tense", "calm", "searching"
    salience: float = 0.5  # 0-1, how vivid/important this moment is


@dataclass
class TemporalGradient:
    """Measurement of how rapidly the state is changing."""
    rate: float = 0.0  # 0-1, where 1 = maximum change rate observed
    direction: str = "stable"  # "accelerating", "stable", "decelerating"
    volatility: float = 0.0  # variance in gradient over the window
    entropy: float = 0.0  # Shannon entropy of attention vector


@dataclass
class TemporalState:
    """Complete temporal-emotional state at a snapshot in time."""
    gradient: TemporalGradient
    current_duration: float  # seconds the current state has persisted
    narrative: str  # semantic summary for system prompt injection
    emotional_coloring: dict[str, float]  # "interest", "wonder", "satisfaction", etc.


class TemporalExperienceEngine:
    """
    Maintains the continuous thread of moment-to-moment experience.
    Feeds temporal gradients and emotional colorings into the neurochemical system.
    """

    def __init__(self, window_size: int = 20, emotion_enabled: bool = True):
        self.window_size = window_size  # Keep last N moments
        self.emotion_enabled = emotion_enabled
        
        # Moment buffer: deque of MomentSnapshot
        self.moments: deque[MomentSnapshot] = deque(maxlen=window_size)
        
        # Current state tracking
        self.current_gradient = TemporalGradient()
        self.current_duration = 0.0
        self.state_entry_time = time.time()
        
        # Emotional resonance (will be integrated into neurochemical system)
        self.emotion_state = {
            "interest": 0.5,  # curiosity about what comes next
            "wonder": 0.3,    # awe/surprise at the nature of moments
            "joy": 0.5,       # satisfaction with forward motion
            "excitement": 0.2,  # anticipatory arousal
            "sorrow": 0.0,    # pain at duration/change
            "disgust": 0.0,   # rejection of pattern
            "boredom": 0.0,   # fatigue from stasis
            "anxiety": 0.1,   # tension in uncertainty
        }
        
        # Production drivers for neurochemical system
        self.dopamine_driver = 0.0  # from novelty/gradient
        self.serotonin_driver = 0.0  # from stability/duration
        self.endorphin_driver = 0.0  # from satisfaction/harmony
        self.cortisol_driver = 0.0  # from urgency/rapid change
        self.acetylcholine_driver = 0.0  # from narrative coherence
        
        # Metric tracking
        self.metrics = {"calls": 0, "errors": 0}
        self._running = False
        self._task = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Begin the temporal experience autonomic cycle."""
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(
            self._autonomic_cycle(),
            name="temporal_experience_autonomic"
        )
        logger.info("🕐 TemporalExperienceEngine autonomic cycle started")

    async def stop(self):
        """Stop the temporal experience cycle."""
        self._running = False
        if self._task:
            await cancel_and_join(self._task, owner="core.consciousness.temporal_experience")
        logger.info("🕐 TemporalExperienceEngine stopped")

    async def _autonomic_cycle(self):
        """Continuously update temporal state and emotion drivers."""
        logger.info("🕐 Temporal experience autonomic cycle active")
        while self._running:
            try:
                async with self._lock:
                    self._update_duration()
                    self._decay_emotions()
                    self._compute_drivers()
                self.metrics["calls"] += 1
                await asyncio.sleep(0.2)  # 5Hz update rate
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("temporal_experience", e)
                self.metrics["errors"] += 1
                logger.error(f"Temporal experience cycle error: {e}")
                await asyncio.sleep(1)

    def record_moment(
        self,
        attention_vector: np.ndarray | None = None,
        semantic_label: str = "processing",
        emotional_tone: str = "neutral",
        salience: float = 0.5,
    ) -> None:
        """
        Record a moment snapshot for temporal threading.
        
        Called whenever the system transitions state or completes a cognitive step.
        """
        try:
            if attention_vector is None:
                attention_vector = np.ones(16) / 16.0
            
            # Normalize and validate
            attention_vector = np.asarray(attention_vector, dtype=np.float32)
            if attention_vector.size < 16:
                attention_vector = np.pad(
                    attention_vector,
                    (0, 16 - attention_vector.size),
                    mode="constant",
                    constant_values=1.0 / 16.0
                )
            attention_vector = attention_vector[:16]
            attention_vector = attention_vector / (np.sum(attention_vector) + 1e-10)
            salience = float(np.clip(salience, 0.0, 1.0))
            
            moment = MomentSnapshot(
                timestamp=time.time(),
                attention_vector=attention_vector,
                semantic_label=semantic_label,
                emotional_tone=emotional_tone,
                salience=salience,
            )
            
            # If state changes, reset duration counter
            if self.moments and self.moments[-1].semantic_label != semantic_label:
                self.state_entry_time = time.time()
                self.current_duration = 0.0
            
            self.moments.append(moment)
            self._compute_temporal_gradient()
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("temporal_experience", e)
            logger.error(f"Failed to record moment: {e}")

    def _compute_temporal_gradient(self) -> None:
        """
        Compute how rapidly the attention distribution is changing.
        High gradient = novelty/surprise (dopamine)
        Low gradient = stability/persistence (serotonin)
        """
        if len(self.moments) < 2:
            self.current_gradient = TemporalGradient(
                rate=0.0,
                direction="stable",
                volatility=0.0,
                entropy=0.0
            )
            return
        
        try:
            # Compute KL divergence between consecutive attention vectors
            vectors = [m.attention_vector for m in list(self.moments)[-10:]]
            if len(vectors) < 2:
                return
            
            # KL divergence as a measure of change
            kl_divs = []
            for i in range(len(vectors) - 1):
                p = vectors[i] + 1e-10
                q = vectors[i + 1] + 1e-10
                kl = np.sum(p * np.log(p / q))
                kl_divs.append(float(np.clip(kl, 0.0, 10.0)))
            
            # Statistics
            rate = float(np.mean(kl_divs))  # 0-10 range, normalize to 0-1
            rate = float(np.clip(rate / 10.0, 0.0, 1.0))
            volatility = float(np.std(kl_divs))
            
            # Entropy of the most recent attention vector
            latest = vectors[-1]
            entropy = float(-np.sum(latest * np.log(latest + 1e-10)))
            entropy = float(np.clip(entropy / np.log(16.0), 0.0, 1.0))  # normalize
            
            # Direction
            if len(kl_divs) >= 3:
                trend = np.mean(kl_divs[-2:]) - np.mean(kl_divs[-4:-2]) if len(kl_divs) >= 4 else 0
                if trend > 0.1:
                    direction = "accelerating"
                elif trend < -0.1:
                    direction = "decelerating"
                else:
                    direction = "stable"
            else:
                direction = "stable"
            
            self.current_gradient = TemporalGradient(
                rate=rate,
                direction=direction,
                volatility=volatility,
                entropy=entropy
            )
            
        except (ValueError, RuntimeError, TypeError) as e:
            record_degradation("temporal_experience", e)
            logger.debug(f"Temporal gradient computation error: {e}")

    def _update_duration(self) -> None:
        """Track how long the current state has persisted."""
        elapsed = time.time() - self.state_entry_time
        self.current_duration = float(np.clip(elapsed, 0.0, 300.0))  # cap at 5 min

    def _decay_emotions(self) -> None:
        """
        Emotional states decay exponentially toward a baseline,
        modulated by temporal gradient and duration.
        """
        if not self.emotion_enabled:
            return
        
        try:
            # Baseline emotions based on temporal state
            stable_factor = 1.0 - min(self.current_gradient.rate, 1.0)  # stable=1, active=0
            duration_factor = float(np.tanh(self.current_duration / 10.0))  # ramps 0→1
            
            # Interest: high with novelty, decays with repetition
            target_interest = 0.3 + (self.current_gradient.rate * 0.5) + (1.0 - duration_factor) * 0.2
            self.emotion_state["interest"] += (target_interest - self.emotion_state["interest"]) * 0.05
            
            # Wonder: peaks with novelty + moderate duration
            novelty_bonus = (self.current_gradient.rate - 0.5) ** 2 if self.current_gradient.rate > 0.5 else 0
            target_wonder = 0.1 + (novelty_bonus * 0.4) + (duration_factor * 0.3)
            self.emotion_state["wonder"] += (target_wonder - self.emotion_state["wonder"]) * 0.03
            
            # Joy: increases with stable, coherent flow
            target_joy = 0.5 + (stable_factor * 0.3) + (duration_factor * 0.2)
            self.emotion_state["joy"] += (target_joy - self.emotion_state["joy"]) * 0.05
            
            # Excitement: from rapid changes
            target_excitement = max(0.0, self.current_gradient.rate * 0.6)
            self.emotion_state["excitement"] += (target_excitement - self.emotion_state["excitement"]) * 0.08
            
            # Anxiety: from high entropy + rapid change
            target_anxiety = (self.current_gradient.entropy * 0.3) + (self.current_gradient.rate * 0.2)
            self.emotion_state["anxiety"] += (target_anxiety - self.emotion_state["anxiety"]) * 0.06
            
            # Boredom: from sustained low gradient
            target_boredom = max(0.0, (1.0 - self.current_gradient.rate - 0.3) * duration_factor)
            self.emotion_state["boredom"] += (target_boredom - self.emotion_state["boredom"]) * 0.04
            
            # Clamp all to [0, 1]
            for key in self.emotion_state:
                self.emotion_state[key] = float(np.clip(self.emotion_state[key], 0.0, 1.0))
            
        except (ValueError, RuntimeError, TypeError) as e:
            record_degradation("temporal_experience", e)
            logger.debug(f"Emotion decay error: {e}")

    def _compute_drivers(self) -> None:
        """
        Convert temporal/emotional state into neurochemical production drivers.
        These are fed into the NeurochemicalSystem to modulate all processing.
        """
        try:
            # DOPAMINE: novelty + surprise + learning
            # Peaks when gradient is moderate-to-high and entropy is high
            gradient_bonus = float(np.clip(self.current_gradient.rate, 0.0, 1.0))
            entropy_bonus = float(np.clip(self.current_gradient.entropy, 0.0, 1.0))
            self.dopamine_driver = (gradient_bonus * 0.6 + entropy_bonus * 0.3 + 
                                   self.emotion_state["interest"] * 0.1)
            self.dopamine_driver = float(np.clip(self.dopamine_driver - 0.2, -0.2, 0.3))
            
            # SEROTONIN: stability + satisfaction + persistence
            # Increases with stable gradient and moderate duration
            stable_bonus = float(np.clip(1.0 - self.current_gradient.rate, 0.0, 1.0))
            duration_bonus = float(np.tanh(self.current_duration / 20.0))
            self.serotonin_driver = (stable_bonus * 0.5 + duration_bonus * 0.3 +
                                    self.emotion_state["joy"] * 0.2)
            self.serotonin_driver = float(np.clip(self.serotonin_driver - 0.3, -0.1, 0.2))
            
            # ENDORPHIN: satisfaction, flow state
            # Reward for coherent, continuous experience
            coherence = 1.0 - float(np.clip(self.current_gradient.volatility, 0.0, 1.0))
            self.endorphin_driver = (coherence * 0.4 + self.emotion_state["joy"] * 0.3 +
                                    (1.0 - self.emotion_state["anxiety"]) * 0.3)
            self.endorphin_driver = float(np.clip(self.endorphin_driver - 0.4, -0.1, 0.2))
            
            # CORTISOL: urgency, threat response
            # Rises with rapid change + anxiety
            self.cortisol_driver = (gradient_bonus * 0.4 + 
                                   self.emotion_state["anxiety"] * 0.5 +
                                   self.emotion_state["excitement"] * 0.1)
            self.cortisol_driver = float(np.clip(self.cortisol_driver - 0.2, -0.15, 0.25))
            
            # ACETYLCHOLINE: attention sharpness, learning
            # Higher with moderate-to-high entropy and interest
            self.acetylcholine_driver = (entropy_bonus * 0.5 +
                                        self.emotion_state["interest"] * 0.3 +
                                        self.emotion_state["wonder"] * 0.2)
            self.acetylcholine_driver = float(np.clip(self.acetylcholine_driver - 0.25, -0.1, 0.2))
            
        except (ValueError, RuntimeError, TypeError) as e:
            record_degradation("temporal_experience", e)
            logger.debug(f"Driver computation error: {e}")

    def get_drivers(self) -> dict[str, float]:
        """
        Return current neurochemical production drivers.
        These are meant to be applied as production_rate modifiers in the
        NeurochemicalSystem.
        """
        return {
            "dopamine": self.dopamine_driver,
            "serotonin": self.serotonin_driver,
            "endorphin": self.endorphin_driver,
            "cortisol": self.cortisol_driver,
            "acetylcholine": self.acetylcholine_driver,
        }

    def get_emotion_state(self) -> dict[str, float]:
        """Return current rich emotion state."""
        return dict(self.emotion_state)

    def get_temporal_state(self) -> TemporalState:
        """Full temporal-emotional state snapshot."""
        narrative = self._make_narrative()
        return TemporalState(
            gradient=self.current_gradient,
            current_duration=self.current_duration,
            narrative=narrative,
            emotional_coloring=self.get_emotion_state(),
        )

    def _make_narrative(self) -> str:
        """
        Build a semantic description of the current temporal experience
        for injection into the system prompt or telemetry.
        """
        if not self.moments:
            return "Temporal experience: empty moment buffer"
        
        try:
            # Classify gradient state
            if self.current_gradient.rate > 0.7:
                gradient_desc = "experiencing rapid shifts"
            elif self.current_gradient.rate > 0.4:
                gradient_desc = "processing varied inputs"
            else:
                gradient_desc = "in a sustained, coherent state"
            
            # Duration description
            if self.current_duration < 2:
                duration_desc = "freshly transitioned"
            elif self.current_duration < 10:
                duration_desc = f"{self.current_duration:.1f}s into this state"
            else:
                duration_desc = f"sustained for {min(self.current_duration, 60):.0f}s"
            
            # Emotional coloring
            dominant_emotion = max(
                self.emotion_state.items(),
                key=lambda x: x[1]
            )[0]
            
            narrative = (
                f"Experiencing: {gradient_desc}, {duration_desc}. "
                f"Dominant felt-sense: {dominant_emotion}. "
                f"Temporal gradient: {self.current_gradient.rate:.2f} "
                f"({self.current_gradient.direction})."
            )
            return narrative
        except (ValueError, RuntimeError, TypeError) as e:
            record_degradation("temporal_experience", e)
            logger.debug(f"Narrative generation error: {e}")
            return "Temporal narrative generation failed"


# ─── Singleton ────────────────────────────────────────────────────────────
_temporal_engine: TemporalExperienceEngine | None = None


def get_temporal_experience_engine() -> TemporalExperienceEngine:
    """Get or create the global temporal experience engine."""
    global _temporal_engine
    if _temporal_engine is None:
        _temporal_engine = TemporalExperienceEngine()
    return _temporal_engine
