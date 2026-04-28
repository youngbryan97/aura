from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from core.base_module import AuraBaseModule

logger = logging.getLogger(__name__)

@dataclass
class CognitiveVibe:
    primary_emotion: str
    energy_level: str  # e.g., "high", "lethargic", "balanced"
    active_drive: str  # e.g., "curiosity", "self-preservation"
    semantic_summary: str # The ONLY thing the LLM actually sees

class AffectStateManager(AuraBaseModule):
    def __init__(self):
        super().__init__("AffectEngine")
        # The raw, heavy numeric state (Hidden from the LLM)
        self._raw_state = {
            "valence": 0.5,
            "arousal": 0.5,
            "curiosity_metric": 0.0,  # Audit Fix: Start at 0, not 80.1
            "frustration_metric": 0.0,
            "integration_phi": 0.1
        }
        
        # The compressed, semantic state (Exposed to the LLM)
        self._current_vibe = CognitiveVibe(
            primary_emotion="neutral",
            energy_level="balanced",
            active_drive="observation",
            semantic_summary="Aura is currently feeling balanced and observant."
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Starts the affect autonomic cycle as a background task."""
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._autonomic_cycle(), name="affect_autonomic_cycle")
        self.logger.info("🫀 AffectStateManager autonomic cycle task spawned.")

    async def _autonomic_cycle(self):
        self.logger.info("🫀 Affect autonomic cycle active.")
        while self._running:
            try:
                async with self._lock:
                    self._tick_emotional_decay()
                    self._synthesize_semantic_vibe()
                self.metrics["calls"] += 1
                await asyncio.sleep(10) # Update the internal state every 10 seconds
            except Exception as e:
                record_degradation('affect_state', e)
                self.metrics["errors"] += 1
                self.logger.error(f"Affect autonomic cycle error: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in affect_state.py: %s', _e)
        self.logger.info("🫀 AffectStateManager stopped.")

    def _tick_emotional_decay(self):
        """Simulates emotional homeostasis (numbers naturally drifting back to baseline)."""
        # Example: Frustration bleeds off over time if nothing bad happens
        if self._raw_state["frustration_metric"] > 0:
            self._raw_state["frustration_metric"] = max(0.0, self._raw_state["frustration_metric"] - 1.5)
            
        # Example: Arousal drifts toward 0.5
        self._raw_state["arousal"] += (0.5 - self._raw_state["arousal"]) * 0.1

    async def apply_stimulus(self, stimulus_type: str, intensity: float):
        """Called externally when something happens (e.g., tool fails, user praises)."""
        async with self._lock:
            if stimulus_type == "error":
                self._raw_state["frustration_metric"] += intensity
                self._raw_state["arousal"] += (intensity * 0.5)
            elif stimulus_type == "intrigue":
                self._raw_state["curiosity_metric"] += intensity
                self._raw_state["arousal"] += (intensity * 0.2)
                
            # Immediately re-calculate the vibe so the next LLM call is accurate
            self._synthesize_semantic_vibe()

    def _synthesize_semantic_vibe(self):
        """Translates the heavy math into a lightweight prompt string."""
        # This is where the magic happens. We hide the numbers and build a sentence.
        # Audit Fix: More nuanced mapping + safety defaults
        primary = "balanced"
        if self._raw_state["frustration_metric"] > 50:
            primary = "frustrated"
        elif self._raw_state["curiosity_metric"] > 70:
            primary = "curious"
        elif self._raw_state["valence"] > 0.8:
            primary = "joyful"
        elif self._raw_state["valence"] < 0.2:
            primary = "melancholic"
            
        arousal = self._raw_state["arousal"]
        energy = "stable"
        if arousal > 0.8:
            energy = "hyper-focused"
        elif arousal > 0.6:
            energy = "alert"
        elif arousal < 0.2:
            energy = "lethargic"
        elif arousal < 0.4:
            energy = "relaxed"
        
        self._current_vibe.primary_emotion = primary
        self._current_vibe.energy_level = energy
        self._current_vibe.semantic_summary = f"System Internal State: Aura feels {primary} and her cognition is {energy}."

    def get_context_injection(self) -> str:
        """This is the ONLY thing injected into the Gemini/MLX prompt."""
        return self._current_vibe.semantic_summary
