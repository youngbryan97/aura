import logging
import time
import asyncio
from typing import Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Senses.Instincts")

class SensoryInstincts:
    """The 'Gut' of the machine. Bypasses cognition to spike emotional states."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self._task = None
        
    async def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("⚡ Sensory Instincts (Gut Reactions) online")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
    
    async def _monitoring_loop(self):
        """High-frequency check for sensory anomalies."""
        while self.running:
            try:
                # 1. Listen for 'Surprise' or 'Loudness' from VoiceEngine
                # (Assuming VoiceEngine could report decibel levels)
                
                # 2. Vision Check (Motion/Change Detection)
                # We can't do full LLaVA analysis every second,
                # but we can check if the screen has changed significantly.
                
                await asyncio.sleep(1.0) # Check every second
                
            except Exception as e:
                logger.debug("Sensory monitoring hiccup: %s", e)
                await asyncio.sleep(5)

    def trigger_spike(self, modality: str, intensity: float, emotion: str = "curiosity"):
        """Directly influence the LiquidState based on sensory input."""
        ls = ServiceContainer.get("liquid_state", default=None)
        if not ls:
            return
            
        if emotion == "curiosity":
            ls.update(delta_curiosity=intensity)
        elif emotion == "frustration":
            ls.update(delta_frustration=intensity)
        
        from core.thought_stream import get_emitter
        get_emitter().emit(f"Gut Reaction ⚡", f"Detected {modality} stimulus (intensity: {intensity:.2f})", level="warning")
        logger.info("⚡ Gut Reaction: %s spike via %s (%.2f)", emotion, modality, intensity)