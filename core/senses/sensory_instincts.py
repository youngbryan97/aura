from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
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
        self._task = get_task_tracker().create_task(self._monitoring_loop())
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
                record_degradation('sensory_instincts', e)
                logger.debug("Sensory monitoring hiccup: %s", e)
                await asyncio.sleep(5)

    def trigger_spike(self, modality: str, intensity: float, emotion: str = "curiosity"):
        """Directly influence the LiquidState based on sensory input."""
        applied_intensity = max(0.0, float(intensity))
        try:
            authority = ServiceContainer.get("substrate_authority", default=None)
            if authority is not None:
                from core.consciousness.substrate_authority import (
                    ActionCategory,
                    AuthorizationDecision,
                )

                verdict = authority.authorize(
                    content=f"{emotion}:{modality}:{applied_intensity:.2f}",
                    source="sensory_instincts",
                    category=ActionCategory.STATE_MUTATION,
                    priority=min(1.0, applied_intensity),
                    is_critical=False,
                )
                if verdict.decision == AuthorizationDecision.BLOCK:
                    logger.info("⚡ Gut Reaction blocked by substrate authority (%s)", verdict.reason)
                    return False
                if verdict.decision == AuthorizationDecision.CONSTRAIN:
                    applied_intensity *= 0.5
        except Exception as exc:
            record_degradation('sensory_instincts', exc)
            logger.debug("SensoryInstincts authority gate unavailable: %s", exc)

        ls = ServiceContainer.get("liquid_state", default=None)
        if not ls:
            return False
            
        if emotion == "curiosity":
            ls.update(delta_curiosity=applied_intensity)
        elif emotion == "frustration":
            ls.update(delta_frustration=applied_intensity)
        
        from core.thought_stream import get_emitter
        get_emitter().emit(f"Gut Reaction ⚡", f"Detected {modality} stimulus (intensity: {applied_intensity:.2f})", level="warning")
        logger.info("⚡ Gut Reaction: %s spike via %s (%.2f)", emotion, modality, applied_intensity)
        return True
