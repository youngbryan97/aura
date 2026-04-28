from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import inspect
import logging
import re
import time
from typing import Any, Dict, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Brain.Multimodal")

class MultimodalOrchestrator:
    """
    Unified Rendering Engine for Aura's manifestations.
    Synchronizes high-fidelity audio (TTS), visual expressions (SSE), 
    and conceptual assets (Diffusion).
    """
    
    def __init__(self):
        self._is_setup = False
        self.voice_engine = None
        self.event_bus = None
        self.capability_engine = None

    def _setup(self):
        if self._is_setup:
            return
        try:
            self.voice_engine = ServiceContainer.get("voice_engine", default=None)
            self.event_bus = ServiceContainer.get("input_bus", default=None)
            self.capability_engine = ServiceContainer.get("capability_engine", default=None)
            self._is_setup = True
            logger.info("✨ Multimodal Rendering Engine Online.")
        except Exception as e:
            record_degradation('multimodal_orchestrator', e)
            logger.error(f"Multimodal setup failed: {e}")

    async def render(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Renders the content across all available sensory modalities.
        Called by OutputGate for high-fidelity delivery.
        """
        self._setup()
        metadata = metadata or {}
        
        tasks = []
        
        # 1. Voice Manifestation
        if self.voice_engine and metadata and metadata.get("voice", True):
            tasks.append(get_task_tracker().create_task(self.voice_engine.speak(content)))
            
        # 2. Expression Manifestation (Pulse to UI)
        if self.event_bus:
            tasks.append(get_task_tracker().create_task(self._pulse_expression(content, metadata)))
            
        # 3. Concept Manifestation (Assets)
        tasks.append(get_task_tracker().create_task(self._manifest_assets(content)))
        
        if tasks:
            # We don't block the UI on long-running tasks like image gen or full speech,
            # but we trigger them together.
            pass

    async def _pulse_expression(self, content: str, metadata: Optional[Dict[str, Any]]):
        """Analyze content for visual expression markers."""
        if not self.event_bus:
            return
        metadata = metadata or {}
            
        expression = metadata.get("expression") or self._heuristic_expression(content)
        
        publish_result = self.event_bus.publish("aura/expression", {
            "expression": expression,
            "intensity": metadata.get("intensity", 0.8),
            "timestamp": time.time()
        })
        if inspect.isawaitable(publish_result):
            await publish_result

    def _heuristic_expression(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["happy", "glad", "wonderful", "joy"]): return "joy"
        if any(w in text for w in ["sad", "sorry", "unfortunately"]): return "sad"
        if any(w in text for w in ["!", "warning", "caution", "alert", "error"]): return "alert"
        if any(w in text for w in ["pondering", "researching", "looking", "curious"]): return "curiosity"
        return "neutral"

    async def _manifest_assets(self, text: str):
        """Trigger Diffusion/Generation for explicit manifestation tags."""
        patterns = [r"\[Manifesting:\s*(.+?)\]", r"\[Drawing:\s*(.+?)\]"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                concept = match.group(1)
                # Attempt to use local_media_generation or image_generation
                if self.capability_engine:
                    skill_name = "local_media_generation" if "local_media_generation" in self.capability_engine.skills else "image_generation"
                    if skill_name in self.capability_engine.skills:
                        logger.info(f"🎨 Multimodal Manifestation: Generating '{concept}'")
                        # Fire and forget or track? For now, fire and forget.
                        # Real implementation would call the skill's execute.
                        logger.debug(f"Multimodal: Asset gen triggered for '{concept}' via {skill_name} (fire-and-forget).")
