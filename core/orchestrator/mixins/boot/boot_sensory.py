import asyncio
import logging
from typing import Any

from core.container import ServiceContainer

logger = logging.getLogger(__name__)


class BootSensoryMixin:
    """Provides initialization for sensory inputs & barrier systems."""

    terminal_monitor: Any
    reasoning_queue: Any
    instincts: Any

    async def _init_sensory_systems(self):
        """Initialize ears and other sensory inputs."""
        # Defer intensive sensory IO to background tasks
        try:
            from core.senses.ears import SovereignEars
            from core.senses.screen_vision import LocalVision

            # 1. Ears (Hearing)
            async def _init_ears():
                try:
                    ears = SovereignEars()
                    ServiceContainer.register_instance("ears", ears)
                    logger.info("👂 Sovereign Ears Active")
                except Exception as e:
                    logger.error("👂 Ears init failed: %s", e)

            # 2. Vision (Eyes)
            async def _init_vision():
                try:
                    vision = LocalVision()
                    # Register as both names to satisfy different client components
                    ServiceContainer.register_instance("vision_engine", vision)
                    ServiceContainer.register_instance("vision", vision)
                    logger.info("👁️  Sovereign Vision Active")
                except Exception as e:
                    logger.error("👁️  Vision init failed: %s", e)

            # Defer sensory IO to background but keep them as tasks
            await asyncio.gather(_init_ears(), _init_vision())

            from core.terminal_monitor import get_terminal_monitor

            self.terminal_monitor = get_terminal_monitor()
            ServiceContainer.register_instance("terminal_monitor", self.terminal_monitor)

            # [Phase 14] Immune System & Sanitization
            from core.adaptation.immune_system import ImmuneSystem
            from core.utils.sanitizer import BloodBrainBarrier

            ServiceContainer.register_instance("immune_system", ImmuneSystem())
            ServiceContainer.register_instance("blood_brain_barrier", BloodBrainBarrier())

            # Start Background Reasoning Queue
            from core.brain.reasoning_queue import get_reasoning_queue

            self.reasoning_queue = get_reasoning_queue()
            # Task creation deferred to _start_sensory_systems
            logger.info("🧠 Background Reasoning Queue Ready (Start Deferred)")

            # Sensory Instincts (v11.0 Gut Reactions)
            try:
                from core.senses.sensory_instincts import SensoryInstincts

                self.instincts = SensoryInstincts(self)
                logger.info("✓ Sensory Instincts initialized")
            except Exception as e:
                logger.error("Failed to init Sensory Instincts: %s", e)
                self.instincts = None
        except Exception as e:
            logger.error("🛑 Halting: Critical validation failures.")
            self.terminal_monitor = None

    async def _start_sensory_systems(self):
        if hasattr(self, "reasoning_queue") and self.reasoning_queue:
            # H-17 FIX: Track background tasks
            from core.utils.task_tracker import get_task_tracker

            get_task_tracker().track(self.reasoning_queue.start(), name="reasoning_queue"
            )
            logger.info("🧠 Background Reasoning Queue Started")

    async def _init_voice_subsystem(self):
        """Initialize the Voice Engine & Multimodal Orchestrator in the background."""
        from core.senses.voice_engine import get_voice_engine

        async def _init_voice():
            try:
                voice = get_voice_engine()
                # Warm TTS only. STT stays cold until the user explicitly enables voice input.
                if hasattr(voice, "ensure_tts_async"):
                    await voice.ensure_tts_async()
                else:
                    await voice.ensure_models_async()
                ServiceContainer.register_instance("voice_engine", voice)
                logger.info("🎙️  Voice Engine initialized and registered in background")
            except Exception as e:
                logger.error("🛑 Voice Engine background init failed: %s", e)

        get_task_tracker().create_task(_init_voice())

        from core.brain.multimodal_orchestrator import MultimodalOrchestrator

        ServiceContainer.register_instance(
            "multimodal_orchestrator", MultimodalOrchestrator()
        )
