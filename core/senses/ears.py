"""Sovereign Ears: Auditory Perception System
------------------------------------------
Handles audio input, Voice Activity Detection (VAD), and Transcription.
Now unified to wrap around SovereignVoiceEngine v5.0 for reliability.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from .voice_engine import VoiceState, get_voice_engine
from .sensory_registry import SensoryCapabilityFlags, get_capabilities

logger = logging.getLogger("Aura.Senses.Ears")

class SovereignEars:
    """Wrapper for the SovereignVoiceEngine to provide a consistent 'Ears' interface
    across the orchestrator.
    """

    def __init__(self, engine=None):
        from .sensory_client import get_sensory_client
        self.capabilities = get_capabilities()
        self.client = get_sensory_client()
        
        if engine:
            self._engine = engine
        else:
            from ..container import ServiceContainer
            self._engine = ServiceContainer.get("voice_engine", default=None)
        
        if not self.capabilities.hearing_enabled:
            logger.warning("👂 SovereignEars: Hearing is DISABLED (Capability Flag Off)")
        else:
            logger.info("👂 SovereignEars: Bridged to Isolated Sensory Process")

    def _resolve_engine(self):
        if self._engine:
            return self._engine
        try:
            from ..container import ServiceContainer

            self._engine = ServiceContainer.get("voice_engine", default=None)
        except Exception as e:
            record_degradation('ears', e)
            logger.debug("👂 SovereignEars: voice engine lookup deferred: %s", e)
        return self._engine

    def should_auto_listen(self) -> bool:
        engine = self._resolve_engine()
        return bool(
            self.capabilities.hearing_enabled
            and engine
            and getattr(engine, "should_auto_listen", lambda: False)()
        )

    async def start_listening(self, callback: Callable[[str], None]):
        """Starts capture only if capability is enabled."""
        engine = self._resolve_engine()
        if not self.capabilities.hearing_enabled or not engine:
            logger.warning("👂 SovereignEars: Cannot start listening (Missing capability or engine)")
            return
        
        # Verify worker is responsive before starting capture
        # This prevents library-level deadlocks from blocking the main thread
        
        async def _async_callback(text: str):
            res = callback(text)
            if asyncio.iscoroutine(res) or asyncio.isfuture(res) or hasattr(res, "__await__"):
                await res
            
        engine.on_transcript(_async_callback)
        await engine.start_listening()
        logger.info("👂 Ears listening (Guarded by Isolated Senses)")

    def transcribe(self, audio_source) -> str:
        """Transcribe audio from a file path or array using the VoiceEngine's model.
        This is a synchronous wrapper for the engine's STT (faster-whisper).
        """
        engine = self._resolve_engine()
        if not engine:
            return ""
            
        engine.ensure_stt()

        # Access the model directly for sync calls (Faster-WhisperSegments)
        if hasattr(engine, 'stt_model') and engine.stt_model:
            logger.info("👂 Ears: Synchronous transcription requested.")
            segments, _ = engine.stt_model.transcribe(
                audio_source,
                language="en",
                beam_size=5
            )
            return " ".join([seg.text for seg in segments]).strip()
        
        return ""

    def mock_hear(self, text: str):
        """Inject text as if heard (for testing)."""
        if not (self._engine and hasattr(self._engine, "_on_transcript") and self._engine._on_transcript):
            return

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._engine._on_transcript(text))
        except RuntimeError:
            # No loop running, but we should not use asyncio.run inside this library
            # as it often collides with the larger service lifecycle.
            logger.warning("mock_hear: No running event loop. Transcript not dispatched.")
        except Exception as e:
            record_degradation('ears', e)
            logger.error("Mock hear failed: %s", e)
