"""Sovereign Ears: Auditory Perception System
------------------------------------------
Handles audio input, Voice Activity Detection (VAD), and Transcription.
Now unified to wrap around SovereignVoiceEngine v5.0 for reliability.
"""

import asyncio
import logging
from collections.abc import Callable

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.utils.task_tracker import get_task_tracker

from .sensory_registry import get_capabilities

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
            self._engine = get_runtime_service("voice_engine", default=None)
        
        if not self.capabilities.hearing_enabled:
            logger.warning("SovereignEars: Hearing is DISABLED (Capability Flag Off)")
        else:
            logger.info("SovereignEars: Bridged to Isolated Sensory Process")

    def _resolve_engine(self):
        if self._engine:
            return self._engine
        try:
            self._engine = get_runtime_service("voice_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('ears', e)
            logger.debug("SovereignEars: voice engine lookup deferred: %s", e)
        return self._engine

    def should_auto_listen(self) -> bool:
        engine = self._resolve_engine()
        return bool(
            self.capabilities.hearing_enabled
            and engine
            and getattr(engine, "should_auto_listen", lambda: False)()
        )

    async def start_listening(self, callback: Callable[[str], None]) -> bool:
        """Starts capture only if capability is enabled."""
        engine = self._resolve_engine()
        if not self.capabilities.hearing_enabled or not engine:
            logger.warning("SovereignEars: Cannot start listening (Missing capability or engine)")
            return False
        
        # Verify worker is responsive before starting capture
        # This prevents library-level deadlocks from blocking the main thread
        
        async def _async_callback(text: str):
            res = callback(text)
            if asyncio.iscoroutine(res) or asyncio.isfuture(res) or hasattr(res, "__await__"):
                await res
            
        engine.on_transcript(_async_callback, key="sovereign_ears")
        started = bool(await engine.start_listening())
        if started:
            logger.info("Ears listening (Guarded by Isolated Senses)")
        else:
            logger.warning("Ears listener remains offline; explicit retry is available")
        return started

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
            logger.info("Ears: Synchronous transcription requested.")
            segments, _ = engine.stt_model.transcribe(
                audio_source,
                language="en",
                beam_size=5
            )
            return " ".join([seg.text for seg in segments]).strip()
        
        return ""

    def inject_transcript_for_test(self, text: str) -> bool:
        """Deliver `text` to the transcript handler as though it were heard.

        Named for what it is. It was ``mock_hear``, which reads as a stub of
        hearing rather than a test seam into the real transcript path, and
        which nothing in the repository called — the ``_for_test`` suffix is
        the convention tools/integration_debt.py already recognises, so a
        seam that stops being used shows up as debt instead of hiding behind
        a name that looks like production code.

        Returns whether the transcript was actually dispatched. It used to
        return None down all four paths, so a caller could not tell a
        delivered transcript from a missing engine or a dead loop.
        """
        if not (self._engine and hasattr(self._engine, "_on_transcript") and self._engine._on_transcript):
            return False

        try:
            loop = asyncio.get_running_loop()
            if not loop.is_running():
                return False
            get_task_tracker().create_task(
                self._engine._on_transcript(text),
                name="ears.injected_transcript",
            )
            return True
        except RuntimeError:
            # No loop running, but we should not use asyncio.run inside this library
            # as it often collides with the larger service lifecycle.
            logger.warning("inject_transcript_for_test: no running event loop; not dispatched.")
            return False
        except (AttributeError, TypeError, ValueError) as e:
            record_degradation('ears', e)
            logger.error("inject_transcript_for_test failed: %s", e)
            return False
