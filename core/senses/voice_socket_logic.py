import asyncio
import logging
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.VoiceProcessor")


class _CanonicalWhisperProxy:
    """Expose Whisper's API without leaking the canonical raw model reference."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def transcribe(self, *args: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        session = getattr(self._engine, "stt_model_session", None)
        if not callable(session):
            raise RuntimeError("canonical_voice_engine_missing_stt_session")
        with session() as model:
            if model is None:
                raise RuntimeError("canonical_whisper_model_unavailable")
            segments, info = model.transcribe(*args, **kwargs)
            return list(segments), info

def get_whisper_model(model_name="tiny"):
    """Return a governed proxy for the canonical voice engine's STT model."""

    engine = get_runtime_service("voice_engine", default=None)
    if engine is None:
        try:
            from core.senses.voice_engine import get_voice_engine

            engine = get_voice_engine()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("voice_socket_logic", exc)
            logger.error("Canonical voice engine unavailable; websocket STT disabled: %s", exc)
            return None
    try:
        configured = str(getattr(engine, "whisper_model_name", "") or "")
        if configured and configured != str(model_name):
            logger.debug(
                "Websocket requested Whisper %s; sharing canonical configured model %s",
                model_name,
                configured,
            )
        if not bool(engine.ensure_stt()):
            return None
        if getattr(engine, "stt_model", None) is None:
            return None
        return _CanonicalWhisperProxy(engine)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("voice_socket_logic", exc)
        logger.error("Canonical Whisper model unavailable: %s", exc)
        return None

class VoiceStreamProcessor:
    """Stateful audio processor for Advanced Voice Mode.
    Handles VAD logic and local STT.
    """

    def __init__(self, model_name="tiny", model_instance=None):
        self.vad = None
        
        self.sample_rate = 16000
        self.frame_duration_ms = 20
        self.bytes_per_frame = int(self.sample_rate * (self.frame_duration_ms / 1000) * 2)
        self.buffer = b""
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.max_silence = 25 # ~500ms

        # STT Model (Local)
        if model_instance:
            self.model = model_instance
        else:
            self.model = get_whisper_model(model_name)

    def add_audio(self, chunk: bytes):
        self.buffer += chunk

    def is_speech_finished(self) -> bool:
        """Processes buffer and returns True if a complete utterance is detected."""
        while len(self.buffer) >= self.bytes_per_frame:
            frame = self.buffer[:self.bytes_per_frame]
            self.buffer = self.buffer[self.bytes_per_frame:]
            
            # VAD logic is now handled by energy-based detection in the frontend or this fallback
            # calculates RMS of the frame and compares against a threshold
            try:
                audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(audio_np**2))
                is_speech = rms > 300 # Threshold for local sensitivity
            # not a failure: a frame this cannot measure is not speech, and False is the
            # refusing direction for a voice gate.
            except (RuntimeError, AttributeError, TypeError, ValueError):
                is_speech = False

            if is_speech:
                if not self.is_speaking:
                    logger.info("🎤 Speech detected...")
                self.is_speaking = True
                self.silence_frames = 0
                self.speech_buffer.append(frame)
            elif self.is_speaking:
                self.speech_buffer.append(frame)
                self.silence_frames += 1
                if self.silence_frames >= self.max_silence:
                    logger.info("🛑 End of speech detected.")
                    return True
        return False

    async def get_transcript(self) -> str:
        """Transcribes the captured speech buffer using local Whisper."""
        if not self.model or not self.speech_buffer:
            return ""
        
        try:
            audio_data = b"".join(self.speech_buffer)
            # Convert 16-bit PCM to float32 normalized
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run transcription in a thread to keep the event loop alive
            def _sync_transcribe():
                segments, _ = self.model.transcribe(audio_np, beam_size=1)
                return "".join([s.text for s in segments]).strip()

            text = await asyncio.to_thread(_sync_transcribe)
            
            # Phase 4: Spinal Cord Reflex Engine (Audio Interrupts)
            orch = get_runtime_service("orchestrator", default=None)
            if orch and hasattr(orch, "reflex_engine") and orch.reflex_engine:
                clean_t = text.upper().strip()
                # Remove punctuation for emergency matching
                import re
                clean_t = re.sub(r'[^\w\s]', '', clean_t)
                
                if clean_t in ("STOP", "HALT", "ABORT", "CANCEL", "SHUT UP", "STOP TALKING", "QUIET"):
                    logger.critical("🚨 [VOICE] Emergency Interrupt Detected: '%s'", text)
                    await orch.reflex_engine.process_emergency_interrupt(clean_t, context="audio_stream")
                    # If it's a STOP command, we might not even want to return the text to the main
                    # cognitive loop, or we return a specific token. For now, returning it is fine
                    # because the reflex engine already purged the action queue.
            
            self.reset()
            return text
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('voice_socket_logic', e)
            logger.error("Transcription error: %s", e)
            self.reset()
            return ""

    def reset(self):
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.buffer = b""
