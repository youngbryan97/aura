# skills/listen.py
# AURA v5.3: Sovereign Listener (Local-Only)

import asyncio
import logging
import struct
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import record_degradation

try:
    import sounddevice as sd
    _SOUNDDEVICE_IMPORT_ERROR: Exception | None = None
except (ImportError, AttributeError, RuntimeError) as exc:  # pragma: no cover - depends on host audio bindings
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc

from core.exceptions import ContainerError
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill
from core.voice.microphone_authority import record_sounddevice_array

logger = logging.getLogger("Skills.Audio")

# ----------------------------------------------------
# Global Audio State
# ----------------------------------------------------
_AUDIO_LOCK = threading.Lock()
_AUDIO_INITIALIZED = False
MIC_TIMEOUT_SECONDS = 10

# ----------------------------------------------------
# Audio Initialization (Safe for macOS)
# ----------------------------------------------------

def _initialize_audio():
    global _AUDIO_INITIALIZED
    if _AUDIO_INITIALIZED:
        return
    if sd is None:
        error = RuntimeError(f"sounddevice unavailable: {_SOUNDDEVICE_IMPORT_ERROR}")
        record_degradation('listen', error)
        logger.error("Audio initialization failed: %s", error)
        raise error
    try:
        # Pre-warm PortAudio
        sd.query_devices()
        logger.info("PortAudio subsystem pre-warmed.")
        _AUDIO_INITIALIZED = True
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        record_degradation('listen', e)
        logger.error("Audio initialization failed: %s", e)
        raise

def _get_default_input_device():
    if sd is None:
        error = RuntimeError(f"sounddevice unavailable: {_SOUNDDEVICE_IMPORT_ERROR}")
        record_degradation('listen', error)
        logger.error("Error querying audio devices: %s", error)
        raise RuntimeError(f"No audio input device available. Ensure a microphone is connected. Details: {error}") from error
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if "Microphone" in device['name'] or "Built-in" in device['name']:
                    return i
        return sd.default.device[0]
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        record_degradation('listen', e)
        logger.error("Error querying audio devices: %s", e)
        raise RuntimeError(f"No audio input device available. Ensure a microphone is connected. Details: {e}") from e

# ----------------------------------------------------
# Recording (Thread Safe)
# ----------------------------------------------------

def _record_sync(duration: float, fs: int = 16000) -> str:
    """Synchronous recording — Faster-Whisper prefers 16kHz."""
    with _AUDIO_LOCK:
        _initialize_audio()
        
        device = _get_default_input_device()
        if device is None:
            logger.error("❌ No input device found!")
            raise RuntimeError("No input device available")

        device_info = sd.query_devices(device)
        logger.info("🎙️ Selected Device: %s (Index: %s)", device_info['name'], device)
        
        # Set explicitly for macOS stability
        sd.default.device = (device, None)
        sd.default.samplerate = fs
        sd.default.channels = 1

        logger.info("🎙️ Recording from device %s at %sHz for %ss...", device, fs, duration)
        
        recording = record_sounddevice_array(
            sd,
            holder="skills.listen",
            source="listen_skill",
            mode="focused",
            frames=int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="int16",
            device=device,
            preemptible=False,
        )
        
        # Save to temp WAV file through the canonical write gateway.
        temp_path = str(Path(tempfile.gettempdir()) / f"aura_listen_{uuid.uuid4().hex}.wav")
        try:
            get_file_write_gateway().write_bytes(
                temp_path,
                _wav_bytes(recording, fs),
                source="skills.listen.recording",
            )
            return temp_path
        except OSError as e:
            record_degradation('listen', e)
            Path(temp_path).unlink(missing_ok=True)
            raise e


def _wav_bytes(recording: np.ndarray, fs: int) -> bytes:
    """Encode mono int16 PCM samples as a WAV byte stream."""
    pcm = recording.astype("<i2", copy=False).tobytes()
    channels = 1
    sample_width = 2
    bits_per_sample = sample_width * 8
    byte_rate = int(fs) * channels * sample_width
    block_align = channels * sample_width
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, int(fs), byte_rate, block_align, bits_per_sample)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )

# ----------------------------------------------------
# AudioListenerSkill Implementation
# ----------------------------------------------------

class ListenInput(BaseModel):
    duration: float = Field(5.0, description="Seconds to listen for audio input.")

class AudioListenerSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "listen"
    description = "Listen to microphone input and transcribe speech to text using local Whisper."
    effect_scope = "read_write_artifacts"
    input_model = ListenInput

    def __init__(self):
        super().__init__()
        self._voice_engine = None
        
    def _get_engine(self):
        """Resolve voice engine from ServiceContainer."""
        if self._voice_engine is None:
            try:
                from core.container import ServiceContainer
                self._voice_engine = ServiceContainer.get("voice_engine")
            except (ImportError, AttributeError, RuntimeError, ContainerError) as e:
                record_degradation('listen', e)
                logger.error("Failed to resolve voice_engine: %s", e)
        return self._voice_engine

    async def execute(self, params: ListenInput, context: dict[str, Any]) -> dict[str, Any]:
        """Execute audio capture and transcription."""
        if isinstance(params, dict):
            try:
                params = ListenInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('listen', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        duration = float(params.duration)
        fs = 16000 # Standard for Whisper
        
        try:
            # Record
            temp_wav = await asyncio.wait_for(
                asyncio.to_thread(_record_sync, duration, fs),
                timeout=duration + MIC_TIMEOUT_SECONDS
            )
            
            logger.info("Audio captured: %s. Transcribing locally...", temp_wav)
            
            try:
                engine = self._get_engine()
                if not engine:
                    return {"ok": False, "error": "Sovereign Voice Engine unavailable."}
                try:
                    text = await asyncio.to_thread(engine.transcribe, temp_wav)
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation(
                        "listen",
                        e,
                        severity="warning",
                        action="returned recorded-audio fallback after unified transcription failed",
                        extra={"duration_s": duration},
                    )
                    logger.error("Unified transcription failed: %s", e)
                    text = f"[Audio Recorded, Unified Transcription Failed: {e}]"

                return {
                    "ok": True,
                    "transcription": text,
                    "summary": f"Heard: {text}",
                }
            finally:
                try:
                    await get_file_write_gateway().delete_path_async(
                        temp_wav, source="skills.listen.cleanup"
                    )
                except OSError as exc:
                    logger.debug("Temporary audio cleanup failed: %s", exc)
            
        except TimeoutError:
            return {
                "ok": False,
                "error": "Microphone access timed out."
            }
        except (RuntimeError, AttributeError) as e:
            record_degradation('listen', e)
            logger.error("Audio capture failed: %s", e)
            return {"ok": False, "error": f"Audio capture failed: {e}"}
