# skills/listen.py
# AURA v5.3: Sovereign Listener (Local-Only)

import asyncio
import logging
import os
import tempfile
import threading
import time
import wave
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

import numpy as np
import sounddevice as sd

from core.skills.base_skill import BaseSkill

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
    try:
        # Pre-warm PortAudio
        sd.query_devices()
        logger.info("PortAudio subsystem pre-warmed.")
        _AUDIO_INITIALIZED = True
    except Exception as e:
        logger.error("Audio initialization failed: %s", e)
        raise

def _get_default_input_device():
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if "Microphone" in device['name'] or "Built-in" in device['name']:
                    return i
        return sd.default.device[0]
    except Exception as e:
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
        
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype='int16',
            device=device
        )
        sd.wait()
        
        # Save to temp WAV file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, 'wb') as tmp:
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) 
                    wf.setframerate(fs)
                    wf.writeframes(recording.tobytes())
            return temp_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

# ----------------------------------------------------
# AudioListenerSkill Implementation
# ----------------------------------------------------

class ListenInput(BaseModel):
    duration: float = Field(5.0, description="Seconds to listen for audio input.")

class AudioListenerSkill(BaseSkill):
    name = "listen"
    description = "Listen to microphone input and transcribe speech to text using local Whisper."
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
            except Exception as e:
                logger.error("Failed to resolve voice_engine: %s", e)
        return self._voice_engine

    async def execute(self, params: ListenInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio capture and transcription."""
        if isinstance(params, dict):
            try:
                params = ListenInput(**params)
            except Exception as e:
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
            
            engine = self._get_engine()
            if not engine:
                return {"ok": False, "error": "Sovereign Voice Engine unavailable."}
                
            try:
                # Transcribe using unified voice engine
                text = await asyncio.to_thread(engine.transcribe, temp_wav)
            except Exception as e:
                logger.error("Unified transcription failed: %s", e)
                text = f"[Audio Recorded, Unified Transcription Failed: {e}]"
            
            # Cleanup
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except OSError as _e:
                logger.debug('Ignored OSError in listen.py: %s', _e)

            return {
                "ok": True,
                "transcription": text,
                "summary": f"Heard: {text}"
            }
            
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": "Microphone access timed out."
            }
        except Exception as e:
            logger.error("Audio capture failed: %s", e)
            return {"ok": False, "error": f"Audio capture failed: {e}"}
