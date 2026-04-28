"""
SovereignVoiceEngine v5.0 — Server-Side Capture + Mycelial Roots
=================================================================

This version captures audio directly from the system microphone using
`sounddevice`, completely bypassing the browser's getUserMedia (which
fails in PyWebView on macOS due to WebKit permission restrictions).

Architecture:
  System Mic → sounddevice callback → PCM buffer → Whisper STT → transcript
  transcript → EventBus("user_input") → Orchestrator
  
  All transitions pulse the Mycelial Network hyphae for real-time
  connectivity tracking and Soul Graph visualization.

TTS uses pyttsx3 (macOS native NSSpeechSynthesizer under the hood).
"""

from core.runtime.errors import record_degradation
import base64

from core.utils.exceptions import capture_and_log
from core.utils.concurrency import RobustLock
import subprocess
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

import numpy as np
import logging
import queue
import asyncio

# ── Optional imports with graceful degradation ────────────
_WhisperModel = None
_whisper_import_attempted = False

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None

import io
import wave
import urllib.request
import logging
import os

logger = logging.getLogger("Aura.VoiceEngine")


def _get_whisper_model_class():
    """Import faster-whisper on demand so STT does not preload PyAV at module import."""
    global _WhisperModel, _whisper_import_attempted
    if _WhisperModel is not None:
        return _WhisperModel
    if _whisper_import_attempted:
        return None

    _whisper_import_attempted = True
    try:
        from faster_whisper import WhisperModel as whisper_model_cls
        _WhisperModel = whisper_model_cls
    except ImportError:
        logger.error("❌ faster-whisper not installed — STT unavailable")
    except Exception as exc:
        record_degradation('voice_engine', exc)
        logger.error("❌ faster-whisper import failed — STT unavailable: %s", exc)
    return _WhisperModel

try:
    from TTS.api import TTS
except Exception as e:
    TTS = None
    e_str = str(e)
    # [STABILITY] Silence redundant torchcodec/transformers warnings on macOS
    if "torchcodec" in e_str or "isin_mps_friendly" in e_str:
        logger.debug("TTS Import: Suppressing expected library quirk: %s", e_str)
    elif "No module named 'TTS'" in e_str and pyttsx3 is not None:
        logger.info("TTS backend unavailable; native pyttsx3 fallback will be used.")
    else:
        logger.warning("TTS Import Error: %s", e_str)

# ── Constants ─────────────────────────────────────────────
SAMPLE_RATE = 16000       # 16kHz for Whisper
CHANNELS = 1              # Mono
BLOCK_SIZE = 1600         # ~100ms chunks at 16kHz
SILENCE_THRESHOLD = 0.01  # RMS below this = silence
SILENCE_TIMEOUT = 1.5     # seconds of silence before processing
MIN_AUDIO_LENGTH = 0.5    # minimum seconds of audio to process
MAX_AUDIO_LENGTH = 15.0   # maximum seconds before forced processing


class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class SovereignVoiceEngine:
    """Aura's Ears & Mouth — Server-side capture with Mycelial integration.
    
    Key difference from v4.0: captures audio directly from the system
    microphone_microphone using sounddevice, so no browser permissions are needed.
    """
    
    _NOISE_PHRASES = frozenset({
        # Whisper hallucinations
        "thank you", "thanks for watching", "you", "bye",
        "the", "it", "a", "hmm", "um", "uh", "oh",
        "it they were", "with the fact that",
        # Common TV/YouTube phrases
        "subscribe", "like and subscribe", "hit that bell",
        "thanks for tuning in", "welcome back",
        "stay tuned", "coming up next", "we'll be right back",
        "this is", "breaking news", "let's go",
        "and that's", "so what do you think",
        # Short/meaningless
        "okay", "alright", "yeah", "yes", "no", "right",
        "i mean", "you know", "so", "well",
    })

    def __init__(self,
                 whisper_model: str = "base",
                 data_dir: Optional[str] = None):
        from core.common.paths import DATA_DIR
        self.data_dir = Path(data_dir or (DATA_DIR / "voice_models"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ── STT State ─────────────────────────────────────
        self.stt_model = None
        self._stt_initialized = False
        self._audio_buffer = queue.Queue()
        self.whisper_model_name = whisper_model

        # ── TTS State ─────────────────────────────────────
        self._tts_lock = threading.Lock()
        self._tts_async_lock: Optional[RobustLock] = None  # Lazy async mutex
        self.tts_engine = None  # Defer init
        
        # ── Mycelial / Affective State ────────────────────
        self._mycelium = None
        self._homeostasis = None
        self._substrate = None
        
        self._init_remaining()

    @property
    def tts_async_lock(self) -> RobustLock:
        """Loop-aware robust lock for TTS synthesis management."""
        if self._tts_async_lock is None:
            self._tts_async_lock = RobustLock("Voice.TTSAsyncLock")
        return self._tts_async_lock

    def _init_remaining(self):
        """Complete initialization that was blocked by logic error."""
        # ── Mic Capture State ─────────────────────────────
        self._mic_stream = None
        self._mic_listening = False

        # ── TTS State ─────────────────────────────────────
        self._tts_initialized = False
        self._voice_map = {}
        self._streaming = False

        # ── General State ─────────────────────────────────
        self.state = VoiceState.IDLE
        self._is_feeding = False
        auto_listen_env = os.environ.get("AURA_AUTO_LISTEN", "0").strip().lower()
        self.auto_listen_enabled = auto_listen_env in {"1", "true", "yes", "on"}
        self.microphone_enabled = self.auto_listen_enabled
        # TTS output gate — independent of STT. Defaults to True so voice works at boot.
        # Set to False via mute() to silence Aura's speech output from the UI toggle.
        self.speaking_enabled = True
        # Issue 34: Lazy initialize event if loop isn't ready
        self._interrupt_event = None 
        self.is_speaking = False

        # ── Callbacks ─────────────────────────────────────
        self._on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_tts_audio: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_state_change: Optional[Callable[[VoiceState], Awaitable[None]]] = None
        self._on_vad_change: Optional[Callable[[bool], None]] = None # Pulse when VAD detection changes

        # ── SSE & Threading ───────────────────────────────
        self._sse_queues: List[asyncio.Queue] = []
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

        # ── Piper / High Fidelity Configuration ──────────
        self.use_piper = True # Default to higher fidelity if available
        self.piper_voice_name = "en_US-amy-medium"
        self._piper_voice = None

        # ── XTTS / Persona Cloning Configuration ─────────
        self.use_xtts = True  # Enable Sara v3 Persona
        self.xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._xtts_engine = None
        self._speaker_wavs = []
        self._voice_ref_dir = self.data_dir.parent / "voice_references"

        logger.info("🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized")
        if self.auto_listen_enabled:
            logger.info("🎙️ Voice auto-listen ENABLED via AURA_AUTO_LISTEN.")
        else:
            logger.info("🎙️ Voice input standing by. STT will load on explicit mic enablement.")
        
        # Start presence pulse in background (BUG-035)
        if self.loop and self.loop.is_running():
            self.loop.create_task(self._pulse_presence())
            logger.debug("VoiceEngine: presence pulse started on active loop")
        else:
            logger.debug("VoiceEngine: presence pulse deferred (no running loop)")

    # ══════════════════════════════════════════════════════
    # MYCELIAL INTEGRATION
    # ══════════════════════════════════════════════════════

    def _get_mycelium(self):
        """Lazy-resolve the Mycelial Network from the container."""
        if self._mycelium is None:
            try:
                from core.container import ServiceContainer
                self._mycelium = ServiceContainer.get("mycelial_network", default=None)
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._mycelium

    def _get_homeostasis(self):
        """Lazy-resolve HomeostaticCoupling for sentient modulation."""
        if self._homeostasis is None:
            try:
                from core.container import ServiceContainer
                self._homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._homeostasis

    def _get_substrate(self):
        """Lazy-resolve LiquidSubstrate for direct affective bypass."""
        if self._substrate is None:
            try:
                from core.container import ServiceContainer
                self._substrate = ServiceContainer.get("liquid_substrate", default=None)
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._substrate

    def _pulse_hypha(self, source: str, target: str, success: bool = True):
        """Pulse a mycelial connection to signal activity."""
        mycelium = self._get_mycelium()
        if mycelium:
            try:
                hypha = mycelium.get_hypha(source, target)
                if hypha:
                    hypha.pulse(success=success)
                else:
                    # Auto-establish if missing
                    mycelium.establish_connection(source, target, priority=1.0)
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})

    def _signal_mycelium(self, source: str, target: str, payload: dict):
        """Route a signal through the Mycelial Network."""
        mycelium = self._get_mycelium()
        if mycelium:
            try:
                mycelium.route_signal(source, target, payload)
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})

    async def _pulse_presence(self):
        """Signals voice engine presence to the system (BUG-035)."""
        while True:
            try:
                from core.container import ServiceContainer
                bus = ServiceContainer.get("mycelium", default=None)
                if bus:
                    await bus.emit("aura.voice.presence", {
                        "status": "online",
                        "state": self.state.name,
                        "timestamp": time.time()
                    })
                # Also pulse a hypha if mycelium is ready
                self._pulse_hypha("voice_engine", "orchestrator", success=True)
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.debug("VoiceEngine: presence pulse failed: %s", e)
            await asyncio.sleep(30) # Pulse every 30s

    def _get_affective_prosody(self) -> dict:
        """
        AffectiveBypass: Direct orbital mapping from physiology to prosody.
        Bypasses the "Language Center" (LLM) for reflexive vocal shifts.
        """
        # Default baseline
        prosody = {"speed": 1.0, "pitch": 1.0, "volume": 1.0, "instability": 0.0}
        
        homeostasis = self._get_homeostasis()
        substrate = self._get_substrate()
        
        if homeostasis:
            mods = homeostasis.get_modifiers()
            # 1. Vitality governs base volume and speed
            prosody["volume"] = 0.8 + (float(mods.overall_vitality) * 0.4) # 0.8 to 1.2
            prosody["speed"] = 0.9 + (float(mods.overall_vitality) * 0.2)  # 0.9 to 1.1 baseline
            
        if substrate:
            try:
                # Use raw substrate activations for high-frequency bypass
                x = substrate.x
                v = substrate.v
                
                # Biometric Mapping Logic:
                # - High Arousal (x[1]) -> Higher speed, slightly higher pitch
                # - Low Valence (x[0]) -> Lower pitch (sadness/seriousness)
                # - High Volatility (v) -> Voice instability / trembling
                
                arousal = float((x[1] + 1.0) / 2.0)  # 0 to 1
                valence = float(x[0])               # -1 to 1
                volatility = float(np.mean(np.abs(v)) * 10.0)
                
                prosody["speed"] *= (1.0 + (arousal - 0.5) * 0.4)  # Boost speed by up to 20%
                prosody["pitch"] = 1.0 + (valence * 0.1) + (arousal * 0.05)
                prosody["instability"] = min(1.0, volatility * 2.0)
                
                # Special Case: Microtubule Coherence (Focus)
                if hasattr(substrate, 'microtubule_coherence'):
                    coherence = float(substrate.microtubule_coherence)
                    if coherence < 0.5:
                        prosody["instability"] = max(prosody["instability"], 0.5)
                
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
                
        # Signal Mycelial Roots about the expressive state shift
        self._signal_mycelium("voice_engine", "prosody", {
            "event": "affective_bypass_pulse",
            "prosody": {k: round(float(v), 2) for k, v in prosody.items()}
        })
                
        return prosody

    def _get_sensory_thresholds(self) -> dict:
        """Calculate STT thresholds based on internal state.
        'Curious' Aura is more tolerant of noise; 'Exhausted' Aura is 'irritable' (high gate).
        """
        homeostasis = self._get_homeostasis()
        thresholds = {
            "rms": SILENCE_THRESHOLD,
            "conf": -0.7 # MIN_AVG_LOGPROB
        }
        
        if homeostasis:
            mods = homeostasis.get_modifiers()
            # High curiosity/vitality -> lower thresholds (listen to everything)
            if mods.overall_vitality > 0.8:
                thresholds["rms"] *= 0.8
                thresholds["conf"] = -0.9 # More tolerant
            # Low vitality -> higher thresholds (filter out more noise, save energy)
            elif mods.overall_vitality < 0.4:
                thresholds["rms"] *= 1.5
                thresholds["conf"] = -0.5 # Very picky
                
        # Signal Mycelial Roots about sensory gating
        if homeostasis:
            now = time.time()
            # Debounce to prevent 30Hz mycelial spam loop building RAM infinitely
            if not hasattr(self, '_last_threshold_time') or (now - self._last_threshold_time) > 2.0:
                self._last_threshold_time = now
                self._signal_mycelium("voice_engine", "sensory_gate", {
                    "event": "threshold_shift",
                    "rms_gate": round(thresholds["rms"], 4),
                    "conf_gate": round(thresholds["conf"], 2)
                })
                
        return thresholds

    # ══════════════════════════════════════════════════════
    # MODEL INITIALIZATION
    # ══════════════════════════════════════════════════════

    def ensure_models(self):
        """Lazy-load STT and TTS models (synchronous — use ensure_models_async from async code)."""
        self.ensure_stt()
        self.ensure_tts()

    def ensure_stt(self):
        """Lazy-load only the STT stack."""
        if not self._stt_initialized:
            self._init_stt()

    def ensure_tts(self):
        """Lazy-load only the TTS stack."""
        if not self._tts_initialized:
            self._init_tts()

    def should_auto_listen(self) -> bool:
        """Whether mic capture should auto-start during boot."""
        return bool(self.auto_listen_enabled and self.microphone_enabled)

    async def ensure_models_async(self):
        """Non-blocking model load — offloads to thread so event loop isn't frozen."""
        if not self._stt_initialized or not self._tts_initialized:
            await asyncio.get_running_loop().run_in_executor(None, self.ensure_models)

    async def ensure_stt_async(self):
        """Non-blocking STT load only."""
        if not self._stt_initialized:
            await asyncio.get_running_loop().run_in_executor(None, self.ensure_stt)

    async def ensure_tts_async(self):
        """Non-blocking TTS load only."""
        if not self._tts_initialized:
            await asyncio.get_running_loop().run_in_executor(None, self.ensure_tts)

    def _init_stt(self):
        whisper_model_cls = _get_whisper_model_class()
        if whisper_model_cls is None:
            return
        try:
            logger.info("Loading Whisper model: %s...", self.whisper_model_name)
            
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            import torch
            device = "auto"
            compute_type = "default"
            
            if torch.backends.mps.is_available():
                # faster-whisper (CTranslate2) does not support MPS yet
                # attempting to use it causes a noisy fallback error.
                device = "cpu"
                compute_type = "int8"
            elif torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
            
            actual_device = device  # track what we actually end up using
            try:
                self.stt_model = whisper_model_cls(
                    self.whisper_model_name,
                    device=device,
                    compute_type=compute_type
                )
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.warning("Primary STT init failed on %s, falling back to CPU: %s", device, e)
                actual_device = "cpu"  # UPDATE: track the fallback
                self.stt_model = whisper_model_cls(
                    self.whisper_model_name,
                    device="cpu",
                    compute_type="int8" # Improved CPU performance
                )
            self._stt_initialized = True
            self._pulse_hypha("voice_engine", "cognition", success=True)
            logger.info("✅ Whisper STT online (model=%s, device=%s)", self.whisper_model_name, actual_device)
        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("Failed to init STT: %s", e)
            self._pulse_hypha("voice_engine", "cognition", success=False)

    def _init_tts(self):
        if self.use_xtts and TTS:
            try:
                self._init_xtts()
                return
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.error("Failed to init XTTS: %s", e)

        if self.use_piper and PiperVoice:
            try:
                model_dir = self.data_dir / "piper_voices"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{self.piper_voice_name}.onnx"
                config_path = model_dir / f"{self.piper_voice_name}.onnx.json"
                
                if not model_path.exists():
                     self._download_piper_voice(model_dir)
                
                self._piper_voice = PiperVoice.load(str(model_path), config_path=str(config_path))
                self._tts_initialized = True
                logger.info("✅ Piper Voice '%s' loaded (High Fidelity)", self.piper_voice_name)
                self._pulse_hypha("cognition", "voice_engine", success=True)
                return
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.warning("Failed to init Piper: %s. Falling back to pyttsx3.", e)

        if pyttsx3 is None:
            logger.error("❌ pyttsx3 not installed — TTS unavailable")
            return
        try:
            self.tts_engine = pyttsx3.init()
            self._tts_initialized = True
            self._pulse_hypha("cognition", "voice_engine", success=True)
            logger.info("✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)")
        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("Failed to init TTS: %s", e)
            self._pulse_hypha("cognition", "voice_engine", success=False)

    def _download_piper_voice(self, model_dir: Path):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
        parts = self.piper_voice_name.split("-")
        lang_code = parts[0]
        lang = lang_code.split("_")[0]
        speaker = parts[1] if len(parts) > 1 else "default"
        quality = parts[2] if len(parts) > 2 else "medium"
        
        vpath = f"{lang}/{lang_code}/{speaker}/{quality}"
        for fname in [f"{self.piper_voice_name}.onnx", f"{self.piper_voice_name}.onnx.json"]:
            dest = model_dir / fname
            if not dest.exists():
                url = f"{base_url}/{vpath}/{fname}"
                logger.info("Downloading %s...", fname)
                try:
                    urllib.request.urlretrieve(url, str(dest))
                except Exception:
                    # Fallback to direct HF link if structure differs
                    alt_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{vpath}/{fname}"
                    urllib.request.urlretrieve(alt_url, str(dest))

    def _init_xtts(self):
        """Initialize the Sara v3 XTTS-v2 voice clone."""
        if TTS is None:
            raise ImportError("TTS library not installed")
        
        logger.info("🎬 Initializing Sara v3 (XTTS-v2)...")
        # Initialize the model (downloads automatically to ~/.local/share/tts if not present)
        self._xtts_engine = TTS(self.xtts_model_name).to("mps") # Native Apple Silicon GPU
        
        # Load speaker references
        if self._voice_ref_dir.exists():
            # Get all wav/mp3/mp4 files (we'll focus on .wav for XTTS)
            for ext in ["*.wav", "*.mp3"]:
                self._speaker_wavs.extend([str(f) for f in self._voice_ref_dir.glob(ext)])
        
        if not self._speaker_wavs:
            logger.warning("No speaker references found for Sara v3 in %s. Fallback to default speaker.", self._voice_ref_dir)
        else:
            logger.info("🧬 Loaded %d vocal references for Sara v3.", len(self._speaker_wavs))

        self._tts_initialized = True
        logger.info("✅ Sara v3 Persona Online (High Fidelity XTTS-v2)")
        self._pulse_hypha("cognition", "voice_engine", success=True)

    # ══════════════════════════════════════════════════════
    # SERVER-SIDE MICROPHONE CAPTURE
    # ══════════════════════════════════════════════════════

    async def start_listening(self):
        """Start capturing audio from the microphone."""
        if not self.microphone_enabled:
            logger.warning("Microphone is disabled in config")
            return False

        if sd is None:
            logger.error("❌ sounddevice not installed — cannot capture mic")
            return False

        if self._mic_listening:
            logger.warning("Already listening")
            return True

        # Ensure STT model is ready
        if not self._stt_initialized:
            self._init_stt()
            if not self._stt_initialized:
                logger.error("Cannot start listening — STT model failed to load")
                return False

        try:
            # v7.0 HARDENING: Wrap Mic activation in a circuit breaker
            from core.resilience.resilience import SmartCircuitBreaker
            if not hasattr(self, "_mic_breaker"):
                self._mic_breaker = SmartCircuitBreaker("Microphone", failure_threshold=2, base_recovery_timeout=300)
            
            async def _mic_payload():
                # Start the STT worker thread
                self._is_feeding = True
                threading.Thread(target=self._stt_worker, daemon=True, name="VoiceSTTWorker").start()

                # Open the mic stream — the callback feeds chunks to the buffer
                self._mic_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCK_SIZE,
                    callback=self._mic_callback,
                )
                await asyncio.to_thread(self._mic_stream.start)
                self._mic_listening = True

                self._pulse_hypha("voice_engine", "cognition", success=True)
                self._signal_mycelium(
                    "voice_engine",
                    "cognition",
                    {"event": "mic_activated", "sample_rate": SAMPLE_RATE},
                )
                return True

            # Use await directly since we are now an async method
            success = await _mic_payload()
            logger.info(
                "🎙️ Server-side mic capture ACTIVE (sounddevice, %dHz mono)",
                SAMPLE_RATE,
            )
            return success

        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("Failed to start mic capture: %s", e, exc_info=True)
            self._pulse_hypha("voice_engine", "cognition", success=False)
            return False

    def stop_listening(self):
        """Stop microphone capture."""
        self._mic_listening = False
        self._is_feeding = False

        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
            self._mic_stream = None

        self._signal_mycelium("voice_engine", "cognition", {
            "event": "mic_deactivated"
        })
        logger.info("🎙️ Mic capture stopped")

    def _mic_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread, must be fast."""
        if status:
            logger.debug("Mic status: %s", status)
        if self._mic_listening and self.microphone_enabled:
            # indata is numpy int16 array, convert to bytes
            self._audio_buffer.put(bytes(indata))

    # ══════════════════════════════════════════════════════
    # BROWSER PCM INPUT (fallback path)
    # ══════════════════════════════════════════════════════

    async def feed_chunk(self, pcm_data: bytes):
        """Accept raw 16kHz PCM from browser WebSocket (fallback path).
        
        Primary path is now server-side mic capture via start_listening().
        This method remains for compatibility with browser-based capture.
        """
        if not self.microphone_enabled:
            return

        if not self._stt_initialized:
            await self.ensure_stt_async()

        if not self._is_feeding:
            self._is_feeding = True
            self._stt_thread = threading.Thread(
                target=self._stt_worker, daemon=True, name="VoiceSTTWorker"
            )
            self._stt_thread.start()

        self._audio_buffer.put(pcm_data)
        self._pulse_hypha("voice_engine", "cognition")

    # ══════════════════════════════════════════════════════
    # STT PROCESSING
    # ══════════════════════════════════════════════════════

    def _stt_worker(self):
        """Background thread: accumulates audio, detects silence, transcribes."""
        accumulated = b""
        last_voice_time = time.time()
        is_speaking = False

        logger.info("🧵 STT worker thread started")

        while self._is_feeding:
            try:
                chunk = self._audio_buffer.get(timeout=0.1)
                accumulated += chunk
                
                # BUG-015: Prevent infinite buffer growth
                if len(accumulated) > 1024 * 1024 * 5: # 5MB safety cap (~2.5 mins)
                    logger.error("⚠️ STT Buffer Safety: Clearing massive accumulated audio buffer (%.2f MB)", len(accumulated)/1024/1024)
                    accumulated = b""

                # Energy-based VAD
                audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_np ** 2))

                # Homeostatic Sensory Gating: Adjust thresholds based on internal state
                stt_gates = self._get_sensory_thresholds()
                current_silence_threshold = stt_gates["rms"]

                if rms > current_silence_threshold:
                    last_voice_time = time.time()
                    if not is_speaking:
                        is_speaking = True
                        if self._on_vad_change: self._on_vad_change(True)
                        logger.debug("🎙️ Voice activity detected (RMS=%.4f, Gate=%.4f)", rms, current_silence_threshold)
                        # Add Barge-in Detection
                        from core.senses.voice_engine import VoiceState
                        if self.state == VoiceState.SPEAKING and not self.interrupt_flag.is_set():
                            logger.warning("🎙️ Barge-in detected in VAD! Interrupting Aura...")
                            self.interrupt_flag.set()
                
                # Detect end of speech (user was speaking, now silent for > threshold)
                silence_detected = (
                    not is_speaking  # VAD says silent
                    and len(accumulated) > 0
                    and (time.time() - last_voice_time) > SILENCE_TIMEOUT
                )
                max_length_hit = (
                    len(accumulated) / (SAMPLE_RATE * 2) > MAX_AUDIO_LENGTH
                )

                if silence_detected or max_length_hit:
                    audio_seconds = len(accumulated) / (SAMPLE_RATE * 2)
                    if audio_seconds > MIN_AUDIO_LENGTH:
                        self._process_transcript(accumulated)
                    accumulated = b""
                    # Reset last_voice_time to prevent immediate re-trigger
                    last_voice_time = time.time()

            except queue.Empty:
                # Check for stale audio during silence if we weren't speaking
                if accumulated and not is_speaking:
                    elapsed = time.time() - last_voice_time
                    if elapsed > SILENCE_TIMEOUT:
                        audio_seconds = len(accumulated) / (SAMPLE_RATE * 2)
                        if audio_seconds > MIN_AUDIO_LENGTH:
                            self._process_transcript(accumulated)
                        accumulated = b""
                continue

        logger.info("🧵 STT worker thread exiting")

    def _process_transcript(self, audio_bytes: bytes):
        """Run Whisper on accumulated audio and dispatch the result.
        
        Multi-layer ambient audio filtering:
          1. RMS volume gate — reject quiet ambient sounds (TV, distant conversations)
          2. Whisper confidence threshold — reject low-probability transcriptions
          3. Noise phrase list — reject common STT hallucinations and TV phrases
        """
        if not self.stt_model:
            return

        # Convert bytes to float32 numpy array for Whisper
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_seconds = len(audio_np) / SAMPLE_RATE

        # ── Layer 1: RMS Volume Gate ──────────────────────────────
        # Reject audio that's too quiet — ambient TV/phone audio is typically
        # 15-25dB quieter than direct speech into the mic.
        rms = np.sqrt(np.mean(audio_np ** 2)) if len(audio_np) > 0 else 0
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Threshold: -35 dB. Normal direct speech is -20 to -10 dB.
        # TV at room volume from 6ft away is typically -40 to -35 dB.
        MIN_RMS_DB = -35.0
        if rms_db < MIN_RMS_DB:
            logger.debug("STT: rejected by volume gate (%.1f dB < %.1f dB threshold)", rms_db, MIN_RMS_DB)
            return

        try:
            from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
            sentinel = get_gpu_sentinel()
            
            # STT is a REFLEX task - it should pre-empt the LLM
            acquired = sentinel.acquire(priority=GPUPriority.REFLEX, timeout=10)
            if not acquired:
                logger.warning("STT: GPU Sentinel timeout")
                return

            try:
                segments_gen, info = self.stt_model.transcribe(
                    audio_np,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                # Collect segments with their probabilities
                segments = list(segments_gen)
            finally:
                sentinel.release()
            text = " ".join([seg.text for seg in segments]).strip()

            if not text or len(text) <= 4 or len(text.split()) < 2:
                logger.debug("STT: silence/noise (%.1fs audio, text='%s')", audio_seconds, text[:30] if text else "")
                return

            # ── Layer 2: Whisper Confidence Threshold ─────────────
            # Each segment has an avg_logprob. Low confidence = ambient/garbled audio.
            avg_prob: float = 0.0
            if segments:
                try:
                    avg_prob = sum(seg.avg_logprob for seg in segments) / len(segments)
                except (AttributeError, ZeroDivisionError):
                    avg_prob = 0.0
                
                # Homeostatic Gating: Irritable (high gate) vs. Curious (low gate)
                stt_gates = self._get_sensory_thresholds()
                min_conf = stt_gates["conf"]
                
                if avg_prob < min_conf:
                    logger.debug("STT: rejected by confidence (avg_logprob=%.2f < %.2f): '%s'",
                               avg_prob, min_conf, text[:50])
                    return

            text_lower = text.strip().lower()
            if text_lower in self._NOISE_PHRASES:
                logger.debug("STT: rejected hallucination: '%s'", text)
                return
            
            # Passed all filters — this is likely real user speech
            logger.info("🎙️ STT Result (%.1fs audio, %.1fdB, conf=%.2f): %s", 
                       audio_seconds, rms_db, 
                       avg_prob if segments else 0, text)
            self._pulse_hypha("voice_engine", "cognition", success=True)

            # Dispatch transcript
            self._dispatch_transcript(text)

        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("Transcription error: %s", e)
            self._pulse_hypha("voice_engine", "cognition", success=False)

    def _dispatch_transcript(self, text: str):
        """Route transcript to the orchestrator via callback + EventBus."""
        # Path 1: Direct callback (if registered by SovereignEars)
        loop = self.loop
        if self._on_transcript and loop and loop.is_running():
            try:
                loop.call_soon_threadsafe(
                    lambda t=text: loop.create_task(
                        self._handle_transcript(t),
                        name=f"transcript_{hash(t) & 0xFFFF}"
                    )
                )
            except RuntimeError as e:
                logger.debug("VoiceEngine: transcript dispatch skipped (loop closed): %s", e)

        # Path 2: EventBus dispatch (always, for redundancy)
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            bus.publish_threadsafe("user_input", {"message": text, "source": "voice"})
            logger.info("🍄 Transcript routed via EventBus: %s", text[:60])
        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("EventBus dispatch failed: %s", e)

        # Pulse the mycelial connection
        self._signal_mycelium("voice_engine", "cognition", {
            "event": "transcript",
            "text": text[:100]
        })

    async def _handle_transcript(self, text: str):
        """Async handler for direct callback path."""
        await self._set_state(VoiceState.PROCESSING)
        try:
            if self._on_transcript is not None and callable(self._on_transcript):
                logger.debug("Routing transcript to brain: %s", text[:50])
                res = self._on_transcript(text)
                if asyncio.iscoroutine(res) or asyncio.isfuture(res) or hasattr(res, "__await__"):
                    await res
                logger.debug("Transcript successfully routed.")
        except Exception as e:
            record_degradation('voice_engine', e)
            logger.error("Direct transcript callback failed: %s", e, exc_info=True)
        finally:
            await self._set_state(VoiceState.IDLE)

    # ══════════════════════════════════════════════════════
    # TTS (Text-to-Speech)
    # ══════════════════════════════════════════════════════

    async def speak(self, text: str):
        """Standard entry point for speech."""
        await self.synthesize_speech(text)

    async def synthesize_speech(self, text: str):
        """Speak text aloud using the best available engine."""
        if not text or not text.strip():
            return
        # TTS mute guard — respect speaking_enabled flag (set by UI voice toggle)
        if not getattr(self, "speaking_enabled", True):
            logger.debug("🔇 TTS suppressed: speaking_enabled=False")
            return

        if not await self.tts_async_lock.acquire_robust(timeout=10.0):
            logger.warning("⚠️ Failed to acquire TTS lock within 10s. Skipping synthesis.")
            return

        try:
            # Fallback initialization check
            if not self._xtts_engine and not self._piper_voice and getattr(self, "tts_engine", None) is None:
                await self.ensure_tts_async()
            
            await self._set_state(VoiceState.SPEAKING)
            self._pulse_hypha("cognition", "voice_engine")
            
            try:
                if self._xtts_engine:
                    await self._synthesize_xtts(text)
                elif self._piper_voice:
                    await self._synthesize_piper(text)
                elif self.tts_engine:
                    await self._synthesize_pyttsx3(text)
                
                self._pulse_hypha("cognition", "voice_engine", success=True)
                logger.debug("🗣️ Speech complete: %s", text[:60])
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.error("❌ TTS Synthesis failed: %s", e)
                self._pulse_hypha("cognition", "voice_engine", success=False)
            finally:
                self.is_speaking = False
                await self._set_state(VoiceState.IDLE)
        finally:
            if self.tts_async_lock.locked():
                self.tts_async_lock.release()

    async def _play_locally(self, audio_data: bytes):
        """Play PCM/WAV audio data locally on macOS using afplay."""
        if not audio_data:
            return

        def _play():
            try:
                temp_wav = self.data_dir / "tts_play_cache.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio_data)
                
                self._current_afplay = subprocess.Popen(["afplay", str(temp_wav)])
                while self._current_afplay.poll() is None:
                    if hasattr(self, 'interrupt_flag') and self.interrupt_flag.is_set():
                        self._current_afplay.terminate()
                        break
                    time.sleep(0.05)
            except Exception as e:
                record_degradation('voice_engine', e)
                logger.error("Local playback failed: %s", e)

        loop = getattr(self, "loop", None) or asyncio.get_running_loop()
        await loop.run_in_executor(None, _play)

    async def _emit_tts_audio(self, audio_data: bytes):
        """Mirror generated audio to browser subscribers and optional callbacks."""
        if not audio_data:
            return

        raw_pcm = audio_data[44:] if audio_data.startswith(b"RIFF") else audio_data

        if self._on_tts_audio:
            result = self._on_tts_audio(raw_pcm)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result) or hasattr(result, "__await__"):
                await result

        if not self._sse_queues:
            return

        payload = {
            "type": "audio",
            "data": base64.b64encode(raw_pcm).decode("ascii"),
            "timestamp": time.time(),
        }
        stale_queues: List[asyncio.Queue] = []
        for queue_ref in list(self._sse_queues):
            try:
                queue_ref.put_nowait(payload)
            except asyncio.QueueFull:
                stale_queues.append(queue_ref)
            except Exception as exc:
                record_degradation('voice_engine', exc)
                logger.debug("Voice SSE delivery failed: %s", exc)
                stale_queues.append(queue_ref)

        for queue_ref in stale_queues:
            if queue_ref in self._sse_queues:
                self._sse_queues.remove(queue_ref)

    async def _synthesize_xtts(self, text: str):
        """High-fidelity voice cloning via XTTS-v2."""
        def _get_audio():
            refs = self._speaker_wavs[:5] if self._speaker_wavs else None
            out_path = self.data_dir / "xtts_temp.wav"
            
            from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
            sentinel = get_gpu_sentinel()
            
            acquired = sentinel.acquire(priority=GPUPriority.REFLEX, timeout=10)
            if not acquired:
                logger.warning("XTTS: GPU Sentinel timeout")
                return None

            try:
                self._xtts_engine.tts_to_file(
                    text=text,
                    speaker_wav=refs,
                    language="en",
                    file_path=str(out_path)
                )
            finally:
                sentinel.release()
            
            with open(out_path, "rb") as f:
                return f.read()

        loop = getattr(self, "loop", None) or asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(None, _get_audio)
        
        if not audio_data:
            return

        await self._emit_tts_audio(audio_data)

        await self._play_locally(audio_data)

    async def speak_stream(self, text_iterator) -> str:
        """Plays TTS audio and returns exactly what was successfully spoken."""
        if not await self.tts_async_lock.acquire_robust(timeout=5.0):
             return "Lock timeout"

        try:
            self.interrupt_flag.clear()
            spoken_text_buffer = []
            await self._set_state(VoiceState.SPEAKING)
            self._pulse_hypha("cognition", "voice_engine")

            try:
                if not self._tts_initialized:
                    await self.ensure_tts_async()

                it = text_iterator.__aiter__()
                while True:
                    try:
                        if self.interrupt_flag.is_set():
                            logger.info("🛑 Aura interrupted. Halting synthesis.")
                            break
                        
                        text_chunk = await it.__anext__()
                        if not text_chunk or not text_chunk.strip():
                            continue
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        record_degradation('voice_engine', e)
                        logger.error(f"Error in voice stream: {e}")
                        break

                    if self._xtts_engine:
                        await self._synthesize_xtts(text_chunk)
                    elif self._piper_voice:
                        await self._synthesize_piper(text_chunk)
                    elif self.tts_engine:
                        await self._synthesize_pyttsx3(text_chunk)

                    if self.interrupt_flag.is_set():
                        break
                    
                    spoken_text_buffer.append(text_chunk)

            except Exception as e:
                record_degradation('voice_engine', e)
                logger.error(f"Playback error in stream: {e}")
                self._pulse_hypha("cognition", "voice_engine", success=False)
            finally:
                self.is_speaking = False
                await self._set_state(VoiceState.IDLE)
                self._pulse_hypha("cognition", "voice_engine", success=True)
                
            return " ".join(spoken_text_buffer).strip()
        finally:
            if self.tts_async_lock.locked():
                self.tts_async_lock.release()

    async def _synthesize_piper(self, text: str):
        """High-fidelity synthesis via Piper."""
        # Generate audio with prosody modulation
        prosody = self._get_affective_prosody()
        
        def _get_audio():
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                self._piper_voice.synthesize(text, wf)
            return buf.getvalue()

        # Issue 40: Track task and fix variable name
        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(None, _get_audio)
        
        # Issue 35: Guard against None audio
        if not audio_data:
            return
        await self._emit_tts_audio(audio_data)
        await self._play_locally(audio_data)
        
        # ALSO play locally for "server-side" voice consistency
        # Note: This uses standard system 'play' or similar if needed,
        # but for now we assume browsers or other services play the bytes.
        # If user wants direct system output:
        # await self._play_locally(audio_data)

    async def _synthesize_pyttsx3(self, text: str):
        """Fallback synthesis via pyttsx3 with prosody modulation."""
        prosody = self._get_affective_prosody()
        
        def _say():
            with self._tts_lock:
                # Modulate speed (baseline 200, scale 150-250)
                rate = int(200 * prosody["speed"])
                self.tts_engine.setProperty('rate', rate)
                
                # Modulate volume (0.0 to 1.0)
                self.tts_engine.setProperty('volume', prosody["volume"])
                
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()

        # Issue 37: Fallback to running loop
        loop = self.loop or asyncio.get_running_loop()
        await loop.run_in_executor(None, _say)

    # ══════════════════════════════════════════════════════
    # CONTROL & STATUS
    # ══════════════════════════════════════════════════════

    def mute(self):
        """Disable both microphone input (STT) and speaker output (TTS)."""
        self.microphone_enabled = False
        self.speaking_enabled = False
        self.stop_listening()
        logger.info("🔇 Voice Engine muted (STT + TTS disabled)")

    def unmute(self):
        """Enable microphone input and speaker output, restart capture."""
        self.microphone_enabled = True
        self.speaking_enabled = True
        # Issue 36: Schedule via create_task for async start_listening
        loop = None
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.start_listening())
        except RuntimeError as _e:
            # Fallback if unmuted from non-async context
            logger.debug('Ignored RuntimeError in voice_engine.py: %s', _e)
        logger.info("🔊 Voice Engine unmuted (STT + TTS enabled)")

    async def reset(self):
        """Full reset — stop listening, clear buffers."""
        self.stop_listening()
        self._is_feeding = False
        if hasattr(self, "_stt_thread") and self._stt_thread.is_alive():
            # Thread will exit on next loop iteration due to _is_feeding=False
            pass

        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except queue.Empty:
                break

    async def subscribe(self, q: asyncio.Queue = None) -> asyncio.Queue:
        if q is None:
            q = asyncio.Queue(maxsize=100)
        self._sse_queues.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        if q in self._sse_queues:
            self._sse_queues.remove(q)

    def on_transcript(self, callback: Callable[[str], Awaitable[None]]):
        """Register a callback for transcription results."""
        self._on_transcript = callback

    async def _set_state(self, new_state: VoiceState):
        if self.state != new_state:
            self.state = new_state
            if self._on_state_change:
                await self._on_state_change(new_state)

    def get_status(self) -> dict:
        tts_type = "Not loaded"
        if self._xtts_engine: tts_type = "Sara v3 (XTTS-v2)"
        elif self._piper_voice: tts_type = f"Piper ({self.piper_voice_name})"
        elif self.tts_engine: tts_type = "pyttsx3 (Native)"
        
        return {
            "state": self.state.name,
            "stt": "Whisper (Direct)" if self._stt_initialized else "Not loaded",
            "tts": tts_type,
            "mic": self.microphone_enabled,
            "speaking": self.speaking_enabled,
            "auto_listen": self.auto_listen_enabled,
            "listening": self._mic_listening,
            "server_capture": True,
        }

# ── Singleton ─────────────────────────────────────────────

_voice_engine: Optional[SovereignVoiceEngine] = None


def get_voice_engine(**kwargs) -> SovereignVoiceEngine:
    global _voice_engine
    if _voice_engine is None:
        _voice_engine = SovereignVoiceEngine(**kwargs)
    return _voice_engine
