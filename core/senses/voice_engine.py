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

import asyncio
import base64
import contextlib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
import queue
import subprocess
import threading
import time
import wave
from collections.abc import Awaitable, Callable, Iterator
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import FileWriteBatchEntry, get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.concurrency import RobustLock
from core.utils.exceptions import capture_and_log
from core.utils.task_tracker import get_task_tracker
from core.voice.microphone_authority import (
    MicrophoneDenial,
    MicrophoneLease,
    get_audio_ingress_broker,
    get_microphone_authority,
)


def _user_voice_output_enabled() -> bool:
    """Honor the user's ``voice.output_enabled`` toggle (default on).

    Reads the persisted runtime setting the settings UI writes, so disabling
    speech output in the UI silences TTS without a restart. Defaults to enabled
    if the setting is unset or unreadable. See docs/SETTINGS_WIRING_AUDIT.md.
    """
    return bool(get_runtime_setting("voice.output_enabled", True))


def _user_voice_input_enabled() -> bool:
    """Return the authoritative microphone-input gate."""

    return bool(get_runtime_setting("voice.input_enabled", True))


def _user_voice_auto_listen_enabled() -> bool:
    """Return the authoritative automatic-capture preference."""

    return bool(get_runtime_setting("voice.auto_listen", False))

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

logger = logging.getLogger("Aura.VoiceEngine")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled", "accepted"}


def _runtime_shutdown_requested() -> bool:
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return bool(is_shutdown_requested())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _coqui_license_accepted() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "AURA_COQUI_CPML_ACCEPTED",
            "AURA_COQUI_COMMERCIAL_LICENSED",
            "COQUI_TOS_AGREED",
        )
    )


def _direct_stt_command_dispatch_enabled() -> bool:
    """Return true only when raw STT should become chat input directly.

    Normal desktop voice uses the wake-word/session layer as the authority
    boundary.  Raw microphone transcripts are still published to WorldState so
    wake-word detection and perceptual grounding can see them, but they must not
    bypass that boundary and enter the user-input EventBus as commands.
    """

    return _env_flag("AURA_VOICE_DIRECT_EVENTBUS", False) or _env_flag(
        "AURA_VOICE_ALWAYS_DIRECT_TO_CHAT",
        False,
    )


def _get_whisper_model_class():
    """Import faster-whisper on demand so STT does not preload PyAV at module import."""
    global _WhisperModel, _whisper_import_attempted
    if _WhisperModel is not None:
        return _WhisperModel
    if _whisper_import_attempted:
        return None

    try:
        from core.runtime.third_party_imports import import_attribute_serialized

        whisper_model_cls = import_attribute_serialized(
            "faster_whisper",
            "WhisperModel",
        )
        _WhisperModel = whisper_model_cls
    except ImportError:
        logger.warning("faster-whisper not installed — STT unavailable")
    except (AttributeError, RuntimeError) as exc:
        record_degradation('voice_engine', exc)
        logger.error("❌ faster-whisper import failed — STT unavailable: %s", exc)
    finally:
        _whisper_import_attempted = True
    return _WhisperModel


def _sounddevice_available() -> bool:
    return sd is not None


def _stt_dependency_available() -> bool:
    """Whether speech-to-text can actually run.

    find_spec answers "is this module on disk", which is not the question a
    readiness surface is asking. A package that is present but raises on
    import — a moved API, a missing transitive dependency, an incompatible
    version — reported available here, so the UI showed STT ready for a
    dead path. That is how the Coqui voice path stayed broken unnoticed.

    Ground truth, in order: a successful import proves availability; a
    RECORDED FAILED import disproves it; only when neither has happened does
    presence on disk stand in, and then it is a presence signal, not a
    promise.
    """
    if _WhisperModel is not None:
        return True
    if _whisper_import_attempted:
        # We tried and it did not work. Presence on disk cannot overrule
        # having actually failed to import it.
        return False
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return False

TTS = None
_tts_api_import_attempted = False
_tts_api_import_error: str | None = None


def _load_tts_api():
    """Load Coqui TTS lazily after installing the transformers compatibility shim."""
    global TTS, _tts_api_import_attempted, _tts_api_import_error
    if TTS is not None:
        return TTS
    if _tts_api_import_attempted:
        return None

    _tts_api_import_attempted = True
    try:
        from core.utils.transformers_tts_compat import install_transformers_tts_compat

        install_transformers_tts_compat()
        TTS = importlib.import_module("TTS.api").TTS
        _tts_api_import_error = None
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
        TTS = None
        _tts_api_import_error = str(e)
        e_str = str(e)
        if "No module named 'TTS'" in e_str and pyttsx3 is not None:
            logger.info("TTS backend unavailable; native pyttsx3 fallback will be used.")
        else:
            logger.warning("TTS Import Error: %s", e_str)
    return TTS


def _tts_dependency_available() -> bool:
    """Whether the Coqui TTS backend can actually run.

    Same defect as the STT check above, and this is the one that bit: the
    XTTS path was dead from an import error while this returned True,
    because ``_tts_api_import_error`` was recorded and then never consulted.
    A status surface that reports a backend available after watching its
    import fail is worse than one that reports nothing.
    """
    if TTS is not None:
        return True
    if _tts_api_import_attempted:
        return False
    try:
        return importlib.util.find_spec("TTS") is not None
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return False


def _piper_dependency_available() -> bool:
    return PiperVoice is not None

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
                 data_dir: str | None = None,
                 *,
                 model_lane_controller: Any | None = None):
        from core.utils.paths import DATA_DIR
        self.data_dir = Path(data_dir or (DATA_DIR / "voice_models"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._model_lane_controller_override = model_lane_controller

        # ── STT State ─────────────────────────────────────
        self.stt_model = None
        self._stt_initialized = False
        self._stt_init_lock = threading.Lock()
        self._stt_usage_lock = threading.RLock()
        self._stt_active_users = 0
        self._stt_init_task: asyncio.Task[bool] | None = None
        self._stt_load_state = "not_loaded"
        self._stt_last_error = ""
        self._stt_lane_lease = None
        self._closing_event = threading.Event()
        self._audio_buffer = queue.Queue()
        self.whisper_model_name = whisper_model

        # ── TTS State ─────────────────────────────────────
        self._tts_lock = threading.Lock()
        self._tts_async_lock: RobustLock | None = None  # Lazy async mutex
        self.tts_engine = None  # Defer init
        
        # ── Mycelial / Affective State ────────────────────
        self._mycelium = None
        self._homeostasis = None
        self._substrate = None
        self._voice_owner_generation = f"voice-engine:{os.getpid()}:{id(self)}"
        
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
        self._stt_thread: threading.Thread | None = None
        self._mic_start_task: asyncio.Task[Any] | None = None
        self._mic_start_cancel_event: threading.Event | None = None
        self._mic_lease: MicrophoneLease | None = None
        self._mic_monitor_task: asyncio.Task[Any] | None = None
        self._mic_recovery_task: asyncio.Task[Any] | None = None
        self._mic_last_frame_at = 0.0
        self._mic_frames_seen = 0
        self._mic_device_generation = 0
        self._mic_device_state = "idle"
        self._mic_device_reason = ""
        self._mic_capture_requested = False

        # ── TTS State ─────────────────────────────────────
        self._tts_initialized = False
        self._tts_lane_lease = None
        self._voice_map = {}
        self._streaming = False

        # ── General State ─────────────────────────────────
        self.state = VoiceState.IDLE
        self._is_feeding = False
        # Runtime settings are the authority. Launch-profile environment flags
        # used to force capture on even when the desktop setting said off,
        # producing a successful settings receipt while PortAudio kept the mic
        # live. Input permission and auto-listen are deliberately independent:
        # input may be permitted while automatic capture remains idle.
        self.auto_listen_enabled = _user_voice_auto_listen_enabled()
        self.microphone_enabled = _user_voice_input_enabled()
        self.speaking_enabled = _user_voice_output_enabled()
        self._settings_lock = threading.RLock()
        self._current_afplay: Any | None = None
        # Issue 34: Lazy initialize event if loop isn't ready
        self._interrupt_event = None 
        self.is_speaking = False
        # [STABILITY v58] interrupt_flag used by speak_stream and barge-in detection
        try:
            self.interrupt_flag = asyncio.Event()
        except RuntimeError:
            # No running event loop yet; will be created on first use
            self.interrupt_flag = threading.Event()

        # ── Callbacks ─────────────────────────────────────
        self._on_transcript: Callable[[str], Awaitable[None]] | None = None
        self._transcript_callbacks: dict[str, Callable[[str], Awaitable[None]]] = {}
        self._candidate_transcript_callbacks: set[str] = set()
        self._anonymous_transcript_callbacks: list[Callable[[str], Awaitable[None]]] = []
        self._last_audio_source_assessment: dict[str, Any] = {}
        self._last_threshold_payload: dict[str, Any] = {}
        self._last_threshold_signal_time = 0.0
        self._on_tts_audio: Callable[[bytes], Awaitable[None]] | None = None
        self._on_state_change: Callable[[VoiceState], Awaitable[None]] | None = None
        self._on_vad_change: Callable[[bool], None] | None = None # Pulse when VAD detection changes

        # ── SSE & Threading ───────────────────────────────
        self._sse_queues: list[asyncio.Queue] = []
        try:
            self.loop = asyncio.get_running_loop()
            self._owner_loop_thread_id: int | None = threading.get_ident()
        except RuntimeError:
            self.loop = None
            self._owner_loop_thread_id = None

        # ── Piper / High Fidelity Configuration ──────────
        self.use_piper = True # Default to higher fidelity if available
        self.piper_voice_name = "en_US-amy-medium"
        self._piper_voice = None

        # ── XTTS / Persona Cloning Configuration ─────────
        self.use_xtts = _env_flag("AURA_ENABLE_XTTS", False)
        self.xtts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._xtts_engine = None
        self._speaker_wavs = []
        self._voice_ref_dir = self.data_dir.parent / "voice_references"
        if self.use_xtts and not _coqui_license_accepted():
            self.use_xtts = False
            logger.warning(
                "XTTS disabled: set AURA_COQUI_CPML_ACCEPTED=1 or "
                "AURA_COQUI_COMMERCIAL_LICENSED=1 after accepting Coqui terms. "
                "Using Piper/pyttsx3 fallback."
            )

        logger.info("🎙️ SovereignVoiceEngine v5.0 (Server-Side + Mycelial) initialized")
        if self.should_auto_listen():
            logger.info("🎙️ Voice auto-listen enabled by verified runtime settings.")
        elif self.microphone_enabled:
            logger.info("🎙️ Voice input permitted; automatic capture is disabled.")
        else:
            logger.info("🎙️ Voice input disabled by verified runtime settings.")
        
        # Start presence pulse in background (BUG-035)
        if self.loop and self.loop.is_running():
            get_task_tracker().create_task(
                self._pulse_presence(),
                name="voice_engine.pulse_presence",
            )
            logger.debug("VoiceEngine: presence pulse started on active loop")
        else:
            logger.debug("VoiceEngine: presence pulse deferred (no running loop)")

        # Ingestion rate-limiting & loop-prevention state
        self._last_transcript_time = 0.0
        self._last_transcript_text = ""

    # ══════════════════════════════════════════════════════
    # MYCELIAL INTEGRATION
    # ══════════════════════════════════════════════════════

    def _get_mycelium(self):
        """Lazy-resolve the Mycelial Network from the container."""
        if getattr(self, "_mycelium", None) is None:
            try:
                from core.container import ServiceContainer
                self._mycelium = ServiceContainer.get("mycelial_network", default=None)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._mycelium

    def _get_homeostasis(self):
        """Lazy-resolve HomeostaticCoupling for sentient modulation."""
        if self._homeostasis is None:
            try:
                from core.container import ServiceContainer
                self._homeostasis = ServiceContainer.get("homeostatic_coupling", default=None)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._homeostasis

    def _get_substrate(self):
        """Lazy-resolve LiquidSubstrate for direct affective bypass."""
        if self._substrate is None:
            try:
                from core.container import ServiceContainer
                self._substrate = ServiceContainer.get("liquid_substrate", default=None)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
        return self._substrate

    def _pulse_hypha(self, source: str, target: str, success: bool = True):
        """Pulse a mycelial connection to signal activity."""
        mycelium = self._get_mycelium()
        if mycelium:
            try:
                if not mycelium.pulse_hypha(source, target, success=success):
                    # Auto-establish if missing
                    mycelium.establish_connection(source, target, priority=1.0)
                    mycelium.pulse_hypha(source, target, success=success)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})

    def _signal_mycelium(self, source: str, target: str, payload: dict):
        """Route a signal through the Mycelial Network."""
        mycelium = self._get_mycelium()
        if mycelium:
            try:
                mycelium.route_signal(source, target, payload)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})

    async def _pulse_presence(self):
        """Signals voice engine presence to the system (BUG-035)."""
        while getattr(self, '_running', True):
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
            except (ImportError, AttributeError, RuntimeError) as e:
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
                
            except (RuntimeError, AttributeError, TypeError) as e:
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
            payload = {
                "event": "threshold_shift",
                "rms_gate": round(thresholds["rms"], 4),
                "conf_gate": round(thresholds["conf"], 2),
            }
            changed = payload != self._last_threshold_payload
            heartbeat_due = (now - self._last_threshold_signal_time) > 30.0
            if changed or heartbeat_due:
                self._last_threshold_payload = dict(payload)
                self._last_threshold_signal_time = now
                self._signal_mycelium("voice_engine", "sensory_gate", payload)
                
        return thresholds

    # ══════════════════════════════════════════════════════
    # MODEL INITIALIZATION
    # ══════════════════════════════════════════════════════

    def ensure_models(self):
        """Lazy-load STT and TTS models (synchronous — use ensure_models_async from async code)."""
        self.ensure_stt()
        self.ensure_tts()

    def ensure_stt(self) -> bool:
        """Lazy-load only the STT stack."""
        if self._stt_initialized:
            return True
        return self._init_stt()

    def ensure_tts(self):
        """Lazy-load only the TTS stack."""
        if not self._tts_initialized:
            self._init_tts()

    @staticmethod
    def _whisper_footprint_gb(model_name: str) -> float:
        lowered = str(model_name or "base").lower()
        if "large" in lowered:
            return 4.0
        if "medium" in lowered:
            return 2.0
        if "small" in lowered:
            return 1.0
        if "tiny" in lowered:
            return 0.25
        return 0.5

    async def _evict_stt_lane(self, _owner: Any, reason: str) -> bool:
        released = await asyncio.to_thread(
            self._release_stt_model_if_idle,
            reason=f"stt_lane_eviction:{reason}",
        )
        if not released:
            logger.warning("STT model preemption refused during active use: %s", reason)
        return released

    async def _compensate_stt_lane(self, _owner: Any, reason: str) -> bool:
        logger.info("Restoring STT lane after failed candidate: %s", reason)
        return await self.ensure_stt_async()

    def _release_stt_model(self, *, reason: str) -> bool:
        with self._stt_init_lock:
            with self._stt_usage_lock:
                return self._release_stt_model_locked(reason=reason)

    def _release_stt_model_if_idle(self, *, reason: str) -> bool:
        if not self._stt_init_lock.acquire(blocking=False):
            return False
        try:
            if not self._stt_usage_lock.acquire(blocking=False):
                return False
            try:
                return self._release_stt_model_locked(reason=reason)
            finally:
                self._stt_usage_lock.release()
        finally:
            self._stt_init_lock.release()

    def _release_stt_model_locked(self, *, reason: str) -> bool:
        self.stt_model = None
        self._stt_initialized = False
        if self._stt_load_state != "stopping":
            self._stt_load_state = "not_loaded"
        lease, self._stt_lane_lease = self._stt_lane_lease, None
        if lease is not None:
            lease.release(reason=reason)
        return self.stt_model is None

    @contextlib.contextmanager
    def stt_model_session(self) -> Iterator[Any | None]:
        """Hold exact model ownership for one complete transcription."""

        if not self.ensure_stt():
            yield None
            return
        with self._stt_usage_lock:
            model = self.stt_model
            if model is None:
                yield None
                return
            self._stt_active_users += 1
            try:
                yield model
            finally:
                self._stt_active_users = max(0, self._stt_active_users - 1)

    async def _evict_tts_lane(self, _owner: Any, reason: str) -> bool:
        if self.is_speaking:
            logger.warning("TTS model preemption refused during active speech: %s", reason)
            return False
        return await asyncio.to_thread(
            self._release_tts_model,
            reason=f"tts_lane_eviction:{reason}",
        )

    async def _compensate_tts_lane(self, _owner: Any, reason: str) -> bool:
        logger.info("Restoring TTS lane after failed candidate: %s", reason)
        await self.ensure_tts_async()
        return bool(self._tts_initialized)

    def _release_tts_model(self, *, reason: str) -> bool:
        with self._tts_lock:
            self._xtts_engine = None
            self._piper_voice = None
            self.tts_engine = None
            self._tts_initialized = False
            lease, self._tts_lane_lease = self._tts_lane_lease, None
            if lease is not None:
                lease.release(reason=reason)
            return True

    def _acquire_voice_model_lane(
        self,
        *,
        owner_suffix: str,
        model_path: str,
        request_gb: float,
        evict: Callable[..., Any],
        compensate: Callable[..., Any],
    ) -> Any:
        from core.runtime.model_lane_control import (
            acquire_synchronous_in_process_model_lane,
        )

        return acquire_synchronous_in_process_model_lane(
            owner_id=f"voice:{id(self)}:{owner_suffix}",
            model_path=model_path,
            purpose="serve",
            request_gb=request_gb,
            priority=30,
            preemptible=False,
            evict=evict,
            compensate=compensate,
            metadata={
                "engine": "voice",
                "model_role": owner_suffix,
                "lifecycle_state": "loading",
            },
            controller=self._model_lane_controller_override,
        )

    def should_auto_listen(self) -> bool:
        """Whether mic capture should auto-start during boot."""
        return bool(self.auto_listen_enabled and self.microphone_enabled)

    def _bind_owner_loop(self) -> asyncio.AbstractEventLoop:
        """Bind loop-affine voice work to the loop actually running it."""

        loop = asyncio.get_running_loop()
        self.loop = loop
        self._owner_loop_thread_id = threading.get_ident()
        return loop

    def _request_speech_stop(self) -> bool:
        """Interrupt active synthesis/playback from any settings worker thread."""

        completed = threading.Event()

        def _interrupt() -> None:
            try:
                flag = getattr(self, "interrupt_flag", None)
                if flag is not None and hasattr(flag, "set"):
                    flag.set()
                player = getattr(self, "_current_afplay", None)
                if player is not None:
                    try:
                        if player.poll() is None:
                            player.terminate()
                    except (
                        AttributeError,
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ):
                        pass
            finally:
                completed.set()

        loop = getattr(self, "loop", None)
        owner_thread = getattr(self, "_owner_loop_thread_id", None)
        if (
            loop is not None
            and loop.is_running()
            and owner_thread is not None
            and owner_thread != threading.get_ident()
        ):
            try:
                loop.call_soon_threadsafe(_interrupt)
                return completed.wait(timeout=1.0)
            except RuntimeError:
                pass
        _interrupt()
        return completed.is_set()

    def _start_capture_from_settings(self) -> dict[str, str]:
        """Start capture on the owner loop and report what actually happened."""

        if not self.microphone_enabled:
            return {
                "owner": "voice_input",
                "status": "deferred",
                "detail": "auto-listen armed; microphone input gate is disabled",
            }
        if not self.auto_listen_enabled:
            return {
                "owner": "voice_input",
                "status": "applied",
                "detail": "microphone input permitted; automatic capture remains idle",
            }
        if bool(getattr(self, "_mic_listening", False)):
            return {
                "owner": "voice_input",
                "status": "applied",
                "detail": "server microphone capture is active",
            }

        loop = getattr(self, "loop", None)
        if loop is None or not loop.is_running():
            return {
                "owner": "voice_input",
                "status": "deferred",
                "detail": "auto-listen persisted; voice owner loop is not running yet",
            }
        if getattr(self, "_owner_loop_thread_id", None) == threading.get_ident():
            get_task_tracker().create_task(
                self.start_listening(),
                name="voice_engine.settings_start_listening",
            )
            return {
                "owner": "voice_input",
                "status": "deferred",
                "detail": "server microphone startup scheduled on the voice owner loop",
            }

        try:
            timeout_s = max(
                0.5,
                min(30.0, float(os.environ.get("AURA_VOICE_SETTINGS_APPLY_TIMEOUT_S", "12"))),
            )
        except (TypeError, ValueError):
            timeout_s = 12.0
        future = asyncio.run_coroutine_threadsafe(self.start_listening(), loop)
        try:
            started = bool(future.result(timeout=timeout_s))
        except TimeoutError:
            return {
                "owner": "voice_input",
                "status": "deferred",
                "detail": "server microphone startup is still in progress",
            }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "owner": "voice_input",
                "status": "failed",
                "detail": f"{type(exc).__name__}:{str(exc)[:180]}",
            }
        if started and bool(getattr(self, "_mic_listening", False)):
            return {
                "owner": "voice_input",
                "status": "applied",
                "detail": "server microphone capture started and verified",
            }
        return {
            "owner": "voice_input",
            "status": "failed",
            "detail": "server microphone capture did not become active",
        }

    def apply_runtime_setting(
        self,
        key: str,
        _previous: Any,
        value: Any,
    ) -> dict[str, str]:
        """Apply one persisted voice setting to the resident hardware owner."""

        if key not in {
            "voice.input_enabled",
            "voice.output_enabled",
            "voice.auto_listen",
        }:
            return {
                "owner": "voice_runtime",
                "status": "unchanged",
                "detail": "setting is outside the resident voice bridge",
            }

        with self._settings_lock:
            if key == "voice.input_enabled":
                self.microphone_enabled = bool(value)
            elif key == "voice.auto_listen":
                self.auto_listen_enabled = bool(value)
            else:
                self.speaking_enabled = bool(value)

        if key == "voice.output_enabled":
            interrupted = self.speaking_enabled or self._request_speech_stop()
            return {
                "owner": "voice_output",
                "status": "applied" if interrupted else "failed",
                "detail": (
                    "speech output enabled"
                    if self.speaking_enabled
                    else "speech output gate closed but owner-loop interruption was not verified"
                    if not interrupted
                    else "speech output disabled and active playback interrupted"
                ),
            }

        if not self.microphone_enabled or not self.auto_listen_enabled:
            self.stop_listening()
            if bool(getattr(self, "_mic_listening", False)) or getattr(
                self, "_mic_stream", None
            ) is not None:
                return {
                    "owner": "voice_input",
                    "status": "failed",
                    "detail": "server microphone capture remained active after disable",
                }
            return {
                "owner": "voice_input",
                "status": "applied",
                "detail": (
                    "microphone input disabled and capture stopped"
                    if not self.microphone_enabled
                    else "automatic capture disabled and microphone stopped"
                ),
            }
        return self._start_capture_from_settings()

    async def ensure_models_async(self):
        """Non-blocking model load — offloads to thread so event loop isn't frozen."""
        await asyncio.gather(self.ensure_stt_async(), self.ensure_tts_async())

    async def ensure_stt_async(self) -> bool:
        """Load STT once without letting one cancelled waiter cancel shared work."""
        if self._stt_initialized:
            return True
        if self._voice_closing():
            self._stt_load_state = "stopping"
            return False

        task = self._stt_init_task
        if task is None or task.done():
            task = get_task_tracker().create_task(
                asyncio.to_thread(self.ensure_stt),
                name="voice_engine.ensure_stt",
            )
            self._stt_init_task = task
        try:
            return bool(await asyncio.shield(task))
        finally:
            if task.done() and self._stt_init_task is task:
                self._stt_init_task = None

    async def ensure_tts_async(self):
        """Non-blocking TTS load only."""
        if not self._tts_initialized:
            await asyncio.get_running_loop().run_in_executor(None, self.ensure_tts)

    def _voice_closing(self) -> bool:
        return self._closing_event.is_set() or _runtime_shutdown_requested()

    def _init_stt(self) -> bool:
        with self._stt_init_lock:
            lane_lease = None
            if self._stt_initialized:
                return True
            if self._voice_closing():
                self._stt_load_state = "stopping"
                return False

            whisper_model_cls = _get_whisper_model_class()
            if whisper_model_cls is None:
                self._stt_load_state = "unavailable"
                self._stt_last_error = "faster_whisper_unavailable"
                return False

            local_files_only = not _env_flag("AURA_STT_ALLOW_MODEL_DOWNLOAD", False)
            self._stt_load_state = "loading"
            self._stt_last_error = ""
            try:
                from core.runtime.model_lane_control import ModelLaneControlError

                logger.info(
                    "Loading Whisper model: %s (local_files_only=%s)...",
                    self.whisper_model_name,
                    local_files_only,
                )

                try:
                    lane_lease = self._acquire_voice_model_lane(
                        owner_suffix="stt",
                        model_path=f"faster-whisper/{self.whisper_model_name}",
                        request_gb=self._whisper_footprint_gb(self.whisper_model_name),
                        evict=self._evict_stt_lane,
                        compensate=self._compensate_stt_lane,
                    )
                except ModelLaneControlError as exc:
                    self._stt_load_state = "unavailable"
                    self._stt_last_error = f"model_lane_refused:{exc}"[:240]
                    logger.warning("STT model admission refused: %s", exc)
                    return False

                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                device = "cpu"
                compute_type = "int8"

                if _env_flag("AURA_STT_ENABLE_TORCH_DEVICE_PROBE", False):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            device = "cuda"
                            compute_type = "float16"
                    except (
                        ImportError,
                        AttributeError,
                        RuntimeError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        record_degradation(
                            "voice_engine",
                            exc,
                            severity="warning",
                            action=(
                                "continued STT init with CPU int8 after optional torch "
                                "probe failed"
                            ),
                        )

                actual_device = device
                constructor_kwargs = {
                    "device": device,
                    "compute_type": compute_type,
                    "local_files_only": local_files_only,
                }
                try:
                    model = whisper_model_cls(
                        self.whisper_model_name,
                        **constructor_kwargs,
                    )
                except (RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
                    if device == "cpu":
                        raise
                    record_degradation("voice_engine", exc)
                    logger.warning(
                        "Primary STT init failed on %s, falling back to CPU: %s",
                        device,
                        exc,
                    )
                    actual_device = "cpu"
                    model = whisper_model_cls(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="int8",
                        local_files_only=local_files_only,
                    )

                if self._voice_closing():
                    self._stt_load_state = "stopping"
                    lane_lease.release(reason="stt_closed_during_model_load")
                    return False

                if not lane_lease.set_preemptible(True):
                    raise ModelLaneControlError(
                        "stt_model_lane_activation_refused_after_load"
                    )
                self.stt_model = model
                self._stt_lane_lease = lane_lease
                lane_lease = None
                self._stt_initialized = True
                self._stt_load_state = "ready"
                self._pulse_hypha("voice_engine", "sensory_gate", success=True)
                logger.info(
                    "✅ Whisper STT online (model=%s, device=%s)",
                    self.whisper_model_name,
                    actual_device,
                )
                return True
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                OSError,
                ConnectionError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                if lane_lease is not None:
                    lane_lease.release(reason="stt_model_load_failed")
                self.stt_model = None
                self._stt_initialized = False
                self._stt_load_state = "unavailable"
                self._stt_last_error = f"{type(exc).__name__}: {exc}"[:240]
                record_degradation("voice_engine", exc)
                logger.error(
                    "Failed to init STT from %s cache: %s",
                    "local" if local_files_only else "configured remote/local",
                    exc,
                )
                self._pulse_hypha("voice_engine", "sensory_gate", success=False)
                return False

    def _init_tts(self):
        tts_api = _load_tts_api() if self.use_xtts else None
        if self.use_xtts and tts_api:
            lane_lease = None
            try:
                lane_lease = self._acquire_voice_model_lane(
                    owner_suffix="xtts",
                    model_path=self.xtts_model_name,
                    request_gb=4.0,
                    evict=self._evict_tts_lane,
                    compensate=self._compensate_tts_lane,
                )
                self._init_xtts()
                if not lane_lease.set_preemptible(True):
                    self._xtts_engine = None
                    self._tts_initialized = False
                    raise RuntimeError("xtts_model_lane_activation_refused_after_load")
                self._tts_lane_lease = lane_lease
                lane_lease = None
                return
            except (EOFError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                if lane_lease is not None:
                    lane_lease.release(reason="xtts_model_load_failed")
                elif self._tts_lane_lease is not None and not self._tts_initialized:
                    self._release_tts_model(reason="xtts_model_init_incomplete")
                record_degradation('voice_engine', e)
                logger.error("Failed to init XTTS: %s", e)

        if self.use_piper and PiperVoice:
            lane_lease = None
            try:
                model_dir = self.data_dir / "piper_voices"
                from core.governance_context import local_internal_governed_scope

                with local_internal_governed_scope(
                    "voice_engine.piper_model_directory",
                    domain="file_write",
                ):
                    get_file_write_gateway().ensure_directory(
                        model_dir,
                        source="core.senses.voice_engine.piper_model_directory",
                    )
                model_path = model_dir / f"{self.piper_voice_name}.onnx"
                config_path = model_dir / f"{self.piper_voice_name}.onnx.json"
                
                if not model_path.exists():
                     self._download_piper_voice(model_dir)

                lane_lease = self._acquire_voice_model_lane(
                    owner_suffix="piper",
                    model_path=str(model_path),
                    request_gb=0.25,
                    evict=self._evict_tts_lane,
                    compensate=self._compensate_tts_lane,
                )
                self._piper_voice = PiperVoice.load(str(model_path), config_path=str(config_path))
                if not lane_lease.set_preemptible(True):
                    self._piper_voice = None
                    raise RuntimeError("piper_model_lane_activation_refused_after_load")
                self._tts_lane_lease = lane_lease
                lane_lease = None
                self._tts_initialized = True
                logger.info("✅ Piper Voice '%s' loaded (High Fidelity)", self.piper_voice_name)
                self._pulse_hypha("cognition", "voice_engine", success=True)
                return
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                if lane_lease is not None:
                    lane_lease.release(reason="piper_model_load_failed")
                elif self._tts_lane_lease is not None and not self._tts_initialized:
                    self._release_tts_model(reason="piper_model_init_incomplete")
                record_degradation('voice_engine', e)
                logger.warning("Failed to init Piper: %s. Falling back to pyttsx3.", e)

        if pyttsx3 is None:
            logger.warning("pyttsx3 not installed — TTS unavailable")
            return
        try:
            self.tts_engine = pyttsx3.init()
            self._tts_initialized = True
            self._pulse_hypha("cognition", "voice_engine", success=True)
            logger.info("✅ pyttsx3 TTS online (macOS NSSpeechSynthesizer)")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('voice_engine', e)
            logger.error("Failed to init TTS: %s", e)
            self._pulse_hypha("cognition", "voice_engine", success=False)

    def _download_piper_voice(self, model_dir: Path):
        from core.governance_context import local_internal_governed_scope

        if model_dir.is_symlink():
            raise RuntimeError("refusing Piper model installation through a symlink")
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
        parts = self.piper_voice_name.split("-")
        lang_code = parts[0]
        lang = lang_code.split("_")[0]
        speaker = parts[1] if len(parts) > 1 else "default"
        quality = parts[2] if len(parts) > 2 else "medium"

        vpath = f"{lang}/{lang_code}/{speaker}/{quality}"
        pending: list[FileWriteBatchEntry] = []
        for fname in [f"{self.piper_voice_name}.onnx", f"{self.piper_voice_name}.onnx.json"]:
            dest = model_dir / fname
            if dest.is_symlink():
                raise RuntimeError(f"refusing Piper model asset symlink: {dest.name}")
            if not dest.exists():
                url = f"{base_url}/{vpath}/{fname}"
                logger.info("Downloading %s...", fname)
                payload = self._download_piper_asset(
                    url,
                    fallback_url=f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{vpath}/{fname}",
                    fname=fname,
                )
                self._validate_piper_asset(fname, payload)
                pending.append(FileWriteBatchEntry(path=dest, payload=payload))

        if pending:
            # Download the complete missing set before any replacement, then
            # commit it as one rollback-capable governed batch. A failed config
            # fetch can no longer leave a model-only half-installation.
            with local_internal_governed_scope(
                "voice_engine.download_piper_voice",
                domain="file_write",
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(
                    model_dir,
                    source="core.senses.voice_engine.download_piper_voice",
                )
                gateway.write_bytes_batch(
                    pending,
                    source="core.senses.voice_engine.download_piper_voice",
                )

    @staticmethod
    def _validate_piper_asset(fname: str, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload:
            raise RuntimeError(f"Piper asset is empty or not bytes: {fname}")
        if payload.lstrip().lower().startswith((b"<html", b"<!doctype")):
            raise RuntimeError(f"Piper asset response is HTML, not a model asset: {fname}")
        if fname.endswith(".json"):
            try:
                parsed = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Piper config is not valid JSON: {fname}") from exc
            if not isinstance(parsed, dict) or not parsed:
                raise RuntimeError(f"Piper config must be a non-empty object: {fname}")
        elif len(payload) < 1024:
            raise RuntimeError(f"Piper model payload is implausibly small: {fname}")

    @staticmethod
    def _download_piper_asset(url: str, *, fallback_url: str, fname: str) -> bytes:
        headers = {"User-Agent": "Aura/5.1 voice-engine"}
        gateway = get_network_gateway()
        response = gateway.request(
            "GET",
            url,
            headers=headers,
            timeout=60,
            read_only=True,
            source="core.senses.voice_engine.download_piper_asset",
        )
        if not response.get("ok"):
            response = gateway.request(
                "GET",
                fallback_url,
                headers=headers,
                timeout=60,
                read_only=True,
                source="core.senses.voice_engine.download_piper_asset.fallback",
            )
        if not response.get("ok"):
            raise OSError(response.get("error") or f"failed to download Piper asset {fname}")
        content = response.get("content")
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"downloaded Piper asset {fname} was empty or invalid")
        return content

    def _init_xtts(self):
        """Initialize the Sara v3 XTTS-v2 voice clone."""
        if not _coqui_license_accepted():
            raise RuntimeError("XTTS requires explicit Coqui license acceptance in environment")
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        if TTS is None:
            if _load_tts_api() is None:
                raise ImportError(_tts_api_import_error or "TTS library not installed")
        
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
        self._bind_owner_loop()
        if not self.microphone_enabled:
            logger.warning("Microphone is disabled in config")
            return False

        if sd is None:
            logger.warning("sounddevice not installed — cannot capture mic")
            return False

        if self._mic_listening:
            logger.warning("Already listening")
            return True

        if self._voice_closing():
            logger.info("Microphone start refused: voice runtime is stopping")
            return False

        if self._mic_start_task is not None and not self._mic_start_task.done():
            logger.info("Microphone start already in progress")
            return False
        self._mic_capture_requested = True

        # Ensure STT model is ready
        if not self._stt_initialized:
            await self.ensure_stt_async()
            if not self._stt_initialized:
                logger.error("Cannot start listening — STT model failed to load")
                return False

        # Acquire before touching PortAudio. The browser duplex lane and every
        # native capture owner share this authority, so a focused conversation
        # can deliberately preempt passive listening without two handles ever
        # contending for CoreAudio.
        start_cancelled = threading.Event()
        self._mic_start_cancel_event = start_cancelled
        authority = get_microphone_authority()
        lease = authority.acquire(
            self._voice_owner_generation,
            principal="owner:local",
            source="sounddevice",
            mode="passive",
            preemptible=True,
            revoke_callback=self._on_microphone_lease_revoked,
        )
        if isinstance(lease, MicrophoneDenial):
            logger.info(
                "Microphone capture not admitted: %s (%s)",
                lease.reason,
                lease.detail,
            )
            if lease.reason == "device_busy" and self._mic_capture_requested:
                self._mic_device_state = "waiting"
                self._mic_device_reason = "device_busy"
                self._register_microphone_waiter(authority)
            return False
        self._mic_lease = lease

        try:
            # v7.0 HARDENING: Wrap Mic activation in a circuit breaker
            from core.resilience.resilience import SmartCircuitBreaker
            if not hasattr(self, "_mic_breaker"):
                self._mic_breaker = SmartCircuitBreaker("Microphone", failure_threshold=2, base_recovery_timeout=300)

            def _open_and_start_stream():
                lease_active, _reason = authority.validate(lease)
                if not lease_active:
                    return None
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCK_SIZE,
                    callback=self._mic_callback,
                )
                if start_cancelled.is_set() or self._voice_closing():
                    self._close_mic_stream(stream)
                    return None
                stream_started = False
                try:
                    stream.start()
                    stream_started = True
                finally:
                    if not stream_started:
                        self._close_mic_stream(stream)
                if start_cancelled.is_set() or self._voice_closing():
                    self._close_mic_stream(stream)
                    return None
                return stream

            try:
                start_timeout_s = max(
                    0.05,
                    float(__import__("core.runtime.flags", fromlist=["declare", "FlagKind"]).declare(
                        "AURA_MIC_START_TIMEOUT_S",
                        kind=__import__("core.runtime.flags", fromlist=["FlagKind"]).FlagKind.FLOAT,
                        default=6.0,
                        description="Microphone stream start budget",
                        owner="core.senses.voice_engine",
                    ).value()),
                )
            except (TypeError, ValueError):
                start_timeout_s = 6.0

            # Opening CoreAudio can block inside PortAudio/TCC on macOS. The
            # thread owns late cleanup so cancellation cannot leak a stream.
            start_task = get_task_tracker().create_task(
                asyncio.to_thread(_open_and_start_stream),
                name="voice_engine.open_microphone",
            )
            self._mic_start_task = start_task
            try:
                stream = await asyncio.wait_for(
                    asyncio.shield(start_task),
                    timeout=start_timeout_s,
                )
            except TimeoutError:
                start_cancelled.set()
                start_task.add_done_callback(
                    lambda completed: self._finish_late_mic_start(
                        completed,
                        authority=authority,
                        lease=lease,
                        recovery_reason="startup_timeout",
                    )
                )
                raise
            except asyncio.CancelledError:
                start_cancelled.set()
                start_task.add_done_callback(
                    lambda completed: self._finish_late_mic_start(
                        completed,
                        authority=authority,
                        lease=lease,
                    )
                )
                raise
            finally:
                if start_task.done() and self._mic_start_task is start_task:
                    self._mic_start_task = None

            lease_active, _lease_reason = authority.validate(lease)
            if stream is None or self._voice_closing() or not lease_active:
                self._close_mic_stream(stream)
                authority.release(lease, reason="startup_cancelled")
                if self._mic_lease is lease:
                    self._mic_lease = None
                return False
            self._mic_stream = stream

            # Start the STT worker thread only after the stream is live.
            self._is_feeding = True
            self._stt_thread = threading.Thread(
                target=self._stt_worker,
                daemon=True,
                name="VoiceSTTWorker",
            )
            self._stt_thread.start()
            self._mic_listening = True
            self._mic_last_frame_at = time.monotonic()
            self._mic_frames_seen = 0
            self._mic_device_generation += 1
            self._mic_device_state = "active"
            self._mic_device_reason = "capture_started"
            self._start_microphone_monitor(lease, stream)

            self._pulse_hypha("voice_engine", "cognition", success=True)
            self._signal_mycelium(
                "voice_engine",
                "cognition",
                {"event": "mic_activated", "sample_rate": SAMPLE_RATE},
            )
            logger.info(
                "🎙️ Server-side mic capture ACTIVE (sounddevice, %dHz mono)",
                SAMPLE_RATE,
            )
            return True

        except TimeoutError as e:
            record_degradation("voice_engine", e)
            logger.warning(
                "Mic stream startup timed out; voice capture will retry on demand "
                "without blocking desktop boot."
            )
            self._is_feeding = False
            self._mic_listening = False
            self._pulse_hypha("voice_engine", "cognition", success=False)
            return False
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as e:
            record_degradation('voice_engine', e)
            logger.error("Failed to start mic capture: %s", e, exc_info=True)
            recoverable_device_failure = isinstance(e, OSError)
            self.stop_listening(preserve_request=recoverable_device_failure)
            if recoverable_device_failure:
                self._schedule_microphone_recovery("stream_start_failed")
            self._pulse_hypha("voice_engine", "cognition", success=False)
            return False

    def _on_microphone_lease_revoked(self, reason: str) -> None:
        """Close native capture when a focused session or setting supersedes it."""
        logger.info("Resident microphone lease revoked: %s", reason)
        preempted = str(reason or "").startswith("preempted_by:")
        self.stop_listening(preserve_request=preempted)
        if preempted:
            self._mic_device_state = "waiting"
            self._mic_device_reason = str(reason)
            self._register_microphone_waiter(get_microphone_authority())

    def _on_microphone_available(self, _group: str) -> None:
        self._unregister_microphone_waiter(get_microphone_authority())
        self._schedule_microphone_recovery("resource_available")

    def _register_microphone_waiter(self, authority: Any) -> bool:
        register = getattr(authority, "register_availability_waiter", None)
        if not callable(register):
            logger.warning(
                "Microphone authority cannot wake a displaced resident capture owner"
            )
            return False
        register(
            self._voice_owner_generation,
            principal="owner:local",
            source="sounddevice",
            callback=self._on_microphone_available,
        )
        return True

    def _unregister_microphone_waiter(self, authority: Any) -> None:
        unregister = getattr(authority, "unregister_availability_waiter", None)
        if callable(unregister):
            unregister(self._voice_owner_generation)

    def _start_microphone_monitor(
        self,
        lease: MicrophoneLease,
        stream: Any,
    ) -> None:
        task = self._mic_monitor_task
        if task is not None and not task.done():
            task.cancel()
        self._mic_monitor_task = get_task_tracker().create_task(
            self._monitor_microphone_stream(lease, stream),
            name="voice_engine.monitor_microphone",
        )

    async def _monitor_microphone_stream(
        self,
        lease: MicrophoneLease,
        stream: Any,
    ) -> None:
        """Detect a dead PortAudio stream even when no callback can report it."""
        try:
            while self._mic_listening and self._mic_lease is lease:
                await asyncio.sleep(0.5)
                lease_ok, _reason = get_microphone_authority().validate(lease)
                if not lease_ok:
                    return
                active = getattr(stream, "active", None)
                if active is False:
                    self._schedule_microphone_recovery("stream_inactive")
                    return
                if (
                    self._mic_frames_seen > 0
                    and time.monotonic() - self._mic_last_frame_at > 3.0
                ):
                    self._schedule_microphone_recovery("callback_stalled")
                    return
        except asyncio.CancelledError:
            raise
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_engine.device_monitor",
                exc,
                severity="warning",
                action="scheduled microphone recovery after monitor failure",
                enforce_failure_policy=False,
            )
            self._schedule_microphone_recovery("monitor_failed")

    def _schedule_microphone_recovery(self, reason: str) -> None:
        if (
            not self._mic_capture_requested
            or not self.microphone_enabled
            or self._voice_closing()
        ):
            return

        def _schedule() -> None:
            task = self._mic_recovery_task
            if task is not None and not task.done():
                return
            self._mic_recovery_task = get_task_tracker().create_task(
                self._recover_microphone(str(reason or "device_fault")),
                name="voice_engine.recover_microphone",
            )

        loop = getattr(self, "loop", None)
        if loop is None or not loop.is_running():
            return
        if getattr(self, "_owner_loop_thread_id", None) == threading.get_ident():
            _schedule()
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_schedule)

    async def _recover_microphone(self, reason: str) -> bool:
        """Reopen a lost native device without losing model or mind state."""
        self._mic_device_generation += 1
        self._mic_device_state = "recovering"
        self._mic_device_reason = reason
        self.stop_listening(preserve_request=True)
        for delay in (0.0, 0.25, 0.75, 1.5, 3.0):
            if (
                not self._mic_capture_requested
                or not self.microphone_enabled
                or self._voice_closing()
            ):
                return False
            if delay:
                await asyncio.sleep(delay)
            if await self.start_listening():
                self._mic_device_state = "active"
                self._mic_device_reason = f"recovered:{reason}"
                return True
        self._mic_device_state = "unavailable"
        self._mic_device_reason = f"reopen_failed:{reason}"
        return False

    @staticmethod
    def _close_mic_stream(stream: Any) -> None:
        if stream is None:
            return
        try:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                stop()
        except (RuntimeError, AttributeError, OSError, TypeError, ValueError):
            pass
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except (RuntimeError, AttributeError, OSError, TypeError, ValueError):
            pass

    def _finish_late_mic_start(
        self,
        task: asyncio.Task[Any],
        *,
        authority: Any | None = None,
        lease: MicrophoneLease | None = None,
        recovery_reason: str = "",
    ) -> None:
        if self._mic_start_task is task:
            self._mic_start_task = None
        try:
            stream = task.result()
        except (asyncio.CancelledError, RuntimeError, OSError, TypeError, ValueError):
            if authority is not None:
                authority.release(lease, reason="late_start_finished")
            if self._mic_lease is lease:
                self._mic_lease = None
            if recovery_reason:
                self._schedule_microphone_recovery(recovery_reason)
            return
        self._close_mic_stream(stream)
        if authority is not None:
            authority.release(lease, reason="late_start_finished")
        if self._mic_lease is lease:
            self._mic_lease = None
        if recovery_reason:
            self._schedule_microphone_recovery(recovery_reason)

    def stop_listening(self, *, preserve_request: bool = False):
        """Stop microphone capture."""
        if not preserve_request:
            self._mic_capture_requested = False
            self._unregister_microphone_waiter(get_microphone_authority())
        current_task = None
        with contextlib.suppress(RuntimeError):
            current_task = asyncio.current_task()
        monitor_task, self._mic_monitor_task = self._mic_monitor_task, None
        if monitor_task is not None and monitor_task is not current_task:
            monitor_task.cancel()
        recovery_task = self._mic_recovery_task
        if (
            not preserve_request
            and recovery_task is not None
            and recovery_task is not current_task
        ):
            recovery_task.cancel()
            self._mic_recovery_task = None
        if self._mic_start_cancel_event is not None:
            self._mic_start_cancel_event.set()
        self._mic_listening = False
        self._is_feeding = False

        if self._mic_stream:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_engine', e)
                capture_and_log(e, {'module': __name__})
            self._mic_stream = None

        thread, self._stt_thread = self._stt_thread, None
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=1.5)
            if thread.is_alive():
                record_degradation(
                    "voice_engine",
                    TimeoutError("VoiceSTTWorker did not stop within 1.5s"),
                    severity="warning",
                    action="left bounded STT worker cleanup visible to runtime hygiene",
                    enforce_failure_policy=False,
                )

        lease, self._mic_lease = self._mic_lease, None
        get_microphone_authority().release(lease, reason="resident_capture_stopped")

        self._signal_mycelium("voice_engine", "cognition", {
            "event": "mic_deactivated"
        })
        logger.info("🎙️ Mic capture stopped")
        if not preserve_request:
            self._mic_device_generation += 1
            self._mic_device_state = "stopped"
            self._mic_device_reason = "capture_stopped"
        return not self._mic_listening and self._mic_stream is None

    def on_stop(self) -> None:
        self._closing_event.set()
        self.stop_listening()
        self._release_stt_model(reason="voice_engine_stopped")
        self._release_tts_model(reason="voice_engine_stopped")

    def _mic_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread, must be fast."""
        if status:
            logger.debug("Mic status: %s", status)
        if self._mic_listening and self.microphone_enabled:
            self._mic_last_frame_at = time.monotonic()
            self._mic_frames_seen += 1
            # indata is numpy int16 array, convert to bytes
            payload = bytes(indata)
            if get_audio_ingress_broker().admit(self._mic_lease, len(payload)):
                self._audio_buffer.put(payload)

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

                # Rate-limited liveness heartbeat: makes "Aura doesn't respond to
                # voice" diagnosable. It proves mic frames are actually arriving
                # and shows whether ambient/speech RMS crosses the VAD gate — the
                # first fork in triaging a silent voice path.
                _hb_now = time.time()
                if _hb_now - getattr(self, "_vad_heartbeat_at", 0.0) > 15.0:
                    logger.info(
                        "🎙️ Voice listening: RMS=%.4f gate=%.4f speaking=%s buffered=%dB",
                        rms, current_silence_threshold, is_speaking, len(accumulated),
                    )
                    self._vad_heartbeat_at = _hb_now

                if rms > current_silence_threshold:
                    last_voice_time = time.time()
                    if not is_speaking:
                        is_speaking = True
                        if self._on_vad_change:
                            self._on_vad_change(True)
                        logger.debug("🎙️ Voice activity detected (RMS=%.4f, Gate=%.4f)", rms, current_silence_threshold)
                        # Add Barge-in Detection
                        from core.senses.voice_engine import VoiceState
                        if self.state == VoiceState.SPEAKING and not self.interrupt_flag.is_set():
                            logger.warning("🎙️ Barge-in detected in VAD! Interrupting Aura...")
                            self.interrupt_flag.set()
                elif is_speaking and (time.time() - last_voice_time) > SILENCE_TIMEOUT:
                    # UN-LATCH: without this, the first gate crossing pinned
                    # is_speaking=True forever, end-of-utterance could never
                    # fire (it requires `not is_speaking`), and user speech
                    # accumulated until the 5MB safety wipe DISCARDED it —
                    # the live "Aura doesn't answer when I talk" silence.
                    is_speaking = False
                    if self._on_vad_change:
                        self._on_vad_change(False)
                    logger.debug(
                        "🎙️ Voice activity ended (RMS=%.4f, Gate=%.4f) — utterance closing",
                        rms, current_silence_threshold,
                    )
                
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
        min_rms_db = -35.0
        if rms_db < min_rms_db:
            logger.debug("STT: rejected by volume gate (%.1f dB < %.1f dB threshold)", rms_db, min_rms_db)
            return

        try:
            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

            sentinel = get_gpu_sentinel()
            
            # STT is a REFLEX task - it should pre-empt the LLM
            acquired = sentinel.acquire(priority=GPUPriority.REFLEX, timeout=10)
            if not acquired:
                logger.warning("STT: GPU Sentinel timeout")
                return

            try:
                with self.stt_model_session() as stt_model:
                    if stt_model is None:
                        return
                    segments_gen, _info = stt_model.transcribe(
                        audio_np,
                        beam_size=5,
                        language="en",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    # Materialize while the model ownership session is held.
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

            source_assessment = self._classify_audio_source(
                text,
                rms_db=rms_db,
                transcript_confidence=avg_prob if segments else 0.0,
                duration_s=audio_seconds,
            )

            # Dispatch transcript
            self._dispatch_transcript(
                text,
                source_assessment=source_assessment,
            )

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('voice_engine', e)
            logger.error("Transcription error: %s", e)
            self._pulse_hypha("voice_engine", "cognition", success=False)

    def _classify_audio_source(
        self,
        text: str,
        *,
        rms_db: float,
        transcript_confidence: float,
        duration_s: float,
    ) -> dict[str, Any]:
        from core.senses.audio_attention import classify_audio_attention

        active_app = ""
        visual_context: dict[str, Any] = {}
        try:
            from core.world_state import get_world_state

            world_state = get_world_state()
            active_app = str(
                getattr(world_state, "active_foreground_app", "")
                or getattr(world_state, "active_app_context", "")
                or ""
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Audio source attribution lacks active-app context: %s", exc)

        try:
            from core.container import ServiceContainer

            interaction_signals = ServiceContainer.get("interaction_signals", default=None)
            if interaction_signals is not None and hasattr(interaction_signals, "get_status"):
                status = interaction_signals.get_status()
                if isinstance(status, dict) and isinstance(status.get("vision"), dict):
                    visual_context = dict(status["vision"])
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Audio source attribution lacks fresh camera context: %s", exc)

        assessment = classify_audio_attention(
            text,
            rms_db=rms_db,
            transcript_confidence=transcript_confidence,
            duration_s=duration_s,
            active_app=active_app,
            explicit_command=_direct_stt_command_dispatch_enabled(),
            visual_context=visual_context,
        ).as_dict()
        self._last_audio_source_assessment = assessment
        logger.info(
            "🎧 Audio attribution: source=%s confidence=%.2f attention=%s response_authorized=%s",
            assessment.get("source"),
            float(assessment.get("confidence", 0.0) or 0.0),
            assessment.get("attention_mode"),
            assessment.get("response_authorized"),
        )
        return assessment

    def begin_owner_voice_conversation(self) -> None:
        """The owner opened a voice conversation from the UI. Speech is for her.

        The wake-word boundary exists so AMBIENT audio — a video, a nearby
        conversation — cannot hijack the typed chat lane. It was also gating the
        case where the owner had just pressed a control labelled "Start voice
        conversation", which is the opposite situation: an explicit invitation.

        Measured live: the mic came up, Whisper transcribed the owner's speech
        correctly, every utterance was filed as `transcript_candidate` with
        `requires_wake_word_session=True`, and nothing ever answered. From the
        outside she could hear him and would not respond.

        No TTL: the session lasts exactly as long as the owner leaves voice on,
        which is what the control's own "Stop the voice conversation" label
        promises. Ambient enablement (wake-word listening, boot) does not call
        this, so that boundary is untouched.
        """

        self._owner_voice_conversation = True
        self._owner_voice_conversation_started_at = time.time()
        logger.info(
            "🎙️ Owner voice conversation OPEN — speech routes to chat without a "
            "wake word until voice is switched off."
        )

    def end_owner_voice_conversation(self) -> None:
        if getattr(self, "_owner_voice_conversation", False):
            logger.info("🎙️ Owner voice conversation CLOSED — wake word required again.")
        self._owner_voice_conversation = False
        self._owner_voice_conversation_started_at = 0.0
        self._owner_voice_chunk_at = 0.0

    #: How long after the last owner-streamed audio chunk the conversation stays
    #: open. The UI stops streaming the moment the owner presses stop, so this is
    #: an idle window, not a conversation timeout — long enough to think between
    #: sentences, short enough that a closed conversation does not linger.
    OWNER_VOICE_CHUNK_IDLE_S: float = 45.0

    def note_owner_voice_chunk(self) -> None:
        """Audio arrived from the owner's own UI voice control.

        This is the signal the wake-word boundary was missing. Chunks on this
        path exist only because the owner pressed the control and is deliberately
        speaking to her, which is categorically different from a microphone that
        happens to be listening.
        """

        if not getattr(self, "_owner_voice_conversation", False):
            self.begin_owner_voice_conversation()
        self._owner_voice_chunk_at = time.time()

    def owner_voice_conversation_active(self) -> bool:
        """True only while the owner has an explicit voice conversation open."""

        if not getattr(self, "_owner_voice_conversation", False):
            return False
        last_chunk = float(getattr(self, "_owner_voice_chunk_at", 0.0) or 0.0)
        if last_chunk > 0.0:
            # Streaming from the UI keeps it open on its own; the server-side mic
            # need not be the source.
            if (time.time() - last_chunk) <= self.OWNER_VOICE_CHUNK_IDLE_S:
                return True
            return False
        # Opened by an explicit microphone enablement instead: a conversation
        # cannot outlive the microphone it speaks through.
        return bool(getattr(self, "microphone_enabled", False))

    #: Classifier verdicts that place the speaker AT the machine rather than
    #: somewhere in the building or inside a video. ``ambient_speech`` and
    #: ``unknown_speech`` are excluded on purpose: both mean the evidence was
    #: too thin to say who spoke, and "cannot tell" must not become "answer it".
    _SPEAKER_AT_THE_MACHINE_SOURCES = frozenset(
        {
            "direct_user",
            "direct_address",
            "nearby_visible_speaker",
            "nearby_person",
        }
    )

    def _ambient_speech_is_addressed_to_her(
        self, source_assessment: dict[str, Any] | None = None
    ) -> bool:
        """Whether speech in the room may be answered without a wake phrase.

        OWNER REPORT, 2026-08-10: "if im talking to her from my computer, she
        should just talk back like normal chat. and she should be able to act
        from it as well." Requiring "Hey Aura" before every sentence is not how
        someone talks to something in the room with them; it made an open
        microphone behave like a closed one.

        The wake phrase was never the real protection anyway — it is a
        password anyone's television can say. What actually distinguishes the
        owner speaking from a video playing is audio provenance, which
        :func:`attribute_wake_audio` already decides: an unverified speaker is
        refused while any other process is making sound. That guard also
        closes the echo loop, because her own speech goes out through a player
        this sees, so she cannot answer herself.

        Two independent things have to hold, and neither is a phrase:

        * the audio-attention classifier must place the speech at the machine —
          near-field energy, a visible speaker, or direct address. Media
          playback, distant room noise and speech with no source evidence are
          all refused, which is what keeps a documentary from talking to her.
        * audio provenance must show nothing else making sound, which is what
          keeps HER from talking to herself: her replies go out through a
          player this sees.

        Set ``AURA_VOICE_REQUIRE_WAKE_PHRASE=1`` to restore the wake-phrase
        boundary for a shared or noisy room.
        """
        if _env_flag("AURA_VOICE_REQUIRE_WAKE_PHRASE", False):
            return False

        assessment = dict(
            source_assessment or self._last_audio_source_assessment or {}
        )
        source = str(assessment.get("source") or "")
        if source not in self._SPEAKER_AT_THE_MACHINE_SOURCES:
            logger.info(
                "🔇 Ambient speech not answered — source=%r is not a person at "
                "the machine.",
                source or "unclassified",
            )
            return False

        try:
            from core.voice.audio_provenance import attribute_wake_audio

            attribution = attribute_wake_audio(assessment)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_engine.ambient_attribution",
                exc,
                action="required a wake phrase for this utterance",
            )
            return False
        if attribution.get("owner_attributed"):
            return True
        logger.info(
            "🔇 Ambient speech not answered — %s",
            attribution.get("owner_attribution_reason", "unattributed"),
        )
        return False

    def _dispatch_transcript(
        self,
        text: str,
        *,
        source_assessment: dict[str, Any] | None = None,
    ):
        """Route transcript to the orchestrator via callback + EventBus."""
        now = time.time()
        
        # 1. Rate-limiting (max 1 command per 2 seconds)
        last_time = getattr(self, "_last_transcript_time", 0.0)
        if now - last_time < 2.0:
            logger.warning("VoiceEngine: transcript rate-limited (too frequent): %r", text)
            return

        # 2. Deduplication (prevent duplicate commands within 5 seconds)
        normalized = text.strip().lower()
        last_text = getattr(self, "_last_transcript_text", "")
        if normalized == last_text and now - last_time < 5.0:
            logger.warning("VoiceEngine: transcript deduplicated (duplicate command): %r", text)
            return

        # Record validation state
        self._last_transcript_time = now
        self._last_transcript_text = normalized
        direct_command_dispatch = (
            _direct_stt_command_dispatch_enabled()
            or self.owner_voice_conversation_active()
            or self._ambient_speech_is_addressed_to_her(source_assessment)
        )
        audio_source = dict(source_assessment or self._last_audio_source_assessment or {})
        audio_source["response_authorized"] = bool(direct_command_dispatch)

        # 3. Record the transcript for wake-word/perception.  By default this is
        # a candidate transcript, not an authorized user command.  The wake-word
        # detector is the command boundary for normal desktop voice.
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.last_voice_transcript = text
            ws.last_voice_transcript_at = time.time()
            ws.voice_activity_detected = True
            if hasattr(ws, "last_voice_activity_at"):
                ws.last_voice_activity_at = ws.last_voice_transcript_at
            ws.last_audio_source_assessment = dict(audio_source)
            ws.record_event(
                description=(
                    f"User voice command: {text}"
                    if direct_command_dispatch
                    else f"Voice transcript candidate: {text}"
                ),
                source="user" if direct_command_dispatch else "voice_stt_candidate",
                salience=1.0 if direct_command_dispatch else 0.35,
                ttl=600.0 if direct_command_dispatch else 90.0,
                transcript=text,
                authorized_command=bool(direct_command_dispatch),
                requires_wake_word_session=not direct_command_dispatch,
                audio_source=audio_source,
            )
            logger.info(
                "🎙️ Recorded %svoice transcript in WorldState",
                "" if direct_command_dispatch else "candidate ",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("voice_engine.world_state_transcript_event", e)
            logger.error("Failed to record transcript event in WorldState: %s", e)

        # Path 1: Direct callback (if registered by SovereignEars)
        loop = self.loop
        has_direct_callback = bool(
            self._on_transcript
            or self._transcript_callbacks
            or self._anonymous_transcript_callbacks
        )
        if has_direct_callback and loop and loop.is_running():
            try:
                loop.call_soon_threadsafe(
                    lambda t=text: get_task_tracker().create_task(
                        self._handle_transcript(
                            t,
                            authorized_command=direct_command_dispatch,
                        ),
                        name=f"transcript_{hash(t) & 0xFFFF}"
                    )
                )
            except RuntimeError as e:
                logger.debug("VoiceEngine: transcript dispatch skipped (loop closed): %s", e)

        # Path 2: EventBus dispatch only in explicit raw-dictation mode.  Normal
        # voice commands route through wake_word._dispatch_to_conversation_lane()
        # after a wake/session boundary, so ambient STT cannot hijack chat.
        if direct_command_dispatch:
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                bus.publish_threadsafe("user_input", {"message": text, "source": "voice"})
                logger.info("🍄 Transcript routed via EventBus: %s", text[:60])
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('voice_engine', e)
                logger.error("EventBus dispatch failed: %s", e)
        else:
            # Heard, transcribed, and deliberately not answered. Without a wake
            # word or an open voice conversation this is ambient speech, and
            # ambient speech must not be able to drive chat — that boundary is
            # correct and stays.
            #
            # Dropping it in SILENCE is not. "when i talk to my computer
            # nothing happens. no response or anything" — reported live
            # 2026-08-10, while the log showed her transcribing "Hey, or can
            # you hear me right now?" perfectly. An invisible boundary is
            # indistinguishable from a dead microphone, so the one person who
            # could act on it had no way to learn what to do.
            logger.info(
                "🎙️ Heard %r but did not answer: no wake word and no open voice "
                "conversation. Say 'Hey Aura' first, or press Voice to talk "
                "without one.",
                text[:120],
            )

        # Pulse the mycelial connection.  Unauthorized STT is perception, not a
        # foreground thought: route candidates through the sensory gate so media
        # or nearby voices cannot contaminate the live typed chat context.
        mycelium_target = "cognition" if direct_command_dispatch else "sensory_gate"
        self._signal_mycelium("voice_engine", mycelium_target, {
            "event": "transcript" if direct_command_dispatch else "transcript_candidate",
            "text": text[:100],
            "authorized_command": bool(direct_command_dispatch),
            "conversation_context_eligible": bool(direct_command_dispatch),
            "audio_source": audio_source,
        })

    async def _handle_transcript(
        self,
        text: str,
        *,
        authorized_command: bool = True,
    ):
        """Async handler for direct callback path."""
        await self._set_state(VoiceState.PROCESSING)
        try:
            await self._run_transcript_callbacks(
                text,
                authorized_command=authorized_command,
            )
            logger.debug("Transcript successfully routed.")
        except (RuntimeError, AttributeError, TypeError) as e:
            record_degradation('voice_engine', e)
            logger.error("Direct transcript callback failed: %s", e, exc_info=True)
        finally:
            await self._set_state(VoiceState.IDLE)

    # ══════════════════════════════════════════════════════
    # TTS (Text-to-Speech)
    # ══════════════════════════════════════════════════════

    async def _play_locally(self, audio_data: bytes):
        """Play PCM/WAV audio data locally on macOS using afplay."""
        if not audio_data:
            return False

        def _play():
            try:
                from core.governance_context import local_internal_governed_scope

                temp_wav = self.data_dir / "tts_play_cache.wav"
                # Establish authority inside the worker rather than relying on
                # whichever caller context happened to schedule playback.
                with local_internal_governed_scope(
                    "voice_engine.play_locally", domain="tool_execution"
                ):
                    get_file_write_gateway().write_bytes(
                        temp_wav,
                        audio_data,
                        source="core.senses.voice_engine.play_locally",
                    )
                    self._current_afplay = get_subprocess_gateway().spawn(
                        ["afplay", str(temp_wav)],
                        source="core.senses.voice_engine.play_locally",
                        accelerator_capability="none",
                    )
                while self._current_afplay.poll() is None:
                    if hasattr(self, 'interrupt_flag') and self.interrupt_flag.is_set():
                        self._current_afplay.terminate()
                        break
                    try:
                        self._current_afplay.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        continue
                return self._current_afplay.poll() == 0
            except (subprocess.SubprocessError, OSError, RuntimeError, ValueError) as e:
                record_degradation('voice_engine', e)
                logger.error("Local playback failed: %s", e)
                return False

        played = bool(await asyncio.to_thread(_play))
        if played:
            mycelium = self._get_mycelium()
            if mycelium is not None and hasattr(mycelium, "attest_neural_root"):
                evidence = {
                    "service": "afplay",
                    "audio_route": "coreaudio_default_output",
                    "playback_completed": True,
                }
                mycelium.attest_neural_root(
                    "voice_engine",
                    root_kind="service",
                    target_id="afplay",
                    owner_generation=getattr(
                        self,
                        "_voice_owner_generation",
                        f"voice-engine:{os.getpid()}:{id(self)}",
                    ),
                    evidence=evidence,
                    liveness_contract="on_demand",
                )
                mycelium.attest_neural_root(
                    "service:afplay",
                    root_kind="hardware",
                    target_id="coreaudio:default_output",
                    owner_generation=getattr(
                        self,
                        "_voice_owner_generation",
                        f"voice-engine:{os.getpid()}:{id(self)}",
                    ),
                    evidence=evidence,
                    liveness_contract="on_demand",
                )
        return played

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
        stale_queues: list[asyncio.Queue] = []
        for queue_ref in list(self._sse_queues):
            try:
                queue_ref.put_nowait(payload)
            except asyncio.QueueFull:
                stale_queues.append(queue_ref)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
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
            
            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

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

        audio_data = await asyncio.to_thread(_get_audio)
        
        if not audio_data:
            return

        await self._emit_tts_audio(audio_data)

        await self._play_locally(audio_data)

    async def synthesize_speech(self, text: str):
        """Single-string wrapper for TTS synthesis (used by local_voice_cortex)."""
        if not text or not text.strip():
            return ""
        if not getattr(self, "speaking_enabled", True):
            logger.debug("🔇 TTS suppressed: speaking_enabled=False")
            return ""
        if not _user_voice_output_enabled():
            logger.debug("🔇 TTS suppressed: voice.output_enabled=False (user setting)")
            return ""

        async def _iter():
            yield text
        return await self.speak_stream(_iter())

    async def speak(self, text: str):
        """Alias for synthesize_speech."""
        await self.synthesize_speech(text)

    async def speak_stream(self, text_iterator) -> str:
        """Plays TTS audio and returns exactly what was successfully spoken."""
        if not getattr(self, "speaking_enabled", True):
            logger.debug("🔇 TTS stream suppressed: speaking_enabled=False")
            return ""
        if not _user_voice_output_enabled():
            logger.debug("🔇 TTS stream suppressed: voice.output_enabled=False (user setting)")
            return ""

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
                while not self.interrupt_flag.is_set():
                    try:
                        if self.interrupt_flag.is_set():
                            logger.info("🛑 Aura interrupted. Halting synthesis.")
                            break
                        
                        text_chunk = await it.__anext__()
                        if not text_chunk or not text_chunk.strip():
                            continue
                    except StopAsyncIteration:
                        break
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        record_degradation('voice_engine', e)
                        logger.error("Error in voice stream: %s", e)
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

            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('voice_engine', e)
                logger.error("Playback error in stream: %s", e)
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

        await asyncio.to_thread(_say)

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
        try:
            asyncio.get_running_loop()
            get_task_tracker().create_task(
                self.start_listening(),
                name="voice_engine.start_listening",
            )
        except RuntimeError as _e:
            # Fallback if unmuted from non-async context
            logger.debug('Ignored RuntimeError in voice_engine.py: %s', _e)
        logger.info("🔊 Voice Engine unmuted (STT + TTS enabled)")

    async def reset(self):
        """Full reset — stop listening, clear buffers."""
        self.stop_listening()
        self._is_feeding = False
        thread = self._stt_thread
        if thread is not None and thread.is_alive():
            # Thread will exit on next loop iteration due to _is_feeding=False
            pass  # no-op: intentional

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

    async def _run_transcript_callbacks(
        self,
        text: str,
        *,
        authorized_command: bool = True,
    ) -> None:
        callbacks: list[Callable[[str], Awaitable[None]]] = []
        for key in list(self._transcript_callbacks.keys()):
            if authorized_command or key in self._candidate_transcript_callbacks:
                callbacks.append(self._transcript_callbacks[key])
        if authorized_command:
            callbacks.extend(self._anonymous_transcript_callbacks)

        if authorized_command and not callbacks and self._on_transcript is not None:
            callbacks.append(self._on_transcript)

        for callback in callbacks:
            try:
                logger.debug("Routing transcript through callback %s: %s", callback, text[:50])
                res = callback(text)
                if inspect.isawaitable(res):
                    await res
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "voice_engine",
                    exc,
                    action="continued transcript fanout after one voice callback failed",
                )
                logger.error("Transcript callback failed: %s", exc, exc_info=True)

    def on_transcript(
        self,
        callback: Callable[[str], Awaitable[None]],
        *,
        key: str | None = None,
        replace: bool = False,
        candidate_safe: bool = False,
    ):
        """Register a transcript callback without stealing existing listeners.

        Candidate-safe listeners may observe ambient STT for perception or
        wake-word detection. Other listeners receive only authorized commands,
        so background audio cannot enter the user-input cognition path.
        """
        if replace:
            self._transcript_callbacks.clear()
            self._candidate_transcript_callbacks.clear()
            self._anonymous_transcript_callbacks.clear()
        if key:
            self._transcript_callbacks[key] = callback
            if candidate_safe:
                self._candidate_transcript_callbacks.add(key)
            else:
                self._candidate_transcript_callbacks.discard(key)
        elif callback not in self._anonymous_transcript_callbacks:
            self._anonymous_transcript_callbacks.append(callback)
        self._on_transcript = self._run_transcript_callbacks

    async def _set_state(self, new_state: VoiceState):
        if self.state != new_state:
            self.state = new_state
            if self._on_state_change:
                await self._on_state_change(new_state)

    def get_status(self) -> dict:
        tts_type = "Not loaded"
        if self._xtts_engine:
            tts_type = "Sara v3 (XTTS-v2)"
        elif self._piper_voice:
            tts_type = f"Piper ({self.piper_voice_name})"
        elif self.tts_engine:
            tts_type = "pyttsx3 (Native)"
        coqui_tts_available = _tts_dependency_available()
        piper_tts_available = _piper_dependency_available()
        pyttsx3_available = pyttsx3 is not None
        
        return {
            "state": self.state.name,
            "stt": "Whisper (Direct)" if self._stt_initialized else "Not loaded",
            "tts": tts_type,
            "mic": self.microphone_enabled,
            "speaking": self.speaking_enabled,
            "auto_listen": self.auto_listen_enabled,
            "listening": self._mic_listening,
            "server_capture": _sounddevice_available(),
            "capture_available": _sounddevice_available(),
            "stt_available": _stt_dependency_available(),
            "tts_available": coqui_tts_available or piper_tts_available or pyttsx3_available,
            "coqui_tts_available": coqui_tts_available,
            "piper_tts_available": piper_tts_available,
            "pyttsx3_available": pyttsx3_available,
            "stt_initialized": self._stt_initialized,
            "stt_load_state": self._stt_load_state,
            "stt_last_error": self._stt_last_error,
            "stt_local_files_only": not _env_flag("AURA_STT_ALLOW_MODEL_DOWNLOAD", False),
            "stt_init_in_flight": bool(
                self._stt_init_task is not None and not self._stt_init_task.done()
            ),
            "tts_initialized": self._tts_initialized,
            "mic_start_in_flight": bool(
                self._mic_start_task is not None and not self._mic_start_task.done()
            ),
            "closing": self._voice_closing(),
            "capture_backend": "sounddevice" if _sounddevice_available() else "unavailable",
            "stt_backend": "faster_whisper" if _stt_dependency_available() else "unavailable",
            "tts_backend": tts_type,
            "tts_import_error": _tts_api_import_error,
            "device": {
                "generation": self._mic_device_generation,
                "state": self._mic_device_state,
                "reason": self._mic_device_reason,
                "capture_active": bool(
                    self._mic_listening and self._mic_stream is not None
                ),
                "frames_seen": self._mic_frames_seen,
                "last_frame_age_s": (
                    round(max(0.0, time.monotonic() - self._mic_last_frame_at), 3)
                    if self._mic_last_frame_at
                    else None
                ),
            },
        }

# ── Singleton ─────────────────────────────────────────────

_voice_engine: SovereignVoiceEngine | None = None


def get_voice_engine(**kwargs) -> SovereignVoiceEngine:
    global _voice_engine
    if _voice_engine is None:
        _voice_engine = SovereignVoiceEngine(**kwargs)
    return _voice_engine
