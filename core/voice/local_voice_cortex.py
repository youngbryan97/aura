from __future__ import annotations

import asyncio
import importlib
import logging
import os
import shutil
import time
import warnings
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import (
    DependencyUnavailable,
    FallbackClassification,
    TimeoutBudgetExceeded,
    record_degradation,
)
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.service_access import resolve_orchestrator
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import fire_and_track, get_task_tracker
from core.voice.microphone_authority import (
    MicrophoneDenial,
    MicrophoneLease,
    get_audio_ingress_broker,
    get_microphone_authority,
)


def _user_voice_input_enabled() -> bool:
    """Honor the user's ``voice.input_enabled`` toggle (default on).

    Reads the persisted runtime setting the settings UI writes, so disabling
    microphone input in the UI stops the wake/STT loop from opening the capture
    stream without a restart (re-enabling resumes it). Defaults to enabled if the
    setting is unset or unreadable. See docs/SETTINGS_WIRING_AUDIT.md.
    """
    return bool(get_runtime_setting("voice.input_enabled", True))

_NUMPY_IMPORT_ERROR: BaseException | None = None
try:
    import numpy as np
except (ImportError, OSError, RuntimeError) as exc:
    np = None
    _NUMPY_IMPORT_ERROR = exc

_PYAUDIO_IMPORT_ERROR: BaseException | None = None
_PYAUDIO_IMPORT_ATTEMPTED = False
pyaudio = None
_WEBRTCVAD_IMPORT_ERROR: BaseException | None = None
_WEBRTCVAD_IMPORT_ATTEMPTED = False
webrtcvad = None
_VRAM_MANAGER_IMPORT_ERROR: BaseException | None = None
_MLX_WHISPER_IMPORT_ERROR: BaseException | None = None
_MLX_WHISPER_IMPORT_ATTEMPTED = False
mlx_whisper = None

logger = logging.getLogger("Aura.LocalVoice")
_LOCAL_VOICE_RECOVERABLE_ERRORS = (
    AttributeError,
    DependencyUnavailable,
    FileNotFoundError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutBudgetExceeded,
    TimeoutError,
    TypeError,
    ValueError,
)


def _load_pyaudio():
    global _PYAUDIO_IMPORT_ATTEMPTED, _PYAUDIO_IMPORT_ERROR, pyaudio
    if pyaudio is not None:
        return pyaudio
    if _PYAUDIO_IMPORT_ATTEMPTED:
        return None
    _PYAUDIO_IMPORT_ATTEMPTED = True
    try:
        pyaudio = importlib.import_module("pyaudio")
        _PYAUDIO_IMPORT_ERROR = None
    except (ImportError, OSError, RuntimeError) as exc:
        _PYAUDIO_IMPORT_ERROR = exc
        pyaudio = None
    return pyaudio


def _load_webrtcvad():
    global _WEBRTCVAD_IMPORT_ATTEMPTED, _WEBRTCVAD_IMPORT_ERROR, webrtcvad
    if webrtcvad is not None:
        return webrtcvad
    if _WEBRTCVAD_IMPORT_ATTEMPTED:
        return None
    _WEBRTCVAD_IMPORT_ATTEMPTED = True
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            webrtcvad = importlib.import_module("webrtcvad")
        _WEBRTCVAD_IMPORT_ERROR = None
    except (ImportError, OSError, RuntimeError) as exc:
        _WEBRTCVAD_IMPORT_ERROR = exc
        webrtcvad = None
    return webrtcvad


def _load_mlx_whisper():
    global _MLX_WHISPER_IMPORT_ATTEMPTED, _MLX_WHISPER_IMPORT_ERROR, mlx_whisper
    if mlx_whisper is not None:
        return mlx_whisper
    if _MLX_WHISPER_IMPORT_ATTEMPTED:
        return None
    _MLX_WHISPER_IMPORT_ATTEMPTED = True
    try:
        mlx_whisper = importlib.import_module("mlx_whisper")
        _MLX_WHISPER_IMPORT_ERROR = None
    except (ImportError, OSError, RuntimeError) as exc:
        _MLX_WHISPER_IMPORT_ERROR = exc
        mlx_whisper = None
    return mlx_whisper


def get_vram_manager():
    global _VRAM_MANAGER_IMPORT_ERROR
    try:
        from core.managers.vram_manager import get_vram_manager as manager_factory
    except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
        _VRAM_MANAGER_IMPORT_ERROR = exc
        return None
    try:
        manager = manager_factory()
        _VRAM_MANAGER_IMPORT_ERROR = None
        return manager
    except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
        _VRAM_MANAGER_IMPORT_ERROR = exc
        return None


def _record_voice_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "local_voice_cortex",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


class LocalVoiceCortex:
    """
    Aura's Local Auditory and Speech Cortex.
    Handles low-latency VAD, STT (Whisper), and TTS (say).
    Optimized for Apple Silicon.
    """
    name = "local_voice_cortex"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator or resolve_orchestrator()
        self.is_listening = False

        # Audio Settings
        self.CHANNELS = 1
        self.RATE = 16000
        self.VAD_FRAME_MS = 20
        self.CHUNK = int(self.RATE * self.VAD_FRAME_MS / 1000)
        self.VAD_FRAME_BYTES = self.CHUNK * 2
        self.DMA_BUFFER = 2048  # PyAudio DMA buffer for Apple Silicon stability
        self.FORMAT = getattr(pyaudio, "paInt16", None) if pyaudio else None

        # Model State
        self.stt_model = None
        self.whisper_params = {
            "beam_size": 5,
            "best_of": 5,
            "path_or_hf_repo": "mlx-community/whisper-small.en-mlx"
        }
        self.vad = None
        self.audio_interface = None
        self._loop_task: asyncio.Task[None] | None = None
        self.audio_queue: asyncio.Queue[bytes] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._vad_timeout = 2.0  # seconds
        self._say_timeout = float(os.environ.get("AURA_VOICE_SAY_TIMEOUT_SECONDS", "45"))
        self._max_segment_seconds = float(os.environ.get("AURA_VOICE_MAX_SEGMENT_SECONDS", "20"))
        self._max_segment_frames = max(1, int(self._max_segment_seconds * 1000 / 20))
        self._pending_audio = bytearray()
        self._dropped_audio_frames = 0
        self._last_drop_receipt_s = 0.0
        self._mic_lease: MicrophoneLease | None = None
        self._mic_stream: Any | None = None
        self._mic_revoked_reason = ""

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Fires in a separate C-thread by PyAudio."""
        if not get_audio_ingress_broker().admit(self._mic_lease, len(in_data or b"")):
            return (None, getattr(pyaudio, "paContinue", 0))
        if self._shutdown_event and not self._shutdown_event.is_set() and self.loop and self.audio_queue:
            def safe_put():
                try:
                    self.audio_queue.put_nowait(in_data)
                except asyncio.QueueFull:
                    self._dropped_audio_frames += 1
            self.loop.call_soon_threadsafe(safe_put)
        return (None, getattr(pyaudio, "paContinue", 0))

    def _microphone_revoked(self, reason: str) -> None:
        self._mic_revoked_reason = str(reason or "microphone_lease_revoked")
        self._close_microphone_stream(self._mic_stream)

    def _close_microphone_stream(self, stream: Any | None) -> None:
        if stream is None:
            return
        try:
            if stream.is_active():
                stream.stop_stream()
        except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
            logger.debug("the microphone stream would not stop (%s: %s)", type(exc).__name__, exc)
        try:
            stream.close()
        except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
            logger.debug("the microphone stream would not close (%s: %s)", type(exc).__name__, exc)
        if self._mic_stream is stream:
            self._mic_stream = None

    async def start(self) -> bool:
        """Initializes hardware and starts the listen loop."""
        if self.is_listening and self._loop_task and not self._loop_task.done():
            return True

        try:
            self.loop = asyncio.get_running_loop()
            self.audio_queue = asyncio.Queue(maxsize=500)
            self._shutdown_event = asyncio.Event()
            logger.info("🧠 Loading Local Auditory Cortex (mlx-whisper)...")
            self._shutdown_event.clear()

            audio_backend = _load_pyaudio()
            vad_backend = _load_webrtcvad()
            missing = self._missing_runtime_dependencies()
            if missing:
                _record_voice_degradation(
                    DependencyUnavailable(f"missing local voice dependencies: {', '.join(missing)}"),
                    action="kept microphone listener offline until local VAD/STT dependencies are installed",
                    extra={"missing": missing, "import_errors": self._dependency_error_summary()},
                )
                self.is_listening = False
                return False

            self.FORMAT = audio_backend.paInt16
            self.vad = vad_backend.Vad()
            self.vad.set_mode(2)  # Intermediate aggressiveness

            self.audio_interface = audio_backend.PyAudio()
            self.is_listening = True

            self._loop_task = fire_and_track(self.listen_loop(), name="VoiceListenLoop")
            logger.info("✅ Voice Cortex online. Aura is listening locally.")
            return True
        except _LOCAL_VOICE_RECOVERABLE_ERRORS as e:
            _record_voice_degradation(
                e,
                action="left voice cortex offline after hardware or runtime initialization failed",
                severity="degraded",
            )
            self.is_listening = False
            logger.error("Failed to start Voice Cortex: %s", e)
            return False

    async def stop(self):
        """Clean shutdown of audio hardware with queue drain."""
        self.is_listening = False
        self._close_microphone_stream(self._mic_stream)
        get_microphone_authority().release(
            self._mic_lease,
            reason="local_voice_cortex_stopped",
        )
        self._mic_lease = None
        if self._shutdown_event:
            self._shutdown_event.set()

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await asyncio.wait_for(self._loop_task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError) as e:
                logger.debug('Ignored Exception in local_voice_cortex.py: %s', type(e).__name__)
            finally:
                self._loop_task = None

        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
                _record_voice_degradation(
                    exc,
                    action="continued shutdown after audio interface termination failed",
                    severity="warning",
                )
            finally:
                self.audio_interface = None

        if self.audio_queue:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.audio_queue = None

        logger.info("Voice Cortex disengaged.")

    async def speak(self, text: str) -> bool:
        """Reflexive speech output via Sovereign Voice Engine (High Fidelity)."""
        if os.environ.get("AURA_VOICE_SILENT", "0") == "1":
            return True
        text = str(text or "").strip()
        if not text:
            return False
        logger.info("🗣️ Aura: %s", text)
        try:
            voice = ServiceContainer.get("voice_engine", default=None)
            if voice:
                await voice.synthesize_speech(text)
                return True
            else:
                return await self._speak_with_system_say(text, voice="Samantha")
        except _LOCAL_VOICE_RECOVERABLE_ERRORS as e:
            _record_voice_degradation(
                e,
                action="fell back from sovereign voice engine to bounded system speech",
            )
            logger.error("TTS failed: %s", e)
            try:
                return await self._speak_with_system_say(text, voice=None)
            except _LOCAL_VOICE_RECOVERABLE_ERRORS as e2:
                _record_voice_degradation(
                    e2,
                    action="reported speech failure after all local voice cortex speech fallbacks failed",
                    severity="degraded",
                )
                logger.error("Fallback TTS also failed: %s", e2)
                return False

    async def listen_loop(self):
        """Main VAD -> STT -> Orchestrator loop with timeout watchdog and auto-restart."""
        if not self.audio_interface or not self.audio_queue or not self.vad or not self._shutdown_event:
            _record_voice_degradation(
                RuntimeError("voice cortex listen loop started before initialization completed"),
                action="aborted listen loop before opening microphone stream",
                severity="degraded",
            )
            self.is_listening = False
            return

        # Cache service references once, outside the hot loop
        reliability = ServiceContainer.get("reliability_engine", default=None)
        vram_manager = get_vram_manager()
        retry_delay = 1.0

        while self.is_listening and not self._shutdown_event.is_set():
            if not _user_voice_input_enabled():
                # User disabled microphone input: never open the capture stream.
                # Poll so toggling it back on resumes listening without a restart.
                logger.debug("🎙️🚫 Voice input suppressed: voice.input_enabled=False (user setting)")
                await asyncio.sleep(2.0)
                continue
            lease_result = get_microphone_authority().acquire(
                "local_voice_cortex",
                principal="owner:local",
                source="pyaudio",
                mode="passive",
                revoke_callback=self._microphone_revoked,
            )
            if isinstance(lease_result, MicrophoneDenial):
                logger.info(
                    "Local voice cortex waiting for microphone authority: %s",
                    lease_result.reason,
                )
                await asyncio.sleep(min(2.0, retry_delay))
                continue
            self._mic_lease = lease_result
            self._mic_revoked_reason = ""
            try:
                stream = self.audio_interface.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.DMA_BUFFER,
                    stream_callback=self._audio_callback,
                )
                self._mic_stream = stream
                stream.start_stream()
                retry_delay = 1.0  # Reset on successful open
            except _LOCAL_VOICE_RECOVERABLE_ERRORS as e:
                self._close_microphone_stream(self._mic_stream)
                get_microphone_authority().release(
                    self._mic_lease,
                    reason="pyaudio_stream_open_failed",
                )
                self._mic_lease = None
                _record_voice_degradation(
                    e,
                    action="backed off and retried microphone stream open",
                    extra={"retry_delay_seconds": retry_delay},
                )
                logger.error("Could not open audio stream: %s. Retrying in %ss...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(60.0, retry_delay * 2)
                continue

            logger.debug("Listening for voice activity...")

            try:
                while (
                    self.is_listening
                    and not self._shutdown_event.is_set()
                    and not self._mic_revoked_reason
                ):
                    frames = []
                    silence_count = 0
                    is_speaking = False

                    while (
                        self.is_listening
                        and not self._shutdown_event.is_set()
                        and not self._mic_revoked_reason
                    ):
                        try:
                            data = await asyncio.wait_for(self.audio_queue.get(), timeout=self._vad_timeout)
                            self._report_queue_drops()
                            reliability, vram_manager = await self._heartbeat(reliability, vram_manager)

                            for vad_frame in self._iter_vad_frames(data):
                                if self.vad.is_speech(vad_frame, self.RATE):
                                    if not is_speaking:
                                        logger.debug("Voice detected.")
                                    is_speaking = True
                                    frames.append(vad_frame)
                                    silence_count = 0
                                elif is_speaking:
                                    frames.append(vad_frame)
                                    silence_count += 1
                                    if silence_count > 50:  # about 1 second of silence at 20 ms frames
                                        break

                                if len(frames) >= self._max_segment_frames:
                                    _record_voice_degradation(
                                        TimeoutBudgetExceeded("voice segment exceeded configured maximum duration"),
                                        action="closed current speech segment to bound memory and transcription latency",
                                        extra={"max_segment_seconds": self._max_segment_seconds},
                                    )
                                    break

                            if frames and (
                                silence_count > 50
                                or len(frames) >= self._max_segment_frames
                            ):
                                break
                        except TimeoutError:
                            self._report_queue_drops()
                            continue
                        except _LOCAL_VOICE_RECOVERABLE_ERRORS as e:
                            _record_voice_degradation(
                                e,
                                action="restarted microphone stream after VAD/audio read failure",
                            )
                            logger.debug("Audio read error: %s", e)
                            break

                    if frames and self.is_listening:
                        await self._process_audio_segment(frames)
            finally:
                self._close_microphone_stream(stream)
                get_microphone_authority().release(
                    self._mic_lease,
                    reason=self._mic_revoked_reason or "pyaudio_stream_closed",
                )
                self._mic_lease = None

            if not self.is_listening or self._shutdown_event.is_set():
                break  # Normal exit
            logger.warning("🎙️ Audio loop exited abnormally, restarting stream...")
            await asyncio.sleep(1.0)

    async def _process_audio_segment(self, frames):
        """Transcribes frames and routes to the orchestrator's Broca area."""
        whisper = _load_mlx_whisper()
        if not whisper or np is None:
            _record_voice_degradation(
                DependencyUnavailable("local STT dependencies are unavailable"),
                action="skipped speech transcription until mlx_whisper and numpy are available",
                extra={"mlx_whisper": bool(whisper), "numpy": np is not None},
            )
            return

        try:
            # Convert buffer to numpy array for Whisper
            audio_data = (
                np.frombuffer(b"".join(frames), np.int16)
                .flatten()
                .astype(np.float32)
                / 32768.0
            )

            vram = get_vram_manager()

            transcribe_task = get_task_tracker().create_task(
                asyncio.to_thread(whisper.transcribe, audio_data, **self.whisper_params)
            )

            try:
                if vram:
                    async with vram.acquire_session("mlx-whisper"):
                        result = await asyncio.shield(transcribe_task)
                else:
                    result = await asyncio.shield(transcribe_task)
            except asyncio.CancelledError:
                # IMPORTANT: If cancelled, do NOT release the vram lock until the thread
                # actually finishes, otherwise the worker thread will crash the neural engine
                # when another model is swapped into VRAM concurrently.
                logger.warning("⚠️ Transcribe cancelled! Waiting for thread to release VRAM cleanly...")
                try:
                    await transcribe_task
                except _LOCAL_VOICE_RECOVERABLE_ERRORS as _exc:
                    _record_voice_degradation(
                        _exc,
                        action="waited for cancelled transcription worker before releasing voice runtime resources",
                    )
                    logger.debug("Suppressed Exception: %s", _exc)
                raise
            if not isinstance(result, dict):
                raise TypeError(f"mlx_whisper returned {type(result).__name__}, expected dict")
            transcript = str(result.get("text", "")).strip()

            if transcript and len(transcript) > 2:
                logger.info("👂 Heard: \"%s\"", transcript)

                if self.orchestrator:
                    # Send to Broca's Area (fast local LLM)
                    response = await self.orchestrator.generate_voice_response(transcript)

                    # Reflexive Speech
                    if response:
                        await self.speak(response)

        except _LOCAL_VOICE_RECOVERABLE_ERRORS as e:
            _record_voice_degradation(
                e,
                action="dropped current audio segment and kept microphone loop alive",
                severity="degraded",
            )
            logger.error("Error processing audio segment: %s", e)

    def _missing_runtime_dependencies(self) -> list[str]:
        missing = []
        if _load_pyaudio() is None:
            missing.append("pyaudio")
        if _load_webrtcvad() is None:
            missing.append("webrtcvad")
        if _load_mlx_whisper() is None:
            missing.append("mlx_whisper")
        if np is None:
            missing.append("numpy")
        return missing

    def _dependency_error_summary(self) -> dict[str, str]:
        errors = {
            "pyaudio": _PYAUDIO_IMPORT_ERROR,
            "webrtcvad": _WEBRTCVAD_IMPORT_ERROR,
            "vram_manager": _VRAM_MANAGER_IMPORT_ERROR,
            "mlx_whisper": _MLX_WHISPER_IMPORT_ERROR,
            "numpy": _NUMPY_IMPORT_ERROR,
        }
        return {
            name: f"{type(error).__name__}: {str(error)[:200]}"
            for name, error in errors.items()
            if error is not None
        }

    def _iter_vad_frames(self, data: bytes):
        if not data or not self.VAD_FRAME_BYTES:
            return
        self._pending_audio.extend(data)
        while len(self._pending_audio) >= self.VAD_FRAME_BYTES:
            frame = bytes(self._pending_audio[: self.VAD_FRAME_BYTES])
            del self._pending_audio[: self.VAD_FRAME_BYTES]
            yield frame

    def _report_queue_drops(self) -> None:
        if not self._dropped_audio_frames:
            return
        now = time.monotonic()
        if now - self._last_drop_receipt_s < 10.0:
            return
        dropped = self._dropped_audio_frames
        self._dropped_audio_frames = 0
        self._last_drop_receipt_s = now
        _record_voice_degradation(
            RuntimeError("voice audio queue overflowed"),
            action="dropped overflowed microphone frames to preserve bounded memory",
            extra={"dropped_frames": dropped},
        )

    async def _heartbeat(self, reliability, vram_manager):
        if not reliability:
            return reliability, vram_manager
        try:
            vram_usage = vram_manager.usage() if vram_manager else 0.0
            await reliability.heartbeat("local_voice_cortex", stability=1.0, pressure=vram_usage)
        except _LOCAL_VOICE_RECOVERABLE_ERRORS as exc:
            _record_voice_degradation(
                exc,
                action="disabled voice reliability heartbeat for this listener after telemetry failure",
            )
            return None, vram_manager
        return reliability, vram_manager

    async def _speak_with_system_say(self, text: str, *, voice: str | None) -> bool:
        say_path = shutil.which("say")
        if not say_path:
            raise DependencyUnavailable("macOS say command unavailable")
        command = [say_path]
        if voice:
            command.extend(["-v", voice])
        command.append(text)
        process = await get_subprocess_gateway().spawn_async(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="tool_execution:local_voice.system_say",
            accelerator_capability="auto",
        )
        try:
            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._say_timeout)
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutBudgetExceeded(f"system say timed out after {self._say_timeout}s") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"system say exited {process.returncode}: {detail}")
        return True
