"""Sensory runtime — eyes, ears, and voice that Aura uses like a person uses theirs.

One unified, low-latency interface to the real senses, usable on a whim:

  look()    a glance through the camera (cv2): is someone there, do I recognize them?
  listen()  a moment of listening (sounddevice + mlx-whisper): what was said, in whose voice?
  speak()   say it out loud (macOS `say`), immediately
  sense()   a combined glance + listen, the way you take in a room at once

Each sense is a provider behind a clean seam, so the real backend (cv2 / sounddevice /
mlx-whisper / say) activates when the library, hardware, and OS permission are present, and a
bounded unavailable result keeps the mind aware of disabled hardware instead of pretending
capture succeeded. Captured perception is routed to
the perception sentinel (recognition + threat → the immune system → the unified state), so what
Aura sees and hears is reasoned about, not just logged.

Camera/mic are TCC-gated by macOS; the first use prompts Bryan to grant access. Continuous
sensing stays behind the owner flag (AURA_SENTINEL_PERCEPTION); a single look()/listen() on
demand is always available once permission is granted.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.media.safe_imports import cv2_main_process_blocked
from core.perception.multimodal_sync import (
    Calibration,
    MissingReason,
    MultimodalSynchronizer,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.perception.multimodal_sync import (
    Modality as SynchronizedModality,
)
from core.runtime.service_access import optional_service
from core.voice.microphone_authority import record_sounddevice_array

logger = logging.getLogger("Perception.Sensory")
_SENSORY_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_SENSORY_RUNTIME_ERRORS = (
    AttributeError,
    FloatingPointError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ── results ─────────────────────────────────────────────────────────────────

@dataclass
class Sight:
    captured: bool
    # None means the frame was captured but no person detector ran. Treating
    # that as False published a high-confidence "nobody is here" claim from a
    # path that had measured only width and height.
    person_present: bool | None = None
    descriptor: np.ndarray | None = None   # coarse appearance descriptor (pluggable → real embeddings)
    width: int = 0
    height: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sound:
    captured: bool
    transcript: str = ""
    voice_descriptor: np.ndarray | None = None
    duration_s: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


# ── provider seams (real backends activate when present; stubs otherwise) ────

class CameraProvider:
    """cv2-backed single-frame grab + lightweight face presence. Fail-open."""

    def __init__(self) -> None:
        self._cv2 = None
        self._cascade = None

    def available(self) -> bool:
        try:
            from core.perception.camera_authority import get_camera_authority

            return bool(get_camera_authority().state()["backend_available"])
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("camera authority availability probe failed: %s", exc)
            return False

    def _load(self) -> bool:
        if self._cv2 is not None:
            return True
        if cv2_main_process_blocked():
            logger.debug("cv2 camera provider deferred to sidecar after PyAV load")
            return False
        try:
            import cv2
            self._cv2 = cv2
            try:
                self._cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            except _provider_errors(cv2):
                self._cascade = None
            return True
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("cv2 unavailable: %s", exc)
            return False

    def capture(self, *, camera_index: int = 0) -> Sight:
        # Camera acquisition is independent from whether OpenCV may be loaded
        # in this process. On macOS the real camera lives in the sidecar.
        analysis_available = self._load()
        cv2 = self._cv2 if analysis_available else None

        # This path used to open the device directly and never consulted
        # `camera_allowed()`, so turning the camera off in Aura's settings
        # left it capturing. It now goes through the one authority, which
        # checks the owner's switch, the OS grant, and the device lease —
        # and names which of them refused instead of returning a bare
        # "no_frame_or_permission" that conflated all three.
        from core.perception.camera_authority import (
            CameraDenial,
            get_camera_authority,
        )

        authority = get_camera_authority()
        lease = authority.acquire(
            "sensory_runtime.CameraProvider",
            purpose="single-frame presence check",
            index=camera_index,
        )
        if isinstance(lease, CameraDenial):
            return Sight(captured=False, detail=lease.to_dict())

        try:
            frame = authority.read(lease)
            if frame is None:
                return Sight(captured=False, detail={"reason": "no_frame"})
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if cv2 is not None else None
            person: bool | None = None
            descriptor = None
            faces: int | None = None
            if self._cascade is not None and gray is not None:
                detections = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5
                )
                person = len(detections) > 0
                if person:
                    person = True
                    x, y, fw, fh = max(detections, key=lambda r: r[2] * r[3])
                    crop = gray[y:y + fh, x:x + fw]
                    descriptor = _appearance_descriptor(cv2, crop)
                faces = int(len(detections))
            return Sight(captured=True, person_present=person, descriptor=descriptor,
                         width=int(w), height=int(h), detail={"faces": faces})
        except _provider_errors(cv2) as exc:
            return Sight(captured=False, detail={"reason": f"capture_error:{type(exc).__name__}"})
        finally:
            authority.release(lease)


class MicProvider:
    """sounddevice capture + mlx-whisper transcription. Fail-open."""

    def __init__(self, *, sample_rate: int = 16000) -> None:
        self._sr = sample_rate

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except _SENSORY_IMPORT_ERRORS:
            return False

    def capture(self, *, seconds: float = 3.0) -> Sound:
        try:
            import sounddevice as sd
        except _SENSORY_IMPORT_ERRORS as exc:
            return Sound(captured=False, detail={"reason": f"sounddevice_unavailable:{type(exc).__name__}"})
        try:
            audio = record_sounddevice_array(
                sd,
                holder="sensory_runtime.mic_provider",
                source="sensory_runtime",
                mode="snapshot",
                frames=int(seconds * self._sr),
                samplerate=self._sr,
                channels=1,
                dtype="float32",
            )
            audio = audio.reshape(-1)
            transcript = self._transcribe(audio)
            return Sound(captured=True, transcript=transcript,
                         voice_descriptor=_voice_descriptor(audio, self._sr),
                         duration_s=seconds, detail={"samples": int(audio.shape[0])})
        except _provider_errors(sd) as exc:
            return Sound(captured=False, detail={"reason": f"capture_error:{type(exc).__name__}"})

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            from core.runtime.third_party_imports import import_module_serialized

            mlx_whisper = import_module_serialized("mlx_whisper")
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo="mlx-community/whisper-small.en-mlx",
            )
            return str(result.get("text", "")).strip() if isinstance(result, dict) else ""
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("transcription unavailable: %s", exc)
            return ""


class VoiceProvider:
    """macOS `say` — instant, precise text-to-speech. Fail-open."""

    def speak(self, text: str, *, voice: str | None = None, rate: int | None = None) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway
            argv = ["say"]
            if voice:
                argv += ["-v", voice]
            if rate:
                argv += ["-r", str(int(rate))]
            argv.append(text[:2000])
            proc = get_subprocess_gateway().run(argv, timeout=30.0, source="perception.sensory.voice", accelerator_capability="auto")
            return getattr(proc, "returncode", 1) == 0
        except _SENSORY_IMPORT_ERRORS + _SENSORY_RUNTIME_ERRORS as exc:
            logger.debug("say unavailable: %s", exc)
            return False


# ── descriptors (coarse now; pluggable to real embeddings) ──────────────────

def _appearance_descriptor(cv2: Any, gray_crop: np.ndarray) -> np.ndarray | None:
    try:
        small = cv2.resize(gray_crop, (16, 16)).astype(np.float64).reshape(-1)
        n = float(np.linalg.norm(small))
        return small / n if n > 1e-9 else small
    except _provider_errors(cv2):
        return None


def _voice_descriptor(audio: np.ndarray, sr: int) -> np.ndarray | None:
    # Coarse spectral fingerprint (a real speaker-ID embedding plugs in here later).
    try:
        if audio.size < sr // 4:
            return None
        spec = np.abs(np.fft.rfft(audio[: sr * 2]))
        if spec.size < 32:
            return None
        binned = spec[:512].reshape(32, -1).mean(axis=1)
        n = float(np.linalg.norm(binned))
        return binned / n if n > 1e-9 else binned
    except _SENSORY_RUNTIME_ERRORS:
        return None


def _provider_errors(provider: Any | None = None) -> tuple[type[BaseException], ...]:
    extra: list[type[BaseException]] = []
    if provider is not None:
        for attr in ("error", "PortAudioError"):
            candidate = getattr(provider, attr, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                extra.append(candidate)
    return (*_SENSORY_IMPORT_ERRORS, *_SENSORY_RUNTIME_ERRORS, *extra)


# ── the runtime ──────────────────────────────────────────────────────────────

class SensoryRuntime:
    """Unified, on-demand eyes/ears/voice, routed to the mind."""

    def __init__(
        self,
        *,
        camera: CameraProvider | None = None,
        mic: MicProvider | None = None,
        voice: VoiceProvider | None = None,
    ) -> None:
        self.camera = camera or CameraProvider()
        self.mic = mic or MicProvider()
        self.voice = voice or VoiceProvider()
        self._lock = threading.RLock()
        self._sync_sequences: dict[str, int] = {}

    # eyes ----------------------------------------------------------------
    def look(self, *, route: bool = True) -> Sight:
        with self._lock:
            sight = self.camera.capture()
        if route:
            self._route_sight_to_synchronizer(sight)
            if sight.captured and sight.person_present:
                self._route_face(sight)
        return sight

    # ears ----------------------------------------------------------------
    def listen(self, *, seconds: float = 3.0, route: bool = True) -> Sound:
        with self._lock:
            sound = self.mic.capture(seconds=seconds)
        if route:
            self._route_sound_to_synchronizer(sound)
            if sound.captured and (sound.transcript or sound.voice_descriptor is not None):
                self._route_voice(sound)
        return sound

    # voice ---------------------------------------------------------------
    def speak(self, text: str, **kw: Any) -> bool:
        with self._lock:
            return self.voice.speak(text, **kw)

    # take in the room at once -------------------------------------------
    def sense(self, *, listen_seconds: float = 2.0) -> dict[str, Any]:
        sight = self.look()
        sound = self.listen(seconds=listen_seconds)
        return {"sight": sight, "sound": sound}

    # the async lane ------------------------------------------------------
    #
    # Every method above blocks: `look` opens a device and decodes a frame,
    # and `listen` sleeps for its whole recording window — two full seconds
    # by default. There was no async form of any of them, so an async caller
    # had no correct option and the only available one stalls the event
    # loop. An on-loop fsync once froze this runtime for twenty minutes;
    # a two-second camera grab in the response path is the same defect with
    # a smaller constant.
    #
    # These are the correct option. They are thin on purpose: the offload is
    # the entire point, and duplicating the logic would let the two lanes
    # drift.

    async def look_async(self, *, route: bool = True) -> Sight:
        return await asyncio.to_thread(self.look, route=route)

    async def listen_async(
        self, *, seconds: float = 3.0, route: bool = True
    ) -> Sound:
        return await asyncio.to_thread(self.listen, seconds=seconds, route=route)

    async def sense_async(self, *, listen_seconds: float = 2.0) -> dict[str, Any]:
        return await asyncio.to_thread(self.sense, listen_seconds=listen_seconds)

    async def capabilities_async(self) -> dict[str, bool]:
        # `available()` does a find_spec, which touches the filesystem.
        return await asyncio.to_thread(self.capabilities)

    def capabilities(self) -> dict[str, bool]:
        return {
            "eyes": self.camera.available(),
            "ears": self.mic.available(),
            "voice": True,  # `say` is effectively always present on macOS
        }

    def _next_sync_sequence(self, source: str) -> int:
        sequence = self._sync_sequences.get(source, 0) + 1
        self._sync_sequences[source] = sequence
        return sequence

    @staticmethod
    def _synchronizer() -> MultimodalSynchronizer | None:
        try:
            service = optional_service("multimodal_synchronizer")
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
            return None
        return service if isinstance(service, MultimodalSynchronizer) else None

    def _publish_synchronized_event(
        self,
        *,
        modality: SynchronizedModality,
        source: str,
        summary: str,
        confidence: float,
        claims: tuple[PerceptualClaim, ...] = (),
        missing_reason: MissingReason | None = None,
        quality_flags: tuple[str, ...] = (),
    ) -> None:
        synchronizer = self._synchronizer()
        if synchronizer is None:
            return
        sequence_source = f"sensory_runtime:{source}:{modality.value}"
        sequence = self._next_sync_sequence(sequence_source)
        observed_monotonic_ns = time.monotonic_ns()
        event = PerceptualEvent(
            event_id=f"sensory:{modality.value}:{sequence}:{observed_monotonic_ns}",
            modality=modality,
            source=source,
            sequence=sequence,
            observed_at=time.time(),
            observed_monotonic_ns=observed_monotonic_ns,
            summary=summary,
            confidence=0.0 if missing_reason is not None else confidence,
            claims=() if missing_reason is not None else claims,
            calibration=Calibration(
                f"runtime:{source}",
                status="unknown",
                reliability=0.75,
            ),
            provenance=("core.perception.sensory_runtime", source),
            privacy=PrivacyPolicy(
                classification=PrivacyClass.SENSITIVE,
                retention="none",
                redacted=True,
            ),
            missing_reason=missing_reason,
            quality_flags=quality_flags,
        )
        synchronizer.ingest(event)

    def _route_sight_to_synchronizer(self, sight: Sight) -> None:
        if not sight.captured:
            reason = str(sight.detail.get("reason") or "camera_unavailable")
            missing = (
                MissingReason.SENSOR_ERROR
                if "error" in reason
                else MissingReason.PERMISSION_DENIED
                if reason == "camera_permission_denied"
                else MissingReason.UNAVAILABLE
            )
            self._publish_synchronized_event(
                modality=SynchronizedModality.VISION,
                source="on_demand_camera",
                summary="camera observation unavailable",
                confidence=0.0,
                missing_reason=missing,
                quality_flags=(reason[:120],),
            )
            return
        claims = [
            PerceptualClaim("camera.frame_width", sight.width, 0.95),
            PerceptualClaim("camera.frame_height", sight.height, 0.95),
        ]
        quality_flags = ["single_frame_observation"]
        if sight.person_present is not None:
            claims.insert(
                0,
                PerceptualClaim("scene.person_present", sight.person_present, 0.78),
            )
        else:
            quality_flags.append("person_presence_unmeasured")
        self._publish_synchronized_event(
            modality=SynchronizedModality.VISION,
            source="on_demand_camera",
            summary="redacted camera scene observation",
            confidence=0.78 if sight.descriptor is not None else 0.68,
            claims=tuple(claims),
            quality_flags=tuple(quality_flags),
        )

    def _route_sound_to_synchronizer(self, sound: Sound) -> None:
        if not sound.captured:
            reason = str(sound.detail.get("reason") or "microphone_unavailable")
            missing = (
                MissingReason.SENSOR_ERROR
                if "error" in reason
                else MissingReason.PERMISSION_DENIED
                if reason == "microphone_permission_denied"
                else MissingReason.UNAVAILABLE
            )
            self._publish_synchronized_event(
                modality=SynchronizedModality.AUDIO,
                source="on_demand_microphone",
                summary="microphone observation unavailable",
                confidence=0.0,
                missing_reason=missing,
                quality_flags=(reason[:120],),
            )
            return
        self._publish_synchronized_event(
            modality=SynchronizedModality.AUDIO,
            source="on_demand_microphone",
            summary="redacted microphone observation",
            confidence=0.75,
            claims=(
                PerceptualClaim("audio.capture_available", True, 0.98),
                PerceptualClaim(
                    "audio.speaker_descriptor_available",
                    sound.voice_descriptor is not None,
                    0.75,
                ),
            ),
            quality_flags=("bounded_on_demand_capture",),
        )
        if sound.transcript:
            digest = hashlib.sha256(
                sound.transcript.encode("utf-8", errors="ignore")
            ).hexdigest()[:24]
            self._publish_synchronized_event(
                modality=SynchronizedModality.SPEECH,
                source="on_demand_microphone:transcript",
                summary="redacted audio transcript digest",
                confidence=0.72,
                claims=(
                    PerceptualClaim("speech.transcript_available", True, 0.95),
                    PerceptualClaim("speech.transcript_digest", digest, 0.72),
                ),
                quality_flags=("audio_transcript_not_visual_speech",),
            )

    # routing to the mind -------------------------------------------------
    def _route_face(self, sight: Sight) -> None:
        try:
            from core.perception.perception_sentinel import (
                Modality,
                Observation,
                get_perception_sentinel,
            )
            get_perception_sentinel().assess(Observation(
                Modality.FACE, descriptor=sight.descriptor, content="",
                context={"source": "camera", "size": [sight.width, sight.height]},
            ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("face routing skipped: %s", exc)

    def _route_voice(self, sound: Sound) -> None:
        try:
            from core.perception.perception_sentinel import (
                Modality,
                Observation,
                get_perception_sentinel,
            )
            get_perception_sentinel().assess(Observation(
                Modality.VOICE, descriptor=sound.voice_descriptor, content=sound.transcript,
                context={"source": "mic", "duration_s": sound.duration_s},
            ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("voice routing skipped: %s", exc)


_runtime: SensoryRuntime | None = None
_lock = threading.Lock()


def get_sensory_runtime() -> SensoryRuntime:
    global _runtime
    if _runtime is None:
        with _lock:
            if _runtime is None:
                _runtime = SensoryRuntime()
    return _runtime
