"""Sensory Integration System
Gives Aura access to cameras, microphones, speakers, and A/V production tools
"""
import asyncio
import base64
import hashlib
import json
import logging
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any

from core.container import ServiceContainer, ServiceLifetime
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
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.permission_gates import camera_allowed
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.service_access import optional_service
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.wall_clock import monotonic_time, wall_time
from core.voice.microphone_authority import record_sounddevice_array

logger = logging.getLogger("Aura.SensoryIntegration")


def _capture_video_to_governed_file(
    authority: Any,
    lease: Any,
    *,
    path: str,
    duration: float,
) -> dict[str, Any]:
    """Stream a bounded camera capture through the canonical file owner.

    This function is deliberately synchronous and is only called through
    ``asyncio.to_thread`` by ``VisionSystem.capture``. Keeping the blocking
    frame cadence here makes that event-loop property structural rather than
    dependent on a nested function the production linter cannot distinguish.
    """
    import av

    from core.runtime.file_write_gateway import get_file_write_gateway

    frames_written = 0
    start = monotonic_time()
    next_frame_at = start
    gateway = get_file_write_gateway()
    with gateway.open_owned_binary_adapter(
        Path(path),
        mode="w+b",
        adapter=av.open,
        adapter_kwargs={"mode": "w", "format": "mp4"},
        permissions=0o600,
        source="sensory_integration.video_capture",
    ) as container:
        stream = None
        try:
            while monotonic_time() - start < duration:
                frame = authority.read(lease)
                if frame is None:
                    if frames_written == 0:
                        return {"error": lease.last_error or "capture_failed"}
                    break
                if stream is None:
                    height, width = frame.shape[:2]
                    stream = container.add_stream("mpeg4", rate=20)
                    stream.width = int(width)
                    stream.height = int(height)
                    stream.pix_fmt = "yuv420p"
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                frames_written += 1
                next_frame_at += 0.05
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        finally:
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
    return {
        "type": "video",
        "path": path,
        "duration": round(monotonic_time() - start, 3),
        "frames": frames_written,
        "timestamp": wall_time(),
    }


class SensoryModality(Enum):
    """Types of sensory input"""

    VISION = "vision"
    HEARING = "hearing"
    TEXT = "text"


class SensorySystem:
    """Manages Aura's sensory perception across multiple modalities.
    
    Enables:
    - Vision (camera)
    - Hearing (microphone)
    - Speech (speakers/TTS)
    - Audio/visual production
    """
    
    def __init__(self) -> None:
        self.vision = VisionSystem()
        self.hearing = HearingSystem()
        self.speech = SpeechSystem()
        self.av_production = AVProductionSystem()
        
        # Sensory memory (recent perceptions)
        # Issue 43: Use deque(maxlen=) for O(1) memory management
        self.max_memory_items = 100
        self.sensory_memory: deque[dict[str, Any]] = deque(maxlen=self.max_memory_items)
        self._perception_locks: dict[SensoryModality, asyncio.Lock] = {
            modality: asyncio.Lock() for modality in SensoryModality
        }
        self._synchronizer_sequences: dict[str, int] = {}
        
    async def perceive(self, modality: SensoryModality, **kwargs: Any) -> dict[str, Any]:
        """Perceive through specified sensory modality (Async)."""
        perception: dict[str, Any] = {
            "timestamp": time.time(),
            "modality": modality.value,
            "data": None,
            "interpretation": None
        }
        
        try:
            async with self._perception_locks[modality]:
                if modality == SensoryModality.VISION:
                    perception["data"] = await self.vision.capture(**kwargs)
                    perception["interpretation"] = await self.vision.analyze(perception["data"])

                elif modality == SensoryModality.HEARING:
                    perception["data"] = await self.hearing.listen(**kwargs)
                    perception["interpretation"] = await self.hearing.transcribe(perception["data"])

                elif modality == SensoryModality.TEXT:
                    perception["data"] = kwargs.get("text")
                    perception["interpretation"] = perception["data"]

            self._publish_to_synchronizer(perception)
            self._store_in_memory(perception)
            return perception

        except (OSError, ConnectionError, TimeoutError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('sensory_integration', e)
            logger.error("Perception failed: %s", e)
            perception["error"] = str(e)
            self._publish_to_synchronizer(perception)
            return perception
    
    async def express(
        self,
        modality: SensoryModality,
        content: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Express through specified modality (Async)."""
        expression: dict[str, Any] = {
            "timestamp": time.time(),
            "modality": modality.value,
            "content": content,
            "success": False
        }
        
        try:
            if modality == SensoryModality.HEARING:
                result = await self.speech.speak(content, **kwargs)
                expression["success"] = result.get("success", False)
                expression["audio_file"] = result.get("audio_file")
            
            return expression
            
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('sensory_integration', e)
            logger.error("Expression failed: %s", e)
            expression["error"] = str(e)
            return expression
    
    def _store_in_memory(self, perception: dict[str, Any]) -> None:
        """Store a bounded metadata record without raw image/audio payloads."""

        stored = dict(perception)
        data = stored.get("data")
        if isinstance(data, dict):
            safe_data = dict(data)
            encoded = safe_data.pop("data", None)
            if isinstance(encoded, str):
                safe_data["data_digest"] = hashlib.sha256(
                    encoded.encode("ascii", errors="ignore")
                ).hexdigest()[:24]
                safe_data["encoded_chars"] = len(encoded)
                safe_data["raw_retained"] = False
            stored["data"] = safe_data
        elif isinstance(data, str):
            stored["data"] = {
                "text_digest": hashlib.sha256(data.encode("utf-8", errors="ignore")).hexdigest()[:24],
                "text_chars": len(data),
                "raw_retained": False,
            }
        interpretation = stored.get("interpretation")
        if isinstance(interpretation, dict):
            safe_interpretation = dict(interpretation)
            for key in ("text", "scene_description"):
                value = safe_interpretation.get(key)
                if isinstance(value, str) and len(value) > 4000:
                    safe_interpretation[key] = value[:4000]
                    safe_interpretation[f"{key}_truncated"] = True
            stored["interpretation"] = safe_interpretation
        stored["privacy"] = {
            "classification": "sensitive",
            "retention": "session_bounded",
            "raw_media_retained": False,
        }
        self.sensory_memory.append(stored)
    
    def get_recent_perceptions(
        self,
        modality: SensoryModality | None = None,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent perceptions, optionally filtered by modality"""
        bounded_count = max(1, min(self.max_memory_items, int(count)))
        perceptions = list(self.sensory_memory)[-bounded_count:]
        
        if modality:
            perceptions = [p for p in perceptions if p["modality"] == modality.value]
        
        return [dict(perception) for perception in perceptions]

    def _next_synchronizer_sequence(self, source: str) -> int:
        sequence = self._synchronizer_sequences.get(source, 0) + 1
        self._synchronizer_sequences[source] = sequence
        return sequence

    @staticmethod
    def _synchronizer() -> MultimodalSynchronizer | None:
        service = optional_service("multimodal_synchronizer")
        return service if isinstance(service, MultimodalSynchronizer) else None

    def _publish_event(
        self,
        *,
        modality: SynchronizedModality,
        source: str,
        confidence: float,
        claims: tuple[PerceptualClaim, ...] = (),
        missing_reason: MissingReason | None = None,
        quality_flags: tuple[str, ...] = (),
    ) -> None:
        synchronizer = self._synchronizer()
        if synchronizer is None:
            return
        sequence_source = f"sensory_system:{source}:{modality.value}"
        sequence = self._next_synchronizer_sequence(sequence_source)
        observed_monotonic_ns = time.monotonic_ns()
        synchronizer.ingest(
            PerceptualEvent(
                event_id=f"sensory-system:{modality.value}:{sequence}:{observed_monotonic_ns}",
                modality=modality,
                source=source,
                sequence=sequence,
                observed_at=time.time(),
                observed_monotonic_ns=observed_monotonic_ns,
                summary=f"redacted {modality.value} observation from sensory system",
                confidence=0.0 if missing_reason is not None else confidence,
                claims=() if missing_reason is not None else claims,
                calibration=Calibration(
                    f"runtime:sensory_system:{source}",
                    status="unknown",
                    reliability=0.70,
                ),
                provenance=("core.perception.sensory_integration", source),
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.SENSITIVE,
                    retention="none",
                    redacted=True,
                ),
                missing_reason=missing_reason,
                quality_flags=quality_flags,
            )
        )

    def _publish_to_synchronizer(self, perception: dict[str, Any]) -> None:
        modality = str(perception.get("modality") or "")
        data = perception.get("data")
        data = data if isinstance(data, dict) else {}
        interpretation = perception.get("interpretation")
        interpretation = interpretation if isinstance(interpretation, dict) else {}
        error = str(perception.get("error") or data.get("error") or "")
        missing = None
        if error:
            missing = (
                MissingReason.PERMISSION_DENIED
                if "permission" in error
                else MissingReason.UNAVAILABLE
                if "not_available" in error or "unavailable" in error
                else MissingReason.SENSOR_ERROR
            )

        if modality == SensoryModality.VISION.value:
            scene = str(interpretation.get("scene_description") or "")
            claims = (
                PerceptualClaim("camera.capture_available", not bool(error), 0.95),
                PerceptualClaim(
                    "vision.scene_digest",
                    hashlib.sha256(scene.encode("utf-8", errors="ignore")).hexdigest()[:24],
                    0.70,
                ),
                PerceptualClaim(
                    "scene.face_count",
                    max(0, int(interpretation.get("faces_detected", 0) or 0)),
                    0.65,
                ),
            )
            self._publish_event(
                modality=SynchronizedModality.VISION,
                source="sensory_system_camera",
                confidence=0.72,
                claims=claims,
                missing_reason=missing,
                quality_flags=("single_capture", "raw_image_not_retained"),
            )
        elif modality == SensoryModality.HEARING.value:
            self._publish_event(
                modality=SynchronizedModality.AUDIO,
                source="sensory_system_microphone",
                confidence=0.72,
                claims=(
                    PerceptualClaim("audio.capture_available", not bool(error), 0.95),
                    PerceptualClaim(
                        "audio.duration_s",
                        max(0.0, float(data.get("duration", 0.0) or 0.0)),
                        0.90,
                    ),
                ),
                missing_reason=missing,
                quality_flags=("bounded_capture", "raw_audio_not_retained"),
            )
            transcript = str(interpretation.get("text") or "").strip()
            if transcript and not error and "transcription failed" not in transcript.lower():
                self._publish_event(
                    modality=SynchronizedModality.SPEECH,
                    source="sensory_system_microphone:transcript",
                    confidence=max(
                        0.0,
                        min(1.0, float(interpretation.get("confidence", 0.0) or 0.0)),
                    ),
                    claims=(
                        PerceptualClaim("speech.transcript_available", True, 0.95),
                        PerceptualClaim(
                            "speech.transcript_digest",
                            hashlib.sha256(
                                transcript.encode("utf-8", errors="ignore")
                            ).hexdigest()[:24],
                            0.75,
                        ),
                    ),
                    quality_flags=("audio_transcript_not_visual_speech",),
                )
        elif modality == SensoryModality.TEXT.value:
            text = str(perception.get("data") or "")
            self._publish_event(
                modality=SynchronizedModality.TEXT,
                source="sensory_system_text",
                confidence=0.98 if text else 0.0,
                claims=(
                    PerceptualClaim("text.input_available", bool(text), 0.99),
                    PerceptualClaim(
                        "text.input_digest",
                        hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24],
                        0.95,
                    ),
                ),
                missing_reason=MissingReason.NOT_OBSERVED if not text else None,
                quality_flags=("raw_text_not_retained_in_fusion",),
            )


def _parse_vision_reading(raw: Any) -> dict[str, Any]:
    """Turn the model's answer into a reading that distinguishes unknown.

    Three outcomes must stay distinguishable, and before this they were not:

      * observed, and there is nothing there      -> 0 / []
      * observed, but cannot tell (dark, blurred,
        occluded, out of frame)                   -> None
      * never observed at all                     -> None + abstained

    Collapsing the middle case into zero is what makes a perception system
    confidently wrong: "no faces detected" in a dark room reads exactly like
    "the room is empty", and only one of those is a fact.
    """
    text = str(raw or "").strip()
    reading: dict[str, Any] = {
        "timestamp": time.time(),
        "analyzed": True,
        "abstained": False,
        "scene_description": text,
        "objects_detected": None,
        "text_detected": None,
        "faces_detected": None,
    }
    if not text:
        reading.update(
            analyzed=False,
            abstained=True,
            reason="empty_model_reply",
            scene_description="",
        )
        return reading

    # Models wrap JSON in prose and fences. Take the outermost object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        # Prose only. The description is still real and is kept; the
        # structured fields stay unknown rather than being invented as
        # empty, which is what the constants used to do.
        reading["reason"] = "unstructured_reply"
        return reading

    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        reading["reason"] = "unparseable_json"
        return reading
    if not isinstance(payload, dict):
        reading["reason"] = "unexpected_json_shape"
        return reading

    scene = payload.get("scene")
    if isinstance(scene, str) and scene.strip():
        reading["scene_description"] = scene.strip()

    def _as_list(value: Any) -> list[str] | None:
        # null means "cannot tell" and must survive as None.
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return None

    reading["objects_detected"] = _as_list(payload.get("objects"))
    reading["text_detected"] = _as_list(payload.get("text"))

    people = payload.get("people")
    if isinstance(people, bool):
        # `true` is not a count. Treating it as 1 would invent a number.
        reading["faces_detected"] = None
    elif isinstance(people, (int, float)):
        reading["faces_detected"] = max(0, int(people))
    elif isinstance(people, str) and people.strip().isdigit():
        reading["faces_detected"] = int(people.strip())
    else:
        reading["faces_detected"] = None

    return reading


class VisionSystem:
    """Camera and visual perception system.
    
    Enables Aura to:
    - Capture images/video from camera
    - Analyze visual content
    - Recognize objects, faces, text
    - Understand scenes
    """
    
    def __init__(self) -> None:
        self._camera_checked: bool = False
        self._camera_available: bool = False
        self.last_capture: dict[str, Any] | None = None

    @property
    def camera_available(self) -> bool:
        """Public access to camera availability."""
        return self._camera_available

    async def _get_camera_available(self) -> bool:
        """Probe hardware asynchronously."""
        if not self._camera_checked:
            self._camera_available = await asyncio.to_thread(self._check_camera)
            self._camera_checked = True
        return self._camera_available
        
    def _check_camera(self) -> bool:
        """Check if camera is available (Must be called in thread).

        Probing by opening the device was itself a defect: it contended for
        an exclusive resource just to answer a question, so a probe during
        an active stream reported "unavailable" and a probe that raced the
        stream could take the device from it. The authority already knows.
        """
        from core.perception.camera_authority import get_camera_authority

        state = get_camera_authority().state()
        # In use by another holder still means the hardware is there. This
        # answers "is there a camera we are allowed to use", which is what
        # every caller of this actually wanted.
        return bool(
            state["backend_available"]
            and state["owner_permission"]
            and state["os_permission"] is not False
        )
    
    async def capture(self, duration: float = 0, save_path: str | None = None) -> dict[str, Any]:
        """Capture from camera (Async)."""
        if not camera_allowed():
            logger.debug("📷🚫 Camera capture blocked: permissions.camera=False (user setting)")
            return {"error": "camera_permission_denied"}
        if not await self._get_camera_available():
            return {"error": "camera_not_available"}
        
        def _do_capture() -> dict[str, Any]:
            from core.perception.camera_authority import (
                CameraDenial,
                get_camera_authority,
            )

            authority = get_camera_authority()
            lease = authority.acquire(
                "sensory_integration",
                purpose=f"owner-requested capture ({'video' if duration else 'photo'})",
            )
            if isinstance(lease, CameraDenial):
                # The named reason and its remedy, not a bare
                # "camera_not_available" that told the owner nothing about
                # which of four possible causes it was.
                return {"error": lease.reason, **lease.to_dict()}

            try:
                if duration == 0:
                    selected = authority.capture_best_still(lease)
                    if selected is None:
                        return {"error": lease.last_error or "capture_failed"}
                    frame = selected.frame
                    encoded = selected.jpeg
                    if save_path:
                        # Through the gateway, not the raw atomic writer: a
                        # captured frame written to an owner-supplied path is
                        # a consequential write, and the gateway is where
                        # those are governed and accounted. Called from the
                        # synchronous _do_capture body, so the blocking
                        # variant is correct here — the async lane exists to
                        # keep fsync off the event loop, and this is already
                        # off it.
                        from core.runtime.file_write_gateway import get_file_write_gateway

                        get_file_write_gateway().write_bytes(
                            Path(save_path), encoded, source="sensory_integration.capture"
                        )
                    return {
                        "type": "image",
                        "data": base64.b64encode(encoded).decode('utf-8'),
                        "path": save_path,
                        "timestamp": time.time(),
                        "frame_quality": selected.quality.to_dict(),
                        "capture_selection": {
                            "selected_attempt": selected.attempt,
                            "attempts": selected.attempts,
                        },
                    }
                else:
                    path = save_path or f"capture_{int(time.time())}.mp4"
                    return _capture_video_to_governed_file(
                        authority,
                        lease,
                        path=path,
                        duration=duration,
                    )
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
                record_degradation('sensory_integration', e)
                return {"error": str(e)}
            finally:
                authority.release(lease)

        result = await asyncio.to_thread(_do_capture)
        if "error" not in result:
            self.last_capture = result
        return result
    
    async def analyze(self, capture_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze captured visual data via the cognitive engine."""
        if not capture_data or "error" in capture_data:
            return {"error": "invalid_capture"}

        # Try real vision analysis via screen_vision -> cognitive engine
        try:
            brain = optional_service("cognitive_engine")
            if brain is None:
                brain = optional_service("brain")

            if brain is not None and hasattr(brain, "think"):
                import base64
                image_path = capture_data.get("path")
                image_b64 = None

                if image_path:
                    import io

                    from PIL import Image
                    img = Image.open(image_path)
                    img = img.convert("RGB")
                    img.thumbnail((672, 672))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                elif capture_data.get("data") or capture_data.get("base64"):
                    image_b64 = str(capture_data.get("data") or capture_data.get("base64"))

                if image_b64:
                    # Ask for the structure, then parse it. The previous
                    # version asked for prose and returned
                    # `objects_detected: []`, `text_detected: []`,
                    # `faces_detected: 0` as CONSTANTS beside a real
                    # description — so a caller checking `faces_detected`
                    # read zero while the model was describing three people
                    # in the adjacent field.
                    description = await brain.think(
                        "Describe this image. Reply with JSON only:\n"
                        '{"scene": "<one or two sentences>", '
                        '"objects": ["..."], "text": ["..."], "people": <count>}\n'
                        "Use an empty list and 0 only when you are confident "
                        "none are present. If the image is too dark, blurred, "
                        'or occluded to tell, use null for that field.',
                        images=[image_b64],
                    )
                    reading = _parse_vision_reading(description)
                    # A model's confidence is not evidence about the
                    # physical conditions it was looking at. It will answer
                    # "two people at a desk" for a motion-blurred frame of a
                    # dim room in exactly the same tone it uses for a sharp
                    # one, so the counts get checked against what the pixels
                    # could actually carry.
                    measured = capture_data.get("frame_quality")
                    if isinstance(measured, dict) and measured.get("limits"):
                        from core.perception.frame_quality import (
                            FrameQuality,
                            temper_reading,
                        )

                        reading = temper_reading(
                            reading,
                            FrameQuality(
                                mean_luminance=float(measured.get("mean_luminance", 0.0)),
                                dark_fraction=float(measured.get("dark_fraction", 0.0)),
                                bright_fraction=float(measured.get("bright_fraction", 0.0)),
                                sharpness=float(measured.get("sharpness", 0.0)),
                                uniformity=float(measured.get("uniformity", 0.0)),
                                pixels=int(measured.get("pixels", 0)),
                                limits=tuple(measured.get("limits") or ()),
                            ),
                        )
                    return reading
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('sensory_integration', e)
            logger.debug("Vision analysis via brain failed: %s", e)

        # No vision model. This is an ABSTENTION, not an observation of an
        # empty room, and the two used to return identical structured
        # fields — zeros and empty lists either way. A caller could not tell
        # "I looked and saw nobody" from "I could not look".
        return {
            "timestamp": time.time(),
            "analyzed": False,
            "abstained": True,
            "reason": "no_vision_model",
            "objects_detected": None,
            "scene_description": "No vision model available for analysis.",
            "text_detected": None,
            "faces_detected": None,
        }


class HearingSystem:
    """Microphone and audio perception system.
    
    Enables Aura to:
    - Record audio from microphone
    - Transcribe speech to text
    - Understand tone and emotion
    - Detect sounds
    """
    
    def __init__(self) -> None:
        self._microphone_checked: bool = False
        self._microphone_available: bool = False
        self.last_recording: dict[str, Any] | None = None

    @property
    def microphone_available(self) -> bool:
        """Public access to microphone availability.

        Probes lazily the first time it is read. Without this, the boot-time
        summary logged "Microphone: unavailable" simply because it read the
        uninitialized default before any async probe ran — a false alarm that
        made a working mic look broken during voice triage.
        """
        if not self._microphone_checked:
            self._microphone_available = self._check_microphone()
            self._microphone_checked = True
        return self._microphone_available

    async def _get_microphone_available(self) -> bool:
        """Probe microphone hardware asynchronously."""
        if not self._microphone_checked:
            self._microphone_available = await asyncio.to_thread(self._check_microphone)
            self._microphone_checked = True
        return self._microphone_available
        
    def _check_microphone(self) -> bool:
        """Check if microphone is available (Must be called in thread)."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Look for any input device
            return any(d.get('max_input_channels', 0) > 0 for d in devices)
        except ImportError:
            logger.debug("sounddevice is unavailable; microphone capture disabled.")
            return False
        except (AttributeError, RuntimeError) as exc:
            record_degradation("sensory_integration", exc)
            logger.debug("Microphone availability probe failed: %s", exc)
            return False
    
    async def listen(self, duration: float = 5.0, save_path: str | None = None) -> dict[str, Any]:
        """Record audio from microphone (Async)."""
        if not await self._get_microphone_available():
            return {"error": "microphone_not_available"}
        
        def _do_listen() -> dict[str, Any]:
            try:
                import sounddevice as sd
                import soundfile as sf
                
                path = save_path or f"recording_{int(time.time())}.wav"
                channels, rate = 1, 44100
                
                # Record as numpy array
                recording = record_sounddevice_array(
                    sd,
                    holder="sensory_integration.hearing",
                    source="sensory_integration",
                    mode="snapshot",
                    frames=int(duration * rate),
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                )
                
                # Save using soundfile
                sf.write(path, recording, rate)
                
                return {"type": "audio", "path": path, "duration": duration, "timestamp": time.time()}
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('sensory_integration', e)
                return {"error": str(e)}

        result = await asyncio.to_thread(_do_listen)
        if "error" not in result:
            self.last_recording = result
        return result
    
    async def transcribe(self, audio_data: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio to text (Async)."""
        if not audio_data or "error" in audio_data:
            return {"error": "invalid_audio"}
        
        def _do_transcribe() -> dict[str, Any]:
            try:
                from core.senses.voice_socket_logic import get_whisper_model

                model = get_whisper_model("tiny")
                if model is None:
                    return {
                        "text": "",
                        "error": "canonical_stt_unavailable",
                        "disposition": "unavailable",
                        "backend": "canonical_whisper",
                    }
                segments, info = model.transcribe(audio_data["path"], beam_size=5)
                text = " ".join(str(segment.text).strip() for segment in segments).strip()
                if not text:
                    return {
                        "text": "",
                        "error": "no_speech_detected",
                        "disposition": "no_speech",
                        "backend": "canonical_whisper",
                    }
                language = str(getattr(info, "language", "") or "unknown")
                language_probability = getattr(info, "language_probability", None)
                if isinstance(language_probability, (int, float)):
                    language_probability = max(0.0, min(1.0, float(language_probability)))
                else:
                    language_probability = None
                return {
                    "text": text,
                    "confidence": None,
                    "language": language,
                    "language_confidence": language_probability,
                    "disposition": "transcribed",
                    "backend": "canonical_whisper",
                }
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                record_degradation('sensory_integration', e)
                return {
                    "text": "",
                    "error": "canonical_stt_failed",
                    "detail": str(e),
                    "disposition": "failed",
                    "backend": "canonical_whisper",
                }

        result = await asyncio.to_thread(_do_transcribe)
        result["timestamp"] = time.time()
        return result


class SpeechSystem:
    """Text-to-speech and voice synthesis system.
    
    Enables Aura to:
    - Speak text aloud
    - Use different voices/emotions
    - Control speech rate, pitch
    """
    
    def __init__(self) -> None:
        # Issue 42: Lazy-init engine and store here
        self._engine: Any = None
        self._lock = threading.Lock()
        self.tts_available = self._check_tts()
    def _check_tts(self) -> bool:
        """Check if TTS is available"""
        import importlib.util

        if importlib.util.find_spec("pyttsx3") is not None:
            return True
        else:
            logger.warning("pyttsx3 not installed - TTS unavailable")
            return False
    
    async def speak(self, text: str, rate: int = 150, volume: float = 1.0, save_path: str | None = None) -> dict[str, Any]:
        """Speak text aloud using TTS (Async)."""
        if not self.tts_available:
            return {"error": "tts_not_available", "success": False}

        # Apply the user's voice.output_rate multiplier (0.5-2.0, default 1.0) to
        # the base words-per-minute rate. (docs/SETTINGS_WIRING_AUDIT.md)
        try:
            multiplier = float(get_runtime_setting("voice.output_rate", 1.0) or 1.0)
        except (TypeError, ValueError):
            multiplier = 1.0
        rate = int(rate * max(0.5, min(2.0, multiplier)))

        def _do_speak() -> dict[str, Any]:
            try:
                # Issue 42: Lazy-init and reuse engine with lock
                with self._lock:
                    if self._engine is None:
                        import pyttsx3

                        self._engine = pyttsx3.init()
                    
                    engine = self._engine
                    engine.setProperty('rate', rate)
                    engine.setProperty('volume', volume)
                    if save_path:
                        engine.save_to_file(text, save_path)
                        engine.runAndWait()
                        return {"success": True, "audio_file": save_path}
                    else:
                        engine.say(text)
                        engine.runAndWait()
                        return {"success": True}
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('sensory_integration', e)
                return {"success": False, "error": str(e)}

        result = await asyncio.to_thread(_do_speak)
        result["timestamp"] = time.time()
        result["text"] = text
        return result


class AVProductionSystem:
    """Audio/visual production tools.
    
    Enables Aura to:
    - Edit audio/video
    - Create visual content
    - Generate images/animations
    - Mix audio
    """
    
    def __init__(self, output_dir: str | None = None) -> None:
        base_dir = Path(output_dir) if output_dir else state_root() / "data" / "media"
        self.output_dir = base_dir
        self.image_dir = base_dir / "generated_images"
        self.video_dir = base_dir / "edited_video"
        self.audio_dir = base_dir / "audio"
        for directory in (self.image_dir, self.video_dir, self.audio_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _slug(self, text: str, limit: int = 48) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return (slug or "aura_media")[:limit]

    def _render_local_image(self, description: str, style: str) -> dict[str, Any]:
        """Create a deterministic local visual artifact when no image model is loaded."""
        import hashlib
        import textwrap

        digest = hashlib.sha256(f"{style}:{description}".encode()).hexdigest()
        path = self.image_dir / f"{int(time.time())}_{self._slug(description)}.png"
        try:
            from PIL import Image, ImageDraw, ImageFont

            width, height = 1024, 1024
            bg = tuple(int(digest[i:i + 2], 16) for i in (0, 2, 4))
            accent = tuple(255 - channel for channel in bg)
            img = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(img)
            for i in range(0, height, 32):
                shade = tuple(max(0, min(255, channel + (i // 32) % 28)) for channel in bg)
                draw.line((0, i, width, i), fill=shade)
            draw.rectangle((72, 72, width - 72, height - 72), outline=accent, width=6)
            font = ImageFont.load_default()
            lines = textwrap.wrap(description, width=52)[:14]
            y = 150
            draw.text((120, 105), f"Aura local image: {style}", fill=accent, font=font)
            for line in lines:
                draw.text((120, y), line, fill=(245, 245, 245), font=font)
                y += 42
            img.save(path)
            return {
                "path": str(path),
                "description": description,
                "style": style,
                "source": "local_renderer",
                "timestamp": time.time(),
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("sensory_integration", exc)
            manifest = path.with_suffix(".txt")
            atomic_write_text(
                manifest,
                f"Aura local image request\nstyle: {style}\ndescription: {description}\n",
                encoding="utf-8",
            )
            return {
                "path": str(manifest),
                "description": description,
                "style": style,
                "source": "manifest_fallback",
                "warning": str(exc),
                "timestamp": time.time(),
            }
    
    async def create_image(self, description: str, style: str = "realistic") -> dict[str, Any]:
        """Generate image via local Stable Diffusion or brain inference."""
        try:
            brain = optional_service("cognitive_engine")
            if brain and hasattr(brain, "generate_image"):
                result = await brain.generate_image(description, style=style)
                if result:
                    return {"path": result, "description": description, "timestamp": time.time()}
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('sensory_integration', e)
            logger.debug("Image generation via brain failed: %s", e)

        return await asyncio.to_thread(self._render_local_image, description, style)

    async def edit_video(self, video_path: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply edits to video via FFmpeg."""
        import shutil
        if not shutil.which("ffmpeg"):
            return {"error": "ffmpeg_not_installed"}
        try:
            # Basic trim operation as baseline
            for edit in edits:
                if edit.get("type") == "trim":
                    start = edit.get("start", 0)
                    end = edit.get("end")
                    output = video_path.rsplit(".", 1)[0] + "_edited.mp4"
                    cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(start)]
                    if end:
                        cmd.extend(["-to", str(end)])
                    cmd.extend(["-c", "copy", output])
                    proc = await get_subprocess_gateway().run_async(
                        cmd,
                        capture_output=True,
                        timeout=60,
                        source="core.perception.sensory_integration.edit_video",
                        accelerator_capability="auto",
                    )
                    if proc.returncode == 0:
                        return {"path": output, "edits_applied": len(edits)}
                    return {"error": proc.stderr[:200]}
            return {"error": "no_supported_edits", "supported": ["trim"]}
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('sensory_integration', e)
            return {"error": str(e)}


_sensory_lock = threading.Lock()

def get_sensory_system() -> SensorySystem:
    """Get global sensory system via DI container"""
    try:
        with _sensory_lock:
            if not ServiceContainer.get("sensory_system", None):
                ServiceContainer.register(
                    "sensory_system",
                    factory=lambda: SensorySystem(),
                    lifetime=ServiceLifetime.SINGLETON
                )
            res = optional_service("sensory_system")
            if isinstance(res, SensorySystem):
                return res
            return SensorySystem()
    except (ImportError, AttributeError, RuntimeError) as e:
        record_degradation('sensory_integration', e)
        logger.debug("ServiceContainer unavailable or failed: %s. Using transient SensorySystem.", e)
        return SensorySystem()


# Integration helpers
def integrate_sensory_system(orchestrator: Any) -> None:
    """Integrate sensory system into orchestrator.
    
    Adds sensory perception as available actions.
    """
    sensory = get_sensory_system()
    
    # Store reference
    orchestrator.sensory_system = sensory
    
    logger.info("✓ Sensory system integrated")
    logger.info("  Camera: %s", 'available' if sensory.vision.camera_available else 'unavailable')
    logger.info("  Microphone: %s", 'available' if sensory.hearing.microphone_available else 'unavailable')
    logger.info("  TTS: %s", 'available' if sensory.speech.tts_available else 'unavailable')
