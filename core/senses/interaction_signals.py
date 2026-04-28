from __future__ import annotations
from core.runtime.errors import record_degradation

from core.utils.task_tracker import get_task_tracker

import asyncio
import base64
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from core.utils.queues import BackpressuredQueue

logger = logging.getLogger("Aura.Senses.InteractionSignals")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _freshness_weight(updated_at: float, half_life_s: float) -> float:
    if updated_at <= 0.0 or half_life_s <= 0.0:
        return 0.0
    age = max(0.0, time.time() - updated_at)
    if age <= 0.0:
        return 1.0
    return float(math.exp((-math.log(2.0) * age) / half_life_s))


@dataclass
class TypingSignalState:
    updated_at: float = 0.0
    active: bool = False
    chars_per_minute: float = 0.0
    burstiness: float = 0.0
    hesitation: float = 0.0
    correction_rate: float = 0.0
    pause_before_submit_ms: float = 0.0
    session_ms: float = 0.0
    message_chars: int = 0
    label: str = "idle"


@dataclass
class VoiceSignalState:
    updated_at: float = 0.0
    speech_ratio: float = 0.0
    rms: float = 0.0
    peak: float = 0.0
    zcr: float = 0.0
    clipping_ratio: float = 0.0
    activation: float = 0.0
    steadiness: float = 0.5
    stress_cue: float = 0.0
    label: str = "quiet"


@dataclass
class VisionSignalState:
    updated_at: float = 0.0
    face_present: bool = False
    face_count: int = 0
    face_area_ratio: float = 0.0
    gaze_direction: str = "unknown"
    head_pose: str = "unknown"
    attention_available: float = 0.0
    eyes_detected: int = 0


@dataclass
class FusedInteractionState:
    updated_at: float = 0.0
    activation: float = 0.0
    steadiness: float = 0.5
    engagement: float = 0.0
    hesitation: float = 0.0
    attention_available: float = 0.5
    question_pressure: float = 0.5
    verbosity_bias: str = "balanced"
    pacing: str = "steady"
    summary: str = "No live human-signal cues yet."
    active_modalities: list[str] = field(default_factory=list)


class InteractionSignalsEngine:
    """Fuse real, testable user interaction signals into a bounded live state."""

    _TYPING_HALF_LIFE_S = 8.0
    _VOICE_HALF_LIFE_S = 5.0
    _VISION_HALF_LIFE_S = 6.0

    def __init__(self) -> None:
        self._typing_queue = BackpressuredQueue(maxsize=96)
        self._voice_queue = BackpressuredQueue(maxsize=128)
        self._vision_queue = BackpressuredQueue(maxsize=24)
        self._tasks: list[asyncio.Task[Any]] = []
        self._start_lock = asyncio.Lock()
        self._started = False

        self._typing = TypingSignalState()
        self._voice = VoiceSignalState()
        self._vision = VisionSignalState()
        self._fused = FusedInteractionState()

        self._face_cascade = None
        self._eye_cascade = None
        self._vision_backend_ready = False
        self._vision_backend_reason = ""

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self._tasks = [
                get_task_tracker().create_task(self._typing_consumer(), name="interaction_signals.typing"),
                get_task_tracker().create_task(self._voice_consumer(), name="interaction_signals.voice"),
                get_task_tracker().create_task(self._vision_consumer(), name="interaction_signals.vision"),
            ]
            self._started = True

    async def stop(self) -> None:
        self._started = False
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    async def publish_typing(self, payload: Dict[str, Any]) -> None:
        await self.ensure_started()
        await self._typing_queue.put(dict(payload), timeout=0.1)

    async def publish_voice(self, payload: Dict[str, Any]) -> None:
        await self.ensure_started()
        await self._voice_queue.put(dict(payload), timeout=0.1)

    async def publish_vision_frame(self, jpeg_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self.ensure_started()
        await self._vision_queue.put(
            {
                "jpeg_bytes": bytes(jpeg_bytes),
                "metadata": dict(metadata or {}),
                "timestamp": time.time(),
            },
            timeout=0.1,
        )

    def get_status(self) -> Dict[str, Any]:
        fused = self._compute_fused_state()
        return {
            "typing": asdict(self._typing),
            "voice": asdict(self._voice),
            "vision": asdict(self._vision),
            "fused": asdict(fused),
            "queues": {
                "typing_depth": self._typing_queue.qsize(),
                "voice_depth": self._voice_queue.qsize(),
                "vision_depth": self._vision_queue.qsize(),
            },
            "vision_backend": {
                "ready": self._vision_backend_ready,
                "reason": self._vision_backend_reason,
            },
        }

    def get_prompt_guidance(self) -> str:
        fused = self._compute_fused_state()
        if not fused.active_modalities:
            return ""

        voice_label = self._voice.label if _freshness_weight(self._voice.updated_at, self._VOICE_HALF_LIFE_S) > 0.15 else "unknown"
        typing_label = self._typing.label if _freshness_weight(self._typing.updated_at, self._TYPING_HALF_LIFE_S) > 0.15 else "unknown"
        vision_bits = []
        if _freshness_weight(self._vision.updated_at, self._VISION_HALF_LIFE_S) > 0.15:
            if self._vision.face_present:
                vision_bits.append(f"face present, gaze={self._vision.gaze_direction}, head={self._vision.head_pose}")
            else:
                vision_bits.append("no face currently visible")

        return "\n".join(
            [
                "## LIVE HUMAN SIGNALS (observed cues, not certainty)",
                f"- Engagement: {fused.summary}",
                f"- Typing: {typing_label}. Voice: {voice_label}.",
                f"- Attention available: {fused.attention_available:.2f}. Hesitation: {fused.hesitation:.2f}.",
                f"- Response shaping: keep pacing {fused.pacing}, keep verbosity {fused.verbosity_bias}, and let question pressure track {fused.question_pressure:.2f}.",
                *(f"- Camera cue: {', '.join(vision_bits)}." for _ in [0] if vision_bits),
                "- Use these cues to shape tone and pacing. Do not claim certainty about the user's inner emotions unless they say them directly.",
            ]
        )

    async def _typing_consumer(self) -> None:
        while True:
            payload = await self._typing_queue.get()
            try:
                self._typing = self._update_typing_state(payload)
                self._fused = self._compute_fused_state()
            except Exception as exc:
                record_degradation('interaction_signals', exc)
                logger.debug("Typing signal update failed: %s", exc)
            finally:
                self._typing_queue.task_done()

    async def _voice_consumer(self) -> None:
        while True:
            payload = await self._voice_queue.get()
            try:
                self._voice = self._update_voice_state(payload)
                self._fused = self._compute_fused_state()
            except Exception as exc:
                record_degradation('interaction_signals', exc)
                logger.debug("Voice signal update failed: %s", exc)
            finally:
                self._voice_queue.task_done()

    async def _vision_consumer(self) -> None:
        while True:
            payload = await self._vision_queue.get()
            try:
                jpeg_bytes = bytes(payload.get("jpeg_bytes") or b"")
                metadata = dict(payload.get("metadata") or {})
                analysis = await asyncio.to_thread(self._analyze_vision_frame_sync, jpeg_bytes, metadata)
                self._vision = self._update_vision_state(analysis)
                self._fused = self._compute_fused_state()
            except Exception as exc:
                record_degradation('interaction_signals', exc)
                logger.debug("Vision signal update failed: %s", exc)
            finally:
                self._vision_queue.task_done()

    def _update_typing_state(self, payload: Dict[str, Any]) -> TypingSignalState:
        updated_at = _safe_float(payload.get("timestamp"), time.time())
        session_ms = max(1.0, _safe_float(payload.get("session_ms"), 0.0))
        message_chars = max(0, int(_safe_float(payload.get("message_chars"), 0.0)))
        key_count = max(1.0, _safe_float(payload.get("key_count"), message_chars))
        correction_count = max(0.0, _safe_float(payload.get("correction_count"), 0.0))
        pause_before_submit_ms = max(0.0, _safe_float(payload.get("pause_before_submit_ms"), 0.0))
        max_pause_ms = max(0.0, _safe_float(payload.get("max_pause_ms"), 0.0))
        is_submitted = bool(payload.get("submitted", False))
        active = bool(payload.get("active", False)) and not is_submitted

        chars_per_minute = (message_chars / session_ms) * 60000.0 if session_ms > 0 else 0.0
        correction_rate = correction_count / max(1.0, key_count)
        pace_pressure = _clamp01((chars_per_minute - 70.0) / 250.0)
        pause_pressure = _clamp01(max(pause_before_submit_ms, max_pause_ms) / 2400.0)
        hesitation = _clamp01((pause_pressure * 0.62) + (correction_rate * 1.4 * 0.38))
        if is_submitted and pause_before_submit_ms > 1200.0 and message_chars < 48:
            hesitation = _clamp01(hesitation + 0.1)

        if active and pace_pressure > 0.72:
            label = "rapid"
        elif hesitation > 0.7:
            label = "hesitant"
        elif chars_per_minute < 85.0 and message_chars > 0:
            label = "considered"
        elif active:
            label = "flowing"
        else:
            label = "idle"

        return TypingSignalState(
            updated_at=updated_at,
            active=active,
            chars_per_minute=round(chars_per_minute, 2),
            burstiness=round(pace_pressure, 4),
            hesitation=round(hesitation, 4),
            correction_rate=round(correction_rate, 4),
            pause_before_submit_ms=round(pause_before_submit_ms, 1),
            session_ms=round(session_ms, 1),
            message_chars=message_chars,
            label=label,
        )

    def _update_voice_state(self, payload: Dict[str, Any]) -> VoiceSignalState:
        updated_at = _safe_float(payload.get("timestamp"), time.time())
        speech_ratio = _clamp01(_safe_float(payload.get("speech_ratio"), 0.0))
        rms = max(0.0, _safe_float(payload.get("rms_avg"), 0.0))
        peak = max(0.0, _safe_float(payload.get("peak_avg"), 0.0))
        zcr = _clamp01(_safe_float(payload.get("zcr_avg"), 0.0))
        clipping_ratio = _clamp01(_safe_float(payload.get("clipping_ratio"), 0.0))
        rms_std = max(0.0, _safe_float(payload.get("rms_std"), 0.0))

        loudness = _clamp01((rms * 14.0) + (peak * 0.8))
        activation = _clamp01((loudness * 0.58) + (speech_ratio * 0.42))
        steadiness = _clamp01(
            1.0
            - min(
                1.0,
                (zcr * 0.9)
                + (clipping_ratio * 1.4)
                + (rms_std * 10.0 * 0.35),
            )
        )
        stress_cue = _clamp01((activation * 0.55) + ((1.0 - steadiness) * 0.45))

        if speech_ratio < 0.08:
            label = "quiet"
        elif stress_cue > 0.72 and activation > 0.42:
            label = "stressed"
        elif activation > 0.72:
            label = "activated"
        elif activation < 0.36 and steadiness > 0.64:
            label = "calm"
        else:
            label = "steady"

        return VoiceSignalState(
            updated_at=updated_at,
            speech_ratio=round(speech_ratio, 4),
            rms=round(rms, 5),
            peak=round(peak, 5),
            zcr=round(zcr, 4),
            clipping_ratio=round(clipping_ratio, 4),
            activation=round(activation, 4),
            steadiness=round(steadiness, 4),
            stress_cue=round(stress_cue, 4),
            label=label,
        )

    def _update_vision_state(self, payload: Dict[str, Any]) -> VisionSignalState:
        return VisionSignalState(
            updated_at=_safe_float(payload.get("updated_at"), time.time()),
            face_present=bool(payload.get("face_present", False)),
            face_count=max(0, int(_safe_float(payload.get("face_count"), 0.0))),
            face_area_ratio=round(_clamp01(_safe_float(payload.get("face_area_ratio"), 0.0)), 4),
            gaze_direction=str(payload.get("gaze_direction") or "unknown"),
            head_pose=str(payload.get("head_pose") or "unknown"),
            attention_available=round(_clamp01(_safe_float(payload.get("attention_available"), 0.0)), 4),
            eyes_detected=max(0, int(_safe_float(payload.get("eyes_detected"), 0.0))),
        )

    def _compute_fused_state(self) -> FusedInteractionState:
        now = time.time()
        typing_weight = _freshness_weight(self._typing.updated_at, self._TYPING_HALF_LIFE_S)
        voice_weight = _freshness_weight(self._voice.updated_at, self._VOICE_HALF_LIFE_S)
        vision_weight = _freshness_weight(self._vision.updated_at, self._VISION_HALF_LIFE_S)

        active_modalities: list[str] = []
        total_weight = 0.0
        activation_sum = 0.0
        steadiness_sum = 0.0
        engagement_sum = 0.0

        if typing_weight > 0.15:
            active_modalities.append("typing")
            total_weight += typing_weight
            activation_sum += typing_weight * max(self._typing.burstiness, 0.12 if self._typing.active else 0.0)
            steadiness_sum += typing_weight * max(0.2, 1.0 - (self._typing.hesitation * 0.75))
            engagement_sum += typing_weight * max(self._typing.burstiness, 0.25 if self._typing.message_chars else 0.0)

        if voice_weight > 0.15:
            active_modalities.append("voice")
            total_weight += voice_weight
            activation_sum += voice_weight * self._voice.activation
            steadiness_sum += voice_weight * self._voice.steadiness
            engagement_sum += voice_weight * max(self._voice.speech_ratio, self._voice.activation * 0.6)

        if vision_weight > 0.15:
            active_modalities.append("vision")
            total_weight += vision_weight
            activation_sum += vision_weight * (0.18 if self._vision.face_present else 0.0)
            steadiness_sum += vision_weight * max(0.15, self._vision.attention_available)
            engagement_sum += vision_weight * self._vision.attention_available

        activation = activation_sum / total_weight if total_weight > 0 else 0.0
        steadiness = steadiness_sum / total_weight if total_weight > 0 else 0.5
        engagement = engagement_sum / total_weight if total_weight > 0 else 0.0

        hesitation = max(
            self._typing.hesitation * typing_weight,
            self._voice.stress_cue * voice_weight * 0.8,
        )
        hesitation = _clamp01(hesitation)

        if vision_weight > 0.15:
            attention_available = self._vision.attention_available
        else:
            attention_available = _clamp01(0.42 + (typing_weight * 0.18) + (voice_weight * 0.14) - (hesitation * 0.25))

        question_pressure = _clamp01(0.62 + (engagement * 0.18) - (hesitation * 0.52) - ((1.0 - attention_available) * 0.18))

        if hesitation > 0.7 or attention_available < 0.34:
            verbosity_bias = "concise"
        elif engagement > 0.72 and attention_available > 0.62:
            verbosity_bias = "expansive"
        else:
            verbosity_bias = "balanced"

        if hesitation > 0.7 or self._voice.label == "stressed":
            pacing = "gentle"
        elif activation > 0.72 and engagement > 0.6:
            pacing = "energetic"
        else:
            pacing = "steady"

        summary_bits: list[str] = []
        if self._typing.active:
            summary_bits.append(f"typing feels {self._typing.label}")
        elif typing_weight > 0.15 and self._typing.message_chars:
            summary_bits.append(f"recent typing felt {self._typing.label}")
        if voice_weight > 0.15 and self._voice.label != "quiet":
            summary_bits.append(f"voice sounds {self._voice.label}")
        if vision_weight > 0.15:
            if self._vision.face_present:
                summary_bits.append(f"camera shows attention {self._vision.gaze_direction}")
            else:
                summary_bits.append("camera does not currently see a face")
        summary = ", ".join(summary_bits) if summary_bits else "No live human-signal cues yet."

        return FusedInteractionState(
            updated_at=now,
            activation=round(_clamp01(activation), 4),
            steadiness=round(_clamp01(steadiness), 4),
            engagement=round(_clamp01(engagement), 4),
            hesitation=round(hesitation, 4),
            attention_available=round(_clamp01(attention_available), 4),
            question_pressure=round(question_pressure, 4),
            verbosity_bias=verbosity_bias,
            pacing=pacing,
            summary=summary,
            active_modalities=active_modalities,
        )

    def _ensure_vision_backend(self) -> bool:
        if self._face_cascade is not None and self._eye_cascade is not None:
            self._vision_backend_ready = True
            return True

        try:
            import cv2

            face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
            face_cascade = cv2.CascadeClassifier(face_path)
            eye_cascade = cv2.CascadeClassifier(eye_path)
            if face_cascade.empty() or eye_cascade.empty():
                self._vision_backend_reason = "opencv_cascade_load_failed"
                self._vision_backend_ready = False
                return False
            self._face_cascade = face_cascade
            self._eye_cascade = eye_cascade
            self._vision_backend_ready = True
            self._vision_backend_reason = ""
            return True
        except Exception as exc:
            record_degradation('interaction_signals', exc)
            self._vision_backend_ready = False
            self._vision_backend_reason = str(exc)
            return False

    def _analyze_vision_frame_sync(self, jpeg_bytes: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not jpeg_bytes:
            return {"updated_at": time.time()}
        if not self._ensure_vision_backend():
            return {"updated_at": time.time()}

        try:
            import cv2
            import numpy as np

            frame_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                return {"updated_at": time.time()}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.18,
                minNeighbors=5,
                minSize=(56, 56),
            )
            if len(faces) == 0:
                return {
                    "updated_at": time.time(),
                    "face_present": False,
                    "face_count": 0,
                    "attention_available": 0.0,
                    "gaze_direction": "absent",
                    "head_pose": "absent",
                }

            x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
            face_roi = gray[y : y + h, x : x + w]
            upper_face = face_roi[: max(1, int(h * 0.62)), :]
            eyes = self._eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(12, 12),
            )
            eyes = sorted(eyes, key=lambda item: item[2] * item[3], reverse=True)[:2]

            frame_h, frame_w = gray.shape[:2]
            face_center_x = (x + (w / 2.0)) / max(1.0, float(frame_w))
            face_center_y = (y + (h / 2.0)) / max(1.0, float(frame_h))
            yaw_hint = (face_center_x - 0.5) * 2.0
            pitch_hint = (face_center_y - 0.42) * 2.0

            if yaw_hint < -0.2:
                head_pose = "left"
            elif yaw_hint > 0.2:
                head_pose = "right"
            elif pitch_hint < -0.16:
                head_pose = "up"
            elif pitch_hint > 0.2:
                head_pose = "down"
            else:
                head_pose = "center"

            gaze_direction = "unknown"
            pupil_positions: list[tuple[float, float]] = []
            for ex, ey, ew, eh in eyes:
                eye_roi = upper_face[ey : ey + eh, ex : ex + ew]
                if eye_roi.size == 0:
                    continue
                eye_blur = cv2.GaussianBlur(eye_roi, (5, 5), 0)
                threshold = int(max(18, min(120, float(np.percentile(eye_blur, 28)))))
                pupil_mask = eye_blur < threshold
                coords = np.column_stack(np.where(pupil_mask))
                if coords.size == 0:
                    continue
                center_y, center_x = coords.mean(axis=0)
                pupil_positions.append((float(center_x) / max(1.0, float(ew)), float(center_y) / max(1.0, float(eh))))

            if pupil_positions:
                avg_x = sum(pos[0] for pos in pupil_positions) / len(pupil_positions)
                avg_y = sum(pos[1] for pos in pupil_positions) / len(pupil_positions)
                if avg_x < 0.4:
                    gaze_direction = "left"
                elif avg_x > 0.6:
                    gaze_direction = "right"
                elif avg_y < 0.38:
                    gaze_direction = "up"
                elif avg_y > 0.66:
                    gaze_direction = "down"
                else:
                    gaze_direction = "center"
            elif head_pose == "center":
                gaze_direction = "center"

            face_area_ratio = (w * h) / max(1.0, float(frame_w * frame_h))
            attention_available = 0.28
            if gaze_direction == "center":
                attention_available += 0.32
            if head_pose == "center":
                attention_available += 0.2
            if len(eyes) >= 2:
                attention_available += 0.1
            attention_available += min(0.2, face_area_ratio * 3.5)
            if not pupil_positions and gaze_direction == "unknown":
                attention_available -= 0.12

            return {
                "updated_at": time.time(),
                "face_present": True,
                "face_count": int(len(faces)),
                "face_area_ratio": round(_clamp01(face_area_ratio), 4),
                "gaze_direction": gaze_direction,
                "head_pose": head_pose,
                "attention_available": round(_clamp01(attention_available), 4),
                "eyes_detected": int(len(eyes)),
                "width": int(metadata.get("width") or frame_w),
                "height": int(metadata.get("height") or frame_h),
            }
        except Exception as exc:
            record_degradation('interaction_signals', exc)
            logger.debug("Vision frame analysis failed: %s", exc)
            return {"updated_at": time.time()}


def decode_data_url_image(data_url: str) -> bytes:
    payload = str(data_url or "").strip()
    if "," in payload:
        _prefix, payload = payload.split(",", 1)
    return base64.b64decode(payload, validate=True)
