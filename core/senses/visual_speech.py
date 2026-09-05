"""core/senses/visual_speech.py
────────────────────────────
Visual speech perception (the lip-reading channel), honestly scoped.

What this genuinely does today:
- tracks a mouth region from camera frames (face detection + geometric
  mouth ROI, no model downloads);
- measures syllabic-band mouth-motion energy and converts it into a
  calibrated speaking-probability with hysteresis — real visual
  speech-activity detection, suitable for "Bryan is talking to me even
  though the mic missed it" and audio-visual disambiguation;
- extracts coarse viseme features (mouth openness / width / motion)
  per frame — features, not words.

What it deliberately does NOT claim: word-level lip reading. That
requires a trained visual-speech-recognition model. The pipeline has a
governed ONNX seam (``attach_vsr_model``) that accepts a user-supplied
model file; until one is attached, ``transcript`` stays ``None`` and
confidence reporting says why. No silent fabrication.

cv2 policy: imports honor the main-process camera policy via
core.media.safe_imports — in the main runtime process the detector
falls back to injected detectors (the sidecar owns real capture).
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Senses.VisualSpeech")

_VISUAL_SPEECH_ERRORS = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)

# Syllable-rate band for human speech articulation, in Hz.
_SPEECH_BAND_LOW = 1.5
_SPEECH_BAND_HIGH = 8.0
_DEFAULT_FPS = 15.0
_ACTIVITY_WINDOW_FRAMES = 30
# Logistic calibration for motion-energy → speaking probability.
_CALIBRATION_MIDPOINT = 0.035
_CALIBRATION_STEEPNESS = 120.0
# Landmark aperture deltas are smaller and cleaner than intensity motion.
_LANDMARK_MIDPOINT = 0.006
_LANDMARK_STEEPNESS = 700.0
_SPEAK_ON_THRESHOLD = 0.65
_SPEAK_OFF_THRESHOLD = 0.35


@dataclass
class MouthRegion:
    x: int
    y: int
    width: int
    height: int

    def clamp(self, frame_height: int, frame_width: int) -> MouthRegion:
        x = max(0, min(self.x, frame_width - 2))
        y = max(0, min(self.y, frame_height - 2))
        return MouthRegion(
            x=x,
            y=y,
            width=max(2, min(self.width, frame_width - x)),
            height=max(2, min(self.height, frame_height - y)),
        )


@dataclass
class VisualSpeechObservation:
    at: float
    face_present: bool
    mouth_region: MouthRegion | None
    motion_energy: float
    speaking_probability: float
    speaking: bool
    viseme_features: list[float] = field(default_factory=list)
    transcript: str | None = None
    transcript_source: str = "unavailable_no_vsr_model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "face_present": self.face_present,
            "motion_energy": round(self.motion_energy, 6),
            "speaking_probability": round(self.speaking_probability, 4),
            "speaking": self.speaking,
            "viseme_features": [round(f, 5) for f in self.viseme_features],
            "transcript": self.transcript,
            "transcript_source": self.transcript_source,
        }


def _default_face_detector() -> Callable[[np.ndarray], tuple[int, int, int, int] | None] | None:
    """Haar face detector from cv2's bundled cascades (no downloads).
    Returns None when cv2 is policy-blocked or unavailable."""
    try:
        from core.media.safe_imports import cv2_main_process_blocked

        if cv2_main_process_blocked():
            return None
        import cv2

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return None

        def detect(gray: np.ndarray) -> tuple[int, int, int, int] | None:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                return None
            # Largest face wins, deterministically.
            x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            return int(x), int(y), int(w), int(h)

        return detect
    except _VISUAL_SPEECH_ERRORS as exc:
        record_degradation("senses.visual_speech.detector", exc)
        return None


# ── Landmark-grade lip tracking (mediapipe FaceMesh, bundled model) ──

# FaceMesh indices: inner lips 13 (upper) / 14 (lower), mouth corners
# 61 / 291, face-height reference 10 (forehead) / 152 (chin).
_LIP_UPPER, _LIP_LOWER = 13, 14
_MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
_FACE_TOP, _FACE_BOTTOM = 10, 152


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def lip_metrics_from_points(points: dict[int, tuple[float, float]]) -> dict[str, float] | None:
    """Normalized lip geometry from landmark points (pixel or unit space).
    Aperture and width are scaled by face height, so they are camera- and
    distance-invariant — the viseme-grade signal."""
    required = (_LIP_UPPER, _LIP_LOWER, _MOUTH_LEFT, _MOUTH_RIGHT,
                _FACE_TOP, _FACE_BOTTOM)
    if any(index not in points for index in required):
        return None
    face_height = math.dist(points[_FACE_TOP], points[_FACE_BOTTOM])
    if face_height <= 1e-9:
        return None
    return {
        "aperture": math.dist(points[_LIP_UPPER], points[_LIP_LOWER]) / face_height,
        "width": math.dist(points[_MOUTH_LEFT], points[_MOUTH_RIGHT]) / face_height,
    }


class LandmarkLipTracker:
    """MediaPipe FaceMesh lip tracker. The mesh model ships inside the
    wheel — no downloads. Fails soft to None (caller falls back)."""

    def __init__(self) -> None:
        import mediapipe

        self._mesh = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def lip_metrics(self, frame: np.ndarray) -> dict[str, float] | None:
        try:
            rgb = np.ascontiguousarray(frame[..., ::-1]) if frame.ndim == 3 else (
                np.repeat(frame[..., None], 3, axis=2))
            result = self._mesh.process(rgb)
            faces = getattr(result, "multi_face_landmarks", None)
            if not faces:
                return None
            landmarks = faces[0].landmark
            points = {
                index: (landmarks[index].x, landmarks[index].y)
                for index in (_LIP_UPPER, _LIP_LOWER, _MOUTH_LEFT,
                              _MOUTH_RIGHT, _FACE_TOP, _FACE_BOTTOM)
            }
            return lip_metrics_from_points(points)
        except _VISUAL_SPEECH_ERRORS as exc:
            record_degradation("senses.visual_speech.landmarks", exc)
            return None


def mouth_roi_from_face(face: tuple[int, int, int, int]) -> MouthRegion:
    """Geometric mouth region: lower third of the face, central half width."""
    x, y, w, h = face
    return MouthRegion(
        x=x + w // 4,
        y=y + (2 * h) // 3,
        width=w // 2,
        height=h // 3,
    )


class VisualSpeechPipeline:
    """Frame-in, observation-out visual speech perception."""

    def __init__(
        self,
        *,
        fps: float = _DEFAULT_FPS,
        face_detector: Callable[[np.ndarray], tuple[int, int, int, int] | None] | None = None,
        use_landmarks: bool = True,
        lip_tracker: Any = None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self._face_detector = face_detector if face_detector is not None else _default_face_detector()
        self._lip_tracker: Any = None
        if lip_tracker is not None:
            self._lip_tracker = lip_tracker
        elif use_landmarks and mediapipe_available():
            try:
                self._lip_tracker = LandmarkLipTracker()
            except _VISUAL_SPEECH_ERRORS as exc:
                record_degradation("senses.visual_speech.landmark_init", exc)
        self._previous_aperture: float | None = None
        self._previous_mouth: np.ndarray | None = None
        from core.senses.viseme_decoder import LipReadResult, VisemeDecoder

        self._viseme_decoder = VisemeDecoder()
        self.last_lip_read: LipReadResult | None = None
        self._energy_window: deque[float] = deque(maxlen=_ACTIVITY_WINDOW_FRAMES)
        self._speaking = False
        self._vsr_session = None
        self._vsr_path: str | None = None
        self.observations = 0

    @property
    def detector_available(self) -> bool:
        return self._face_detector is not None

    @property
    def vsr_model_attached(self) -> bool:
        return self._vsr_session is not None

    # ── model seam (no downloads; user-supplied file only) ─────

    def attach_vsr_model(self, model_path: str | Path) -> dict[str, Any]:
        """Attach a user-supplied ONNX visual-speech-recognition model.
        Until this succeeds, transcripts remain None by design."""
        path = Path(model_path)
        if not path.exists() or path.suffix.lower() != ".onnx":
            return {
                "ok": False,
                "error": f"VSR model must be an existing .onnx file (got {path})",
            }
        try:
            import onnxruntime

            self._vsr_session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._vsr_path = str(path)
            logger.info("Visual speech: attached VSR model %s", path.name)
            return {"ok": True, "model": path.name}
        # onnxruntime raises pybind11 exception types that subclass
        # Exception directly (InvalidProtobuf, Fail, …) — a malformed
        # model file must degrade cleanly, never crash perception.
        except Exception as exc:  # noqa: BLE001
            record_degradation("senses.visual_speech.vsr_attach", exc)
            self._vsr_session = None
            return {"ok": False, "error": f"Could not load VSR model: {exc}"}

    # ── per-frame processing ───────────────────────────────────

    def process_frame(self, frame: np.ndarray, *, at: float | None = None) -> VisualSpeechObservation:
        """Consume one frame (grayscale or BGR uint8) and update state."""
        at = time.time() if at is None else float(at)
        if self._lip_tracker is not None:
            landmark_obs = self._process_landmark_frame(frame, at)
            if landmark_obs is not None:
                return landmark_obs
            # Landmark path found no face this frame; fall through to the
            # detector path so behavior degrades, never gaps.
        gray = self._to_gray(frame)
        face = self._face_detector(gray) if self._face_detector else None
        if face is None:
            self._previous_mouth = None
            self._energy_window.append(0.0)
            self._speaking = False
            self.observations += 1
            return VisualSpeechObservation(
                at=at, face_present=False, mouth_region=None,
                motion_energy=0.0, speaking_probability=0.0, speaking=False,
            )

        region = mouth_roi_from_face(face).clamp(*gray.shape[:2])
        mouth = gray[
            region.y: region.y + region.height,
            region.x: region.x + region.width,
        ].astype(np.float64) / 255.0

        energy = self._motion_energy(mouth)
        self._energy_window.append(energy)
        band_energy = self._band_limited_energy()
        probability = 1.0 / (1.0 + math.exp(
            -_CALIBRATION_STEEPNESS * (band_energy - _CALIBRATION_MIDPOINT)
        ))
        # Hysteresis: flip on above the high mark, off below the low mark.
        if not self._speaking and probability >= _SPEAK_ON_THRESHOLD:
            self._speaking = True
        elif self._speaking and probability <= _SPEAK_OFF_THRESHOLD:
            self._speaking = False

        self.observations += 1
        return VisualSpeechObservation(
            at=at,
            face_present=True,
            mouth_region=region,
            motion_energy=energy,
            speaking_probability=probability,
            speaking=self._speaking,
            viseme_features=self._viseme_features(mouth),
            transcript=None,
            transcript_source=(
                "vsr_model_attached_but_decoding_not_wired"
                if self._vsr_session is not None
                else "unavailable_no_vsr_model"
            ),
        )

    # ── internals ──────────────────────────────────────────────

    def _process_landmark_frame(
        self, frame: np.ndarray, at: float
    ) -> VisualSpeechObservation | None:
        """Viseme-grade path: normalized lip aperture from face landmarks.
        Returns None when no face is tracked (caller falls back)."""
        metrics = self._lip_tracker.lip_metrics(np.asarray(frame))
        if metrics is None:
            # No landmarks this frame: reset aperture history and let the
            # detector path own the frame (and the energy window).
            self._previous_aperture = None
            return None
        aperture = float(metrics["aperture"])
        width = float(metrics["width"])
        previous = self._previous_aperture
        self._previous_aperture = aperture
        delta = (aperture - previous) if previous is not None else 0.0
        energy = abs(delta)
        # The SIGNED derivative goes into the band window: rectifying it
        # would double the articulation frequency out of the speech band.
        self._energy_window.append(delta)
        band_energy = self._band_limited_energy()
        probability = 1.0 / (1.0 + math.exp(
            -_LANDMARK_STEEPNESS * (band_energy - _LANDMARK_MIDPOINT)
        ))
        was_speaking = self._speaking
        if not self._speaking and probability >= _SPEAK_ON_THRESHOLD:
            self._speaking = True
        elif self._speaking and probability <= _SPEAK_OFF_THRESHOLD:
            self._speaking = False

        # Bounded-vocabulary lip reading: feed the viseme decoder while
        # speech is active; decode on the utterance boundary.
        transcript: str | None = None
        transcript_source = "no_utterance"
        self._viseme_decoder.feed(aperture, width, speaking=self._speaking)
        if was_speaking and not self._speaking:
            result = self._viseme_decoder.decode()
            self.last_lip_read = result
            if result.word is not None:
                transcript = result.word
                transcript_source = "viseme_command_decoder"
            else:
                transcript_source = f"viseme_decoder_{result.reason}"

        self.observations += 1
        return VisualSpeechObservation(
            at=at,
            face_present=True,
            mouth_region=None,
            motion_energy=energy,
            speaking_probability=probability,
            speaking=self._speaking,
            viseme_features=[
                round(min(1.0, aperture / 0.15), 5),
                round(min(1.0, width / 0.8), 5),
                round(min(1.0, energy / 0.05), 5),
            ],
            transcript=transcript,
            transcript_source=transcript_source,
        )

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            # BGR → luma without requiring cv2 in this process.
            arr = (
                0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]
            )
        if arr.ndim != 2:
            raise ValueError("frame must be HxW or HxWx3")
        return arr.astype(np.uint8)

    def _motion_energy(self, mouth: np.ndarray) -> float:
        previous = self._previous_mouth
        if previous is None or previous.shape != mouth.shape:
            self._previous_mouth = mouth
            return 0.0
        energy = float(np.mean(np.abs(mouth - previous)))
        self._previous_mouth = mouth
        return energy

    def _band_limited_energy(self) -> float:
        """Mean spectral energy of the mouth-motion signal restricted to
        the syllabic band — steady drift and camera noise fall outside."""
        window = np.array(self._energy_window, dtype=np.float64)
        if window.size < 8:
            return 0.0
        window = window - np.mean(window)
        spectrum = np.abs(np.fft.rfft(window)) / window.size
        freqs = np.fft.rfftfreq(window.size, d=1.0 / self.fps)
        band = (freqs >= _SPEECH_BAND_LOW) & (freqs <= _SPEECH_BAND_HIGH)
        if not np.any(band):
            return float(np.mean(spectrum[1:])) if spectrum.size > 1 else 0.0
        return float(np.sum(spectrum[band]))

    @staticmethod
    def _viseme_features(mouth: np.ndarray) -> list[float]:
        """Coarse geometric articulation features in [0, 1]:
        openness (dark-pixel ratio), vertical aperture profile, contrast."""
        if mouth.size == 0:
            return [0.0, 0.0, 0.0]
        darkness = mouth < max(0.15, float(np.percentile(mouth, 20)))
        openness = float(np.mean(darkness))
        # Vertical aperture: the tallest dark column fraction — how far
        # open the mouth is, independent of its width.
        aperture_profile = np.mean(darkness, axis=0)
        aperture = float(np.max(aperture_profile)) if aperture_profile.size else 0.0
        contrast = float(np.clip(np.std(mouth) * 4.0, 0.0, 1.0))
        return [round(openness, 5), round(aperture, 5), round(contrast, 5)]
