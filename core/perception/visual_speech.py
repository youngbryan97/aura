"""Consented visual-speech recognition with evidence-driven abstention.

Audio transcription is never accepted as visual speech. The extractor supplies
mouth crops and visual quality evidence; a visual-only decoder supplies a
candidate transcript. This layer enforces consent, source quality, speaker-track
ambiguity, optional A/V alignment, calibrated uncertainty, privacy, and causal
publication to the canonical multimodal synchronizer.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from core.container import ServiceContainer
from core.perception.multimodal_sync import (
    Calibration,
    MissingReason,
    Modality,
    MultimodalSynchronizer,
    PerceptualClaim,
    PerceptualEvent,
    PrivacyClass,
    PrivacyPolicy,
)
from core.runtime.service_access import optional_service


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _inspect_video_source(video_path: str | Path) -> tuple[Path, bool, int]:
    path = Path(video_path).expanduser().resolve()
    try:
        return path, path.is_file(), path.stat().st_size
    except OSError:
        return path, False, 0


class VisualSpeechStatus(StrEnum):
    TRANSCRIBED = "transcribed"
    CANDIDATE = "candidate"
    ABSTAINED = "abstained"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class VisualSpeechConsent:
    """Explicit, expiring authority for visual-speech processing."""

    consent_id: str
    subject_id: str
    purpose: str
    issued_at: float
    expires_at: float
    allow_visual_speech: bool = True
    allow_audio_alignment: bool = False
    allow_raw_retention: bool = False

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("consent_id", self.consent_id, 160),
            ("subject_id", self.subject_id, 160),
            ("purpose", self.purpose, 240),
        ):
            if not value or len(value) > limit:
                raise ValueError(f"{name} must be present and bounded")
        if not math.isfinite(self.issued_at) or not math.isfinite(self.expires_at):
            raise ValueError("consent timestamps must be finite")
        if self.expires_at <= self.issued_at:
            raise ValueError("consent expiry must follow issuance")
        if self.expires_at - self.issued_at > 24 * 60 * 60:
            raise ValueError("visual-speech consent cannot exceed 24 hours")
        if self.allow_raw_retention:
            raise ValueError("raw visual-speech retention is not supported")

    def denial_reason(self, now: float) -> str | None:
        if not self.allow_visual_speech:
            return "visual_speech_not_consented"
        if now < self.issued_at:
            return "consent_not_yet_valid"
        if now >= self.expires_at:
            return "consent_expired"
        return None


@dataclass(frozen=True)
class AudioActivitySample:
    timestamp_s: float
    activity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("audio activity timestamp must be finite and non-negative")
        if not math.isfinite(self.activity) or not 0.0 <= self.activity <= 1.0:
            raise ValueError("audio activity must be between 0 and 1")


@dataclass(frozen=True)
class AlignmentEvidence:
    evaluated: bool
    correlation: float = 0.0
    offset_ms: float = 0.0
    matched_samples: int = 0
    passed: bool = False
    reason: str = "not_requested"


@dataclass(frozen=True)
class VisualSpeechEvidence:
    """Ephemeral mouth crops plus bounded, non-biometric quality evidence."""

    source_digest: str
    mouth_crops: NDArray[np.uint8] = field(repr=False, compare=False)
    timestamps_s: tuple[float, ...]
    mouth_activity: tuple[float, ...]
    source_fps: float
    sampled_fps: float
    duration_s: float
    decoded_frames: int
    mouth_frames: int
    face_detection_coverage: float
    mouth_landmark_coverage: float
    mean_brightness: float
    mean_blur_variance: float
    mean_mouth_motion: float
    competing_face_ratio: float
    ambiguous_face_frames: int
    track_switches: int
    speaker_track_id: str
    source_audio_present: bool
    source_audio_presence_known: bool
    extractor: str
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_digest or len(self.source_digest) > 128:
            raise ValueError("source digest must be present and bounded")
        if self.mouth_crops.dtype != np.uint8 or self.mouth_crops.ndim != 4:
            raise ValueError("mouth crops must be uint8 [T,H,W,C]")
        if (
            self.mouth_crops.shape[1] < 32
            or self.mouth_crops.shape[2] < 32
            or self.mouth_crops.shape[1] > 256
            or self.mouth_crops.shape[2] > 256
            or self.mouth_crops.shape[3] not in (1, 3)
        ):
            raise ValueError("mouth crop shape is outside decoder bounds")
        if self.mouth_crops.shape[0] != self.mouth_frames:
            raise ValueError("mouth crop count must match mouth_frames")
        if len(self.timestamps_s) != self.mouth_frames:
            raise ValueError("timestamps must match mouth_frames")
        if len(self.mouth_activity) != self.mouth_frames:
            raise ValueError("mouth activity must match mouth_frames")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.timestamps_s
        ):
            raise ValueError("mouth timestamps must be finite and non-negative")
        if any(
            current <= previous
            for previous, current in zip(
                self.timestamps_s,
                self.timestamps_s[1:],
                strict=False,
            )
        ):
            raise ValueError("mouth timestamps must be strictly increasing")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.mouth_activity
        ):
            raise ValueError("mouth activity must be normalized")
        if self.decoded_frames < self.mouth_frames or self.mouth_frames < 0:
            raise ValueError("frame counts are inconsistent")
        for value in (
            self.source_fps,
            self.sampled_fps,
            self.duration_s,
            self.face_detection_coverage,
            self.mouth_landmark_coverage,
            self.mean_brightness,
            self.mean_blur_variance,
            self.mean_mouth_motion,
            self.competing_face_ratio,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("visual-speech quality values must be finite and non-negative")
        for value in (
            self.face_detection_coverage,
            self.mouth_landmark_coverage,
            self.competing_face_ratio,
        ):
            if value > 1.0:
                raise ValueError("visual-speech coverage ratios cannot exceed 1")
        if not self.speaker_track_id or len(self.speaker_track_id) > 160:
            raise ValueError("speaker track id must be present and bounded")
        if not self.extractor or len(self.extractor) > 160:
            raise ValueError("extractor must be present and bounded")
        if len(self.quality_flags) > 16:
            raise ValueError("quality flags exceed bounded limit")


@dataclass(frozen=True)
class VisualSpeechPolicy:
    min_duration_s: float = 0.8
    max_duration_s: float = 20.0
    max_source_bytes: int = 512 * 1024 * 1024
    target_fps: float = 25.0
    max_frames: int = 500
    min_mouth_frames: int = 20
    min_face_coverage: float = 0.80
    min_landmark_coverage: float = 0.75
    min_brightness: float = 28.0
    min_blur_variance: float = 12.0
    min_mouth_motion: float = 0.75
    max_competing_face_ratio: float = 0.65
    max_ambiguous_face_fraction: float = 0.10
    max_track_switches: int = 1
    min_alignment_correlation: float = 0.20
    max_alignment_offset_ms: float = 300.0
    alignment_search_ms: int = 500
    decoder_timeout_s: float = 180.0
    actionable_confidence: float = 0.70

    def __post_init__(self) -> None:
        if not 0.1 <= self.min_duration_s < self.max_duration_s <= 120.0:
            raise ValueError("visual-speech duration policy is invalid")
        if not 1 <= self.max_frames <= 5000 or self.min_mouth_frames > self.max_frames:
            raise ValueError("visual-speech frame policy is invalid")
        if not 1_024 <= self.max_source_bytes <= 2 * 1024 * 1024 * 1024:
            raise ValueError("visual-speech source byte limit is invalid")
        for value in (
            self.min_face_coverage,
            self.min_landmark_coverage,
            self.max_competing_face_ratio,
            self.max_ambiguous_face_fraction,
            self.min_alignment_correlation,
            self.actionable_confidence,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("visual-speech ratio thresholds must be between 0 and 1")
        if not 1.0 <= self.target_fps <= 120.0:
            raise ValueError("visual-speech target fps is invalid")
        if not 1.0 <= self.decoder_timeout_s <= 1800.0:
            raise ValueError("visual-speech decoder timeout is invalid")


@dataclass(frozen=True)
class BackendPrediction:
    transcript: str
    confidence: float | None
    calibrated: bool
    backend: str
    model_id: str
    score: float | None = None
    alternatives: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if len(self.transcript) > 4000:
            raise ValueError("visual-speech transcript must be bounded")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("backend confidence must be between 0 and 1")
        if not self.backend or len(self.backend) > 160:
            raise ValueError("backend name must be present and bounded")
        if not self.model_id or len(self.model_id) > 240:
            raise ValueError("model id must be present and bounded")
        if len(self.alternatives) > 8:
            raise ValueError("backend alternatives exceed bounded limit")


class VisualSpeechExtractor(Protocol):
    def extract(self, video_path: Path, policy: VisualSpeechPolicy) -> VisualSpeechEvidence: ...


class VisualSpeechBackend(Protocol):
    def available(self) -> tuple[bool, str]: ...

    async def infer(
        self,
        mouth_crops: NDArray[np.uint8],
        *,
        fps: float,
    ) -> BackendPrediction: ...


@dataclass(frozen=True)
class VisualSpeechResult:
    status: VisualSpeechStatus
    transcript: str
    confidence: float
    actionable: bool
    calibrated: bool
    reason: str
    backend: str
    model_id: str
    consent_id: str
    subject_id: str
    speaker_track_id: str
    speaker_association: str
    evidence_digest: str
    alignment: AlignmentEvidence
    quality: dict[str, float | int | str | bool]
    quality_flags: tuple[str, ...]
    privacy: dict[str, str | bool]
    created_at: float

    def __post_init__(self) -> None:
        if len(self.transcript) > 4000:
            raise ValueError("visual-speech result transcript must be bounded")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("visual-speech result confidence must be between 0 and 1")
        if self.actionable and (
            self.status is not VisualSpeechStatus.TRANSCRIBED or not self.calibrated
        ):
            raise ValueError("only calibrated transcribed results can be actionable")
        if not math.isfinite(self.created_at) or self.created_at <= 0.0:
            raise ValueError("visual-speech result timestamp must be positive and finite")
        for name, value, limit in (
            ("reason", self.reason, 320),
            ("backend", self.backend, 160),
            ("model_id", self.model_id, 240),
            ("consent_id", self.consent_id, 160),
            ("subject_id", self.subject_id, 160),
            ("speaker_track_id", self.speaker_track_id, 160),
            ("evidence_digest", self.evidence_digest, 128),
        ):
            if not value or len(value) > limit:
                raise ValueError(f"{name} must be present and bounded")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["status"] = self.status.value
        return payload


class VisualSpeechEngine:
    """Govern visual-only speech recognition and publish uncertainty."""

    def __init__(
        self,
        *,
        extractor: VisualSpeechExtractor,
        backend: VisualSpeechBackend,
        policy: VisualSpeechPolicy | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.extractor = extractor
        self.backend = backend
        self.policy = policy or VisualSpeechPolicy()
        self._wall_clock = wall_clock
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._sequence_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = False
        self._requests = 0
        self._status_counts: dict[str, int] = {}
        self._latest_status: dict[str, object] | None = None
        self.shutdown_timeout_s = 10.0

    async def start(self) -> None:
        """Publish the governed engine without eagerly loading its 1 GiB model."""
        if self._started:
            return
        ServiceContainer.register_instance("visual_speech", self, required=False)
        self._started = True

    async def stop(self) -> None:
        """Release any model-lane lease held by the decoder backend."""
        unload = getattr(self.backend, "unload", None)
        if callable(unload):
            result = unload(reason="visual_speech_service_stopped")
            if inspect.isawaitable(result):
                await result
        self._started = False

    def get_status(self) -> dict[str, object]:
        """Return bounded operational evidence without transcript or subject data."""
        available, availability_reason = self.backend.available()
        backend_status: dict[str, object] = {
            "available": available,
            "availability_reason": availability_reason[:160],
            "type": type(self.backend).__name__,
        }
        status_getter = getattr(self.backend, "get_status", None)
        if callable(status_getter):
            raw_status: Any = status_getter()
            if isinstance(raw_status, dict):
                for key in (
                    "loaded",
                    "lane_owned",
                    "integrity_verified",
                    "loads",
                    "inferences",
                    "last_error",
                    "backend",
                    "modality",
                    "checkpoint_sha256",
                    "runtime_sha256",
                    "upstream_commit",
                    "reported_lrs3_wer_percent",
                    "confidence_calibrated",
                ):
                    value = raw_status.get(key)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        backend_status[key] = value
        with self._state_lock:
            latest = dict(self._latest_status) if self._latest_status is not None else None
            return {
                "started": self._started,
                "requests": self._requests,
                "status_counts": dict(self._status_counts),
                "latest": latest,
                "backend": backend_status,
                "privacy": self._privacy_status(),
                "raw_retention_supported": False,
                "actionability_requires_calibrated_confidence": True,
            }

    async def transcribe_video(
        self,
        video_path: str | Path,
        *,
        consent: VisualSpeechConsent,
        audio_activity: Sequence[AudioActivitySample] = (),
    ) -> VisualSpeechResult:
        now = self._wall_clock()
        denial = consent.denial_reason(now)
        if denial is not None:
            result = self._result_without_evidence(
                VisualSpeechStatus.DENIED,
                denial,
                consent,
                created_at=now,
            )
            self._publish(result, missing_reason=MissingReason.PERMISSION_DENIED)
            return result
        if audio_activity and not consent.allow_audio_alignment:
            result = self._result_without_evidence(
                VisualSpeechStatus.DENIED,
                "audio_alignment_not_consented",
                consent,
                created_at=now,
            )
            self._publish(result, missing_reason=MissingReason.PERMISSION_DENIED)
            return result

        path, source_exists, source_bytes = await asyncio.to_thread(
            _inspect_video_source,
            video_path,
        )
        if not source_exists:
            result = self._result_without_evidence(
                VisualSpeechStatus.UNAVAILABLE,
                "video_source_not_found",
                consent,
                created_at=now,
            )
            self._publish(result, missing_reason=MissingReason.UNAVAILABLE)
            return result
        if source_bytes <= 0 or source_bytes > self.policy.max_source_bytes:
            result = self._result_without_evidence(
                VisualSpeechStatus.ABSTAINED,
                "video_source_size_out_of_bounds",
                consent,
                created_at=now,
            )
            self._publish(result, missing_reason=MissingReason.SENSOR_ERROR)
            return result

        async with self._lock:
            try:
                evidence = await asyncio.to_thread(self.extractor.extract, path, self.policy)
            except asyncio.CancelledError:
                raise
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                result = self._result_without_evidence(
                    VisualSpeechStatus.ERROR,
                    f"extractor_error:{type(exc).__name__}",
                    consent,
                    created_at=self._wall_clock(),
                )
                self._publish(result, missing_reason=MissingReason.SENSOR_ERROR)
                return result

            quality_reasons = self._quality_rejection_reasons(evidence)
            alignment = self._align(evidence, audio_activity)
            if alignment.evaluated and not alignment.passed:
                quality_reasons.append(f"av_alignment:{alignment.reason}")
            if quality_reasons:
                result = self._result_from_evidence(
                    status=VisualSpeechStatus.ABSTAINED,
                    transcript="",
                    confidence=0.0,
                    actionable=False,
                    calibrated=False,
                    reason=",".join(quality_reasons[:8]),
                    backend="not_run",
                    model_id="not_run",
                    consent=consent,
                    evidence=evidence,
                    alignment=alignment,
                    created_at=self._wall_clock(),
                )
                self._publish(result, missing_reason=MissingReason.SENSOR_ERROR)
                self._zero_crops(evidence.mouth_crops)
                return result

            available, availability_reason = self.backend.available()
            if not available:
                result = self._result_from_evidence(
                    status=VisualSpeechStatus.UNAVAILABLE,
                    transcript="",
                    confidence=0.0,
                    actionable=False,
                    calibrated=False,
                    reason=f"decoder_unavailable:{availability_reason[:160]}",
                    backend=type(self.backend).__name__,
                    model_id="unavailable",
                    consent=consent,
                    evidence=evidence,
                    alignment=alignment,
                    created_at=self._wall_clock(),
                )
                self._publish(result, missing_reason=MissingReason.UNAVAILABLE)
                self._zero_crops(evidence.mouth_crops)
                return result

            try:
                prediction = await asyncio.wait_for(
                    self.backend.infer(evidence.mouth_crops, fps=evidence.sampled_fps),
                    timeout=self.policy.decoder_timeout_s,
                )
            except asyncio.CancelledError:
                self._zero_crops(evidence.mouth_crops)
                raise
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                result = self._result_from_evidence(
                    status=VisualSpeechStatus.ERROR,
                    transcript="",
                    confidence=0.0,
                    actionable=False,
                    calibrated=False,
                    reason=f"decoder_error:{type(exc).__name__}",
                    backend=type(self.backend).__name__,
                    model_id="error",
                    consent=consent,
                    evidence=evidence,
                    alignment=alignment,
                    created_at=self._wall_clock(),
                )
                self._publish(result, missing_reason=MissingReason.SENSOR_ERROR)
                self._zero_crops(evidence.mouth_crops)
                return result

            transcript = " ".join(prediction.transcript.strip().split())[:4000]
            if not transcript:
                result = self._result_from_evidence(
                    status=VisualSpeechStatus.ABSTAINED,
                    transcript="",
                    confidence=0.0,
                    actionable=False,
                    calibrated=prediction.calibrated,
                    reason="decoder_returned_empty_transcript",
                    backend=prediction.backend,
                    model_id=prediction.model_id,
                    consent=consent,
                    evidence=evidence,
                    alignment=alignment,
                    created_at=self._wall_clock(),
                )
                self._publish(result, missing_reason=MissingReason.SENSOR_ERROR)
                self._zero_crops(evidence.mouth_crops)
                return result

            quality_confidence = self._quality_confidence(evidence, alignment)
            if prediction.calibrated and prediction.confidence is not None:
                confidence = _clamp01(prediction.confidence * quality_confidence)
                actionable = confidence >= self.policy.actionable_confidence
                status = (
                    VisualSpeechStatus.TRANSCRIBED
                    if actionable
                    else VisualSpeechStatus.CANDIDATE
                )
                reason = "calibrated_visual_only_transcript"
            else:
                confidence = min(0.49, 0.55 * quality_confidence)
                actionable = False
                status = VisualSpeechStatus.CANDIDATE
                reason = "uncalibrated_visual_only_candidate"

            result = self._result_from_evidence(
                status=status,
                transcript=transcript,
                confidence=confidence,
                actionable=actionable,
                calibrated=prediction.calibrated,
                reason=reason,
                backend=prediction.backend,
                model_id=prediction.model_id,
                consent=consent,
                evidence=evidence,
                alignment=alignment,
                created_at=self._wall_clock(),
            )
            self._publish(
                result,
                missing_reason=(
                    MissingReason.UNCALIBRATED
                    if not result.calibrated
                    else None
                ),
            )
            self._zero_crops(evidence.mouth_crops)
            return result

    def _quality_rejection_reasons(self, evidence: VisualSpeechEvidence) -> list[str]:
        reasons: list[str] = []
        policy = self.policy
        if evidence.duration_s < policy.min_duration_s:
            reasons.append("video_too_short")
        if evidence.duration_s > policy.max_duration_s:
            reasons.append("video_too_long")
        if evidence.mouth_frames < policy.min_mouth_frames:
            reasons.append("insufficient_mouth_frames")
        if evidence.face_detection_coverage < policy.min_face_coverage:
            reasons.append("face_detection_coverage_low")
        if evidence.mouth_landmark_coverage < policy.min_landmark_coverage:
            reasons.append("mouth_landmark_coverage_low")
        if evidence.mean_brightness < policy.min_brightness:
            reasons.append("poor_lighting")
        if evidence.mean_blur_variance < policy.min_blur_variance:
            reasons.append("blur_too_high")
        if evidence.mean_mouth_motion < policy.min_mouth_motion:
            reasons.append("no_reliable_mouth_motion")
        if evidence.competing_face_ratio > policy.max_competing_face_ratio:
            reasons.append("speaker_face_ambiguous")
        ambiguous_fraction = evidence.ambiguous_face_frames / max(1, evidence.decoded_frames)
        if ambiguous_fraction > policy.max_ambiguous_face_fraction:
            reasons.append("multiple_speakers_ambiguous")
        if evidence.track_switches > policy.max_track_switches:
            reasons.append("speaker_track_unstable")
        return reasons

    def _align(
        self,
        evidence: VisualSpeechEvidence,
        audio_activity: Sequence[AudioActivitySample],
    ) -> AlignmentEvidence:
        if not audio_activity:
            return AlignmentEvidence(evaluated=False, reason="video_only")
        if len(audio_activity) < 5 or len(evidence.timestamps_s) < 5:
            return AlignmentEvidence(evaluated=True, reason="insufficient_alignment_samples")

        audio_times = np.asarray([sample.timestamp_s for sample in audio_activity], dtype=np.float64)
        audio_values = np.asarray([sample.activity for sample in audio_activity], dtype=np.float64)
        video_times = np.asarray(evidence.timestamps_s, dtype=np.float64)
        video_values = np.asarray(evidence.mouth_activity, dtype=np.float64)
        best_correlation = -1.0
        best_offset_ms = 0.0
        best_count = 0
        step_ms = 40
        for offset_ms in range(
            -self.policy.alignment_search_ms,
            self.policy.alignment_search_ms + step_ms,
            step_ms,
        ):
            shifted = video_times + offset_ms / 1000.0
            indices = np.searchsorted(audio_times, shifted)
            indices = np.clip(indices, 1, len(audio_times) - 1)
            left = indices - 1
            choose_left = np.abs(audio_times[left] - shifted) <= np.abs(
                audio_times[indices] - shifted
            )
            nearest = np.where(choose_left, left, indices)
            distances = np.abs(audio_times[nearest] - shifted)
            mask = distances <= 0.08
            if int(mask.sum()) < 5:
                continue
            x = video_values[mask]
            y = audio_values[nearest[mask]]
            if float(np.std(x)) < 1e-6 or float(np.std(y)) < 1e-6:
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(x, y)[0, 1])
                if not math.isfinite(correlation):
                    correlation = 0.0
            if correlation > best_correlation:
                best_correlation = correlation
                best_offset_ms = float(offset_ms)
                best_count = int(mask.sum())

        passed = (
            best_count >= 5
            and best_correlation >= self.policy.min_alignment_correlation
            and abs(best_offset_ms) <= self.policy.max_alignment_offset_ms
        )
        reason = (
            "aligned"
            if passed
            else "correlation_low"
            if best_correlation < self.policy.min_alignment_correlation
            else "offset_exceeds_limit"
        )
        return AlignmentEvidence(
            evaluated=True,
            correlation=max(-1.0, min(1.0, best_correlation)),
            offset_ms=best_offset_ms,
            matched_samples=best_count,
            passed=passed,
            reason=reason,
        )

    def _quality_confidence(
        self,
        evidence: VisualSpeechEvidence,
        alignment: AlignmentEvidence,
    ) -> float:
        policy = self.policy
        brightness = _clamp01(evidence.mean_brightness / max(1.0, policy.min_brightness * 2.0))
        blur = _clamp01(evidence.mean_blur_variance / max(1.0, policy.min_blur_variance * 4.0))
        motion = _clamp01(evidence.mean_mouth_motion / max(1.0, policy.min_mouth_motion * 4.0))
        score = (
            0.25 * evidence.face_detection_coverage
            + 0.25 * evidence.mouth_landmark_coverage
            + 0.15 * brightness
            + 0.15 * blur
            + 0.20 * motion
        )
        if alignment.evaluated:
            score *= 0.75 + 0.25 * _clamp01(alignment.correlation)
        return _clamp01(score)

    def _result_without_evidence(
        self,
        status: VisualSpeechStatus,
        reason: str,
        consent: VisualSpeechConsent,
        *,
        created_at: float,
    ) -> VisualSpeechResult:
        return VisualSpeechResult(
            status=status,
            transcript="",
            confidence=0.0,
            actionable=False,
            calibrated=False,
            reason=reason,
            backend="not_run",
            model_id="not_run",
            consent_id=consent.consent_id,
            subject_id=consent.subject_id,
            speaker_track_id="unobserved",
            speaker_association="unobserved",
            evidence_digest="none",
            alignment=AlignmentEvidence(evaluated=False),
            quality={},
            quality_flags=(),
            privacy=self._privacy_status(),
            created_at=created_at,
        )

    def _result_from_evidence(
        self,
        *,
        status: VisualSpeechStatus,
        transcript: str,
        confidence: float,
        actionable: bool,
        calibrated: bool,
        reason: str,
        backend: str,
        model_id: str,
        consent: VisualSpeechConsent,
        evidence: VisualSpeechEvidence,
        alignment: AlignmentEvidence,
        created_at: float,
    ) -> VisualSpeechResult:
        return VisualSpeechResult(
            status=status,
            transcript=transcript,
            confidence=_clamp01(confidence),
            actionable=actionable,
            calibrated=calibrated,
            reason=reason[:320],
            backend=backend[:160],
            model_id=model_id[:240],
            consent_id=consent.consent_id,
            subject_id=consent.subject_id,
            speaker_track_id=evidence.speaker_track_id,
            speaker_association="single_visible_track_not_identity_verified",
            evidence_digest=evidence.source_digest,
            alignment=alignment,
            quality={
                "duration_s": round(evidence.duration_s, 4),
                "decoded_frames": evidence.decoded_frames,
                "mouth_frames": evidence.mouth_frames,
                "face_detection_coverage": round(evidence.face_detection_coverage, 4),
                "mouth_landmark_coverage": round(evidence.mouth_landmark_coverage, 4),
                "mean_brightness": round(evidence.mean_brightness, 4),
                "mean_blur_variance": round(evidence.mean_blur_variance, 4),
                "mean_mouth_motion": round(evidence.mean_mouth_motion, 4),
                "competing_face_ratio": round(evidence.competing_face_ratio, 4),
                "ambiguous_face_frames": evidence.ambiguous_face_frames,
                "track_switches": evidence.track_switches,
                "source_audio_present": evidence.source_audio_present,
                "source_audio_presence_known": evidence.source_audio_presence_known,
                "extractor": evidence.extractor,
            },
            quality_flags=evidence.quality_flags[:16],
            privacy=self._privacy_status(),
            created_at=created_at,
        )

    @staticmethod
    def _privacy_status() -> dict[str, str | bool]:
        return {
            "classification": "sensitive",
            "raw_video_retained": False,
            "mouth_crops_retained": False,
            "biometric_identity_claimed": False,
            "transcript_retention": "caller_controlled",
        }

    @staticmethod
    def _zero_crops(crops: NDArray[np.uint8]) -> None:
        crops.fill(0)

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _publish(
        self,
        result: VisualSpeechResult,
        *,
        missing_reason: MissingReason | None = None,
    ) -> None:
        transcript_digest = hashlib.sha256(
            result.transcript.encode("utf-8", errors="ignore")
        ).hexdigest()[:24]
        with self._state_lock:
            self._requests += 1
            status_name = result.status.value
            self._status_counts[status_name] = self._status_counts.get(status_name, 0) + 1
            self._latest_status = {
                "status": status_name,
                "reason": result.reason,
                "confidence": result.confidence,
                "actionable": result.actionable,
                "calibrated": result.calibrated,
                "backend": result.backend,
                "model_id": result.model_id,
                "transcript_available": bool(result.transcript),
                "transcript_digest": transcript_digest,
                "speaker_identity_verified": False,
                "created_at": result.created_at,
            }
        service = optional_service("multimodal_synchronizer")
        if not isinstance(service, MultimodalSynchronizer):
            return
        sequence = self._next_sequence()
        observed_monotonic_ns = time.monotonic_ns()
        claims = (
            PerceptualClaim("visual_speech.video_only", True, 1.0),
            PerceptualClaim("visual_speech.transcript_available", bool(result.transcript), 0.95),
            PerceptualClaim("visual_speech.transcript_digest", transcript_digest, result.confidence),
            PerceptualClaim("visual_speech.actionable", result.actionable, 1.0),
            PerceptualClaim("visual_speech.speaker_track", result.speaker_track_id, 0.70),
        )
        flags = [
            "visual_only_lip_reading",
            "raw_video_not_retained",
            "speaker_identity_not_verified",
            "calibrated" if result.calibrated else "uncalibrated",
            f"status:{result.status.value}",
        ]
        flags.extend(result.quality_flags[:8])
        service.ingest(
            PerceptualEvent(
                event_id=f"visual-speech:{sequence}:{observed_monotonic_ns}",
                modality=Modality.SPEECH,
                source=f"visual_speech:{result.backend}"[:160],
                sequence=sequence,
                observed_at=result.created_at,
                observed_monotonic_ns=observed_monotonic_ns,
                summary="redacted visual-only speech result",
                confidence=0.0 if missing_reason is not None else result.confidence,
                claims=() if missing_reason is not None else claims,
                calibration=Calibration(
                    calibration_id=f"visual_speech:{result.model_id}"[:160],
                    status="valid" if result.calibrated else "unknown",
                    reliability=1.0 if result.calibrated else 0.70,
                ),
                provenance=(
                    "core.perception.visual_speech",
                    f"consent:{result.consent_id}"[:200],
                    f"evidence:{result.evidence_digest}"[:200],
                ),
                privacy=PrivacyPolicy(
                    classification=PrivacyClass.SENSITIVE,
                    retention="none",
                    consent_scope=result.consent_id,
                    redacted=True,
                ),
                missing_reason=missing_reason,
                quality_flags=tuple(flags[:16]),
            )
        )


_ENGINE: VisualSpeechEngine | None = None


def get_visual_speech_engine() -> VisualSpeechEngine:
    """Return the process-wide consented visual-speech service."""
    global _ENGINE
    if _ENGINE is None:
        from core.perception.visual_speech_auto_avsr import get_auto_avsr_backend
        from core.perception.visual_speech_tracking import NativeVisualSpeechExtractor

        _ENGINE = VisualSpeechEngine(
            extractor=NativeVisualSpeechExtractor(),
            backend=get_auto_avsr_backend(),
        )
    return _ENGINE


__all__ = [
    "AlignmentEvidence",
    "AudioActivitySample",
    "BackendPrediction",
    "VisualSpeechBackend",
    "VisualSpeechConsent",
    "VisualSpeechEngine",
    "VisualSpeechEvidence",
    "VisualSpeechExtractor",
    "VisualSpeechPolicy",
    "VisualSpeechResult",
    "VisualSpeechStatus",
    "get_visual_speech_engine",
]
