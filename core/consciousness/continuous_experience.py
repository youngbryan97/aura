"""Hash-chained continuous experience stream.

This module is the durable bridge between Aura's moment-by-moment
phenomenology, Unity binding, and outcome learning.  ``StreamOfBeing`` already
creates vivid NowMoments; Unity already binds self/world/action into coherent
states.  The missing runtime object was a replayable "movie reel" that keeps
those frames in order, notices when errors start compounding, and exports only
privacy-safe context for later cognition.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import AtomicWriteError, atomic_write_json, read_json_envelope
from core.runtime.errors import record_degradation

STREAM_SCHEMA_VERSION = 1
DEFAULT_MAX_FRAMES = 7200
PRIVATE_RETENTION_S = 24 * 60 * 60
STANDARD_RETENTION_S = 30 * 24 * 60 * 60
PUBLIC_REEL_LIMIT = 24


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return lower


def _compact(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _redact(text: str) -> str:
    if not text:
        return ""
    compact = _compact(text, 160)
    digest = hashlib.sha256(compact.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"[private:{digest}]"


def _as_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in list(values or []) if str(item))


@dataclass(frozen=True)
class ExperienceFrame:
    """One committed frame in Aura's lived stream."""

    frame_id: str
    sequence: int
    timestamp: float
    scene_id: str
    summary: str
    focus: str = ""
    objective: str = ""
    source: str = "unknown"
    unity_id: str = ""
    unity_level: str = "unknown"
    unity_score: float = 0.0
    fragmentation_score: float = 0.0
    ownership_confidence: float = 1.0
    agency_score: float = 1.0
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    outcome_score: float | None = None
    harm_score: float = 0.0
    surprise: float = 0.0
    lesson: str = ""
    transfer_tags: tuple[str, ...] = field(default_factory=tuple)
    repair_needed: bool = False
    repair_reasons: tuple[str, ...] = field(default_factory=tuple)
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    privacy_tier: str = "standard"
    previous_hash: str = ""
    frame_hash: str = ""

    def payload_for_hash(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["frame_hash"] = ""
        return payload

    def with_hashes(self, *, sequence: int, scene_id: str, previous_hash: str) -> ExperienceFrame:
        payload = {
            **self.to_dict(),
            "sequence": sequence,
            "scene_id": scene_id,
            "previous_hash": previous_hash,
            "frame_hash": "",
        }
        frame_hash = _hash_payload(payload)
        return ExperienceFrame.from_dict({**payload, "frame_hash": frame_hash})

    def to_dict(self, *, redacted: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["transfer_tags"] = list(self.transfer_tags)
        payload["repair_reasons"] = list(self.repair_reasons)
        payload["source_refs"] = list(self.source_refs)
        if redacted and self.privacy_tier == "private":
            payload["summary"] = _redact(self.summary)
            payload["focus"] = _redact(self.focus)
            payload["objective"] = _redact(self.objective)
            payload["lesson"] = _redact(self.lesson)
            payload["source_refs"] = []
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperienceFrame:
        payload = dict(raw or {})
        return cls(
            frame_id=str(payload.get("frame_id") or ""),
            sequence=int(payload.get("sequence", 0) or 0),
            timestamp=float(payload.get("timestamp", time.time()) or time.time()),
            scene_id=str(payload.get("scene_id") or ""),
            summary=str(payload.get("summary") or ""),
            focus=str(payload.get("focus") or ""),
            objective=str(payload.get("objective") or ""),
            source=str(payload.get("source") or "unknown"),
            unity_id=str(payload.get("unity_id") or ""),
            unity_level=str(payload.get("unity_level") or "unknown"),
            unity_score=_clamp(payload.get("unity_score", 0.0)),
            fragmentation_score=_clamp(payload.get("fragmentation_score", 0.0)),
            ownership_confidence=_clamp(payload.get("ownership_confidence", 1.0)),
            agency_score=_clamp(payload.get("agency_score", 1.0)),
            affect_valence=max(-1.0, min(1.0, float(payload.get("affect_valence", 0.0) or 0.0))),
            affect_arousal=_clamp(payload.get("affect_arousal", 0.0)),
            outcome_score=(
                None
                if payload.get("outcome_score") is None
                else _clamp(payload.get("outcome_score"))
            ),
            harm_score=_clamp(payload.get("harm_score", 0.0)),
            surprise=_clamp(payload.get("surprise", 0.0)),
            lesson=str(payload.get("lesson") or ""),
            transfer_tags=_as_tuple(payload.get("transfer_tags")),
            repair_needed=bool(payload.get("repair_needed", False)),
            repair_reasons=_as_tuple(payload.get("repair_reasons")),
            source_refs=_as_tuple(payload.get("source_refs")),
            privacy_tier=str(payload.get("privacy_tier") or "standard"),
            previous_hash=str(payload.get("previous_hash") or ""),
            frame_hash=str(payload.get("frame_hash") or ""),
        )


@dataclass
class ExperienceEpisode:
    """A short scene in the continuous stream."""

    scene_id: str
    started_at: float
    updated_at: float
    frame_ids: list[str] = field(default_factory=list)
    rolling_summary: str = ""
    transfer_tags: set[str] = field(default_factory=set)
    risk_score: float = 0.0

    def add(self, frame: ExperienceFrame) -> None:
        self.updated_at = frame.timestamp
        self.frame_ids.append(frame.frame_id)
        self.frame_ids = self.frame_ids[-120:]
        self.transfer_tags.update(frame.transfer_tags)
        self.risk_score = max(
            self.risk_score,
            frame.fragmentation_score,
            frame.harm_score,
            frame.surprise if frame.surprise >= 0.5 else 0.0,
        )
        focus = frame.focus or frame.objective or frame.summary
        if focus:
            self.rolling_summary = _compact(focus, 180)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "frame_ids": list(self.frame_ids),
            "rolling_summary": self.rolling_summary,
            "transfer_tags": sorted(self.transfer_tags),
            "risk_score": round(self.risk_score, 3),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperienceEpisode:
        payload = dict(raw or {})
        return cls(
            scene_id=str(payload.get("scene_id") or ""),
            started_at=float(payload.get("started_at", time.time()) or time.time()),
            updated_at=float(payload.get("updated_at", time.time()) or time.time()),
            frame_ids=[str(item) for item in list(payload.get("frame_ids") or [])],
            rolling_summary=str(payload.get("rolling_summary") or ""),
            transfer_tags={str(item) for item in list(payload.get("transfer_tags") or [])},
            risk_score=_clamp(payload.get("risk_score", 0.0)),
        )


@dataclass(frozen=True)
class CompoundingErrorReport:
    active: bool
    severity: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    recommended_mode: str = "continue"
    affected_frame_ids: tuple[str, ...] = field(default_factory=tuple)
    transfer_tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContinuousExperienceStream:
    """Durable "movie-like" sequence of lived frames."""

    def __init__(
        self,
        *,
        max_frames: int = DEFAULT_MAX_FRAMES,
        persist_path: str | Path | None = None,
        private_retention_s: float = PRIVATE_RETENTION_S,
        standard_retention_s: float = STANDARD_RETENTION_S,
        autosave: bool = True,
    ) -> None:
        self.max_frames = max(20, int(max_frames))
        self.persist_path = Path(persist_path) if persist_path else None
        self.private_retention_s = max(60.0, float(private_retention_s))
        self.standard_retention_s = max(self.private_retention_s, float(standard_retention_s))
        self.autosave = bool(autosave)
        self._frames: deque[ExperienceFrame] = deque(maxlen=self.max_frames)
        self._episodes: dict[str, ExperienceEpisode] = {}
        self._scene_counter = 0
        self._compounding_report = CompoundingErrorReport(False)
        if self.persist_path and self.persist_path.exists():
            self.load()

    @property
    def frames(self) -> list[ExperienceFrame]:
        return list(self._frames)

    @property
    def current_frame(self) -> ExperienceFrame | None:
        return self._frames[-1] if self._frames else None

    @property
    def compounding_report(self) -> CompoundingErrorReport:
        return self._compounding_report

    def append_frame(self, frame: ExperienceFrame) -> ExperienceFrame:
        previous = self.current_frame
        sequence = previous.sequence + 1 if previous else 1
        if frame.scene_id:
            scene_id = frame.scene_id
        elif previous:
            scene_id = self._scene_for(previous, frame)
        else:
            scene_id = self._next_scene_id()
        frame_id = frame.frame_id or f"exp_{int(frame.timestamp * 1000)}_{sequence}"
        committed = ExperienceFrame.from_dict({**frame.to_dict(), "frame_id": frame_id})
        committed = committed.with_hashes(
            sequence=sequence,
            scene_id=scene_id,
            previous_hash=previous.frame_hash if previous else "",
        )
        self._frames.append(committed)
        self._episode_for(committed).add(committed)
        self._compounding_report = self._detect_compounding_errors()
        self.enforce_retention()
        if self.autosave:
            self.save()
        return committed

    def append_now_moment(
        self,
        moment: Any,
        *,
        objective: str = "",
        privacy_tier: str = "standard",
    ) -> ExperienceFrame:
        substrate = getattr(moment, "substrate", None)
        affect = getattr(moment, "affect", None)
        focus = _compact(getattr(moment, "attentional_focus", "") or objective or "the present moment")
        emotion = _compact(getattr(affect, "dominant_emotion", "") or "neutral", 60)
        texture = _compact(getattr(substrate, "texture_word", "") or "steady", 60)
        summary = _compact(getattr(moment, "interior_text", "") or f"{emotion} / {texture} attending to {focus}", 280)
        frame = ExperienceFrame(
            frame_id="",
            sequence=0,
            timestamp=float(getattr(moment, "timestamp", time.time()) or time.time()),
            scene_id="",
            summary=summary,
            focus=focus,
            objective=_compact(objective),
            source="stream_of_being",
            affect_valence=max(-1.0, min(1.0, float(getattr(substrate, "valence", 0.0) or 0.0))),
            affect_arousal=_clamp(getattr(substrate, "arousal", 0.0)),
            transfer_tags=("phenomenology", f"emotion:{emotion}", f"texture:{texture}"),
            privacy_tier=privacy_tier,
        )
        return self.append_frame(frame)

    def append_from_unity(
        self,
        unity_state: Any,
        *,
        report: Any = None,
        objective: str = "",
        now_moment: Any = None,
        outcome: Any = None,
        privacy_tier: str = "standard",
    ) -> ExperienceFrame:
        focus = self._focus_from_unity(unity_state) or _compact(objective) or "current unified field"
        repair_reasons = tuple(str(item) for item in list(getattr(unity_state, "repair_reasons", []) or []))
        report_causes = tuple(
            str(item[0]) for item in list(getattr(report, "top_causes", []) or []) if item
        )
        substrate = getattr(now_moment, "substrate", None)
        lesson = _compact(getattr(outcome, "lesson", "") or "")
        tags = {
            "unity",
            f"unity:{getattr(unity_state, 'level', 'unknown')}",
            *repair_reasons,
            *report_causes,
        }
        if outcome is not None:
            tags.add(f"action:{getattr(outcome, 'action', '')}")
            if float(getattr(outcome, "surprise", 0.0) or 0.0) >= 0.5:
                tags.add("prediction_mismatch")
            if float(getattr(outcome, "harm_score", 0.0) or 0.0) > 0.0:
                tags.add("harm")
        source_refs = [str(getattr(unity_state, "unity_id", "") or "")]
        for content in list(getattr(unity_state, "contents", []) or [])[:8]:
            ref = getattr(content, "evidence_ref", None) or getattr(content, "content_id", "")
            if ref:
                source_refs.append(str(ref))
        summary = (
            f"{focus}; unity={getattr(unity_state, 'level', 'unknown')} "
            f"{float(getattr(unity_state, 'unity_score', 0.0) or 0.0):.2f}"
        )
        if repair_reasons:
            summary += f"; repair={', '.join(repair_reasons[:3])}"
        if lesson:
            summary += f"; lesson={lesson}"
        frame = ExperienceFrame(
            frame_id="",
            sequence=0,
            timestamp=float(getattr(unity_state, "created_at", time.time()) or time.time()),
            scene_id="",
            summary=_compact(summary, 320),
            focus=focus,
            objective=_compact(objective),
            source="unity_runtime",
            unity_id=str(getattr(unity_state, "unity_id", "") or ""),
            unity_level=str(getattr(unity_state, "level", "unknown") or "unknown"),
            unity_score=_clamp(getattr(unity_state, "unity_score", 0.0)),
            fragmentation_score=_clamp(getattr(unity_state, "fragmentation_score", 0.0)),
            ownership_confidence=_clamp(getattr(unity_state, "agency_ownership_score", 1.0)),
            agency_score=_clamp(getattr(unity_state, "action_readiness_score", 1.0)),
            affect_valence=max(-1.0, min(1.0, float(getattr(substrate, "valence", 0.0) or 0.0))),
            affect_arousal=_clamp(getattr(substrate, "arousal", 0.0)),
            outcome_score=(
                None
                if outcome is None
                else _clamp(getattr(outcome, "success_score", None))
            ),
            harm_score=_clamp(getattr(outcome, "harm_score", 0.0) if outcome is not None else 0.0),
            surprise=_clamp(getattr(outcome, "surprise", 0.0) if outcome is not None else 0.0),
            lesson=lesson,
            transfer_tags=tuple(sorted(tag for tag in tags if tag)),
            repair_needed=bool(getattr(unity_state, "repair_needed", False)),
            repair_reasons=repair_reasons,
            source_refs=tuple(dict.fromkeys(source_refs)),
            privacy_tier=privacy_tier,
        )
        return self.append_frame(frame)

    def learning_context(self, *, target_domain: str = "", tags: Iterable[str] = ()) -> dict[str, Any]:
        lessons = self.transfer_lessons(target_domain=target_domain, tags=tags)
        return {
            "current_frame": self.current_frame.to_dict(redacted=True) if self.current_frame else None,
            "compounding_error": self._compounding_report.to_dict(),
            "transfer_lessons": lessons,
            "safe_to_act": not self._compounding_report.active,
            "recommended_mode": self._compounding_report.recommended_mode,
        }

    def transfer_lessons(self, *, target_domain: str = "", tags: Iterable[str] = ()) -> list[dict[str, Any]]:
        wanted = {str(item) for item in tags if str(item)}
        if target_domain:
            wanted.add(str(target_domain))
        lessons: list[dict[str, Any]] = []
        for frame in reversed(self._frames):
            frame_tags = set(frame.transfer_tags)
            if not frame.lesson and not frame.repair_reasons:
                continue
            if wanted and not (wanted & frame_tags):
                generic = {"prediction_mismatch", "harm", "ownership_ambiguity", "repair_only"}
                if not (generic & frame_tags):
                    continue
            lessons.append(
                {
                    "frame_id": frame.frame_id,
                    "lesson": frame.lesson or "; ".join(frame.repair_reasons),
                    "source": frame.source,
                    "confidence": round(max(frame.surprise, frame.harm_score, frame.fragmentation_score, 0.45), 3),
                    "transfer_tags": sorted(frame_tags),
                }
            )
            if len(lessons) >= 12:
                break
        return lessons

    def export_reel(self, *, limit: int = PUBLIC_REEL_LIMIT, redacted: bool = True) -> dict[str, Any]:
        frames = list(self._frames)[-max(1, int(limit)) :]
        return {
            "schema_version": STREAM_SCHEMA_VERSION,
            "frame_count": len(self._frames),
            "latest_hash": self.current_frame.frame_hash if self.current_frame else "",
            "compounding_error": self._compounding_report.to_dict(),
            "frames": [frame.to_dict(redacted=redacted) for frame in frames],
        }

    def delete_where(self, predicate: Callable[[ExperienceFrame], bool]) -> int:
        retained = [frame for frame in self._frames if not predicate(frame)]
        removed = len(self._frames) - len(retained)
        if not removed:
            return 0
        self._frames = deque(retained, maxlen=self.max_frames)
        self._rebuild_after_deletion()
        if self.autosave:
            self.save()
        return removed

    def delete_privacy_tier(self, privacy_tier: str) -> int:
        return self.delete_where(lambda frame: frame.privacy_tier == privacy_tier)

    def enforce_retention(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else float(now)

        def expired(frame: ExperienceFrame) -> bool:
            age = now - frame.timestamp
            if frame.privacy_tier == "private":
                return age > self.private_retention_s
            return age > self.standard_retention_s

        retained = [frame for frame in self._frames if not expired(frame)]
        removed = len(self._frames) - len(retained)
        if removed:
            self._frames = deque(retained, maxlen=self.max_frames)
            self._rebuild_after_deletion()
        return removed

    def validate_replay(self) -> dict[str, Any]:
        previous_hash = ""
        for frame in self._frames:
            expected = frame.with_hashes(
                sequence=frame.sequence,
                scene_id=frame.scene_id,
                previous_hash=previous_hash,
            ).frame_hash
            if frame.previous_hash != previous_hash:
                return {
                    "valid": False,
                    "failed_frame_id": frame.frame_id,
                    "reason": "previous_hash_mismatch",
                }
            if frame.frame_hash != expected:
                return {
                    "valid": False,
                    "failed_frame_id": frame.frame_id,
                    "reason": "frame_hash_mismatch",
                }
            previous_hash = frame.frame_hash
        return {
            "valid": True,
            "frame_count": len(self._frames),
            "latest_hash": previous_hash,
        }

    def save(self) -> None:
        if not self.persist_path:
            return
        payload = {
            "frames": [frame.to_dict() for frame in self._frames],
            "episodes": [episode.to_dict() for episode in self._episodes.values()],
            "scene_counter": self._scene_counter,
            "compounding_report": self._compounding_report.to_dict(),
            "saved_at": time.time(),
        }
        try:
            atomic_write_json(
                self.persist_path,
                payload,
                schema_name="continuous_experience_stream",
                schema_version=STREAM_SCHEMA_VERSION,
            )
        except (AtomicWriteError, OSError, TypeError, ValueError) as exc:
            record_degradation(
                "continuous_experience",
                exc,
                severity="warning",
                action="continued after experience stream save failed",
            )

    def load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            envelope = read_json_envelope(self.persist_path)
            payload = dict(envelope.get("payload") or {})
        except AtomicWriteError:
            try:
                raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
                payload = raw if isinstance(raw, dict) else {}
            except (OSError, json.JSONDecodeError):
                raise
        frames = [
            ExperienceFrame.from_dict(item)
            for item in list(payload.get("frames") or [])
            if isinstance(item, dict)
        ]
        self._frames = deque(frames[-self.max_frames :], maxlen=self.max_frames)
        self._episodes = {
            str(item.get("scene_id")): ExperienceEpisode.from_dict(item)
            for item in list(payload.get("episodes") or [])
            if isinstance(item, dict) and item.get("scene_id")
        }
        self._scene_counter = int(payload.get("scene_counter", 0) or 0)
        report = payload.get("compounding_report") or {}
        self._compounding_report = CompoundingErrorReport(
            active=bool(report.get("active", False)),
            severity=_clamp(report.get("severity", 0.0)),
            reasons=_as_tuple(report.get("reasons")),
            recommended_mode=str(report.get("recommended_mode") or "continue"),
            affected_frame_ids=_as_tuple(report.get("affected_frame_ids")),
            transfer_tags=_as_tuple(report.get("transfer_tags")),
        )

    def _next_scene_id(self) -> str:
        self._scene_counter += 1
        return f"scene_{self._scene_counter:06d}"

    def _scene_for(self, previous: ExperienceFrame, frame: ExperienceFrame) -> str:
        if frame.timestamp - previous.timestamp > 15 * 60:
            return self._next_scene_id()
        if frame.source != previous.source and {frame.source, previous.source} != {"unity_runtime", "stream_of_being"}:
            return self._next_scene_id()
        if frame.objective and previous.objective and frame.objective != previous.objective:
            return self._next_scene_id()
        if frame.focus and previous.focus and frame.focus != previous.focus:
            overlap = set(frame.focus.lower().split()) & set(previous.focus.lower().split())
            if len(overlap) < 2:
                return self._next_scene_id()
        return previous.scene_id

    def _episode_for(self, frame: ExperienceFrame) -> ExperienceEpisode:
        episode = self._episodes.get(frame.scene_id)
        if episode is None:
            episode = ExperienceEpisode(
                scene_id=frame.scene_id,
                started_at=frame.timestamp,
                updated_at=frame.timestamp,
            )
            self._episodes[frame.scene_id] = episode
        return episode

    def _detect_compounding_errors(self) -> CompoundingErrorReport:
        recent = list(self._frames)[-6:]
        if len(recent) < 3:
            return CompoundingErrorReport(False)
        reasons: list[str] = []
        bad: list[ExperienceFrame] = []
        for frame in recent:
            frame_bad = False
            if frame.outcome_score is not None and frame.outcome_score <= 0.35:
                reasons.append("low_outcome_score")
                frame_bad = True
            if frame.harm_score >= 0.2:
                reasons.append("harm_accumulating")
                frame_bad = True
            if frame.surprise >= 0.55:
                reasons.append("prediction_mismatch_repeated")
                frame_bad = True
            if frame.fragmentation_score >= 0.45 or frame.repair_needed:
                reasons.append("unity_repair_pressure")
                frame_bad = True
            if frame_bad:
                bad.append(frame)
        consecutive_bad = 0
        for frame in reversed(recent):
            is_bad = frame in bad
            if is_bad:
                consecutive_bad += 1
            else:
                break
        active = consecutive_bad >= 3 or len(bad) >= 4
        if not active:
            return CompoundingErrorReport(False)
        tags = sorted({tag for frame in bad for tag in frame.transfer_tags})
        severity = _clamp((len(bad) / len(recent)) * 0.45 + max(
            max(frame.harm_score, frame.surprise, frame.fragmentation_score)
            for frame in bad
        ) * 0.55)
        return CompoundingErrorReport(
            active=True,
            severity=round(severity, 3),
            reasons=tuple(sorted(set(reasons))),
            recommended_mode="observe_stabilize_replay",
            affected_frame_ids=tuple(frame.frame_id for frame in bad[-6:]),
            transfer_tags=tuple(tags),
        )

    def _focus_from_unity(self, unity_state: Any) -> str:
        focus_id = getattr(unity_state, "global_focus_id", None)
        contents = list(getattr(unity_state, "contents", []) or [])
        for content in contents:
            if getattr(content, "content_id", None) == focus_id:
                return _compact(getattr(content, "summary", ""), 180)
        if contents:
            best = sorted(contents, key=lambda item: float(getattr(item, "salience", 0.0) or 0.0), reverse=True)[0]
            return _compact(getattr(best, "summary", ""), 180)
        return ""

    def _rebuild_after_deletion(self) -> None:
        raw_frames = [frame.to_dict() for frame in self._frames]
        self._frames = deque(maxlen=self.max_frames)
        self._episodes = {}
        self._scene_counter = 0
        previous_hash = ""
        previous_scene = ""
        for idx, raw in enumerate(raw_frames, start=1):
            frame = ExperienceFrame.from_dict(raw)
            scene_id = frame.scene_id or previous_scene or self._next_scene_id()
            rebuilt = frame.with_hashes(
                sequence=idx,
                scene_id=scene_id,
                previous_hash=previous_hash,
            )
            self._frames.append(rebuilt)
            self._episode_for(rebuilt).add(rebuilt)
            previous_hash = rebuilt.frame_hash
            previous_scene = rebuilt.scene_id
        self._compounding_report = self._detect_compounding_errors()


_CONTINUOUS_STREAM: ContinuousExperienceStream | None = None


def default_experience_stream_path() -> Path | None:
    override = os.environ.get("AURA_CONTINUOUS_EXPERIENCE_PATH")
    if override == "":
        return None
    if override:
        return Path(override).expanduser()
    if os.environ.get("AURA_TEST_MODE") == "1":
        return None
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "continuous_experience" / "stream.json"
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        record_degradation("continuous_experience", exc, severity="debug", action="using in-memory experience stream")
        return None


def get_continuous_experience_stream(
    *,
    persist_path: str | Path | None = None,
) -> ContinuousExperienceStream:
    global _CONTINUOUS_STREAM
    if _CONTINUOUS_STREAM is None:
        _CONTINUOUS_STREAM = ContinuousExperienceStream(
            persist_path=persist_path if persist_path is not None else default_experience_stream_path()
        )
        try:
            from core.container import ServiceContainer

            ServiceContainer.set("continuous_experience_stream", _CONTINUOUS_STREAM, required=False)
        except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
            record_degradation(
                "continuous_experience",
                exc,
                severity="debug",
                action="continued without service registration",
            )
    return _CONTINUOUS_STREAM


def reset_continuous_experience_stream() -> None:
    global _CONTINUOUS_STREAM
    _CONTINUOUS_STREAM = None


__all__ = [
    "CompoundingErrorReport",
    "ContinuousExperienceStream",
    "ExperienceEpisode",
    "ExperienceFrame",
    "get_continuous_experience_stream",
    "reset_continuous_experience_stream",
]
