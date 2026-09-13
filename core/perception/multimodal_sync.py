"""Bounded, privacy-aware synchronization for continuous perception.

Sensors run at different rates and fail independently. This module turns their
typed observations into one event-time frame without treating a cached value as
fresh or allowing the most recent writer to silently win a contradiction.

The synchronizer deliberately stores summaries and scalar claims, not raw
images, audio, or transcript text. Raw media remains with the governed sensor
owner and may be referenced only through a bounded provenance identifier.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

type ClaimValue = str | int | float | bool | None
type BeliefStatus = Literal["supported", "resolved", "contested"]
type CalibrationStatus = Literal["valid", "expired", "unknown", "failed"]
type RetentionPolicy = Literal["none", "ephemeral", "session", "durable"]


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


class Modality(StrEnum):
    VISION = "vision"
    AUDIO = "audio"
    SPEECH = "speech"
    SPATIAL = "spatial"
    DEVICE = "device"
    BODY = "body"
    TEXT = "text"


class MissingReason(StrEnum):
    NOT_OBSERVED = "not_observed"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    SENSOR_ERROR = "sensor_error"
    REDACTED = "redacted"
    UNCALIBRATED = "uncalibrated"


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    SENSITIVE = "sensitive"


@dataclass(frozen=True)
class Calibration:
    """Calibration state attached to a sensor observation."""

    calibration_id: str
    status: CalibrationStatus = "unknown"
    reliability: float = 0.5
    calibrated_at: float | None = None
    valid_until: float | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id or len(self.calibration_id) > 160:
            raise ValueError("calibration_id must be present and bounded")
        if not math.isfinite(self.reliability) or not 0.0 <= self.reliability <= 1.0:
            raise ValueError("calibration reliability must be between 0 and 1")
        for value in (self.calibrated_at, self.valid_until):
            if value is not None and not math.isfinite(value):
                raise ValueError("calibration timestamps must be finite")
        if (
            self.calibrated_at is not None
            and self.valid_until is not None
            and self.valid_until < self.calibrated_at
        ):
            raise ValueError("calibration validity cannot predate calibration")

    def reliability_at(self, wall_time: float) -> float:
        if self.status == "failed":
            return 0.0
        if self.status == "expired":
            return self.reliability * 0.25
        if self.valid_until is not None and wall_time > self.valid_until:
            return self.reliability * 0.25
        if self.status == "unknown":
            return self.reliability * 0.70
        return self.reliability


@dataclass(frozen=True)
class PrivacyPolicy:
    """Retention and consent state carried with each observation."""

    classification: PrivacyClass = PrivacyClass.PRIVATE
    retention: RetentionPolicy = "none"
    consent_scope: str | None = None
    redacted: bool = True
    raw_retained: bool = False

    def __post_init__(self) -> None:
        if self.consent_scope is not None and len(self.consent_scope) > 160:
            raise ValueError("consent scope must be bounded")
        if self.raw_retained and self.retention == "none":
            raise ValueError("raw data cannot be retained under retention=none")
        if (
            self.classification is PrivacyClass.SENSITIVE
            and not self.redacted
            and not self.consent_scope
        ):
            raise ValueError("unredacted sensitive evidence requires explicit consent scope")


@dataclass(frozen=True)
class PerceptualClaim:
    """One scalar proposition that can be reconciled across modalities."""

    key: str
    value: ClaimValue
    confidence: float

    def __post_init__(self) -> None:
        if not self.key or len(self.key) > 160:
            raise ValueError("claim key must be present and bounded")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("claim confidence must be between 0 and 1")
        if isinstance(self.value, str) and len(self.value) > 320:
            raise ValueError("claim string values must be bounded")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("claim float values must be finite")


@dataclass(frozen=True)
class PerceptualEvent:
    """A single sensor result in monotonic event time."""

    event_id: str
    modality: Modality
    source: str
    sequence: int
    observed_at: float
    observed_monotonic_ns: int
    summary: str
    confidence: float
    claims: tuple[PerceptualClaim, ...] = ()
    calibration: Calibration = field(
        default_factory=lambda: Calibration("unspecified", status="unknown", reliability=0.5)
    )
    provenance: tuple[str, ...] = ()
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    missing_reason: MissingReason | None = None
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id or len(self.event_id) > 192:
            raise ValueError("event_id must be present and bounded")
        if not self.source or len(self.source) > 160:
            raise ValueError("event source must be present and bounded")
        if self.sequence < 0:
            raise ValueError("event sequence cannot be negative")
        if not math.isfinite(self.observed_at) or self.observed_at <= 0.0:
            raise ValueError("observed_at must be a positive finite wall timestamp")
        if self.observed_monotonic_ns <= 0:
            raise ValueError("observed_monotonic_ns must be positive")
        if len(self.summary) > 320:
            raise ValueError("event summary must be bounded")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be between 0 and 1")
        if self.missing_reason is not None and self.claims:
            raise ValueError("missing observations cannot carry positive claims")
        if self.missing_reason is not None and self.confidence != 0.0:
            raise ValueError("missing observations must have zero confidence")
        if len(self.claims) > 64 or len(self.provenance) > 16 or len(self.quality_flags) > 16:
            raise ValueError("event evidence collections exceed bounded limits")
        if any(not item or len(item) > 200 for item in self.provenance):
            raise ValueError("provenance identifiers must be present and bounded")
        if any(not item or len(item) > 120 for item in self.quality_flags):
            raise ValueError("quality flags must be present and bounded")

    def effective_confidence(self, wall_time: float) -> float:
        return _clamp01(self.confidence * self.calibration.reliability_at(wall_time))


@dataclass(frozen=True)
class IngestReceipt:
    event_id: str
    accepted: bool
    reason: str
    queue_depth: int
    overflow_drop: bool = False
    out_of_order: bool = False


@dataclass(frozen=True)
class FusedBelief:
    key: str
    value: ClaimValue
    confidence: float
    status: BeliefStatus
    supporting_event_ids: tuple[str, ...]
    alternatives: tuple[tuple[ClaimValue, float], ...] = ()


@dataclass(frozen=True)
class Contradiction:
    key: str
    alternatives: tuple[tuple[ClaimValue, float], ...]
    resolved: bool
    selected_value: ClaimValue = None


@dataclass(frozen=True)
class CausalDirectives:
    attention_targets: tuple[str, ...]
    memory_candidates: tuple[str, ...]
    planning_constraints: tuple[str, ...]
    repair_requirements: tuple[str, ...]


@dataclass(frozen=True)
class FusedPerceptualFrame:
    """One synchronized frame and the decisions derived from its evidence."""

    frame_id: str
    emitted_at: float
    anchor_monotonic_ns: int
    observations: Mapping[Modality, PerceptualEvent]
    missing: Mapping[Modality, MissingReason]
    beliefs: tuple[FusedBelief, ...]
    contradictions: tuple[Contradiction, ...]
    confidence: float
    uncertainty: float
    directives: CausalDirectives
    queue_overflow_drops: int
    late_events: int

    def has_usable(self, modality: Modality) -> bool:
        return modality in self.observations and modality not in self.missing

    @property
    def unresolved_contradictions(self) -> tuple[Contradiction, ...]:
        return tuple(item for item in self.contradictions if not item.resolved)

    def belief(self, key: str) -> FusedBelief | None:
        return next((item for item in self.beliefs if item.key == key), None)

    def to_status(self) -> dict[str, object]:
        """Return a bounded diagnostic view with no raw sensory content."""

        observation_status = {
            modality.value: {
                "event_id": event.event_id,
                "source": event.source,
                "confidence": round(event.effective_confidence(self.emitted_at), 4),
                "age_ms": round(
                    max(0, self.anchor_monotonic_ns - event.observed_monotonic_ns) / 1_000_000,
                    3,
                ),
                "privacy": event.privacy.classification.value,
                "redacted": event.privacy.redacted,
                "calibration": event.calibration.status,
                "quality_flags": list(event.quality_flags),
            }
            for modality, event in sorted(self.observations.items(), key=lambda item: item[0].value)
        }
        return {
            "frame_id": self.frame_id,
            "emitted_at": self.emitted_at,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "observations": observation_status,
            "missing": {
                modality.value: reason.value
                for modality, reason in sorted(self.missing.items(), key=lambda item: item[0].value)
            },
            "beliefs": [
                {
                    "key": belief.key,
                    "value_available": belief.value is not None,
                    "confidence": round(belief.confidence, 4),
                    "status": belief.status,
                }
                for belief in self.beliefs[:32]
            ],
            "contradictions": [
                {
                    "key": item.key,
                    "resolved": item.resolved,
                    "selected_value_available": item.selected_value is not None,
                    "candidate_count": len(item.alternatives),
                    "candidate_confidences": [score for _value, score in item.alternatives],
                }
                for item in self.contradictions[:16]
            ],
            "unresolved_contradictions": len(self.unresolved_contradictions),
            "directives": {
                "attention_targets": list(self.directives.attention_targets),
                "memory_candidates": list(self.directives.memory_candidates),
                "planning_constraints": list(self.directives.planning_constraints),
                "repair_requirements": list(self.directives.repair_requirements),
            },
            "queue_overflow_drops": self.queue_overflow_drops,
            "late_events": self.late_events,
        }


def _default_max_age() -> Mapping[Modality, float]:
    return {
        Modality.VISION: 6.0,
        Modality.AUDIO: 2.0,
        Modality.SPEECH: 8.0,
        Modality.SPATIAL: 6.0,
        Modality.DEVICE: 3.0,
        Modality.BODY: 3.0,
        Modality.TEXT: 15.0,
    }


@dataclass(frozen=True)
class FusionPolicy:
    queue_limit: int = 64
    retained_event_age_s: float = 60.0
    future_tolerance_s: float = 0.25
    reorder_tolerance_s: float = 1.0
    conflict_resolution_margin: float = 0.20
    memory_confidence_threshold: float = 0.65
    expected_modalities: tuple[Modality, ...] = tuple(Modality)
    critical_modalities: tuple[Modality, ...] = (Modality.DEVICE, Modality.BODY)
    max_age_s: Mapping[Modality, float] = field(default_factory=_default_max_age)

    def __post_init__(self) -> None:
        if not 4 <= self.queue_limit <= 4096:
            raise ValueError("queue_limit must be between 4 and 4096")
        for value in (
            self.retained_event_age_s,
            self.future_tolerance_s,
            self.reorder_tolerance_s,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("fusion timing limits must be finite and non-negative")
        if not 0.0 <= self.conflict_resolution_margin <= 1.0:
            raise ValueError("conflict margin must be between 0 and 1")
        if not 0.0 <= self.memory_confidence_threshold <= 1.0:
            raise ValueError("memory threshold must be between 0 and 1")
        if not self.expected_modalities:
            raise ValueError("at least one expected modality is required")
        if not set(self.critical_modalities).issubset(set(self.expected_modalities)):
            raise ValueError("critical modalities must also be expected")
        for modality in self.expected_modalities:
            age = self.max_age_s.get(modality)
            if age is None or not math.isfinite(age) or age <= 0.0:
                raise ValueError(f"missing positive max age for {modality.value}")


class MultimodalSynchronizer:
    """Thread-safe bounded event ledger and deterministic fusion engine."""

    def __init__(
        self,
        policy: FusionPolicy | None = None,
        *,
        monotonic_clock: Callable[[], int] = time.monotonic_ns,
        wall_clock: Callable[[], float] = lambda: time.time(),
    ) -> None:
        self.policy = policy or FusionPolicy()
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._queues: dict[Modality, deque[PerceptualEvent]] = {
            modality: deque() for modality in self.policy.expected_modalities
        }
        self._lock = threading.RLock()
        self._last_sequence: dict[str, int] = {}
        self._seen_event_ids: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._seen_limit = self.policy.queue_limit * len(self.policy.expected_modalities) * 4
        self._accepted = 0
        self._rejected = 0
        self._overflow_drops = 0
        self._late_events = 0
        self._fusions = 0
        self._latest: FusedPerceptualFrame | None = None

    @property
    def latest(self) -> FusedPerceptualFrame | None:
        with self._lock:
            return self._latest

    def ingest(self, event: PerceptualEvent) -> IngestReceipt:
        with self._lock:
            queue = self._queues.get(event.modality)
            if queue is None:
                self._rejected += 1
                return IngestReceipt(event.event_id, False, "unexpected_modality", 0)
            if event.event_id in self._seen_event_ids:
                self._rejected += 1
                return IngestReceipt(event.event_id, False, "duplicate_event", len(queue))

            now_ns = self._monotonic_clock()
            future_ns = int(self.policy.future_tolerance_s * 1_000_000_000)
            retained_ns = int(self.policy.retained_event_age_s * 1_000_000_000)
            if event.observed_monotonic_ns > now_ns + future_ns:
                self._rejected += 1
                return IngestReceipt(event.event_id, False, "future_event_time", len(queue))
            if now_ns - event.observed_monotonic_ns > retained_ns:
                self._rejected += 1
                return IngestReceipt(event.event_id, False, "expired_before_ingest", len(queue))

            previous_sequence = self._last_sequence.get(event.source)
            out_of_order = previous_sequence is not None and event.sequence <= previous_sequence
            if out_of_order:
                reorder_ns = int(self.policy.reorder_tolerance_s * 1_000_000_000)
                latest_source_time = max(
                    (
                        item.observed_monotonic_ns
                        for candidate_queue in self._queues.values()
                        for item in candidate_queue
                        if item.source == event.source
                    ),
                    default=event.observed_monotonic_ns,
                )
                if latest_source_time - event.observed_monotonic_ns > reorder_ns:
                    self._rejected += 1
                    return IngestReceipt(
                        event.event_id,
                        False,
                        "outside_reorder_window",
                        len(queue),
                        out_of_order=True,
                    )
                self._late_events += 1
            else:
                self._last_sequence[event.source] = event.sequence

            overflow = len(queue) >= self.policy.queue_limit
            if overflow:
                queue.popleft()
                self._overflow_drops += 1
            queue.append(event)
            self._remember_event_id(event.event_id)
            self._accepted += 1
            return IngestReceipt(
                event.event_id,
                True,
                "accepted_out_of_order" if out_of_order else "accepted",
                len(queue),
                overflow_drop=overflow,
                out_of_order=out_of_order,
            )

    def _remember_event_id(self, event_id: str) -> None:
        if len(self._seen_order) >= self._seen_limit:
            evicted = self._seen_order.popleft()
            self._seen_event_ids.discard(evicted)
        self._seen_order.append(event_id)
        self._seen_event_ids.add(event_id)

    def fuse(
        self,
        frame_id: str,
        *,
        anchor_monotonic_ns: int | None = None,
        emitted_at: float | None = None,
    ) -> FusedPerceptualFrame:
        if not frame_id or len(frame_id) > 192:
            raise ValueError("frame_id must be present and bounded")
        with self._lock:
            anchor_ns = anchor_monotonic_ns or self._monotonic_clock()
            wall_time = emitted_at or self._wall_clock()
            observations: dict[Modality, PerceptualEvent] = {}
            missing: dict[Modality, MissingReason] = {}
            arbitration_events: list[PerceptualEvent] = []

            for modality in self.policy.expected_modalities:
                candidates = [
                    event
                    for event in self._queues[modality]
                    if event.observed_monotonic_ns <= anchor_ns
                ]
                if not candidates:
                    missing[modality] = MissingReason.NOT_OBSERVED
                    continue
                max_age_ns = int(self.policy.max_age_s[modality] * 1_000_000_000)
                latest_by_source: dict[str, PerceptualEvent] = {}
                for event in candidates:
                    previous = latest_by_source.get(event.source)
                    if previous is None or (
                        event.observed_monotonic_ns,
                        event.sequence,
                        event.event_id,
                    ) > (
                        previous.observed_monotonic_ns,
                        previous.sequence,
                        previous.event_id,
                    ):
                        latest_by_source[event.source] = event
                usable = [
                    event
                    for event in latest_by_source.values()
                    if anchor_ns - event.observed_monotonic_ns <= max_age_ns
                    and event.missing_reason is None
                    and event.calibration.reliability_at(wall_time) > 0.0
                ]
                if usable:
                    selected = max(
                        usable,
                        key=lambda event: (
                            event.effective_confidence(wall_time),
                            event.observed_monotonic_ns,
                            event.sequence,
                            event.event_id,
                        ),
                    )
                    observations[modality] = selected
                    arbitration_events.extend(usable)
                    continue

                latest = max(
                    latest_by_source.values(),
                    key=lambda event: (event.observed_monotonic_ns, event.sequence, event.event_id),
                )
                if anchor_ns - latest.observed_monotonic_ns > max_age_ns:
                    missing[modality] = MissingReason.STALE
                elif latest.missing_reason is not None:
                    missing[modality] = latest.missing_reason
                else:
                    missing[modality] = MissingReason.UNCALIBRATED

            beliefs, contradictions = self._arbitrate(arbitration_events, wall_time)
            confidence = self._frame_confidence(
                observations,
                missing,
                contradictions,
                wall_time,
            )
            uncertainty = _clamp01(
                1.0
                - confidence
                + 0.10 * sum(1 for item in contradictions if not item.resolved)
            )
            directives = self._causal_directives(missing, beliefs, contradictions, confidence)
            frame = FusedPerceptualFrame(
                frame_id=frame_id,
                emitted_at=wall_time,
                anchor_monotonic_ns=anchor_ns,
                observations=dict(observations),
                missing=dict(missing),
                beliefs=beliefs,
                contradictions=contradictions,
                confidence=confidence,
                uncertainty=uncertainty,
                directives=directives,
                queue_overflow_drops=self._overflow_drops,
                late_events=self._late_events,
            )
            self._latest = frame
            self._fusions += 1
            return frame

    def _arbitrate(
        self,
        observations: list[PerceptualEvent],
        wall_time: float,
    ) -> tuple[tuple[FusedBelief, ...], tuple[Contradiction, ...]]:
        claims_by_key: dict[str, list[tuple[PerceptualEvent, PerceptualClaim]]] = {}
        for event in observations:
            for claim in event.claims:
                claims_by_key.setdefault(claim.key, []).append((event, claim))

        beliefs: list[FusedBelief] = []
        contradictions: list[Contradiction] = []
        for key in sorted(claims_by_key):
            evidence = claims_by_key[key]
            by_value: dict[tuple[str, str], list[tuple[PerceptualEvent, PerceptualClaim]]] = {}
            values: dict[tuple[str, str], ClaimValue] = {}
            for event, claim in evidence:
                value_key = (type(claim.value).__name__, repr(claim.value))
                by_value.setdefault(value_key, []).append((event, claim))
                values[value_key] = claim.value

            scored: list[tuple[ClaimValue, float, tuple[str, ...]]] = []
            for value_key, items in by_value.items():
                non_support = 1.0
                event_ids: list[str] = []
                for event, claim in items:
                    support = _clamp01(event.effective_confidence(wall_time) * claim.confidence)
                    non_support *= 1.0 - support
                    event_ids.append(event.event_id)
                scored.append((values[value_key], _clamp01(1.0 - non_support), tuple(event_ids)))
            scored.sort(key=lambda item: (-item[1], type(item[0]).__name__, repr(item[0])))
            winner_value, winner_score, winner_ids = scored[0]
            alternatives = tuple((value, round(score, 6)) for value, score, _ in scored[:8])
            if len(scored) == 1:
                beliefs.append(
                    FusedBelief(
                        key=key,
                        value=winner_value,
                        confidence=winner_score,
                        status="supported",
                        supporting_event_ids=winner_ids,
                        alternatives=alternatives,
                    )
                )
                continue

            runner_up = scored[1][1]
            margin = winner_score - runner_up
            resolved = margin >= self.policy.conflict_resolution_margin
            contradiction = Contradiction(
                key=key,
                alternatives=alternatives,
                resolved=resolved,
                selected_value=winner_value if resolved else None,
            )
            contradictions.append(contradiction)
            beliefs.append(
                FusedBelief(
                    key=key,
                    value=winner_value if resolved else None,
                    confidence=winner_score if resolved else _clamp01(margin),
                    status="resolved" if resolved else "contested",
                    supporting_event_ids=winner_ids if resolved else tuple(
                        item.event_id for item, _claim in evidence
                    ),
                    alternatives=alternatives,
                )
            )
        return tuple(beliefs), tuple(contradictions)

    def _frame_confidence(
        self,
        observations: Mapping[Modality, PerceptualEvent],
        missing: Mapping[Modality, MissingReason],
        contradictions: tuple[Contradiction, ...],
        wall_time: float,
    ) -> float:
        expected_count = max(1, len(self.policy.expected_modalities))
        coverage = len(observations) / expected_count
        evidence_quality = (
            sum(event.effective_confidence(wall_time) for event in observations.values())
            / max(1, len(observations))
        )
        critical_coverage = sum(
            1 for modality in self.policy.critical_modalities if modality in observations
        ) / max(1, len(self.policy.critical_modalities))
        unresolved = sum(1 for item in contradictions if not item.resolved)
        denied = sum(1 for reason in missing.values() if reason is MissingReason.PERMISSION_DENIED)
        score = (
            0.50 * evidence_quality
            + 0.25 * coverage
            + 0.25 * critical_coverage
            - 0.12 * unresolved
            - 0.05 * denied
        )
        return _clamp01(score)

    def _causal_directives(
        self,
        missing: Mapping[Modality, MissingReason],
        beliefs: tuple[FusedBelief, ...],
        contradictions: tuple[Contradiction, ...],
        confidence: float,
    ) -> CausalDirectives:
        attention: list[str] = []
        planning: list[str] = []
        repair: list[str] = []

        for modality, reason in sorted(missing.items(), key=lambda item: item[0].value):
            target = f"perception-gap:{modality.value}:{reason.value}"
            attention.append(target)
            if modality in self.policy.critical_modalities:
                planning.append(f"acquire-evidence:{modality.value}")
            if reason is MissingReason.PERMISSION_DENIED:
                repair.append(f"request-consent:{modality.value}")
            elif reason is MissingReason.STALE:
                repair.append(f"refresh-sensor:{modality.value}")
            elif reason in (
                MissingReason.UNAVAILABLE,
                MissingReason.SENSOR_ERROR,
                MissingReason.UNCALIBRATED,
            ):
                repair.append(f"restore-sensor:{modality.value}:{reason.value}")

        for contradiction in contradictions:
            attention.append(f"sensor-conflict:{contradiction.key}")
            if not contradiction.resolved:
                planning.append(f"verify-before-action:{contradiction.key}")
                repair.append(f"reobserve:{contradiction.key}")

        if confidence < 0.65:
            planning.append("prefer-reversible-information-gathering")
        memory = tuple(
            belief.key
            for belief in beliefs
            if belief.status != "contested"
            and belief.confidence >= self.policy.memory_confidence_threshold
        )[:32]
        return CausalDirectives(
            attention_targets=tuple(attention[:32]),
            memory_candidates=memory,
            planning_constraints=tuple(planning[:32]),
            repair_requirements=tuple(repair[:32]),
        )

    def get_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "accepted_events": self._accepted,
                "rejected_events": self._rejected,
                "queue_overflow_drops": self._overflow_drops,
                "late_events": self._late_events,
                "fusions": self._fusions,
                "queue_depths": {
                    modality.value: len(queue)
                    for modality, queue in sorted(self._queues.items(), key=lambda item: item[0].value)
                },
                "queue_limit": self.policy.queue_limit,
                "latest": self._latest.to_status() if self._latest is not None else None,
            }


__all__ = [
    "Calibration",
    "CausalDirectives",
    "Contradiction",
    "FusedBelief",
    "FusedPerceptualFrame",
    "FusionPolicy",
    "IngestReceipt",
    "MissingReason",
    "Modality",
    "MultimodalSynchronizer",
    "PerceptualClaim",
    "PerceptualEvent",
    "PrivacyClass",
    "PrivacyPolicy",
]
