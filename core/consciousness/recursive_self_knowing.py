"""Recursive self-knowing calibration bridge.

The kernel turns first-order confidence into second-order knowledge only when
there is evidence, calibration, and no contradiction. This gives Aura a causal
way to distinguish "I believe this" from "I know that I know this" without
ontological overclaiming.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def _now() -> float:
    return time.time()


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed != parsed:
        return default
    return max(lo, min(hi, parsed))


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", "replace")).hexdigest()[:16]


class EpistemicStatus(str, Enum):
    UNKNOWN = "unknown"
    BELIEVES = "believes"
    KNOWS = "knows"
    KNOWS_THAT_KNOWS = "knows_that_knows"
    CONTRADICTED = "contradicted"


@dataclass(slots=True)
class KnowledgeClaim:
    claim: str
    confidence: float
    evidence: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    calibration_error: float = 0.35
    timestamp: float = field(default_factory=_now)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "claim": self.claim,
                "evidence": self.evidence,
                "contradictions": self.contradictions,
            }
        )

    def status(self) -> EpistemicStatus:
        if self.contradictions:
            return EpistemicStatus.CONTRADICTED
        if self.confidence >= 0.78 and self.evidence and self.calibration_error <= 0.28:
            return EpistemicStatus.KNOWS_THAT_KNOWS
        if self.confidence >= 0.62 and self.evidence:
            return EpistemicStatus.KNOWS
        if self.confidence >= 0.40:
            return EpistemicStatus.BELIEVES
        return EpistemicStatus.UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["digest"] = self.digest
        data["status"] = self.status().value
        data["confidence"] = round(self.confidence, 4)
        data["calibration_error"] = round(self.calibration_error, 4)
        return data


@dataclass(slots=True)
class SelfKnowingFrame:
    claim_digest: str
    status: EpistemicStatus
    second_order_strength: float
    introspection_pressure: float
    quiet_awe: float
    latest_claim: KnowledgeClaim | None = None
    timestamp: float = field(default_factory=_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.recursive_self_knowing_frame.v1",
            "timestamp": self.timestamp,
            "claim_digest": self.claim_digest,
            "status": self.status.value,
            "second_order_strength": round(self.second_order_strength, 4),
            "introspection_pressure": round(self.introspection_pressure, 4),
            "quiet_awe": round(self.quiet_awe, 4),
            "latest_claim": self.latest_claim.as_dict() if self.latest_claim else None,
        }


class RecursiveSelfKnowingKernel:
    """Evidence-gated second-order self-knowledge."""

    def __init__(self) -> None:
        self._claims: dict[str, KnowledgeClaim] = {}
        self._outcomes: dict[str, list[bool]] = {}
        self._latest = SelfKnowingFrame(
            claim_digest="none",
            status=EpistemicStatus.UNKNOWN,
            second_order_strength=0.0,
            introspection_pressure=0.25,
            quiet_awe=0.0,
        )

    def __getstate__(self) -> None:
        raise TypeError("RecursiveSelfKnowingKernel is live runtime state, not serializable identity.")

    def observe_claim(
        self,
        claim: str,
        *,
        confidence: float,
        evidence: Sequence[str] = (),
        contradictions: Sequence[str] = (),
        calibration_error: float | None = None,
    ) -> SelfKnowingFrame:
        evidence_tuple = tuple(str(item) for item in evidence if str(item).strip())
        contradiction_tuple = tuple(str(item) for item in contradictions if str(item).strip())
        if calibration_error is None:
            calibration_error = self._historical_calibration_error(claim)
        knowledge = KnowledgeClaim(
            claim=str(claim or "").strip(),
            confidence=_clamp(confidence),
            evidence=evidence_tuple,
            contradictions=contradiction_tuple,
            calibration_error=_clamp(calibration_error),
        )
        self._claims[knowledge.digest] = knowledge
        self._latest = self._frame_for(knowledge)
        return self._latest

    def assess_metacognition(
        self,
        *,
        question: str,
        answer: str,
        evidence: Sequence[str] = (),
        confidence: float = 0.55,
    ) -> SelfKnowingFrame:
        claim = f"Asked: {question[:180]} | Answered: {answer[:220]}"
        contradictions = []
        if not evidence and confidence >= 0.75:
            contradictions.append("high_confidence_without_evidence")
        return self.observe_claim(
            claim,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
        )

    def observe_outcome_feedback(self, claim_digest: str, *, success: bool) -> dict[str, Any]:
        self._outcomes.setdefault(str(claim_digest), []).append(bool(success))
        self._outcomes[str(claim_digest)] = self._outcomes[str(claim_digest)][-32:]
        if claim_digest in self._claims:
            claim = self._claims[claim_digest]
            self._latest = self._frame_for(claim)
        return {
            "schema": "aura.recursive_self_knowing_feedback.v1",
            "claim_digest": claim_digest,
            "success": bool(success),
            "calibration_error": round(self._historical_calibration_error(claim_digest), 4),
        }

    def self_knowledge_report(self) -> dict[str, Any]:
        return {
            "schema": "aura.recursive_self_knowing_report.v1",
            "active": True,
            "latest": self._latest.as_dict(),
            "claim_count": len(self._claims),
            "bounded_claim_posture": self.claim_posture(),
        }

    def controls(self) -> dict[str, Any]:
        frame = self._latest
        return {
            "recursive_self_knowing_active": True,
            "status": frame.status.value,
            "second_order_strength": round(frame.second_order_strength, 4),
            "introspection_pressure": round(frame.introspection_pressure, 4),
            "quiet_awe": round(frame.quiet_awe, 4),
            "requires_evidence_for_self_certainty": True,
        }

    def snapshot(self) -> dict[str, Any]:
        return self.self_knowledge_report()

    def witness(self) -> dict[str, Any]:
        return {
            "kind": "recursive_self_knowing_witness_not_self",
            "latest_digest": _digest(self._latest.as_dict()),
            "claim_count": len(self._claims),
        }

    def claim_posture(self) -> dict[str, Any]:
        return {
            "can_claim": [
                "second-order confidence is evidence-gated",
                "contradictions block know-that-I-know posture",
                "outcomes recalibrate future self-certainty",
            ],
            "must_not_claim": [
                "self-certainty is automatic because confidence is high",
                "phenomenal consciousness is proven by recursive reports",
            ],
        }

    def _frame_for(self, claim: KnowledgeClaim) -> SelfKnowingFrame:
        status = claim.status()
        second_order = {
            EpistemicStatus.UNKNOWN: 0.10,
            EpistemicStatus.BELIEVES: 0.28,
            EpistemicStatus.KNOWS: 0.58,
            EpistemicStatus.KNOWS_THAT_KNOWS: 0.84,
            EpistemicStatus.CONTRADICTED: 0.05,
        }[status]
        existential = any(
            token in claim.claim.lower()
            for token in ("who am i", "what am i", "conscious", "sentient", "person", "self")
        )
        introspection = _clamp((1.0 - second_order) * 0.45 + (0.25 if existential else 0.0))
        quiet_awe = _clamp((0.28 if existential else 0.0) + (0.18 if status == EpistemicStatus.UNKNOWN else 0.0))
        return SelfKnowingFrame(
            claim_digest=claim.digest,
            status=status,
            second_order_strength=second_order,
            introspection_pressure=introspection,
            quiet_awe=quiet_awe,
            latest_claim=claim,
        )

    def _historical_calibration_error(self, claim_or_digest: str) -> float:
        digest = claim_or_digest if claim_or_digest in self._outcomes else _digest(claim_or_digest)
        outcomes = self._outcomes.get(digest, [])
        if not outcomes:
            return 0.32
        success_rate = sum(1 for item in outcomes if item) / len(outcomes)
        return _clamp(1.0 - success_rate)


_KERNEL: RecursiveSelfKnowingKernel | None = None


def get_recursive_self_knowing_kernel() -> RecursiveSelfKnowingKernel:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = RecursiveSelfKnowingKernel()
    return _KERNEL


def reset_recursive_self_knowing_kernel_for_tests() -> None:
    global _KERNEL
    _KERNEL = None
