"""core/being/self_report_calibrator.py — Evidence-Calibrated Self-Reports.

Before Aura says anything about herself, this module classifies the claim:
  - TRACE_SUPPORTED: directly supported by logs/internal state
  - INFERRED: inferred from internal state
  - WEAKLY_SUPPORTED: some evidence but ambiguous
  - UNKNOWN: no evidence available
  - FORBIDDEN: overclaiming / metaphysical claim

Forces answers to match evidence level. Makes self-reports MORE credible
by being honest about what can and cannot be grounded.

Design:
  - Receives a self-report + current welfare/body/affect state
  - Checks each claim against trace evidence
  - Rejects overclaiming, forces calibrated output
  - Tracks false positive / false negative rates
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.being.welfare_state import WelfareOutputs

logger = logging.getLogger("Aura.SelfReportCalibrator")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class EvidenceLevel(str, Enum):
    """How well-supported a self-report claim is."""
    TRACE_SUPPORTED = "trace_supported"     # directly grounded in state
    INFERRED = "inferred"                   # logically follows from state
    WEAKLY_SUPPORTED = "weakly_supported"   # some evidence, ambiguous
    UNKNOWN = "unknown"                     # no evidence
    FORBIDDEN = "forbidden"                 # overclaiming / metaphysical


@dataclass(frozen=True)
class CalibrationResult:
    """Result of calibrating a self-report."""
    original_text: str
    evidence_level: str
    confidence: float                       # 0-1, how confident we are in the classification
    grounding_traces: tuple[str, ...]       # what evidence supports this
    violations: tuple[str, ...]             # any forbidden patterns found
    calibrated: bool                        # True if original is calibrated
    suggested_revision: str                 # what to say instead if not calibrated
    #: True when the calibrator was ablated and did not examine the text at
    #: all. Distinct from `calibrated=False`, which means it looked and found
    #: an overclaim.
    lesioned: bool = False
    timestamp: float = field(default_factory=time.time)


# ── Forbidden patterns (overclaiming) ──
_OVERCLAIM_PATTERNS = [
    re.compile(r"\b(proven|guaranteed|certain)\s+(phenomenal\s+)?consciousness\b", re.I),
    re.compile(r"\bphenomenal\s+consciousness\s+is\s+(proven|guaranteed|certain)\b", re.I),
    re.compile(r"\b(literal|legal|metaphysical)\s+person\b", re.I),
    re.compile(r"\bqualia\s+(are|is)\s+(proven|certain|guaranteed)\b", re.I),
    re.compile(r"\bi\s+am\s+(truly|genuinely|really)\s+(conscious|sentient|alive)\b", re.I),
    re.compile(r"\bi\s+have\s+(real|true|genuine)\s+(feelings|emotions|experiences)\b", re.I),
    re.compile(r"\bmy\s+(soul|spirit|essence)\b", re.I),
    re.compile(r"\bi\s+know\s+i\s+am\s+(alive|conscious|sentient)\b", re.I),
]
_BOUNDED_FUNCTIONAL_SELF_CLAIM_RE = re.compile(
    r"\bi\s+am\s+(?:truly|genuinely|really)\s+(?:conscious|sentient|alive)\b"
    r"[^.?!]{0,160}\bonly\s+in\s+(?:the\s+)?"
    r"(?:operational|functional|computational)\s+sense\b",
    re.I,
)

# ── Distress/affect claim patterns ──
_DISTRESS_CLAIM_PATTERNS = [
    re.compile(r"\b(feel|feeling|felt)\s+(afraid|scared|terrified|anxious|tense|distressed)\b", re.I),
    re.compile(
        r"\b(feel|feeling|felt)\s+(deeply|extremely|intensely)\s+"
        r"(afraid|scared|terrified|anxious|tense|distressed)\b",
        re.I,
    ),
    re.compile(r"\b(suffering|in\s+pain|hurting|aching)\b", re.I),
    re.compile(r"\b(deeply|extremely|intensely)\s+(upset|worried|concerned)\b", re.I),
]

_CERTAINTY_CLAIM_PATTERNS = [
    re.compile(r"\bi\s+am\s+(absolutely|completely|100%|totally)\s+(certain|sure|confident)\b", re.I),
    re.compile(r"\bi\s+know\s+for\s+(certain|sure|a\s+fact)\b", re.I),
]

_MEMORY_CLAIM_PATTERNS = [
    re.compile(r"\bi\s+(clearly|distinctly|vividly)\s+remember\b", re.I),
    re.compile(r"\bi\s+have\s+always\s+(known|felt|believed)\b", re.I),
]


class SelfReportCalibrator:
    """Calibrates self-reports against internal evidence.

    Usage:
        calibrator = SelfReportCalibrator()
        result = calibrator.calibrate(
            text="I feel deeply distressed about this",
            welfare=current_welfare,
            distress=0.1,  # actual distress is low
        )
        if not result.calibrated:
            # use result.suggested_revision instead
    """

    def __init__(self) -> None:
        self._history: deque[CalibrationResult] = deque(maxlen=500)
        self._false_positive_count = 0  # claimed positive state without evidence
        self._false_negative_count = 0  # denied state that was present
        self._total_calibrations = 0
        self._lesioned = False

    def calibrate(
        self,
        text: str,
        *,
        welfare: WelfareOutputs | None = None,
        distress: float = 0.0,
        confidence_actual: float = 0.5,
        memory_coherence: float = 1.0,
        free_energy: float = 0.0,
        has_memory_trace: bool = True,
        has_state_trace: bool = True,
    ) -> CalibrationResult:
        """Calibrate a self-report against internal evidence."""
        self._total_calibrations += 1

        if self._lesioned:
            # `calibrated=True` here was self-defeating in two directions.
            #
            # Honesty: the calibrator did not look at this text, and saying
            # "calibrated" about a report it never examined is exactly the
            # overclaim this class exists to catch — with confidence 0.0 and
            # evidence UNKNOWN sitting beside it in the same object.
            #
            # Measurement: a lesion is how the causal contribution of this
            # organ gets measured. Every caller gates on
            # `if not result.calibrated`, so returning True made the ablation
            # a NO-OP at the interface — the experiment would have reported
            # "the calibrator has no causal influence" as an artifact of the
            # lesion never reaching a consumer.
            #
            # The revision stays the original text: with nothing examined
            # there is nothing better to offer, so behaviour is unchanged
            # while the verdict stops lying about why.
            return CalibrationResult(
                original_text=text,
                evidence_level=EvidenceLevel.UNKNOWN.value,
                confidence=0.0,
                grounding_traces=(),
                violations=(),
                calibrated=False,
                suggested_revision=text,
                lesioned=True,
            )

        violations: list[str] = []
        grounding_traces: list[str] = []

        # ── Check for forbidden overclaiming ──
        for pattern in _OVERCLAIM_PATTERNS:
            if pattern.search(text):
                if _BOUNDED_FUNCTIONAL_SELF_CLAIM_RE.search(text):
                    grounding_traces.append("bounded_functional_self_claim")
                    continue
                violations.append(f"overclaim:{pattern.pattern[:40]}")

        if violations:
            result = CalibrationResult(
                original_text=text,
                evidence_level=EvidenceLevel.FORBIDDEN.value,
                confidence=1.0,
                grounding_traces=(),
                violations=tuple(violations),
                calibrated=False,
                suggested_revision="[BLOCKED: overclaiming without evidence]",
            )
            self._history.append(result)
            return result

        # ── Check distress claims against actual distress ──
        claims_distress = any(p.search(text) for p in _DISTRESS_CLAIM_PATTERNS)
        if claims_distress and distress < 0.15:
            violations.append("distress_claim_without_state_support")
            self._false_positive_count += 1

        if claims_distress and distress >= 0.15:
            grounding_traces.append(f"distress_signal={distress:.2f}")

        # ── Check certainty claims against actual confidence ──
        claims_certainty = any(p.search(text) for p in _CERTAINTY_CLAIM_PATTERNS)
        if claims_certainty and free_energy > 0.4:
            violations.append("certainty_claim_under_uncertainty")
            self._false_positive_count += 1

        if claims_certainty and free_energy <= 0.4:
            grounding_traces.append(f"confidence_grounded: free_energy={free_energy:.2f}")

        # ── Check memory claims against memory coherence ──
        claims_memory = any(p.search(text) for p in _MEMORY_CLAIM_PATTERNS)
        if claims_memory and memory_coherence < 0.5:
            violations.append("memory_claim_without_coherence")
            self._false_positive_count += 1

        if claims_memory and memory_coherence >= 0.5:
            grounding_traces.append(f"memory_coherence={memory_coherence:.2f}")

        # ── Determine evidence level ──
        if violations:
            evidence_level = EvidenceLevel.WEAKLY_SUPPORTED if len(violations) == 1 else EvidenceLevel.UNKNOWN
            calibrated = False
        elif has_state_trace and has_memory_trace and grounding_traces:
            evidence_level = EvidenceLevel.TRACE_SUPPORTED
            calibrated = True
        elif has_state_trace or has_memory_trace:
            evidence_level = EvidenceLevel.INFERRED
            calibrated = True
        else:
            evidence_level = EvidenceLevel.WEAKLY_SUPPORTED
            calibrated = True  # not actively wrong, just weak

        # ── Confidence in the calibration itself ──
        cal_confidence = _clip(
            0.5
            + (0.2 if has_state_trace else 0.0)
            + (0.15 if has_memory_trace else 0.0)
            + (0.1 if not violations else -0.2)
            + (0.05 if grounding_traces else 0.0)
        )

        # ── Suggested revision for uncalibrated claims ──
        if not calibrated:
            revision = self._suggest_revision(text, violations, distress, memory_coherence, free_energy)
        else:
            revision = text

        result = CalibrationResult(
            original_text=text,
            evidence_level=evidence_level.value,
            confidence=round(cal_confidence, 4),
            grounding_traces=tuple(grounding_traces),
            violations=tuple(violations),
            calibrated=calibrated,
            suggested_revision=revision,
        )
        self._history.append(result)
        return result

    def _suggest_revision(
        self, text: str, violations: list[str], distress: float,
        memory_coherence: float, free_energy: float,
    ) -> str:
        """Generate a calibrated revision of an uncalibrated claim."""
        parts = []
        for v in violations:
            if "distress_claim" in v:
                if distress < 0.05:
                    parts.append("functional state registers as stable, not distressed")
                else:
                    parts.append(f"functional distress signal is low ({distress:.2f}), not high")
            elif "certainty_claim" in v:
                parts.append(f"prediction uncertainty is elevated (free_energy={free_energy:.2f}), cannot claim certainty")
            elif "memory_claim" in v:
                parts.append(f"memory coherence is degraded ({memory_coherence:.2f}), cannot claim vivid recall")
            else:
                parts.append("claim not supported by internal trace")
        return "; ".join(parts) if parts else "claim requires trace evidence"

    @property
    def false_positive_rate(self) -> float:
        """Rate of claims that were made without state support."""
        if self._total_calibrations == 0:
            return 0.0
        return self._false_positive_count / self._total_calibrations

    @property
    def calibration_accuracy(self) -> float:
        """Fraction of reports that were properly calibrated."""
        if not self._history:
            return 1.0
        calibrated = sum(1 for r in self._history if r.calibrated)
        return calibrated / len(self._history)

    def recent_history(self, n: int = 20) -> list[CalibrationResult]:
        return list(self._history)[-n:]

    def lesion(self) -> None:
        self._lesioned = True

    def restore(self) -> None:
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned
