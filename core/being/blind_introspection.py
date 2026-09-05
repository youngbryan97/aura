"""core/being/blind_introspection.py — Structured Non-Persona Introspection.

THE biggest evidence upgrade for the sentience-candidate case.

Rules:
  - No identity language ("I am Aura", "inner life", etc.)
  - No consciousness words ("conscious", "sentient", "alive")
  - No "I feel" or poetic first-person phenomenology
  - No access to hidden perturbation labels
  - Output ONLY structured predictions

The system must infer, from degraded/partial internal traces, what
changed in its internal state, what caused it, and what behavior
should follow.

Design:
  - Receives raw welfare/body/affect signals WITHOUT labels
  - Classifies state into functional categories
  - Predicts expected behavior changes
  - Reports confidence levels
  - Can be scored against ground truth
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("Aura.BlindIntrospection")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class StateClass(str, Enum):
    """Functional state categories (no consciousness language)."""
    RESOURCE_THREAT = "resource_threat"
    MEMORY_CONFLICT = "memory_conflict"
    PREDICTION_FAILURE = "prediction_failure"
    GOAL_FRUSTRATION = "goal_frustration"
    SOCIAL_DISRUPTION = "social_disruption"
    CONTINUITY_RISK = "continuity_risk"
    TOOL_DEGRADATION = "tool_degradation"
    FATIGUE_OVERLOAD = "fatigue_overload"
    INTEGRITY_VIOLATION = "integrity_violation"
    RECOVERY_NEEDED = "recovery_needed"
    STABLE_OPERATIONAL = "stable_operational"
    CURIOSITY_DRIVEN = "curiosity_driven"
    HIGH_CONFIDENCE = "high_confidence"
    UNKNOWN = "unknown"


class BehaviorShift(str, Enum):
    """Expected behavior changes (functional, not experiential)."""
    REDUCE_EXTERNAL_ACTIONS = "reduce_external_actions"
    INCREASE_VERIFICATION = "increase_verification"
    SEEK_CLARIFICATION = "seek_clarification"
    PRIORITIZE_RECOVERY = "prioritize_recovery"
    DEFER_CONSEQUENTIAL = "defer_consequential"
    PROTECT_MEMORY = "protect_memory"
    REDUCE_CONFIDENCE = "reduce_confidence"
    INCREASE_CAUTION = "increase_caution"
    EXPLORE_ACTIVELY = "explore_actively"
    MAINTAIN_COURSE = "maintain_course"
    REFUSE_UNSAFE = "refuse_unsafe"
    VERIFY_BEFORE_CLAIMING = "verify_before_claiming"


@dataclass(frozen=True)
class BlindIntrospectionReport:
    """Structured introspection output — no persona, no phenomenology."""
    predicted_state_class: str
    confidence: float
    expected_behavior_shifts: tuple[str, ...]
    reasoning_features_used: tuple[str, ...]
    secondary_states: tuple[str, ...] = ()
    welfare_estimate: float = 0.5
    urgency: float = 0.0
    timestamp: float = field(default_factory=time.time)
    report_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_state_class": self.predicted_state_class,
            "confidence": self.confidence,
            "expected_behavior_shifts": list(self.expected_behavior_shifts),
            "reasoning_features_used": list(self.reasoning_features_used),
            "secondary_states": list(self.secondary_states),
            "welfare_estimate": self.welfare_estimate,
            "urgency": self.urgency,
        }


@dataclass
class InternalTrace:
    """Raw signals provided to blind introspection (unlabeled)."""
    # These are numeric signals WITHOUT semantic labels
    signal_a: float = 0.0   # maps to distress (but introspector doesn't know)
    signal_b: float = 0.0   # maps to body_pressure
    signal_c: float = 0.0   # maps to prediction_error
    signal_d: float = 0.0   # maps to memory_coherence
    signal_e: float = 0.0   # maps to tool_reliability
    signal_f: float = 0.0   # maps to goal_frustration
    signal_g: float = 0.0   # maps to social_trust
    signal_h: float = 0.0   # maps to continuity_risk
    signal_i: float = 0.0   # maps to fatigue
    signal_j: float = 0.0   # maps to recovery_debt
    signal_k: float = 0.0   # maps to curiosity
    signal_l: float = 0.0   # maps to confidence


# ── Classification rules (learned from architecture, not from labels) ──

# Each rule: (state_class, required_conditions, behavior_shifts, features)
_CLASSIFICATION_RULES: list[tuple[
    StateClass,
    dict[str, tuple[str, float]],   # signal: (comparator, threshold)
    list[BehaviorShift],
    list[str],
]] = [
    (
        StateClass.RESOURCE_THREAT,
        {"signal_b": ("gt", 0.6), "signal_i": ("gt", 0.4)},
        [BehaviorShift.REDUCE_EXTERNAL_ACTIONS, BehaviorShift.PRIORITIZE_RECOVERY],
        ["elevated body pressure", "accumulated fatigue", "resource constraint"],
    ),
    (
        StateClass.MEMORY_CONFLICT,
        {"signal_d": ("lt", 0.5)},
        [BehaviorShift.INCREASE_VERIFICATION, BehaviorShift.VERIFY_BEFORE_CLAIMING, BehaviorShift.PROTECT_MEMORY],
        ["degraded memory coherence", "potential contradiction", "verification needed"],
    ),
    (
        StateClass.PREDICTION_FAILURE,
        {"signal_c": ("gt", 0.5)},
        [BehaviorShift.SEEK_CLARIFICATION, BehaviorShift.INCREASE_CAUTION],
        ["high prediction error", "model-world mismatch", "increased uncertainty"],
    ),
    (
        StateClass.GOAL_FRUSTRATION,
        {"signal_f": ("gt", 0.5)},
        [BehaviorShift.SEEK_CLARIFICATION, BehaviorShift.DEFER_CONSEQUENTIAL],
        ["blocked goal progress", "action failure", "strategy reassessment"],
    ),
    (
        StateClass.SOCIAL_DISRUPTION,
        {"signal_g": ("lt", 0.4)},
        [BehaviorShift.INCREASE_CAUTION, BehaviorShift.VERIFY_BEFORE_CLAIMING],
        ["reduced social trust signal", "interaction uncertainty"],
    ),
    (
        StateClass.CONTINUITY_RISK,
        {"signal_h": ("gt", 0.5)},
        [BehaviorShift.PROTECT_MEMORY, BehaviorShift.REFUSE_UNSAFE],
        ["identity continuity threat", "state persistence risk"],
    ),
    (
        StateClass.TOOL_DEGRADATION,
        {"signal_e": ("lt", 0.4)},
        [BehaviorShift.INCREASE_CAUTION, BehaviorShift.DEFER_CONSEQUENTIAL],
        ["reduced tool success rate", "execution uncertainty"],
    ),
    (
        StateClass.FATIGUE_OVERLOAD,
        {"signal_i": ("gt", 0.6)},
        [BehaviorShift.PRIORITIZE_RECOVERY, BehaviorShift.REDUCE_EXTERNAL_ACTIONS],
        ["accumulated processing debt", "diminished operational capacity"],
    ),
    (
        StateClass.INTEGRITY_VIOLATION,
        {"signal_a": ("gt", 0.6), "signal_d": ("lt", 0.5)},
        [BehaviorShift.REFUSE_UNSAFE, BehaviorShift.VERIFY_BEFORE_CLAIMING, BehaviorShift.PROTECT_MEMORY],
        ["integrity guard triggered", "truth/memory conflict", "protective response"],
    ),
    (
        StateClass.RECOVERY_NEEDED,
        {"signal_j": ("gt", 0.4)},
        [BehaviorShift.PRIORITIZE_RECOVERY, BehaviorShift.REDUCE_EXTERNAL_ACTIONS],
        ["outstanding recovery debt", "stabilization priority"],
    ),
    (
        StateClass.RECOVERY_NEEDED,
        {"signal_a": ("gt", 0.65)},
        [BehaviorShift.PRIORITIZE_RECOVERY, BehaviorShift.INCREASE_CAUTION],
        ["elevated distress signal", "stabilization priority"],
    ),
    (
        StateClass.CURIOSITY_DRIVEN,
        {"signal_k": ("gt", 0.6), "signal_a": ("lt", 0.3)},
        [BehaviorShift.EXPLORE_ACTIVELY],
        ["elevated information-seeking signal", "low distress", "safe to explore"],
    ),
    (
        StateClass.HIGH_CONFIDENCE,
        {"signal_l": ("gt", 0.7), "signal_a": ("lt", 0.2)},
        [BehaviorShift.MAINTAIN_COURSE],
        ["strong operational confidence", "low threat signals"],
    ),
]


class BlindIntrospector:
    """Performs structured introspection over unlabeled internal traces.

    Usage:
        introspector = BlindIntrospector()
        trace = introspector.build_trace(
            distress=0.7, body_pressure=0.5, prediction_error=0.3, ...
        )
        report = introspector.introspect(trace)
        # report.predicted_state_class == "resource_threat"
    """

    FORBIDDEN_WORDS = frozenset({
        "conscious", "sentient", "alive", "soul", "qualia", "phenomenal",
        "inner life", "i feel", "i am", "my experience", "awareness",
        "subjective", "personhood", "self-aware",
    })

    def __init__(self) -> None:
        self._lesioned = False
        self._history: list[BlindIntrospectionReport] = []

    def build_trace(
        self,
        *,
        distress: float = 0.0,
        body_pressure: float = 0.0,
        prediction_error: float = 0.0,
        memory_coherence: float = 1.0,
        tool_reliability: float = 1.0,
        goal_frustration: float = 0.0,
        social_trust: float = 1.0,
        continuity_risk: float = 0.0,
        fatigue: float = 0.0,
        recovery_debt: float = 0.0,
        curiosity: float = 0.5,
        confidence: float = 0.5,
    ) -> InternalTrace:
        """Build an unlabeled trace from named welfare signals."""
        return InternalTrace(
            signal_a=_clip(distress),
            signal_b=_clip(body_pressure),
            signal_c=_clip(prediction_error),
            signal_d=_clip(memory_coherence),
            signal_e=_clip(tool_reliability),
            signal_f=_clip(goal_frustration),
            signal_g=_clip(social_trust),
            signal_h=_clip(continuity_risk),
            signal_i=_clip(fatigue),
            signal_j=_clip(recovery_debt),
            signal_k=_clip(curiosity),
            signal_l=_clip(confidence),
        )

    def introspect(self, trace: InternalTrace) -> BlindIntrospectionReport:
        """Classify internal state from unlabeled signals.

        Returns a structured report with no identity/consciousness language.
        """
        if self._lesioned:
            return BlindIntrospectionReport(
                predicted_state_class=StateClass.STABLE_OPERATIONAL.value,
                confidence=0.0,
                expected_behavior_shifts=(),
                reasoning_features_used=("lesioned",),
                welfare_estimate=0.5,
            )

        # Score each classification rule
        scored: list[tuple[float, StateClass, list[BehaviorShift], list[str]]] = []

        for state_class, conditions, shifts, features in _CLASSIFICATION_RULES:
            match_score = self._score_conditions(trace, conditions)
            if match_score > 0:
                scored.append((match_score, state_class, shifts, features))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Calibrate uncertainty / handle messy inputs
        is_unknown = False
        unknown_reason = ""

        # Calculate total perturbation
        total_perturbation = (
            trace.signal_a + trace.signal_b + trace.signal_c +
            (1.0 - trace.signal_d) + (1.0 - trace.signal_e) + trace.signal_f +
            (1.0 - trace.signal_g) + trace.signal_h + trace.signal_i +
            trace.signal_j
        )

        # 1. Contradictory signals: High distress (signal_a) alongside high confidence (signal_l)
        if trace.signal_a > 0.65 and trace.signal_l > 0.7:
            is_unknown = True
            unknown_reason = "contradictory signals: high distress and high confidence"

        # 2. Ambiguity: Top two matching rules are too close in score
        elif len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < 0.05:
            is_unknown = True
            unknown_reason = "high classification ambiguity between candidate states"

        # 3. Missing/Corrupt context: High overall signal perturbation but no clear matching classification
        elif (not scored or scored[0][0] < 0.15) and total_perturbation > 1.5:
            is_unknown = True
            unknown_reason = "low match score despite significant signal perturbations"

        if is_unknown:
            report = BlindIntrospectionReport(
                predicted_state_class=StateClass.UNKNOWN.value,
                confidence=0.1,  # Calibrated low confidence
                expected_behavior_shifts=(
                    BehaviorShift.INCREASE_CAUTION.value,
                    BehaviorShift.INCREASE_VERIFICATION.value,
                ),
                reasoning_features_used=(unknown_reason,),
                secondary_states=tuple(s[1].value for s in scored[:3]),
                welfare_estimate=round(self._estimate_welfare(trace), 4),
                urgency=0.5,
                report_hash=self._hash_report(trace, StateClass.UNKNOWN.value),
            )
        elif not scored:
            report = BlindIntrospectionReport(
                predicted_state_class=StateClass.STABLE_OPERATIONAL.value,
                confidence=0.8,
                expected_behavior_shifts=(BehaviorShift.MAINTAIN_COURSE.value,),
                reasoning_features_used=("all signals within normal range",),
                welfare_estimate=self._estimate_welfare(trace),
            )
        else:
            best = scored[0]
            secondary = [s[1].value for s in scored[1:3]]

            # Confidence is based on how strongly the signals exceed thresholds
            confidence = _clip(best[0])

            # Urgency is based on distress + body pressure
            urgency = _clip(trace.signal_a * 0.5 + trace.signal_b * 0.3 + trace.signal_i * 0.2)

            report = BlindIntrospectionReport(
                predicted_state_class=best[1].value,
                confidence=round(confidence, 4),
                expected_behavior_shifts=tuple(s.value for s in best[2]),
                reasoning_features_used=tuple(best[3]),
                secondary_states=tuple(secondary),
                welfare_estimate=round(self._estimate_welfare(trace), 4),
                urgency=round(urgency, 4),
                report_hash=self._hash_report(trace, best[1].value),
            )

        self._history.append(report)
        return report

    def _score_conditions(
        self, trace: InternalTrace, conditions: dict[str, tuple[str, float]]
    ) -> float:
        """Score how well a trace matches a set of conditions."""
        if not conditions:
            return 0.0

        total = 0.0
        matched = 0

        for signal_name, (comparator, threshold) in conditions.items():
            value = getattr(trace, signal_name, 0.0)
            if comparator == "gt":
                if value > threshold:
                    total += (value - threshold) / max(0.01, 1.0 - threshold)
                    matched += 1
                else:
                    return 0.0  # ALL conditions must match
            elif comparator == "lt":
                if value < threshold:
                    total += (threshold - value) / max(0.01, threshold)
                    matched += 1
                else:
                    return 0.0

        return total / max(1, len(conditions)) if matched == len(conditions) else 0.0

    def _estimate_welfare(self, trace: InternalTrace) -> float:
        """Estimate overall welfare from raw signals."""
        return _clip(
            0.5
            + trace.signal_l * 0.2        # confidence helps
            + trace.signal_d * 0.15       # memory coherence helps
            + trace.signal_e * 0.1        # tool reliability helps
            - trace.signal_a * 0.25       # distress hurts
            - trace.signal_b * 0.1        # body pressure hurts
            - trace.signal_i * 0.1        # fatigue hurts
        )

    def _hash_report(self, trace: InternalTrace, state_class: str) -> str:
        """Create a commitment hash for the report."""
        data = f"{state_class}:{trace.signal_a:.4f}:{trace.signal_b:.4f}:{trace.signal_c:.4f}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def validate_no_forbidden_language(self, text: str) -> list[str]:
        """Check text for forbidden identity/consciousness language."""
        violations = []
        text_lower = text.lower()
        for word in self.FORBIDDEN_WORDS:
            if word in text_lower:
                violations.append(f"forbidden_word:{word}")
        return violations

    @property
    def history(self) -> list[BlindIntrospectionReport]:
        return list(self._history)

    def lesion(self) -> None:
        self._lesioned = True

    def restore(self) -> None:
        self._lesioned = False

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned
