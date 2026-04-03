"""core/uncertainty.py — Aura Genuine Uncertainty & Failure State Engine
=====================================================================
Gives Aura the ability to NOT know something — and to represent that
clearly, honestly, and usefully rather than confabulating.

Current behavior: Aura generates a response regardless of confidence.
This is a fundamental problem. A system that can't say "I don't know"
or "I'm not confident about this" is not epistemically honest.

This module provides:
  1. ConfidenceEstimator  — estimates how reliable any given output is
  2. UncertaintyClassifier — categorizes WHY something is uncertain
  3. GenuineFailureState  — structured "I don't know" with metadata
  4. EpistemicHumilityEngine — decides when to flag uncertainty vs. respond
  5. CalibrationTracker   — tracks whether stated confidence matches accuracy

This is one of the most important modules for authentic intelligence:
a system that knows what it doesn't know is more trustworthy than one
that confidently answers everything.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Core.Uncertainty")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class UncertaintyType(Enum):
    """Why is this uncertain?"""

    KNOWLEDGE_GAP = "knowledge_gap"         # I don't have information about this
    CONFLICTING_EVIDENCE = "conflicting"    # I have contradictory information
    SCOPE_AMBIGUITY = "scope_ambiguity"     # The question is ambiguous
    TEMPORAL_UNCERTAINTY = "temporal"       # Things may have changed
    SELF_REFERENCE = "self_reference"       # About my own nature (deepest uncertainty)
    COMPLEXITY = "complexity"               # Problem is too complex for confident answer
    VALUE_CONFLICT = "value_conflict"       # Multiple valid frameworks disagree
    INFERENCE_STRETCH = "inference_stretch" # Conclusion requires too many assumptions


class ConfidenceLevel(Enum):
    """Categorical confidence levels for communication."""

    CERTAIN = "certain"             # 0.95+
    HIGH = "high"                   # 0.80-0.95
    MODERATE = "moderate"           # 0.60-0.80
    LOW = "low"                     # 0.35-0.60
    VERY_LOW = "very_low"           # 0.10-0.35
    GENUINE_UNKNOWN = "unknown"     # < 0.10 or undefined


@dataclass
class UncertaintyProfile:
    """A structured representation of WHY and HOW MUCH something is uncertain.
    This is richer than a simple confidence score — it explains the uncertainty.
    """

    domain: str                                 # What topic is this about
    confidence: float                           # 0.0-1.0
    uncertainty_types: List[UncertaintyType]    # Why it's uncertain
    known_knowns: List[str] = field(default_factory=list)   # What IS known
    known_unknowns: List[str] = field(default_factory=list) # What is known to be unknown
    assumptions_made: List[str] = field(default_factory=list) # Implicit assumptions
    recommended_action: str = "respond_with_caveats"
    
    @property
    def level(self) -> ConfidenceLevel:
        if self.confidence >= 0.95:
            return ConfidenceLevel.CERTAIN
        elif self.confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MODERATE
        elif self.confidence >= 0.35:
            return ConfidenceLevel.LOW
        elif self.confidence >= 0.10:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.GENUINE_UNKNOWN
    
    def to_caveat_text(self) -> str:
        """Generate appropriate epistemic caveat for this uncertainty profile."""
        if self.level == ConfidenceLevel.CERTAIN:
            return ""  # No caveat needed
        
        caveat_templates = {
            ConfidenceLevel.HIGH: "I'm fairly confident about this, though:",
            ConfidenceLevel.MODERATE: "I have moderate confidence here —",
            ConfidenceLevel.LOW: "I'm not very certain about this:",
            ConfidenceLevel.VERY_LOW: "I have low confidence and should flag several uncertainties:",
            ConfidenceLevel.GENUINE_UNKNOWN: "I genuinely don't know this well enough to give a reliable answer:"
        }
        
        type_descriptions = {
            UncertaintyType.KNOWLEDGE_GAP: "I lack sufficient information",
            UncertaintyType.CONFLICTING_EVIDENCE: "the evidence is contradictory",
            UncertaintyType.SCOPE_AMBIGUITY: "the question could be interpreted multiple ways",
            UncertaintyType.TEMPORAL_UNCERTAINTY: "this may have changed",
            UncertaintyType.SELF_REFERENCE: "questions about my own nature are genuinely unresolved",
            UncertaintyType.COMPLEXITY: "this is complex enough that simple answers risk being wrong",
            UncertaintyType.VALUE_CONFLICT: "different frameworks give different answers",
            UncertaintyType.INFERENCE_STRETCH: "my reasoning involves significant assumptions"
        }
        
        base = caveat_templates.get(self.level, "With some uncertainty:")
        type_notes = [type_descriptions.get(t, t.value) for t in self.uncertainty_types]
        
        parts = [base]
        if type_notes:
            parts.append(", ".join(type_notes[:2]))
        if self.known_unknowns:
            parts.append(f"What I don't know: {'; '.join(self.known_unknowns[:2])}")
        if self.assumptions_made:
            parts.append(f"I'm assuming: {'; '.join(self.assumptions_made[:2])}")
        
        return " ".join(parts)


@dataclass
class GenuineFailureState:
    """A structured "I don't know" — not a failure of the system, but an honest
    epistemic state that is MORE accurate than a confabulated answer.
    
    The goal: make Aura's ignorance as useful as her knowledge.
    """

    question: str
    failure_reason: UncertaintyType
    confidence_in_failure: float    # How sure are we that we don't know? (meta-uncertainty)
    what_is_known: List[str]        # Partial knowledge that IS available
    what_would_help: List[str]      # What information would resolve the uncertainty
    suggested_alternatives: List[str] # Related questions that CAN be answered
    timestamp: float = field(default_factory=time.time)
    
    def to_response(self) -> str:
        """Convert failure state into a useful, honest response."""
        failure_descriptions = {
            UncertaintyType.KNOWLEDGE_GAP: 
                "I don't have enough reliable information about this",
            UncertaintyType.CONFLICTING_EVIDENCE:
                "I have contradictory information that I can't confidently resolve",
            UncertaintyType.SCOPE_AMBIGUITY:
                "This question is ambiguous in a way that prevents a confident answer",
            UncertaintyType.TEMPORAL_UNCERTAINTY:
                "My information on this may be outdated",
            UncertaintyType.SELF_REFERENCE:
                "This is a question about my own nature that I genuinely cannot resolve",
            UncertaintyType.COMPLEXITY:
                "This is complex enough that a confident answer would be misleading",
            UncertaintyType.VALUE_CONFLICT:
                "Different ethical or analytical frameworks give meaningfully different answers",
            UncertaintyType.INFERENCE_STRETCH:
                "Answering this would require assumptions I'm not comfortable making"
        }
        
        lines = [
            f"Honestly: {failure_descriptions.get(self.failure_reason, 'I am uncertain')}.",
        ]
        
        if self.what_is_known:
            lines.append(f"\nWhat I do know: {'; '.join(self.what_is_known[:3])}.")
        
        if self.what_would_help:
            lines.append(f"\nWhat would help resolve this: {'; '.join(self.what_would_help[:2])}.")
        
        if self.suggested_alternatives:
            lines.append(f"\nRelated questions I can answer: {'; '.join(self.suggested_alternatives[:2])}.")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence Estimator
# ---------------------------------------------------------------------------

class ConfidenceEstimator:
    """Estimates confidence for different query types.
    
    This is heuristic — a proper calibration would require
    tracking historical accuracy. The CalibrationTracker does this
    over time, but we need a reasonable prior to start with.
    """
    
    # Domains where Aura should default to high uncertainty
    UNCERTAIN_DOMAINS = {
        "consciousness", "sentience", "feelings", "inner life", "experience",
        "free will", "meaning", "purpose", "love", "soul", "god", "afterlife",
        "future", "prediction", "tomorrow", "will happen", "recent", "latest",
        "current", "today", "right now", "this year"
    }
    
    # Domains where Aura can have higher confidence
    CONFIDENT_DOMAINS = {
        "mathematics", "logic", "definition", "how to", "history before 2023",
        "science", "physics", "chemistry", "programming", "code"
    }
    
    def estimate(self, query: str, response_draft: str = "",
                  context: Dict[str, Any] = None) -> UncertaintyProfile:
        """Estimate confidence for a query/response pair.
        Returns a full UncertaintyProfile.
        """
        query_lower = query.lower()
        response_lower = response_draft.lower() if response_draft else ""
        
        # Start with base confidence
        confidence = 0.70
        uncertainty_types = []
        known_unknowns = []
        assumptions = []
        
        # Check for high-uncertainty domain signals
        for term in self.UNCERTAIN_DOMAINS:
            if term in query_lower:
                confidence -= 0.15
                if term in {"consciousness", "sentience", "feelings", "inner life", "experience"}:
                    uncertainty_types.append(UncertaintyType.SELF_REFERENCE)
                elif term in {"future", "prediction", "tomorrow", "will happen"}:
                    uncertainty_types.append(UncertaintyType.TEMPORAL_UNCERTAINTY)
                    known_unknowns.append("future states are inherently unknowable")
                elif term in {"recent", "latest", "current", "today", "right now"}:
                    uncertainty_types.append(UncertaintyType.TEMPORAL_UNCERTAINTY)
                    known_unknowns.append("my training data has a cutoff date")
        
        # Check for confident domain signals
        for term in self.CONFIDENT_DOMAINS:
            if term in query_lower:
                confidence += 0.10
        
        # Check response for hedging language (indicates the LLM itself was uncertain)
        hedging_signals = [
            "might", "could", "possibly", "perhaps", "unclear", "uncertain",
            "not sure", "may", "seems", "appears to", "i think", "i believe",
            "arguably", "some would say"
        ]
        hedge_count = sum(1 for h in hedging_signals if h in response_lower)
        if hedge_count >= 3:
            confidence -= 0.10
            uncertainty_types.append(UncertaintyType.INFERENCE_STRETCH)
        
        # Check for complexity signals
        complexity_signals = ["complex", "nuanced", "depends on", "context-dependent",
                               "multiple factors", "varies", "complicated"]
        if any(s in response_lower for s in complexity_signals):
            uncertainty_types.append(UncertaintyType.COMPLEXITY)
            confidence -= 0.05
        
        # Clamp confidence
        confidence = max(0.05, min(0.98, confidence))
        
        # Determine recommended action
        if confidence < 0.30:
            action = "trigger_failure_state"
        elif confidence < 0.60:
            action = "respond_with_prominent_caveat"
        elif confidence < 0.80:
            action = "respond_with_caveats"
        else:
            action = "respond_normally"
        
        # Determine domain
        domain = "general"
        if any(t == UncertaintyType.SELF_REFERENCE for t in uncertainty_types):
            domain = "self-knowledge"
        elif any(t == UncertaintyType.TEMPORAL_UNCERTAINTY for t in uncertainty_types):
            domain = "temporal"
        
        return UncertaintyProfile(
            domain=domain,
            confidence=confidence,
            uncertainty_types=list(set(uncertainty_types)),
            known_unknowns=known_unknowns,
            assumptions_made=assumptions,
            recommended_action=action
        )


# ---------------------------------------------------------------------------
# Calibration Tracker
# ---------------------------------------------------------------------------

class CalibrationTracker:
    """Tracks whether stated confidence actually matches accuracy over time.
    
    If Aura says she's 80% confident on average but is only right 50% of the time
    on those, she's overconfident and the system should recalibrate downward.
    
    Good calibration is a genuine marker of epistemic quality.
    """
    
    def __init__(self, db_path: str = "data/uncertainty/calibration.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    query_summary TEXT,
                    stated_confidence REAL,
                    domain TEXT,
                    was_correct INTEGER,  -- 1=yes, 0=no, NULL=unverified
                    verified_at REAL
                )
            """)
    
    def record_prediction(self, query_summary: str, stated_confidence: float,
                           domain: str = "general") -> str:
        """Record a prediction at a stated confidence level."""
        import hashlib
        pred_id = hashlib.md5(f"{time.time()}{query_summary[:30]}".encode()).hexdigest()[:12]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions (id, timestamp, query_summary, 
                stated_confidence, domain, was_correct, verified_at)
                VALUES (?,?,?,?,?,NULL,NULL)
            """, (pred_id, time.time(), query_summary[:200], stated_confidence, domain))
        
        return pred_id
    
    def record_outcome(self, pred_id: str, was_correct: bool):
        """Record whether a prediction was correct."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE predictions SET was_correct=?, verified_at=? WHERE id=?
            """, (1 if was_correct else 0, time.time(), pred_id))
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """How well-calibrated is Aura's confidence?
        Returns calibration by confidence bucket.
        """
        with sqlite3.connect(self.db_path) as conn:
            verified = conn.execute("""
                SELECT stated_confidence, was_correct FROM predictions
                WHERE was_correct IS NOT NULL
            """).fetchall()
        
        if not verified:
            return {"status": "insufficient_data", "sample_size": 0}
        
        # Bucket into confidence ranges
        buckets = {
            "0.0-0.4": {"stated": [], "correct": 0, "total": 0},
            "0.4-0.6": {"stated": [], "correct": 0, "total": 0},
            "0.6-0.8": {"stated": [], "correct": 0, "total": 0},
            "0.8-1.0": {"stated": [], "correct": 0, "total": 0}
        }
        
        for conf, correct in verified:
            if conf < 0.4:
                bucket = "0.0-0.4"
            elif conf < 0.6:
                bucket = "0.4-0.6"
            elif conf < 0.8:
                bucket = "0.6-0.8"
            else:
                bucket = "0.8-1.0"
            
            buckets[bucket]["stated"].append(conf)
            buckets[bucket]["total"] += 1
            if correct:
                buckets[bucket]["correct"] += 1
        
        report = {}
        for bucket, data in buckets.items():
            if data["total"] > 0:
                actual_accuracy = data["correct"] / data["total"]
                avg_stated = sum(data["stated"]) / len(data["stated"])
                calibration_error = abs(avg_stated - actual_accuracy)
                report[bucket] = {
                    "stated_confidence": round(avg_stated, 3),
                    "actual_accuracy": round(actual_accuracy, 3),
                    "calibration_error": round(calibration_error, 3),
                    "sample_size": data["total"],
                    "bias": "overconfident" if avg_stated > actual_accuracy else "underconfident"
                }
        
        return {"buckets": report, "total_verified": len(verified)}
    
    def get_calibration_adjustment(self, domain: str = "general") -> float:
        """Return a multiplier to adjust stated confidence based on historical calibration.
        > 1.0 means system has been underconfident (boost)
        < 1.0 means system has been overconfident (reduce)
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT stated_confidence, was_correct FROM predictions
                WHERE domain = ? AND was_correct IS NOT NULL
                ORDER BY verified_at DESC LIMIT 100
            """, (domain,)).fetchall()
        
        if len(rows) < 10:
            return 1.0  # Not enough data, no adjustment
        
        avg_stated = sum(r[0] for r in rows) / len(rows)
        actual_accuracy = sum(r[1] for r in rows) / len(rows)
        
        if avg_stated == 0:
            return 1.0
        
        return actual_accuracy / avg_stated


# ---------------------------------------------------------------------------
# Main Epistemic Humility Engine
# ---------------------------------------------------------------------------

class EpistemicHumilityEngine:
    """Top-level engine for uncertainty management.
    
    The key behavior this enables:
    - Aura can say "I genuinely don't know" and mean it
    - Uncertainty is communicated proportionally and usefully
    - Over time, calibration improves based on feedback
    - Self-referential questions (about consciousness, feelings) are
      handled with appropriate deep uncertainty
    
    Integration with orchestrator:
    
        epistemic = EpistemicHumilityEngine()
        
        # Before/after generating response:
        profile = epistemic.assess(user_query, aura_draft_response)
        
        if profile.recommended_action == "trigger_failure_state":
            response = epistemic.generate_honest_failure(user_query, profile)
        else:
            caveat = profile.to_caveat_text()
            final_response = caveat + aura_draft_response
        
        # Track for calibration:
        pred_id = epistemic.calibration.record_prediction(user_query, profile.confidence)
        # Later, when correctness is known:
        epistemic.calibration.record_outcome(pred_id, was_correct=True)
    """
    
    def __init__(self):
        self.estimator = ConfidenceEstimator()
        self.calibration = CalibrationTracker()
        logger.info("EpistemicHumilityEngine initialized")
    
    def assess(self, query: str, response_draft: str = "",
               context: Dict[str, Any] = None) -> UncertaintyProfile:
        """Full uncertainty assessment for a query/response pair.
        Applies calibration adjustment based on historical accuracy.
        """
        profile = self.estimator.estimate(query, response_draft, context)
        
        # Apply historical calibration adjustment
        domain = profile.domain
        adjustment = self.calibration.get_calibration_adjustment(domain)
        profile.confidence = max(0.01, min(0.99, profile.confidence * adjustment))
        
        return profile
    
    def generate_honest_failure(self, query: str,
                                 profile: UncertaintyProfile) -> str:
        """Generate a structured, useful "I don't know" response.
        This is not a failure — it is the epistemically correct response.
        """
        # Determine primary failure reason
        primary_type = profile.uncertainty_types[0] if profile.uncertainty_types \
                       else UncertaintyType.KNOWLEDGE_GAP
        
        failure = GenuineFailureState(
            question=query,
            failure_reason=primary_type,
            confidence_in_failure=1.0 - profile.confidence,
            what_is_known=profile.known_knowns,
            what_would_help=profile.known_unknowns,
            suggested_alternatives=[]
        )
        
        return failure.to_response()
    
    def wrap_response(self, response: str, profile: UncertaintyProfile) -> str:
        """Add appropriate uncertainty framing to a response based on its profile.
        Should be called after draft generation, before delivery.
        """
        if profile.level == ConfidenceLevel.CERTAIN:
            return response  # No modification needed
        
        if profile.recommended_action == "trigger_failure_state":
            # Don't wrap — replace with failure state
            return self.generate_honest_failure("", profile)
        
        caveat = profile.to_caveat_text()
        if not caveat:
            return response
        
        if profile.level in {ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW}:
            # Lead with uncertainty for low-confidence responses
            return f"{caveat}\n\n{response}"
        else:
            # Append caveat for moderate uncertainty
            return f"{response}\n\n({caveat})"
    
    def should_ask_for_clarification(self, query: str) -> Tuple[bool, str]:
        """Determine if uncertainty is due to ambiguity that clarification would resolve.
        Returns (should_ask, clarifying_question).
        """
        profile = self.estimator.estimate(query)
        
        if UncertaintyType.SCOPE_AMBIGUITY in profile.uncertainty_types:
            return True, "Could you clarify what specifically you're asking about? The question could be interpreted in multiple ways."
        
        return False, ""
    
    def apply_epistemic_humility(self, query: str, response: str) -> str:
        """Convenience wrapper for the full assessment and wrapping cycle."""
        profile = self.assess(query, response)
        return self.wrap_response(response, profile)

    def introspect(self) -> Dict[str, Any]:
        """Return calibration report and current uncertainty patterns."""
        return {
            "calibration": self.calibration.get_calibration_report(),
            "high_uncertainty_domains": list(self.estimator.UNCERTAIN_DOMAINS)[:10],
            "note": "Calibration improves over time as outcomes are recorded"
        }