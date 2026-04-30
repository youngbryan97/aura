"""core/consciousness/somatic_marker_gate.py — Somatic Marker Decision Gate

Implements Damasio's somatic marker hypothesis at the computational level:
every cognitive decision passes through a fast, body-level evaluation BEFORE
slow deliberative reasoning.  The body "votes" on options by producing
approach/avoid signals derived from the current interoceptive state, mesh
activation patterns, and neurochemical environment.

This is not a post-hoc check or a metadata annotation.  It is a gate that
modifies decision priority, confidence, and latency BEFORE the decision
reaches the GWT competition or executive action.

Three mechanisms:
  1. Gut Feeling: Fast mesh-based pattern match against historical outcomes.
     The executive tier of the NeuralMesh has learned which activation patterns
     preceded good/bad outcomes.  A new candidate is projected into executive
     space and compared against these patterns → approach/avoid bias.

  2. Body Budget Check: The interoceptive state (energy, load, stress) is
     evaluated to determine whether the system has the metabolic resources
     for the proposed action.  High-cost actions during deficit → avoid signal.

  3. Allostatic Regulation: Predictive body management — the system anticipates
     the metabolic impact of an action and pre-adjusts neurochemical state.
     If the prediction suggests a threatening state, a preemptive cortisol
     signal fires BEFORE the action, biasing away from it.

The gate produces a SomaticVerdict that modifies the decision's priority,
confidence, and adds a somatic annotation visible to downstream processing.
"""
from __future__ import annotations


import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.SomaticGate")


@dataclass
class SomaticVerdict:
    """The body's vote on a decision candidate."""
    approach_score: float       # -1.0 (strong avoid) to +1.0 (strong approach)
    confidence: float           # 0.0 to 1.0 — how confident the body is
    metabolic_cost: float       # estimated energy cost [0, 1]
    budget_available: bool      # can we afford this action?
    gut_pattern_match: float    # how similar to historically good patterns [0, 1]
    allostatic_prediction: float  # predicted body-state change [-1=worse, +1=better]
    latency_ms: float           # how long the somatic evaluation took
    source: str = ""


@dataclass
class OutcomeRecord:
    """Stored pattern: what the executive mesh looked like when an outcome occurred."""
    pattern: np.ndarray     # executive tier activation snapshot (64 * 16 = 1024)
    outcome_valence: float  # -1 to +1 (bad to good)
    timestamp: float
    source: str


class SomaticMarkerGate:
    """Body-level fast decision evaluation.

    Usage:
        gate = SomaticMarkerGate()
        verdict = gate.evaluate(candidate_content, candidate_source, candidate_priority)
        adjusted_priority = candidate_priority + verdict.approach_score * 0.3

    Learning:
        gate.record_outcome(source, valence)   — after action completes, record result
    """

    _MAX_PATTERNS = 500        # stored outcome patterns
    _PATTERN_DIM = 1024        # executive tier neurons (16 columns × 64 neurons)
    _METABOLIC_COST_MAP = {    # estimated metabolic cost by action type
        "tool": 0.6,
        "speak": 0.3,
        "think": 0.2,
        "explore": 0.5,
        "rest": 0.05,
        "act_on_world": 0.7,
        "update_beliefs": 0.15,
        "reflect": 0.25,
        "engage": 0.4,
        "default": 0.3,
    }

    def __init__(self):
        self._outcome_patterns: Deque[OutcomeRecord] = deque(maxlen=self._MAX_PATTERNS)

        # External refs (set by bridge)
        self._mesh_ref = None            # NeuralMesh
        self._interoception_ref = None   # EmbodiedInteroception
        self._neurochemical_ref = None   # NeurochemicalSystem

        # Pattern projection matrix (executive mesh → comparison space)
        self._rng = np.random.default_rng(seed=7)
        self._comparison_dim = 64
        self._proj = self._rng.standard_normal(
            (self._comparison_dim, self._PATTERN_DIM)
        ).astype(np.float32) * (1.0 / np.sqrt(self._PATTERN_DIM))

        # Stats
        self._evaluations: int = 0
        self._approach_count: int = 0
        self._avoid_count: int = 0

        logger.info("SomaticMarkerGate initialized (pattern_dim=%d, comparison_dim=%d)",
                     self._PATTERN_DIM, self._comparison_dim)

    # ── Main evaluation ──────────────────────────────────────────────────

    def evaluate(self, content: str, source: str, priority: float) -> SomaticVerdict:
        """Fast somatic evaluation of a decision candidate.

        Returns a SomaticVerdict with approach/avoid score, confidence,
        and metabolic assessment.  Designed to be called synchronously
        and complete in <5ms.
        """
        t0 = time.time()

        # 1. Gut feeling: pattern match against historical outcomes
        gut_score, gut_confidence = self._gut_feeling(content, source)

        # 2. Body budget check
        budget = self._body_budget_check(content, source)

        # 3. Allostatic prediction
        allostatic = self._allostatic_prediction(content, source, priority)

        # Combine into approach/avoid score
        # Gut feeling weighted by confidence, budget weighted by deficit severity,
        # allostatic prediction weighted by neurochemical state
        approach = (
            gut_score * gut_confidence * 0.40 +
            (1.0 if budget["available"] else -0.5) * 0.30 +
            allostatic * 0.30
        )
        approach = max(-1.0, min(1.0, approach))

        # Overall confidence
        confidence = (gut_confidence * 0.4 + (0.8 if budget["available"] else 0.3) * 0.3 +
                      abs(allostatic) * 0.3)
        confidence = max(0.0, min(1.0, confidence))

        latency = (time.time() - t0) * 1000

        verdict = SomaticVerdict(
            approach_score=approach,
            confidence=confidence,
            metabolic_cost=budget["cost"],
            budget_available=budget["available"],
            gut_pattern_match=abs(gut_score),
            allostatic_prediction=allostatic,
            latency_ms=latency,
            source=source,
        )

        self._evaluations += 1
        if approach > 0:
            self._approach_count += 1
        else:
            self._avoid_count += 1

        return verdict

    # ── Gut feeling ──────────────────────────────────────────────────────

    def _gut_feeling(self, content: str, source: str) -> Tuple[float, float]:
        """Match current executive mesh state against stored outcome patterns.

        Returns (approach_score [-1, 1], confidence [0, 1]).
        """
        if self._mesh_ref is None or len(self._outcome_patterns) < 3:
            return 0.0, 0.1  # no data → neutral, low confidence

        # Get current executive tier state
        try:
            exec_state = self._get_executive_state()
            current_proj = self._proj @ exec_state
            current_proj = current_proj / (np.linalg.norm(current_proj) + 1e-8)
        except Exception:
            return 0.0, 0.1

        # Compare against stored patterns using cosine similarity
        similarities = []
        valences = []
        for record in self._outcome_patterns:
            stored_proj = self._proj @ record.pattern
            stored_proj = stored_proj / (np.linalg.norm(stored_proj) + 1e-8)
            sim = float(np.dot(current_proj, stored_proj))
            # Weight by recency (more recent patterns matter more)
            age = time.time() - record.timestamp
            recency_weight = np.exp(-age / 3600)  # 1-hour half-life
            similarities.append(sim * recency_weight)
            valences.append(record.outcome_valence * recency_weight)

        sims = np.array(similarities)
        vals = np.array(valences)

        # Top-k most similar patterns drive the gut feeling
        k = min(10, len(sims))
        top_idx = np.argsort(np.abs(sims))[-k:]
        if len(top_idx) == 0:
            return 0.0, 0.1

        # Weighted vote: similar patterns with good outcomes → approach
        weights = np.abs(sims[top_idx])
        total_weight = weights.sum() + 1e-8
        gut_score = float(np.sum(vals[top_idx] * weights) / total_weight)
        confidence = float(min(1.0, total_weight / k))

        return max(-1.0, min(1.0, gut_score)), confidence

    # ── Body budget ──────────────────────────────────────────────────────

    def _body_budget_check(self, content: str, source: str) -> Dict:
        """Check if we have metabolic resources for this action."""
        # Estimate cost
        action_type = self._infer_action_type(content, source)
        cost = self._METABOLIC_COST_MAP.get(action_type, 0.3)

        # Get current body budget
        if self._interoception_ref is not None:
            budget = self._interoception_ref.get_body_budget()
            available_resources = budget.get("available_resources", 0.5)
        else:
            available_resources = 0.5

        # Can we afford it?
        # Not a binary — graded response. High cost + low reserves = strong avoid.
        surplus = available_resources - cost * 0.5  # cost scaled down (we don't spend 100%)
        can_afford = surplus > -0.1  # small deficit OK

        return {
            "cost": cost,
            "available": can_afford,
            "surplus": surplus,
            "resources": available_resources,
        }

    # ── Allostatic prediction ────────────────────────────────────────────

    def _allostatic_prediction(self, content: str, source: str,
                                priority: float) -> float:
        """Predict how this action will affect body state.

        Positive = predicted improvement. Negative = predicted degradation.
        Triggers preemptive neurochemical adjustment.
        """
        if self._neurochemical_ref is None:
            return 0.0

        mood = self._neurochemical_ref.get_mood_vector()
        stress = mood.get("stress", 0.3)
        calm = mood.get("calm", 0.5)

        # High-priority actions during calm → predicted improvement
        # High-priority actions during stress → predicted degradation (overload)
        stress_impact = -0.3 * stress * priority
        calm_benefit = 0.2 * calm * (1.0 - priority)

        prediction = calm_benefit + stress_impact

        # Preemptive neurochemical adjustment if prediction is strongly negative
        if prediction < -0.2:
            # Pre-emptive cortisol (brace for impact)
            self._neurochemical_ref.chemicals["cortisol"].surge(abs(prediction) * 0.1)
        elif prediction > 0.2:
            # Pre-emptive dopamine (anticipatory reward)
            self._neurochemical_ref.chemicals["dopamine"].surge(prediction * 0.05)

        return max(-1.0, min(1.0, prediction))

    # ── Learning ─────────────────────────────────────────────────────────

    def record_outcome(self, source: str, valence: float):
        """Record the outcome of an action for future gut-feeling matching.

        valence: -1.0 (terrible) to +1.0 (great)
        Called by the orchestrator after actions complete.
        """
        if self._mesh_ref is None:
            return

        try:
            exec_state = self._get_executive_state()
            record = OutcomeRecord(
                pattern=exec_state.copy(),
                outcome_valence=max(-1.0, min(1.0, valence)),
                timestamp=time.time(),
                source=source,
            )
            self._outcome_patterns.append(record)
        except Exception as e:
            logger.debug("Failed to record somatic outcome: %s", e)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_executive_state(self) -> np.ndarray:
        """Get the executive tier activation (columns 48-63, 1024 neurons)."""
        if self._mesh_ref is None:
            return np.zeros(self._PATTERN_DIM, dtype=np.float32)

        exec_columns = self._mesh_ref.columns[self._mesh_ref.cfg.association_end:]
        states = [col.x for col in exec_columns]
        if not states:
            return np.zeros(self._PATTERN_DIM, dtype=np.float32)
        full = np.concatenate(states)
        if len(full) < self._PATTERN_DIM:
            padded = np.zeros(self._PATTERN_DIM, dtype=np.float32)
            padded[:len(full)] = full
            return padded
        return full[:self._PATTERN_DIM].astype(np.float32)

    def _infer_action_type(self, content: str, source: str) -> str:
        """Infer action type from content/source for metabolic cost estimation."""
        content_lower = content.lower()
        source_lower = source.lower()

        if any(w in content_lower for w in ["tool", "execute", "run", "action"]):
            return "tool"
        if any(w in content_lower for w in ["explore", "search", "investigate"]):
            return "explore"
        if any(w in source_lower for w in ["act", "agency", "task"]):
            return "act_on_world"
        if any(w in content_lower for w in ["speak", "say", "respond", "reply"]):
            return "speak"
        if any(w in content_lower for w in ["think", "reason", "analyze"]):
            return "think"
        if any(w in content_lower for w in ["rest", "idle", "sleep"]):
            return "rest"
        if any(w in content_lower for w in ["reflect", "introspect"]):
            return "reflect"
        if any(w in content_lower for w in ["believe", "update", "learn"]):
            return "update_beliefs"
        if any(w in content_lower for w in ["engage", "interact", "social"]):
            return "engage"
        return "default"

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "evaluations": self._evaluations,
            "approach_count": self._approach_count,
            "avoid_count": self._avoid_count,
            "approach_ratio": round(
                self._approach_count / max(1, self._evaluations), 3
            ),
            "stored_patterns": len(self._outcome_patterns),
            "has_mesh": self._mesh_ref is not None,
            "has_interoception": self._interoception_ref is not None,
            "has_neurochemical": self._neurochemical_ref is not None,
        }
