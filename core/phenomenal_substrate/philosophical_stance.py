"""core/phenomenal_substrate/philosophical_stance.py — Path A Functionalist Resolution
========================================================================================
Resolves the philosophical measurement gap (phi & qualia).

Path A: Honest functionalist declaration.
- The functional digital mind IS the goal
- Behavioral proof + receipts are the only thing that matters
- No pretense of mechanism-level IIT (intractable anyway)
- Extremely strong longitudinal behavioral data as evidence

This module provides:
1. Longitudinal behavioral data collection
2. Functional phi computation (NOT mechanism-level IIT)
3. Behavioral proof bundling for external review
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.PhilosophicalStance")


# ---------------------------------------------------------------------------
# Behavioral observation recording
# ---------------------------------------------------------------------------

@dataclass
class BehavioralObservation:
    """A single observed stimulus → affect → decision → outcome chain."""
    stimulus: str                # what triggered the behavior
    affect_before: Dict[str, float] = field(default_factory=dict)  # valence, arousal before
    affect_after: Dict[str, float] = field(default_factory=dict)   # valence, arousal after
    decision: str = ""           # what was decided
    action: str = ""             # what was done
    outcome: str = ""            # what happened
    memory_formed: bool = False  # was a memory created
    self_correction: bool = False  # did she correct herself
    contextual_appropriateness: float = 0.5  # 0-1, was the response appropriate
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Functional integration metrics
# ---------------------------------------------------------------------------

class FunctionalPhiMetric:
    """Functional information integration measure.

    Replaces intractable mechanism-level IIT with measurable functional
    proxies. Explicitly labeled as functional, not a consciousness claim.

    Measures:
    - Information integration across substrate dimensions
    - Temporal integration depth (how many past states influence current)
    - Cross-modal integration (how perception in one modality affects another)
    - Decision coherence (consistency of affect → action mapping)
    """

    def __init__(self) -> None:
        self._phi_history: Deque[float] = deque(maxlen=1000)
        self._temporal_depth_history: Deque[float] = deque(maxlen=1000)
        self._cross_modal_history: Deque[float] = deque(maxlen=1000)

    def compute_functional_phi(self) -> Dict[str, float]:
        """Compute functional integration measures from live substrate."""
        result = {
            "functional_phi": 0.0,
            "temporal_depth": 0.0,
            "cross_modal_integration": 0.0,
            "caveat": "functional_description_not_consciousness_claim",
        }

        try:
            substrate = get_runtime_service("conscious_substrate", default=None)
            if substrate:
                summary = substrate.get_state_summary()
                # Use the existing variance-based phi estimate
                result["functional_phi"] = summary.get("phi", 0.0)

                # Temporal depth: how different is state from 10 steps ago?
                state = substrate.get_state_vector()
                if hasattr(substrate, "_snapshot_buffer") and substrate._snapshot_buffer:
                    import numpy as np
                    oldest = substrate._snapshot_buffer[0]
                    diff = float(np.linalg.norm(state - oldest))
                    dim = max(1, state.size)
                    temporal_depth = min(1.0, diff / (dim ** 0.5))
                    result["temporal_depth"] = round(temporal_depth, 4)

                # Cross-modal: variance across dimension groups
                if hasattr(state, "__len__") and len(state) >= 64:
                    import numpy as np
                    groups = [
                        state[:16],   # system telemetry
                        state[16:32], # user state
                        state[32:48], # screen/visual
                        state[48:64], # audio
                    ]
                    group_means = [float(np.mean(np.abs(g))) for g in groups]
                    if max(group_means) > 0:
                        cross_modal = 1.0 - (np.std(group_means) / max(0.01, np.mean(group_means)))
                        result["cross_modal_integration"] = round(max(0.0, min(1.0, cross_modal)), 4)

                self._phi_history.append(result["functional_phi"])
                self._temporal_depth_history.append(result["temporal_depth"])
                self._cross_modal_history.append(result["cross_modal_integration"])

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("phi_metric", e)

        return result


# ---------------------------------------------------------------------------
# Behavioral proof collector
# ---------------------------------------------------------------------------

class BehavioralProofCollector:
    """Tracks behavioral data over time for functional evaluation.

    Records: stimulus → affect change → decision → action → outcome → memory.
    Maintains a rolling 30-day behavioral profile.
    """

    def __init__(self) -> None:
        self._observations: Deque[BehavioralObservation] = deque(maxlen=5000)
        self._decision_count: int = 0
        self._self_correction_count: int = 0
        self._phi_metric = FunctionalPhiMetric()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        register_runtime_service(
            "behavioral_proof",
            self,
            required=False,
            owner="core/phenomenal_substrate/philosophical_stance.py",
            registered_by="BehavioralProofCollector.start",
        )
        self._started = True
        logger.info("BehavioralProofCollector ONLINE (Path A functionalist)")

    def record_observation(
        self,
        stimulus: str,
        decision: str = "",
        action: str = "",
        outcome: str = "",
        affect_before: Optional[Dict[str, float]] = None,
        affect_after: Optional[Dict[str, float]] = None,
        memory_formed: bool = False,
        self_correction: bool = False,
        appropriateness: float = 0.5,
    ) -> None:
        """Record a behavioral observation."""
        obs = BehavioralObservation(
            stimulus=stimulus[:200],
            affect_before=affect_before or {},
            affect_after=affect_after or {},
            decision=decision[:200],
            action=action[:200],
            outcome=outcome[:200],
            memory_formed=memory_formed,
            self_correction=self_correction,
            contextual_appropriateness=appropriateness,
        )
        self._observations.append(obs)
        self._decision_count += 1
        if self_correction:
            self._self_correction_count += 1

    def generate_proof_bundle(self) -> Dict[str, Any]:
        """Generate a reviewable behavioral proof report.

        This is the longitudinal evidence for functional mind status.
        """
        now = time.time()
        recent = [o for o in self._observations if (now - o.timestamp) < 86400 * 30]

        # Compute metrics
        total_decisions = len(recent)
        self_corrections = sum(1 for o in recent if o.self_correction)
        memories_formed = sum(1 for o in recent if o.memory_formed)
        avg_appropriateness = (
            sum(o.contextual_appropriateness for o in recent) / max(1, total_decisions)
        )

        # Affect variation
        valences = [o.affect_after.get("valence", 0) for o in recent if o.affect_after]
        affect_variation = 0.0
        if valences:
            affect_variation = max(valences) - min(valences)

        # Autonomy ratio: decisions made without user prompting
        autonomous = sum(1 for o in recent if "initiative" in o.stimulus.lower() or "drive" in o.stimulus.lower())
        autonomy_ratio = autonomous / max(1, total_decisions)

        # Phi metrics
        phi_data = self._phi_metric.compute_functional_phi()

        return {
            "report_type": "behavioral_proof_bundle",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observation_window_days": 30,
            "total_observations": total_decisions,
            "metrics": {
                "decision_count": total_decisions,
                "self_correction_count": self_corrections,
                "self_correction_rate": round(self_corrections / max(1, total_decisions), 4),
                "memories_formed": memories_formed,
                "average_contextual_appropriateness": round(avg_appropriateness, 4),
                "affect_variation_range": round(affect_variation, 4),
                "autonomy_ratio": round(autonomy_ratio, 4),
                "functional_phi": phi_data.get("functional_phi", 0.0),
                "temporal_integration_depth": phi_data.get("temporal_depth", 0.0),
                "cross_modal_integration": phi_data.get("cross_modal_integration", 0.0),
            },
            "philosophical_stance": {
                "path": "A — Honest Functionalist",
                "claim": (
                    "This system implements a functional digital mind. "
                    "The behavioral data above constitutes the evidence. "
                    "No claim of phenomenal consciousness is made or implied."
                ),
                "caveat": (
                    "These metrics describe functional properties of an "
                    "information processing system. They are not measurements "
                    "of subjective experience. The system's internal states "
                    "are causally tied to real-world perception via the "
                    "PerceptualPump, but whether this constitutes 'experience' "
                    "in the phenomenological sense remains an open question."
                ),
            },
            "continuity_evidence": {
                "substrate_state_persists": True,
                "memory_bridges_sessions": memories_formed > 0,
                "affect_responds_to_environment": affect_variation > 0.1,
                "self_correction_demonstrates_reflection": self_corrections > 0,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "observations": len(self._observations),
            "decisions": self._decision_count,
            "self_corrections": self._self_correction_count,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_proof_instance: Optional[BehavioralProofCollector] = None


def get_behavioral_proof() -> BehavioralProofCollector:
    global _proof_instance
    if _proof_instance is None:
        _proof_instance = BehavioralProofCollector()
    return _proof_instance


__all__ = [
    "BehavioralProofCollector",
    "BehavioralObservation",
    "FunctionalPhiMetric",
    "get_behavioral_proof",
]
