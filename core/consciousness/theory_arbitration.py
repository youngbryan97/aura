"""
Theory Arbitration Meta-Framework

The consciousness stack contains GWT, IIT, RPT, HOT, Multiple Drafts,
predictive coding, and more. These theories do NOT all say the same thing.
They make different, sometimes incompatible predictions. Running them
simultaneously without a meta-framework means the system can always find
some theory that validates any behavior — making it unfalsifiable.

This module:
1. Classifies each theory's role (mechanistic commitment, measurement
   heuristic, or adversarial test harness)
2. Logs when theories make divergent predictions about the same input
3. Tracks which theory's predictions best match actual behavior over time
4. Produces adversarial test results that could be published

This makes the system FALSIFIABLE. It can fail. That's what makes it
scientifically serious rather than infinitely accommodating.
"""
from __future__ import annotations


import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Consciousness.TheoryArbitration")


class TheoryRole(str, Enum):
    """How a theory is used in the architecture."""
    MECHANISTIC = "mechanistic_commitment"    # This is literally how the system works
    MEASUREMENT = "measurement_heuristic"     # This is how we verify/quantify
    ADVERSARIAL = "adversarial_test_harness"  # This theory should predict X; test it


@dataclass
class TheoryRegistration:
    """A registered theory with its role and prediction track record."""
    name: str
    role: TheoryRole
    module_path: str                    # e.g. "core/consciousness/phi_core.py"
    description: str
    predictions_made: int = 0
    predictions_correct: int = 0
    predictions_wrong: int = 0
    last_prediction_at: float = 0.0
    evidence_for: float = 0.0          # Accumulated evidence strength
    evidence_against: float = 0.0


@dataclass
class PredictionRecord:
    """A logged prediction from one theory about a specific event."""
    theory: str
    event_id: str
    prediction: str                    # What the theory predicts
    confidence: float = 0.5
    actual_outcome: str = ""           # What actually happened
    correct: Optional[bool] = None
    timestamp: float = 0.0


class TheoryArbitrationFramework:
    """Meta-framework for managing competing consciousness theories.

    Each theory is registered with its role. When theories make divergent
    predictions, the divergence is logged and the actual outcome updates
    the track record of each theory.

    This doesn't decide which theory is "right." It accumulates evidence
    over runtime so that Aura's behavior can empirically pick sides.
    """

    def __init__(self):
        self._theories: Dict[str, TheoryRegistration] = {}
        self._predictions: deque[PredictionRecord] = deque(maxlen=200)
        self._divergences: deque[Dict[str, Any]] = deque(maxlen=50)
        self._register_default_theories()
        logger.info("TheoryArbitrationFramework initialized with %d theories.", len(self._theories))

    def _register_default_theories(self):
        """Register Aura's theory stack with explicit roles."""
        registrations = [
            TheoryRegistration(
                name="gwt",
                role=TheoryRole.MECHANISTIC,
                module_path="core/consciousness/global_workspace.py",
                description="Global Workspace Theory (Baars): conscious content requires broadcast to all modules",
            ),
            TheoryRegistration(
                name="iit_4_0",
                role=TheoryRole.MEASUREMENT,
                module_path="core/consciousness/phi_core.py",
                description="IIT 4.0 (Tononi): consciousness = integrated information (phi) with exclusion postulate",
            ),
            TheoryRegistration(
                name="predictive_coding",
                role=TheoryRole.MECHANISTIC,
                module_path="core/consciousness/predictive_hierarchy.py",
                description="Hierarchical predictive coding (Friston): every level predicts down, errors propagate up",
            ),
            TheoryRegistration(
                name="rpt",
                role=TheoryRole.ADVERSARIAL,
                module_path="core/consciousness/neural_mesh.py",
                description="Recurrent Processing Theory (Lamme): consciousness requires exec→sensory feedback, not broadcast",
            ),
            TheoryRegistration(
                name="hot",
                role=TheoryRole.ADVERSARIAL,
                module_path="core/consciousness/hot_engine.py",
                description="Higher-Order Thought (Rosenthal): consciousness requires a representation OF the mental state",
            ),
            TheoryRegistration(
                name="multiple_drafts",
                role=TheoryRole.ADVERSARIAL,
                module_path="core/consciousness/multiple_drafts.py",
                description="Multiple Drafts (Dennett): no central broadcast event; parallel streams, retroactive probe",
            ),
            TheoryRegistration(
                name="ast",
                role=TheoryRole.MECHANISTIC,
                module_path="core/consciousness/attention_schema.py",
                description="Attention Schema Theory (Graziano): brain builds a simplified model of its own attention",
            ),
            TheoryRegistration(
                name="free_energy",
                role=TheoryRole.MECHANISTIC,
                module_path="core/consciousness/free_energy.py",
                description="Free Energy Principle (Friston): behavior minimizes surprise + complexity",
            ),
            TheoryRegistration(
                name="enactivism",
                role=TheoryRole.ADVERSARIAL,
                module_path="core/consciousness/embodied_interoception.py",
                description="Enactivism (Varela/Thompson): mind requires closed sensorimotor loop with world resistance",
            ),
            TheoryRegistration(
                name="illusionism",
                role=TheoryRole.MEASUREMENT,
                module_path="core/consciousness/illusionism_layer.py",
                description="Illusionism (Frankish/Dennett): phenomenal consciousness may be useful fiction, not extra property",
            ),
        ]
        for reg in registrations:
            self._theories[reg.name] = reg

    def log_prediction(
        self,
        theory: str,
        event_id: str,
        prediction: str,
        confidence: float = 0.5,
    ):
        """Log a prediction from a theory about a specific event.

        Called when a theory-specific module makes a claim about what
        should happen next.
        """
        if theory not in self._theories:
            return

        record = PredictionRecord(
            theory=theory,
            event_id=event_id,
            prediction=prediction,
            confidence=min(1.0, max(0.0, confidence)),
            timestamp=time.time(),
        )
        self._predictions.append(record)
        self._theories[theory].predictions_made += 1
        self._theories[theory].last_prediction_at = time.time()

    def resolve_prediction(
        self,
        event_id: str,
        actual_outcome: str,
    ):
        """Resolve all predictions for an event with the actual outcome.

        Called when the actual behavior/state is known. Updates the
        track record of each theory that made a prediction.
        """
        for record in self._predictions:
            if record.event_id != event_id or record.correct is not None:
                continue

            record.actual_outcome = actual_outcome
            # Simple match: does the prediction string appear in the outcome?
            # (More sophisticated matching could use embedding similarity)
            predicted_lower = record.prediction.lower()
            actual_lower = actual_outcome.lower()
            if predicted_lower in actual_lower or actual_lower in predicted_lower:
                record.correct = True
                self._theories[record.theory].predictions_correct += 1
                self._theories[record.theory].evidence_for += record.confidence
            else:
                record.correct = False
                self._theories[record.theory].predictions_wrong += 1
                self._theories[record.theory].evidence_against += record.confidence

    def log_divergence(
        self,
        event_id: str,
        theory_a: str,
        prediction_a: str,
        theory_b: str,
        prediction_b: str,
    ):
        """Log when two theories make different predictions about the same event.

        These divergences are the most scientifically valuable data: they're
        the adversarial tests that the consciousness field has been calling for.
        """
        self._divergences.append({
            "event_id": event_id,
            "theory_a": theory_a,
            "prediction_a": prediction_a,
            "theory_b": theory_b,
            "prediction_b": prediction_b,
            "timestamp": time.time(),
            "resolved": False,
            "winner": None,
        })
        logger.info(
            "TheoryArbitration: DIVERGENCE %s vs %s on event %s",
            theory_a, theory_b, event_id,
        )

    def get_theory_rankings(self) -> List[Dict[str, Any]]:
        """Return theories ranked by prediction accuracy."""
        rankings = []
        for name, theory in self._theories.items():
            total = theory.predictions_correct + theory.predictions_wrong
            accuracy = theory.predictions_correct / max(1, total)
            rankings.append({
                "name": name,
                "role": theory.role.value,
                "predictions_made": theory.predictions_made,
                "accuracy": round(accuracy, 4),
                "evidence_for": round(theory.evidence_for, 4),
                "evidence_against": round(theory.evidence_against, 4),
                "net_evidence": round(theory.evidence_for - theory.evidence_against, 4),
            })
        rankings.sort(key=lambda x: x["net_evidence"], reverse=True)
        return rankings

    def get_context_block(self) -> str:
        """Context block showing theory status. Only fires occasionally."""
        divergences = [d for d in self._divergences if not d.get("resolved")]
        if not divergences:
            return ""
        latest = divergences[-1]
        return (
            f"## THEORY ARBITRATION\n"
            f"Active divergence: {latest['theory_a']} vs {latest['theory_b']}\n"
            f"Predictions differ — behavior will provide evidence for one side."
        )

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "theories": {
                name: {
                    "role": t.role.value,
                    "predictions": t.predictions_made,
                    "accuracy": round(
                        t.predictions_correct / max(1, t.predictions_correct + t.predictions_wrong), 4
                    ),
                }
                for name, t in self._theories.items()
            },
            "unresolved_divergences": len([d for d in self._divergences if not d.get("resolved")]),
            "total_predictions": sum(t.predictions_made for t in self._theories.values()),
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[TheoryArbitrationFramework] = None


def get_theory_arbitration() -> TheoryArbitrationFramework:
    global _instance
    if _instance is None:
        _instance = TheoryArbitrationFramework()
    return _instance
