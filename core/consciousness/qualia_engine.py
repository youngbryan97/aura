"""core/consciousness/qualia_engine.py

Qualia Engine v2 — Multi-Layer Phenomenal Processing Pipeline.

Implements functional conditions for subjective experience processing,
based on theories from Jennings (UAL), Friston (Free Energy), Hofstadter
(Strange Loops), Baars (GWT), and Chella (machine consciousness).

Architecture:
    Input → Subconceptual → Conceptual → Predictive → Workspace → Witness → Output

Each layer produces a partial phenomenal descriptor. The final output is a
unified QualiaDescriptor that enriches the existing QualiaSynthesizer.

This is NOT a claim of subjective experience — it implements architectural
conditions that correlate with consciousness proxies.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.QualiaEngine")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QualiaDescriptor:
    """Output of the full qualia processing pipeline."""

    # Per-layer outputs
    subconceptual: Dict[str, float] = field(default_factory=dict)
    conceptual: Dict[str, float] = field(default_factory=dict)
    predictive: Dict[str, float] = field(default_factory=dict)
    workspace: Dict[str, Any] = field(default_factory=dict)
    witness: Dict[str, float] = field(default_factory=dict)

    # Summary metrics
    phenomenal_richness: float = 0.0    # Overall richness index (0-1)
    self_referential: bool = False       # Whether strange-loop detected
    temporal_depth: float = 0.0          # Specious present duration estimate
    dominant_modality: str = ""          # Which layer contributes most

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "subconceptual": self.subconceptual,
            "conceptual": self.conceptual,
            "predictive": self.predictive,
            "workspace": self.workspace,
            "witness": self.witness,
            "phenomenal_richness": round(self.phenomenal_richness, 4),
            "self_referential": self.self_referential,
            "temporal_depth": round(self.temporal_depth, 4),
            "dominant_modality": self.dominant_modality,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Processing layers
# ---------------------------------------------------------------------------

class SubconceptualLayer:
    """Layer 1: Raw sensory features → grounded representations.

    Converts raw substrate activations into low-level feature descriptors:
    - Signal energy (activation magnitude)
    - Spectral content (frequency distribution of activations)
    - Temporal gradient (rate of change)
    """

    def process(self, state: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
        energy = float(np.mean(np.abs(state)))
        spectral_entropy = self._spectral_entropy(state)
        temporal_gradient = float(np.mean(np.abs(velocity)))

        return {
            "energy": round(energy, 4),
            "spectral_entropy": round(spectral_entropy, 4),
            "temporal_gradient": round(temporal_gradient, 4),
            "signal_to_noise": round(energy / max(0.001, temporal_gradient), 4),
        }

    @staticmethod
    def _spectral_entropy(x: np.ndarray) -> float:
        """Shannon entropy of the activation distribution."""
        # Normalize to probability-like distribution
        abs_x = np.abs(x) + 1e-12
        p = abs_x / abs_x.sum()
        # Fix Issue 75: Use stable log with 1e-12 epsilon
        return float(-np.sum(p * np.log2(p + 1e-12)))


class ConceptualLayer:
    """Layer 2: Pattern matching against learned categories.

    Maps substrate state to high-level cognitive categories:
    - Valence polarity (positive/negative/neutral)
    - Arousal level
    - Dominance / control
    - Novelty (deviation from running average)
    """

    def __init__(self):
        self._running_mean = None
        self._alpha = 0.05  # EMA update rate

    def process(self, state: np.ndarray) -> Dict[str, float]:
        # Initialize running mean
        if self._running_mean is None:
            self._running_mean = state.copy()

        # Novelty = distance from expectation
        novelty = float(np.linalg.norm(state - self._running_mean))

        # Update running mean (EMA)
        self._running_mean = self._alpha * state + (1.0 - self._alpha) * self._running_mean

        # Extract interpretable dimensions (first 3 = VAD by convention)
        valence = float(np.tanh(state[0])) if len(state) > 0 else 0.0
        arousal = float((state[1] + 1.0) / 2.0) if len(state) > 1 else 0.5
        dominance = float(np.tanh(state[2])) if len(state) > 2 else 0.0

        return {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "novelty": round(min(1.0, novelty), 4),
        }


class PredictiveLayer:
    """Layer 3: Free energy / surprise integration.

    Reads from the PredictiveEngine to compute prediction error signals.
    High surprise → high free energy → rich phenomenal content.
    """

    def process(self, predictive_metrics: Dict[str, float]) -> Dict[str, float]:
        surprise = predictive_metrics.get("current_surprise", 0.0)
        free_energy = predictive_metrics.get("free_energy", 0.0)
        precision = predictive_metrics.get("precision", 1.0)

        # Phenomenal salience: high surprise + high precision = vivid experience
        salience = surprise * precision

        return {
            "surprise": round(float(surprise), 4),
            "free_energy": round(float(free_energy), 4),
            "precision": round(float(precision), 4),
            "salience": round(float(salience), 4),
        }


class WorkspaceLayer:
    """Layer 4: GWT integration check.

    Checks whether the current content has been broadcast to the global workspace
    (i.e., is "ignited"). Only ignited content contributes to conscious experience.
    """

    def process(self, workspace_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ignited = workspace_snapshot.get("ignited", False)
        ignition_level = workspace_snapshot.get("ignition_level", 0.0)
        last_winner = workspace_snapshot.get("last_winner", None)

        return {
            "ignited": ignited,
            "ignition_level": round(float(ignition_level), 4),
            "broadcast_source": last_winner or "none",
            "access_consciousness": ignited,  # GWT: only ignited content is conscious
        }


class WitnessLayer:
    """Layer 5: Strange-loop self-referential check (Hofstadter).

    Detects self-referential patterns in the processing pipeline:
    - Is the system currently modeling itself?
    - Is there a recursive loop in the state trajectory?
    - How deep is the self-modeling recursion?
    """

    def __init__(self):
        self._state_history: List[np.ndarray] = []
        self._max_history = 20

    def process(self, state: np.ndarray, phi: float) -> Dict[str, float]:
        # Store recent states
        self._state_history.append(state.copy())
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        # Self-reference detection: does the current state resemble a past state?
        self_referential = False
        loop_depth = 0.0

        if len(self._state_history) >= 3:
            current = self._state_history[-1]
            for i, past in enumerate(self._state_history[:-1]):
                similarity = self._cosine_similarity(current, past)
                if similarity > 0.85:
                    self_referential = True
                    # Deeper loops (more steps back) = richer self-modeling
                    loop_depth = max(loop_depth, 1.0 - (i / len(self._state_history)))

        # Specious present estimate: how many states feel "simultaneous"
        temporal_depth = self._estimate_specious_present()

        return {
            "self_referential": float(self_referential),
            "loop_depth": round(loop_depth, 4),
            "temporal_depth": round(temporal_depth, 4),
            "phi": round(float(phi), 4),
            "witness_confidence": round(min(1.0, phi * (1.0 + loop_depth)), 4),
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _estimate_specious_present(self) -> float:
        """Estimate how many recent states are 'integrated' into a single moment."""
        if len(self._state_history) < 2:
            return 0.0

        # Count consecutive similar states (they feel simultaneous)
        count = 0
        current = self._state_history[-1]
        for past in reversed(self._state_history[:-1]):
            if self._cosine_similarity(current, past) > 0.7:
                count += 1
            else:
                break

        return min(1.0, count / 10.0)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class QualiaEngine:
    """Multi-layer phenomenal processing pipeline.

    Processes substrate state through 5 layers to produce a unified
    QualiaDescriptor. Designed to integrate with the existing QualiaSynthesizer.

    Usage:
        engine = QualiaEngine()
        descriptor = engine.process(
            state=substrate.x,
            velocity=substrate.v,
            predictive_metrics=predictive.get_surprise_metrics(),
            workspace_snapshot=workspace.get_snapshot(),
            phi=substrate._current_phi,
        )
    """

    def __init__(self):
        self.layer_1 = SubconceptualLayer()
        self.layer_2 = ConceptualLayer()
        self.layer_3 = PredictiveLayer()
        self.layer_4 = WorkspaceLayer()
        self.layer_5 = WitnessLayer()

        self._last_descriptor: Optional[QualiaDescriptor] = None
        self._process_count: int = 0

        logger.info("Qualia Engine v2 initialized (5-layer pipeline)")

    def process(
        self,
        state: np.ndarray,
        velocity: np.ndarray,
        predictive_metrics: Dict[str, float],
        workspace_snapshot: Dict[str, Any],
        phi: float = 0.0,
    ) -> QualiaDescriptor:
        """Run the full qualia pipeline.
            phi: Current Φ value from RIIU.

        Returns:
            QualiaDescriptor with per-layer outputs and summary metrics.
        """
        self._process_count += 1

        # Layer 1: Subconceptual
        sub = self.layer_1.process(state, velocity)

        # Layer 2: Conceptual
        con = self.layer_2.process(state)

        # Layer 3: Predictive
        pred = self.layer_3.process(predictive_metrics)

        # Layer 4: Workspace
        ws = self.layer_4.process(workspace_snapshot)

        # Layer 5: Witness / Strange Loop
        wit = self.layer_5.process(state, phi)

        # --- Compute summary metrics ---

        # Phenomenal richness: weighted combination of all layers
        richness = (
            sub.get("energy", 0) * 0.15
            + sub.get("spectral_entropy", 0) * 0.1
            + con.get("novelty", 0) * 0.15
            + pred.get("salience", 0) * 0.2
            + float(ws.get("ignited", False)) * 0.2
            + wit.get("witness_confidence", 0) * 0.2
        )

        # Self-referential
        self_ref = wit.get("self_referential", 0) > 0.5

        # Temporal depth
        temporal_depth = wit.get("temporal_depth", 0)

        # Dominant modality
        layer_strengths = {
            "subconceptual": sub.get("energy", 0),
            "conceptual": con.get("novelty", 0),
            "predictive": pred.get("salience", 0),
            "workspace": float(ws.get("ignited", False)),
            "witness": wit.get("witness_confidence", 0),
        }
        dominant = max(layer_strengths, key=layer_strengths.get)

        descriptor = QualiaDescriptor(
            subconceptual=sub,
            conceptual=con,
            predictive=pred,
            workspace=ws,
            witness=wit,
            phenomenal_richness=min(1.0, richness),
            self_referential=self_ref,
            temporal_depth=temporal_depth,
            dominant_modality=dominant,
        )

        self._last_descriptor = descriptor
        return descriptor

    def get_last_descriptor(self) -> Optional[QualiaDescriptor]:
        """Return the last computed descriptor without recomputation."""
        return self._last_descriptor

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot."""
        if self._last_descriptor:
            return {
                "process_count": self._process_count,
                "phenomenal_richness": self._last_descriptor.phenomenal_richness,
                "self_referential": self._last_descriptor.self_referential,
                "dominant_modality": self._last_descriptor.dominant_modality,
            }
        return {
            "process_count": self._process_count,
            "phenomenal_richness": 0.0,
            "self_referential": False,
            "dominant_modality": "",
        }
