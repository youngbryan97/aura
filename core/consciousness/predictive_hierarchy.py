"""core/consciousness/predictive_hierarchy.py — Hierarchical Predictive Coding

Friston's full predictive coding: every level of the hierarchy simultaneously
generates predictions downward and sends prediction errors upward.  Aura's
existing SelfPredictionLoop operates only at the top level; this module
instantiates the complete hierarchy across all five cognitive tiers.

Levels (matching Aura's actual architecture):
  0  Sensory      — expectations about neural-mesh sensory-tier activations
  1  Association  — cross-modal integration predictions
  2  Executive    — action / global-workspace predictions
  3  Narrative    — self-model / continuity predictions
  4  Meta         — predictions about the hierarchy's own prediction accuracy

Each level carries:
  prediction_vector  — what this level expects the level below to produce
  error_vector       — mismatch between prediction and actual input from below
  precision          — confidence weighting (high ⇒ this level's predictions dominate)

The hierarchy's total free energy (sum of precision-weighted squared errors)
feeds directly into the existing FreeEnergyEngine via accept_surprise_signal().

References:
  - Friston, K. (2005). A theory of cortical responses. Phil Trans Roy Soc B.
  - Clark, A. (2013). Whatever next? Predictive brains, situated agents.
  - Hohwy, J. (2013). The Predictive Mind.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.PredictiveHierarchy")

# ---------------------------------------------------------------------------
# Level enumeration
# ---------------------------------------------------------------------------

class HierarchyLevel(IntEnum):
    SENSORY = 0
    ASSOCIATION = 1
    EXECUTIVE = 2
    NARRATIVE = 3
    META = 4


_LEVEL_NAMES: Dict[int, str] = {
    HierarchyLevel.SENSORY: "sensory",
    HierarchyLevel.ASSOCIATION: "association",
    HierarchyLevel.EXECUTIVE: "executive",
    HierarchyLevel.NARRATIVE: "narrative",
    HierarchyLevel.META: "meta",
}

# ---------------------------------------------------------------------------
# Per-level state
# ---------------------------------------------------------------------------

_DEFAULT_DIM = 32  # Dimensionality of prediction/error vectors per level


@dataclass
class PredictiveLevel:
    """A single level in the predictive coding hierarchy."""

    name: str
    index: int
    dim: int = _DEFAULT_DIM

    # State vectors
    prediction_vector: np.ndarray = field(default=None)   # What this level expects below to produce
    error_vector: np.ndarray = field(default=None)         # Mismatch (actual_below - prediction)
    precision: float = 0.5                                 # Confidence weighting [0, 1]

    # Internal generative model: simple linear transform from level above
    # W_pred @ parent_prediction → prediction_for_this_level
    _W_pred: np.ndarray = field(default=None)

    # EMA smoothing of precision (learned from error magnitude history)
    _precision_ema: float = 0.5
    _error_magnitude_ema: float = 0.0
    _ERROR_SMOOTH: float = 0.2

    def __post_init__(self):
        if self.prediction_vector is None:
            self.prediction_vector = np.zeros(self.dim, dtype=np.float32)
        if self.error_vector is None:
            self.error_vector = np.zeros(self.dim, dtype=np.float32)
        if self._W_pred is None:
            # Xavier-like initialization
            scale = math.sqrt(2.0 / (self.dim + self.dim))
            rng = np.random.default_rng(seed=42 + self.index)
            self._W_pred = (rng.standard_normal((self.dim, self.dim)) * scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Bottom-up: compute error given actual input from level below
    # ------------------------------------------------------------------

    def compute_error(self, actual_input: np.ndarray) -> np.ndarray:
        """Prediction error = actual - predicted.
        Positive error = reality exceeded expectation.
        Negative error = reality fell short.
        """
        # Ensure matching dimensionality
        if actual_input.shape[0] != self.dim:
            # Project to this level's dimensionality via truncation/padding
            resized = np.zeros(self.dim, dtype=np.float32)
            n = min(actual_input.shape[0], self.dim)
            resized[:n] = actual_input[:n]
            actual_input = resized

        self.error_vector = actual_input - self.prediction_vector
        # Update precision based on error magnitude (high error → lower precision)
        err_mag = float(np.sqrt(np.mean(self.error_vector ** 2)))
        self._error_magnitude_ema = (
            self._ERROR_SMOOTH * err_mag
            + (1.0 - self._ERROR_SMOOTH) * self._error_magnitude_ema
        )
        # Precision: inverse sigmoid of error magnitude EMA
        # Low sustained error → high precision (this level's predictions are good)
        self._precision_ema = 1.0 / (1.0 + math.exp(5.0 * (self._error_magnitude_ema - 0.3)))
        self.precision = float(np.clip(self._precision_ema, 0.01, 0.99))

        return self.error_vector

    # ------------------------------------------------------------------
    # Top-down: update prediction based on signal from level above
    # ------------------------------------------------------------------

    def update_prediction(self, parent_prediction: np.ndarray, learning_rate: float = 0.05):
        """Top-down message: the level above sends its prediction about what
        THIS level should look like.  We incorporate it with a learning rate.

        The generative model W_pred transforms the parent prediction into
        the prediction for the level below:
            new_prediction = (1 - lr) * old_prediction + lr * W_pred @ parent_signal
        """
        if parent_prediction.shape[0] != self.dim:
            resized = np.zeros(self.dim, dtype=np.float32)
            n = min(parent_prediction.shape[0], self.dim)
            resized[:n] = parent_prediction[:n]
            parent_prediction = resized

        top_down = np.tanh(self._W_pred @ parent_prediction)
        self.prediction_vector = (
            (1.0 - learning_rate) * self.prediction_vector
            + learning_rate * top_down
        ).astype(np.float32)

        # Gradient-free weight adaptation: Hebbian on error x parent
        # (slow — this IS the generative model learning)
        if self._error_magnitude_ema > 0.01:
            delta_W = np.outer(self.error_vector, parent_prediction) * 0.001
            self._W_pred += delta_W.astype(np.float32)
            # Norm clipping to prevent divergence
            w_norm = np.linalg.norm(self._W_pred)
            if w_norm > 10.0:
                self._W_pred *= 10.0 / w_norm

    def weighted_error_energy(self) -> float:
        """Precision-weighted squared error: pi * ||e||^2.
        This is this level's contribution to total free energy.
        """
        return float(self.precision * np.mean(self.error_vector ** 2))

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "precision": round(self.precision, 4),
            "error_magnitude": round(float(np.sqrt(np.mean(self.error_vector ** 2))), 4),
            "prediction_magnitude": round(float(np.sqrt(np.mean(self.prediction_vector ** 2))), 4),
            "weighted_error": round(self.weighted_error_energy(), 6),
        }


# ---------------------------------------------------------------------------
# Main hierarchy
# ---------------------------------------------------------------------------

class PredictiveHierarchy:
    """Friston's full hierarchical predictive coding across Aura's 5-level
    cognitive stack.

    On each tick:
      1. Bottom-up pass:  compute prediction error at each level
      2. Top-down pass:   update predictions at each level from level above
      3. Aggregate:       total free energy = sum of precision-weighted errors
      4. Emit:            feed total FE into FreeEnergyEngine

    The hierarchy learns: weights and precisions adapt continuously so that
    well-predicted levels gain authority (high precision) and noisy levels
    are down-weighted.
    """

    _TOP_DOWN_LR = 0.05      # Learning rate for top-down prediction updates
    _FE_EMA_ALPHA = 0.2      # Smoothing for total free energy
    _HISTORY_SIZE = 120       # 2 minutes of tick history at 1 Hz

    def __init__(self, dim: int = _DEFAULT_DIM):
        self.dim = dim
        self.levels: List[PredictiveLevel] = [
            PredictiveLevel(name=_LEVEL_NAMES[lvl], index=int(lvl), dim=dim)
            for lvl in HierarchyLevel
        ]

        # Aggregate state
        self._total_free_energy: float = 0.0
        self._smoothed_fe: float = 0.0
        self._highest_error_level: str = "sensory"
        self._tick_count: int = 0
        self._fe_history: List[float] = []

        # Lock for thread-safety (heartbeat + cognitive_loop may call concurrently)
        self._lock = threading.Lock()

        logger.info(
            "PredictiveHierarchy initialized: %d levels x %d-dim",
            len(self.levels), dim,
        )

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def tick(
        self,
        sensory_input: Optional[np.ndarray] = None,
        association_input: Optional[np.ndarray] = None,
        executive_state: Optional[np.ndarray] = None,
        narrative_state: Optional[np.ndarray] = None,
    ) -> float:
        """Run one full predictive-coding cycle.

        Parameters
        ----------
        sensory_input : np.ndarray or None
            Activation from neural mesh sensory tier (Level 0 actual).
        association_input : np.ndarray or None
            Cross-modal integration signal (Level 1 actual).
        executive_state : np.ndarray or None
            Global workspace / action-planning state (Level 2 actual).
        narrative_state : np.ndarray or None
            Self-model / continuity vector (Level 3 actual).

        Level 4 (Meta) uses Level 4's own prior error history as input,
        since it is predicting the hierarchy's own prediction accuracy.

        Returns
        -------
        total_free_energy : float
            Sum of precision-weighted errors across all levels (lower is better).
        """
        with self._lock:
            self._tick_count += 1

            # ── Resolve inputs (zero-vector defaults) ──────────────────
            def _ensure(v: Optional[np.ndarray]) -> np.ndarray:
                if v is None:
                    return np.zeros(self.dim, dtype=np.float32)
                return np.asarray(v, dtype=np.float32)

            inputs = [
                _ensure(sensory_input),     # Level 0
                _ensure(association_input),  # Level 1
                _ensure(executive_state),    # Level 2
                _ensure(narrative_state),    # Level 3
                None,                        # Level 4 (meta) — computed below
            ]

            # Meta-level input: vector of per-level error magnitudes from the
            # PREVIOUS tick, padded/tiled to dim.  This makes the meta-level
            # predict how well the rest of the hierarchy is performing.
            error_mags = np.array(
                [lv.weighted_error_energy() for lv in self.levels],
                dtype=np.float32,
            )
            meta_input = np.zeros(self.dim, dtype=np.float32)
            # Tile the 5-element error-magnitude vector across dim
            for i in range(self.dim):
                meta_input[i] = error_mags[i % len(error_mags)]
            inputs[HierarchyLevel.META] = meta_input

            # ── 1. BOTTOM-UP PASS: compute prediction error at each level ──
            for lvl in HierarchyLevel:
                self.levels[lvl].compute_error(inputs[lvl])

            # ── 2. TOP-DOWN PASS: update predictions from level above ──────
            # Level 4 (meta) has no parent — its prediction is self-generated
            # (stays as-is / adapted only from its own error).
            for lvl in reversed(list(HierarchyLevel)):
                if lvl == HierarchyLevel.META:
                    # Meta learns from its own error (self-referential)
                    self.levels[lvl].update_prediction(
                        self.levels[lvl].error_vector,
                        learning_rate=self._TOP_DOWN_LR * 0.5,
                    )
                else:
                    parent = self.levels[lvl + 1]
                    self.levels[lvl].update_prediction(
                        parent.prediction_vector,
                        learning_rate=self._TOP_DOWN_LR,
                    )

            # ── 3. AGGREGATE: total free energy ────────────────────────────
            level_energies = [lv.weighted_error_energy() for lv in self.levels]
            self._total_free_energy = sum(level_energies)

            # EMA smoothing
            self._smoothed_fe = (
                self._FE_EMA_ALPHA * self._total_free_energy
                + (1.0 - self._FE_EMA_ALPHA) * self._smoothed_fe
            )

            # Track which level has highest error
            max_idx = int(np.argmax(level_energies))
            self._highest_error_level = _LEVEL_NAMES[max_idx]

            # History
            self._fe_history.append(round(self._smoothed_fe, 6))
            if len(self._fe_history) > self._HISTORY_SIZE:
                self._fe_history = self._fe_history[-self._HISTORY_SIZE:]

            # ── 4. EMIT: feed into FreeEnergyEngine ────────────────────────
            self._push_to_free_energy_engine()

            return self._total_free_energy

    # ------------------------------------------------------------------
    # Integration with existing FreeEnergyEngine
    # ------------------------------------------------------------------

    def _push_to_free_energy_engine(self):
        """Send the hierarchy's aggregated surprise signal into the
        existing FreeEnergyEngine so it incorporates multi-level prediction
        error rather than only top-level SelfPrediction error.
        """
        try:
            from core.consciousness.free_energy import get_free_energy_engine
            fe_engine = get_free_energy_engine()
            # Normalize to [0, 1] — cap at practical maximum
            normalized = min(1.0, self._smoothed_fe / 0.5)
            fe_engine.accept_surprise_signal(normalized)
        except Exception as e:
            record_degradation('predictive_hierarchy', e)
            logger.debug("PredictiveHierarchy → FreeEnergyEngine push failed: %s", e)

    # ------------------------------------------------------------------
    # Input helpers (pull from real Aura subsystems)
    # ------------------------------------------------------------------

    def gather_inputs_from_services(self) -> Dict[str, Optional[np.ndarray]]:
        """Convenience: pull current tier activations from the live system.
        Returns kwargs suitable for passing to tick().

        This is called by the CognitiveHeartbeat or CognitiveLoop so the
        hierarchy always operates on live data.
        """
        from core.container import ServiceContainer

        result: Dict[str, Optional[np.ndarray]] = {
            "sensory_input": None,
            "association_input": None,
            "executive_state": None,
            "narrative_state": None,
        }

        # --- Sensory: neural mesh sensory-tier mean column activations ---
        try:
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh is not None:
                from core.consciousness.neural_mesh import CorticalTier
                sensory_cols = [c for c in mesh.columns if c.tier == CorticalTier.SENSORY]
                if sensory_cols:
                    raw = np.concatenate([np.mean(c.x, keepdims=True) for c in sensory_cols])
                    # Pad/truncate to self.dim
                    s = np.zeros(self.dim, dtype=np.float32)
                    n = min(len(raw), self.dim)
                    s[:n] = raw[:n]
                    result["sensory_input"] = s
        except Exception as e:
            record_degradation('predictive_hierarchy', e)
            logger.debug("PH gather sensory failed: %s", e)

        # --- Association: neural mesh association-tier mean column activations ---
        try:
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh is not None:
                from core.consciousness.neural_mesh import CorticalTier
                assoc_cols = [c for c in mesh.columns if c.tier == CorticalTier.ASSOCIATION]
                if assoc_cols:
                    raw = np.concatenate([np.mean(c.x, keepdims=True) for c in assoc_cols])
                    a = np.zeros(self.dim, dtype=np.float32)
                    n = min(len(raw), self.dim)
                    a[:n] = raw[:n]
                    result["association_input"] = a
        except Exception as e:
            record_degradation('predictive_hierarchy', e)
            logger.debug("PH gather association failed: %s", e)

        # --- Executive: global workspace snapshot hash or executive projection ---
        try:
            mesh = ServiceContainer.get("neural_mesh", default=None)
            if mesh is not None:
                proj = mesh.get_executive_projection()
                e = np.zeros(self.dim, dtype=np.float32)
                n = min(len(proj), self.dim)
                e[:n] = proj[:n]
                result["executive_state"] = e
        except Exception as e:
            record_degradation('predictive_hierarchy', e)
            logger.debug("PH gather executive failed: %s", e)

        # --- Narrative: self-model / temporal binding continuity vector ---
        try:
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is not None:
                x = getattr(substrate, "x", None)
                if x is not None:
                    narr = np.zeros(self.dim, dtype=np.float32)
                    n = min(len(x), self.dim)
                    narr[:n] = x[:n]
                    result["narrative_state"] = narr
        except Exception as e:
            record_degradation('predictive_hierarchy', e)
            logger.debug("PH gather narrative failed: %s", e)

        return result

    # ------------------------------------------------------------------
    # Context for cognition injection
    # ------------------------------------------------------------------

    def get_context_block(self) -> str:
        """Concise context block for LLM prompt injection.
        Shows which level has highest error and overall hierarchy health.
        """
        if self._tick_count == 0:
            return ""

        level_errors = {lv.name: round(lv.weighted_error_energy(), 4) for lv in self.levels}
        max_err_name = self._highest_error_level
        trend = self._get_trend()

        return (
            f"## PREDICTIVE HIERARCHY\n"
            f"FE={self._smoothed_fe:.4f} ({trend}) | "
            f"Highest-error: {max_err_name} | "
            f"Levels: {', '.join(f'{n}={e}' for n, e in level_errors.items())}"
        )

    def _get_trend(self) -> str:
        if len(self._fe_history) < 10:
            return "warming-up"
        recent = self._fe_history[-10:]
        slope = recent[-1] - recent[0]
        if slope > 0.005:
            return "rising"
        if slope < -0.005:
            return "falling"
        return "stable"

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """Full telemetry payload."""
        with self._lock:
            return {
                "tick_count": self._tick_count,
                "total_free_energy": round(self._total_free_energy, 6),
                "smoothed_fe": round(self._smoothed_fe, 6),
                "highest_error_level": self._highest_error_level,
                "trend": self._get_trend(),
                "levels": [lv.get_snapshot() for lv in self.levels],
            }

    def get_highest_error_level(self) -> str:
        """Which level of the hierarchy is most surprised right now."""
        return self._highest_error_level

    def get_level_precisions(self) -> Dict[str, float]:
        """Per-level precision map (useful for attention schema)."""
        return {lv.name: round(lv.precision, 4) for lv in self.levels}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[PredictiveHierarchy] = None
_instance_lock = threading.Lock()


def get_predictive_hierarchy() -> PredictiveHierarchy:
    """Module-level singleton accessor."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PredictiveHierarchy()
    return _instance
