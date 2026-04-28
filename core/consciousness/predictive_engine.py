"""core/consciousness/predictive_engine.py

Implements Predictive Coding / Free Energy Principle.
The brain is a prediction machine. It constantly generates expectations about the future.
"Surprise" (Prediction Error) is the difference between Expectation and Reality.
Minimizing surprise is the core drive of the system.
"""

from core.runtime.errors import record_degradation
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.Predictive")

@dataclass
class Prediction:
    """An expectation about the future state"""
    source_module: str
    content: str
    expected_state_vector: Optional[np.ndarray] = None # Predicted substrate state
    expected_changes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8 # Precision
    timestamp: float = field(default_factory=time.time)

class PredictiveEngine:
    """Manages the hierarchy of predictions and error minimization.
    Combines world-model state predictions and substrate-level vector predictions.
    """
    
    def __init__(self, world_model=None, neuron_count: int = 64):
        self.world_model = world_model
        self.neuron_count = neuron_count
        self.active_predictions: List[Prediction] = []
        self.total_surprise = 0.0
        self.surprise_history: List[float] = []
        
        # Simple generative model for substrate
        self.internal_model = np.zeros(neuron_count)
        
        # IWMT / Free Energy State (Phase XVI)
        self.free_energy = 0.0
        self.precision = 1.0 # Confidence weight
        
        logger.info("Predictive Engine initialized (Unified).")
        
    async def predict_next_state(self, action: Dict[str, Any]) -> Prediction:
        """Predict the outcome of an action on the world state.
        """
        prediction = Prediction(
            source_module="GenerativeModel",
            content=f"Predicting outcome of {action.get('type', 'action')}",
            timestamp=time.time()
        )
        
        if self.world_model:
            try:
                current_summary = self.world_model.get_summary()
                action_type = action.get("type", "unknown")
                
                if action_type == "search":
                    prediction.expected_changes["total_beliefs"] = current_summary.get("total_beliefs", 0) + 2
                elif action_type == "apply_fix":
                    prediction.expected_changes["strong"] = current_summary.get("strong", 0) + 1
            except Exception as e:
                record_degradation('predictive_engine', e)
                logger.debug("World model prediction failed: %s", e)

        # Substrate prediction (momentum heuristic)
        prediction.expected_state_vector = self.internal_model * 0.95

        # Adaptive prediction: if surprise has been rising, widen prediction
        # to expect more change (reflect rising uncertainty)
        trend = self.get_surprise_trend()
        if trend == "rising" and prediction.expected_state_vector is not None:
            # Nudge prediction toward larger state changes
            prediction.expected_state_vector *= 0.90
            prediction.confidence = max(0.3, prediction.confidence - 0.1)
        elif trend == "falling":
            prediction.confidence = min(0.95, prediction.confidence + 0.05)

        self.active_predictions.append(prediction)
        if len(self.active_predictions) > 10:
            self.active_predictions.pop(0)

        return prediction

    def compute_surprise(self, actual_state_summary: Dict[str, Any], actual_substrate_x: Optional[np.ndarray] = None) -> float:
        """Compare actual state with predicted state to compute surprise (prediction error).
        """
        if not self.active_predictions:
            return 0.0
            
        best_p = self.active_predictions[-1]
        surprise = 0.0
        
        # 1. World Model Surprise
        for key, p_val in best_p.expected_changes.items():
            a_val = actual_state_summary.get(key, 0)
            surprise += abs(p_val - a_val)
            
        # 2. Substrate Surprise
        if actual_substrate_x is not None and best_p.expected_state_vector is not None:
            error_vector = actual_substrate_x - best_p.expected_state_vector
            sub_surprise = np.mean(np.square(error_vector))
            surprise += sub_surprise * 5.0 # Weight substrate surprise
            
            # Update internal model (Learning)
            alpha = 0.1
            self.internal_model = (1 - alpha) * self.internal_model + alpha * actual_substrate_x

        # Normalize/Scale surprise
        normalized_surprise = min(1.0, surprise / 5.0)
        
        # --- Phase XVI: IWMT / Free Energy Calculation ---
        # F = Complexity - Accuracy (Simplified variational free energy)
        # Complexity is the drift in internal model, Accuracy is the surprise
        complexity = np.mean(np.abs(self.internal_model)) * 0.1
        accuracy = 1.0 - normalized_surprise
        self.free_energy = complexity + (1.0 - accuracy)
        
        # Update Precision (Inverse Variance)
        # High surprise lowers precision (confidence)
        self.precision = (self.precision * 0.95) + ((1.0 - normalized_surprise) * 0.05)
        
        self.total_surprise = (self.total_surprise * 0.9) + (normalized_surprise * 0.1)
        self.surprise_history.append(normalized_surprise)
        
        if len(self.surprise_history) > 100:
            self.surprise_history.pop(0)
            
        if normalized_surprise > 0.6:
            logger.warning("HIGH SURPRISE: %.2f", normalized_surprise)
            
        return normalized_surprise

    def get_surprise_metrics(self) -> Dict[str, float]:
        return {
            "current_surprise": self.surprise_history[-1] if self.surprise_history else 0.0,
            "average_surprise": float(np.mean(self.surprise_history)) if self.surprise_history else 0.0,
            "total_accumulated": self.total_surprise,
            "free_energy": float(self.free_energy),
            "precision": float(self.precision),
            "surprise_trend": self.get_surprise_trend(),
        }

    # ── New integration methods ───────────────────────────────────────────

    def get_surprise_trend(self) -> str:
        """Analyze last 20 surprise values — 'rising', 'falling', or 'stable'."""
        window = self.surprise_history[-20:]
        if len(window) < 3:
            return "stable"
        # Simple linear slope via numpy
        x = np.arange(len(window), dtype=float)
        y = np.array(window, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        if slope > 0.005:
            return "rising"
        elif slope < -0.005:
            return "falling"
        return "stable"

    def get_prediction_confidence(self) -> float:
        """Returns current precision value (0-1)."""
        return float(self.precision)

    def get_surprise_signal(self) -> float:
        """Returns total_surprise — the value the heartbeat reads."""
        return float(self.total_surprise)

    def accept_feedback(self, actual_outcome: Dict[str, Any]):
        """Accept actual outcome, compute surprise, and push to free_energy_engine."""
        actual_substrate = actual_outcome.get("substrate_x", None)
        if actual_substrate is not None and not isinstance(actual_substrate, np.ndarray):
            actual_substrate = np.array(actual_substrate, dtype=float)

        surprise = self.compute_surprise(actual_outcome, actual_substrate)

        try:
            fe = ServiceContainer.get("free_energy_engine", default=None)
            if fe and hasattr(fe, "accept_surprise_signal"):
                fe.accept_surprise_signal(surprise)
        except Exception as e:
            record_degradation('predictive_engine', e)
            logger.debug("accept_feedback → free_energy_engine: %s", e)

    def get_context_block(self) -> str:
        """Concise prediction stats for context injection (max 200 chars)."""
        current = self.surprise_history[-1] if self.surprise_history else 0.0
        trend = self.get_surprise_trend()
        return (
            f"[PRED] surprise={current:.2f} trend={trend} "
            f"precision={self.precision:.2f} "
            f"free_energy={self.free_energy:.2f}"
        )