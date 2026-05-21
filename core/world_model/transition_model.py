"""core/world_model/transition_model.py
=====================================
Online-updating LMS Delta Rule linear transition predictor for substrate state-action transitions.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.resilience.cognitive_ledger import Transition, TransitionType, get_cognitive_ledger

logger = logging.getLogger("Aura.WorldModel.TransitionModel")

_SINGLETON_LOCK = threading.Lock()
_transition_model_instance: Optional[TransitionModel] = None


class TransitionModel:
    """Mathematical linear transition model that predicts next substrate state and learns online."""

    # 8-dimensional sparse action mapping
    ACTION_MAP = {
        "file_read": 0,
        "file_write": 1,
        "file_delete": 2,
        "command_execute": 3,
        "run_test": 4,
        "patch_code": 5,
        "commit_code": 6,
        "reflect": 7
    }

    def __init__(self, learning_rate: float = 0.05) -> None:
        self._lock = threading.Lock()
        self.learning_rate = learning_rate
        
        # Dimensions: state (7) + action (8) = 15. Outputs state (7).
        # W maps R^15 -> R^7. Initialize to small random normal weights.
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(7, 15)).astype(np.float32)
        
        # Tracking variables
        self.last_state: Optional[np.ndarray] = None
        self.last_action_vec: Optional[np.ndarray] = None
        self.last_action_name: str = "reflect"
        
        self.prediction_error_history: List[float] = []
        logger.info("TransitionModel initialized with learning rate: %.4f", self.learning_rate)

    def extract_state_vector(self) -> np.ndarray:
        """Extracts the 7-dimensional continuous substrate state vector."""
        state = np.zeros(7, dtype=np.float32)
        
        # 1. Liquid Substrate metrics (dims 0-4)
        substrate = ServiceContainer.get("liquid_substrate", default=None)
        if substrate:
            try:
                with substrate.sync_lock:
                    state[0] = float(substrate.x[substrate.idx_valence])
                    state[1] = float(substrate.x[substrate.idx_arousal])
                    state[2] = float(substrate.x[substrate.idx_frustration])
                    state[3] = float(substrate.x[substrate.idx_curiosity])
                    state[4] = float(substrate.x[substrate.idx_focus])
            except Exception as exc:
                logger.debug("Failed extracting substrate state: %s", exc)
                
        # 2. Free Energy Engine (dim 5)
        free_energy = ServiceContainer.get("free_energy_engine", default=None)
        if free_energy:
            try:
                state[5] = float(free_energy.smoothed_fe)
            except Exception as exc:
                logger.debug("Failed extracting free energy: %s", exc)
                
        # 3. Precision Engine (dim 6)
        precision = ServiceContainer.get("precision_engine", default=None)
        if precision:
            try:
                # Thread-safe read of FHN arousal
                state[6] = float(precision.fhn.arousal)
            except Exception as exc:
                logger.debug("Failed extracting precision arousal: %s", exc)
                
        return state

    def encode_action_vector(self, action_name: str) -> np.ndarray:
        """Creates an 8-dimensional one-hot representation of a symbolic action."""
        vec = np.zeros(8, dtype=np.float32)
        idx = self.ACTION_MAP.get(action_name, 7)  # Fallback to reflect
        vec[idx] = 1.0
        return vec

    def predict_next_state(self, current_state: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """Computes the transition prediction: s_hat_t+1 = s_t + W * [s_t, a_t]."""
        with self._lock:
            # Concatenate state and action vector to form joint input x_t (R^15)
            x_t = np.concatenate([current_state, action_vec])
            
            # Predict difference vector
            delta_s = np.dot(self.W, x_t)
            
            # Predicted state is s_t + delta_s
            predicted_state = current_state + delta_s
            
            # Clamp predicted state to physical boundaries [0.0, 1.0]
            return np.clip(predicted_state, 0.0, 1.0)

    def simulate_action(self, action_name: str) -> Dict[str, Any]:
        """Simulates the consequences of executing an action, modeling state changes,
        resource cost, risk, reversibility, and potential structural effects.
        """
        current_state = self.extract_state_vector()
        action_vec = self.encode_action_vector(action_name)
        predicted_next = self.predict_next_state(current_state, action_vec)
        
        # Calculate state differences
        state_deltas = (predicted_next - current_state).tolist()
        
        # Establish domain-specific heuristics for physical side-effects
        # (reversibility, resource consumption, file impact, memory impact)
        reversibility = 1.0
        risk = 0.05
        resource_cost = "low"
        files_changed = []
        memory_changed = False
        user_visible = False
        possible_failures = []
        
        if action_name == "file_write":
            reversibility = 0.8  # Overwriting files is partially reversible via backup
            risk = 0.15
            resource_cost = "low"
            files_changed = ["target_file"]
        elif action_name == "file_delete":
            reversibility = 0.1  # Deleting files is highly irreversible
            risk = 0.4
            resource_cost = "low"
            files_changed = ["target_file"]
            possible_failures = ["FileNotFoundError", "PermissionError"]
        elif action_name == "command_execute":
            reversibility = 0.5
            risk = 0.3
            resource_cost = "medium"
            user_visible = True
            possible_failures = ["ProcessTerminated", "TimeoutExpired"]
        elif action_name == "run_test":
            reversibility = 1.0  # Safe and read-only
            risk = 0.05
            resource_cost = "medium"
        elif action_name == "patch_code":
            reversibility = 0.7  # Reversible via git rollback
            risk = 0.25
            resource_cost = "low"
            files_changed = ["source_file"]
            possible_failures = ["SyntaxError", "PytestFailure"]
        elif action_name == "commit_code":
            reversibility = 0.9  # Fully reversible via git revert
            risk = 0.1
            resource_cost = "low"
            memory_changed = True
        elif action_name == "reflect":
            reversibility = 1.0
            risk = 0.0
            resource_cost = "none"
            memory_changed = True
            
        return {
            "proposed_action": action_name,
            "predicted_state_deltas": state_deltas,
            "resource_cost": resource_cost,
            "risk": risk,
            "reversibility": reversibility,
            "files_affected": files_changed,
            "memory_affected": memory_changed,
            "user_visible": user_visible,
            "possible_failures": possible_failures
        }

    def process_step(self, action_name: str) -> float:
        """Processes one causal world step: learns from past error, makes next prediction.
        
        Returns the magnitude of the prediction error.
        """
        current_state = self.extract_state_vector()
        action_vec = self.encode_action_vector(action_name)
        
        error_magnitude = 0.0
        
        with self._lock:
            # If we have a past state-action prediction, evaluate and update weights
            if self.last_state is not None and self.last_action_vec is not None:
                # 1. Compute past prediction s_hat_t
                x_prev = np.concatenate([self.last_state, self.last_action_vec])
                s_hat = np.clip(self.last_state + np.dot(self.W, x_prev), 0.0, 1.0)
                
                # 2. Compute prediction error: e_t = s_actual - s_hat
                e_t = current_state - s_hat
                error_magnitude = float(np.linalg.norm(e_t))
                
                self.prediction_error_history.append(error_magnitude)
                if len(self.prediction_error_history) > 100:
                    self.prediction_error_history.pop(0)
                    
                # 3. Apply Delta Rule LMS Weight Update: W <- W + lr * (e_t x x_prev^T)
                # Outer product produces a 7x15 gradient correction matrix
                weight_correction = self.learning_rate * np.outer(e_t, x_prev)
                self.W = np.clip(self.W + weight_correction, -2.0, 2.0)
                
                # 4. Record to CognitiveLedger in WAL sqlite mode
                self._record_to_ledger(self.last_state, self.last_action_name, s_hat, current_state, e_t)
                
            # Store current state-action pair for next step evaluation
            self.last_state = current_state
            self.last_action_vec = action_vec
            self.last_action_name = action_name
            
        return error_magnitude

    def _record_to_ledger(
        self, s_prev: np.ndarray, action: str, s_predicted: np.ndarray, s_actual: np.ndarray, e_t: np.ndarray
    ) -> None:
        """Assembles a Transition event and journals it directly to the CognitiveLedger."""
        try:
            prior_hash = hashlib.sha256(s_prev.tobytes()).hexdigest()[:12]
            next_hash = hashlib.sha256(s_actual.tobytes()).hexdigest()[:12]
            error_val = float(np.linalg.norm(e_t))
            
            t = Transition.create(
                ttype=TransitionType.BELIEF_REVISION,
                subsystem="world_model",
                cause=f"action_executed:{action}",
                payload={
                    "prior_state": s_prev.round(4).tolist(),
                    "action": action,
                    "predicted_state": s_predicted.round(4).tolist(),
                    "actual_state": s_actual.round(4).tolist(),
                    "prediction_error": e_t.round(4).tolist(),
                    "error_magnitude": round(error_val, 4)
                },
                prior_hash=prior_hash,
                confidence=float(1.0 - min(1.0, error_val)),
                uncertainty=float(min(1.0, error_val))
            )
            t.next_state_hash = next_hash
            
            ledger = get_cognitive_ledger()
            if ledger:
                ledger.append(t)
        except Exception as exc:
            record_degradation("transition_ledger_log", exc)
            logger.error("Failed to append transition prediction to CognitiveLedger: %s", exc)

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns diagnostic state and weight properties for serialization/telemetry."""
        with self._lock:
            mean_error = float(np.mean(self.prediction_error_history)) if self.prediction_error_history else 0.0
            return {
                "weights_shape": self.W.shape,
                "weights_mean": float(self.W.mean()),
                "weights_abs_sum": float(np.abs(self.W).sum()),
                "last_action": self.last_action_name,
                "mean_prediction_error": round(mean_error, 4),
                "error_history_len": len(self.prediction_error_history)
            }


def get_transition_model() -> TransitionModel:
    """Thread-safe accessor for the TransitionModel singleton."""
    global _transition_model_instance
    if _transition_model_instance is None:
        with _SINGLETON_LOCK:
            if _transition_model_instance is None:
                _transition_model_instance = TransitionModel()
                # Auto register in ServiceContainer
                ServiceContainer.register_instance("transition_model", _transition_model_instance)
    return _transition_model_instance
