"""core/brain/homeostatic_modulator.py
====================================
Computes real-time inference modulation parameters from homeostatic loops.
Maps FitzHugh-Nagumo, FreeEnergy, and LiquidSubstrate states to temperature,
top_p, repetition_penalty, and token-level logit biases.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.runtime.atomic_writer import atomic_write_text

logger = logging.getLogger("Aura.Brain.HomeostaticModulator")


@dataclass
class InferenceModulation:
    """Modulation parameters to be injected directly into LLM inference."""

    temperature: float
    top_p: float
    repetition_penalty: float
    logit_bias: Dict[int, float]
    head_weights: np.ndarray
    urgency: float
    source_snapshot: Dict[str, Any] = field(default_factory=dict)


class SubstrateLogitProjection:
    """Learned sparse mapping from substrate state to token logit biases.

    Uses Hebbian plasticity: when generation succeeds (positive coherence,
    low surprise), strengthen the connection between current substrate state
    and generated tokens. When surprise is high or coherence is negative, weaken.
    """

    def __init__(self, substrate_dim: int = 512, save_path: str | None = None) -> None:
        self.substrate_dim = substrate_dim
        # Map: token_id -> np.ndarray of shape (substrate_dim,) containing association weights
        self.weights: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        
        if save_path:
            self.save_path = Path(save_path)
        else:
            try:
                from core.config import config as aura_config
                self.save_path = aura_config.paths.data_dir / "substrate_logit_projection.json"
            except Exception:
                self.save_path = Path("data/substrate_logit_projection.json")

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def get_biases(self, substrate_state: np.ndarray) -> Dict[int, float]:
        """Compute logit biases for all tokens based on the current substrate state."""
        with self._lock:
            if not self.weights or len(substrate_state) == 0:
                return {}

            biases: Dict[int, float] = {}
            # Resize substrate state to match weights dimensions if necessary
            state = substrate_state
            if len(state) != self.substrate_dim:
                resized = np.zeros(self.substrate_dim, dtype=np.float32)
                copy_len = min(len(state), self.substrate_dim)
                resized[:copy_len] = state[:copy_len]
                state = resized

            # Compute dot product for each token
            for token_id, w in self.weights.items():
                dot = float(np.dot(w, state))
                # Soft clamp logit bias to [-2.0, 2.0] to prevent model breakdown
                bias = float(np.clip(dot * 0.5, -2.0, 2.0))
                if abs(bias) > 0.05:
                    biases[token_id] = bias

            return biases

    def learn_step(
        self,
        substrate_state: np.ndarray,
        token_ids: list[int],
        feedback_coherence: float,
        surprise: float,
        lr: float = 0.005
    ) -> None:
        """Update projection weights via reward-modulated Hebbian learning.

        Feedback coherence: positive when output is aligned with substrate goals.
        Surprise: prediction error/surprise of the generated text (perplexity).
        """
        with self._lock:
            if len(substrate_state) == 0 or not token_ids:
                return

            state = substrate_state
            if len(state) != self.substrate_dim:
                resized = np.zeros(self.substrate_dim, dtype=np.float32)
                copy_len = min(len(state), self.substrate_dim)
                resized[:copy_len] = state[:copy_len]
                state = resized

            # Reward signal: High coherence and low surprise = high reward
            # E.g., reward range roughly [-1.0, 1.0]
            reward = feedback_coherence * math.exp(-surprise)

            for token_id in token_ids:
                if token_id not in self.weights:
                    self.weights[token_id] = np.zeros(self.substrate_dim, dtype=np.float32)
                
                # Hebbian weight update: dW = lr * reward * state
                self.weights[token_id] += lr * reward * state
                
                # Weight decay/regularization to prevent unbounded growth
                self.weights[token_id] *= 0.98
                
                # Clip weight vector for stability
                self.weights[token_id] = np.clip(self.weights[token_id], -1.0, 1.0)

            # Prune near-zero weights to keep memory usage sparse
            inactive = [tid for tid, w in self.weights.items() if np.linalg.norm(w) < 1e-4]
            for tid in inactive:
                del self.weights[tid]

    # -- Persistence -----------------------------------------------------------

    def save(self) -> None:
        """Persist weights to disk."""
        with self._lock:
            payload = {
                "substrate_dim": self.substrate_dim,
                "weights": {
                    str(tid): w.tolist()
                    for tid, w in self.weights.items()
                }
            }
            try:
                atomic_write_text(self.save_path, json.dumps(payload, indent=2), encoding="utf-8")
                logger.debug("Persisted SubstrateLogitProjection to: %s", self.save_path)
            except Exception as exc:
                logger.error("Failed to save SubstrateLogitProjection: %s", exc)

    def load(self) -> None:
        """Restore weights from disk."""
        if not self.save_path.exists():
            return
        try:
            payload = json.loads(self.save_path.read_text(encoding="utf-8"))
            self.substrate_dim = int(payload.get("substrate_dim", self.substrate_dim))
            weights_raw = payload.get("weights", {})
            with self._lock:
                self.weights = {
                    int(tid): np.asarray(w, dtype=np.float32)
                    for tid, w in weights_raw.items()
                }
            logger.info("Loaded SubstrateLogitProjection from: %s", self.save_path)
        except Exception as exc:
            logger.error("Failed to load SubstrateLogitProjection: %s", exc)


class HomeostaticModulator:
    """Coordinates and converts live states from the FitzHugh-Nagumo oscillator,
    Free Energy engine, and Liquid Substrate into InferenceModulation parameters.
    """

    def __init__(self, substrate_dim: int = 512) -> None:
        self.projection = SubstrateLogitProjection(substrate_dim=substrate_dim)

    def compute_modulation(self) -> InferenceModulation:
        """Read and bundle current state values into an InferenceModulation object."""
        from core.container import ServiceContainer

        # 1. Retrieve engines
        precision_engine = ServiceContainer.get("precision_engine", default=None)
        free_energy_engine = ServiceContainer.get("free_energy_engine", default=None)
        substrate = ServiceContainer.get("liquid_substrate", default=None)

        # 2. Extract raw states
        arousal = 0.5
        fatigue = 0.0
        head_weights = np.ones(32, dtype=np.float32)
        if precision_engine:
            arousal = float(precision_engine.fhn.arousal)
            fatigue = float(precision_engine.fhn.fatigue)
            head_weights = precision_engine.get_head_weights()

        free_energy = 0.3
        urgency = 0.5
        if free_energy_engine:
            # Handle smoothed free energy
            free_energy = float(getattr(free_energy_engine, "_smoothed_fe", 0.3))
            # Determine action urgency
            urgency = float(free_energy_engine.get_action_urgency()) if hasattr(free_energy_engine, "get_action_urgency") else free_energy

        frustration = 0.0
        substrate_state = np.zeros(self.projection.substrate_dim, dtype=np.float32)
        if substrate:
            frustration = float(substrate.x[substrate.idx_frustration]) if substrate.idx_frustration < len(substrate.x) else 0.0
            # Get full substrate activation vector
            with substrate.sync_lock:
                substrate_state = substrate.x.copy()

        # 3. Parameter Mapping
        # Temperature is modulated by precision arousal (high arousal -> focused, lower temp; low arousal -> exploratory, higher temp)
        # Use precision_engine's calculation, with fallback
        if precision_engine:
            temperature = float(precision_engine.get_temperature())
        else:
            temperature = float(0.95 - 0.40 * arousal)

        # Repetition penalty scales up with substrate frustration to avoid looping
        repetition_penalty = float(1.1 + 0.3 * frustration)

        # Top_p scales down with free energy to constrain responses when highly Surprised/Complex
        top_p = float(max(0.6, min(1.0, 0.95 - 0.25 * free_energy)))

        # Logit bias derived from substrate Hebbian projection
        logit_bias = self.projection.get_biases(substrate_state)

        # 4. Snapshot source values for telemetry/debugging
        source_snapshot = {
            "fhn_arousal": arousal,
            "fhn_fatigue": fatigue,
            "free_energy": free_energy,
            "substrate_frustration": frustration,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logit_bias_count": len(logit_bias)
        }

        return InferenceModulation(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            head_weights=head_weights,
            urgency=urgency,
            source_snapshot=source_snapshot
        )
