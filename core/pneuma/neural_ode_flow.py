"""core/pneuma/neural_ode_flow.py
PNEUMA Layer 2 — Neural ODE Continuous Belief Flow.

Integrates belief state vectors continuously using RK4 with adaptive step size.
The flow network models how belief distributions evolve over time under the
influence of new evidence, affect, and prior knowledge.

Belief vector b ∈ ℝ^d represents a compressed embedding of Aura's current
world-model confidence distribution. The ODE:

    db/dt = f_θ(b, t) + η(t)

where f_θ is the learned flow field (approximated via small MLP) and η(t) is
noise from the FHN metabolic state.
"""

from core.runtime.errors import record_degradation
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PNEUMA.NeuralODEFlow")


class BeliefFlowNetwork:
    """Lightweight MLP that approximates the belief flow field f_θ(b, t).

    Architecture: b (dim) → tanh → hidden → tanh → b (dim)
    Weights are random at init and evolve via Hebbian-like updates from
    evidence (no backprop dependency required — this runs online).
    """

    def __init__(self, dim: int = 64, hidden: int = 128, seed: int = 42, hebbian_top_k: int = 8):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (hidden, dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (dim, hidden)).astype(np.float32)
        self.b2 = np.zeros(dim, dtype=np.float32)
        self.dim = dim
        self.hidden = hidden
        self._lr = 1e-4  # slow Hebbian learning rate
        self._hebbian_top_k = max(1, min(int(hebbian_top_k or 1), dim))

    def forward(self, b: np.ndarray, t: float) -> np.ndarray:
        """Compute flow field f_θ(b, t)."""
        # Time encoding: sin/cos pair appended conceptually via modulation
        t_mod = np.sin(t * 0.1) * 0.1
        h = np.tanh(self.W1 @ b + self.b1 + t_mod)
        out = np.tanh(self.W2 @ h + self.b2)
        return out

    def hebbian_update(self, b_pre: np.ndarray, b_post: np.ndarray):
        """Soft Hebbian update: W += lr * outer(b_post, b_pre)."""
        if b_pre.shape[0] != self.dim or b_post.shape[0] != self.dim:
            return
        try:
            delta = (b_post - b_pre).astype(np.float32, copy=False)
            if not np.any(delta):
                return
            h = np.tanh(self.W1 @ b_pre + self.b1)
            abs_delta = np.abs(delta)
            if self._hebbian_top_k < self.dim:
                active_idx = np.argpartition(abs_delta, -self._hebbian_top_k)[-self._hebbian_top_k:]
            else:
                active_idx = np.arange(self.dim)
            if active_idx.size == 0:
                return
            rows = self.W2[active_idx].copy()
            rows += self._lr * np.outer(delta[active_idx], h)
            np.clip(rows, -2.0, 2.0, out=rows)
            self.W2[active_idx] = rows
        except Exception as _exc:
            record_degradation('neural_ode_flow', _exc)
            logger.debug("Suppressed Exception: %s", _exc)


@dataclass
class BeliefState:
    vector: np.ndarray
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    source: str = "unknown"


class NeuralODEFlow:
    """Continuous belief integrator using RK4.

    Maintains a rolling belief state vector that is updated:
    1. Autonomously each tick via the flow network
    2. On evidence injection from EpistemicFilter
    3. On affect changes from AffectiveCircumplex
    """

    def __init__(self, dim: int = 64, max_dt: float = 1.0):
        self.dim = dim
        self.max_dt = max_dt
        self.flow_net = BeliefFlowNetwork(dim=dim)
        self._state = BeliefState(
            vector=np.zeros(dim, dtype=np.float32),
            confidence=0.5,
            source="init",
        )
        self._t = 0.0
        self._history: List[BeliefState] = []
        self._max_history = 200
        self._lock = threading.Lock()
        self._step_count = 0
        self._hebbian_update_every = 3
        logger.info("NeuralODEFlow online (dim=%d)", dim)

    @property
    def current_belief(self) -> BeliefState:
        with self._lock:
            return self._state

    def _rk4_step(self, b: np.ndarray, t: float, dt: float) -> np.ndarray:
        """RK4 integration of db/dt = f_θ(b, t)."""
        k1 = self.flow_net.forward(b, t)
        k2 = self.flow_net.forward(b + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.flow_net.forward(b + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.flow_net.forward(b + dt * k3, t + dt)
        return b + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, dt: Optional[float] = None) -> BeliefState:
        """Advance belief state by dt seconds."""
        with self._lock:
            if dt is None:
                dt = min(self.max_dt, time.time() - self._state.timestamp)
            dt = max(0.001, min(self.max_dt, dt))

            new_vec = self._rk4_step(self._state.vector, self._t, dt)
            new_vec = np.clip(new_vec, -3.0, 3.0)
            self._t += dt

            prev = self._state
            self._state = BeliefState(
                vector=new_vec.astype(np.float32),
                timestamp=time.time(),
                confidence=self._state.confidence,
                source="ode_step",
            )
            self._step_count += 1
            should_update_hebbian = (
                self._step_count == 1
                or (self._step_count % self._hebbian_update_every) == 0
                or dt >= (self.max_dt * 0.5)
            )
            if should_update_hebbian:
                self.flow_net.hebbian_update(prev.vector, new_vec)
            self._record_history(self._state)
            return self._state

    def inject_evidence(self, embedding: np.ndarray, weight: float = 0.3, source: str = "evidence"):
        """Perturb belief vector with new evidence embedding.

        Blends current belief with evidence: b' = (1-w)*b + w*e
        """
        if embedding.shape[0] != self.dim:
            # Resize via truncation / zero-padding
            e = np.zeros(self.dim, dtype=np.float32)
            n = min(self.dim, len(embedding))
            e[:n] = embedding[:n]
            embedding = e

        with self._lock:
            weight = max(0.0, min(1.0, weight))
            new_vec = (1.0 - weight) * self._state.vector + weight * embedding.astype(np.float32)
            self._state = BeliefState(
                vector=new_vec,
                timestamp=time.time(),
                confidence=min(1.0, self._state.confidence + 0.05),
                source=source,
            )
            self._record_history(self._state)

    def inject_affect(self, valence: float, arousal: float):
        """Modulate belief vector with affective shift.

        Affect creates a small directional perturbation in belief space.
        """
        perturbation = np.zeros(self.dim, dtype=np.float32)
        # First two dims encode valence/arousal directly
        perturbation[0] = (valence - 0.5) * 0.1
        perturbation[1] = (arousal - 0.5) * 0.1
        with self._lock:
            self._state.vector = np.clip(
                self._state.vector + perturbation, -3.0, 3.0
            )

    def get_belief_distance(self, other_vec: np.ndarray) -> float:
        """L2 distance between current belief and another vector."""
        if other_vec.shape[0] != self.dim:
            return float("inf")
        with self._lock:
            return float(np.linalg.norm(self._state.vector - other_vec))

    def get_state_dict(self) -> dict:
        with self._lock:
            return {
                "belief_norm": round(float(np.linalg.norm(self._state.vector)), 4),
                "belief_confidence": round(self._state.confidence, 4),
                "ode_time": round(self._t, 2),
                "history_len": len(self._history),
            }

    def _record_history(self, state: BeliefState):
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
