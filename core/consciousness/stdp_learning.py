"""STDP-inspired online learning for the liquid substrate.

Inspired by BrainCog's Spike-Timing-Dependent Plasticity (STDP) rules.
Instead of fixed Hebbian learning rates, this modulates synaptic weight
updates based on prediction error from the active inference loop.

When the free energy engine reports high surprise, synaptic plasticity
increases — the substrate updates eligible traces more decisively because
something unexpected happened. Prediction error supplies the signed
modulatory reward: high error is negative, so the faster update weakens or
reverses the traces that helped produce the bad prediction instead of
reinforcing them.

This creates a genuine reward-modulated learning system: the substrate's
internal dynamics literally change based on how well it's predicting
the world.

Algorithm (Reward-modulated STDP, Izhikevich 2007):
  1. Compute eligibility trace: e(t) = e(t-1) * decay + STDP(pre, post)
  2. Get reward signal: r(t) = -tanh(prediction_error)
  3. Weight update: dw = learning_rate * r(t) * e(t)
  4. Apply to substrate connectivity matrix W

The STDP window follows the classical asymmetric shape:
  - Pre before post (causal): potentiation (strengthen)
  - Post before pre (anti-causal): depression (weaken)
"""
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.STDPLearning")

# STDP parameters
TAU_PLUS = 20.0          # Potentiation time constant (ms)
TAU_MINUS = 20.0         # Depression time constant (ms)
A_PLUS = 0.005           # Potentiation amplitude
A_MINUS = 0.005 * 1.05   # Depression amplitude (slightly stronger for stability)
ELIGIBILITY_DECAY = 0.95  # Eligibility trace decay per tick
BASE_LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 0.0001
WEIGHT_CLIP = 2.0         # Max absolute weight value


class STDPLearningEngine:
    """Reward-modulated STDP for the liquid substrate."""

    def __init__(self, n_neurons: int = 64):
        self.n = n_neurons

        # Eligibility traces: accumulate STDP signals between reward deliveries
        self._eligibility = np.zeros((n_neurons, n_neurons), dtype=np.float32)

        # Spike timing records (last spike time per neuron)
        self._last_spike_time = np.full(n_neurons, -1000.0, dtype=np.float32)

        # Current modulated learning rate
        self._learning_rate = BASE_LEARNING_RATE

        # Stats
        self._total_updates = 0
        self._total_potentiations = 0
        self._total_depressions = 0
        self._last_reward = 0.0
        self._last_surprise = 0.0

    def record_spikes(self, activations: np.ndarray, t: float):
        """Record which neurons fired and update eligibility traces.

        Vectorized implementation: uses NumPy broadcasting to compute
        all pairwise STDP contributions in O(N) instead of O(N²).

        Args:
            activations: Current activation vector (n_neurons,). Values > 0.5
                        are considered "spikes".
            t: Current time in milliseconds.
        """
        threshold = 0.5
        spiking = activations > threshold
        spike_indices = np.where(spiking)[0]

        if len(spike_indices) == 0:
            # Still decay eligibility even with no spikes
            self._eligibility *= ELIGIBILITY_DECAY
            return

        # Vectorized STDP: compute dt for all (pre, post) pairs at once
        # dt_matrix[i, j] = t - last_spike_time[i] for post neuron j
        # Potentiation: pre fires before post (dt > 0)
        # Depression: post fires before pre (dt < 0)
        dt_vec = t - self._last_spike_time  # (N,) time since each neuron last fired

        # Create spike mask for broadcasting
        post_mask = spiking.astype(np.float32)  # (N,) 1 where post fired

        # Potentiation contribution: for each pre neuron, if any post fired
        # dt_vec[pre] > 0 means pre fired before now (causal)
        pot_contribution = A_PLUS * np.exp(-np.clip(dt_vec, 0, 200) / TAU_PLUS)
        # Shape: (N_pre,) x (N_post,) -> outer product
        pot_matrix = np.outer(pot_contribution, post_mask)

        # Depression contribution: for post neurons that fired,
        # their prior spike was at t (they just fired), so for pre neurons
        # that fired AFTER, dt < 0 -> depression
        # We use the negative dt for pre neurons that fired recently
        dep_dt = -dt_vec  # negative = post before pre
        dep_contribution = -A_MINUS * np.exp(np.clip(dep_dt, -200, 0) / TAU_MINUS)
        dep_matrix = np.outer(np.ones(self.n, dtype=np.float32), post_mask) * dep_contribution[:, np.newaxis]

        self._eligibility += pot_matrix + dep_matrix

        # Zero self-connections
        np.fill_diagonal(self._eligibility, 0.0)

        # Update last spike times
        self._last_spike_time[spiking] = t

        # Decay eligibility traces
        self._eligibility *= ELIGIBILITY_DECAY

        # NaN guard
        self._eligibility = np.nan_to_num(self._eligibility, nan=0.0)

    def deliver_reward(self, surprise: float, prediction_error: float) -> np.ndarray:
        """Apply reward-modulated weight update based on prediction error.

        Args:
            surprise: Current surprise value from free energy engine (0-1).
            prediction_error: Current prediction error (higher = worse prediction).

        Returns:
            Weight delta matrix (n x n) to apply to substrate connectivity.
        """
        # Reward signal: negative prediction error (lower error = less
        # negative reward). Surprise changes step size below; this signed
        # reward decides whether eligible traces are reinforced or depressed.
        reward = -np.tanh(prediction_error)
        self._last_reward = float(reward)
        self._last_surprise = float(surprise)

        # Modulate learning rate by surprise. High surprise increases the
        # magnitude of corrective plasticity; the reward sign above still
        # decides the direction of the update.
        self._learning_rate = np.clip(
            BASE_LEARNING_RATE * (1.0 + surprise * 5.0),
            MIN_LEARNING_RATE,
            MAX_LEARNING_RATE,
        )

        # Compute weight delta: dw = lr * reward * eligibility
        dw = self._learning_rate * reward * self._eligibility

        # Clip to prevent runaway weights
        dw = np.clip(dw, -WEIGHT_CLIP * 0.01, WEIGHT_CLIP * 0.01)

        # Stats
        self._total_updates += 1
        self._total_potentiations += int(np.sum(dw > 0))
        self._total_depressions += int(np.sum(dw < 0))

        return dw

    def apply_to_connectivity(self, W: np.ndarray, dw: np.ndarray) -> np.ndarray:
        """Apply the weight delta to a connectivity matrix.

        Args:
            W: Current connectivity matrix (n x n).
            dw: Weight delta from deliver_reward().

        Returns:
            Updated connectivity matrix.
        """
        W_new = W + dw

        # Clip weights
        W_new = np.clip(W_new, -WEIGHT_CLIP, WEIGHT_CLIP)

        # Zero diagonal (no self-connections)
        np.fill_diagonal(W_new, 0.0)

        # Symmetry breaking: prevent W from becoming symmetric (W = W^T)
        # which collapses dynamical richness. Add a small antisymmetric
        # perturbation every 50 updates.
        self._total_updates_since_symmetry_break = getattr(
            self, '_total_updates_since_symmetry_break', 0) + 1
        if self._total_updates_since_symmetry_break >= 50:
            asymmetry = 0.001 * (W_new - W_new.T)
            W_new += asymmetry
            W_new = np.clip(W_new, -WEIGHT_CLIP, WEIGHT_CLIP)
            np.fill_diagonal(W_new, 0.0)
            self._total_updates_since_symmetry_break = 0

        return W_new

    def get_status(self) -> Dict:
        return {
            "learning_rate": round(self._learning_rate, 6),
            "last_reward": round(self._last_reward, 4),
            "last_surprise": round(self._last_surprise, 4),
            "total_updates": self._total_updates,
            "total_potentiations": self._total_potentiations,
            "total_depressions": self._total_depressions,
            "eligibility_norm": round(float(np.linalg.norm(self._eligibility)), 4),
            "eligibility_sparsity": round(
                float(np.sum(np.abs(self._eligibility) < 1e-6)) / max(self.n * self.n, 1), 3
            ),
        }


_instance: Optional[STDPLearningEngine] = None


def get_stdp_engine(n_neurons: int = 64) -> STDPLearningEngine:
    global _instance
    if _instance is None:
        _instance = STDPLearningEngine(n_neurons=n_neurons)
    return _instance
