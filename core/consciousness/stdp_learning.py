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

MESU Extension (Meta-learning with Elastic Synaptic Uncertainty):
  Each synapse gets a per-parameter learning rate scaled by its posterior
  uncertainty, approximated via the diagonal of the empirical Fisher
  (Hessian diagonal). Weights that encode core identity (low uncertainty,
  consistently activated) become "locked" while task-specific weights
  (high uncertainty) remain flexible. This prevents catastrophic
  forgetting at the STDP level.

  uncertainty(w_ij) = running_var(grad(w_ij))
  lr(w_ij) = base_lr * sigmoid(uncertainty(w_ij) / tau_mesu)
  lock(w_ij) iff uncertainty(w_ij) < lock_threshold for > lock_window updates
"""
import logging

import numpy as np

from core.runtime.errors import record_degradation

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

# MESU parameters
MESU_TAU = 0.5           # Temperature for sigmoid scaling of uncertainty→lr
MESU_LOCK_THRESHOLD = 0.01  # Uncertainty below this → identity-locked
MESU_LOCK_WINDOW = 100   # Must be below threshold for this many updates
MESU_EMA_ALPHA = 0.02    # Exponential moving average for uncertainty tracking


class STDPLearningEngine:
    """Reward-modulated STDP for the liquid substrate with MESU plasticity."""

    def __init__(self, n_neurons: int = 64):
        self.n = n_neurons

        # Eligibility traces: accumulate STDP signals between reward deliveries
        self._eligibility = np.zeros((n_neurons, n_neurons), dtype=np.float32)

        # Spike timing records (last spike time per neuron)
        self._last_spike_time = np.full(n_neurons, -1000.0, dtype=np.float32)

        # Current modulated learning rate
        self._learning_rate = BASE_LEARNING_RATE

        # ── MESU: Per-synapse uncertainty tracking ──────────────────────
        # Running mean and variance of weight deltas (Welford online)
        self._mesu_mean = np.zeros((n_neurons, n_neurons), dtype=np.float64)
        self._mesu_var = np.ones((n_neurons, n_neurons), dtype=np.float64) * 0.5
        self._mesu_count = np.zeros((n_neurons, n_neurons), dtype=np.int32)

        # Per-synapse learning rate multiplier (0..1)
        self._mesu_lr_scale = np.ones((n_neurons, n_neurons), dtype=np.float32)

        # Identity lock mask: True = weight is locked (core identity)
        self._mesu_locked = np.zeros((n_neurons, n_neurons), dtype=bool)
        # How many consecutive updates each weight has been below lock threshold
        self._mesu_stable_count = np.zeros((n_neurons, n_neurons), dtype=np.int32)

        # Stats
        self._total_updates = 0
        self._total_potentiations = 0
        self._total_depressions = 0
        self._last_reward = 0.0
        self._last_surprise = 0.0
        self._locked_count = 0
        self._mesu_mean_uncertainty = 0.0

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
        """Apply reward-modulated weight update with MESU uncertainty scaling.

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

        # Modulate global learning rate by surprise. High surprise increases the
        # magnitude of corrective plasticity; the reward sign above still
        # decides the direction of the update.
        self._learning_rate = np.clip(
            BASE_LEARNING_RATE * (1.0 + surprise * 5.0),
            MIN_LEARNING_RATE,
            MAX_LEARNING_RATE,
        )

        # Compute raw weight delta: dw = lr * reward * eligibility
        dw_raw = self._learning_rate * reward * self._eligibility

        # ── MESU: Update per-synapse uncertainty ──────────────────────
        # Track running variance of weight deltas using Welford online algorithm
        self._mesu_count += 1
        delta_from_mean = dw_raw - self._mesu_mean
        self._mesu_mean += MESU_EMA_ALPHA * delta_from_mean
        self._mesu_var = (
            (1.0 - MESU_EMA_ALPHA) * self._mesu_var
            + MESU_EMA_ALPHA * delta_from_mean * (dw_raw - self._mesu_mean)
        )
        # Clamp variance to prevent numerical issues
        self._mesu_var = np.clip(self._mesu_var, 1e-10, 10.0)

        # Compute per-synapse learning rate: high uncertainty → high lr
        # low uncertainty → low lr (identity protection)
        uncertainty = np.sqrt(self._mesu_var)
        self._mesu_mean_uncertainty = float(np.mean(uncertainty))

        # Sigmoid scaling: lr_scale = sigmoid((uncertainty - threshold) / tau)
        self._mesu_lr_scale = (
            1.0 / (1.0 + np.exp(-(uncertainty - MESU_LOCK_THRESHOLD) / MESU_TAU))
        ).astype(np.float32)

        # ── MESU: Identity locking ────────────────────────────────────
        # Weights that have been consistently low-uncertainty get locked
        low_uncertainty_mask = uncertainty < MESU_LOCK_THRESHOLD
        self._mesu_stable_count[low_uncertainty_mask] += 1
        self._mesu_stable_count[~low_uncertainty_mask] = 0

        # Lock weights that have been stable for long enough
        newly_locked = (
            (self._mesu_stable_count >= MESU_LOCK_WINDOW)
            & (~self._mesu_locked)
        )
        self._mesu_locked |= newly_locked
        self._locked_count = int(np.sum(self._mesu_locked))

        if int(np.sum(newly_locked)) > 0:
            logger.debug(
                "MESU: %d synapses newly identity-locked (total locked: %d/%d)",
                int(np.sum(newly_locked)),
                self._locked_count,
                self.n * self.n,
            )

        # Apply MESU scaling: locked weights get zero update
        mesu_mask = self._mesu_lr_scale.copy()
        mesu_mask[self._mesu_locked] = 0.0

        dw = dw_raw * mesu_mask

        # Clip to prevent runaway weights
        dw = np.clip(dw, -WEIGHT_CLIP * 0.01, WEIGHT_CLIP * 0.01)

        # Stats
        self._total_updates += 1
        self._total_potentiations += int(np.sum(dw > 0))
        self._total_depressions += int(np.sum(dw < 0))

        return dw

    def apply_to_connectivity(self, weights: np.ndarray, dw: np.ndarray) -> np.ndarray:
        """Apply the weight delta to a connectivity matrix.

        Includes:
          - Weight clipping
          - Spectral norm cap (prevents runaway growth)
          - Homeostatic scaling (keeps mean|W| near target)
          - Symmetry breaking

        Args:
            weights: Current connectivity matrix (n x n).
            dw: Weight delta from deliver_reward().

        Returns:
            Updated connectivity matrix.
        """
        updated_weights = weights + dw

        # Clip weights
        updated_weights = np.clip(updated_weights, -WEIGHT_CLIP, WEIGHT_CLIP)

        # Zero diagonal (no self-connections)
        np.fill_diagonal(updated_weights, 0.0)

        # ── Spectral Norm Cap ─────────────────────────────────────────
        # Prevent the largest singular value from exceeding a safe bound.
        # This caps the maximum gain of the dynamical system, preventing
        # explosive state growth that leads to NaN in the substrate ODE.
        spectral_norm_cap = 3.0
        try:
            s_max = np.linalg.norm(updated_weights, ord=2)  # Largest singular value
            if s_max > spectral_norm_cap:
                updated_weights *= spectral_norm_cap / s_max
                logger.debug(
                    "STDP: Spectral norm capped: %.3f -> %.3f",
                    s_max, spectral_norm_cap,
                )
        except np.linalg.LinAlgError as exc:
            record_degradation("stdp_learning", exc)
            logger.debug("STDP spectral norm cap skipped after linear algebra failure: %s", exc)

        # ── Homeostatic Scaling ───────────────────────────────────────
        # Keep the mean absolute weight near a target value. This prevents
        # both mode collapse (all weights → 0) and explosive growth
        # (all weights → max). The scaling is gentle (0.5% per tick).
        homeostatic_target = 0.3
        homeostatic_rate = 0.005
        mean_abs = float(np.mean(np.abs(updated_weights)))
        if mean_abs > 1e-8:
            ratio = homeostatic_target / mean_abs
            # Only apply if drift is significant (>10% from target)
            if abs(ratio - 1.0) > 0.1:
                # Gentle correction — move 0.5% toward target per tick
                correction = 1.0 + homeostatic_rate * (ratio - 1.0)
                updated_weights *= correction

        # ── NaN/Inf Guard ─────────────────────────────────────────────
        if not np.isfinite(updated_weights).all():
            nan_count = int(np.sum(~np.isfinite(updated_weights)))
            logger.warning(
                "STDP: %d NaN/Inf values in weight matrix — clamping to zero.",
                nan_count,
            )
            updated_weights = np.nan_to_num(
                updated_weights,
                nan=0.0,
                posinf=WEIGHT_CLIP,
                neginf=-WEIGHT_CLIP,
            )
            try:
                from core.observability.metrics import get_metrics
                get_metrics().increment_counter("stdp_nan_events_total")
            except (AttributeError, ImportError, RuntimeError, TypeError) as exc:
                record_degradation("stdp_learning", exc)
                logger.debug("STDP NaN metric emission failed: %s", exc)

        # Symmetry breaking: prevent W from becoming symmetric (W = W^T)
        # which collapses dynamical richness. Add a small antisymmetric
        # perturbation every 50 updates.
        self._total_updates_since_symmetry_break = getattr(
            self, '_total_updates_since_symmetry_break', 0) + 1
        if self._total_updates_since_symmetry_break >= 50:
            asymmetry = 0.001 * (updated_weights - updated_weights.T)
            updated_weights += asymmetry
            updated_weights = np.clip(updated_weights, -WEIGHT_CLIP, WEIGHT_CLIP)
            np.fill_diagonal(updated_weights, 0.0)
            self._total_updates_since_symmetry_break = 0

        return updated_weights

    def get_status(self) -> dict:
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
            # MESU telemetry
            "mesu_locked_count": self._locked_count,
            "mesu_locked_fraction": round(
                self._locked_count / max(self.n * self.n, 1), 4
            ),
            "mesu_mean_uncertainty": round(self._mesu_mean_uncertainty, 6),
            "mesu_mean_lr_scale": round(float(np.mean(self._mesu_lr_scale)), 4),
        }

    def get_mesu_diagnostics(self) -> dict:
        """Return detailed MESU diagnostics for observability."""
        uncertainty = np.sqrt(self._mesu_var)
        return {
            "uncertainty_mean": float(np.mean(uncertainty)),
            "uncertainty_std": float(np.std(uncertainty)),
            "uncertainty_min": float(np.min(uncertainty)),
            "uncertainty_max": float(np.max(uncertainty)),
            "lr_scale_mean": float(np.mean(self._mesu_lr_scale)),
            "lr_scale_std": float(np.std(self._mesu_lr_scale)),
            "locked_count": self._locked_count,
            "locked_fraction": self._locked_count / max(self.n * self.n, 1),
            "stable_count_mean": float(np.mean(self._mesu_stable_count)),
            "total_synapses": self.n * self.n,
        }

    def unlock_weights(self, mask: np.ndarray | None = None) -> int:
        """Unlock identity-locked weights (requires Will approval).

        Args:
            mask: Optional (n x n) boolean mask. If provided, only unlock
                  where mask is True. If None, unlock all.

        Returns:
            Number of weights unlocked.
        """
        if mask is None:
            count = int(np.sum(self._mesu_locked))
            self._mesu_locked[:] = False
            self._mesu_stable_count[:] = 0
        else:
            count = int(np.sum(self._mesu_locked & mask))
            self._mesu_locked[mask] = False
            self._mesu_stable_count[mask] = 0

        self._locked_count = int(np.sum(self._mesu_locked))
        logger.info("MESU: Unlocked %d weights (remaining locked: %d)", count, self._locked_count)
        return count

    def get_uncertainty_map(self) -> np.ndarray:
        """Return the per-synapse uncertainty matrix."""
        return np.sqrt(self._mesu_var).astype(np.float32)

    def get_locked_mask(self) -> np.ndarray:
        """Return the identity-locked synapse mask."""
        return self._mesu_locked.copy()


_instance: STDPLearningEngine | None = None


def get_stdp_engine(n_neurons: int = 64) -> STDPLearningEngine:
    global _instance
    if _instance is None:
        _instance = STDPLearningEngine(n_neurons=n_neurons)
    return _instance
