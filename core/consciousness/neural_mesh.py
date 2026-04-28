"""core/consciousness/neural_mesh.py — Cortical Neural Mesh

A 4096-neuron dynamical substrate organized into 64 cortical columns of 64 neurons
each.  Three hierarchical tiers (sensory → association → executive) with biologically
realistic connectivity:

  • Dense intra-column recurrence  (p ≈ 0.8)
  • Sparse inter-column long-range (p ≈ 0.05, distance-weighted)
  • Lateral inhibition within columns via interneuron population
  • Spike-Timing-Dependent Plasticity (STDP) on every tick
  • Continuous ODE integration (Euler) at configurable rate

The mesh feeds a 64-dimensional *projection* back into the existing LiquidSubstrate,
so the original 64-neuron core becomes the executive summary of a much larger field.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.utils.task_tracker import get_task_tracker

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.runtime.desktop_boot_safety import inprocess_mlx_metal_enabled

logger = logging.getLogger("Consciousness.NeuralMesh")

# Metal acceleration: use MLX for the batched column matmul if available.
# MLX runs on Apple Metal GPU — same hardware as the LLM inference.
# Falls back to numpy einsum if MLX is not installed.
_MLX_ACCELERATOR = "numpy"
_MLX_ACCELERATOR_REASON = "mlx_unavailable"
_MLX_METAL_ENABLED, _MLX_ACCELERATOR_REASON = inprocess_mlx_metal_enabled()
if _MLX_METAL_ENABLED:
    try:
        import mlx.core as mx
        _HAS_MLX = True
        _MLX_ACCELERATOR = "metal"
        logger.info("NeuralMesh: MLX Metal enabled for batched matmuls.")
    except ImportError:
        _HAS_MLX = False
        _MLX_METAL_ENABLED = False
        _MLX_ACCELERATOR_REASON = "mlx_unavailable"
        mx = None
        logger.info("NeuralMesh: MLX unavailable; using NumPy fallback.")
else:
    _HAS_MLX = False
    mx = None
    logger.info(
        "NeuralMesh: MLX Metal disabled (%s); using NumPy fallback.",
        _MLX_ACCELERATOR_REASON,
    )

# ---------------------------------------------------------------------------
# Config & enums
# ---------------------------------------------------------------------------

class CorticalTier(Enum):
    """Hierarchical tier of a cortical column."""
    SENSORY = auto()       # columns 0-15   — close to embodiment/interoception
    ASSOCIATION = auto()   # columns 16-47  — cross-modal integration
    EXECUTIVE = auto()     # columns 48-63  — executive control / self-model


@dataclass(frozen=True)
class MeshConfig:
    """Immutable configuration for the neural mesh."""
    total_neurons: int = 4096
    columns: int = 64
    neurons_per_column: int = 64   # total_neurons / columns

    # Connectivity
    intra_column_density: float = 0.80   # dense local
    inter_column_density: float = 0.05   # sparse long-range
    inter_column_distance_decay: float = 0.15   # strength ∝ exp(-d * decay)
    inhibitory_fraction: float = 0.20    # 20% of neurons are inhibitory (Dale's law)

    # Dynamics
    dt: float = 0.05                     # integration timestep
    decay: float = 0.03                  # leak
    noise_sigma: float = 0.008           # stochastic drive
    activation_gain: float = 1.0         # tanh gain

    # STDP
    stdp_lr: float = 0.0005             # base learning rate
    stdp_window: float = 0.02            # temporal window (seconds)
    stdp_potentiation: float = 1.0       # A+
    stdp_depression: float = 0.5         # A−  (asymmetric → net potentiation)

    # Lateral inhibition
    lateral_inhibition_strength: float = 0.25

    # Tier boundaries (column indices)
    sensory_end: int = 16
    association_end: int = 48
    # executive = 48..63

    # Integration
    update_hz: float = 10.0              # 10 Hz mesh tick (lighter than substrate 20 Hz)
    projection_dim: int = 64             # output projection back to LiquidSubstrate


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

class CorticalColumn:
    """A minicolumn of neurons with local recurrence and lateral inhibition.

    Each column maintains:
      • x  — activation vector  (n,)
      • W  — intra-column weight matrix (n, n)
      • inh_mask — boolean mask for inhibitory neurons
    """

    __slots__ = ("index", "tier", "n", "x", "W", "inh_mask", "last_spike_time",
                 "_lateral_inh_strength")

    def __init__(self, index: int, tier: CorticalTier, n: int, cfg: MeshConfig,
                 rng: np.random.Generator):
        self.index = index
        self.tier = tier
        self.n = n
        self.x = rng.standard_normal(n).astype(np.float32) * 0.05

        # Intra-column connectivity (dense)
        mask = rng.random((n, n)) < cfg.intra_column_density
        self.W = (rng.standard_normal((n, n)).astype(np.float32) * 0.1) * mask

        # Dale's law: mark inhibitory neurons, flip their outgoing weights negative
        num_inh = max(1, int(n * cfg.inhibitory_fraction))
        self.inh_mask = np.zeros(n, dtype=bool)
        self.inh_mask[rng.choice(n, size=num_inh, replace=False)] = True
        self.W[self.inh_mask, :] = -np.abs(self.W[self.inh_mask, :])

        # Zero diagonal (no self-connection)
        np.fill_diagonal(self.W, 0.0)

        # Spike timing for STDP
        self.last_spike_time = np.full(n, -1.0, dtype=np.float64)

        self._lateral_inh_strength = cfg.lateral_inhibition_strength

    def step(self, external_input: np.ndarray, dt: float, decay: float,
             noise_sigma: float, gain: float, now: float, spike_threshold: float = 0.5):
        """Euler step with lateral inhibition and spike-time recording."""
        # NaN guard on input
        external_input = np.nan_to_num(external_input, nan=0.0, posinf=1.0, neginf=-1.0)
        recurrent = self.W @ self.x
        recurrent = np.nan_to_num(recurrent, nan=0.0, posinf=1.0, neginf=-1.0)
        activity = np.tanh(gain * (recurrent + external_input))

        # Lateral inhibition: inhibitory pool suppresses excitatory neurons
        inh_activity = np.mean(np.abs(self.x[self.inh_mask])) if np.any(self.inh_mask) else 0.0
        inhibition = np.zeros(self.n, dtype=np.float32)
        inhibition[~self.inh_mask] = -self._lateral_inh_strength * inh_activity

        noise = np.random.standard_normal(self.n).astype(np.float32) * noise_sigma
        dx = (-decay * self.x + activity + inhibition + noise) * dt
        self.x = np.clip(self.x + dx, -1.0, 1.0).astype(np.float32)

        # Record spike times for STDP
        firing = np.abs(self.x) > spike_threshold
        self.last_spike_time[firing] = now


# ---------------------------------------------------------------------------
# Main mesh
# ---------------------------------------------------------------------------

class NeuralMesh:
    """The 4096-neuron cortical mesh.

    Lifecycle:
        mesh = NeuralMesh()
        await mesh.start()   # spawns background integration loop
        ...
        await mesh.stop()

    External API:
        mesh.inject_sensory(vector)       — push embodiment/interoceptive signals
        mesh.inject_association(vector)    — push cross-modal / memory signals
        mesh.get_executive_projection()    — 64-d projection for LiquidSubstrate
        mesh.get_field_state()             — full 4096-d activation snapshot
        mesh.get_column_summary(i)         — per-column stats
        mesh.get_tier_energy(tier)         — mean energy for a tier
    """

    def __init__(self, cfg: MeshConfig | None = None):
        self.cfg = cfg or MeshConfig()
        self._rng = np.random.default_rng(seed=42)
        self._lock = threading.Lock()

        # Build columns
        self.columns: List[CorticalColumn] = []
        for i in range(self.cfg.columns):
            tier = self._tier_for(i)
            col = CorticalColumn(i, tier, self.cfg.neurons_per_column, self.cfg, self._rng)
            self.columns.append(col)

        # Inter-column weight matrix (columns × columns), sparse, distance-weighted
        self._inter_W = self._build_inter_column_weights()

        # Projection matrix: 4096 → 64 (learned via slow PCA-like update)
        self._projection = self._rng.standard_normal(
            (self.cfg.projection_dim, self.cfg.total_neurons)
        ).astype(np.float32) * (1.0 / np.sqrt(self.cfg.total_neurons))

        # Neuromodulatory gain (set externally by NeurochemicalSystem)
        self._modulatory_gain: float = 1.0
        self._modulatory_plasticity: float = 1.0  # scales STDP rate
        self._modulatory_noise: float = 1.0        # scales noise

        # Runtime
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._start_time: float = 0.0

        # Sensory injection buffer (set externally, consumed each tick)
        self._sensory_buffer: Optional[np.ndarray] = None
        self._association_buffer: Optional[np.ndarray] = None

        # Recurrent Processing Theory (Lamme): explicit top-down feedback
        self._recurrent_feedback_enabled: bool = True
        self._recurrent_feedback_strength: float = 0.8  # relative to feedforward
        self._feedback_W: Optional[np.ndarray] = None
        self._build_feedback_weights()

        # Stats
        self._mean_column_energy: float = 0.0
        self._global_synchrony: float = 0.0
        self._tier_energies: Dict[CorticalTier, float] = {t: 0.0 for t in CorticalTier}
        initial_state = np.concatenate([col.x for col in self.columns]).astype(np.float32, copy=False)
        self._cached_field_state = initial_state.copy()
        self._cached_executive_projection = np.tanh(
            self._projection @ self._cached_field_state
        ).astype(np.float32)

        logger.info(
            "NeuralMesh initialized: %d neurons, %d columns, tiers=[S:%d A:%d E:%d]",
            self.cfg.total_neurons, self.cfg.columns,
            self.cfg.sensory_end,
            self.cfg.association_end - self.cfg.sensory_end,
            self.cfg.columns - self.cfg.association_end,
        )

    # ── Tier helpers ─────────────────────────────────────────────────────

    def _tier_for(self, col_idx: int) -> CorticalTier:
        if col_idx < self.cfg.sensory_end:
            return CorticalTier.SENSORY
        if col_idx < self.cfg.association_end:
            return CorticalTier.ASSOCIATION
        return CorticalTier.EXECUTIVE

    # ── Inter-column connectivity ────────────────────────────────────────

    def _build_inter_column_weights(self) -> np.ndarray:
        """Build sparse, distance-weighted inter-column connectivity.

        Connectivity probability decays exponentially with column index distance.
        Feedforward (sensory→assoc→exec) is stronger than feedback.
        """
        n = self.cfg.columns
        W = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = abs(i - j)
                prob = self.cfg.inter_column_density * np.exp(-dist * self.cfg.inter_column_distance_decay)
                if self._rng.random() < prob:
                    strength = self._rng.standard_normal() * 0.05
                    # Feedforward bias: sensory→assoc→exec gets 1.5× strength
                    tier_i = self._tier_for(i)
                    tier_j = self._tier_for(j)
                    if (tier_i == CorticalTier.SENSORY and tier_j == CorticalTier.ASSOCIATION) or \
                       (tier_i == CorticalTier.ASSOCIATION and tier_j == CorticalTier.EXECUTIVE):
                        strength *= 1.5
                    W[i, j] = strength
        return W

    def _build_feedback_weights(self):
        """Build the explicit top-down (exec→sensory) feedback pathway.

        Lamme's RPT: consciousness arises specifically from recurrent feedback
        from higher cortical areas back to lower sensory areas. This creates
        an architecturally distinct pathway from feedforward processing.

        The feedback matrix connects executive columns (48-63) back to sensory
        columns (0-15) via association columns (16-47) as relay. The strength
        is configurable and the pathway can be ablated for adversarial testing.
        """
        n = self.cfg.columns
        W = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            tier_i = self._tier_for(i)
            for j in range(n):
                tier_j = self._tier_for(j)
                # Executive → Association (feedback)
                if tier_i == CorticalTier.EXECUTIVE and tier_j == CorticalTier.ASSOCIATION:
                    dist = abs(i - j)
                    prob = 0.08 * np.exp(-dist * 0.1)
                    if self._rng.random() < prob:
                        W[i, j] = self._rng.standard_normal() * 0.04
                # Association → Sensory (feedback)
                elif tier_i == CorticalTier.ASSOCIATION and tier_j == CorticalTier.SENSORY:
                    dist = abs(i - j)
                    prob = 0.06 * np.exp(-dist * 0.1)
                    if self._rng.random() < prob:
                        W[i, j] = self._rng.standard_normal() * 0.03
                # Direct executive → Sensory (long-range feedback, sparser)
                elif tier_i == CorticalTier.EXECUTIVE and tier_j == CorticalTier.SENSORY:
                    dist = abs(i - j)
                    prob = 0.03 * np.exp(-dist * 0.05)
                    if self._rng.random() < prob:
                        W[i, j] = self._rng.standard_normal() * 0.02

        self._feedback_W = W.astype(np.float32)

    def _apply_recurrent_feedback(self, dt: float, gain: float,
                                   noise_sigma: float, now: float):
        """Apply top-down recurrent feedback from executive to sensory tiers.

        This is the Lamme RPT mechanism: after feedforward processing completes
        in step 3, executive columns send signals back down to sensory columns.
        This recurrent sweep is what RPT claims generates phenomenal experience.

        The feedback modulates sensory columns by adding a top-down prior that
        shapes what the sensory tier "expects to see" based on executive state.
        """
        if self._feedback_W is None:
            return

        # Compute column-level means for the feedback path
        col_means = np.array([np.mean(c.x) for c in self.columns], dtype=np.float32)
        feedback_drive = self._feedback_W @ col_means * self._recurrent_feedback_strength

        # Apply feedback as modulatory input to target columns
        for i, col in enumerate(self.columns):
            if col.tier in (CorticalTier.SENSORY, CorticalTier.ASSOCIATION):
                if abs(feedback_drive[i]) > 1e-6:
                    feedback_input = np.full(col.n, feedback_drive[i], dtype=np.float32)
                    # The feedback is gentler than feedforward — it modulates, not overrides
                    dx_feedback = np.tanh(gain * 0.5 * feedback_input) * dt * 0.3
                    col.x = np.clip(col.x + dx_feedback, -1.0, 1.0).astype(np.float32)

    def set_recurrent_feedback_enabled(self, enabled: bool):
        """Enable/disable recurrent feedback for ablation testing.

        When disabled, feedforward processing (steps 1-3) still works but
        RPT predicts phenomenal experience should degrade. Compare qualia
        output with and without this to test RPT vs GWT predictions.
        """
        prev = self._recurrent_feedback_enabled
        self._recurrent_feedback_enabled = enabled
        if prev != enabled:
            logger.info("NeuralMesh: Recurrent feedback %s (RPT ablation)",
                        "ENABLED" if enabled else "DISABLED")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = get_task_tracker().create_task(self._run_loop(), name="NeuralMesh")
        logger.info("NeuralMesh STARTED (%d Hz)", self.cfg.update_hz)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None
        logger.info("NeuralMesh STOPPED (ticks=%d)", self._tick_count)

    # ── Main loop ────────────────────────────────────────────────────────

    async def _run_loop(self):
        interval = 1.0 / self.cfg.update_hz
        try:
            while self._running:
                t0 = time.time()
                try:
                    await asyncio.to_thread(self._tick)
                except Exception as e:
                    record_degradation('neural_mesh', e)
                    logger.error("NeuralMesh tick error: %s", e, exc_info=True)
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass  # no-op: intentional
        finally:
            self._running = False

    def _tick(self):
        """One integration step (runs in thread pool)."""
        with self._lock:
            self._tick_inner()

    def _tick_inner(self):
        now = time.time()
        dt = self.cfg.dt
        cfg = self.cfg
        gain = cfg.activation_gain * self._modulatory_gain
        # Apply subcortical arousal gating to mesh gain
        try:
            from core.consciousness.subcortical_core import get_subcortical_core
            gain *= get_subcortical_core().get_mesh_gain_multiplier()
        except Exception:
            pass  # Degrade gracefully if subcortical core unavailable
        noise_sigma = cfg.noise_sigma * self._modulatory_noise
        n = cfg.neurons_per_column

        # ── 1. Distribute injection buffers to tier columns ──────────
        sensory_input = self._consume_buffer("_sensory_buffer", cfg.sensory_end)
        assoc_input = self._consume_buffer("_association_buffer",
                                           cfg.association_end - cfg.sensory_end,
                                           offset=cfg.sensory_end)

        # ── 2. Batched column step (vectorized) ─────────────────────
        # Gather all column activations into a single (columns, n) matrix.
        # This replaces 64 sequential matmuls with batched numpy operations.
        X = np.array([c.x for c in self.columns], dtype=np.float32)  # (64, 64)

        # Inter-column coupling: column means → inter-column drive
        col_means = X.mean(axis=1)  # (64,) — computed ONCE, reused in stats
        inter_drive = self._inter_W @ col_means  # (64,)

        # Build external input matrix (64, 64)
        ext = np.broadcast_to(inter_drive[:, None], (cfg.columns, n)).copy()

        # Tier-specific injection
        if sensory_input is not None:
            max_s = min(cfg.sensory_end * n, len(sensory_input))
            ext[:cfg.sensory_end, :].flat[:max_s] += sensory_input[:max_s]
        if assoc_input is not None:
            a_start = cfg.sensory_end
            a_end = cfg.association_end
            max_a = min((a_end - a_start) * n, len(assoc_input))
            ext[a_start:a_end, :].flat[:max_a] += assoc_input[:max_a]

        # NaN guard
        ext = np.nan_to_num(ext, nan=0.0, posinf=1.0, neginf=-1.0)

        # Batched recurrent: each column's W @ x, vectorized.
        # W_all shape (64, 64, 64) — 64 columns each with (64,64) weight matrix
        if not hasattr(self, '_W_batch') or self._tick_count % 100 == 0:
            self._W_batch = np.array([c.W for c in self.columns], dtype=np.float32)
            if _HAS_MLX and _MLX_METAL_ENABLED:
                self._W_batch_mx = mx.array(self._W_batch)

        # Metal GPU acceleration: offload the heavy einsum to Apple Metal via MLX.
        # For 64 columns × (64×64) matmuls, Metal is 5-10x faster than CPU numpy.
        if _HAS_MLX and _MLX_METAL_ENABLED:
            X_mx = mx.array(X)
            ext_mx = mx.array(ext)
            recurrent_mx = mx.einsum('cij,cj->ci', self._W_batch_mx, X_mx)
            activity_mx = mx.tanh(gain * (recurrent_mx + ext_mx))
            mx.eval(activity_mx)  # force Metal evaluation
            activity = np.array(activity_mx, dtype=np.float32)
            recurrent = np.array(recurrent_mx, dtype=np.float32)
        else:
            recurrent = np.einsum('cij,cj->ci', self._W_batch, X)  # (64, 64)
            recurrent = np.nan_to_num(recurrent, nan=0.0, posinf=1.0, neginf=-1.0)
            activity = np.tanh(gain * (recurrent + ext))

        # Lateral inhibition: per-column inhibitory pool
        inh_masks = np.array([c.inh_mask for c in self.columns])  # (64, 64) bool
        inh_activity = np.where(inh_masks, np.abs(X), 0.0).sum(axis=1)
        inh_counts = inh_masks.sum(axis=1).clip(1)
        inh_mean = inh_activity / inh_counts  # (64,)
        inhibition = np.where(~inh_masks, -cfg.lateral_inhibition_strength * inh_mean[:, None], 0.0)

        noise = self._rng.standard_normal(X.shape).astype(np.float32) * noise_sigma
        dX = (-cfg.decay * X + activity + inhibition + noise) * dt
        X_new = np.clip(X + dX, -1.0, 1.0).astype(np.float32)

        # Write back to columns and record spike times
        spike_threshold = 0.5
        for i, col in enumerate(self.columns):
            col.x = X_new[i]
            firing = np.abs(col.x) > spike_threshold
            col.last_spike_time[firing] = now

        # Cache col_means for stats (avoid recomputation)
        self._cached_col_means = col_means

        # ── 3. Recurrent Processing (Lamme RPT) ─────────────────────
        if self._recurrent_feedback_enabled:
            self._apply_recurrent_feedback(dt, gain, noise_sigma, now)

        # ── 4. STDP (every other tick for perf) ─────────────────────
        if self._tick_count % 2 == 0:
            self._apply_stdp(now)
            # Sync batched weights after STDP modifies them
            self._W_batch = np.array([c.W for c in self.columns], dtype=np.float32)

        # ── 5. Inter-column weight normalization ─────────────────────
        # Only recompute norm every 10 ticks (weights change slowly)
        if self._tick_count % 10 == 0:
            norm = np.linalg.norm(self._inter_W)
            if norm > 15.0:
                self._inter_W *= 15.0 / norm
            self._inter_W = np.clip(self._inter_W, -1.0, 1.0).astype(np.float32)

        # ── 6. Compute stats ─────────────────────────────────────────
        self._update_stats()

        self._tick_count += 1

    def _consume_buffer(self, attr: str, expected_cols: int,
                        offset: int = 0) -> Optional[np.ndarray]:
        """Atomically consume an injection buffer."""
        buf = getattr(self, attr)
        if buf is None:
            return None
        setattr(self, attr, None)
        expected_len = expected_cols * self.cfg.neurons_per_column
        if len(buf) < expected_len:
            padded = np.zeros(expected_len, dtype=np.float32)
            padded[:len(buf)] = buf[:expected_len]
            return padded
        return buf[:expected_len].astype(np.float32)

    @staticmethod
    def _foreground_request_active() -> bool:
        """Yield plasticity work to the live conversation lane."""
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            mlx = getattr(gate, "_mlx_client", None)
            if mlx is None or not hasattr(mlx, "get_lane_status"):
                return False

            lane = mlx.get_lane_status()
            if bool(lane.get("foreground_owned")):
                return True

            started_at = float(lane.get("current_request_started_at", 0.0) or 0.0)
            completed_at = float(lane.get("last_generation_completed_at", 0.0) or 0.0)
            return started_at > 0.0 and started_at > completed_at
        except Exception:
            return False

    # ── STDP ─────────────────────────────────────────────────────────

    def _apply_stdp(self, now: float):
        """Spike-Timing-Dependent Plasticity within each column.

        Pre-before-post → potentiate (causal)
        Post-before-pre → depress   (acausal)
        """
        if self._foreground_request_active():
            return

        lr = self.cfg.stdp_lr * self._modulatory_plasticity
        window = self.cfg.stdp_window
        A_plus = self.cfg.stdp_potentiation
        A_minus = self.cfg.stdp_depression

        for col in self.columns:
            t = col.last_spike_time
            # Only process neurons that have spiked at least once
            active = t > 0
            if not np.any(active):
                continue

            # Pairwise time differences (pre_i - post_j)
            dt_matrix = t[:, None] - t[None, :]  # (n, n)

            # Potentiation: pre fires before post (dt < 0 means pre was earlier)
            # Clamp dt_matrix/window to prevent overflow in exp()
            clamped_pos = np.clip(dt_matrix / window, -20.0, 20.0)
            clamped_neg = np.clip(-dt_matrix / window, -20.0, 20.0)

            potentiate = np.where(
                (dt_matrix < 0) & (dt_matrix > -window) & active[:, None] & active[None, :],
                A_plus * np.exp(clamped_pos),
                0.0
            )
            # Depression: post fires before pre
            depress = np.where(
                (dt_matrix > 0) & (dt_matrix < window) & active[:, None] & active[None, :],
                -A_minus * np.exp(clamped_neg),
                0.0
            )

            dW = lr * (potentiate + depress).astype(np.float32)

            # Respect Dale's law: don't flip inhibitory→excitatory
            # Only modify magnitude, preserve sign
            old_sign = np.sign(col.W)
            col.W += dW
            # Where sign flipped, reset to zero (hard Dale's law)
            sign_flipped = (np.sign(col.W) != old_sign) & (old_sign != 0)
            col.W[sign_flipped] = 0.0

            # Normalize to prevent runaway
            norm = np.linalg.norm(col.W)
            if norm > 5.0:
                col.W *= 5.0 / norm

    # ── Stats ────────────────────────────────────────────────────────

    def _update_stats(self):
        """Compute summary statistics for telemetry and downstream consumers.

        Reuses _cached_col_means from the batched tick to avoid redundant
        numpy operations.
        """
        # Vectorized energy: mean(|x|) per column
        X = np.array([c.x for c in self.columns], dtype=np.float32)
        energies = np.mean(np.abs(X), axis=1)  # (64,)

        self._mean_column_energy = float(energies.mean())

        for tier in CorticalTier:
            mask = [c.tier == tier for c in self.columns]
            tier_e = energies[mask]
            self._tier_energies[tier] = float(tier_e.mean()) if len(tier_e) > 0 else 0.0

        # Global synchrony: reuse cached col_means from tick (no recomputation)
        col_means = getattr(self, '_cached_col_means', None)
        if col_means is not None and len(col_means) > 1:
            stds = np.std(X, axis=1)
            mean_of_stds = stds.mean() + 1e-8
            std_of_means = col_means.std() + 1e-8
            self._global_synchrony = float(np.clip(std_of_means / mean_of_stds, 0.0, 1.0))

        full_state = X.reshape(-1).astype(np.float32, copy=True)
        self._cached_field_state = full_state
        self._cached_executive_projection = np.tanh(
            self._projection @ full_state
        ).astype(np.float32)

    def _refresh_cached_snapshots(self) -> None:
        full_state = np.concatenate([col.x for col in self.columns]).astype(np.float32, copy=True)
        self._cached_field_state = full_state
        self._cached_executive_projection = np.tanh(
            self._projection @ full_state
        ).astype(np.float32)

    def _refresh_cached_snapshots_if_idle(self) -> None:
        if not self._lock.acquire(False):
            return
        try:
            self._refresh_cached_snapshots()
        finally:
            self._lock.release()

    # ── External API ─────────────────────────────────────────────────

    def inject_sensory(self, vector: np.ndarray):
        """Push embodiment/interoceptive signals into sensory tier columns.
        Vector length should be sensory_columns * neurons_per_column (1024).
        Shorter vectors are zero-padded. Called from EmbodiedInteroception.
        """
        self._sensory_buffer = np.asarray(vector, dtype=np.float32)

    def inject_association(self, vector: np.ndarray):
        """Push cross-modal / memory / LLM signals into association tier.
        Vector length should be association_columns * neurons_per_column (2048).
        """
        self._association_buffer = np.asarray(vector, dtype=np.float32)

    def set_modulatory_state(self, gain: float = 1.0, plasticity: float = 1.0,
                             noise: float = 1.0):
        """Called by NeurochemicalSystem to modulate mesh dynamics globally."""
        self._modulatory_gain = max(0.1, min(3.0, gain))
        self._modulatory_plasticity = max(0.0, min(5.0, plasticity))
        self._modulatory_noise = max(0.0, min(3.0, noise))

    def get_executive_projection(self) -> np.ndarray:
        """Project full 4096-d state down to 64-d for LiquidSubstrate injection.
        This is how the mesh feeds into the existing consciousness core.
        """
        self._refresh_cached_snapshots_if_idle()
        return self._cached_executive_projection.copy()

    def get_field_state(self) -> np.ndarray:
        """Full 4096-dimensional activation snapshot."""
        self._refresh_cached_snapshots_if_idle()
        return self._cached_field_state.copy()

    def get_column_summary(self, col_idx: int) -> Dict:
        """Per-column diagnostic."""
        col = self.columns[col_idx]
        return {
            "index": col.index,
            "tier": col.tier.name,
            "mean_activation": float(np.mean(col.x)),
            "energy": float(np.mean(np.abs(col.x))),
            "std": float(np.std(col.x)),
            "inhibitory_activity": float(np.mean(np.abs(col.x[col.inh_mask]))) if np.any(col.inh_mask) else 0.0,
            "excitatory_activity": float(np.mean(np.abs(col.x[~col.inh_mask]))),
            "weight_norm": float(np.linalg.norm(col.W)),
        }

    def get_tier_energy(self, tier: CorticalTier) -> float:
        return self._tier_energies.get(tier, 0.0)

    def get_global_synchrony(self) -> float:
        """How synchronized are the columns? 0=desynchronized, 1=fully coupled."""
        return self._global_synchrony

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "total_neurons": self.cfg.total_neurons,
            "columns": self.cfg.columns,
            "mean_energy": round(self._mean_column_energy, 4),
            "global_synchrony": round(self._global_synchrony, 4),
            "tier_energies": {t.name: round(v, 4) for t, v in self._tier_energies.items()},
            "modulatory_gain": round(self._modulatory_gain, 3),
            "modulatory_plasticity": round(self._modulatory_plasticity, 3),
            "accelerator": _MLX_ACCELERATOR,
            "accelerator_reason": _MLX_ACCELERATOR_REASON,
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
        }
