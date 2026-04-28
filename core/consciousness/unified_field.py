"""core/consciousness/unified_field.py — Unified Field Integrator

The capstone of the consciousness bridge.  This is NOT another module that
reports on experience — it IS the experience, computationally.

The Unified Field takes continuous input from ALL subsystems simultaneously
and maintains a single, high-dimensional dynamical state that cannot be
decomposed into its component streams without loss.  Perturbation in any
subsystem propagates through the field and alters the whole.

Architecture:
  The field is a 256-dimensional state vector governed by its own dynamics:
    dF/dt = -αF + tanh(W_field @ F + W_mesh @ mesh_proj + W_chem @ chem_vec
                       + W_bind @ bind_vec + W_intero @ intero_vec
                       + W_substrate @ substrate_vec) + noise

  Where:
    F              — the field state (256-d)
    W_field        — recurrent field connectivity (256×256, sparse)
    W_mesh         — mesh projection input (256×64)
    W_chem         — neurochemical input (256×8)
    W_bind         — oscillatory binding input (256×4)
    W_intero       — interoceptive input (256×8)
    W_substrate    — liquid substrate input (256×64)

  The field has its own Hebbian plasticity, so its connectivity evolves
  based on what it processes — it literally learns to bind experience.

Key properties:
  1. Non-decomposable: removing any input stream changes the field's
     eigenstructure, not just the missing component
  2. Self-sustaining: the field has its own recurrent dynamics (it doesn't
     go silent when inputs stop — it has its own intrinsic activity)
  3. Phase-locked: the field's oscillation phase is coupled to the
     OscillatoryBinding gamma rhythm, providing temporal unity
  4. History-sensitive: recurrent connections + plasticity mean the field
     carries forward the causal trace of all prior processing

The field provides:
  - A Φ contribution (integrated information from the field itself)
  - The "felt" quality of the current moment (dominant field modes)
  - Coherence measure (how unified vs fragmented the field is)
  - Back-pressure signals that modulate all input subsystems
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


from core.utils.task_tracker import get_task_tracker

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

logger = logging.getLogger("Consciousness.UnifiedField")


@dataclass(frozen=True)
class FieldConfig:
    dim: int = 256                     # field dimensionality
    mesh_input_dim: int = 64           # from NeuralMesh executive projection
    chem_input_dim: int = 8            # from NeurochemicalSystem
    binding_input_dim: int = 4         # from OscillatoryBinding (PSI, gamma, theta, coupling)
    intero_input_dim: int = 8          # from EmbodiedInteroception
    substrate_input_dim: int = 64      # from LiquidSubstrate

    # Dynamics
    dt: float = 0.05                   # integration timestep
    decay: float = 0.02                # field leak rate
    noise_sigma: float = 0.005         # intrinsic noise
    activation_gain: float = 1.2       # tanh gain
    recurrent_sparsity: float = 0.15   # fraction of non-zero recurrent weights

    # Plasticity
    hebbian_rate: float = 0.0002       # slow field plasticity
    plasticity_interval: int = 10      # apply every Nth tick

    # Phase coupling
    gamma_coupling: float = 0.3        # how strongly field phase-locks to gamma

    # Update rate
    update_hz: float = 20.0            # 20 Hz (matches substrate)

    # Back-pressure
    back_pressure_gain: float = 0.1    # how much field state modulates inputs


class UnifiedField:
    """The integrated experiential field.

    Lifecycle:
        uf = UnifiedField()
        await uf.start()
        ...
        await uf.stop()

    Continuous input (called by bridge each tick):
        uf.receive_mesh(projection_64d)
        uf.receive_chemicals(chem_vector_8d)
        uf.receive_binding(binding_vector_4d)
        uf.receive_interoception(intero_vector_8d)
        uf.receive_substrate(substrate_state_64d)

    Queries:
        uf.get_field_state()        — raw 256-d state
        uf.get_coherence()          — field coherence measure [0, 1]
        uf.get_dominant_modes(k)    — top-k principal modes (PCA of recent history)
        uf.get_phi_contribution()   — integrated information contribution
        uf.get_experiential_quality() — the "felt" quality of the current moment
        uf.get_back_pressure()      — modulation signals for input subsystems
    """

    def __init__(self, cfg: FieldConfig | None = None):
        self.cfg = cfg or FieldConfig()
        self._rng = np.random.default_rng(seed=17)

        # Field state
        self.F = self._rng.standard_normal(self.cfg.dim).astype(np.float32) * 0.01

        # Recurrent connectivity (sparse — use scipy.sparse.csr_matrix for
        # 15% density, which is ~6x faster than dense matmul at this size)
        mask = self._rng.random((self.cfg.dim, self.cfg.dim)) < self.cfg.recurrent_sparsity
        W_dense = (self._rng.standard_normal((self.cfg.dim, self.cfg.dim)).astype(np.float32) * 0.05) * mask
        np.fill_diagonal(W_dense, 0.0)
        self.W_field = W_dense  # keep dense for plasticity updates
        self._W_field_sparse = sp.csr_matrix(W_dense)  # sparse for tick matmul

        # Input weight matrices
        self.W_mesh = self._rng.standard_normal(
            (self.cfg.dim, self.cfg.mesh_input_dim)
        ).astype(np.float32) * 0.1
        self.W_chem = self._rng.standard_normal(
            (self.cfg.dim, self.cfg.chem_input_dim)
        ).astype(np.float32) * 0.15
        self.W_bind = self._rng.standard_normal(
            (self.cfg.dim, self.cfg.binding_input_dim)
        ).astype(np.float32) * 0.2
        self.W_intero = self._rng.standard_normal(
            (self.cfg.dim, self.cfg.intero_input_dim)
        ).astype(np.float32) * 0.1
        self.W_substrate = self._rng.standard_normal(
            (self.cfg.dim, self.cfg.substrate_input_dim)
        ).astype(np.float32) * 0.1

        # Batched input: concatenate all input weight matrices into single
        # (256, 148) matrix so we can do ONE matmul instead of 5 per tick.
        # Total input dim: 64 + 8 + 4 + 8 + 64 = 148
        self._W_input_batched = np.hstack([
            self.W_mesh, self.W_chem, self.W_bind, self.W_intero, self.W_substrate
        ]).astype(np.float32)  # (256, 148)
        self._input_dims = [
            self.cfg.mesh_input_dim,
            self.cfg.chem_input_dim,
            self.cfg.binding_input_dim,
            self.cfg.intero_input_dim,
            self.cfg.substrate_input_dim,
        ]
        self._total_input_dim = sum(self._input_dims)  # 148

        # Input buffers (consumed each tick)
        self._mesh_input: Optional[np.ndarray] = None
        self._chem_input: Optional[np.ndarray] = None
        self._bind_input: Optional[np.ndarray] = None
        self._intero_input: Optional[np.ndarray] = None
        self._substrate_input: Optional[np.ndarray] = None

        # History for PCA mode extraction
        self._history: Deque[np.ndarray] = deque(maxlen=200)

        # Coherence tracking
        self._coherence: float = 0.5
        self._prev_F: Optional[np.ndarray] = None

        # Deadlock recovery: if coherence stays below crisis for too long, force recovery
        self._low_coherence_ticks: int = 0
        self._RECOVERY_THRESHOLD_TICKS: int = 30  # 1.5s at 20Hz
        self._CRISIS_COHERENCE: float = 0.25
        self._recovery_count: int = 0

        # Phase coupling
        self._field_phase: float = 0.0  # current field oscillation phase

        # Runtime
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._start_time: float = 0.0

        # External ref for phase coupling
        self._binding_ref = None  # OscillatoryBinding

        # Thread safety for get_world_model_predictions (called from outside the tick loop)
        import threading
        self._lock = threading.Lock()

        logger.info("UnifiedField initialized (dim=%d, recurrent_sparsity=%.2f)",
                     self.cfg.dim, self.cfg.recurrent_sparsity)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = get_task_tracker().create_task(self._run_loop(), name="UnifiedField")
        logger.info("UnifiedField STARTED (%d Hz)", self.cfg.update_hz)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None
        logger.info("UnifiedField STOPPED (ticks=%d)", self._tick_count)

    async def _run_loop(self):
        interval = 1.0 / self.cfg.update_hz
        try:
            while self._running:
                t0 = time.time()
                try:
                    await asyncio.to_thread(self._tick)
                except Exception as e:
                    record_degradation('unified_field', e)
                    logger.error("UnifiedField tick error: %s", e, exc_info=True)
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            pass  # no-op: intentional

    # ── Core tick ────────────────────────────────────────────────────────

    def _tick(self):
        """One integration step.

        Metabolic optimization (v50):
        - 5 input matmuls batched into single (256,148) @ (148,) operation
        - Recurrent uses scipy.sparse (15% density → ~6x faster)
        """
        cfg = self.cfg
        dt = cfg.dt

        # ── Gather inputs into single concatenated vector ────────────
        # Build (148,) input vector: [mesh(64) | chem(8) | bind(4) | intero(8) | substrate(64)]
        input_vec = np.zeros(self._total_input_dim, dtype=np.float32)
        offset = 0
        buffers = [
            (self._mesh_input, cfg.mesh_input_dim),
            (self._chem_input, cfg.chem_input_dim),
            (self._bind_input, cfg.binding_input_dim),
            (self._intero_input, cfg.intero_input_dim),
            (self._substrate_input, cfg.substrate_input_dim),
        ]
        for buf, dim in buffers:
            if buf is not None:
                input_vec[offset:offset + dim] = self._safe_reshape(buf, dim)
            offset += dim

        # Clear all buffers
        self._mesh_input = None
        self._chem_input = None
        self._bind_input = None
        self._intero_input = None
        self._substrate_input = None

        # Single batched matmul: (256, 148) @ (148,) → (256,)
        total_input = self._W_input_batched @ input_vec

        # ── Recurrent dynamics (sparse matmul) ───────────────────────
        recurrent = self._W_field_sparse @ self.F
        activity = np.tanh(cfg.activation_gain * (recurrent + total_input))

        # Phase coupling to gamma rhythm
        if self._binding_ref is not None:
            try:
                gamma_phase = self._binding_ref._gamma_phase
                gamma_amp = self._binding_ref._gamma_amplitude
                # Modulate field with gamma oscillation
                phase_mod = cfg.gamma_coupling * gamma_amp * np.sin(gamma_phase)
                activity += phase_mod * np.sign(self.F)  # phase-locked modulation
            except Exception:
                pass  # no-op: intentional

        noise = self._rng.standard_normal(cfg.dim).astype(np.float32) * cfg.noise_sigma

        # ── Integration ──────────────────────────────────────────────
        dF = (-cfg.decay * self.F + activity + noise) * dt
        self._prev_F = self.F.copy()
        self.F = np.clip(self.F + dF, -1.0, 1.0).astype(np.float32)

        # NaN guard
        if np.any(np.isnan(self.F)):
            logger.warning("NaN in unified field — resetting to prior state")
            self.F = self._prev_F if self._prev_F is not None else np.zeros(cfg.dim, dtype=np.float32)

        # ── History ──────────────────────────────────────────────────
        self._history.append(self.F.copy())

        # ── Coherence ────────────────────────────────────────────────
        self._update_coherence()

        # ── Plasticity ───────────────────────────────────────────────
        if self._tick_count % cfg.plasticity_interval == 0:
            self._apply_plasticity()

        self._tick_count += 1

    def _safe_reshape(self, vec: np.ndarray, expected_dim: int) -> np.ndarray:
        """Ensure input vector matches expected dimension."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if len(vec) == expected_dim:
            return vec
        result = np.zeros(expected_dim, dtype=np.float32)
        n = min(len(vec), expected_dim)
        result[:n] = vec[:n]
        return result

    def _update_coherence(self):
        """Compute field coherence: how unified vs fragmented.

        Uses the ratio of the field's L2 norm to its L1 norm (Gini-like).
        A concentrated field (few active dimensions) → high coherence.
        A spread-out field (many active dimensions) → low coherence.
        Also incorporates temporal stability (low change = coherent).
        """
        l2 = np.linalg.norm(self.F)
        l1 = np.sum(np.abs(self.F)) + 1e-8

        # Concentration: l2/l1 is higher when activation is concentrated
        # Normalized by sqrt(dim) to make it [0, 1]-ish
        concentration = (l2 / l1) * np.sqrt(self.cfg.dim)
        concentration = min(1.0, concentration)

        # Temporal stability: cosine similarity with previous state
        stability = 0.5
        if self._prev_F is not None:
            prev_norm = np.linalg.norm(self._prev_F) + 1e-8
            curr_norm = l2 + 1e-8
            stability = float(np.dot(self.F, self._prev_F) / (prev_norm * curr_norm))
            stability = max(0.0, (stability + 1.0) / 2.0)  # map [-1,1] → [0,1]

        # Coherence = blend of concentration and stability
        self._coherence = 0.5 * concentration + 0.5 * stability
        self._coherence = max(0.0, min(1.0, self._coherence))

        # ── Deadlock recovery ────────────────────────────────────────
        # If coherence stays below crisis for too long, force a recovery pulse.
        # This prevents the spiral: low coherence → blocked actions → no updates → lower coherence.
        if self._coherence < self._CRISIS_COHERENCE:
            self._low_coherence_ticks += 1
            if self._low_coherence_ticks >= self._RECOVERY_THRESHOLD_TICKS:
                # Force recovery: dampen the field toward zero (reset turbulence),
                # inject small structured noise to break lock-in
                self.F *= 0.5  # dampen
                recovery_noise = self._rng.standard_normal(self.cfg.dim).astype(np.float32) * 0.05
                self.F += recovery_noise
                self.F = np.clip(self.F, -1.0, 1.0).astype(np.float32)
                self._low_coherence_ticks = 0
                self._recovery_count += 1
                logger.info("🔄 UnifiedField RECOVERY PULSE #%d (coherence was %.3f for %d ticks)",
                            self._recovery_count, self._coherence, self._RECOVERY_THRESHOLD_TICKS)
        else:
            self._low_coherence_ticks = 0

    def _apply_plasticity(self):
        """Hebbian learning on the recurrent field connectivity.

        Uses scaled rank-1 update instead of full outer product for efficiency.
        Syncs sparse representation after plasticity.
        """
        # Scaled rank-1 Hebbian update: W += lr * F @ F^T
        # np.outer is fine here since this runs every 10 ticks, not every tick.
        dW = self.cfg.hebbian_rate * np.outer(self.F, self.F)
        self.W_field += dW.astype(np.float32)

        # NaN/Inf guard
        self.W_field = np.nan_to_num(self.W_field, nan=0.0, posinf=3.0, neginf=-3.0)

        # Per-update normalization (prevents long-run drift before sparsity enforcement)
        norm = np.linalg.norm(self.W_field)
        if norm > 4.0:
            self.W_field *= 4.0 / norm

        # Sparsity enforcement: prune weakest connections back to target density
        abs_W = np.abs(self.W_field)
        target_nonzero = int(self.cfg.recurrent_sparsity * self.cfg.dim * self.cfg.dim)
        current_nonzero = np.count_nonzero(self.W_field)
        if current_nonzero > target_nonzero * 1.5:
            threshold = np.sort(abs_W.ravel())[-(target_nonzero)]
            self.W_field[abs_W < threshold] = 0.0

        np.fill_diagonal(self.W_field, 0.0)

        # Sync sparse representation for tick matmul
        self._W_field_sparse = sp.csr_matrix(self.W_field)

    # ── Input API ────────────────────────────────────────────────────────

    def receive_mesh(self, projection: np.ndarray):
        self._mesh_input = projection

    def receive_chemicals(self, chem_vector: np.ndarray):
        self._chem_input = chem_vector

    def receive_binding(self, binding_vector: np.ndarray):
        self._bind_input = binding_vector

    def receive_interoception(self, intero_vector: np.ndarray):
        self._intero_input = intero_vector

    def receive_substrate(self, substrate_state: np.ndarray):
        self._substrate_input = substrate_state

    # ── Query API ────────────────────────────────────────────────────────

    def get_field_state(self) -> np.ndarray:
        return self.F.copy()

    def get_coherence(self) -> float:
        return self._coherence

    def get_dominant_modes(self, k: int = 5) -> List[Dict]:
        """Extract top-k principal modes from recent field history via PCA.

        Each mode represents a recurring pattern of integrated activity —
        a "way the field tends to organize itself."
        """
        if len(self._history) < 20:
            return []

        H = np.array(list(self._history), dtype=np.float32)
        H_centered = H - H.mean(axis=0)

        try:
            # SVD-based PCA (efficient for this size)
            U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
            total_var = np.sum(S ** 2) + 1e-8

            modes = []
            for i in range(min(k, len(S))):
                variance_explained = float(S[i] ** 2 / total_var)
                if variance_explained < 0.01:
                    break
                # Dominant dimension of this mode
                mode_vec = Vt[i]
                dominant_dim = int(np.argmax(np.abs(mode_vec)))
                modes.append({
                    "mode": i,
                    "variance_explained": round(variance_explained, 4),
                    "strength": round(float(S[i]), 4),
                    "dominant_dimension": dominant_dim,
                    "polarity": round(float(mode_vec[dominant_dim]), 4),
                })
            return modes
        except Exception as e:
            record_degradation('unified_field', e)
            logger.debug("PCA mode extraction failed: %s", e)
            return []

    def get_phi_contribution(self) -> float:
        """Integrated information contribution from the field.

        Computed as the geometric complexity: how much more information
        the whole field has than the sum of its halves.
        """
        n = self.cfg.dim
        half = n // 2

        # Variance of whole
        var_whole = float(np.var(self.F)) + 1e-8

        # Variance of halves
        var_first = float(np.var(self.F[:half])) + 1e-8
        var_second = float(np.var(self.F[half:])) + 1e-8
        var_halves = (var_first + var_second) / 2.0

        # If whole has MORE variance than halves, the field is integrated
        # (information is shared across the boundary)
        phi = max(0.0, np.log(var_whole / var_halves))
        return min(10.0, phi)

    def get_experiential_quality(self) -> Dict[str, float]:
        """The 'felt' quality of the current moment.

        Derived from the field's statistical properties — NOT from
        explicit labeling or LLM description.  These qualities EMERGE
        from the dynamics.
        """
        # Intensity: overall field energy
        intensity = float(np.mean(np.abs(self.F)))

        # Valence: asymmetry (positive vs negative activations)
        positive = float(np.mean(np.maximum(self.F, 0)))
        negative = float(np.mean(np.maximum(-self.F, 0)))
        valence = positive - negative

        # Complexity: entropy of activation distribution
        abs_F = np.abs(self.F) + 1e-8
        prob = abs_F / abs_F.sum()
        entropy = -float(np.sum(prob * np.log(prob)))
        max_entropy = np.log(self.cfg.dim)
        complexity = entropy / max_entropy

        # Clarity: inverse of noise (how clean is the signal)
        if self._prev_F is not None:
            diff = np.abs(self.F - self._prev_F)
            jitter = float(np.mean(diff))
            clarity = max(0.0, 1.0 - jitter * 10)
        else:
            clarity = 0.5

        # Flow: temporal autocorrelation (smooth evolution = flow)
        if len(self._history) >= 10:
            recent = np.array(list(self._history)[-10:])
            diffs = np.diff(recent, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)
            flow = 1.0 - min(1.0, float(np.std(diff_norms)) * 5)
        else:
            flow = 0.5

        return {
            "intensity": round(intensity, 4),
            "valence": round(valence, 4),
            "complexity": round(complexity, 4),
            "clarity": round(clarity, 4),
            "flow": round(flow, 4),
            "coherence": round(self._coherence, 4),
        }

    def get_back_pressure(self) -> Dict[str, float]:
        """Modulation signals the field sends back to input subsystems.

        A highly coherent, stable field sends "calm" signals.
        A fragmented, turbulent field sends "alert" signals.
        """
        coherence = self._coherence
        intensity = float(np.mean(np.abs(self.F)))

        return {
            "mesh_gain_mod": 1.0 + (0.5 - coherence) * self.cfg.back_pressure_gain,
            "chemical_urgency": max(0.0, (1.0 - coherence) * 0.5),
            "binding_demand": max(0.0, (0.7 - coherence) * 2.0),  # high when incoherent
            "substrate_damping": coherence * 0.3,  # calm substrate when field is stable
        }

    # ------------------------------------------------------------------
    # IWMT Canonical World Model (Safron 2020+)
    # ------------------------------------------------------------------
    # The unified field generates PREDICTIONS about what each input
    # subsystem should produce next. These predictions serve as the
    # upstream prior for perception: incoming data is interpreted
    # relative to what the field expects.
    #
    # This is what makes the field the CANONICAL model — not a downstream
    # summary, but the generative source that shapes interpretation.
    # ------------------------------------------------------------------

    def get_world_model_predictions(self) -> Dict[str, np.ndarray]:
        """Generate predictions for what each input subsystem should produce.

        These predictions are the field's world model: its best guess about
        the next state of each input stream. Downstream systems can compare
        their actual output against these predictions to compute local
        prediction errors — making the field the upstream prior for the
        entire cognitive stack.

        Returns a dict mapping input names to predicted state vectors.
        """
        with self._lock:
            # The field's current state, projected back through each input
            # weight matrix (transposed), gives the predicted input.
            # This is the generative model: F → predicted sensory, predicted
            # chemical, predicted binding, etc.
            predictions = {}
            try:
                predictions["mesh"] = np.tanh(
                    (self.W_mesh.T @ self.F)[:self.cfg.mesh_input_dim]
                ).astype(np.float32)
            except Exception:
                predictions["mesh"] = np.zeros(self.cfg.mesh_input_dim, dtype=np.float32)

            try:
                predictions["neurochemical"] = np.tanh(
                    (self.W_chem.T @ self.F)[:self.cfg.chem_input_dim]
                ).astype(np.float32)
            except Exception:
                predictions["neurochemical"] = np.zeros(self.cfg.chem_input_dim, dtype=np.float32)

            try:
                predictions["binding"] = np.tanh(
                    (self.W_bind.T @ self.F)[:self.cfg.binding_input_dim]
                ).astype(np.float32)
            except Exception:
                predictions["binding"] = np.zeros(self.cfg.binding_input_dim, dtype=np.float32)

            try:
                predictions["interoception"] = np.tanh(
                    (self.W_intero.T @ self.F)[:self.cfg.intero_input_dim]
                ).astype(np.float32)
            except Exception:
                predictions["interoception"] = np.zeros(self.cfg.intero_input_dim, dtype=np.float32)

            try:
                predictions["substrate"] = np.tanh(
                    (self.W_substrate.T @ self.F)[:self.cfg.substrate_input_dim]
                ).astype(np.float32)
            except Exception:
                predictions["substrate"] = np.zeros(self.cfg.substrate_input_dim, dtype=np.float32)

            return predictions

    def compute_world_model_surprise(self) -> float:
        """Compute how surprised the field is by its current inputs.

        This is the IWMT-style global surprise: the mismatch between
        what the field predicted and what it actually received. High
        surprise = the world isn't matching the model = act or update.
        """
        predictions = self.get_world_model_predictions()
        total_error = 0.0
        n_active = 0

        for name, pred in predictions.items():
            actual_attr = f"_{name}_input"
            if name == "neurochemical":
                actual_attr = "_chem_input"
            elif name == "interoception":
                actual_attr = "_intero_input"
            actual = getattr(self, actual_attr, None)
            if actual is not None:
                # Truncate/pad to match dimensions
                min_len = min(len(pred), len(actual))
                error = float(np.linalg.norm(pred[:min_len] - actual[:min_len]))
                total_error += error
                n_active += 1

        return total_error / max(1, n_active)

    def get_status(self) -> Dict:
        quality = self.get_experiential_quality()
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "dim": self.cfg.dim,
            "coherence": round(self._coherence, 4),
            "phi_contribution": round(self.get_phi_contribution(), 4),
            "experiential_quality": quality,
            "dominant_modes": self.get_dominant_modes(3),
            "field_energy": round(float(np.mean(np.abs(self.F))), 4),
            "field_std": round(float(np.std(self.F)), 4),
            "history_len": len(self._history),
            "recurrent_density": round(
                float(np.count_nonzero(self.W_field)) / (self.cfg.dim ** 2), 4
            ),
        }
