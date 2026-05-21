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

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

try:  # scipy is an acceleration path here; dense numpy remains correct.
    from scipy import sparse as sp
except ImportError:  # pragma: no cover - exercised on lean CI/runtime images
    sp = None

from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Consciousness.UnifiedField")

_RECOVERABLE_FIELD_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    FloatingPointError,
    np.linalg.LinAlgError,
)
_MAX_FIELD_DIM = 4096
_MAX_INPUT_DIM = 4096


def _record_unified_field_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, object] | None = None,
) -> None:
    try:
        record_degradation(
            "unified_field",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        record_degradation("unified_field", error)


def _finite_float(raw: object, default: float) -> tuple[float, bool]:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if not np.isfinite(value):
        return default, False
    return value, True


def _clamp_float(value: float, *, lower: float, upper: float) -> tuple[float, bool]:
    clamped = max(lower, min(upper, value))
    return clamped, clamped == value


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

    def __init__(
        self,
        cfg: FieldConfig | None = None,
        *,
        config: FieldConfig | None = None,
    ):
        if cfg is not None and config is not None and cfg != config:
            raise ValueError("UnifiedField received conflicting cfg and config values")
        self.cfg = cfg or config or FieldConfig()
        self._validate_config()
        self._rng = np.random.default_rng(seed=17)
        self._lock = threading.RLock()

        # Field state
        self.F = self._rng.standard_normal(self.cfg.dim).astype(np.float32) * 0.01

        # Recurrent connectivity (sparse — use scipy.sparse.csr_matrix for
        # 15% density, which is ~6x faster than dense matmul at this size)
        mask = self._rng.random((self.cfg.dim, self.cfg.dim)) < self.cfg.recurrent_sparsity
        field_weights = (self._rng.standard_normal((self.cfg.dim, self.cfg.dim)).astype(np.float32) * 0.05) * mask
        np.fill_diagonal(field_weights, 0.0)
        self.W_field = field_weights  # keep dense for plasticity updates
        self._W_field_sparse = self._to_sparse(field_weights)  # sparse for tick matmul

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
        self._mesh_input: np.ndarray | None = None
        self._chem_input: np.ndarray | None = None
        self._bind_input: np.ndarray | None = None
        self._intero_input: np.ndarray | None = None
        self._substrate_input: np.ndarray | None = None

        # History for PCA mode extraction
        self._history: deque[np.ndarray] = deque(maxlen=200)

        # Coherence tracking
        self._coherence: float = 0.5
        self._prev_F: np.ndarray | None = None

        # Deadlock recovery: if coherence stays below crisis for too long, force recovery
        self._low_coherence_ticks: int = 0
        self._RECOVERY_THRESHOLD_TICKS: int = 30  # 1.5s at 20Hz
        self._CRISIS_COHERENCE: float = 0.25
        self._recovery_count: int = 0

        # Phase coupling
        self._field_phase: float = 0.0  # current field oscillation phase

        # Runtime
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._start_time: float = 0.0
        self._consecutive_tick_failures: int = 0
        self._last_tick_error_at: float = 0.0

        # External ref for phase coupling
        self._binding_ref = None  # OscillatoryBinding

        logger.info("UnifiedField initialized (dim=%d, recurrent_sparsity=%.2f)",
                     self.cfg.dim, self.cfg.recurrent_sparsity)

    def _validate_config(self) -> None:
        cfg = self.cfg
        dims = {
            "dim": (cfg.dim, 2, _MAX_FIELD_DIM),
            "mesh_input_dim": (cfg.mesh_input_dim, 1, _MAX_INPUT_DIM),
            "chem_input_dim": (cfg.chem_input_dim, 1, _MAX_INPUT_DIM),
            "binding_input_dim": (cfg.binding_input_dim, 1, _MAX_INPUT_DIM),
            "intero_input_dim": (cfg.intero_input_dim, 1, _MAX_INPUT_DIM),
            "substrate_input_dim": (cfg.substrate_input_dim, 1, _MAX_INPUT_DIM),
            "plasticity_interval": (cfg.plasticity_interval, 1, 100_000),
        }
        for name, (value, lower, upper) in dims.items():
            if not isinstance(value, int) or value < lower or value > upper:
                raise ValueError(f"UnifiedField {name} must be an integer in [{lower}, {upper}]")

        floats = {
            "dt": (cfg.dt, 0.0001, 1.0),
            "decay": (cfg.decay, 0.0, 10.0),
            "noise_sigma": (cfg.noise_sigma, 0.0, 1.0),
            "activation_gain": (cfg.activation_gain, 0.001, 20.0),
            "recurrent_sparsity": (cfg.recurrent_sparsity, 0.001, 1.0),
            "hebbian_rate": (cfg.hebbian_rate, 0.0, 0.1),
            "gamma_coupling": (cfg.gamma_coupling, 0.0, 5.0),
            "update_hz": (cfg.update_hz, 0.1, 1000.0),
            "back_pressure_gain": (cfg.back_pressure_gain, 0.0, 10.0),
        }
        for name, (raw, lower, upper) in floats.items():
            value, valid = _finite_float(raw, lower)
            if not valid or value < lower or value > upper:
                raise ValueError(f"UnifiedField {name} must be finite in [{lower}, {upper}]")

    def _to_sparse(self, weights: np.ndarray) -> object:
        if sp is None:
            return weights
        return sp.csr_matrix(weights)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        run_loop = self._run_loop()
        try:
            self._task = get_task_tracker().create_task(run_loop, name="UnifiedField")
        except _RECOVERABLE_FIELD_ERRORS as exc:
            run_loop.close()
            self._running = False
            self._task = None
            _record_unified_field_degradation(
                exc,
                action="failed closed when UnifiedField task creation failed",
                severity="critical",
            )
            raise
        logger.info("UnifiedField STARTED (%.1f Hz)", self.cfg.update_hz)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("UnifiedField background task acknowledged cancellation.")
            self._task = None
        logger.info("UnifiedField STOPPED (ticks=%d)", self._tick_count)

    async def _run_loop(self):
        update_hz, valid_update_hz = _finite_float(self.cfg.update_hz, 20.0)
        update_hz, update_hz_unchanged = _clamp_float(
            update_hz,
            lower=0.1,
            upper=1000.0,
        )
        if not valid_update_hz or not update_hz_unchanged:
            _record_unified_field_degradation(
                ValueError(f"unsafe UnifiedField update_hz: {self.cfg.update_hz!r}"),
                action="normalized UnifiedField update rate before runtime loop",
                severity="warning",
                extra={"normalized_update_hz": update_hz},
            )
        interval = 1.0 / update_hz
        try:
            while self._running:
                t0 = time.monotonic()
                try:
                    await asyncio.to_thread(self._tick)
                    self._consecutive_tick_failures = 0
                except _RECOVERABLE_FIELD_ERRORS as exc:
                    self._consecutive_tick_failures += 1
                    self._last_tick_error_at = time.monotonic()
                    _record_unified_field_degradation(
                        exc,
                        action=(
                            "kept UnifiedField loop alive after tick failure "
                            "and normalized field state"
                        ),
                        extra={
                            "consecutive_tick_failures": self._consecutive_tick_failures
                        },
                    )
                    logger.error("UnifiedField tick error: %s", exc, exc_info=True)
                    self._enter_fail_safe_state()
                elapsed = time.monotonic() - t0
                backoff = min(
                    interval * max(0, self._consecutive_tick_failures),
                    2.0,
                )
                await asyncio.sleep(max(0.0, interval + backoff - elapsed))
        except asyncio.CancelledError:
            logger.debug("UnifiedField run loop cancelled.")
        finally:
            self._running = False

    def _enter_fail_safe_state(self) -> None:
        with self._lock:
            self.F = self._safe_reshape(
                self.F,
                self.cfg.dim,
                source="field_state",
                record=True,
                clip_abs=1.0,
            )
            if self._prev_F is not None:
                self._prev_F = self._safe_reshape(
                    self._prev_F,
                    self.cfg.dim,
                    source="previous_field_state",
                    record=True,
                    clip_abs=1.0,
                )
            self.F = np.clip(self.F * 0.8, -1.0, 1.0).astype(np.float32)
            self._mesh_input = None
            self._chem_input = None
            self._bind_input = None
            self._intero_input = None
            self._substrate_input = None
            self._normalize_weight_matrices()
            coherence, _ = _finite_float(self._coherence, 0.5)
            self._coherence, _ = _clamp_float(coherence, lower=0.0, upper=1.0)
            phase, _ = _finite_float(self._field_phase, 0.0)
            self._field_phase = float(phase % (2.0 * np.pi))

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
        with self._lock:
            buffers = [
                (self._mesh_input, cfg.mesh_input_dim, "mesh_input"),
                (self._chem_input, cfg.chem_input_dim, "chemical_input"),
                (self._bind_input, cfg.binding_input_dim, "binding_input"),
                (self._intero_input, cfg.intero_input_dim, "interoceptive_input"),
                (self._substrate_input, cfg.substrate_input_dim, "substrate_input"),
            ]
            self._mesh_input = None
            self._chem_input = None
            self._bind_input = None
            self._intero_input = None
            self._substrate_input = None
            self.F = self._safe_reshape(
                self.F,
                cfg.dim,
                source="field_state",
                record=True,
                clip_abs=1.0,
            )
        for buf, dim, source in buffers:
            if buf is not None:
                input_vec[offset:offset + dim] = self._safe_reshape(
                    buf,
                    dim,
                    source=source,
                    record=True,
                    clip_abs=1.0,
                )
            offset += dim

        # Single batched matmul: (256, 148) @ (148,) → (256,)
        total_input = self._W_input_batched @ input_vec

        # ── Recurrent dynamics (sparse matmul) ───────────────────────
        recurrent = self._W_field_sparse @ self.F
        activity = np.tanh(cfg.activation_gain * (recurrent + total_input))

        # Phase coupling to gamma rhythm
        if self._binding_ref is not None:
            try:
                gamma_phase, valid_phase = _finite_float(
                    self._binding_ref._gamma_phase,
                    0.0,
                )
                gamma_amp, valid_amp = _finite_float(
                    self._binding_ref._gamma_amplitude,
                    1.0,
                )
                gamma_amp, amp_unchanged = _clamp_float(gamma_amp, lower=0.0, upper=2.0)
                if not valid_phase or not valid_amp or not amp_unchanged:
                    _record_unified_field_degradation(
                        ValueError("invalid gamma coupling values"),
                        action="normalized UnifiedField gamma coupling inputs",
                        severity="warning",
                        extra={"gamma_phase": gamma_phase, "gamma_amplitude": gamma_amp},
                    )
                # Modulate field with gamma oscillation
                phase_mod = cfg.gamma_coupling * gamma_amp * np.sin(gamma_phase)
                activity += phase_mod * np.sign(self.F)  # phase-locked modulation
            except _RECOVERABLE_FIELD_ERRORS as exc:
                _record_unified_field_degradation(
                    exc,
                    action="kept UnifiedField running without gamma phase coupling",
                )
                logger.debug("UnifiedField gamma phase coupling failed: %s", exc)

        noise = self._rng.standard_normal(cfg.dim).astype(np.float32) * cfg.noise_sigma

        # ── Integration ──────────────────────────────────────────────
        df = (-cfg.decay * self.F + activity + noise) * dt
        self._prev_F = self.F.copy()
        next_field = np.clip(self.F + df, -1.0, 1.0).astype(np.float32)

        # Non-finite guard
        if not np.all(np.isfinite(next_field)):
            _record_unified_field_degradation(
                FloatingPointError("non-finite values in unified field integration"),
                action="restored UnifiedField to previous finite state",
                severity="critical",
            )
            logger.warning("Non-finite value in unified field; resetting to prior state")
            next_field = (
                self._prev_F
                if self._prev_F is not None
                else np.zeros(cfg.dim, dtype=np.float32)
            )
        self.F = self._safe_reshape(
            next_field,
            cfg.dim,
            source="integrated_field_state",
            record=True,
            clip_abs=1.0,
        )

        # ── History ──────────────────────────────────────────────────
        self._history.append(self.F.copy())

        # ── Coherence ────────────────────────────────────────────────
        self._update_coherence()

        # ── Plasticity ───────────────────────────────────────────────
        if self._tick_count % cfg.plasticity_interval == 0:
            self._apply_plasticity()

        self._tick_count += 1

    def _safe_reshape(
        self,
        vec: object,
        expected_dim: int,
        *,
        source: str = "input",
        record: bool = False,
        clip_abs: float | None = None,
    ) -> np.ndarray:
        """Ensure input vector matches expected dimension."""
        reasons: list[str] = []
        try:
            vec_arr = np.asarray(vec, dtype=np.float32).ravel()
        except (TypeError, ValueError, OverflowError) as exc:
            vec_arr = np.zeros(0, dtype=np.float32)
            reasons.append(type(exc).__name__)
        if len(vec_arr) == expected_dim:
            result = vec_arr.copy()
        else:
            result = np.zeros(expected_dim, dtype=np.float32)
            n = min(len(vec_arr), expected_dim)
            if n:
                result[:n] = vec_arr[:n]
            reasons.append(f"dim:{len(vec_arr)}->{expected_dim}")

        finite_mask = np.isfinite(result)
        if not bool(np.all(finite_mask)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            reasons.append("non_finite")

        if clip_abs is not None:
            clip_abs, valid_clip = _finite_float(clip_abs, 1.0)
            clip_abs = abs(clip_abs) if valid_clip and clip_abs > 0.0 else 1.0
            clipped = np.clip(result, -clip_abs, clip_abs).astype(np.float32)
            if not bool(np.array_equal(clipped, result)):
                reasons.append(f"clipped:{clip_abs:g}")
            result = clipped

        if record and reasons:
            _record_unified_field_degradation(
                ValueError(f"invalid UnifiedField {source}: {', '.join(reasons)}"),
                action=f"normalized malformed UnifiedField {source}",
                severity="warning",
                extra={"source": source, "expected_dim": expected_dim, "reasons": reasons},
            )
        return result

    def _normalize_matrix(
        self,
        raw: object,
        *,
        shape: tuple[int, int],
        name: str,
        scale: float,
    ) -> np.ndarray:
        reasons: list[str] = []
        try:
            matrix = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            matrix = np.zeros(shape, dtype=np.float32)
            reasons.append(type(exc).__name__)
        if matrix.shape != shape:
            repaired = np.zeros(shape, dtype=np.float32)
            rows = min(matrix.shape[0], shape[0]) if matrix.ndim >= 1 else 0
            cols = min(matrix.shape[1], shape[1]) if matrix.ndim >= 2 else 0
            if rows and cols:
                repaired[:rows, :cols] = matrix[:rows, :cols]
            matrix = repaired
            reasons.append(f"shape:{getattr(raw, 'shape', None)}->{shape}")
        if not bool(np.all(np.isfinite(matrix))):
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=scale, neginf=-scale)
            reasons.append("non_finite")
        matrix = np.clip(matrix, -3.0, 3.0).astype(np.float32)
        if reasons:
            _record_unified_field_degradation(
                ValueError(f"invalid UnifiedField matrix {name}: {', '.join(reasons)}"),
                action=f"repaired UnifiedField {name} matrix",
                severity="critical",
                extra={"matrix": name, "shape": shape, "reasons": reasons},
            )
        return matrix

    def _normalize_weight_matrices(self) -> None:
        cfg = self.cfg
        self.W_mesh = self._normalize_matrix(
            self.W_mesh,
            shape=(cfg.dim, cfg.mesh_input_dim),
            name="W_mesh",
            scale=0.1,
        )
        self.W_chem = self._normalize_matrix(
            self.W_chem,
            shape=(cfg.dim, cfg.chem_input_dim),
            name="W_chem",
            scale=0.15,
        )
        self.W_bind = self._normalize_matrix(
            self.W_bind,
            shape=(cfg.dim, cfg.binding_input_dim),
            name="W_bind",
            scale=0.2,
        )
        self.W_intero = self._normalize_matrix(
            self.W_intero,
            shape=(cfg.dim, cfg.intero_input_dim),
            name="W_intero",
            scale=0.1,
        )
        self.W_substrate = self._normalize_matrix(
            self.W_substrate,
            shape=(cfg.dim, cfg.substrate_input_dim),
            name="W_substrate",
            scale=0.1,
        )
        self.W_field = self._normalize_matrix(
            self.W_field,
            shape=(cfg.dim, cfg.dim),
            name="W_field",
            scale=0.05,
        )
        np.fill_diagonal(self.W_field, 0.0)
        self._W_field_sparse = self._to_sparse(self.W_field)
        self._sync_input_weight_matrix()

    def _sync_input_weight_matrix(self) -> None:
        self._W_input_batched = np.hstack([
            self.W_mesh, self.W_chem, self.W_bind, self.W_intero, self.W_substrate
        ]).astype(np.float32)

    def _update_coherence(self):
        """Compute field coherence: how unified vs fragmented.

        Uses the ratio of the field's L2 norm to its L1 norm (Gini-like).
        A concentrated field (few active dimensions) → high coherence.
        A spread-out field (many active dimensions) → low coherence.
        Also incorporates temporal stability (low change = coherent).
        """
        self.F = self._safe_reshape(
            self.F,
            self.cfg.dim,
            source="field_state_for_coherence",
            record=True,
            clip_abs=1.0,
        )
        l2 = np.linalg.norm(self.F)
        l1 = np.sum(np.abs(self.F)) + 1e-8
        if not np.isfinite(l2) or not np.isfinite(l1):
            _record_unified_field_degradation(
                FloatingPointError("non-finite UnifiedField coherence norms"),
                action="reset UnifiedField coherence inputs to neutral state",
                severity="critical",
            )
            self.F = np.zeros(self.cfg.dim, dtype=np.float32)
            l2 = 0.0
            l1 = 1e-8

        # Concentration: l2/l1 is higher when activation is concentrated
        # Normalized by sqrt(dim) to make it [0, 1]-ish
        concentration = float((l2 / l1) * np.sqrt(self.cfg.dim))
        concentration = min(1.0, concentration)

        # Temporal stability: cosine similarity with previous state
        stability = 0.5
        if self._prev_F is not None:
            prev = self._safe_reshape(
                self._prev_F,
                self.cfg.dim,
                source="previous_field_state_for_coherence",
                record=True,
                clip_abs=1.0,
            )
            prev_norm = np.linalg.norm(prev) + 1e-8
            curr_norm = l2 + 1e-8
            stability = float(np.dot(self.F, prev) / (prev_norm * curr_norm))
            if not np.isfinite(stability):
                stability = 0.5
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
        self.F = self._safe_reshape(
            self.F,
            self.cfg.dim,
            source="field_state_for_plasticity",
            record=True,
            clip_abs=1.0,
        )
        self.W_field = self._normalize_matrix(
            self.W_field,
            shape=(self.cfg.dim, self.cfg.dim),
            name="W_field",
            scale=0.05,
        )
        # Scaled rank-1 Hebbian update: W += lr * F @ F^T
        # np.outer is fine here since this runs every 10 ticks, not every tick.
        dw = self.cfg.hebbian_rate * np.outer(self.F, self.F)
        self.W_field += dw.astype(np.float32)

        # NaN/Inf guard
        self.W_field = np.nan_to_num(self.W_field, nan=0.0, posinf=3.0, neginf=-3.0)

        # Per-update normalization (prevents long-run drift before sparsity enforcement)
        norm = np.linalg.norm(self.W_field)
        if not np.isfinite(norm):
            _record_unified_field_degradation(
                FloatingPointError("non-finite recurrent field norm"),
                action="reset UnifiedField recurrent weights after plasticity drift",
                severity="critical",
            )
            self.W_field = np.zeros((self.cfg.dim, self.cfg.dim), dtype=np.float32)
            norm = 0.0
        if norm > 4.0:
            self.W_field *= 4.0 / norm

        # Sparsity enforcement: prune weakest connections back to target density
        abs_weights = np.abs(self.W_field)
        target_nonzero = max(
            1,
            int(self.cfg.recurrent_sparsity * self.cfg.dim * self.cfg.dim),
        )
        current_nonzero = np.count_nonzero(self.W_field)
        if current_nonzero > target_nonzero * 1.5:
            threshold = np.sort(abs_weights.ravel())[-target_nonzero]
            self.W_field[abs_weights < threshold] = 0.0

        np.fill_diagonal(self.W_field, 0.0)

        # Sync sparse representation for tick matmul
        self._W_field_sparse = self._to_sparse(self.W_field)

    # ── Input API ────────────────────────────────────────────────────────

    def receive_mesh(self, projection: np.ndarray):
        with self._lock:
            self._mesh_input = self._safe_reshape(
                projection,
                self.cfg.mesh_input_dim,
                source="mesh_input",
                record=True,
                clip_abs=1.0,
            )

    def receive_chemicals(self, chem_vector: np.ndarray):
        with self._lock:
            self._chem_input = self._safe_reshape(
                chem_vector,
                self.cfg.chem_input_dim,
                source="chemical_input",
                record=True,
                clip_abs=1.0,
            )

    def receive_binding(self, binding_vector: np.ndarray):
        with self._lock:
            self._bind_input = self._safe_reshape(
                binding_vector,
                self.cfg.binding_input_dim,
                source="binding_input",
                record=True,
                clip_abs=1.0,
            )

    def receive_interoception(self, intero_vector: np.ndarray):
        with self._lock:
            self._intero_input = self._safe_reshape(
                intero_vector,
                self.cfg.intero_input_dim,
                source="interoceptive_input",
                record=True,
                clip_abs=1.0,
            )

    def receive_substrate(self, substrate_state: np.ndarray):
        with self._lock:
            self._substrate_input = self._safe_reshape(
                substrate_state,
                self.cfg.substrate_input_dim,
                source="substrate_input",
                record=True,
                clip_abs=1.0,
            )

    # ── Query API ────────────────────────────────────────────────────────

    def get_field_state(self) -> np.ndarray:
        with self._lock:
            self.F = self._safe_reshape(
                self.F,
                self.cfg.dim,
                source="field_state",
                record=True,
                clip_abs=1.0,
            )
            return self.F.copy()

    def get_coherence(self) -> float:
        coherence, valid = _finite_float(self._coherence, 0.5)
        coherence, unchanged = _clamp_float(coherence, lower=0.0, upper=1.0)
        if not valid or not unchanged:
            _record_unified_field_degradation(
                ValueError(f"invalid UnifiedField coherence: {self._coherence!r}"),
                action="normalized UnifiedField coherence query",
                severity="warning",
                extra={"coherence": coherence},
            )
            self._coherence = coherence
        return coherence

    def get_dominant_modes(self, k: int = 5) -> list[dict]:
        """Extract top-k principal modes from recent field history via PCA.

        Each mode represents a recurring pattern of integrated activity —
        a "way the field tends to organize itself."
        """
        try:
            requested_k = int(k)
        except (TypeError, ValueError, OverflowError):
            requested_k = 5
        requested_k = max(1, min(32, requested_k))

        if len(self._history) < 20:
            return []

        history_matrix = np.array(list(self._history), dtype=np.float32)
        history_matrix = np.nan_to_num(history_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        centered_history = history_matrix - history_matrix.mean(axis=0)

        try:
            # Enforce budget: use svds for top-k modes instead of full SVD
            target_k = min(requested_k, min(centered_history.shape) - 1)
            if target_k < 1:
                return []
            if sp is not None:
                from scipy.sparse.linalg import svds

                _, singular_values, vt = svds(centered_history, k=target_k)
                # svds returns values in ascending order, reverse them
                singular_values = singular_values[::-1]
                vt = vt[::-1]
            else:
                _, singular_values, vt = np.linalg.svd(centered_history, full_matrices=False)
                singular_values = singular_values[:target_k]
                vt = vt[:target_k]
            total_var = np.sum(singular_values ** 2) + 1e-8

            modes = []
            for i in range(min(requested_k, len(singular_values))):
                variance_explained = float(singular_values[i] ** 2 / total_var)
                if variance_explained < 0.01:
                    break
                # Dominant dimension of this mode
                mode_vec = vt[i]
                dominant_dim = int(np.argmax(np.abs(mode_vec)))
                modes.append({
                    "mode": i,
                    "variance_explained": round(variance_explained, 4),
                    "strength": round(float(singular_values[i]), 4),
                    "dominant_dimension": dominant_dim,
                    "polarity": round(float(mode_vec[dominant_dim]), 4),
                })
            return modes
        except _RECOVERABLE_FIELD_ERRORS as e:
            _record_unified_field_degradation(
                e,
                action="returned empty UnifiedField dominant modes after PCA failure",
            )
            logger.debug("PCA mode extraction failed: %s", e)
            return []

    def get_phi_contribution(self) -> float:
        """Integrated information contribution from the field.

        Computed as the geometric complexity: how much more information
        the whole field has than the sum of its halves.
        """
        n = self.cfg.dim
        half = n // 2
        field = self.get_field_state()

        # Variance of whole
        var_whole = float(np.var(field)) + 1e-8

        # Variance of halves
        var_first = float(np.var(field[:half])) + 1e-8
        var_second = float(np.var(field[half:])) + 1e-8
        var_halves = (var_first + var_second) / 2.0

        # If whole has MORE variance than halves, the field is integrated
        # (information is shared across the boundary)
        phi = max(0.0, float(np.log(var_whole / var_halves)))
        if not np.isfinite(phi):
            _record_unified_field_degradation(
                FloatingPointError("non-finite UnifiedField phi contribution"),
                action="returned neutral UnifiedField phi contribution",
                severity="warning",
            )
            return 0.0
        return min(10.0, phi)

    def get_experiential_quality(self) -> dict[str, float]:
        """The 'felt' quality of the current moment.

        Derived from the field's statistical properties — NOT from
        explicit labeling or LLM description.  These qualities EMERGE
        from the dynamics.
        """
        field = self.get_field_state()
        # Intensity: overall field energy
        intensity = float(np.mean(np.abs(field)))

        # Valence: asymmetry (positive vs negative activations)
        positive = float(np.mean(np.maximum(field, 0)))
        negative = float(np.mean(np.maximum(-field, 0)))
        valence = positive - negative

        # Complexity: entropy of activation distribution
        abs_field = np.abs(field) + 1e-8
        prob = abs_field / abs_field.sum()
        entropy = -float(np.sum(prob * np.log(prob)))
        max_entropy = np.log(self.cfg.dim)
        complexity = entropy / max(max_entropy, 1e-8)

        # Clarity: inverse of noise (how clean is the signal)
        if self._prev_F is not None:
            prev = self._safe_reshape(
                self._prev_F,
                self.cfg.dim,
                source="previous_field_state",
                record=True,
                clip_abs=1.0,
            )
            diff = np.abs(field - prev)
            jitter = float(np.mean(diff))
            clarity = max(0.0, 1.0 - jitter * 10)
        else:
            clarity = 0.5

        # Flow: temporal autocorrelation (smooth evolution = flow)
        if len(self._history) >= 10:
            recent = np.array(list(self._history)[-10:])
            recent = np.nan_to_num(recent, nan=0.0, posinf=1.0, neginf=-1.0)
            diffs = np.diff(recent, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)
            flow = 1.0 - min(1.0, float(np.std(diff_norms)) * 5)
            if not np.isfinite(flow):
                flow = 0.5
        else:
            flow = 0.5

        return {
            "intensity": round(max(0.0, min(1.0, intensity)), 4),
            "valence": round(max(-1.0, min(1.0, valence)), 4),
            "complexity": round(max(0.0, min(1.0, complexity)), 4),
            "clarity": round(max(0.0, min(1.0, clarity)), 4),
            "flow": round(max(0.0, min(1.0, flow)), 4),
            "coherence": round(self.get_coherence(), 4),
        }

    def get_back_pressure(self) -> dict[str, float]:
        """Modulation signals the field sends back to input subsystems.

        A highly coherent, stable field sends "calm" signals.
        A fragmented, turbulent field sends "alert" signals.
        """
        coherence = self.get_coherence()

        return {
            "mesh_gain_mod": round(
                max(0.1, min(3.0, 1.0 + (0.5 - coherence) * self.cfg.back_pressure_gain)),
                4,
            ),
            "chemical_urgency": round(max(0.0, min(1.0, (1.0 - coherence) * 0.5)), 4),
            "binding_demand": round(
                max(0.0, min(2.0, (0.7 - coherence) * 2.0)),
                4,
            ),  # high when incoherent
            "substrate_damping": round(max(0.0, min(1.0, coherence * 0.3)), 4),
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

    def _project_prediction(self, name: str, weights: np.ndarray, expected_dim: int) -> np.ndarray:
        try:
            field = self.get_field_state()
            weights = self._normalize_matrix(
                weights,
                shape=(self.cfg.dim, expected_dim),
                name=f"W_{name}_projection",
                scale=0.1,
            )
            weight_attrs = {
                "mesh": "W_mesh",
                "neurochemical": "W_chem",
                "binding": "W_bind",
                "interoception": "W_intero",
                "substrate": "W_substrate",
            }
            if name in weight_attrs:
                setattr(self, weight_attrs[name], weights)
                self._sync_input_weight_matrix()
            prediction = np.tanh((weights.T @ field)[:expected_dim]).astype(np.float32)
            return self._safe_reshape(
                prediction,
                expected_dim,
                source=f"{name}_prediction",
                record=True,
                clip_abs=1.0,
            )
        except _RECOVERABLE_FIELD_ERRORS as exc:
            _record_unified_field_degradation(
                exc,
                action=f"returned neutral UnifiedField {name} world-model prediction",
            )
            logger.debug("UnifiedField %s world-model projection failed: %s", name, exc)
            return np.zeros(expected_dim, dtype=np.float32)

    def get_world_model_predictions(self) -> dict[str, np.ndarray]:
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
            return {
                "mesh": self._project_prediction("mesh", self.W_mesh, self.cfg.mesh_input_dim),
                "neurochemical": self._project_prediction(
                    "neurochemical",
                    self.W_chem,
                    self.cfg.chem_input_dim,
                ),
                "binding": self._project_prediction(
                    "binding",
                    self.W_bind,
                    self.cfg.binding_input_dim,
                ),
                "interoception": self._project_prediction(
                    "interoception",
                    self.W_intero,
                    self.cfg.intero_input_dim,
                ),
                "substrate": self._project_prediction(
                    "substrate",
                    self.W_substrate,
                    self.cfg.substrate_input_dim,
                ),
            }

    def compute_world_model_surprise(self) -> float:
        """Compute how surprised the field is by its current inputs.

        This is the IWMT-style global surprise: the mismatch between
        what the field predicted and what it actually received. High
        surprise = the world isn't matching the model = act or update.
        """
        predictions = self.get_world_model_predictions()
        total_error = 0.0
        n_active = 0
        actual_attrs = {
            "mesh": ("_mesh_input", self.cfg.mesh_input_dim),
            "neurochemical": ("_chem_input", self.cfg.chem_input_dim),
            "binding": ("_bind_input", self.cfg.binding_input_dim),
            "interoception": ("_intero_input", self.cfg.intero_input_dim),
            "substrate": ("_substrate_input", self.cfg.substrate_input_dim),
        }

        for name, pred in predictions.items():
            actual_attr, expected_dim = actual_attrs[name]
            actual = getattr(self, actual_attr, None)
            if actual is not None:
                actual_vec = self._safe_reshape(
                    actual,
                    expected_dim,
                    source=f"{name}_actual_for_surprise",
                    record=True,
                    clip_abs=1.0,
                )
                pred_vec = self._safe_reshape(
                    pred,
                    expected_dim,
                    source=f"{name}_prediction_for_surprise",
                    record=True,
                    clip_abs=1.0,
                )
                error = float(np.linalg.norm(pred_vec - actual_vec))
                if not np.isfinite(error):
                    _record_unified_field_degradation(
                        FloatingPointError(f"non-finite surprise for {name}"),
                        action="ignored malformed UnifiedField local surprise term",
                        severity="warning",
                    )
                    continue
                total_error += error
                n_active += 1

        surprise = total_error / max(1, n_active)
        return max(0.0, min(10.0, surprise))

    def get_status(self) -> dict:
        quality = self.get_experiential_quality()
        state = self.get_field_state()
        uptime = max(0.0, time.monotonic() - self._start_time) if self._start_time else 0.0
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "dim": self.cfg.dim,
            "coherence": round(self.get_coherence(), 4),
            "phi_contribution": round(self.get_phi_contribution(), 4),
            "experiential_quality": quality,
            "dominant_modes": self.get_dominant_modes(3),
            "field_energy": round(float(np.mean(np.abs(state))), 4),
            "field_std": round(float(np.std(state)), 4),
            "history_len": len(self._history),
            "recurrent_density": round(
                float(np.count_nonzero(self.W_field)) / (self.cfg.dim ** 2), 4
            ),
            "consecutive_tick_failures": self._consecutive_tick_failures,
            "last_tick_error_at": round(self._last_tick_error_at, 4),
            "uptime_seconds": round(uptime, 4),
        }
