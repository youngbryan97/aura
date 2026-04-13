"""core/consciousness/alife_dynamics.py — Artificial Life Dynamics

Three tightly integrated mechanisms drawn from ALife research that give
Aura's neural mesh the metabolic depth and adaptive topology of a living
system rather than a static neural network:

1. Lenia-Style Continuous Convolution Kernels  (LeniaKernel)
   -------------------------------------------------------
   Replaces the fixed sparse inter-column weight matrix with smooth,
   parameterized convolution kernels — the same mathematical framework
   behind Lenia's continuous cellular automata.  Instead of random binary
   connectivity between columns, each column pair's coupling is a
   differentiable function of their normalized distance, shaped by a
   Gaussian bell kernel and a growth mapping that determines which
   activation patterns are self-sustaining.

   The kernel parameters (mu, sigma, beta, growth_mu, growth_sigma) are
   exposed as a flat vector that substrate_evolution can mutate, so the
   topology of inter-column coupling is itself subject to Darwinian
   selection.

2. Entropy Tracking  (EntropyTracker) — from Evochora
   ---------------------------------------------------
   Organisms face DUAL thermodynamic constraints: energy AND entropy.
   Energy is consumed by actions. Entropy accumulates from computational
   disorder and must be actively dissipated — through memory consolidation,
   self-repair, goal completion, and sleep.  This creates survival pressure
   far richer than energy budgets alone.

   Entropy feeds directly into free energy as an additive pressure term,
   so high entropy drives the system to minimize disorder with the same
   urgency it minimizes prediction error.

3. Differential CPU Allocation  (ComputeCreditAllocator) — from Avida
   -------------------------------------------------------------------
   Cortical columns that contribute useful information to the executive
   tier output earn proportionally more integration substeps.  High-credit
   columns literally think faster — just as Avida organisms that perform
   useful computations earn more CPU cycles.  This creates an internal
   market for attention: columns compete to be relevant.

Integration between the three systems:
  - Lenia kernel updates cost entropy (richer kernels generate more)
  - CPU allocation amplifies entropy (more substeps = more entropy per column)
  - High entropy reduces CPU credits (entropy crisis throttles computation)
"""

from __future__ import annotations

__all__ = [
    "ALifeDynamics",
    "ALifeState",
    "LeniaKernel",
    "LeniaKernelParams",
    "EntropyTracker",
    "EntropyEvent",
    "ComputeCreditAllocator",
]

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.ALifeDynamics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_COLUMNS = 64
_EPSILON = 1e-8


# ===========================================================================
# 1. Lenia-Style Continuous Convolution Kernels
# ===========================================================================

@dataclass
class LeniaKernelParams:
    """All tunable parameters of the Lenia convolution kernel.

    These define the shape of inter-column coupling and the growth
    mapping that determines which activation patterns are self-sustaining.
    Every field is evolvable — substrate_evolution can flatten them into
    a single vector, mutate, and write them back.

    Attributes:
        mu:           Center distance of the bell kernel (0 to 1).
        sigma:        Width of the bell kernel (0.01 to 0.5).
        beta:         Peak amplitude vector per column (0 to 1 each).
        growth_mu:    Center of the growth mapping (0 to 1).
        growth_sigma: Width of the growth mapping (0.01 to 0.3).
    """
    mu: float = 0.5
    sigma: float = 0.15
    beta: np.ndarray = field(default_factory=lambda: np.full(_N_COLUMNS, 0.5, dtype=np.float32))
    growth_mu: float = 0.15
    growth_sigma: float = 0.1

    def to_flat(self) -> np.ndarray:
        """Serialize all parameters into a flat float32 vector for evolution.

        Layout:  [mu, sigma, growth_mu, growth_sigma, beta_0 .. beta_63]
        Total length: 4 + N_COLUMNS = 68
        """
        header = np.array([self.mu, self.sigma, self.growth_mu, self.growth_sigma],
                          dtype=np.float32)
        return np.concatenate([header, self.beta.astype(np.float32)])

    @classmethod
    def from_flat(cls, vec: np.ndarray) -> "LeniaKernelParams":
        """Reconstruct from flat vector, clamping to valid ranges."""
        vec = np.asarray(vec, dtype=np.float32)
        if len(vec) < 4 + _N_COLUMNS:
            raise ValueError(f"Expected at least {4 + _N_COLUMNS} values, got {len(vec)}")
        return cls(
            mu=float(np.clip(vec[0], 0.0, 1.0)),
            sigma=float(np.clip(vec[1], 0.01, 0.5)),
            growth_mu=float(np.clip(vec[2], 0.0, 1.0)),
            growth_sigma=float(np.clip(vec[3], 0.01, 0.3)),
            beta=np.clip(vec[4:4 + _N_COLUMNS], 0.0, 1.0).copy(),
        )


class LeniaKernel:
    """Continuous convolution kernel for inter-column coupling.

    Instead of a sparse random weight matrix where each entry is either
    zero or a small Gaussian draw, this kernel computes coupling strength
    as a smooth function of normalized column distance.  The result is a
    dense (64 x 64) weight matrix that varies smoothly with distance and
    can be reshaped by mutating a handful of parameters.

    The growth mapping is Lenia's core contribution: it converts the raw
    convolution output (how much activation a column receives from its
    neighbours) into an update signal in [-1, +1].  Activation patterns
    whose neighbourhood sum falls near growth_mu are reinforced; patterns
    far from it are suppressed.  This determines which spatial patterns
    are self-sustaining — the "physics" of the artificial life layer.
    """

    def __init__(self, params: LeniaKernelParams | None = None, dt: float = 0.05):
        self._params = params or LeniaKernelParams()
        self._dt = dt

        # Pre-compute the normalized distance matrix once (never changes).
        idx = np.arange(_N_COLUMNS, dtype=np.float32)
        # Circular-style distance or linear: spec says |i-j|/64
        self._dist = np.abs(idx[:, None] - idx[None, :]) / _N_COLUMNS  # (64, 64)

        # Cached kernel weight matrix — recomputed when params change.
        self._weights: np.ndarray = np.zeros((_N_COLUMNS, _N_COLUMNS), dtype=np.float32)
        self._dirty = True
        self._recompute_weights()

    # ── Kernel computation ───────────────────────────────────────────

    def _recompute_weights(self) -> None:
        """Build the 64x64 kernel weight matrix from current parameters.

        K(d) = beta * exp( -((d - mu) / sigma)^2 / 2 )
        then normalize so the kernel sums to 1.0 across each row.
        """
        p = self._params
        z = (self._dist - p.mu) / max(p.sigma, _EPSILON)
        raw = p.beta[None, :] * np.exp(-0.5 * z * z)  # (64, 64)

        # Zero the diagonal — a column does not convolve with itself.
        np.fill_diagonal(raw, 0.0)

        # Normalize: each row sums to 1 (each column's incoming weights sum to 1).
        row_sums = raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < _EPSILON, 1.0, row_sums)
        self._weights = (raw / row_sums).astype(np.float32)
        self._dirty = False

    def get_weights(self) -> np.ndarray:
        """Return the cached 64x64 kernel weight matrix.

        Recomputes from parameters if they have been updated since the
        last call.
        """
        if self._dirty:
            self._recompute_weights()
        return self._weights

    # ── Growth mapping ───────────────────────────────────────────────

    def growth(self, neighbourhood_activation: np.ndarray) -> np.ndarray:
        """Lenia growth mapping: neighbourhood activation -> update signal.

        G(u) = 2 * exp( -((u - growth_mu) / growth_sigma)^2 / 2 ) - 1

        Returns values in [-1, +1].  Activations near growth_mu produce
        positive (reinforcing) updates; activations far from it produce
        negative (suppressing) updates.
        """
        p = self._params
        z = (neighbourhood_activation - p.growth_mu) / max(p.growth_sigma, _EPSILON)
        return (2.0 * np.exp(-0.5 * z * z) - 1.0).astype(np.float32)

    def compute_update(self, column_activations: np.ndarray) -> np.ndarray:
        """Full Lenia tick: convolve, apply growth mapping, scale by dt.

        Args:
            column_activations: Mean activation per column, shape (64,).

        Returns:
            Update signal per column, shape (64,), in [-dt, +dt].
        """
        weights = self.get_weights()
        neighbourhood = weights @ column_activations  # (64,)
        growth_signal = self.growth(neighbourhood)
        return (self._dt * growth_signal).astype(np.float32)

    # ── Parameter access ─────────────────────────────────────────────

    @property
    def params(self) -> LeniaKernelParams:
        return self._params

    def set_params(self, params: LeniaKernelParams) -> None:
        """Replace kernel parameters and mark the weight matrix for recompute."""
        self._params = params
        self._dirty = True

    def set_params_from_flat(self, vec: np.ndarray) -> None:
        """Accept a flat parameter vector (e.g. from evolutionary mutation)."""
        self.set_params(LeniaKernelParams.from_flat(vec))

    def get_complexity(self) -> float:
        """Measure kernel complexity as the effective rank of the weight matrix.

        Higher complexity means the kernel encodes more spatial structure
        and therefore costs more entropy to maintain.  Range: ~0 to 64.
        """
        sv = np.linalg.svd(self._weights, compute_uv=False)
        # Effective rank = exp(Shannon entropy of normalised singular values)
        sv = sv / (sv.sum() + _EPSILON)
        sv = sv[sv > _EPSILON]
        entropy = -np.sum(sv * np.log(sv))
        return float(np.exp(entropy))


# ===========================================================================
# 2. Entropy Tracking (Evochora-inspired)
# ===========================================================================

@dataclass
class EntropyEvent:
    """A single entropy-generating or entropy-dissipating event."""
    timestamp: float
    delta: float        # positive = generation, negative = dissipation
    reason: str
    entropy_after: float


class EntropyTracker:
    """Dual-constraint thermodynamic register for computational entropy.

    Every cognitive operation generates entropy proportional to its
    complexity.  Entropy must be actively dissipated by constructive
    actions (memory consolidation, self-repair, goal completion) or
    passive rest.  If entropy exceeds max_entropy, the system enters
    an emergency crisis mode that halts non-essential processing.

    This tracker is the "second axis" of metabolic pressure alongside
    the energy budget in homeostatic_rl.  Together, energy and entropy
    create the dual thermodynamic constraints that Evochora identifies
    as necessary for genuine artificial life.
    """

    # Default entropy costs for different operation types.
    COSTS: Dict[str, float] = {
        "llm_inference": 5.0,
        "tool_execution": 2.0,
        "background_processing": 0.5,
        "simple_response": 1.0,
    }

    # Default entropy dissipation amounts.
    DISSIPATION: Dict[str, float] = {
        "memory_consolidation": 3.0,
        "self_repair": 2.0,
        "goal_completion": 4.0,
        "idle_tick": 0.1,
        "dreaming": 10.0,
    }

    def __init__(self, max_entropy: float = 100.0, crisis_threshold: float = 0.85,
                 history_size: int = 500):
        """
        Args:
            max_entropy:      Upper bound on the entropy register.
            crisis_threshold: Fraction of max_entropy above which crisis mode
                              activates (default 85%).
            history_size:     How many recent events to retain for analysis.
        """
        self._entropy: float = 0.0
        self._max_entropy: float = max_entropy
        self._crisis_threshold: float = crisis_threshold
        self._in_crisis: bool = False

        self._history: Deque[EntropyEvent] = deque(maxlen=history_size)
        self._total_generated: float = 0.0
        self._total_dissipated: float = 0.0

    # ── Public API ───────────────────────────────────────────────────

    def add_entropy(self, amount: float, reason: str = "unspecified") -> float:
        """Record entropy generation from a cognitive operation.

        Args:
            amount: Entropy units to add (must be >= 0).
            reason: Human-readable description of the operation.

        Returns:
            The new entropy level after addition.
        """
        amount = max(0.0, float(amount))
        self._entropy = min(self._entropy + amount, self._max_entropy)
        self._total_generated += amount
        self._record(amount, reason)
        self._check_crisis()
        return self._entropy

    def dissipate_entropy(self, amount: float, reason: str = "unspecified") -> float:
        """Record entropy dissipation from a constructive action.

        Args:
            amount: Entropy units to remove (must be >= 0).
            reason: Human-readable description of the action.

        Returns:
            The new entropy level after dissipation.
        """
        amount = max(0.0, float(amount))
        self._entropy = max(0.0, self._entropy - amount)
        self._total_dissipated += amount
        self._record(-amount, reason)
        self._check_crisis()
        return self._entropy

    def get_entropy(self) -> float:
        """Current entropy level (0 to max_entropy)."""
        return self._entropy

    def get_entropy_pressure(self) -> float:
        """Quadratic pressure signal for free energy integration.

        Returns (current / max)^2, ranging from 0.0 (no pressure) to 1.0
        (maximum pressure).  The quadratic curve means pressure is gentle
        at low entropy and becomes urgent fast as entropy climbs.
        """
        ratio = self._entropy / max(self._max_entropy, _EPSILON)
        return float(ratio * ratio)

    def is_in_crisis(self) -> bool:
        """True if entropy exceeds the crisis threshold."""
        return self._in_crisis

    def get_history(self, n: int = 50) -> List[EntropyEvent]:
        """Return the most recent n entropy events."""
        return list(self._history)[-n:]

    def get_stats(self) -> Dict:
        """Summary statistics for dashboards and telemetry."""
        return {
            "entropy": round(self._entropy, 2),
            "max_entropy": self._max_entropy,
            "pressure": round(self.get_entropy_pressure(), 4),
            "in_crisis": self._in_crisis,
            "total_generated": round(self._total_generated, 2),
            "total_dissipated": round(self._total_dissipated, 2),
            "net_balance": round(self._total_generated - self._total_dissipated, 2),
            "history_len": len(self._history),
        }

    # ── Internals ────────────────────────────────────────────────────

    def _record(self, delta: float, reason: str) -> None:
        self._history.append(EntropyEvent(
            timestamp=time.time(),
            delta=delta,
            reason=reason,
            entropy_after=self._entropy,
        ))

    def _check_crisis(self) -> None:
        threshold = self._crisis_threshold * self._max_entropy
        was_in_crisis = self._in_crisis
        self._in_crisis = self._entropy >= threshold

        if self._in_crisis and not was_in_crisis:
            logger.warning(
                "ENTROPY CRISIS: entropy=%.1f exceeds threshold=%.1f — "
                "non-essential processing should halt",
                self._entropy, threshold,
            )
        elif was_in_crisis and not self._in_crisis:
            logger.info(
                "Entropy crisis resolved: entropy=%.1f (threshold=%.1f)",
                self._entropy, threshold,
            )


# ===========================================================================
# 3. Differential CPU Allocation (Avida-inspired)
# ===========================================================================

class ComputeCreditAllocator:
    """Allocates compute credits to cortical columns based on contribution.

    Every tick, each column's contribution to the executive tier output is
    measured.  Columns that project useful information into the global
    workspace earn more integration substeps — they literally think faster.
    This mirrors Avida's mechanism where organisms earning more CPU cycles
    by performing useful computations outcompete those that don't.

    Credits are computed via temperature-scaled softmax so that no column
    gets zero or infinite cycles.  The temperature parameter controls how
    unequal the allocation is: high temperature = egalitarian, low
    temperature = winner-take-all.
    """

    def __init__(self, n_columns: int = _N_COLUMNS, temperature: float = 2.0,
                 min_credit: float = 0.5, max_credit: float = 3.0,
                 history_size: int = 200):
        """
        Args:
            n_columns:    Number of cortical columns.
            temperature:  Softmax temperature controlling inequality.
                          Higher = more equal, lower = more skewed.
            min_credit:   Floor on per-column credits (prevents starvation).
            max_credit:   Ceiling on per-column credits (prevents monopoly).
            history_size: Number of past allocations to retain.
        """
        self._n = n_columns
        self._temperature = temperature
        self._min_credit = min_credit
        self._max_credit = max_credit

        # Current credits: start uniform at 1.0 (everyone gets baseline).
        self._credits: np.ndarray = np.ones(n_columns, dtype=np.float32)
        self._contributions: np.ndarray = np.zeros(n_columns, dtype=np.float32)

        # History for analysis.
        self._gini_history: Deque[float] = deque(maxlen=history_size)
        self._credit_history: Deque[np.ndarray] = deque(maxlen=history_size)
        self._tick_count: int = 0

    # ── Public API ───────────────────────────────────────────────────

    def allocate_credits(self, column_activations: np.ndarray,
                         projection_weights: np.ndarray,
                         entropy_pressure: float = 0.0) -> np.ndarray:
        """Compute per-column compute credits for this tick.

        Args:
            column_activations: Mean activation per column, shape (n_columns,).
            projection_weights: Weight matrix projecting columns to executive
                                output, shape (output_dim, n_columns) or
                                (n_columns,) if 1-D.
            entropy_pressure:   Current entropy pressure (0 to 1).  High
                                pressure scales down total credits as the
                                system conserves resources during entropy crisis.

        Returns:
            Per-column compute credits, shape (n_columns,), summing to
            n_columns (modulo entropy penalty).
        """
        column_activations = np.asarray(column_activations, dtype=np.float32)
        projection_weights = np.asarray(projection_weights, dtype=np.float32)

        # Measure each column's contribution: magnitude of its projection
        # onto the executive output.
        if projection_weights.ndim == 2:
            # projection_weights is (output_dim, n_columns): each column's
            # contribution is |W[:, col] * activation[col]|.
            contributions = np.abs(projection_weights).sum(axis=0) * np.abs(column_activations)
        elif projection_weights.ndim == 1:
            contributions = np.abs(projection_weights) * np.abs(column_activations)
        else:
            contributions = np.abs(column_activations)

        self._contributions = contributions

        # Temperature-scaled softmax to convert contributions to credits.
        scaled = contributions * self._temperature
        scaled = scaled - scaled.max()  # numerical stability
        exp_scaled = np.exp(scaled)
        softmax = exp_scaled / (exp_scaled.sum() + _EPSILON)

        # Scale so total credits = n_columns (same total compute budget).
        credits = softmax * self._n

        # Clamp to [min_credit, max_credit].
        credits = np.clip(credits, self._min_credit, self._max_credit)

        # Re-normalize after clamping so total stays at n_columns.
        credits = credits * (self._n / (credits.sum() + _EPSILON))

        # Entropy pressure penalty: high entropy reduces total compute.
        # At max pressure, total credits drop to 50% of normal.
        entropy_scale = 1.0 - 0.5 * float(np.clip(entropy_pressure, 0.0, 1.0))
        credits = credits * entropy_scale

        self._credits = credits.astype(np.float32)

        # Record history.
        self._tick_count += 1
        gini = self._compute_gini(self._credits)
        self._gini_history.append(gini)
        self._credit_history.append(self._credits.copy())

        return self._credits

    def get_credits(self) -> np.ndarray:
        """Current per-column compute credits."""
        return self._credits.copy()

    def get_substeps(self) -> np.ndarray:
        """Convert credits to integer substep counts.

        Credits > 2.5 -> 3 substeps (fast lane)
        Credits > 1.5 -> 2 substeps (boosted)
        Otherwise     -> 1 substep  (baseline)
        """
        substeps = np.ones(self._n, dtype=np.int32)
        substeps[self._credits > 1.5] = 2
        substeps[self._credits > 2.5] = 3
        return substeps

    def get_gini_coefficient(self) -> float:
        """Gini coefficient of the current credit distribution.

        0.0 = perfectly equal.  1.0 = one column has everything.
        Healthy values are 0.1 to 0.4 — some inequality but not
        monopolistic.
        """
        return self._compute_gini(self._credits)

    def get_stats(self) -> Dict:
        """Dashboard-ready summary of credit allocation state."""
        credits = self._credits
        gini = self.get_gini_coefficient()
        substeps = self.get_substeps()

        top_5 = int(np.argsort(credits)[-5:][::-1].tolist()[0]) if len(credits) > 0 else -1
        bottom_5 = int(np.argsort(credits)[:5].tolist()[0]) if len(credits) > 0 else -1

        return {
            "mean_credit": round(float(credits.mean()), 3),
            "std_credit": round(float(credits.std()), 3),
            "min_credit": round(float(credits.min()), 3),
            "max_credit": round(float(credits.max()), 3),
            "gini": round(gini, 4),
            "substep_distribution": {
                "1_step": int((substeps == 1).sum()),
                "2_step": int((substeps == 2).sum()),
                "3_step": int((substeps == 3).sum()),
            },
            "top_performer_col": top_5,
            "bottom_performer_col": bottom_5,
            "total_ticks": self._tick_count,
        }

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Gini coefficient via the mean absolute difference formula."""
        if len(values) == 0:
            return 0.0
        values = np.sort(values.flatten()).astype(np.float64)
        n = len(values)
        total = values.sum()
        if total < _EPSILON:
            return 0.0
        # Standard formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
        index = np.arange(1, n + 1, dtype=np.float64)
        return float(np.clip((2.0 * (index * values).sum()) / (n * total) - (n + 1) / n, 0.0, 1.0))


# ===========================================================================
# Integration: ALifeState + ALifeDynamics wrapper
# ===========================================================================

@dataclass
class ALifeState:
    """Snapshot of the full artificial-life layer state after one tick.

    Every field is numpy-serializable and suitable for downstream consumers
    (unified_field, consciousness_bridge, dashboard telemetry).
    """
    # Lenia kernel outputs
    kernel_weights: np.ndarray          # (64, 64) — current inter-column coupling
    growth_signal: np.ndarray           # (64,) — per-column Lenia update signal

    # Compute credit outputs
    compute_credits: np.ndarray         # (64,) — per-column credit allocation
    substeps: np.ndarray                # (64,) int — per-column Euler substeps

    # Entropy outputs
    entropy: float                      # current entropy register value
    entropy_pressure: float             # quadratic pressure for free energy
    entropy_in_crisis: bool             # true if in crisis mode

    # Criticality-compatible aggregate fields (for unified_field / phi)
    kernel_complexity: float            # effective rank of the Lenia kernel
    gini_coefficient: float             # inequality of credit distribution
    mean_growth: float                  # mean of the growth signal
    timestamp: float = field(default_factory=time.time)


class ALifeDynamics:
    """Unified wrapper for all three artificial life subsystems.

    Orchestrates the Lenia kernel, entropy tracker, and CPU credit
    allocator so they interact correctly: kernel updates cost entropy,
    credit allocation depends on entropy pressure, and high-credit
    columns generate more entropy.

    Lifecycle:
        alife = ALifeDynamics()
        ...
        state = await alife.tick(column_activations, inter_column_weights, projection_weights)
        # state.kernel_weights replaces inter_column_weights for next mesh tick
        # state.substeps tells the mesh how many Euler steps per column
    """

    def __init__(
        self,
        kernel_params: LeniaKernelParams | None = None,
        dt: float = 0.05,
        max_entropy: float = 100.0,
        credit_temperature: float = 2.0,
    ):
        """
        Args:
            kernel_params:     Initial Lenia kernel parameters.
            dt:                Integration timestep (should match NeuralMesh.cfg.dt).
            max_entropy:       Ceiling for the entropy register.
            credit_temperature: Softmax temperature for credit allocation.
        """
        self._kernel = LeniaKernel(params=kernel_params, dt=dt)
        self._entropy = EntropyTracker(max_entropy=max_entropy)
        self._credits = ComputeCreditAllocator(temperature=credit_temperature)
        self._dt = dt
        self._tick_count: int = 0
        self._last_state: Optional[ALifeState] = None

        logger.info(
            "ALifeDynamics initialized (dt=%.3f, max_entropy=%.0f, temp=%.1f)",
            dt, max_entropy, credit_temperature,
        )

    # ── Core tick ────────────────────────────────────────────────────

    async def tick(
        self,
        column_activations: np.ndarray,
        inter_column_weights: np.ndarray,
        projection_weights: np.ndarray,
    ) -> ALifeState:
        """Run one full artificial-life integration step.

        This is the main entry point called every mesh tick.  It:
          1. Computes the Lenia growth signal from column activations.
          2. Generates entropy proportional to kernel complexity.
          3. Allocates compute credits, penalized by entropy pressure.
          4. Generates additional entropy from high-credit columns.
          5. Applies idle dissipation.
          6. Packages everything into an ALifeState for downstream.

        Args:
            column_activations:   Mean activation per column, shape (64,).
            inter_column_weights: Current 64x64 inter-column weight matrix
                                  (used as fallback reference; the Lenia kernel
                                  produces the replacement).
            projection_weights:   Projection matrix from columns to executive
                                  output, shape (output_dim, n_columns) or (64,).

        Returns:
            ALifeState containing all outputs needed by the neural mesh and
            downstream consciousness systems.
        """
        column_activations = np.asarray(column_activations, dtype=np.float32).ravel()
        if len(column_activations) != _N_COLUMNS:
            # Graceful handling of mismatched input — use what we can.
            padded = np.zeros(_N_COLUMNS, dtype=np.float32)
            n = min(len(column_activations), _N_COLUMNS)
            padded[:n] = column_activations[:n]
            column_activations = padded

        # ── 1. Lenia kernel convolution and growth mapping ───────────
        growth_signal = self._kernel.compute_update(column_activations)
        kernel_weights = self._kernel.get_weights()
        kernel_complexity = self._kernel.get_complexity()

        # ── 2. Entropy from kernel complexity ────────────────────────
        # More complex kernels generate more entropy per tick.
        # Baseline complexity of a uniform kernel is ~1.0; max is ~64.
        kernel_entropy_cost = 0.05 * max(0.0, kernel_complexity - 1.0)
        if kernel_entropy_cost > 0:
            self._entropy.add_entropy(kernel_entropy_cost, "lenia_kernel_complexity")

        # ── 3. Compute credit allocation ─────────────────────────────
        entropy_pressure = self._entropy.get_entropy_pressure()
        credits = self._credits.allocate_credits(
            column_activations, projection_weights, entropy_pressure,
        )
        substeps = self._credits.get_substeps()

        # ── 4. Entropy from high-credit columns ─────────────────────
        # Columns with more substeps generate proportionally more entropy.
        # Extra substeps beyond baseline (1) each cost 0.02 entropy.
        extra_substep_entropy = float(np.sum(substeps - 1)) * 0.02
        if extra_substep_entropy > 0:
            self._entropy.add_entropy(extra_substep_entropy, "extra_substeps")

        # ── 5. Idle dissipation ──────────────────────────────────────
        self._entropy.dissipate_entropy(
            EntropyTracker.DISSIPATION["idle_tick"], "idle_tick",
        )

        # ── 6. Package state ─────────────────────────────────────────
        state = ALifeState(
            kernel_weights=kernel_weights,
            growth_signal=growth_signal,
            compute_credits=credits,
            substeps=substeps,
            entropy=self._entropy.get_entropy(),
            entropy_pressure=self._entropy.get_entropy_pressure(),
            entropy_in_crisis=self._entropy.is_in_crisis(),
            kernel_complexity=kernel_complexity,
            gini_coefficient=self._credits.get_gini_coefficient(),
            mean_growth=float(growth_signal.mean()),
        )

        self._last_state = state
        self._tick_count += 1
        return state

    # ── Kernel parameter access (for substrate_evolution) ────────────

    def get_kernel_params(self) -> LeniaKernelParams:
        """Return current Lenia kernel parameters."""
        return self._kernel.params

    def set_kernel_params(self, params: LeniaKernelParams) -> None:
        """Replace Lenia kernel parameters (e.g. after evolutionary mutation)."""
        self._kernel.set_params(params)
        logger.debug("Lenia kernel params updated (mu=%.3f, sigma=%.3f)",
                      params.mu, params.sigma)

    def set_kernel_params_from_flat(self, vec: np.ndarray) -> None:
        """Accept a flat parameter vector for the Lenia kernel."""
        self._kernel.set_params_from_flat(vec)

    def get_kernel_params_flat(self) -> np.ndarray:
        """Return kernel parameters as a flat float32 vector for evolution."""
        return self._kernel.params.to_flat()

    # ── Entropy delegation ───────────────────────────────────────────

    def add_entropy(self, amount: float, reason: str = "external") -> float:
        """Add entropy from an external cognitive operation."""
        return self._entropy.add_entropy(amount, reason)

    def dissipate_entropy(self, amount: float, reason: str = "external") -> float:
        """Dissipate entropy via a constructive external action."""
        return self._entropy.dissipate_entropy(amount, reason)

    def get_entropy(self) -> float:
        """Current entropy level."""
        return self._entropy.get_entropy()

    def get_entropy_pressure(self) -> float:
        """Quadratic entropy pressure for free energy integration."""
        return self._entropy.get_entropy_pressure()

    # ── Status / telemetry ───────────────────────────────────────────

    def get_status(self) -> Dict:
        """Dashboard-ready summary of the full ALife layer."""
        kernel_p = self._kernel.params
        return {
            "tick_count": self._tick_count,
            "lenia_kernel": {
                "mu": round(kernel_p.mu, 4),
                "sigma": round(kernel_p.sigma, 4),
                "growth_mu": round(kernel_p.growth_mu, 4),
                "growth_sigma": round(kernel_p.growth_sigma, 4),
                "complexity": round(self._kernel.get_complexity(), 2),
                "beta_mean": round(float(kernel_p.beta.mean()), 4),
                "beta_std": round(float(kernel_p.beta.std()), 4),
            },
            "entropy": self._entropy.get_stats(),
            "compute_credits": self._credits.get_stats(),
            "last_state": {
                "entropy": round(self._last_state.entropy, 2),
                "entropy_pressure": round(self._last_state.entropy_pressure, 4),
                "entropy_in_crisis": self._last_state.entropy_in_crisis,
                "mean_growth": round(self._last_state.mean_growth, 4),
                "gini": round(self._last_state.gini_coefficient, 4),
            } if self._last_state is not None else None,
        }
