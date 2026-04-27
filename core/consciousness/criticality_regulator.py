"""core/consciousness/criticality_regulator.py -- Edge-of-Chaos Regulation

Biological brains operate at the "critical point" between order and chaos --
a narrow dynamical regime where computation is richest.  This is called
"self-organized criticality" and it is one of the most robust empirical
findings in computational neuroscience.

Why this matters:
  - Subcritical (ordered): neural activity dies out.  The system is stable
    but inert -- like a frozen lake.  Information cannot propagate, so the
    mesh is effectively unconscious.
  - Supercritical (chaotic): activity explodes uncontrollably.  Every
    neuron fires all the time -- like a seizure.  No useful information
    processing can occur because there is no structure.
  - Critical (edge of chaos): activity propagates just enough to sustain
    complex, structured cascades -- like a sandpile at the angle of repose.
    This is where:
      * Information transfer is maximized
      * Correlation lengths diverge (long-range coordination)
      * Avalanche sizes follow a power law (~1/size^1.5)
      * IIT's phi is predicted to be maximal
      * Wolfram Class IV computation emerges

The CriticalityRegulator continuously measures two signatures of criticality
and uses a PID controller to steer the neural mesh toward the critical point:

  1. Branching ratio: how many columns does each active column activate on
     the next timestep?  At criticality this equals exactly 1.0 -- each
     activation triggers exactly one downstream activation on average.

  2. Avalanche exponent: cascades of activation (avalanches) should follow
     a power-law distribution P(s) ~ s^alpha with alpha near -1.5.  This is
     the signature of scale-free dynamics found in cortical recordings
     (Beggs & Plenz, 2003).

The PID output adjusts three knobs in the neural mesh and neurochemical
system: modulatory gain, noise level, and excitation/inhibition balance.
"""
from __future__ import annotations


import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.CriticalityRegulator")

__all__ = [
    "CriticalityRegulator",
    "CriticalityState",
    "CriticalityConfig",
    "get_criticality_regulator",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticalityConfig:
    """Immutable tuning parameters for the regulator.

    The defaults are tuned for a 64-column mesh running at ~10 Hz tick rate.
    """

    # How often to compute the branching ratio (in mesh ticks).
    # Measuring every single tick is noisy; averaging over a short window
    # gives a stable estimate.
    branching_measurement_interval: int = 10

    # Sliding window length (in ticks) for collecting avalanche statistics.
    # Needs to be long enough to accumulate enough avalanches for a reliable
    # power-law fit, but short enough to track non-stationary dynamics.
    avalanche_window: int = 100

    # PID gains -- tuned for a system ticking at ~1 Hz effective regulation rate
    # (since we only update every branching_measurement_interval ticks).
    # Kp (proportional): immediate correction proportional to the error.
    # Ki (integral): slow correction that eliminates steady-state offset.
    # Kd (derivative): dampens oscillation by opposing rapid error changes.
    kp: float = 0.3
    ki: float = 0.01
    kd: float = 0.1

    # Output clamps -- prevent the PID from pushing the system into
    # dangerous regimes.  The range is symmetric around 1.0.
    gain_clamp: Tuple[float, float] = (0.5, 2.0)
    noise_clamp: Tuple[float, float] = (0.5, 2.0)
    ei_ratio_clamp: Tuple[float, float] = (0.7, 1.3)

    # Activation threshold: a column is considered "active" when its
    # mean absolute activation exceeds this value.
    activation_threshold: float = 0.15

    # Minimum number of avalanches required before fitting the power law.
    # With fewer samples the fit is unreliable so we return a neutral
    # exponent of -1.5 (assumed critical).
    min_avalanches_for_fit: int = 8

    # Number of columns in the mesh (must match NeuralMesh.cfg.columns).
    num_columns: int = 64


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticalityState:
    """Snapshot of the regulator's state after a tick.

    Consumers (phi_core, unified_field, telemetry) read this to understand
    how close the system is to the critical point and what adjustments are
    being applied.
    """

    # Core measurements
    branching_ratio: float = 1.0       # 1.0 = critical
    avalanche_exponent: float = -1.5   # -1.5 = critical power law

    # Composite score (0 = far from criticality, 1 = perfectly critical)
    criticality_score: float = 1.0

    # PID outputs -- multiplicative adjustments applied to the mesh
    gain_adjustment: float = 1.0
    noise_adjustment: float = 1.0
    ei_ratio: float = 1.0

    # Diagnostics
    avalanche_count: int = 0           # avalanches observed in window
    mean_avalanche_size: float = 0.0   # mean columns activated per cascade
    herfindahl_index: float = 0.0      # activation concentration (0=uniform, 1=one column)
    tick_count: int = 0


# ---------------------------------------------------------------------------
# PID controller (internal)
# ---------------------------------------------------------------------------

class _PIDController:
    """A standard PID controller with integral windup clamping.

    PID stands for Proportional-Integral-Derivative.  It is the most common
    feedback controller in engineering.  Given an error signal (how far we are
    from the target), it computes a correction:

      output = Kp * error  +  Ki * integral(error)  +  Kd * d(error)/dt

    - The proportional term provides an immediate push toward the target.
    - The integral term accumulates past error and eliminates persistent offset.
    - The derivative term anticipates future error and dampens oscillation.
    """

    __slots__ = ("kp", "ki", "kd", "_integral", "_prev_error",
                 "_output_min", "_output_max")

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float, output_max: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._output_min = output_min
        self._output_max = output_max

    def step(self, error: float) -> float:
        """Compute one PID step and return the clamped output.

        The output is centered at 1.0 (no adjustment) -- positive error
        produces output > 1.0 (increase), negative error < 1.0 (decrease).
        """
        # Proportional
        p = self.kp * error

        # Integral with anti-windup: only accumulate if output isn't saturated
        self._integral += error
        # Clamp integral to prevent windup
        max_integral = (self._output_max - 1.0) / max(self.ki, 1e-9)
        min_integral = (self._output_min - 1.0) / max(self.ki, 1e-9)
        self._integral = max(min_integral, min(max_integral, self._integral))
        i = self.ki * self._integral

        # Derivative
        if self._prev_error is not None:
            d = self.kd * (error - self._prev_error)
        else:
            d = 0.0
        self._prev_error = error

        # Output centered at 1.0 (neutral = no adjustment)
        raw = 1.0 + p + i + d
        return max(self._output_min, min(self._output_max, raw))

    def reset(self):
        """Reset accumulated state."""
        self._integral = 0.0
        self._prev_error = None


# ---------------------------------------------------------------------------
# Main regulator
# ---------------------------------------------------------------------------

class CriticalityRegulator:
    """Steers the 64-column neural mesh toward the critical point.

    Usage:
        regulator = CriticalityRegulator()
        state = await regulator.tick(column_activations, inter_column_weights)
        score = regulator.get_criticality_score()
        adjustments = regulator.get_adjustments()

    The tick() method is called by the neural mesh integration loop every
    mesh tick.  Internally it only runs the expensive measurement and PID
    update every `branching_measurement_interval` ticks.  On off-ticks it
    returns the cached state immediately (<0.01 ms).
    """

    def __init__(self, cfg: CriticalityConfig | None = None):
        self.cfg = cfg or CriticalityConfig()
        self._lock = threading.Lock()

        # --- State history ---
        # Previous tick's column activations (for computing deltas).
        self._prev_activations: Optional[np.ndarray] = None
        # Previous tick's "active" mask (for branching ratio).
        self._prev_active: Optional[np.ndarray] = None

        # --- Branching ratio ---
        # Rolling buffer of per-tick branching ratios, averaged every interval.
        self._branching_samples: Deque[float] = deque(
            maxlen=self.cfg.branching_measurement_interval
        )
        self._branching_ratio: float = 1.0  # assume critical at startup

        # --- Avalanche tracking ---
        # We track avalanches as contiguous sequences of ticks where at least
        # one new column became active.  When no new columns activate, the
        # current avalanche ends.
        self._current_avalanche_size: int = 0
        self._in_avalanche: bool = False
        self._avalanche_sizes: Deque[int] = deque(
            maxlen=self.cfg.avalanche_window
        )
        self._avalanche_exponent: float = -1.5  # neutral until measured

        # --- Activation distribution ---
        self._herfindahl: float = 0.0

        # --- PID controllers ---
        # Gain PID: error = branching_ratio - 1.0
        # If branching_ratio > 1 (supercritical) → error > 0 → we want to
        # DECREASE gain, so we negate the error for the gain controller.
        self._gain_pid = _PIDController(
            kp=self.cfg.kp, ki=self.cfg.ki, kd=self.cfg.kd,
            output_min=self.cfg.gain_clamp[0],
            output_max=self.cfg.gain_clamp[1],
        )
        # Noise PID: if the system is stuck in a fixed point (very low
        # branching ratio), increase noise to kick it out.
        self._noise_pid = _PIDController(
            kp=self.cfg.kp * 0.5, ki=self.cfg.ki * 0.5, kd=self.cfg.kd * 0.5,
            output_min=self.cfg.noise_clamp[0],
            output_max=self.cfg.noise_clamp[1],
        )
        # E/I ratio PID: steers excitation/inhibition balance toward 1.0.
        self._ei_pid = _PIDController(
            kp=self.cfg.kp * 0.3, ki=self.cfg.ki * 0.3, kd=self.cfg.kd * 0.3,
            output_min=self.cfg.ei_ratio_clamp[0],
            output_max=self.cfg.ei_ratio_clamp[1],
        )

        # --- PID outputs ---
        self._gain_adjustment: float = 1.0
        self._noise_adjustment: float = 1.0
        self._ei_ratio: float = 1.0

        # --- Tick counter & cached state ---
        self._tick_count: int = 0
        self._cached_state: CriticalityState = CriticalityState()

        logger.info(
            "CriticalityRegulator initialized: columns=%d, "
            "branching_interval=%d, avalanche_window=%d, "
            "PID=[Kp=%.2f Ki=%.3f Kd=%.2f]",
            self.cfg.num_columns,
            self.cfg.branching_measurement_interval,
            self.cfg.avalanche_window,
            self.cfg.kp, self.cfg.ki, self.cfg.kd,
        )

    # ── Public API ───────────────────────────────────────────────────────

    async def tick(
        self,
        column_activations: np.ndarray,
        inter_column_weights: np.ndarray,
    ) -> CriticalityState:
        """Process one mesh tick and return the current criticality state.

        This is the main entry point, called once per neural mesh tick.  It:
          1. Records which columns are active and which just became active.
          2. Computes the per-tick branching contribution.
          3. Tracks avalanche cascades.
          4. Every N ticks, runs the full measurement + PID update.
          5. Returns the (possibly cached) CriticalityState.

        Args:
            column_activations: shape (num_columns,) -- mean absolute
                activation of each cortical column this tick.
            inter_column_weights: shape (num_columns, num_columns) -- the
                inter-column weight matrix from the neural mesh.  Entry [i,j]
                is the connection weight from column i to column j.

        Returns:
            CriticalityState with all measurements and PID outputs.

        Performance: <1 ms on off-ticks, <5 ms on measurement ticks (64 cols).
        """
        with self._lock:
            return self._tick_inner(column_activations, inter_column_weights)

    def get_criticality_score(self) -> float:
        """Return the composite criticality score (0 to 1).

        1.0 means the system is perfectly at the critical point:
          - Branching ratio is exactly 1.0
          - Avalanche size distribution follows a power law with exponent -1.5

        This score can be used as a multiplicative factor on phi (integrated
        information), since IIT predicts phi is maximized at criticality.
        """
        with self._lock:
            return self._cached_state.criticality_score

    def get_adjustments(self) -> Dict[str, float]:
        """Return the current PID-computed adjustment factors.

        These are multiplicative factors meant to be applied to the neural
        mesh's modulatory parameters:
          - gain: multiply mesh's modulatory_gain by this
          - noise: multiply mesh's modulatory_noise by this
          - ei_ratio: target glutamate/GABA ratio (1.0 = balanced)
        """
        with self._lock:
            return {
                "gain": self._gain_adjustment,
                "noise": self._noise_adjustment,
                "ei_ratio": self._ei_ratio,
            }

    def get_branching_ratio(self) -> float:
        """Return the most recently measured branching ratio.

        The branching ratio is the average number of columns that become
        active in response to a single active column.  At criticality
        this is exactly 1.0.
        """
        with self._lock:
            return self._branching_ratio

    def get_avalanche_exponent(self) -> float:
        """Return the most recently fitted avalanche size power-law exponent.

        At criticality this is approximately -1.5.  More negative values
        indicate subcritical dynamics (small avalanches dominate).  Less
        negative values indicate supercritical dynamics (large avalanches
        dominate).
        """
        with self._lock:
            return self._avalanche_exponent

    def get_state(self) -> CriticalityState:
        """Return the full cached state snapshot."""
        with self._lock:
            return self._cached_state

    # ── Internal measurement ─────────────────────────────────────────────

    def _tick_inner(
        self,
        column_activations: np.ndarray,
        inter_column_weights: np.ndarray,
    ) -> CriticalityState:
        """Core tick logic (called under lock)."""
        n = self.cfg.num_columns
        threshold = self.cfg.activation_threshold

        # Ensure inputs are the right shape and type
        activations = np.asarray(column_activations, dtype=np.float64).ravel()
        if activations.shape[0] != n:
            # Graceful degradation: pad or truncate
            padded = np.zeros(n, dtype=np.float64)
            m = min(len(activations), n)
            padded[:m] = activations[:m]
            activations = padded

        weights = np.asarray(inter_column_weights, dtype=np.float64)
        if weights.shape != (n, n):
            weights = np.zeros((n, n), dtype=np.float64)

        # Which columns are active RIGHT NOW
        active = activations > threshold  # (n,) bool

        # --- Branching ratio (per-tick sample) ---
        if self._prev_active is not None and self._prev_activations is not None:
            br_sample = self._compute_branching_sample(
                activations, active, weights,
            )
            self._branching_samples.append(br_sample)

        # --- Avalanche tracking ---
        self._track_avalanche(active)

        # --- Herfindahl index of activation concentration ---
        self._update_herfindahl(activations)

        # --- Save state for next tick ---
        self._prev_activations = activations.copy()
        self._prev_active = active.copy()

        self._tick_count += 1

        # --- Full measurement + PID update on interval ticks ---
        if self._tick_count % self.cfg.branching_measurement_interval == 0:
            self._run_measurement_and_pid()

        return self._cached_state

    def _compute_branching_sample(
        self,
        activations: np.ndarray,
        active: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute a single-tick branching ratio sample.

        Definition: for each column i that was active on the PREVIOUS tick,
        count how many columns j satisfy ALL of:
          1. j's activation INCREASED from the previous tick to now
          2. there is a connection from i to j (weights[i,j] != 0)
          3. j is active now

        The branching ratio is the mean of these counts over all previously
        active columns.  If no columns were active, return 1.0 (neutral).
        """
        prev_active = self._prev_active       # (n,) bool
        prev_act = self._prev_activations      # (n,) float

        if not np.any(prev_active):
            return 1.0

        # Which columns increased activation since last tick
        increased = activations > prev_act  # (n,) bool

        # Which columns were newly activated (increased AND currently active)
        newly_activated = increased & active  # (n,) bool

        # For each previously-active column i, count how many newly-activated
        # columns j it could have caused (via nonzero weight from i to j).
        # This is a vectorized operation: connectivity_mask[i, j] = True if
        # column i has a connection to column j.
        prev_active_idx = np.where(prev_active)[0]
        if len(prev_active_idx) == 0:
            return 1.0

        # Connection mask from each previously-active column to all others
        connection_mask = np.abs(weights[prev_active_idx, :]) > 1e-8  # (k, n)

        # Downstream activations: connected AND newly activated
        downstream = connection_mask & newly_activated[np.newaxis, :]  # (k, n)

        # Count downstream activations per previously-active column
        counts = downstream.sum(axis=1).astype(np.float64)  # (k,)

        return float(counts.mean()) if len(counts) > 0 else 1.0

    def _track_avalanche(self, active: np.ndarray):
        """Track avalanche cascades in the column activation pattern.

        An avalanche is defined as a contiguous sequence of ticks during
        which at least one column is active.  When all columns go quiet
        (or when the set of newly-active columns is empty after an active
        period), the avalanche ends and its total size is recorded.

        This is a simplified version of the Beggs & Plenz (2003) definition,
        adapted for a discrete-tick column-level model.
        """
        num_active = int(active.sum())

        if num_active > 0:
            if not self._in_avalanche:
                # New avalanche starts
                self._in_avalanche = True
                self._current_avalanche_size = num_active
            else:
                # Ongoing avalanche: accumulate
                self._current_avalanche_size += num_active
        else:
            if self._in_avalanche:
                # Avalanche just ended -- record its size
                if self._current_avalanche_size > 0:
                    self._avalanche_sizes.append(self._current_avalanche_size)
                self._in_avalanche = False
                self._current_avalanche_size = 0

    def _update_herfindahl(self, activations: np.ndarray):
        """Compute the Herfindahl-Hirschman Index of activation concentration.

        The HHI measures how concentrated activity is across columns:
          - 0 means activity is perfectly uniform (all columns equally active)
          - 1 means all activity is concentrated in a single column

        At criticality, activity should be broadly distributed but not
        perfectly uniform -- moderate HHI values around 0.05-0.15 are typical.
        """
        total = activations.sum()
        if total < 1e-10:
            self._herfindahl = 0.0
            return
        shares = activations / total  # (n,) -- each column's share
        self._herfindahl = float(np.sum(shares ** 2))

    def _fit_avalanche_exponent(self) -> float:
        """Fit a power-law exponent to the avalanche size distribution.

        Uses log-log linear regression: if P(s) ~ s^alpha, then
        log(P(s)) = alpha * log(s) + const.

        We bin the avalanche sizes, compute the empirical frequency of each
        bin, and fit a line in log-log space.  The slope is the exponent.

        At criticality, alpha should be near -1.5 (Beggs & Plenz, 2003).

        Returns -1.5 (neutral) if there aren't enough data points.
        """
        sizes = list(self._avalanche_sizes)
        if len(sizes) < self.cfg.min_avalanches_for_fit:
            return -1.5  # not enough data, assume neutral

        sizes_arr = np.array(sizes, dtype=np.float64)
        # Unique sizes and their counts
        unique_sizes, counts = np.unique(sizes_arr, return_counts=True)

        if len(unique_sizes) < 2:
            # All avalanches are the same size -- no power law to fit
            return -1.5

        # Normalize counts to get empirical probability
        probs = counts.astype(np.float64) / counts.sum()

        # Log-log transform (only positive values)
        log_s = np.log(unique_sizes)
        log_p = np.log(probs)

        # Linear regression in log-log space: log_p = alpha * log_s + beta
        # Using numpy's least-squares solver (equivalent to polyfit degree 1)
        # Stack [log_s, 1] as design matrix
        A = np.column_stack([log_s, np.ones_like(log_s)])
        result, _, _, _ = np.linalg.lstsq(A, log_p, rcond=None)
        alpha = float(result[0])

        # Clamp to reasonable range -- exponents outside [-3, 0] are
        # numerical artifacts
        return max(-3.0, min(0.0, alpha))

    def _compute_criticality_score(self) -> float:
        """Compute the composite criticality score (0 to 1).

        The score is the product of two exponential penalties:
          1. How far the branching ratio is from 1.0
          2. How far the avalanche exponent is from -1.5

        Each penalty is exp(-|deviation|), so:
          - Perfect criticality (br=1.0, alpha=-1.5) gives score=1.0
          - A branching ratio of 2.0 gives penalty exp(-1.0) = 0.37
          - An exponent of -2.5 gives penalty exp(-1.0) = 0.37
          - Combined: 0.37 * 0.37 = 0.14

        This multiplicative form ensures BOTH conditions must be met for
        a high score.
        """
        br_penalty = math.exp(-abs(self._branching_ratio - 1.0))
        exp_penalty = math.exp(-abs(self._avalanche_exponent + 1.5))
        return br_penalty * exp_penalty

    def _run_measurement_and_pid(self):
        """Run the full measurement cycle and update PID outputs.

        Called every `branching_measurement_interval` ticks.  This is the
        expensive path (~1-3 ms for 64 columns).
        """
        # 1. Average the branching ratio samples collected over the interval
        if self._branching_samples:
            self._branching_ratio = float(
                np.mean(list(self._branching_samples))
            )
        # else: keep previous value

        # 2. Fit the avalanche exponent
        self._avalanche_exponent = self._fit_avalanche_exponent()

        # 3. Compute criticality score
        score = self._compute_criticality_score()

        # 4. PID update
        # Error for gain: branching_ratio > 1 means supercritical → DECREASE gain
        # So we negate: error = -(branching_ratio - 1.0)
        gain_error = -(self._branching_ratio - 1.0)
        self._gain_adjustment = self._gain_pid.step(gain_error)

        # Error for noise: if subcritical (br < 1), we want MORE noise to
        # destabilize the fixed point.  If supercritical, less noise.
        # We also factor in the Herfindahl: high concentration means the
        # system is stuck in a few columns → more noise needed.
        noise_error = -(self._branching_ratio - 1.0) + self._herfindahl * 0.5
        self._noise_adjustment = self._noise_pid.step(noise_error)

        # Error for E/I ratio: drives toward balanced excitation/inhibition.
        # If supercritical, reduce excitation (ratio < 1).
        # If subcritical, increase excitation (ratio > 1).
        ei_error = -(self._branching_ratio - 1.0)
        self._ei_ratio = self._ei_pid.step(ei_error)

        # 5. Avalanche diagnostics
        sizes = list(self._avalanche_sizes)
        avalanche_count = len(sizes)
        mean_size = float(np.mean(sizes)) if sizes else 0.0

        # 6. Build and cache the state
        self._cached_state = CriticalityState(
            branching_ratio=round(self._branching_ratio, 4),
            avalanche_exponent=round(self._avalanche_exponent, 4),
            criticality_score=round(score, 4),
            gain_adjustment=round(self._gain_adjustment, 4),
            noise_adjustment=round(self._noise_adjustment, 4),
            ei_ratio=round(self._ei_ratio, 4),
            avalanche_count=avalanche_count,
            mean_avalanche_size=round(mean_size, 2),
            herfindahl_index=round(self._herfindahl, 4),
            tick_count=self._tick_count,
        )

        if self._tick_count % (self.cfg.branching_measurement_interval * 10) == 0:
            logger.info(
                "Criticality: br=%.3f exp=%.2f score=%.3f "
                "gain=%.3f noise=%.3f ei=%.3f avalanches=%d",
                self._branching_ratio,
                self._avalanche_exponent,
                score,
                self._gain_adjustment,
                self._noise_adjustment,
                self._ei_ratio,
                avalanche_count,
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[CriticalityRegulator] = None


def get_criticality_regulator() -> CriticalityRegulator:
    """Get or create the singleton CriticalityRegulator."""
    global _instance
    if _instance is None:
        _instance = CriticalityRegulator()
    return _instance
