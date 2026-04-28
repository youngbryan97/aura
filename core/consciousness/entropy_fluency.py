"""core/consciousness/entropy_fluency.py
─────────────────────────────────────────
Entropy-Fluency tracker for the substrate.

Background
----------
Beyköylü, Vervaeke, & Meling (2025), "From flow to mystical experiences:
Connecting entropy and fluency along the unifying framework of cognitive
continuum." Their Entropy-Fluency Hypothesis: in a person-world system,
increased entropy signals destabilization; that destabilization, when
followed by reorganization, manifests as increased fluency (the felt
ease characteristic of flow / insight / mystical states).

Why this fits Aura
------------------
The LiquidSubstrate already produces a 64-D state vector at 20 Hz. We
have phi (integration), neurochemicals, and oscillatory binding — but
no metric that captures *destabilization vs reorganization* as a phase.
That is exactly what entropy-fluency provides: an observable that
distinguishes "the system is in transition / probing the affordance
space" from "the system has settled into a smooth attractor."

What this module does
---------------------
1. **Entropy:** Shannon entropy of the binarized substrate state (each
   neuron above/below its running median is one bit; H = -Σ p log p over
   the resulting bit-pattern frequency in a sliding window). The window
   is short enough to capture transitions, long enough to be stable.
2. **Fluency:** smoothed inverse of substrate-state innovation magnitude.
   When the state vector's tick-to-tick delta is small and stable,
   fluency is high; when the system is thrashing, fluency drops.
3. **Phase classifier:** four phases — STABLE (low entropy, high
   fluency), DESTABILIZING (rising entropy, dropping fluency),
   REORGANIZING (high entropy, fluency starting to recover), and
   FLUENT (entropy normalized, fluency high — post-reorganization,
   typically the flow / insight phase).

What downstream code can do with it
------------------------------------
- The autonomy pipeline can use REORGANIZING as a signal that the system
  is actively integrating new information — a good window to commit
  research findings rather than during STABLE.
- The substrate authority can permit identity-adjacent operations during
  FLUENT phases (post-reorganization, integration is high) and resist
  during DESTABILIZING.
- The reflection loop can correlate REORGANIZING ↔ insight reports.

Pure numpy. CPU-only. Designed to run in the substrate's existing tick.
No model loads.
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.EntropyFluency")

# ── Tunables ──────────────────────────────────────────────────────────────

HISTORY_WINDOW = 64        # ticks of state history kept for entropy estimation
INNOVATION_WINDOW = 16     # ticks for fluency innovation smoothing
MIN_HISTORY_FOR_ENTROPY = 8
EWMA_ALPHA = 0.15          # entropy / fluency smoothing factor

# Phase thresholds, normalized to [0, 1]
ENTROPY_HIGH = 0.65
ENTROPY_LOW = 0.35
FLUENCY_HIGH = 0.65
FLUENCY_LOW = 0.40

PHASE_STABLE = "stable"
PHASE_DESTABILIZING = "destabilizing"
PHASE_REORGANIZING = "reorganizing"
PHASE_FLUENT = "fluent"

ALL_PHASES = (PHASE_STABLE, PHASE_DESTABILIZING, PHASE_REORGANIZING, PHASE_FLUENT)


@dataclass
class EntropyFluencyReport:
    entropy: float                 # smoothed [0, 1]
    fluency: float                 # smoothed [0, 1]
    raw_entropy: float             # unsmoothed Shannon entropy of last window
    innovation_magnitude: float    # raw L2 norm of latest delta
    phase: str                     # one of ALL_PHASES
    phase_dwell_ticks: int         # how many ticks we've been in this phase
    transition_just_happened: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entropy": round(float(self.entropy), 4),
            "fluency": round(float(self.fluency), 4),
            "raw_entropy": round(float(self.raw_entropy), 4),
            "innovation_magnitude": round(float(self.innovation_magnitude), 4),
            "phase": self.phase,
            "phase_dwell_ticks": int(self.phase_dwell_ticks),
            "transition_just_happened": bool(self.transition_just_happened),
        }


class EntropyFluencyTracker:
    """Stateful tracker. Call ``update(state_vector)`` once per substrate
    tick; read ``last_report`` whenever downstream code needs the metric.
    """

    def __init__(
        self,
        history_window: int = HISTORY_WINDOW,
        innovation_window: int = INNOVATION_WINDOW,
        ewma_alpha: float = EWMA_ALPHA,
    ) -> None:
        self._history_window = max(MIN_HISTORY_FOR_ENTROPY, int(history_window))
        self._innovation_window = max(2, int(innovation_window))
        self._alpha = float(ewma_alpha)

        self._states: Deque[np.ndarray] = deque(maxlen=self._history_window)
        self._innovations: Deque[float] = deque(maxlen=self._innovation_window)

        self._smoothed_entropy: float = 0.0
        self._smoothed_fluency: float = 1.0
        self._last_state: Optional[np.ndarray] = None
        self._phase: str = PHASE_STABLE
        self._phase_dwell: int = 0
        self._tick_count: int = 0

    # ── Tick interface ────────────────────────────────────────────────────

    def update(self, state_vector: np.ndarray) -> EntropyFluencyReport:
        s = np.asarray(state_vector, dtype=np.float32).ravel()
        self._states.append(s.copy())
        if self._last_state is not None and self._last_state.shape == s.shape:
            innovation = float(np.linalg.norm(s - self._last_state))
        else:
            innovation = 0.0
        self._innovations.append(innovation)
        self._last_state = s
        self._tick_count += 1

        raw_entropy = self._compute_normalized_entropy()
        raw_fluency = self._compute_fluency()

        # EWMA smoothing
        self._smoothed_entropy = (
            self._alpha * raw_entropy + (1.0 - self._alpha) * self._smoothed_entropy
        )
        self._smoothed_fluency = (
            self._alpha * raw_fluency + (1.0 - self._alpha) * self._smoothed_fluency
        )

        new_phase = self._classify_phase(self._smoothed_entropy, self._smoothed_fluency)
        transitioned = (new_phase != self._phase)
        if transitioned:
            logger.debug(
                "EntropyFluency: %s → %s (e=%.2f f=%.2f)",
                self._phase, new_phase, self._smoothed_entropy, self._smoothed_fluency,
            )
            self._phase = new_phase
            self._phase_dwell = 1
        else:
            self._phase_dwell += 1

        return EntropyFluencyReport(
            entropy=self._smoothed_entropy,
            fluency=self._smoothed_fluency,
            raw_entropy=raw_entropy,
            innovation_magnitude=innovation,
            phase=self._phase,
            phase_dwell_ticks=self._phase_dwell,
            transition_just_happened=transitioned,
        )

    @property
    def last_report(self) -> Optional[EntropyFluencyReport]:
        if self._tick_count == 0:
            return None
        return EntropyFluencyReport(
            entropy=self._smoothed_entropy,
            fluency=self._smoothed_fluency,
            raw_entropy=self._compute_normalized_entropy(),
            innovation_magnitude=(self._innovations[-1] if self._innovations else 0.0),
            phase=self._phase,
            phase_dwell_ticks=self._phase_dwell,
            transition_just_happened=False,
        )

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def reset(self) -> None:
        self._states.clear()
        self._innovations.clear()
        self._last_state = None
        self._smoothed_entropy = 0.0
        self._smoothed_fluency = 1.0
        self._phase = PHASE_STABLE
        self._phase_dwell = 0
        self._tick_count = 0

    # ── Metric helpers ────────────────────────────────────────────────────

    def _compute_normalized_entropy(self) -> float:
        """Normalized variance entropy of substrate state across the window.

        Why not Shannon-on-bit-patterns: with 64-D state vectors and
        ~64-tick windows, every sample's exact bit-pattern is almost
        always unique (2^64 possible patterns vs ~64 samples), so a naive
        Shannon-on-patterns saturates near 1.0 and carries no signal.

        What we actually want is "how much is the state moving in its
        own ambient space?" The metric below: per-neuron variance over
        the window, normalized by ambient magnitude (so it's scale-
        invariant), then averaged and squashed to [0, 1] via tanh on a
        characteristic noise scale. Stable attractors → low; chaotic
        thrashing → high. This is the same thing entropy is doing
        conceptually for continuous variables (differential entropy is
        a function of variance for Gaussians), without the headaches of
        a degenerate bit-pattern count.
        """
        if len(self._states) < MIN_HISTORY_FOR_ENTROPY:
            return 0.0
        try:
            X = np.stack(list(self._states), axis=0)  # (T, N)
            T, N = X.shape
            if N == 0 or T < 2:
                return 0.0
            # Per-neuron std and absolute mean
            std_per = X.std(axis=0)                       # (N,)
            abs_mean_per = np.abs(X).mean(axis=0) + 1e-6  # (N,) avoid div-by-0
            relative_std = std_per / abs_mean_per         # (N,) scale-invariant
            # tanh-squash; characteristic scale 1.0 → tanh(1)≈0.76
            squashed = np.tanh(np.asarray(relative_std, dtype=np.float64))
            return float(max(0.0, min(1.0, squashed.mean())))
        except Exception as e:
            record_degradation('entropy_fluency', e)
            logger.debug("entropy compute fallback (zero): %s", e)
            return 0.0

    def _compute_fluency(self) -> float:
        """Smoothed inverse of innovation magnitude, normalized to [0, 1].
        High when the substrate's tick-to-tick delta is small and consistent;
        low when the system is making large, irregular jumps.
        """
        if not self._innovations:
            return 1.0
        innovations = list(self._innovations)
        mean_innov = float(np.mean(innovations))
        std_innov = float(np.std(innovations))
        # We treat fluency as bounded ease: small mean magnitude AND low variance.
        # Normalize via tanh on a characteristic magnitude scale.
        # The substrate's typical innovation should be O(1) or smaller; tanh(x)
        # saturates around 1 by x=2, so we scale.
        ease = 1.0 - math.tanh(mean_innov * 0.5 + std_innov * 0.5)
        return max(0.0, min(1.0, ease))

    def _classify_phase(self, entropy: float, fluency: float) -> str:
        """Map the (entropy, fluency) joint state to a phase. The classifier
        is intentionally simple — the smoothing keeps it from chattering.
        """
        if entropy >= ENTROPY_HIGH and fluency < FLUENCY_LOW:
            return PHASE_DESTABILIZING
        if entropy >= ENTROPY_HIGH and fluency >= FLUENCY_LOW:
            return PHASE_REORGANIZING
        if entropy < ENTROPY_HIGH and fluency >= FLUENCY_HIGH:
            return PHASE_FLUENT if entropy <= ENTROPY_LOW + 0.15 else PHASE_REORGANIZING
        return PHASE_STABLE


# ── Singleton accessor ────────────────────────────────────────────────────

_singleton: Optional[EntropyFluencyTracker] = None


def get_entropy_fluency_tracker() -> EntropyFluencyTracker:
    global _singleton
    if _singleton is None:
        _singleton = EntropyFluencyTracker()
    return _singleton


def reset_entropy_fluency_tracker() -> None:
    """Test/diagnostic helper. Resets the singleton."""
    global _singleton
    _singleton = None
