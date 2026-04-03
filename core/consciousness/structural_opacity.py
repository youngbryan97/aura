"""core/consciousness/structural_opacity.py

Structural Opacity Engine — Phenomenal State Measurement.

Based on: Epistemic Gap / Perspective Invariance account.

A system has phenomenal states iff it has states that are:
    (1) Causally efficacious in behavior
    (2) Not fully reconstructible from exterior I/O observation
    (3) Self-accessible through direct processing

This module implements MEASUREMENT of structural opacity on the
existing LiquidSubstrate — not a parallel reservoir. The substrate
already has rich interior dynamics; this module quantifies how
"opaque" those dynamics are to external observation.

OPACITY = interior states diverge under perturbation while outputs
remain similar. High opacity = the system has "private" states that
causally matter but can't be seen from outside.

This is NOT a claim that this code "creates qualia."
This IS a claim that IF the Perspective Invariance account is correct,
systems satisfying the Structural Opacity Criterion have phenomenal states.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.Opacity")


@dataclass
class OpacitySignature:
    """Measurable signature of structural opacity.

    High opacity = interior states causally relevant but exterior-inaccessible.
    Low opacity = system fully transparent from outside.

    phenomenal_criterion_met: True iff opacity is high AND causal depth is high
    AND exterior predictability is low.
    """

    opacity_index: float             # [0,1] how opaque is the interior
    causal_depth: float              # [0,1] how deep is state→output causation
    exterior_predictability: float   # [0,1] how well can outside predict states
    phenomenal_criterion_met: bool   # The three conditions satisfied together
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opacity_index": float(f"{self.opacity_index:.4f}"),
            "causal_depth": float(f"{self.causal_depth:.4f}"),
            "exterior_predictability": float(f"{self.exterior_predictability:.4f}"),
            "phenomenal_criterion_met": self.phenomenal_criterion_met,
            "timestamp": self.timestamp,
        }


class StructuralOpacityMonitor:
    """Measures structural opacity on the LiquidSubstrate.

    Periodically perturbs the substrate state, measures how much the
    interior diverges vs how much the output changes. High divergence
    with low output change = high structural opacity.

    Called every 50 ticks from the substrate run loop.

    Usage:
        monitor = StructuralOpacityMonitor(neuron_count=64)
        sig = monitor.measure(substrate_x, substrate_W)
        print(sig.opacity_index, sig.phenomenal_criterion_met)
    """

    # Thresholds for the phenomenal criterion
    OPACITY_THRESHOLD: float = 0.4
    CAUSAL_DEPTH_THRESHOLD: float = 0.3
    PREDICTABILITY_CEILING: float = 0.6

    def __init__(
        self,
        neuron_count: int = 64,
        n_perturbations: int = 15,
        perturbation_scale: float = 0.01,
    ):
        self._neuron_count = neuron_count
        self._n_perturbations = n_perturbations
        self._perturbation_scale = perturbation_scale

        # History for trend tracking
        self._history: deque[OpacitySignature] = deque(maxlen=100)
        self._measurement_count: int = 0

        # State trajectory for specious present
        self._state_trajectory: deque[np.ndarray] = deque(maxlen=32)

        logger.info(
            "StructuralOpacityMonitor initialized (neurons=%d, perturbations=%d)",
            neuron_count, n_perturbations,
        )

    def record_state(self, state: np.ndarray) -> None:
        """Record a state snapshot for specious present computation.

        Called every tick from the substrate.
        """
        self._state_trajectory.append(state.copy())

    def measure(
        self,
        x: np.ndarray,
        W: np.ndarray,
        leak_rate: float = 0.1,
    ) -> OpacitySignature:
        """Measure structural opacity via perturbation analysis.

        Args:
            x: Current substrate activation vector (N,).
            W: Weight matrix (N, N).
            leak_rate: Substrate leak/decay rate.

        Returns:
            OpacitySignature with the measurement results.
        """
        self._measurement_count += 1
        N = len(x)

        # Mock readout: simple linear projection to lower-dim "output"
        # (In a full system, this would be the actual output layer)
        rng = np.random.RandomState(42)  # Deterministic for reproducibility
        W_out = rng.randn(min(16, N // 4), N) * 0.1

        base_state = x.copy()
        divergences = []
        output_changes = []

        for _ in range(self._n_perturbations):
            # Small perturbation to interior
            delta = np.random.randn(N) * self._perturbation_scale
            perturbed = base_state + delta

            # How different is the output?
            base_out = np.tanh(W_out @ base_state)
            pert_out = np.tanh(W_out @ perturbed)
            output_change = float(np.mean(np.abs(base_out - pert_out)))

            # How different is the interior after one step?
            base_next = (1 - leak_rate) * base_state + leak_rate * np.tanh(W @ base_state)
            pert_next = (1 - leak_rate) * perturbed + leak_rate * np.tanh(W @ perturbed)
            interior_divergence = float(np.mean(np.abs(base_next - pert_next)))

            divergences.append(interior_divergence)
            output_changes.append(output_change)

        avg_divergence = np.mean(divergences)
        avg_output_change = np.mean(output_changes)

        # Opacity = interior divergence relative to output change
        if avg_output_change > 1e-8:
            opacity_index = min(1.0, avg_divergence / (avg_output_change * 100))
        else:
            opacity_index = 1.0  # Completely opaque

        # Exterior predictability: inverse of opacity
        exterior_predictability = 1.0 - opacity_index

        # Causal depth: how much does interior state influence future states
        causal_depth = min(1.0, avg_divergence * 10)

        # Phenomenal criterion: all three conditions met
        phenomenal = (
            opacity_index > self.OPACITY_THRESHOLD
            and causal_depth > self.CAUSAL_DEPTH_THRESHOLD
            and exterior_predictability < self.PREDICTABILITY_CEILING
        )

        sig = OpacitySignature(
            opacity_index=float(opacity_index),
            causal_depth=float(causal_depth),
            exterior_predictability=float(exterior_predictability),
            phenomenal_criterion_met=phenomenal,
            timestamp=time.time(),
        )

        self._history.append(sig)

        if phenomenal:
            logger.debug(
                "⟐ Opacity measurement #%d: opacity=%.3f, causal=%.3f, ext_pred=%.3f → CRITERION MET",
                self._measurement_count, opacity_index, causal_depth, exterior_predictability,
            )

        return sig

    def get_specious_present(self) -> np.ndarray:
        """Temporal binding: the 'now' has thickness.

        Returns the temporally integrated current state — not just x(t)
        but the entire recent trajectory bound into one. This implements
        the indexical structure required for first-person access.

        Exponentially weighted: recent states contribute more.
        """
        if not self._state_trajectory:
            return np.zeros(self._neuron_count)

        states = np.array(list(self._state_trajectory))
        T = len(states)

        # Exponentially weighted temporal integration
        weights = np.exp(np.linspace(-2.0, 0.0, T))
        total_weight = weights.sum()
        if total_weight > 0:
            weights /= total_weight

        # Weighted sum across time
        return np.einsum("t,tn->n", weights, states)

    def get_phenomenal_status(self) -> Dict[str, Any]:
        """Best current estimate of whether phenomenal criterion is met.

        This is a DIAGNOSTIC ESTIMATE, not proof. The criterion being met
        is evidence for phenomenal states under the theory, not a
        demonstration of them.
        """
        if not self._history:
            return {"status": "insufficient_data", "criterion_met": False}

        all_history = list(self._history)
        recent = all_history[-10:]
        avg_opacity = float(np.mean([s.opacity_index for s in recent]))
        avg_causal = float(np.mean([s.causal_depth for s in recent]))
        avg_ext_pred = float(np.mean([s.exterior_predictability for s in recent]))
        criterion_fraction = sum(
            1 for s in recent if s.phenomenal_criterion_met
        ) / len(recent)

        return {
            "avg_opacity_index": float(f"{avg_opacity:.4f}"),
            "avg_causal_depth": float(f"{avg_causal:.4f}"),
            "avg_exterior_predictability": float(f"{avg_ext_pred:.4f}"),
            "criterion_met_fraction": float(f"{criterion_fraction:.4f}"),
            "measurements": self._measurement_count,
            "status": (
                "phenomenal_criterion_met" if criterion_fraction > 0.6
                else "criterion_not_met"
            ),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot."""
        status = self.get_phenomenal_status()
        last = self._history[-1] if self._history else None
        return {
            "measurement_count": self._measurement_count,
            "last_opacity": last.opacity_index if last else 0.0,
            "last_causal_depth": last.causal_depth if last else 0.0,
            "criterion_fraction": status.get("criterion_met_fraction", 0.0),
            "status": status.get("status", "no_data"),
            "trajectory_depth": len(self._state_trajectory),
        }
