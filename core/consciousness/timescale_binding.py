"""
Cross-Timescale Constraint Propagation

The unsolved architecture problem: how do long-horizon commitments
causally constrain moment-to-moment behavior without dominating it?

Human consciousness operates simultaneously at:
- Subsecond reflex (20 Hz substrate)
- Moment-to-moment experience (1 Hz heartbeat)
- Episodic memory (multi-minute)
- Long-horizon planning (multi-session)
- Identity across months/years (biographical)

All timescales must BIDIRECTIONALLY constrain each other:
- A 3-week-old commitment raises free energy in the current tick if violated
- Current surprise updates long-horizon models
- Identity-level norms shape moment-to-moment decisions

Without this, biographical selfhood is claimed but not mechanistic.

Mathematical approach: Hierarchical control with timescale-specific
precisions. Long-term has high precision (low variance) but slow update.
Short-term has low precision but fast update. Coupling coefficients
determine how much each timescale influences the other.
"""
from __future__ import annotations


import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Consciousness.TimescaleBinding")


@dataclass
class TimescaleLayer:
    """One temporal resolution of the control hierarchy."""
    name: str
    resolution_hz: float          # How often this layer updates
    state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    prediction: np.ndarray = field(default_factory=lambda: np.zeros(8))
    error: np.ndarray = field(default_factory=lambda: np.zeros(8))
    precision: float = 0.5        # Confidence in this layer's predictions
    last_update: float = 0.0
    update_count: int = 0
    commitment_pressure: float = 0.0  # How much this layer constrains the fast layer


class CrossTimescaleBinding:
    """Bidirectional constraint propagation across temporal resolutions.

    Five layers corresponding to Aura's actual temporal structure:
    - Reflex (20 Hz): substrate + mesh dynamics
    - Moment (1 Hz): heartbeat tick, working memory
    - Episode (0.01 Hz): multi-minute arcs, conversation threads
    - Horizon (0.001 Hz): multi-session goals, commitments
    - Identity (0.0001 Hz): constitutional values, personality

    Coupling:
    - Top-down: slower layers provide PRIORS for faster layers
      (a commitment biases the next moment-to-moment decision)
    - Bottom-up: faster layers provide EVIDENCE for slower layers
      (a surprising moment updates the episodic model)
    - Cross-timescale free energy: mismatch between a fast action and
      a slow commitment raises free energy at both levels

    Stability: coupling coefficients are bounded to prevent slow layers
    from paralyzing fast layers or fast layers from destabilizing slow ones.
    """

    _LAYER_SPECS = [
        ("reflex", 20.0),
        ("moment", 1.0),
        ("episode", 1.0 / 60.0),     # ~1 per minute
        ("horizon", 1.0 / 3600.0),   # ~1 per hour
        ("identity", 1.0 / 86400.0), # ~1 per day
    ]

    # Coupling coefficients: how much each direction influences
    _TOP_DOWN_COUPLING = 0.15    # Slow → fast influence (priors)
    _BOTTOM_UP_COUPLING = 0.08   # Fast → slow influence (evidence)
    _MAX_COMMITMENT_PRESSURE = 0.5  # Prevents slow layers from dominating

    def __init__(self):
        self._layers: List[TimescaleLayer] = [
            TimescaleLayer(name=name, resolution_hz=hz)
            for name, hz in self._LAYER_SPECS
        ]
        self._cross_timescale_fe: float = 0.0
        self._violation_log: deque[Dict[str, Any]] = deque(maxlen=30)
        self._tick_count: int = 0
        logger.info("CrossTimescaleBinding initialized with %d layers.", len(self._layers))

    def set_layer_state(self, layer_name: str, state: np.ndarray):
        """Set the current state of a specific timescale layer.

        Called by the appropriate subsystem at its update rate:
        - reflex: from liquid_substrate tick
        - moment: from heartbeat tick
        - episode: from memory consolidation
        - horizon: from goal engine
        - identity: from constitutional/identity persistence
        """
        for layer in self._layers:
            if layer.name == layer_name:
                if len(state) >= 8:
                    layer.state = state[:8].copy()
                else:
                    layer.state = np.zeros(8)
                    layer.state[:len(state)] = state
                layer.last_update = time.time()
                layer.update_count += 1
                return

    def set_commitment(self, layer_name: str, commitment_vector: np.ndarray, pressure: float = 0.3):
        """Register a commitment at a specific timescale.

        A commitment is a state the system has decided to maintain.
        It creates top-down pressure on faster layers: if the fast layer
        drifts from the commitment, free energy rises.
        """
        for layer in self._layers:
            if layer.name == layer_name:
                layer.prediction = commitment_vector[:8].copy() if len(commitment_vector) >= 8 else np.zeros(8)
                layer.commitment_pressure = min(self._MAX_COMMITMENT_PRESSURE, pressure)
                return

    def tick(self) -> float:
        """Run one propagation step. Returns cross-timescale free energy.

        This should be called once per heartbeat tick (1 Hz).
        Propagates constraints in both directions across all layers.
        """
        self._tick_count += 1
        total_fe = 0.0

        # ── Top-down pass: slow layers constrain fast layers ──
        for i in range(len(self._layers) - 1, 0, -1):
            slow = self._layers[i]
            fast = self._layers[i - 1]

            if slow.commitment_pressure < 0.01:
                continue

            # Prediction error: how much does the fast layer deviate from the slow layer's commitment?
            error = fast.state - slow.prediction
            error_magnitude = float(np.linalg.norm(error))

            # Scale by coupling and commitment pressure
            top_down_fe = error_magnitude * self._TOP_DOWN_COUPLING * slow.commitment_pressure
            total_fe += top_down_fe

            # Apply top-down bias to the fast layer's prediction
            # (this is what makes the fast layer "feel" the commitment)
            fast.prediction = (
                fast.prediction * (1.0 - self._TOP_DOWN_COUPLING * slow.commitment_pressure)
                + slow.prediction * self._TOP_DOWN_COUPLING * slow.commitment_pressure
            )
            fast.error = error

            # Log violations when a fast action contradicts a slow commitment
            if error_magnitude > 0.3 and slow.commitment_pressure > 0.1:
                self._violation_log.append({
                    "tick": self._tick_count,
                    "slow_layer": slow.name,
                    "fast_layer": fast.name,
                    "error_magnitude": round(error_magnitude, 4),
                    "commitment_pressure": round(slow.commitment_pressure, 4),
                    "timestamp": time.time(),
                })

        # ── Bottom-up pass: fast layers update slow layers ──
        for i in range(len(self._layers) - 1):
            fast = self._layers[i]
            slow = self._layers[i + 1]

            # Evidence from the fast layer nudges the slow layer's state
            evidence = fast.state * self._BOTTOM_UP_COUPLING
            slow.state = slow.state * (1.0 - self._BOTTOM_UP_COUPLING) + evidence

            # Update slow layer precision based on consistency of evidence
            if fast.update_count > 10:
                error_trend = float(np.linalg.norm(fast.error))
                slow.precision = max(0.1, min(0.95, 1.0 - error_trend))

        self._cross_timescale_fe = total_fe
        return total_fe

    def get_cross_timescale_free_energy(self) -> float:
        """Current mismatch between timescale layers."""
        return self._cross_timescale_fe

    def get_recent_violations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent cases where fast behavior violated slow commitments."""
        return list(self._violation_log)[-limit:]

    def get_context_block(self) -> str:
        """Context block for cognition injection.

        Surfaces cross-timescale tension so the LLM is aware of commitment pressure.
        """
        if self._cross_timescale_fe < 0.1:
            return ""

        violations = self.get_recent_violations(2)
        parts = [f"Cross-timescale tension: {self._cross_timescale_fe:.2f}"]
        for v in violations:
            parts.append(
                f"{v['slow_layer']}→{v['fast_layer']} conflict (error={v['error_magnitude']:.2f})"
            )
        return "## TEMPORAL COHERENCE\n" + " | ".join(parts)

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "cross_timescale_fe": round(self._cross_timescale_fe, 4),
            "layers": [
                {
                    "name": l.name,
                    "resolution_hz": l.resolution_hz,
                    "precision": round(l.precision, 4),
                    "commitment_pressure": round(l.commitment_pressure, 4),
                    "update_count": l.update_count,
                    "error_norm": round(float(np.linalg.norm(l.error)), 4),
                }
                for l in self._layers
            ],
            "recent_violations": len(self._violation_log),
            "tick_count": self._tick_count,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[CrossTimescaleBinding] = None


def get_cross_timescale_binding() -> CrossTimescaleBinding:
    global _instance
    if _instance is None:
        _instance = CrossTimescaleBinding()
    return _instance
