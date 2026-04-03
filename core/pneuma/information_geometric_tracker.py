"""core/pneuma/information_geometric_tracker.py
PNEUMA Layer 3 — Information Geometric Tracker (IGT).

Tracks belief drift on the statistical manifold using Fisher-Rao distance
(Hellinger arc distance on the probability simplex).

Fisher-Rao distance between two categorical distributions p and q:
    d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))

This is equivalent to the geodesic distance on the probability simplex
under the Fisher information metric.

Used to:
  - Detect when belief has drifted significantly (trigger re-grounding)
  - Score belief stability for the FreeEnergyOracle
  - Provide a precision signal for the PrecisionEngine
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger("PNEUMA.IGTracker")

_EPS = 1e-8


def hellinger_arc_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher-Rao / Hellinger arc distance between two distributions.

    Both p, q are normalized to sum-to-1 before computation.

    Returns value in [0, π/2].
    """
    # Normalize to valid probability vectors
    p = np.abs(p).astype(np.float64)
    q = np.abs(q).astype(np.float64)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum < _EPS or q_sum < _EPS:
        return math.pi / 2  # Maximum distance if degenerate
    p = p / p_sum
    q = q / q_sum

    # Bhattacharyya coefficient: sum_i sqrt(p_i * q_i)
    bc = float(np.sum(np.sqrt(p * q)))
    bc = max(-1.0, min(1.0, bc))
    return math.acos(bc)  # ∈ [0, π/2]


@dataclass
class ManifoldPoint:
    distribution: np.ndarray   # probability simplex point
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    fr_distance_from_prev: float = 0.0


class InformationGeometricTracker:
    """Tracks belief drift on the statistical manifold.

    Maintains a sliding window of distribution snapshots and computes:
    - Instantaneous Fisher-Rao distance from previous snapshot
    - Cumulative geodesic path length (total drift)
    - Drift rate (distance per second)
    - Anomaly flag when drift exceeds threshold
    """

    DRIFT_THRESHOLD = 0.8   # radians — above this, belief has shifted significantly
    WINDOW_SIZE = 100

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._window: Deque[ManifoldPoint] = deque(maxlen=self.WINDOW_SIZE)
        self._cumulative_distance: float = 0.0
        self._last_anomaly_at: float = 0.0
        self._drift_rate: float = 0.0
        logger.info("IGTracker online (dim=%d)", dim)

    def update(self, belief_vector: np.ndarray, source: str = "ode") -> float:
        """Register a new belief state and return Fisher-Rao distance from previous.

        belief_vector is treated as an unnormalized distribution.
        """
        dist = np.abs(belief_vector[:self.dim]).astype(np.float64)

        fr_dist = 0.0
        if self._window:
            prev_dist = self._window[-1].distribution
            fr_dist = hellinger_arc_distance(dist, prev_dist)
            self._cumulative_distance += fr_dist

            # Update drift rate
            dt = time.time() - self._window[-1].timestamp
            if dt > 0:
                self._drift_rate = fr_dist / dt

            if fr_dist > self.DRIFT_THRESHOLD:
                logger.info(
                    "IGTracker: Belief drift detected (FR=%.4f) from %s", fr_dist, source
                )
                self._last_anomaly_at = time.time()

        point = ManifoldPoint(
            distribution=dist.copy(),
            source=source,
            fr_distance_from_prev=fr_dist,
        )
        self._window.append(point)
        return fr_dist

    @property
    def is_drifting(self) -> bool:
        """True if a drift anomaly occurred within the last 30 seconds."""
        return (time.time() - self._last_anomaly_at) < 30.0

    @property
    def stability(self) -> float:
        """Belief stability score in [0, 1]. 1 = maximally stable."""
        if not self._window:
            return 1.0
        recent_dists = [p.fr_distance_from_prev for p in list(self._window)[-10:]]
        if not recent_dists:
            return 1.0
        mean_drift = sum(recent_dists) / len(recent_dists)
        return max(0.0, 1.0 - mean_drift / (math.pi / 2))

    def get_geodesic_velocity(self) -> float:
        """Current drift rate (radians/second)."""
        return self._drift_rate

    def get_state_dict(self) -> dict:
        return {
            "cumulative_distance": round(self._cumulative_distance, 4),
            "drift_rate": round(self._drift_rate, 6),
            "stability": round(self.stability, 4),
            "is_drifting": self.is_drifting,
            "window_size": len(self._window),
        }
