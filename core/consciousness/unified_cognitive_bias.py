"""core/consciousness/unified_cognitive_bias.py
===================================================
Unifies the BIAS_DIM vectors from multiple consciousness layers into a
single priority bias that the GlobalWorkspace scorer consumes.

Sources:
    • HemisphericSplit.fused_bias()                — left/right fusion
    • MinimalSelfhood.get_priority_bias()          — chemotaxis/directed
    • RecursiveTheoryOfMind.get_observer_bias()    — scrub-jay effect

Fusion rule:
    final = tanh(
        w_hemi       * hemispheric_bias
      + w_selfhood   * selfhood_bias
      + w_observer   * observer_bias
    )

The default weights are tuned so that each layer can dominate in its
regime: selfhood dominates when body budget is in deficit, observer
dominates when many observers are watching, hemispheric dominates
otherwise.

This keeps the biases COMPOSABLE — each layer's intended contribution
is visible in the fused vector and attributable via
``contribution_summary``.
"""
from __future__ import annotations


import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("Consciousness.UnifiedBias")

BIAS_DIM = 16

DEFAULT_WEIGHTS = {
    "hemi": 0.40,
    "selfhood": 0.35,
    "observer": 0.25,
}


@dataclass
class UnifiedBiasSnapshot:
    fused: np.ndarray
    hemi_contribution: np.ndarray
    selfhood_contribution: np.ndarray
    observer_contribution: np.ndarray
    weights: Dict[str, float]
    observer_presence: float
    ts: float = field(default_factory=time.time)


class UnifiedCognitiveBias:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = dict(weights or DEFAULT_WEIGHTS)
        self._lock = threading.Lock()
        self._last: Optional[UnifiedBiasSnapshot] = None
        logger.info("UnifiedCognitiveBias initialized: weights=%s", self._weights)

    @staticmethod
    def _vec(x: Any) -> np.ndarray:
        if x is None:
            return np.zeros(BIAS_DIM, dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size < BIAS_DIM:
            arr = np.pad(arr, (0, BIAS_DIM - arr.size))
        return arr[:BIAS_DIM]

    def fuse(self,
             hemi_bias: Optional[np.ndarray],
             selfhood_bias: Optional[np.ndarray],
             observer_bias: Optional[np.ndarray],
             observer_presence: float = 0.0
             ) -> UnifiedBiasSnapshot:
        h = self._vec(hemi_bias)
        s = self._vec(selfhood_bias)
        o = self._vec(observer_bias)
        w = self._weights

        fused = np.tanh(
            w["hemi"] * h
            + w["selfhood"] * s
            + w["observer"] * o
        ).astype(np.float32)

        snap = UnifiedBiasSnapshot(
            fused=fused,
            hemi_contribution=(w["hemi"] * h).astype(np.float32),
            selfhood_contribution=(w["selfhood"] * s).astype(np.float32),
            observer_contribution=(w["observer"] * o).astype(np.float32),
            weights=dict(w),
            observer_presence=float(observer_presence),
        )
        with self._lock:
            self._last = snap
        return snap

    def last(self) -> Optional[UnifiedBiasSnapshot]:
        with self._lock:
            return self._last

    def set_weights(self, hemi: float, selfhood: float, observer: float) -> None:
        total = hemi + selfhood + observer
        if total <= 0:
            return
        self._weights = {
            "hemi": hemi / total,
            "selfhood": selfhood / total,
            "observer": observer / total,
        }

    def contribution_summary(self) -> Dict[str, Any]:
        s = self.last()
        if s is None:
            return {"has_snapshot": False}
        return {
            "has_snapshot": True,
            "fused_peak": int(np.argmax(np.abs(s.fused))),
            "hemi_norm": round(float(np.linalg.norm(s.hemi_contribution)), 4),
            "selfhood_norm": round(float(np.linalg.norm(s.selfhood_contribution)), 4),
            "observer_norm": round(float(np.linalg.norm(s.observer_contribution)), 4),
            "observer_presence": round(s.observer_presence, 3),
            "weights": s.weights,
        }


_INSTANCE: Optional[UnifiedCognitiveBias] = None


def get_unified_cognitive_bias() -> UnifiedCognitiveBias:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UnifiedCognitiveBias()
    return _INSTANCE
