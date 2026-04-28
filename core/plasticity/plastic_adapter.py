"""Plastic adapter that reweights feature vectors during life.

The adapter wraps a ``NeuromodulatedPlasticLayer``.  Callers feed it a
feature vector (e.g. from the grounding kernel), get a transformed
vector back, and later call ``update_from_reward`` once external
semantic reward is known.  This is the load-bearing piece that turns
"the matrix changed" into "the matrix changed in a way that improves
future grounded behaviour."
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from core.plasticity.neuromodulated_plasticity import (
    NeuromodulatedPlasticLayer,
    PlasticityConfig,
)


class GroundingPlasticAdapter:
    def __init__(self, *, feature_dim: int = 128, seed: int = 7):
        self.layer = NeuromodulatedPlasticLayer(
            PlasticityConfig(in_dim=feature_dim, out_dim=feature_dim),
            seed=seed,
        )

    def adapt_features(self, features: List[float]) -> List[float]:
        x = np.asarray(features, dtype=np.float32)
        y = self.layer.forward(x)
        norm = np.linalg.norm(y)
        if norm > 1e-9:
            y = y / norm
        return y.tolist()

    def update_from_reward(self, *, reward: float, modulation: float) -> Dict[str, Any]:
        return self.layer.update(reward=reward, modulation=modulation)

    def reset(self) -> None:
        self.layer.reset_plastic_state()

    def snapshot(self) -> Dict[str, Any]:
        return self.layer.snapshot()
