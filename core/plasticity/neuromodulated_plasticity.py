"""Backpropamine-style neuromodulated plastic linear layer.

Effective weight at each step::

    W_eff = W_base + alpha * Hebb

Hebbian + eligibility update::

    E[t+1]    = decay * E[t] + (1 - decay) * outer(y, x)
    Hebb[t+1] = clip(Hebb[t] + modulation * reward * E[t+1], -1, 1)

Hard safety: ``max_delta_norm`` caps a single step's update so a
spurious reward cannot rewrite the layer.  ``reset_plastic_state``
zeroes Hebb + eligibility for episode boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PlasticityConfig:
    in_dim: int
    out_dim: int
    alpha_scale: float = 0.10
    eligibility_decay: float = 0.95
    hebb_clip: float = 1.0
    max_delta_norm: float = 0.25


class NeuromodulatedPlasticLayer:
    def __init__(self, cfg: PlasticityConfig, *, seed: int = 7):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.W_base = (
            rng.standard_normal((cfg.out_dim, cfg.in_dim)).astype(np.float32)
            / np.sqrt(cfg.in_dim)
        )
        self.bias = np.zeros(cfg.out_dim, dtype=np.float32)
        self.alpha = np.full(
            (cfg.out_dim, cfg.in_dim), cfg.alpha_scale, dtype=np.float32
        )
        self.hebb = np.zeros((cfg.out_dim, cfg.in_dim), dtype=np.float32)
        self.eligibility = np.zeros((cfg.out_dim, cfg.in_dim), dtype=np.float32)

        self.last_x: Optional[np.ndarray] = None
        self.last_y: Optional[np.ndarray] = None
        self.total_updates = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        W_eff = self.W_base + self.alpha * self.hebb
        y = np.tanh(W_eff @ x + self.bias)
        self.last_x = x
        self.last_y = y
        return y

    def update(self, *, reward: float, modulation: float) -> Dict[str, Any]:
        if self.last_x is None or self.last_y is None:
            return {"updated": False, "reason": "no_activity"}

        reward = float(np.clip(reward, -1.0, 1.0))
        modulation = float(np.clip(modulation, 0.0, 1.0))

        outer = np.outer(self.last_y, self.last_x).astype(np.float32)
        self.eligibility = (
            self.cfg.eligibility_decay * self.eligibility
            + (1.0 - self.cfg.eligibility_decay) * outer
        )

        delta = modulation * reward * self.eligibility
        norm = float(np.linalg.norm(delta))
        if norm > self.cfg.max_delta_norm:
            delta *= self.cfg.max_delta_norm / max(norm, 1e-9)

        self.hebb = np.clip(
            self.hebb + delta, -self.cfg.hebb_clip, self.cfg.hebb_clip
        ).astype(np.float32)

        self.total_updates += 1
        return {
            "updated": True,
            "reward": reward,
            "modulation": modulation,
            "delta_norm": float(np.linalg.norm(delta)),
            "hebb_norm": float(np.linalg.norm(self.hebb)),
            "total_updates": self.total_updates,
        }

    def reset_plastic_state(self) -> None:
        self.hebb.fill(0.0)
        self.eligibility.fill(0.0)
        self.last_x = None
        self.last_y = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "shape": [self.cfg.out_dim, self.cfg.in_dim],
            "hebb_norm": float(np.linalg.norm(self.hebb)),
            "eligibility_norm": float(np.linalg.norm(self.eligibility)),
            "total_updates": self.total_updates,
        }
