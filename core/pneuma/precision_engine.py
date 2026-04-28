"""core/pneuma/precision_engine.py
PNEUMA Layer 1 — FitzHugh-Nagumo Metabolic Drive + Per-Head Precision Weights.

The FHN oscillator models metabolic arousal (v) and adaptation (w).
Precision weights Π_h are derived from the FHN voltage state and modulate
the QK^T dot-product before softmax (i.e., sharper attention when aroused).

FHN equations:
    dv/dt = v - v³/3 - w + I_ext
    dw/dt = ε(v + a - bw)

Where:
    v  = membrane potential (arousal proxy)
    w  = recovery variable (fatigue proxy)
    I_ext = external stimulus (derived from heartstone Curiosity + somatic stress)
"""

from core.runtime.errors import record_degradation
import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("PNEUMA.PrecisionEngine")


@dataclass
class FHNState:
    v: float = -1.2       # membrane potential
    w: float = -0.6       # recovery variable
    t: float = field(default_factory=time.time)


class FHNOscillator:
    """FitzHugh-Nagumo oscillator — metabolic drive model."""

    def __init__(self, a: float = 0.7, b: float = 0.8, eps: float = 0.08, dt: float = 0.1):
        self.a = a
        self.b = b
        self.eps = eps
        self.dt = dt
        self.state = FHNState()

    def step(self, i_ext: float = 0.5) -> FHNState:
        """Integrate one time step via Euler method."""
        v, w = self.state.v, self.state.w
        dv = v - (v ** 3) / 3.0 - w + i_ext
        dw = self.eps * (v + self.a - self.b * w)
        get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='v', new_value=v + self.dv_clip(dv) * self.dt, cause='FHNOscillator.step')))
        get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='w', new_value=w + dw * self.dt, cause='FHNOscillator.step')))
        self.state.t = time.time()
        return self.state

    def dv_clip(self, dv: float) -> float:
        """Clamp dv to prevent blow-up."""
        return max(-5.0, min(5.0, dv))

    @property
    def arousal(self) -> float:
        """Normalize v ∈ [-3, 3] → [0, 1]."""
        return (self.state.v + 3.0) / 6.0

    @property
    def fatigue(self) -> float:
        """Normalize w ∈ [-2, 2] → [0, 1]."""
        return (self.state.w + 2.0) / 4.0


class HeadPrecisionModule:
    """Computes per-attention-head precision weights from FHN state.

    Precision Π_h(φ) controls sharpness of attention:
        QK^T_modulated = Π_h(φ) ⊙ QK^T / sqrt(d_k)

    High arousal → high precision → focused (peaked) attention.
    Low arousal → low precision → diffuse (exploratory) attention.
    """

    def __init__(self, n_heads: int = 32, base_precision: float = 1.0):
        self.n_heads = n_heads
        self.base_precision = base_precision
        self._weights: np.ndarray = np.ones(n_heads, dtype=np.float32)

    def update(self, fhn: FHNOscillator, somatic_stress: float = 0.0) -> np.ndarray:
        """Recompute precision weights from FHN state + somatic stress.

        Heads are partitioned into:
          - Query heads (first half): scale with arousal
          - Memory heads (second half): inversely scale with fatigue
        """
        arousal = fhn.arousal
        fatigue = fhn.fatigue
        # Combined drive: high curiosity/stress raises precision; high fatigue lowers it
        drive = 0.6 * arousal + 0.4 * somatic_stress - 0.2 * fatigue
        drive = max(0.1, min(2.0, drive + self.base_precision))

        half = self.n_heads // 2
        # Query partition: precision ∝ drive
        self._weights[:half] = float(drive)
        # Memory partition: slight inverse (broader context when fatigued)
        self._weights[half:] = float(max(0.5, 2.0 - drive * 0.5))
        return self._weights.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights


@dataclass
class PrecisionConfig:
    n_heads: int = 32
    fhn_dt: float = 0.05
    fhn_a: float = 0.7
    fhn_b: float = 0.8
    fhn_eps: float = 0.08
    base_precision: float = 1.0
    drive_update_interval: float = 1.0   # seconds between FHN steps


class PrecisionEngine:
    """Top-level precision controller.

    Exposes:
        step(i_ext)           — advance FHN oscillator
        get_head_weights()    — current Π_h weights
        get_temperature()     — temperature scalar derived from arousal
        get_state_dict()      — full diagnostic dict
    """

    def __init__(self, config: Optional[PrecisionConfig] = None):
        self.config = config or PrecisionConfig()
        self.fhn = FHNOscillator(
            a=self.config.fhn_a,
            b=self.config.fhn_b,
            eps=self.config.fhn_eps,
            dt=self.config.fhn_dt,
        )
        self.head_module = HeadPrecisionModule(
            n_heads=self.config.n_heads,
            base_precision=self.config.base_precision,
        )
        self._last_step = 0.0
        logger.info("PrecisionEngine online (n_heads=%d)", self.config.n_heads)

    def step(self, i_ext: Optional[float] = None) -> FHNState:
        """Advance the FHN oscillator by one step.

        i_ext is derived from live somatic + heartstone state if not supplied.
        """
        if i_ext is None:
            i_ext = self._compute_drive()

        state = self.fhn.step(i_ext)
        somatic_stress = self._get_somatic_stress()
        self.head_module.update(self.fhn, somatic_stress)
        self._last_step = time.time()
        return state

    def _compute_drive(self) -> float:
        """Derive FHN external current from live system state."""
        drive = 0.5  # resting drive
        try:
            from core.affect.heartstone_values import get_heartstone_values
            vals = get_heartstone_values().values
            curiosity = vals.get("Curiosity", 0.5)
            drive += 0.3 * (curiosity - 0.5)
        except Exception as _exc:
            record_degradation('precision_engine', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        try:
            from core.affect.affective_circumplex import get_circumplex
            params = get_circumplex().get_llm_params()
            arousal = params.get("arousal", 0.5)
            drive += 0.2 * (arousal - 0.5)
        except Exception as _exc:
            record_degradation('precision_engine', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return max(0.0, min(1.5, drive))

    def _get_somatic_stress(self) -> float:
        try:
            from core.container import ServiceContainer
            soma = ServiceContainer.get("soma", default=None)
            if soma:
                snap = soma.get_body_snapshot()
                return snap.get("affects", {}).get("stress", 0.0)
        except Exception as _exc:
            record_degradation('precision_engine', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.0

    def get_head_weights(self) -> np.ndarray:
        return self.head_module.weights

    def get_temperature(self) -> float:
        """Map FHN arousal to LLM temperature: arousal 0→0.95, 1→0.55."""
        return 0.95 - 0.40 * self.fhn.arousal

    def get_state_dict(self) -> Dict:
        return {
            "fhn_v": round(self.fhn.state.v, 4),
            "fhn_w": round(self.fhn.state.w, 4),
            "arousal": round(self.fhn.arousal, 4),
            "fatigue": round(self.fhn.fatigue, 4),
            "temperature": round(self.get_temperature(), 4),
            "head_weights_mean": round(float(self.head_module.weights.mean()), 4),
        }
