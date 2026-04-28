"""core/brain/llm/continuous_substrate.py
────────────────────────────────────────
Aura v5.0: Continuous Latent Streaming (Substrate Pacing Inversion).

A 64-neuron Liquid Time-Constant ODE that runs at ~20 Hz, integrating affect
inputs with stochastic perturbation to produce a state vector and a derived
telemetry summary (valence/arousal/dominance/phi/etc.) that downstream
subsystems can read.

Implementation notes
--------------------
- CPU-only numpy. No model load. Negligible RAM, low single-core CPU.
- Pure-Python ODE with explicit-Euler integration; sufficient given the
  20 Hz step rate and the bounded tanh nonlinearity.
- ``get_state_summary`` derives valence/arousal/dominance/phi from the
  64-D state vector via fixed projections rather than returning hardcoded
  values. The projections are intentionally simple so the readouts are
  legible while reflecting actual dynamics.
- The previous implementation was a hardcoded-monologue stub. This module
  replaces that. The stub-attestation in README's "What's stubbed and what's
  real" should be updated when this lands in production.

Public API (unchanged from prior stub):
- ``ContinuousSubstrate(model_path, device)``
- ``await substrate.start()``
- ``await substrate.stop()``
- ``substrate.get_latest_monologue(limit)``  — repurposed: returns most-recent
  human-readable substrate snapshots, not LLM tokens.
- ``substrate.get_state_summary() -> Dict``
- ``substrate.clear_buffer()``
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Substrate")

NEURONS = 64
STEP_HZ = 20.0
STEP_DT = 1.0 / STEP_HZ
DECAY = 0.05
NOISE_SIGMA = 0.01
INPUT_GAIN = 0.5
SNAPSHOT_BUFFER = 100


def _make_recurrent_matrix(seed: int = 17) -> np.ndarray:
    """Sparse, slightly anti-symmetric recurrent connectivity. Anti-symmetry
    encourages oscillatory dynamics rather than fixed-point convergence,
    which gives the substrate something to "do" even with steady input."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((NEURONS, NEURONS)) * 0.15
    sparse_mask = rng.random((NEURONS, NEURONS)) < 0.25
    raw = raw * sparse_mask
    skew = 0.5 * (raw - raw.T)
    sym = 0.5 * (raw + raw.T)
    W = 0.6 * skew + 0.4 * sym
    np.fill_diagonal(W, -0.2)
    return W.astype(np.float32)


# Fixed readout projections from 64-D state to interpretable scalars.
_RNG = np.random.default_rng(42)
_VALENCE_PROJ = _RNG.standard_normal(NEURONS).astype(np.float32) / math.sqrt(NEURONS)
_AROUSAL_PROJ = _RNG.standard_normal(NEURONS).astype(np.float32) / math.sqrt(NEURONS)
_DOMINANCE_PROJ = _RNG.standard_normal(NEURONS).astype(np.float32) / math.sqrt(NEURONS)
_CURIOSITY_PROJ = _RNG.standard_normal(NEURONS).astype(np.float32) / math.sqrt(NEURONS)


class ContinuousSubstrate:
    def __init__(self, model_path: str = "", device: str = "cpu"):
        self.model_path = model_path  # retained for API compatibility; unused
        self.device = device
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._snapshot_buffer: Deque[Dict[str, Any]] = deque(maxlen=SNAPSHOT_BUFFER)
        self._state = np.zeros(NEURONS, dtype=np.float32)
        self._W = _make_recurrent_matrix()
        self._input_signal = np.zeros(NEURONS, dtype=np.float32)
        self._step_count = 0
        self._last_phi_estimate = 0.0
        self._phi_window: Deque[np.ndarray] = deque(maxlen=64)

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self.running:
            return
        logger.info(
            "🧠 [SUBSTRATE] Initializing real ODE substrate (%d neurons, %.0f Hz)",
            NEURONS, STEP_HZ,
        )
        self.running = True
        self._task = get_task_tracker().create_task(self._integration_loop())

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.info("🧠 [SUBSTRATE] ODE substrate halted (real-mode).")

    def inject_input(self, vector: np.ndarray) -> None:
        """Apply an external affect signal to the substrate's input bus.
        Vector should be 64-D; shorter inputs are zero-padded, longer are truncated.
        """
        v = np.asarray(vector, dtype=np.float32).ravel()
        buf = np.zeros(NEURONS, dtype=np.float32)
        n = min(NEURONS, v.size)
        buf[:n] = v[:n]
        self._input_signal = buf

    def get_state_summary(self) -> Dict[str, Any]:
        """Telemetry-compatible summary derived from live dynamics."""
        if not self.running and self._step_count == 0:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "phi": 0.0,
                "curiosity": 0.0,
                "status": "idle",
                "buffer_depth": 0,
                "step_count": 0,
            }
        s = self._state
        return {
            "valence": float(np.tanh(np.dot(s, _VALENCE_PROJ))),
            "arousal": float(np.clip(0.5 * (1.0 + np.tanh(np.dot(s, _AROUSAL_PROJ))), 0.0, 1.0)),
            "dominance": float(np.tanh(np.dot(s, _DOMINANCE_PROJ))),
            "phi": float(self._last_phi_estimate),
            "curiosity": float(np.clip(0.5 * (1.0 + np.tanh(np.dot(s, _CURIOSITY_PROJ))), 0.0, 1.0)),
            "energy": float(np.linalg.norm(s) / math.sqrt(NEURONS)),
            "status": "active" if self.running else "halted",
            "buffer_depth": len(self._snapshot_buffer),
            "step_count": self._step_count,
        }

    def get_state_vector(self) -> np.ndarray:
        """Returns the live 64-D state vector (copy)."""
        return self._state.copy()

    def get_latest_monologue(self, limit: int = 5) -> str:
        """Repurposed: returns most-recent human-readable substrate snapshots,
        which downstream code can either log or feed back to the LLM as
        contextual cues. Compatibility shim with the prior stub interface.
        """
        if not self._snapshot_buffer:
            return ""
        recent = list(self._snapshot_buffer)[-limit:]
        return " | ".join(
            f"v={s['valence']:+.2f} a={s['arousal']:.2f} d={s['dominance']:+.2f} φ={s['phi']:.3f}"
            for s in recent
        )

    def clear_buffer(self) -> None:
        self._snapshot_buffer.clear()

    # ── Integration loop ──────────────────────────────────────────────────

    async def _integration_loop(self) -> None:
        try:
            while self.running:
                self._step_once()
                await asyncio.sleep(STEP_DT)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("🛑 [SUBSTRATE] integration loop crashed: %s", e)
            self.running = False

    def _step_once(self) -> None:
        # Liquid Time-Constant ODE (explicit Euler): dx/dt = -decay·x + tanh(W·x + I) + ξ
        dx = (
            -DECAY * self._state
            + np.tanh(self._W @ self._state + INPUT_GAIN * self._input_signal)
        ) * STEP_DT
        noise = np.random.normal(0.0, NOISE_SIGMA, NEURONS).astype(np.float32)
        self._state = self._state + dx + noise * STEP_DT

        # Soft saturation to keep norms bounded.
        norm = float(np.linalg.norm(self._state))
        if norm > math.sqrt(NEURONS):
            self._state *= math.sqrt(NEURONS) / norm

        self._step_count += 1
        self._phi_window.append(self._state.copy())
        if self._step_count % 20 == 0:  # 1 Hz
            self._last_phi_estimate = self._estimate_phi()
            self._snapshot_buffer.append(self.get_state_summary())

    def _estimate_phi(self) -> float:
        """Cheap proxy for integration: variance of the principal pairwise
        correlation across recent state snapshots, scaled to [0, 1]. This is
        not strict-IIT phi (see README's "What's stubbed and what's real" and
        ARCHITECTURE.md §3 level-of-description caveat) — it is an integration
        proxy meant to give a usable telemetry number derived from real state.
        """
        if len(self._phi_window) < 8:
            return 0.0
        try:
            X = np.stack(list(self._phi_window), axis=0)  # (T, N)
            X = X - X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            Xn = X / std
            C = (Xn.T @ Xn) / max(1, X.shape[0] - 1)
            np.fill_diagonal(C, 0.0)
            magnitude = float(np.mean(np.abs(C)))
            return max(0.0, min(1.0, magnitude * 1.5))
        except Exception:
            return 0.0
