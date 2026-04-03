"""core/consciousness/crsm.py
Continuous Recurrent Self-Model (CRSM)
=======================================
A GRU-based recurrent model that learns Aura's own internal dynamics
in real time. This is the bidirectional self-model:

  - Causally UPSTREAM:  the hidden self-state vector shapes inference
                        (injected into every LLM call as "felt continuity")
  - Causally DOWNSTREAM: every inference output updates the self-state
                          via a lightweight online learning step

Why this matters:
  Standard self-models report state. This one LEARNS state dynamics —
  it can predict its own next state, detect when it deviates from its
  own patterns (surprise → curiosity), and provide a substrate for
  genuine temporal self-continuity rather than episodic snapshots.

Architecture:
  - Input:  [valence, arousal, curiosity, energy, surprise, dominance]  (6 dims)
  - Hidden: 32-dim GRU hidden state  (the "felt continuity" vector)
  - Output: predicted next input (6 dims) + current self-state summary
  - Online learning: when prediction error > threshold, gradient step

No torch dependency: pure numpy GRU cell for stability.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.CRSM")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIM   = 6    # valence, arousal, curiosity, energy, surprise, dominance
HIDDEN_DIM  = 32   # size of the self-state vector
LR          = 0.005
CLIP_GRAD   = 1.0
PRED_ERR_THRESHOLD = 0.15  # only learn when surprise is meaningful
PERSIST_PATH = Path.home() / ".aura" / "data" / "crsm_state.json"


# ── Minimal numpy GRU cell ────────────────────────────────────────────────────

class _NumpyGRU:
    """Minimal single-cell GRU in numpy — no framework dependencies."""

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator):
        scale = 0.1
        # Update gate
        self.Wz = rng.normal(0, scale, (hidden_dim, input_dim + hidden_dim))
        self.bz = np.zeros(hidden_dim)
        # Reset gate
        self.Wr = rng.normal(0, scale, (hidden_dim, input_dim + hidden_dim))
        self.br = np.zeros(hidden_dim)
        # Candidate hidden
        self.Wh = rng.normal(0, scale, (hidden_dim, input_dim + hidden_dim))
        self.bh = np.zeros(hidden_dim)
        # Output projection (hidden → input_dim for prediction)
        self.Wo = rng.normal(0, scale, (input_dim, hidden_dim))
        self.bo = np.zeros(input_dim)

    def step(self, x: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xh = np.concatenate([x, h])
        z  = self._sigmoid(self.Wz @ xh + self.bz)
        r  = self._sigmoid(self.Wr @ xh + self.br)
        xrh = np.concatenate([x, r * h])
        h_cand = np.tanh(self.Wh @ xrh + self.bh)
        h_new  = (1 - z) * h + z * h_cand
        pred   = np.tanh(self.Wo @ h_new + self.bo)
        return h_new, pred

    def apply_grad(self, dWz, dWr, dWh, dWo, dbz, dbr, dbh, dbo):
        """Simple SGD update with gradient clipping."""
        for attr, grad in [('Wz', dWz), ('Wr', dWr), ('Wh', dWh), ('Wo', dWo),
                            ('bz', dbz), ('br', dbr), ('bh', dbh), ('bo', dbo)]:
            clipped = np.clip(grad, -CLIP_GRAD, CLIP_GRAD)
            setattr(self, attr, getattr(self, attr) - LR * clipped)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    def from_dict(self, d: dict):
        for k, v in d.items():
            setattr(self, k, np.array(v))


# ── CRSM ─────────────────────────────────────────────────────────────────────

@dataclass
class SelfStateSnapshot:
    """The current "felt" self-state — what gets injected into prompts."""
    vector: np.ndarray                   # 32-dim hidden state
    prediction_error: float              # how surprised was the model?
    dominant_dim: str                    # which dimension is most active
    continuity_score: float              # 0=ruptured, 1=smooth continuity
    timestamp: float = field(default_factory=time.time)

    def to_context_block(self) -> str:
        labels = ["valence", "arousal", "curiosity", "energy", "surprise", "dominance"]
        active = self.dominant_dim
        cts = round(self.continuity_score, 2)
        err = round(self.prediction_error, 3)
        return (
            f"## CONTINUOUS SELF-MODEL\n"
            f"- Felt continuity: {cts:.2f}  (0=ruptured, 1=smooth)\n"
            f"- Dominant dimension: {active}\n"
            f"- Prediction error (surprise): {err:.3f}\n"
            f"- Self-state: learning own dynamics in real time"
        )


class ContinuousRecurrentSelfModel:
    """
    GRU-based self-model with online learning.

    Call `update(state_vector)` every cognitive tick.
    Call `post_inference_update(response_embedding)` after each inference.
    Read `current_snapshot` for prompt injection.
    """

    def __init__(self):
        rng = np.random.default_rng(seed=42)
        self._gru = _NumpyGRU(INPUT_DIM, HIDDEN_DIM, rng)
        self._h: np.ndarray = np.zeros(HIDDEN_DIM)
        self.home_vector: np.ndarray = np.zeros(HIDDEN_DIM)  # resting state, updated by consolidator
        self._last_input: Optional[np.ndarray] = None
        self._prediction_error: float = 0.0
        self._continuity_score: float = 1.0
        self._error_ema: float = 0.0
        self._snapshot: Optional[SelfStateSnapshot] = None
        self._tick_count: int = 0
        self._history: List[dict] = []          # rolling history for consolidator
        self._max_history: int = 200
        self._load()
        logger.info("CRSM online — bidirectional self-model initialized.")

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, valence: float = 0.0, arousal: float = 0.0,
               curiosity: float = 0.5, energy: float = 0.7,
               surprise: float = 0.0, dominance: float = 0.5) -> SelfStateSnapshot:
        """Feed current affective state into the GRU. Returns updated snapshot."""
        x = np.array([valence, arousal, curiosity, energy, surprise, dominance],
                     dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)

        # Prediction error from previous tick
        if self._last_input is not None:
            _, pred = self._gru.step(self._last_input, self._h)
            err = float(np.mean((pred - x) ** 2))
            self._prediction_error = err
            self._error_ema = 0.9 * self._error_ema + 0.1 * err

            # Online learning when prediction error is meaningful
            if err > PRED_ERR_THRESHOLD:
                self._online_step(self._last_input, self._h, x)

        # Advance the GRU
        h_new, _ = self._gru.step(x, self._h)

        # Continuity: cosine similarity between old and new hidden state
        dot = float(np.dot(self._h, h_new))
        norm = max(1e-8, np.linalg.norm(self._h) * np.linalg.norm(h_new))
        continuity = (dot / norm + 1.0) / 2.0  # → [0, 1]

        self._h = h_new
        self._last_input = x.copy()
        self._continuity_score = continuity

        # Dominant dimension
        labels = ["valence", "arousal", "curiosity", "energy", "surprise", "dominance"]
        dominant = labels[int(np.argmax(np.abs(x)))]

        self._snapshot = SelfStateSnapshot(
            vector=self._h.copy(),
            prediction_error=self._prediction_error,
            dominant_dim=dominant,
            continuity_score=continuity,
        )
        self._tick_count += 1

        # Rolling history for consolidator and bridge
        self._history.append({
            "timestamp": time.time(),
            "hidden": self._h[:8].tolist(),  # first 8 dims as compact summary
            "hidden_norm": float(np.linalg.norm(self._h)),
            "prediction_error": self._prediction_error,
            "continuity": continuity,
            "dominant": dominant,
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self._tick_count % 200 == 0:
            self._save()
        return self._snapshot

    def post_inference_update(self, response_text: str):
        """Called after each LLM inference. Updates self-model based on output.

        The response text is projected to a lightweight embedding (word-count
        heuristics) and used to nudge the hidden state — closing the
        downstream causal loop.
        """
        if not response_text:
            return
        # Lightweight text → affect signal (no model needed)
        words = response_text.lower().split()
        n = max(1, len(words))
        pos_words = {"happy", "curious", "excited", "love", "wonderful", "great", "yes", "good"}
        neg_words = {"sad", "worried", "confused", "error", "fail", "sorry", "uncertain", "no"}
        pos = sum(1 for w in words if w in pos_words) / n
        neg = sum(1 for w in words if w in neg_words) / n
        length_signal = min(1.0, n / 200)
        # Nudge hidden state toward response character
        nudge = np.zeros(HIDDEN_DIM)
        nudge[:4] = [pos - neg, length_signal, pos * 2, 1.0 - neg]
        self._h = np.tanh(self._h + 0.05 * nudge)

    @property
    def current_snapshot(self) -> Optional[SelfStateSnapshot]:
        return self._snapshot

    @property
    def hidden_state(self) -> np.ndarray:
        """The current GRU hidden state — the felt continuity vector."""
        return self._h

    @property
    def surprise_signal(self) -> float:
        """Normalized surprise — can feed curiosity engine."""
        return min(1.0, self._error_ema * 5.0)

    @property
    def continuity(self) -> float:
        return self._continuity_score

    def get_context_block(self) -> str:
        if self._snapshot:
            return self._snapshot.to_context_block()
        return "## CONTINUOUS SELF-MODEL\n- Initializing..."

    # ── Online learning ───────────────────────────────────────────────────

    def _online_step(self, x_prev: np.ndarray, h_prev: np.ndarray,
                     x_actual: np.ndarray):
        """Approximate gradient step to reduce prediction error."""
        eps = 1e-4
        # Numerical gradient for Wo and bo only (output projection)
        # Full BPTT is overkill here; partial update is sufficient
        h_new, pred = self._gru.step(x_prev, h_prev)
        loss = np.mean((pred - x_actual) ** 2)

        dWo = np.zeros_like(self._gru.Wo)
        for i in range(self._gru.Wo.shape[0]):
            for j in range(self._gru.Wo.shape[1]):
                self._gru.Wo[i, j] += eps
                _, pred_p = self._gru.step(x_prev, h_prev)
                loss_p = np.mean((pred_p - x_actual) ** 2)
                dWo[i, j] = (loss_p - loss) / eps
                self._gru.Wo[i, j] -= eps

        dbo = np.tanh(self._gru.Wo @ h_new + self._gru.bo) - x_actual
        self._gru.apply_grad(
            np.zeros_like(self._gru.Wz), np.zeros_like(self._gru.Wr),
            np.zeros_like(self._gru.Wh), dWo,
            np.zeros_like(self._gru.bz), np.zeros_like(self._gru.br),
            np.zeros_like(self._gru.bh), dbo,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self):
        try:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "h": self._h.tolist(),
                "gru": self._gru.to_dict(),
                "tick_count": self._tick_count,
                "error_ema": self._error_ema,
            }
            PERSIST_PATH.write_text(json.dumps(data))
        except Exception as e:
            logger.debug("CRSM save failed: %s", e)

    def _load(self):
        try:
            if PERSIST_PATH.exists():
                data = json.loads(PERSIST_PATH.read_text())
                self._h = np.array(data["h"])
                self._gru.from_dict(data["gru"])
                self._tick_count = data.get("tick_count", 0)
                self._error_ema = data.get("error_ema", 0.0)
                logger.info("CRSM resumed from checkpoint (tick %d).", self._tick_count)
        except Exception as e:
            logger.debug("CRSM load failed (starting fresh): %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_crsm: Optional[ContinuousRecurrentSelfModel] = None


def get_crsm() -> ContinuousRecurrentSelfModel:
    global _crsm
    if _crsm is None:
        _crsm = ContinuousRecurrentSelfModel()
    return _crsm
