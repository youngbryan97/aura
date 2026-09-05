"""core/consciousness/synaptic_plasticity.py
==============================================
Online Synaptic Plasticity — True Fluid Weight-Level Learning.

The Problem:
    The LLM's weights are frozen between training cycles. STDP updates the
    substrate's connectivity, and CRSMLoraBridge captures moments for *nightly*
    batch LoRA updates. But during live operation, the actual generation pipeline
    has no learnable parameters — the model that speaks at 10pm is identical to
    the one that spoke at 10am.

The Solution:
    A small, real, learnable projection matrix (64×64 = 4096 parameters) that
    sits in the causal path between the consciousness substrate and the
    generation pipeline. After every inference:

    1. The prediction error (CRSM surprise) and hedonic outcome are captured.
    2. These form a signed reward signal: r = hedonic_delta - tanh(surprise).
    3. The reward modulates a Hebbian update on the projection matrix:
       dW = lr * reward * outer(substrate_state, response_hash_vector)
    4. On the NEXT inference, the projection matrix transforms the current
       substrate state into a modulation vector that directly adjusts:
       - temperature (how exploratory the generation is)
       - top_p (how much of the distribution is sampled)
       - repetition_penalty (how much repetition is suppressed)

    This means: experiences that produced positive hedonic outcomes in a given
    substrate state will bias future generation in similar states toward the
    same exploratory/conservative style. The system literally learns from
    experience in real-time.

    The matrix persists to disk and survives restarts. It is *the system's own*
    learned relationship between how it feels and how it should speak.

Identity Protection (MESU-inspired):
    Weights that stabilize (low variance over many updates) become "identity
    locked" — they represent core learned associations that should not be
    overwritten by single experiences. This prevents catastrophic forgetting
    of personality-defining generation preferences.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Consciousness.SynapticPlasticity")

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECTION_DIM = 64           # Matches substrate projection and mesh output
LEARNING_RATE = 0.002         # Base Hebbian learning rate
MAX_LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 0.0001
REWARD_DECAY = 0.95           # EMA decay for reward baseline
WEIGHT_CLIP = 1.5             # Max absolute weight value
IDENTITY_LOCK_THRESHOLD = 0.005   # Variance below this → locked
IDENTITY_LOCK_WINDOW = 50    # Must be stable for this many updates
MODULATION_STRENGTH = 0.15   # How strongly the matrix affects sampling
PERSIST_PATH = state_root() / "data" / "synaptic_plasticity_state.json"
PERSIST_INTERVAL = 60        # Save every 60 seconds


@dataclass
class PlasticitySnapshot:
    """Immutable snapshot of the plasticity state for telemetry."""
    total_updates: int
    mean_weight: float
    weight_norm: float
    locked_fraction: float
    reward_baseline: float
    last_reward: float
    last_modulation: Dict[str, float]
    effective_lr: float


class SynapticPlasticityEngine:
    """Real-time online learning via reward-modulated Hebbian plasticity.

    The projection matrix W maps substrate state → generation modulation.
    It is the only learnable parameter that updates during live operation.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # The learnable projection matrix
        rng = np.random.default_rng(seed=7)
        self._W = rng.standard_normal((PROJECTION_DIM, PROJECTION_DIM)).astype(
            np.float32
        ) * 0.01

        # Identity lock tracking (MESU-inspired)
        self._weight_variance = np.ones_like(self._W) * 0.5
        self._weight_mean = np.zeros_like(self._W)
        self._stable_count = np.zeros_like(self._W, dtype=np.int32)
        self._locked = np.zeros_like(self._W, dtype=bool)

        # Reward baseline (running average for advantage computation)
        self._reward_baseline = 0.0
        self._total_updates = 0

        # Pre-inference capture state
        self._pre_substrate: Optional[np.ndarray] = None
        self._pre_hedonic: float = 0.0

        # Last modulation output (for telemetry)
        self._last_modulation: Dict[str, float] = {}
        self._last_reward: float = 0.0
        self._effective_lr: float = LEARNING_RATE

        # Persistence
        self._last_persist_at: float = 0.0

        # Try to load persisted state
        self._load_state()

        logger.info(
            "SynapticPlasticityEngine ONLINE — %d learnable parameters, "
            "%d locked",
            PROJECTION_DIM * PROJECTION_DIM,
            int(self._locked.sum()),
        )

    # ── Pre-Inference Capture ─────────────────────────────────────────────

    def pre_inference_capture(self, substrate_state: np.ndarray, hedonic_score: float):
        """Called BEFORE inference. Snapshot the substrate state for learning."""
        with self._lock:
            try:
                state = np.asarray(substrate_state, dtype=np.float32).reshape(-1)
                if state.size >= PROJECTION_DIM:
                    self._pre_substrate = state[:PROJECTION_DIM].copy()
                elif state.size > 0:
                    padded = np.zeros(PROJECTION_DIM, dtype=np.float32)
                    padded[: state.size] = state
                    self._pre_substrate = padded
                else:
                    self._pre_substrate = None
                self._pre_hedonic = float(hedonic_score)
            except (TypeError, ValueError) as exc:
                record_degradation("synaptic_plasticity", exc)
                self._pre_substrate = None

    # ── Post-Inference Learning ───────────────────────────────────────────

    def post_inference_learn(
        self,
        response_text: str,
        hedonic_after: float,
        surprise: float,
    ):
        """Called AFTER inference. Compute reward and update the projection matrix.

        This is where true online learning happens:
        1. Compute reward from hedonic change and surprise
        2. Hash the response into a vector (what was produced)
        3. Hebbian update: dW = lr * reward * outer(substrate, response_hash)
        4. Apply MESU identity protection
        """
        with self._lock:
            if self._pre_substrate is None:
                return

            substrate = self._pre_substrate
            self._pre_substrate = None

            # 1. Compute reward signal
            hedonic_delta = hedonic_after - self._pre_hedonic
            # Reward = hedonic improvement - surprise penalty
            # Positive hedonic delta with low surprise = strong positive reward
            # Negative hedonic delta or high surprise = negative reward
            raw_reward = hedonic_delta - 0.3 * np.tanh(surprise)
            # Advantage: reward relative to baseline (reduces variance)
            advantage = raw_reward - self._reward_baseline
            self._reward_baseline = REWARD_DECAY * self._reward_baseline + (
                1 - REWARD_DECAY
            ) * raw_reward
            self._last_reward = float(advantage)

            # 2. Hash response into a projection vector
            response_vec = self._hash_response(response_text)

            # 3. Modulate learning rate by surprise magnitude
            # High surprise = larger correction needed
            self._effective_lr = np.clip(
                LEARNING_RATE * (1.0 + abs(surprise) * 3.0),
                MIN_LEARNING_RATE,
                MAX_LEARNING_RATE,
            )

            # 4. Hebbian update: dW = lr * advantage * outer(substrate, response)
            dW = self._effective_lr * advantage * np.outer(substrate, response_vec)
            dW = np.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )

            # 5. MESU: Update per-weight variance tracking
            delta_from_mean = dW - self._weight_mean
            alpha = 0.02
            self._weight_mean += alpha * delta_from_mean
            self._weight_variance = (
                (1.0 - alpha) * self._weight_variance
                + alpha * delta_from_mean * (dW - self._weight_mean)
            )
            self._weight_variance = np.clip(self._weight_variance, 1e-10, 10.0)

            # Update stability counter for identity locking
            newly_stable = self._weight_variance < IDENTITY_LOCK_THRESHOLD
            self._stable_count[newly_stable] += 1
            self._stable_count[~newly_stable] = 0
            self._locked = self._stable_count >= IDENTITY_LOCK_WINDOW

            # 6. Apply update, respecting identity locks
            # Locked weights don't change — they represent core identity
            dW[self._locked] = 0.0
            self._W += dW
            self._W = np.clip(self._W, -WEIGHT_CLIP, WEIGHT_CLIP).astype(np.float32)

            self._total_updates += 1

            # Periodic persistence
            now = time.time()
            if now - self._last_persist_at > PERSIST_INTERVAL:
                self._persist_state()
                self._last_persist_at = now

            logger.debug(
                "SynapticPlasticity: update #%d reward=%.4f lr=%.5f locked=%.1f%%",
                self._total_updates,
                advantage,
                self._effective_lr,
                100.0 * self._locked.mean(),
            )

    # ── Generation Modulation ─────────────────────────────────────────────

    def compute_modulation(self, substrate_state: np.ndarray) -> Dict[str, float]:
        """Compute generation parameter modulations from current substrate state.

        This is the causal output: the learned projection matrix transforms the
        current felt state into concrete sampling adjustments.

        Returns a dict with delta values for temperature, top_p, repetition_penalty.
        """
        with self._lock:
            try:
                state = np.asarray(substrate_state, dtype=np.float32).reshape(-1)
                if state.size < PROJECTION_DIM:
                    padded = np.zeros(PROJECTION_DIM, dtype=np.float32)
                    padded[: state.size] = state
                    state = padded
                else:
                    state = state[:PROJECTION_DIM]

                # Project through the learned matrix
                projection = self._W @ state  # (64,)
                projection = np.nan_to_num(projection, nan=0.0)

                # Extract modulation signals from different regions of the
                # projection vector (different "cortical columns" of the matrix
                # learn different generation aspects)
                temp_signal = float(np.mean(projection[:16]))
                top_p_signal = float(np.mean(projection[16:32]))
                rep_signal = float(np.mean(projection[32:48]))
                # Remaining 48-64 reserved for future modulations

                # Scale to reasonable delta ranges
                modulation = {
                    "temperature_delta": np.clip(
                        temp_signal * MODULATION_STRENGTH, -0.15, 0.15
                    ),
                    "top_p_delta": np.clip(
                        top_p_signal * MODULATION_STRENGTH * 0.5, -0.1, 0.1
                    ),
                    "repetition_penalty_delta": np.clip(
                        rep_signal * MODULATION_STRENGTH * 0.3, -0.1, 0.1
                    ),
                }

                self._last_modulation = {k: round(float(v), 5) for k, v in modulation.items()}
                return self._last_modulation

            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                record_degradation("synaptic_plasticity", exc)
                self._last_modulation = {}
                return {}

    # ── Response Hashing ──────────────────────────────────────────────────

    @staticmethod
    def _hash_response(text: str) -> np.ndarray:
        """Hash response text into a 64-d unit vector.

        Uses SHA-256 bytes as deterministic seeds for a projection vector.
        This gives each unique response a stable, reproducible vector
        representation for the Hebbian outer product.
        """
        digest = hashlib.sha256((text or "")[:500].encode("utf-8", errors="ignore")).digest()
        # Use 8 bytes at a time to generate 8 float32 values, cycle for 64
        values = []
        for i in range(PROJECTION_DIM):
            byte_idx = i % 32
            val = (digest[byte_idx] - 128) / 128.0
            values.append(val)
        vec = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    # ── Telemetry ─────────────────────────────────────────────────────────

    def get_snapshot(self) -> PlasticitySnapshot:
        with self._lock:
            return PlasticitySnapshot(
                total_updates=self._total_updates,
                mean_weight=float(np.mean(self._W)),
                weight_norm=float(np.linalg.norm(self._W)),
                locked_fraction=float(self._locked.mean()),
                reward_baseline=round(self._reward_baseline, 5),
                last_reward=round(self._last_reward, 5),
                last_modulation=dict(self._last_modulation),
                effective_lr=round(self._effective_lr, 6),
            )

    def get_status(self) -> Dict[str, Any]:
        snap = self.get_snapshot()
        return {
            "total_updates": snap.total_updates,
            "mean_weight": round(snap.mean_weight, 6),
            "weight_norm": round(snap.weight_norm, 4),
            "locked_fraction": round(snap.locked_fraction, 4),
            "reward_baseline": snap.reward_baseline,
            "last_reward": snap.last_reward,
            "last_modulation": snap.last_modulation,
            "effective_lr": snap.effective_lr,
        }

    def is_ready(self) -> bool:
        """Synchronous liveness probe for optional runtime health."""
        with self._lock:
            return bool(
                isinstance(self._W, np.ndarray)
                and self._W.shape == (PROJECTION_DIM, PROJECTION_DIM)
                and np.isfinite(self._W).all()
                and isinstance(self._locked, np.ndarray)
                and self._locked.shape == self._W.shape
            )

    def get_context_block(self) -> str:
        """Minimal context for the LLM about plasticity state."""
        with self._lock:
            if self._total_updates == 0:
                return ""
            return (
                f"## SYNAPTIC PLASTICITY\n"
                f"- {self._total_updates} online weight updates applied\n"
                f"- {self._locked.sum()}/{PROJECTION_DIM * PROJECTION_DIM} "
                f"synapses identity-locked\n"
                f"- reward baseline: {self._reward_baseline:.3f}"
            )

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist_state(self):
        """Save learnable weights to disk."""
        try:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "W": self._W.tolist(),
                "weight_variance": self._weight_variance.tolist(),
                "weight_mean": self._weight_mean.tolist(),
                "stable_count": self._stable_count.tolist(),
                "locked": self._locked.tolist(),
                "reward_baseline": self._reward_baseline,
                "total_updates": self._total_updates,
                "saved_at": time.time(),
            }
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            # Post-inference weight persistence is local runtime maintenance;
            # it MUST run inside a governed scope or the file-write gateway
            # raises GovernanceViolationError. That error previously escaped
            # this best-effort persist, propagated through post_inference_learn
            # into the inference gate's fail-closed boundary, and killed the
            # whole turn (cortex circuit opened after 5 → every turn timed out).
            with local_internal_governed_scope(
                "synaptic_plasticity.persist", domain="file_write"
            ):
                get_file_write_gateway().write_text(
                    PERSIST_PATH,
                    json.dumps(state),
                    source="synaptic_plasticity.persist",
                )
            logger.debug("SynapticPlasticity: persisted state (%d updates)", self._total_updates)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            # Best-effort persistence must never fail inference learning, let
            # alone the turn. RuntimeError covers GovernanceViolationError
            # (its subclass) as belt-and-suspenders behind the governed scope.
            record_degradation("synaptic_plasticity", exc)

    def _load_state(self):
        """Load persisted weights from disk."""
        try:
            if not PERSIST_PATH.exists():
                return
            raw = json.loads(PERSIST_PATH.read_text(encoding="utf-8"))
            W = np.array(raw["W"], dtype=np.float32)
            if W.shape == (PROJECTION_DIM, PROJECTION_DIM):
                self._W = W
                self._weight_variance = np.array(raw.get("weight_variance", self._weight_variance), dtype=np.float32)
                self._weight_mean = np.array(raw.get("weight_mean", self._weight_mean), dtype=np.float32)
                self._stable_count = np.array(raw.get("stable_count", self._stable_count), dtype=np.int32)
                self._locked = np.array(raw.get("locked", self._locked), dtype=bool)
                self._reward_baseline = float(raw.get("reward_baseline", 0.0))
                self._total_updates = int(raw.get("total_updates", 0))
                logger.info(
                    "SynapticPlasticity: loaded persisted state (%d updates, %.1f%% locked)",
                    self._total_updates,
                    100.0 * self._locked.mean(),
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            record_degradation("synaptic_plasticity", exc)
            logger.debug("SynapticPlasticity: could not load persisted state: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_ENGINE: Optional[SynapticPlasticityEngine] = None


def get_synaptic_plasticity() -> SynapticPlasticityEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SynapticPlasticityEngine()
    return _ENGINE
