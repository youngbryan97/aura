"""core/world_model/learned_world_model.py -- Variational RNN World Model
=========================================================================
A learned causal world model that replaces heuristic predictions with a
trainable Variational Recurrent Neural Network (VRNN).

The VRNN learns to:
  1. Encode observations into a latent space
  2. Predict the next latent state given actions
  3. Compute surprise (prediction error) for the free energy engine
  4. Imagine future trajectories for planning

Architecture:
  - Encoder: observation → latent (μ, σ)
  - Prior: h_t-1 → predicted latent (μ, σ)
  - Decoder: latent → reconstructed observation
  - Transition: (h_t-1, z_t, a_t) → h_t (GRU cell)

The model uses online learning during waking (updating on each observation)
and batch consolidation during dream cycles.

Design principles:
  - Deterministic: uses fixed seeds for initialization
  - Bounded: latent dimensions and hidden size are capped
  - CPU-only: runs on numpy, no GPU required
  - Grounded: prediction error feeds into the free energy engine
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.LearnedWorldModel")

_DATA_DIR = state_root() / "data" / "world_model"
_MODEL_PATH = _DATA_DIR / "vrnn_state.npz"

#: Every parameter the model actually learns. Naming them here makes the list
#: checkable: a weight missing from it is a weight that never moves, which is
#: exactly the defect this file used to have.
_TRAINABLE = (
    "W_enc", "b_enc", "W_prior", "b_prior", "W_dec", "b_dec",
    "W_z", "b_z", "W_r", "b_r", "W_h", "b_h",
)

#: Steps of truncated backpropagation through time. A single step gives the
#: transition weights no gradient at all — the only thing that grades a hidden
#: state is what the *next* step does with it — so this window is the
#: difference between a learned recurrence and a random one.
_BPTT_WINDOW = 8

#: Free-bits floor, in nats per latent dimension. Below this a dimension stops
#: being pushed toward the prior. Without it the cheapest way to reduce the
#: loss is for the posterior to collapse onto the prior and encode nothing,
#: which looks like excellent convergence and is total amnesia.
_FREE_BITS_NATS = 0.02

#: Global gradient-norm clip.
_GRAD_CLIP_NORM = 5.0

#: Training passes per background cycle. Bounded so a backlog costs many small
#: cycles rather than one long one that starves the checkpoint.
_TRAIN_PASSES_PER_CYCLE = 4

#: Seconds between checkpoints. The previous cadence was every 500 steps, and
#: on the live instance the model never reached 500 steps in a session — the
#: checkpoint directory had been empty since the day it was created.
_CHECKPOINT_INTERVAL_S = 120.0


@dataclass
class WorldModelConfig:
    """Configuration for the VRNN world model."""
    observation_dim: int = 64      # Input observation dimension
    latent_dim: int = 32           # Latent state dimension
    hidden_dim: int = 128          # GRU hidden dimension
    action_dim: int = 16           # Action embedding dimension
    learning_rate: float = 0.001   # Online learning rate
    kl_weight: float = 0.1         # KL divergence weight
    max_trajectory_len: int = 50   # Max imagination trajectory
    seed: int = 42                 # Deterministic initialization seed
    replay_buffer_size: int = 500  # Experience replay buffer


@dataclass
class WorldModelPrediction:
    """Output of a world model prediction step."""
    predicted_state: np.ndarray     # Predicted next observation
    surprise: float                 # Prediction error magnitude
    kl_divergence: float            # KL between posterior and prior
    reconstruction_error: float     # Decoder reconstruction loss
    latent_mean: np.ndarray         # Posterior mean
    latent_logvar: np.ndarray       # Posterior log-variance
    confidence: float               # 1.0 - surprise (how confident the prediction is)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surprise": round(self.surprise, 6),
            "kl_divergence": round(self.kl_divergence, 6),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "confidence": round(self.confidence, 4),
            "latent_norm": round(float(np.linalg.norm(self.latent_mean)), 4),
            "timestamp": self.timestamp,
        }


class LearnedWorldModel:
    """Variational RNN world model for causal prediction.

    Usage:
        model = get_learned_world_model()
        prediction = model.observe(observation_vector, action_vector)
        print(f"Surprise: {prediction.surprise:.4f}")

        # Imagine future trajectories
        trajectory = model.imagine(current_obs, action_sequence)
    """

    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        self.config = config or WorldModelConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Dimensions
        obs_d = self.config.observation_dim
        lat_d = self.config.latent_dim
        hid_d = self.config.hidden_dim
        act_d = self.config.action_dim

        # Initialize weights (Xavier initialization with fixed seed)
        scale = lambda fan_in, fan_out: math.sqrt(2.0 / (fan_in + fan_out))

        # Encoder: obs → latent (μ, σ)
        self.W_enc = self._rng.standard_normal((lat_d * 2, obs_d + hid_d)).astype(np.float32) * scale(obs_d + hid_d, lat_d * 2)
        self.b_enc = np.zeros(lat_d * 2, dtype=np.float32)

        # Prior: h → latent (μ, σ)
        self.W_prior = self._rng.standard_normal((lat_d * 2, hid_d)).astype(np.float32) * scale(hid_d, lat_d * 2)
        self.b_prior = np.zeros(lat_d * 2, dtype=np.float32)

        # Decoder: z + h → obs
        self.W_dec = self._rng.standard_normal((obs_d, lat_d + hid_d)).astype(np.float32) * scale(lat_d + hid_d, obs_d)
        self.b_dec = np.zeros(obs_d, dtype=np.float32)

        # GRU transition: (z + action) → h
        gru_input_d = lat_d + act_d
        # Update gate
        self.W_z = self._rng.standard_normal((hid_d, gru_input_d + hid_d)).astype(np.float32) * scale(gru_input_d + hid_d, hid_d)
        self.b_z = np.zeros(hid_d, dtype=np.float32)
        # Reset gate
        self.W_r = self._rng.standard_normal((hid_d, gru_input_d + hid_d)).astype(np.float32) * scale(gru_input_d + hid_d, hid_d)
        self.b_r = np.zeros(hid_d, dtype=np.float32)
        # Candidate
        self.W_h = self._rng.standard_normal((hid_d, gru_input_d + hid_d)).astype(np.float32) * scale(gru_input_d + hid_d, hid_d)
        self.b_h = np.zeros(hid_d, dtype=np.float32)

        # Hidden state
        self.h = np.zeros(hid_d, dtype=np.float32)

        # Experience replay buffer
        self._replay: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.config.replay_buffer_size
        )

        # Optimiser state, one moment pair per trainable parameter.
        self._adam_m = {name: np.zeros_like(getattr(self, name)) for name in _TRAINABLE}
        self._adam_v = {name: np.zeros_like(getattr(self, name)) for name in _TRAINABLE}
        self._adam_t = 0
        self._train_steps = 0
        self._last_loss = 0.0
        self._last_checkpoint = time.time()
        self._pending_since_train = 0
        self._trainer_thread: threading.Thread | None = None
        self._trainer_stop = threading.Event()
        self._train_interval = 2.0

        # Metrics
        self._step_count = 0
        self._total_surprise = 0.0
        self._last_prediction: Optional[WorldModelPrediction] = None
        self._surprise_history: Deque[float] = deque(maxlen=100)

        # Load persisted state
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

        logger.info(
            "LearnedWorldModel initialized: obs=%d, lat=%d, hid=%d, seed=%d",
            obs_d, lat_d, hid_d, self.config.seed,
        )

    # ── Core API ────────────────────────────────────────────────────────

    def observe(
        self,
        observation: np.ndarray,
        action: Optional[np.ndarray] = None,
        *,
        learn: bool = True,
    ) -> WorldModelPrediction:
        """Process an observation and optionally learn from it.

        Args:
            observation: The current observation vector
            action: The action taken (or None for passive observation)
            learn: Whether to update weights (online learning)

        Returns:
            WorldModelPrediction with surprise, KL, etc.
        """
        obs = self._pad_or_truncate(observation, self.config.observation_dim)
        act = self._pad_or_truncate(
            action if action is not None else np.zeros(self.config.action_dim),
            self.config.action_dim,
        )

        # 1. Compute prior: P(z_t | h_t-1)
        prior_params = self.W_prior @ self.h + self.b_prior
        prior_mean, prior_logvar = np.split(prior_params, 2)
        prior_logvar = np.clip(prior_logvar, -5.0, 2.0)

        # 2. Compute posterior: Q(z_t | x_t, h_t-1)
        enc_input = np.concatenate([obs, self.h])
        post_params = self.W_enc @ enc_input + self.b_enc
        post_mean, post_logvar = np.split(post_params, 2)
        post_logvar = np.clip(post_logvar, -5.0, 2.0)

        # 3. Sample z from posterior (reparameterization trick)
        z = self._reparameterize(post_mean, post_logvar)

        # 4. Decode: P(x_t | z_t, h_t-1)
        dec_input = np.concatenate([z, self.h])
        reconstructed = np.tanh(self.W_dec @ dec_input + self.b_dec)

        # 5. GRU transition: h_t = GRU(z_t, a_t, h_t-1)
        h_prev = self.h.copy()
        gru_input = np.concatenate([z, act])
        self.h = self._gru_step(gru_input, self.h)

        # 6. Compute losses
        reconstruction_error = float(np.mean((obs - reconstructed) ** 2))
        kl_divergence = self._kl_divergence(
            post_mean, post_logvar, prior_mean, prior_logvar
        )
        surprise = reconstruction_error + self.config.kl_weight * kl_divergence
        surprise = max(0.0, min(10.0, surprise))

        # 7. Online learning
        if learn:
            # The state the step was *conditioned on*. Replaying from the
            # post-transition state, as this once did, trains the model on a
            # sequence it never actually saw.
            self._replay.append((h_prev, obs.copy(), act.copy()))
            self._pending_since_train += 1

        self._step_count += 1
        self._total_surprise += surprise
        self._surprise_history.append(surprise)

        prediction = WorldModelPrediction(
            predicted_state=reconstructed,
            surprise=surprise,
            kl_divergence=kl_divergence,
            reconstruction_error=reconstruction_error,
            latent_mean=post_mean,
            latent_logvar=post_logvar,
            confidence=max(0.0, 1.0 - min(1.0, surprise)),
        )
        self._last_prediction = prediction

        # Checkpoint on wall-clock, not step count: a model that is stepped
        # rarely still deserves to survive a restart.
        if time.time() - self._last_checkpoint >= _CHECKPOINT_INTERVAL_S:
            self.save()

        return prediction

    def imagine(
        self,
        current_observation: np.ndarray,
        action_sequence: List[np.ndarray],
    ) -> List[WorldModelPrediction]:
        """Imagine a future trajectory given a sequence of actions.

        Uses the prior (not posterior) since future observations
        aren't available. This is the planning pathway.
        """
        trajectory: List[WorldModelPrediction] = []
        h = self.h.copy()  # Don't modify actual hidden state

        for action in action_sequence[:self.config.max_trajectory_len]:
            act = self._pad_or_truncate(action, self.config.action_dim)

            # Prior prediction
            prior_params = self.W_prior @ h + self.b_prior
            prior_mean, prior_logvar = np.split(prior_params, 2)
            prior_logvar = np.clip(prior_logvar, -5.0, 2.0)

            # Sample from prior
            z = self._reparameterize(prior_mean, prior_logvar)

            # Decode
            dec_input = np.concatenate([z, h])
            predicted = np.tanh(self.W_dec @ dec_input + self.b_dec)

            # Transition
            gru_input = np.concatenate([z, act])
            h = self._gru_step(gru_input, h)

            trajectory.append(WorldModelPrediction(
                predicted_state=predicted,
                surprise=0.0,  # Unknown for imagined states
                kl_divergence=0.0,
                reconstruction_error=0.0,
                latent_mean=prior_mean,
                latent_logvar=prior_logvar,
                confidence=0.5,  # Moderate confidence for predictions
            ))

        return trajectory

    # ── Internal Methods ────────────────────────────────────────────────

    def _reparameterize(self, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick for sampling."""
        std = np.exp(0.5 * logvar)
        eps = self._rng.standard_normal(mean.shape).astype(np.float32)
        return mean + eps * std

    def _gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Single GRU step."""
        xh = np.concatenate([x, h])
        z = self._sigmoid(self.W_z @ xh + self.b_z)  # Update gate
        r = self._sigmoid(self.W_r @ xh + self.b_r)  # Reset gate
        xrh = np.concatenate([x, r * h])
        h_candidate = np.tanh(self.W_h @ xrh + self.b_h)
        h_new = (1 - z) * h + z * h_candidate
        # Stability: clip hidden state
        return np.clip(h_new, -5.0, 5.0).astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    @staticmethod
    def _kl_divergence(
        mu1: np.ndarray, logvar1: np.ndarray,
        mu2: np.ndarray, logvar2: np.ndarray,
    ) -> float:
        """KL divergence between two diagonal Gaussians."""
        kl = 0.5 * np.sum(
            logvar2 - logvar1
            + (np.exp(logvar1) + (mu1 - mu2) ** 2) / (np.exp(logvar2) + 1e-8)
            - 1.0
        )
        return max(0.0, float(kl))

    # ── Learning: the real ELBO, backpropagated through time ─────────────
    #
    # What was here before updated ``W_dec`` and ``b_dec`` and nothing else.
    # The encoder, the prior and all three GRU gates kept their random
    # initialisation for the life of the process — so the "variational" model
    # had no variational objective (no KL gradient reached either Gaussian),
    # and the "recurrent" model had no learned recurrence. What it actually
    # was, structurally, is a random-projection encoder feeding a random
    # recurrent map with a trained decoder bolted on the end: a reservoir,
    # arrived at by omission rather than design, and without any of the
    # spectral-radius, leak or washout discipline that makes a reservoir work.
    #
    # This trains all of it, properly: the ELBO (reconstruction + KL) with
    # gradients through the reparameterisation, truncated backpropagation
    # through time so the transition weights get a signal at all, Adam because
    # plain SGD on this loss surface is not stable, and free bits so the
    # posterior cannot collapse onto the prior and quietly stop encoding
    # anything.

    def _forward_step(
        self, h_prev: np.ndarray, obs: np.ndarray, act: np.ndarray
    ) -> dict[str, np.ndarray]:
        """One VRNN step, keeping every intermediate the backward pass needs."""
        prior_params = self.W_prior @ h_prev + self.b_prior
        prior_mean, prior_logvar_raw = np.split(prior_params, 2)
        prior_logvar = np.clip(prior_logvar_raw, -5.0, 2.0)

        enc_input = np.concatenate([obs, h_prev])
        post_params = self.W_enc @ enc_input + self.b_enc
        post_mean, post_logvar_raw = np.split(post_params, 2)
        post_logvar = np.clip(post_logvar_raw, -5.0, 2.0)

        eps = self._rng.standard_normal(post_mean.shape).astype(np.float32)
        std = np.exp(0.5 * post_logvar)
        z = post_mean + eps * std

        dec_input = np.concatenate([z, h_prev])
        pre_dec = self.W_dec @ dec_input + self.b_dec
        recon = np.tanh(pre_dec)

        gru_in = np.concatenate([z, act])
        xh = np.concatenate([gru_in, h_prev])
        gate_z = self._sigmoid(self.W_z @ xh + self.b_z)
        gate_r = self._sigmoid(self.W_r @ xh + self.b_r)
        xrh = np.concatenate([gru_in, gate_r * h_prev])
        h_cand = np.tanh(self.W_h @ xrh + self.b_h)
        h_raw = (1 - gate_z) * h_prev + gate_z * h_cand
        h_new = np.clip(h_raw, -5.0, 5.0)

        return {
            "h_prev": h_prev, "obs": obs, "act": act,
            "prior_mean": prior_mean, "prior_logvar": prior_logvar,
            "prior_clipped": ((prior_logvar_raw < -5.0) | (prior_logvar_raw > 2.0)).astype(np.float32),
            "post_mean": post_mean, "post_logvar": post_logvar,
            "post_clipped": ((post_logvar_raw < -5.0) | (post_logvar_raw > 2.0)).astype(np.float32),
            "eps": eps, "std": std, "z": z,
            "enc_input": enc_input, "dec_input": dec_input, "recon": recon,
            "gru_in": gru_in, "xh": xh, "xrh": xrh,
            "gate_z": gate_z, "gate_r": gate_r, "h_cand": h_cand,
            # dtype is left as computed: forcing float32 here would quantise
            # the state between steps and make the model's own gradients
            # un-checkable against finite differences.
            "h_new": h_new,
            "h_clipped": ((h_raw < -5.0) | (h_raw > 5.0)).astype(np.float32),
        }

    def _backward_step(
        self, cache: dict[str, np.ndarray], grads: dict[str, np.ndarray], d_h_next: np.ndarray
    ) -> np.ndarray:
        """Backprop one step. Returns the gradient flowing into ``h_prev``."""
        obs_d = self.config.observation_dim
        lat_d = self.config.latent_dim
        act_d = self.config.action_dim
        beta = self.config.kl_weight

        h_prev = cache["h_prev"]
        d_h_prev = np.zeros_like(h_prev)

        # ── reconstruction ───────────────────────────────────────────────
        recon = cache["recon"]
        d_recon = -2.0 * (cache["obs"] - recon) / obs_d
        d_pre_dec = d_recon * (1.0 - recon ** 2)
        grads["W_dec"] += np.outer(d_pre_dec, cache["dec_input"])
        grads["b_dec"] += d_pre_dec
        d_dec_input = self.W_dec.T @ d_pre_dec
        d_z = d_dec_input[:lat_d].copy()
        d_h_prev += d_dec_input[lat_d:]

        # ── GRU, carrying the future's gradient back into this step ──────
        d_h = d_h_next * (1.0 - cache["h_clipped"])
        gate_z, h_cand = cache["gate_z"], cache["h_cand"]
        d_gate_z = d_h * (h_cand - h_prev)
        d_h_cand = d_h * gate_z
        d_h_prev += d_h * (1.0 - gate_z)

        d_pre_h = d_h_cand * (1.0 - h_cand ** 2)
        grads["W_h"] += np.outer(d_pre_h, cache["xrh"])
        grads["b_h"] += d_pre_h
        d_xrh = self.W_h.T @ d_pre_h
        d_gru_in = d_xrh[: lat_d + act_d].copy()
        d_rh = d_xrh[lat_d + act_d:]
        d_gate_r = d_rh * h_prev
        d_h_prev += d_rh * cache["gate_r"]

        d_pre_z = d_gate_z * gate_z * (1.0 - gate_z)
        grads["W_z"] += np.outer(d_pre_z, cache["xh"])
        grads["b_z"] += d_pre_z
        d_xh = self.W_z.T @ d_pre_z
        d_gru_in += d_xh[: lat_d + act_d]
        d_h_prev += d_xh[lat_d + act_d:]

        gate_r = cache["gate_r"]
        d_pre_r = d_gate_r * gate_r * (1.0 - gate_r)
        grads["W_r"] += np.outer(d_pre_r, cache["xh"])
        grads["b_r"] += d_pre_r
        d_xh_r = self.W_r.T @ d_pre_r
        d_gru_in += d_xh_r[: lat_d + act_d]
        d_h_prev += d_xh_r[lat_d + act_d:]

        d_z += d_gru_in[:lat_d]

        # ── KL, with free bits so the posterior cannot collapse ──────────
        post_mean, post_logvar = cache["post_mean"], cache["post_logvar"]
        prior_mean, prior_logvar = cache["prior_mean"], cache["prior_logvar"]
        inv_prior_var = np.exp(-prior_logvar)
        delta = post_mean - prior_mean
        kl_per_dim = 0.5 * (
            prior_logvar - post_logvar
            + (np.exp(post_logvar) + delta ** 2) * inv_prior_var
            - 1.0
        )
        # Free bits: dimensions already carrying less than the floor stop
        # being pushed toward the prior. Without this the cheapest way to
        # lower the loss is for the encoder to stop encoding.
        active = (kl_per_dim > _FREE_BITS_NATS).astype(np.float32)

        d_post_mean = d_z + beta * active * delta * inv_prior_var
        d_post_logvar = (
            d_z * 0.5 * cache["eps"] * cache["std"]
            + beta * active * 0.5 * (np.exp(post_logvar) * inv_prior_var - 1.0)
        )
        d_post_logvar *= (1.0 - cache["post_clipped"])
        d_post_params = np.concatenate([d_post_mean, d_post_logvar])
        grads["W_enc"] += np.outer(d_post_params, cache["enc_input"])
        grads["b_enc"] += d_post_params
        d_enc_input = self.W_enc.T @ d_post_params
        d_h_prev += d_enc_input[obs_d:]

        d_prior_mean = -beta * active * delta * inv_prior_var
        d_prior_logvar = beta * active * 0.5 * (
            1.0 - (np.exp(post_logvar) + delta ** 2) * inv_prior_var
        )
        d_prior_logvar *= (1.0 - cache["prior_clipped"])
        d_prior_params = np.concatenate([d_prior_mean, d_prior_logvar])
        grads["W_prior"] += np.outer(d_prior_params, h_prev)
        grads["b_prior"] += d_prior_params
        d_h_prev += self.W_prior.T @ d_prior_params

        return d_h_prev

    def _mini_batch_update(self) -> float:
        """One truncated-BPTT update over a recent window. Returns the loss."""
        if len(self._replay) < _BPTT_WINDOW:
            return 0.0

        window = list(self._replay)[-_BPTT_WINDOW:]
        h_prev = window[0][0].copy()  # detached start state
        caches: list[dict[str, np.ndarray]] = []
        loss = 0.0
        for _, obs, act in window:
            cache = self._forward_step(h_prev, obs, act)
            caches.append(cache)
            loss += float(np.mean((obs - cache["recon"]) ** 2))
            loss += self.config.kl_weight * self._kl_divergence(
                cache["post_mean"], cache["post_logvar"],
                cache["prior_mean"], cache["prior_logvar"],
            )
            h_prev = cache["h_new"]

        grads = {name: np.zeros_like(getattr(self, name)) for name in _TRAINABLE}
        d_h = np.zeros(self.config.hidden_dim, dtype=np.float32)
        for cache in reversed(caches):
            d_h = self._backward_step(cache, grads, d_h)

        scale = 1.0 / len(caches)
        total_norm = math.sqrt(sum(float(np.sum((g * scale) ** 2)) for g in grads.values()))
        clip = min(1.0, _GRAD_CLIP_NORM / (total_norm + 1e-8))
        self._adam_step({name: g * scale * clip for name, g in grads.items()})
        self._last_loss = loss / len(caches)
        self._train_steps += 1
        return self._last_loss

    # ── the training lane ────────────────────────────────────────────────
    #
    # ``observe`` runs inside Aura's live decision loop and must cost a forward
    # pass and nothing else. Training happens on its own thread and applies its
    # result by rebinding parameter arrays rather than mutating them in place.
    # A rebind is atomic under the GIL, so a concurrent forward pass sees
    # either the old array or the new one and never a half-written one — which
    # means the real-time path never takes a lock. The price is that one step
    # can straddle an update, and for a model that is continuously retrained
    # that is not a price at all.

    def start_training(self, *, interval_s: float = 2.0) -> None:
        """Run the training lane in the background. Idempotent."""
        if self._trainer_thread is not None:
            return
        self._train_interval = float(interval_s)
        self._trainer_thread = threading.Thread(
            target=self._train_loop, name="vrnn-trainer", daemon=True
        )
        self._trainer_thread.start()
        logger.info("World model training lane started (every %.1fs)", self._train_interval)

    def stop_training(self) -> None:
        self._trainer_stop.set()

    def _train_loop(self) -> None:
        while not self._trainer_stop.wait(self._train_interval):
            try:
                if self._pending_since_train <= 0:
                    continue
                self._pending_since_train = 0
                for _ in range(_TRAIN_PASSES_PER_CYCLE):
                    if len(self._replay) < _BPTT_WINDOW:
                        break
                    self._mini_batch_update()
                if time.time() - self._last_checkpoint >= _CHECKPOINT_INTERVAL_S:
                    self.save()
            except (ValueError, FloatingPointError, MemoryError, RuntimeError) as exc:
                record_degradation(
                    "learned_world_model", exc, severity="warning",
                    action="world-model training cycle failed; the model keeps its weights",
                )

    def train_now(self, passes: int = 1) -> float:
        """Synchronous training, for tests and for dream-cycle consolidation."""
        loss = 0.0
        for _ in range(max(1, passes)):
            loss = self._mini_batch_update()
        return loss

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        """Adam. Plain SGD on a VRNN objective in float32 does not stay stable."""
        self._adam_t += 1
        lr = self.config.learning_rate
        b1, b2, eps = 0.9, 0.999, 1e-8
        bias1 = 1.0 - b1 ** self._adam_t
        bias2 = 1.0 - b2 ** self._adam_t
        for name, grad in grads.items():
            m = self._adam_m[name] = b1 * self._adam_m[name] + (1 - b1) * grad
            v = self._adam_v[name] = b2 * self._adam_v[name] + (1 - b2) * (grad * grad)
            update = lr * (m / bias1) / (np.sqrt(v / bias2) + eps)
            setattr(self, name, (getattr(self, name) - update).astype(np.float32))

    def _pad_or_truncate(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate a vector to target dimension."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if vec.size == target_dim:
            return vec
        result = np.zeros(target_dim, dtype=np.float32)
        n = min(vec.size, target_dim)
        result[:n] = vec[:n]
        return result

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self) -> bool:
        """Persist weights and hidden state through the governed write path.

        The old version wrote with a bare ``np.savez_compressed`` every 500
        steps, outside the write gateway and outside any governed scope. On the
        live instance the directory had been empty since the day it was
        created — the model had never survived a single restart, so every
        session began from random weights and "online learning" learned
        nothing that lasted an hour.
        """
        try:
            import io

            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                W_enc=self.W_enc, b_enc=self.b_enc,
                W_prior=self.W_prior, b_prior=self.b_prior,
                W_dec=self.W_dec, b_dec=self.b_dec,
                W_z=self.W_z, b_z=self.b_z,
                W_r=self.W_r, b_r=self.b_r,
                W_h=self.W_h, b_h=self.b_h,
                h=self.h,
                step_count=np.array([self._step_count]),
                train_steps=np.array([self._train_steps]),
                adam_t=np.array([self._adam_t]),
            )
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "learned_world_model", domain="state_mutation", receipt_prefix="vrnn-state"
            ):
                gateway.ensure_directory(_MODEL_PATH.parent, source="learned_world_model")
                gateway.write_bytes(_MODEL_PATH, buffer.getvalue(), source="learned_world_model")
            self._last_checkpoint = time.time()
            logger.debug("World model saved (step %d, train %d)", self._step_count, self._train_steps)
            return True
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "learned_world_model", exc, severity="warning",
                action="world-model checkpoint failed; the model continues in memory",
            )
            return False

    def _save(self) -> None:
        """Backwards-compatible alias for existing callers."""
        self.save()

    def _load(self) -> None:
        """Load persisted model weights."""
        try:
            if not _MODEL_PATH.exists():
                return
            data = np.load(str(_MODEL_PATH))
            # Only load if dimensions match
            if data['W_enc'].shape == self.W_enc.shape:
                self.W_enc = data['W_enc']
                self.b_enc = data['b_enc']
                self.W_prior = data['W_prior']
                self.b_prior = data['b_prior']
                self.W_dec = data['W_dec']
                self.b_dec = data['b_dec']
                self.W_z = data['W_z']
                self.b_z = data['b_z']
                self.W_r = data['W_r']
                self.b_r = data['b_r']
                self.W_h = data['W_h']
                self.b_h = data['b_h']
                self.h = data['h']
                self._step_count = int(data['step_count'][0])
                if 'train_steps' in data:
                    self._train_steps = int(data['train_steps'][0])
                if 'adam_t' in data:
                    self._adam_t = int(data['adam_t'][0])
                logger.info(
                    "World model restored (step %d, %d training updates)",
                    self._step_count, self._train_steps,
                )
            else:
                logger.warning("World model dimension mismatch — reinitializing")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("World model load failed: %s", exc)

    # ── Public API ──────────────────────────────────────────────────────

    def get_surprise(self) -> float:
        """Get the most recent surprise value."""
        if self._last_prediction is not None:
            return self._last_prediction.surprise
        return 0.0

    def get_mean_surprise(self) -> float:
        """Get the rolling mean surprise."""
        if not self._surprise_history:
            return 0.0
        return float(np.mean(list(self._surprise_history)))

    def get_status(self) -> Dict[str, Any]:
        """Return model status for observability."""
        return {
            "step_count": self._step_count,
            "mean_surprise": round(self.get_mean_surprise(), 6),
            "last_surprise": round(self.get_surprise(), 6),
            "hidden_norm": round(float(np.linalg.norm(self.h)), 4),
            "replay_buffer_size": len(self._replay),
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 6),
            "training_lane": "background" if self._trainer_thread is not None else "idle",
            "pending_since_train": self._pending_since_train,
            "checkpoint_age_s": round(time.time() - self._last_checkpoint, 1),
            "trainable_parameters": int(sum(getattr(self, n).size for n in _TRAINABLE)),
            "config": {
                "observation_dim": self.config.observation_dim,
                "latent_dim": self.config.latent_dim,
                "hidden_dim": self.config.hidden_dim,
                "seed": self.config.seed,
            },
        }

    def reset_hidden(self) -> None:
        """Reset the hidden state (e.g., on context switch)."""
        self.h = np.zeros(self.config.hidden_dim, dtype=np.float32)


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[LearnedWorldModel] = None


def get_learned_world_model() -> LearnedWorldModel:
    """Get or create the singleton LearnedWorldModel."""
    global _instance
    if _instance is None:
        _instance = LearnedWorldModel()
    return _instance
