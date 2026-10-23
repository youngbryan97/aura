"""Contract tests for the VRNN world model.

The model shipped for months with a training routine that updated the decoder
and nothing else: the encoder, the prior and all three GRU gates kept their
random initialisation forever, so the "variational" model had no KL gradient
and the "recurrent" model had no learned recurrence. It had also never once
persisted its state on the live instance — every boot started from random
weights.

These tests fix all three claims in place. The gradient check is the important
one: hand-written backprop is exactly the kind of code that runs, looks right,
and silently optimises the wrong thing.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.world_model.learned_world_model import (
    _BPTT_WINDOW,
    _FREE_BITS_NATS,
    _TRAINABLE,
    LearnedWorldModel,
    WorldModelConfig,
)


class _FixedRng:
    """Freezes the reparameterisation noise so the loss is a pure function."""

    def __init__(self, eps: np.ndarray) -> None:
        self._eps = eps

    def standard_normal(self, shape):  # noqa: ARG002 — shape is fixed by construction
        return self._eps.copy()


def _small_model(seed: int = 5) -> LearnedWorldModel:
    model = LearnedWorldModel(
        WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=5, action_dim=3, seed=seed)
    )
    # float64: float32 finite differences are dominated by rounding.
    for name in _TRAINABLE:
        setattr(model, name, getattr(model, name).astype(np.float64))
    model.h = model.h.astype(np.float64)
    return model


def _free_bits_kl(cache: dict) -> float:
    inv_prior_var = np.exp(-cache["prior_logvar"])
    delta = cache["post_mean"] - cache["prior_mean"]
    per_dim = 0.5 * (
        cache["prior_logvar"] - cache["post_logvar"]
        + (np.exp(cache["post_logvar"]) + delta ** 2) * inv_prior_var - 1.0
    )
    return float(np.sum(np.maximum(per_dim, _FREE_BITS_NATS)))


def _window_loss(model, window, eps_cache) -> float:
    h = window[0][0].copy()
    total = 0.0
    for i, (_, obs, act) in enumerate(window):
        model._rng = _FixedRng(eps_cache[i])
        cache = model._forward_step(h, obs, act)
        total += float(np.mean((obs - cache["recon"]) ** 2))
        total += model.config.kl_weight * _free_bits_kl(cache)
        h = cache["h_new"]
    return total / len(window)


class TestGradients:
    def test_analytic_gradients_match_finite_differences(self):
        """Every trainable parameter, checked numerically."""
        rng = np.random.default_rng(0)
        model = _small_model()
        window = [
            (
                rng.standard_normal(model.config.hidden_dim) * 0.3,
                rng.standard_normal(model.config.observation_dim) * 0.5,
                rng.standard_normal(model.config.action_dim) * 0.5,
            )
            for _ in range(_BPTT_WINDOW)
        ]
        eps_cache = [rng.standard_normal(model.config.latent_dim) for _ in range(_BPTT_WINDOW)]

        h = window[0][0].copy()
        caches = []
        for i, (_, obs, act) in enumerate(window):
            model._rng = _FixedRng(eps_cache[i])
            cache = model._forward_step(h, obs, act)
            caches.append(cache)
            h = cache["h_new"]
        grads = {n: np.zeros_like(getattr(model, n)) for n in _TRAINABLE}
        d_h = np.zeros(model.config.hidden_dim)
        for cache in reversed(caches):
            d_h = model._backward_step(cache, grads, d_h)
        scale = 1.0 / len(caches)

        delta = 1e-6
        for name in _TRAINABLE:
            flat = getattr(model, name).reshape(-1)
            for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
                original = float(flat[idx])
                flat[idx] = original + delta
                up = _window_loss(model, window, eps_cache)
                flat[idx] = original - delta
                down = _window_loss(model, window, eps_cache)
                flat[idx] = original
                numeric = (up - down) / (2 * delta)
                analytic = grads[name].reshape(-1)[idx] * scale
                assert numeric == pytest.approx(analytic, abs=1e-6, rel=0.02), (
                    f"{name}[{idx}]: analytic {analytic} vs numeric {numeric}"
                )

    def test_every_declared_parameter_actually_moves(self):
        """The original defect: ten of twelve parameter groups never trained."""
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=8, latent_dim=4, hidden_dim=12,
                             action_dim=3, seed=9, learning_rate=0.01)
        )
        before = {n: getattr(model, n).copy() for n in _TRAINABLE}
        rng = np.random.default_rng(1)
        for _ in range(60):
            model.observe(rng.normal(size=8), rng.normal(size=3), learn=True)
        model.train_now(passes=10)
        unchanged = [n for n in _TRAINABLE if np.allclose(before[n], getattr(model, n))]
        assert not unchanged, f"these parameters never trained: {unchanged}"


class TestLearning:
    def _driven_sequence(self, n: int, rng):
        """Observations determined by an unobserved running phase.

        The current observation alone is not enough — the model has to carry
        state to predict it, so a frozen transition cannot do well.
        """
        phase, out = 0.0, []
        for _ in range(n):
            a = int(rng.integers(0, 3))
            act = np.zeros(3)
            act[a] = 1.0
            phase += 0.35 + 0.25 * a
            obs = np.array([
                np.sin(phase), np.cos(phase), np.sin(2 * phase),
                np.cos(2 * phase), np.sin(phase / 2), np.cos(phase / 2),
                0.5 * np.sin(phase), 0.5 * np.cos(phase),
            ])
            out.append((obs, act))
        return out

    def _train(self, steps: int, *, freeze_transition: bool, seed: int = 3):
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=8, latent_dim=6, hidden_dim=24, action_dim=3,
                             learning_rate=0.004, kl_weight=0.05, seed=seed)
        )
        frozen = {n: getattr(model, n).copy() for n in ("W_z", "W_r", "W_h", "b_z", "b_r", "b_h")}
        rng = np.random.default_rng(seed)
        for i, (obs, act) in enumerate(self._driven_sequence(steps, rng)):
            model.observe(obs, act, learn=True)
            if i % 2 == 0 and len(model._replay) >= _BPTT_WINDOW:
                model.train_now()
                if freeze_transition:
                    for name, weight in frozen.items():
                        setattr(model, name, weight.copy())
        return model

    def _holdout_error(self, model, holdout) -> float:
        model.reset_hidden()
        return float(np.mean([
            model.observe(obs, act, learn=False).reconstruction_error for obs, act in holdout
        ]))

    @pytest.mark.slow
    def test_it_beats_predicting_the_mean(self):
        holdout = self._driven_sequence(150, np.random.default_rng(99))
        model = self._train(2500, freeze_transition=False)
        mean_obs = np.mean([o for o, _ in holdout], axis=0)
        marginal = float(np.mean([np.mean((o - mean_obs) ** 2) for o, _ in holdout]))
        assert self._holdout_error(model, holdout) < marginal * 0.6

    @pytest.mark.slow
    def test_the_learned_recurrence_earns_its_place(self):
        """If a frozen transition did as well, the GRU would be decoration."""
        holdout = self._driven_sequence(150, np.random.default_rng(99))
        trained = self._train(2500, freeze_transition=False)
        frozen = self._train(2500, freeze_transition=True)
        assert self._holdout_error(trained, holdout) < self._holdout_error(frozen, holdout)

    @pytest.mark.slow
    def test_the_posterior_does_not_collapse(self):
        """Free bits: without them, encoding nothing is the cheapest solution."""
        model = self._train(1500, freeze_transition=False)
        rng = np.random.default_rng(7)
        kls = [
            model.observe(obs, act, learn=False).kl_divergence
            for obs, act in self._driven_sequence(80, rng)
        ]
        assert float(np.mean(kls)) > 0.05, "the posterior has collapsed onto the prior"


class TestStability:
    def test_the_hidden_state_stays_bounded_under_absurd_input(self):
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        for _ in range(200):
            model.observe(np.full(6, 1e6), np.full(2, -1e6), learn=True)
        assert np.all(np.isfinite(model.h))
        assert np.all(np.abs(model.h) <= 5.0 + 1e-6)

    def test_observations_of_the_wrong_shape_are_accepted(self):
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        assert model.observe(np.ones(3), np.ones(9), learn=True) is not None

    def test_imagination_does_not_disturb_the_real_state(self):
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        for _ in range(20):
            model.observe(np.random.default_rng(0).normal(size=6), np.zeros(2))
        before = model.h.copy()
        trajectory = model.imagine(np.zeros(6), [np.zeros(2)] * 5)
        assert len(trajectory) == 5
        assert np.allclose(before, model.h)


class TestPersistence:
    def test_state_survives_a_restart(self, tmp_path, monkeypatch):
        """It never had, on the live instance: the directory was empty for months."""
        import core.world_model.learned_world_model as module

        monkeypatch.setattr(module, "_model_path", lambda: tmp_path / "vrnn.npz")
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        rng = np.random.default_rng(2)
        for _ in range(40):
            model.observe(rng.normal(size=6), rng.normal(size=2), learn=True)
        model.train_now(passes=5)
        assert model.save()
        assert (tmp_path / "vrnn.npz").exists()

        restored = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        assert np.allclose(restored.W_enc, model.W_enc)
        assert np.allclose(restored.h, model.h)
        assert restored._train_steps == model._train_steps

    def test_a_dimension_mismatch_is_refused(self, tmp_path, monkeypatch):
        import core.world_model.learned_world_model as module

        monkeypatch.setattr(module, "_model_path", lambda: tmp_path / "vrnn.npz")
        LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        ).save()
        other = LearnedWorldModel(
            WorldModelConfig(observation_dim=10, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        assert other.config.observation_dim == 10, "a mismatched checkpoint must not be coerced"

    def test_status_reports_what_is_actually_trainable(self):
        model = LearnedWorldModel(
            WorldModelConfig(observation_dim=6, latent_dim=4, hidden_dim=8, action_dim=2, seed=4)
        )
        status = model.get_status()
        expected = sum(getattr(model, n).size for n in _TRAINABLE)
        assert status["trainable_parameters"] == expected
        assert status["train_steps"] == 0
