"""Memory conditions what her world model predicts, not only how fast it learns.

The first coupling of memory to the world model scaled its step size by the
units where perception and memory meet. A reviewer asked for the prediction
itself, W(t+1) = f(W(t), P(t), M(t), A(t)), with the gain read where copying
memory into the target cannot produce it. Memory now enters beside the action,
in the input that advances the model's hidden state; the observation it
reconstructs is unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.world_model.learned_world_model import LearnedWorldModel, WorldModelConfig
from core.world_model.observe_cycle import (
    ACTION_READINGS,
    MEMORY_CONTEXT_WIDTH,
    action_of,
    conditioning_of,
    memory_context_of,
    observe_cycle,
)


def _state(recalled: str = "the garden in spring", score: float = 0.7, intensity: float = 0.4):
    cognition = SimpleNamespace(
        long_term_memory=[recalled],
        working_memory=[],
        memory_scores=[0.2, score],
        relived={"relived": True, "intensity": intensity},
        active_goals=[],
    )
    world = SimpleNamespace(facts={}, recent_percepts=[])
    return SimpleNamespace(cognition=cognition, world=world)


def test_memory_sits_after_the_action_inside_the_models_action_input():
    state = _state()
    conditioning = conditioning_of(state)
    assert conditioning.shape == (ACTION_READINGS + MEMORY_CONTEXT_WIDTH,)
    assert conditioning.size <= WorldModelConfig().action_dim
    assert np.allclose(conditioning[:ACTION_READINGS], action_of(state))
    assert np.allclose(conditioning[ACTION_READINGS:], memory_context_of(state))
    assert not np.allclose(memory_context_of(state), memory_context_of(_state(recalled="a storm at sea")))


def test_the_same_moment_with_a_different_memory_is_predicted_differently():
    model = LearnedWorldModel(WorldModelConfig(seed=3))
    observation = np.linspace(-0.5, 0.5, 22)
    latent, hidden = model.rollout_start(observation)
    one, _ = model.rollout_step(latent, hidden, conditioning_of(_state()))
    other, _ = model.rollout_step(latent, hidden, conditioning_of(_state(recalled="a storm at sea")))
    assert not np.allclose(one.predicted_state, other.predicted_state)


def test_each_cycle_hands_the_model_her_memory():
    seen = {}

    class Spy:
        def observe(self, observation, action, *, learn=True):
            seen["action"] = np.asarray(action)

        def surprise(self):
            return 0.1

    state = _state()
    state.affect = SimpleNamespace(valence=0.0, arousal=0.5, curiosity=0.5)
    state.soma = SimpleNamespace(hardware={})
    state.motivation = SimpleNamespace(budgets={})
    observe_cycle(state, Spy())
    assert np.allclose(seen["action"][ACTION_READINGS:], memory_context_of(state))


def _held_out_error(mode: str, seed: int, steps: int = 4000) -> float:
    """Next-observation error on the last part of a stream whose next step only memory can tell."""
    rng = np.random.default_rng(seed)
    model = LearnedWorldModel(
        WorldModelConfig(observation_dim=8, latent_dim=4, hidden_dim=24, action_dim=8,
                         seed=seed, replay_buffer_size=400, learning_rate=0.005)
    )
    cues = rng.choice([-1.0, 1.0], steps + 1)
    arriving = rng.uniform(-1, 1, steps + 1)
    stream = np.zeros((steps + 1, 8))
    stream[1:, 0] = arriving[1:]
    stream[1:, 1] = 0.8 * cues[:-1] * arriving[:-1]
    errors = []
    for t in range(steps):
        context = np.zeros(8)
        if mode == "memory":
            context[4] = cues[t]
        elif mode == "shuffled":
            context[4] = cues[rng.integers(0, steps)]
        if t > 0.7 * steps:
            latent, hidden = model.rollout_start(stream[t])
            predicted, _ = model.rollout_step(latent, hidden, context)
            errors.append((predicted.predicted_state[1] - stream[t + 1, 1]) ** 2)
        model.observe(stream[t], context, learn=True)
        if t % 10 == 9 and t < 0.7 * steps:
            model.train_now(passes=4)
    return float(np.mean(errors))


@pytest.mark.parametrize("seed", [0, 1])
def test_a_learned_memory_predicts_what_a_shuffled_one_cannot(seed: int):
    """Measured 21 October: 0.158 and 0.176 with memory, 0.213 and 0.216 shuffled, 0.211 and 0.216 without."""
    with_memory = _held_out_error("memory", seed)
    assert with_memory < _held_out_error("shuffled", seed)
    assert with_memory < _held_out_error("none", seed)
