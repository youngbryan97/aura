"""Search and imagination must roll the same observed state through each action."""

from copy import deepcopy

import numpy as np
import pytest

from core.cognition.mcts_world_model import LearnedMCTSPlanner, MCTSNode
from core.world_model import learned_world_model as dynamics
from core.world_model.unified_world_model import UnifiedWorldModel


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(dynamics, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(dynamics, "_MODEL_PATH", tmp_path / "vrnn.npz")
    return dynamics.LearnedWorldModel(dynamics.WorldModelConfig(
        observation_dim=3, latent_dim=2, hidden_dim=4, action_dim=2,
    ))


def test_first_prediction_depends_on_supplied_observation(model):
    action = np.array([0.4, -0.2])
    first = model.imagine(np.zeros(3), [action])[0].predicted_state
    model._rng = np.random.default_rng(42)
    second = model.imagine(np.ones(3), [action])[0].predicted_state
    # Resetting the random stream is unnecessary for deterministic rollouts.
    model._rng = np.random.default_rng(42)
    control = model.imagine(np.zeros(3), [action])[0].predicted_state
    assert not np.allclose(control, second)
    assert np.allclose(first, control)


def test_first_prediction_depends_on_immediate_action(model):
    observation = np.array([0.2, 0.3, -0.1])
    model._rng = np.random.default_rng(17)
    left = model.imagine(observation, [np.array([1.0, 0.0])])[0]
    model._rng = np.random.default_rng(17)
    right = model.imagine(observation, [np.array([0.0, 1.0])])[0]
    assert not np.allclose(left.predicted_state, right.predicted_state)


def test_rollout_has_no_fictitious_root_action(model):
    observation = np.array([0.3, -0.5, 0.2])
    latent, hidden = model.rollout_start(observation)
    expected, _ = np.split(model.W_enc @ np.concatenate([observation, model.h]) + model.b_enc, 2)
    np.testing.assert_allclose(latent, expected)
    np.testing.assert_array_equal(hidden, model.h)
    assert not np.shares_memory(hidden, model.h)


def test_step_matches_observation_training_chronology(model):
    latent, hidden = model.rollout_start(np.array([0.3, -0.5, 0.2]))
    action = np.array([0.2, 0.6], dtype=np.float32)
    prediction, next_hidden = model.rollout_step(latent, hidden, action)
    expected_hidden = model._gru_step(np.concatenate([latent, action]), hidden)
    prior, _ = np.split(model.W_prior @ expected_hidden + model.b_prior, 2)
    expected_observation = np.tanh(model.W_dec @ np.concatenate([prior, expected_hidden]) + model.b_dec)
    np.testing.assert_allclose(next_hidden, expected_hidden)
    np.testing.assert_allclose(prediction.predicted_state, expected_observation)


def test_tree_and_trajectory_compose_identical_transitions(model):
    actions = [np.array([0.4, -0.2]), np.array([-0.1, 0.7])]
    observation = np.array([0.1, 0.6, -0.4])
    planner = LearnedMCTSPlanner(model, actions, lambda h: float(h.sum()))
    latent, hidden = planner._encode_root(observation)
    node = MCTSNode(latent, hidden)
    trajectory = model.imagine(observation, actions)
    for index, prediction in enumerate(trajectory):
        planner._expand(node)
        node = node.children[index]
        np.testing.assert_allclose(node.latent_state, prediction.latent_mean)
        decoded = np.tanh(model.W_dec @ np.concatenate([node.latent_state, node.hidden_state]) + model.b_dec)
        np.testing.assert_allclose(decoded, prediction.predicted_state)


def test_planning_does_not_mutate_live_state_or_learning_randomness(model):
    hidden = model.h.copy()
    rng_state = deepcopy(model._rng.bit_generator.state)
    actions = [np.ones(2), -np.ones(2)]
    model.imagine(np.ones(3), actions)
    LearnedMCTSPlanner(model, actions, lambda h: float(h.sum()), num_simulations=8).plan(np.ones(3))
    np.testing.assert_array_equal(model.h, hidden)
    assert model._rng.bit_generator.state == rng_state
    assert model._step_count == 0
    assert not model._replay


def test_facade_retains_predicted_observation(model):
    facade = UnifiedWorldModel(learned=model)
    answer = facade.query("imagine", observation=np.ones(3), action_sequence=[np.ones(2)])
    assert answer["facet"] == "learned"
    assert len(answer["result"][0]["predicted_state"]) == 3
    assert answer["result"][0]["prediction_kind"] == "action_rollout"
    assert answer["result"][0]["confidence_calibrated"] is False


def test_typed_query_does_not_resolve_unrelated_neural_facet():
    class State:
        cells = ((1,),)

        def as_text(self):
            return "1"

    class Rules:
        def expect(self, state, action):
            return state

    facade = UnifiedWorldModel(rules=Rules())
    result = facade.query("imagine", observation=State(), action_sequence=["stay"])
    assert result["facet"] == "rules"
    assert result["available"] is True
    assert facade._learned is None
