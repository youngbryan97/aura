import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.brain.homeostatic_modulator import (
    HomeostaticModulator,
    SubstrateLogitProjection,
    InferenceModulation,
)
from core.container import ServiceContainer


@pytest.fixture
def temp_project_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_inference_modulation_dataclass():
    head_weights = np.ones(32, dtype=np.float32)
    mod = InferenceModulation(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        logit_bias={101: 0.5},
        head_weights=head_weights,
        urgency=0.2,
    )
    assert mod.temperature == 0.7
    assert mod.top_p == 0.9
    assert mod.repetition_penalty == 1.1
    assert mod.logit_bias == {101: 0.5}
    assert np.allclose(mod.head_weights, head_weights)
    assert mod.urgency == 0.2


def test_substrate_logit_projection_init_and_paths(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj = SubstrateLogitProjection(substrate_dim=128, save_path=str(save_path))
    assert proj.substrate_dim == 128
    assert proj.weights == {}
    assert proj.save_path == save_path


def test_substrate_logit_projection_get_biases_empty():
    proj = SubstrateLogitProjection(substrate_dim=128, save_path=None)
    state = np.random.rand(128).astype(np.float32)
    assert proj.get_biases(state) == {}


def test_substrate_logit_projection_learn_step_and_get_biases(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj = SubstrateLogitProjection(substrate_dim=4, save_path=str(save_path))

    state = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    proj.learn_step(
        substrate_state=state,
        token_ids=[10, 20],
        feedback_coherence=1.0,
        surprise=0.0,
        lr=0.5,
    )

    # Weights for token 10 and 20 should be updated
    # reward = coherence * exp(-surprise) = 1.0 * 1.0 = 1.0
    # dw = lr * reward * state = 0.5 * 1.0 * [1.0, 0.0, -1.0, 0.5] = [0.5, 0.0, -0.5, 0.25]
    # after decay: w * 0.98 = [0.49, 0.0, -0.49, 0.245]
    assert 10 in proj.weights
    assert 20 in proj.weights
    assert np.allclose(proj.weights[10], [0.49, 0.0, -0.49, 0.245], atol=1e-5)

    biases = proj.get_biases(state)
    # dot = [0.49, 0.0, -0.49, 0.245] . [1.0, 0.0, -1.0, 0.5]
    # dot = 0.49 + 0 + 0.49 + 0.1225 = 1.1025
    # bias = clip(dot * 0.5, -2, 2) = clip(0.55125, -2, 2) = 0.55125
    assert abs(biases[10] - 0.55125) < 1e-4
    assert abs(biases[20] - 0.55125) < 1e-4


def test_substrate_logit_projection_dimension_mismatch(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj = SubstrateLogitProjection(substrate_dim=4, save_path=str(save_path))

    # Test with too long state
    state = np.array([1.0, 0.0, -1.0, 0.5, 999.0], dtype=np.float32)
    proj.learn_step(
        substrate_state=state,
        token_ids=[5],
        feedback_coherence=1.0,
        surprise=0.0,
        lr=0.5,
    )
    assert proj.weights[5].shape == (4,)
    assert np.allclose(proj.weights[5], [0.49, 0.0, -0.49, 0.245])

    # Test get_biases with too short state
    short_state = np.array([1.0, 0.0], dtype=np.float32)
    biases = proj.get_biases(short_state)
    # short_state is padded with zeros: [1.0, 0.0, 0.0, 0.0]
    # dot = [0.49, 0.0, -0.49, 0.245] . [1.0, 0.0, 0.0, 0.0] = 0.49
    # bias = clip(0.49 * 0.5, -2, 2) = 0.245
    assert abs(biases[5] - 0.245) < 1e-4


def test_substrate_logit_projection_save_load(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj1 = SubstrateLogitProjection(substrate_dim=4, save_path=str(save_path))
    state = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    proj1.learn_step(state, [100], 1.0, 0.0, lr=1.0)
    proj1.save()

    assert save_path.exists()

    proj2 = SubstrateLogitProjection(substrate_dim=4, save_path=str(save_path))
    assert 100 in proj2.weights
    assert np.allclose(proj1.weights[100], proj2.weights[100])


def test_substrate_logit_projection_pruning(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj = SubstrateLogitProjection(substrate_dim=4, save_path=str(save_path))

    # Very small update
    state = np.array([1e-5, 0, 0, 0], dtype=np.float32)
    proj.learn_step(state, [7], 1.0, 0.0, lr=1e-5)
    # The weight norm should be below 1e-4, resulting in pruning
    assert 7 not in proj.weights


def test_homeostatic_modulator_no_services():
    # Clear container to ensure fresh state
    with patch.object(ServiceContainer, "get", return_value=None):
        modulator = HomeostaticModulator(substrate_dim=128)
        modulation = modulator.compute_modulation()

        assert modulation.temperature == 0.75  # 0.95 - 0.40 * 0.5
        assert modulation.repetition_penalty == 1.1
        assert modulation.top_p == 0.875  # 0.95 - 0.25 * 0.3
        assert modulation.logit_bias == {}
        assert np.allclose(modulation.head_weights, np.ones(32, dtype=np.float32))
        assert modulation.urgency == 0.5


def test_homeostatic_modulator_with_mocked_services():
    mock_precision = MagicMock()
    mock_precision.fhn.arousal = 0.8
    mock_precision.fhn.fatigue = 0.2
    mock_precision.get_head_weights.return_value = np.linspace(
        0.5, 1.5, 32, dtype=np.float32
    )
    mock_precision.get_temperature.return_value = 0.63

    mock_free_energy = MagicMock()
    mock_free_energy._smoothed_fe = 0.6
    mock_free_energy.get_action_urgency.return_value = 0.75

    mock_substrate = MagicMock()
    mock_substrate.idx_frustration = 3
    mock_substrate.x = np.array([0.1, 0.2, 0.3, 0.5, 0.6], dtype=np.float32)

    def service_lookup(name, default=None):
        if name == "precision_engine":
            return mock_precision
        if name == "free_energy_engine":
            return mock_free_energy
        if name == "liquid_substrate":
            return mock_substrate
        return default

    with patch.object(ServiceContainer, "get", side_effect=service_lookup):
        modulator = HomeostaticModulator(substrate_dim=5)
        # Seed weights in projection to see biases
        modulator.projection.weights[42] = np.array(
            [0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32
        )

        modulation = modulator.compute_modulation()

        assert modulation.temperature == 0.63
        # frustration is substrate.x[3] = 0.5. Rep pen = 1.1 + 0.3 * 0.5 = 1.25
        assert modulation.repetition_penalty == 1.25
        # top_p = max(0.6, min(1.0, 0.95 - 0.25 * 0.6)) = 0.8
        assert abs(modulation.top_p - 0.8) < 1e-6
        assert np.allclose(modulation.head_weights, np.linspace(0.5, 1.5, 32))
        assert modulation.urgency == 0.75
        # bias dot product = 0.5 * 0.1 + 0.5 * 0.2 = 0.15
        # bias = clip(dot * 0.5) = 0.075
        assert 42 in modulation.logit_bias
        assert abs(modulation.logit_bias[42] - 0.075) < 1e-4


def test_modulator_projection_stress_test(temp_project_dir):
    save_path = temp_project_dir / "projection.json"
    proj = SubstrateLogitProjection(substrate_dim=32, save_path=str(save_path))

    # Stress test: run Hebbian learning 10,000 times with random inputs and verify stability
    for i in range(1000):
        state = np.random.uniform(-1.0, 1.0, 32).astype(np.float32)
        tokens = list(np.random.randint(1, 100, 5))
        coherence = float(np.random.uniform(-1.0, 1.0))
        surprise = float(np.random.uniform(0.0, 3.0))
        proj.learn_step(
            state, tokens, feedback_coherence=coherence, surprise=surprise, lr=0.01
        )

    # Verify that all weight vectors remain within clipped range [-1, 1]
    for tid, w in proj.weights.items():
        assert np.all(w >= -1.0)
        assert np.all(w <= 1.0)
        assert not np.any(np.isnan(w))
