import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.brain.inference_feedback import InferenceFeedbackLoop
from core.brain.homeostatic_modulator import InferenceModulation
from core.container import ServiceContainer


@pytest.fixture
def base_modulation():
    return InferenceModulation(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        logit_bias={},
        head_weights=np.ones(32, dtype=np.float32),
        urgency=0.5,
    )


def test_surprise_calculation_with_logprobs(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=10)

    # All highly probable tokens (logprobs close to 0) -> low surprise
    logprobs = [-0.01, -0.05, -0.02]
    metrics = loop.process_output(
        output_text="test message",
        token_ids=[1, 2, 3],
        logprobs=logprobs,
        modulation=base_modulation,
        modulator_projection=None,
    )
    assert metrics["surprise"] < 0.1

    # Highly improbable tokens (large negative logprobs) -> high surprise
    logprobs = [-3.5, -4.0, -3.8]
    metrics = loop.process_output(
        output_text="test message",
        token_ids=[1, 2, 3],
        logprobs=logprobs,
        modulation=base_modulation,
        modulator_projection=None,
    )
    # Clipped at 3.0
    assert metrics["surprise"] == 3.0


def test_surprise_calculation_lexical_fallback(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=10)

    # Empty/repetitive text should yield different surprise than unique text
    metrics_rep = loop.process_output(
        output_text="test test test test",
        token_ids=[1, 1, 1, 1],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )
    metrics_uniq = loop.process_output(
        output_text="this is a unique collection of words",
        token_ids=[1, 2, 3, 4, 5, 6, 7],
        logprobs=None,
        modulation=base_modulation,
        modulator_projection=None,
    )

    # Repetitive text has low unique ratio -> higher surprise fallback score
    assert metrics_rep["surprise"] > metrics_uniq["surprise"]


def test_coherence_calculation_valence_alignment(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    # Mock substrate with positive valence (idx_valence = 0, state[0] = 0.8)
    mock_substrate = MagicMock()
    mock_substrate.idx_valence = 0
    mock_substrate.idx_arousal = 1
    mock_substrate.x = np.array([0.8, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    def service_lookup(name, default=None):
        if name == "liquid_substrate":
            return mock_substrate
        return default

    with patch.object(ServiceContainer, "get", side_effect=service_lookup):
        # 1. Output text with positive valence words (should align -> high coherence)
        metrics_pos = loop.process_output(
            output_text="completed success stable resolved",
            token_ids=[1, 2],
            logprobs=None,
            modulation=base_modulation,
            modulator_projection=None,
        )
        assert metrics_pos["coherence"] > 0.0

        # 2. Output text with negative valence words (should conflict -> low coherence)
        metrics_neg = loop.process_output(
            output_text="failed error danger hazard broken",
            token_ids=[3, 4],
            logprobs=None,
            modulation=base_modulation,
            modulator_projection=None,
        )
        assert metrics_neg["coherence"] < 0.0


def test_engine_feedback_injection(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    mock_substrate = MagicMock()
    mock_substrate.idx_valence = 0
    mock_substrate.idx_arousal = 1
    mock_substrate.x = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    mock_free_energy = MagicMock()
    mock_precision = MagicMock()

    def service_lookup(name, default=None):
        if name == "liquid_substrate":
            return mock_substrate
        if name == "free_energy_engine":
            return mock_free_energy
        if name == "precision_engine":
            return mock_precision
        return default

    with patch.object(ServiceContainer, "get", side_effect=service_lookup):
        loop.process_output(
            output_text="success resolved",
            token_ids=[1, 2],
            logprobs=[-0.1, -0.1],
            modulation=base_modulation,
            modulator_projection=None,
        )

        # Verify free energy received surprise signal
        mock_free_energy.accept_surprise_signal.assert_called_once()
        # Verify liquid substrate received feedback
        mock_substrate.accept_inference_feedback.assert_called_once()
        # Verify precision engine received feedback
        mock_precision.accept_inference_feedback.assert_called_once()


def test_hebbian_projection_updates(base_modulation):
    loop = InferenceFeedbackLoop(substrate_dim=5)

    mock_substrate = MagicMock()
    mock_substrate.idx_valence = 0
    mock_substrate.idx_arousal = 1
    # arousal = substrate.x[1] = 0.5
    mock_substrate.x = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    mock_projection = MagicMock()

    with patch.object(ServiceContainer, "get", return_value=mock_substrate):
        loop.process_output(
            output_text="test resolved",
            token_ids=[42, 43],
            logprobs=[-0.05, -0.05],
            modulation=base_modulation,
            modulator_projection=mock_projection,
        )

        # learning_rate = 0.002 * (1.0 + arousal) = 0.002 * 1.5 = 0.003
        mock_projection.learn_step.assert_called_once()
        args, kwargs = mock_projection.learn_step.call_args
        assert kwargs["lr"] == 0.003
        assert kwargs["token_ids"] == [42, 43]
