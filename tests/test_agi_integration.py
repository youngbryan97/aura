"""tests/test_agi_integration.py
================================
Tests for the AGIIntegrationLayer coordinator — tick execution,
telemetry aggregation, modulation retrieval, inference callbacks,
and graceful degradation.
"""

import asyncio
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from core.agi.agi_integration import AGIIntegrationLayer
from core.brain.homeostatic_modulator import InferenceModulation
from core.container import ServiceContainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_layer():
    """Create a fresh AGIIntegrationLayer for each test."""
    layer = AGIIntegrationLayer()
    yield layer


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


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_integration_layer_initializes_cleanly(integration_layer):
    assert integration_layer.tick_count == 0
    assert integration_layer.last_tick_time == 0.0
    assert not integration_layer._running
    assert integration_layer.modulator is not None
    assert integration_layer.feedback_loop is not None


# ---------------------------------------------------------------------------
# Single tick execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tick_increments_counter(integration_layer):
    """A single tick should increment tick_count and update last_tick_time."""
    # Patch ServiceContainer.get so no real services are touched
    with patch.object(ServiceContainer, "get", return_value=None):
        await integration_layer._run_tick()

    assert integration_layer.tick_count == 1
    assert integration_layer.last_tick_time > 0.0


@pytest.mark.asyncio
async def test_run_tick_steps_precision_engine(integration_layer):
    """Tick should call precision.step() when the engine is registered."""
    mock_precision = MagicMock()

    def lookup(name, default=None):
        if name == "precision_engine":
            return mock_precision
        return default

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        await integration_layer._run_tick()

    mock_precision.step.assert_called_once()


@pytest.mark.asyncio
async def test_run_tick_contraction_every_30_ticks(integration_layer):
    """Dimensional contraction should trigger every 30 ticks."""
    mock_expansion = MagicMock()
    mock_expansion.evaluate_contraction.return_value = []

    def lookup(name, default=None):
        if name == "dimensional_expansion":
            return mock_expansion
        return default

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        # Run 29 ticks — no contraction yet
        for _ in range(29):
            await integration_layer._run_tick()
        mock_expansion.evaluate_contraction.assert_not_called()

        # 30th tick triggers contraction
        await integration_layer._run_tick()
        mock_expansion.evaluate_contraction.assert_called_once()


@pytest.mark.asyncio
async def test_run_tick_saves_projection_every_300s(integration_layer):
    """Projection weights save should trigger after 300 seconds elapse."""
    # Force last_save_time into the past
    integration_layer.last_save_time = time.time() - 301.0

    mock_projection = MagicMock()
    integration_layer.modulator.projection = mock_projection

    with patch.object(ServiceContainer, "get", return_value=None):
        await integration_layer._run_tick()

    mock_projection.save.assert_called_once()


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def test_on_inference_complete_returns_metrics(integration_layer, base_modulation):
    """Callback should return surprise/coherence dict."""
    mock_substrate = MagicMock()
    mock_substrate.idx_valence = 0
    mock_substrate.idx_arousal = 1
    mock_substrate.x = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    def lookup(name, default=None):
        if name == "liquid_substrate":
            return mock_substrate
        return default

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        metrics = integration_layer.on_inference_complete(
            output_text="test output",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.15],
            modulation=base_modulation,
        )

    assert "surprise" in metrics
    assert "coherence" in metrics
    assert isinstance(metrics["surprise"], float)
    assert isinstance(metrics["coherence"], float)


def test_on_inference_complete_graceful_degradation(integration_layer, base_modulation):
    """If feedback loop throws, callback should return safe defaults."""
    with patch.object(
        integration_layer.feedback_loop,
        "process_output",
        side_effect=RuntimeError("boom"),
    ):
        metrics = integration_layer.on_inference_complete(
            output_text="test",
            token_ids=[1],
            logprobs=None,
            modulation=base_modulation,
        )

    assert metrics == {"surprise": 0.5, "coherence": 0.0}


# ---------------------------------------------------------------------------
# Modulation retrieval
# ---------------------------------------------------------------------------

def test_get_modulation_returns_inference_modulation(integration_layer):
    """get_modulation should return an InferenceModulation dataclass."""
    with patch.object(ServiceContainer, "get", return_value=None):
        mod = integration_layer.get_modulation()

    assert isinstance(mod, InferenceModulation)
    assert 0.0 < mod.temperature <= 1.5
    assert 0.0 < mod.top_p <= 1.0


def test_get_modulation_graceful_degradation(integration_layer):
    """If modulator throws, get_modulation should return safe defaults."""
    with patch.object(
        integration_layer.modulator,
        "compute_modulation",
        side_effect=RuntimeError("broken"),
    ):
        mod = integration_layer.get_modulation()

    assert isinstance(mod, InferenceModulation)
    assert mod.temperature == 0.7
    assert mod.top_p == 0.9


# ---------------------------------------------------------------------------
# Telemetry aggregation
# ---------------------------------------------------------------------------

def test_get_unified_telemetry_minimal(integration_layer):
    """Telemetry should include integration block even with no services."""
    with patch.object(ServiceContainer, "get", return_value=None):
        telemetry = integration_layer.get_unified_telemetry()

    assert "integration" in telemetry
    assert telemetry["integration"]["ticks"] == 0
    assert "uptime_seconds" in telemetry["integration"]


def test_get_unified_telemetry_with_services(integration_layer):
    """Telemetry should aggregate data from all registered services."""
    mock_precision = MagicMock()
    mock_precision.get_state_dict.return_value = {"arousal": 0.6, "fatigue": 0.2}

    mock_substrate = MagicMock()
    mock_substrate.idx_valence = 0
    mock_substrate.idx_arousal = 1
    mock_substrate.idx_frustration = 2
    mock_substrate.idx_curiosity = 3
    mock_substrate.idx_focus = 4
    mock_substrate.x = np.array([0.7, 0.5, 0.1, 0.3, 0.8], dtype=np.float64)

    mock_free_energy = MagicMock()
    mock_free_energy.smoothed_fe = 0.42
    mock_free_energy.current_action = "explore"

    mock_expansion = MagicMock()
    mock_expansion.get_status.return_value = {"current_dim": 18, "expanded_count": 2}

    mock_registry = MagicMock()
    mock_registry.synthesized_actuators = {"a": 1}
    mock_registry.actuators = {"a": 1, "b": 2, "c": 3}

    def lookup(name, default=None):
        mapping = {
            "precision_engine": mock_precision,
            "liquid_substrate": mock_substrate,
            "free_energy_engine": mock_free_energy,
            "dimensional_expansion": mock_expansion,
            "actuator_registry": mock_registry,
        }
        return mapping.get(name, default)

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        telemetry = integration_layer.get_unified_telemetry()

    assert "precision" in telemetry
    assert telemetry["precision"]["arousal"] == 0.6

    assert "substrate" in telemetry
    assert telemetry["substrate"]["valence"] == 0.7
    assert telemetry["substrate"]["frustration"] == 0.1

    assert "free_energy" in telemetry
    assert telemetry["free_energy"]["smoothed_free_energy"] == 0.42

    assert "dimensional_expansion" in telemetry
    assert telemetry["dimensional_expansion"]["current_dim"] == 18
    assert telemetry["dimensional_expansion"]["expanded_count"] == 2

    assert "actuators" in telemetry
    assert telemetry["actuators"]["synthesized_count"] == 1
    assert telemetry["actuators"]["total_count"] == 3


# ---------------------------------------------------------------------------
# Tick loop resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_tick_survives_precision_engine_crash(integration_layer):
    """If PrecisionEngine.step() throws, the tick should complete gracefully."""
    mock_precision = MagicMock()
    mock_precision.step.side_effect = RuntimeError("FHN divergence")

    def lookup(name, default=None):
        if name == "precision_engine":
            return mock_precision
        return default

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        # Should NOT raise
        await integration_layer._run_tick()

    assert integration_layer.tick_count == 1


@pytest.mark.asyncio
async def test_run_tick_survives_expansion_crash(integration_layer):
    """If evaluate_contraction() throws, the tick should complete gracefully."""
    mock_expansion = MagicMock()
    mock_expansion.evaluate_contraction.side_effect = ValueError("matrix singular")

    def lookup(name, default=None):
        if name == "dimensional_expansion":
            return mock_expansion
        return default

    # Force tick_count to 29 so the 30th tick triggers contraction
    integration_layer.tick_count = 29

    with patch.object(ServiceContainer, "get", side_effect=lookup):
        await integration_layer._run_tick()

    assert integration_layer.tick_count == 30
    mock_expansion.evaluate_contraction.assert_called_once()


# ---------------------------------------------------------------------------
# Start / Stop lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_stop_lifecycle(integration_layer):
    """Start should set _running, stop should clear it and save projection."""
    mock_tracker = MagicMock()
    mock_task = MagicMock()
    mock_tracker.create_task.return_value = mock_task

    mock_projection = MagicMock()
    integration_layer.modulator.projection = mock_projection

    with patch("core.utils.task_tracker.get_task_tracker", return_value=mock_tracker):
        with patch.object(ServiceContainer, "register"):
            await integration_layer.start()

    assert integration_layer._running is True
    mock_tracker.create_task.assert_called_once()

    await integration_layer.stop()
    assert integration_layer._running is False
    mock_task.cancel.assert_called_once()
    mock_projection.save.assert_called_once()


@pytest.mark.asyncio
async def test_double_start_is_idempotent(integration_layer):
    """Calling start() twice should not spawn a second loop."""
    mock_tracker = MagicMock()
    mock_tracker.create_task.return_value = MagicMock()

    with patch("core.utils.task_tracker.get_task_tracker", return_value=mock_tracker):
        with patch.object(ServiceContainer, "register"):
            await integration_layer.start()
            await integration_layer.start()

    assert mock_tracker.create_task.call_count == 1
    await integration_layer.stop()
