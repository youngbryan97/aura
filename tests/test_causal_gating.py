import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.brain.inference_gate import InferenceGate
from core.consciousness.attention_schema import AttentionSchema, AttentionalFocus
from core.consciousness.free_energy import FreeEnergyEngine, FreeEnergyState


# ==============================================================================
# InferenceGate Phi Gating Tests
# ==============================================================================

def test_inference_gate_get_system_phi_redundancy():
    """Verify that _get_system_phi correctly probes the redundant sources."""
    # 1. Test ClosedCausalLoop _loop_state fallback
    mock_loop = MagicMock()
    mock_loop._loop_state = MagicMock()
    mock_loop._loop_state.phi_estimate = 0.75
    
    with patch("core.container.ServiceContainer.get", side_effect=lambda name, default=None: mock_loop if name == "closed_causal_loop" else default):
        phi = InferenceGate._get_system_phi()
        assert phi == 0.75

    # 2. Test PhiComputer fallback
    mock_pc = MagicMock()
    mock_pc.latest_phi = 0.45
    
    with patch("core.container.ServiceContainer.get", return_value=None), \
         patch("core.consciousness.phi_compute.get_phi_computer", return_value=mock_pc):
        phi = InferenceGate._get_system_phi()
        assert phi == 0.45

    # 3. Test PhiCore fallback
    mock_phi_core = MagicMock()
    mock_phi_core._last_result = MagicMock()
    mock_phi_core._last_result.phi_s = 0.25
    
    with patch("core.container.ServiceContainer.get", side_effect=lambda name, default=None: mock_phi_core if name == "phi_core" else default), \
         patch("core.consciousness.phi_compute.get_phi_computer", return_value=None):
        phi = InferenceGate._get_system_phi()
        assert phi == 0.25

    # 4. Test neutral fallback
    with patch("core.container.ServiceContainer.get", return_value=None), \
         patch("core.consciousness.phi_compute.get_phi_computer", return_value=None):
        phi = InferenceGate._get_system_phi()
        assert phi == 0.5


def test_inference_gate_adaptive_max_tokens_phi_scaling():
    """Verify that _adaptive_max_tokens_for_prompt scales the token budget based on system Phi."""
    # Ensure it only scales user-facing primary tier requests
    
    # Under high Phi (e.g. phi = 0.5 -> scale = 0.6 + 1.0 = 1.6)
    with patch.object(InferenceGate, "_get_system_phi", return_value=0.5):
        adapted = InferenceGate._adaptive_max_tokens_for_prompt(
            prompt="Hello",
            base_tokens=1000,
            origin="user",
            requested_tier="primary",
            is_background=False
        )
        # Expected adapted tokens: 1000 * 1.6 = 1600
        assert 1500 <= adapted <= 1700

    # Under low Phi (e.g. phi = 0.0 -> scale = max(0.5, 0.6 + 0) = 0.6)
    with patch.object(InferenceGate, "_get_system_phi", return_value=0.0):
        adapted = InferenceGate._adaptive_max_tokens_for_prompt(
            prompt="Hello",
            base_tokens=1000,
            origin="user",
            requested_tier="primary",
            is_background=False
        )
        # Expected adapted tokens: 1000 * 0.6 = 600
        assert 550 <= adapted <= 650

    # Under nominal Phi (e.g. phi = 0.2 -> scale = 0.6 + 0.4 = 1.0)
    with patch.object(InferenceGate, "_get_system_phi", return_value=0.2):
        adapted = InferenceGate._adaptive_max_tokens_for_prompt(
            prompt="Hello",
            base_tokens=1000,
            origin="user",
            requested_tier="primary",
            is_background=False
        )
        # Expected adapted tokens: 1000 * 1.0 = 1000
        assert 950 <= adapted <= 1050


# ==============================================================================
# AttentionSchema Free Energy Gating Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_attention_schema_free_energy_gating_unrestricted():
    """Verify that focus shifts occur normally under low Free Energy (F <= 0.6)."""
    schema = AttentionSchema()
    
    # Establish initial focus
    initial_focus = await schema.set_focus(
        content="Analyzing the neural mesh",
        source="curiosity",
        priority=0.4
    )
    assert schema.current_focus.source == "curiosity"
    
    # Under low Free Energy (e.g. F = 0.2), shifting to a different source should succeed
    mock_fe_state = FreeEnergyState(
        surprise=0.1,
        complexity=0.1,
        free_energy=0.2,
        valence=0.6,
        arousal=0.2,
        dominant_action="explore"
    )
    mock_fe_engine = MagicMock()
    mock_fe_engine.current = mock_fe_state
    
    with patch("core.consciousness.free_energy.get_free_energy_engine", return_value=mock_fe_engine):
        new_focus = await schema.set_focus(
            content="Responding to query",
            source="affective_steering",
            priority=0.3
        )
        assert schema.current_focus.source == "affective_steering"
        assert schema.current_focus.content == "Responding to query"


@pytest.mark.asyncio
async def test_attention_schema_free_energy_gating_rigidity():
    """Verify focus stability/rigidity is enforced under high Free Energy (F > 0.6)."""
    schema = AttentionSchema()
    
    # Establish initial focus
    await schema.set_focus(
        content="Analyzing the neural mesh",
        source="curiosity",
        priority=0.5
    )
    assert schema.current_focus.source == "curiosity"
    
    # Under high Free Energy (e.g. F = 0.8), shift is blocked if priority is too low
    mock_fe_state = FreeEnergyState(
        surprise=0.8,
        complexity=0.8,
        free_energy=0.8,
        valence=-0.6,
        arousal=0.8,
        dominant_action="update_beliefs"
    )
    mock_fe_engine = MagicMock()
    mock_fe_engine.current = mock_fe_state
    
    with patch("core.consciousness.free_energy.get_free_energy_engine", return_value=mock_fe_engine):
        # Shift with low priority (0.4) is below the rigidity threshold (0.3 + 0.8 * 0.4 = 0.62)
        blocked_focus = await schema.set_focus(
            content="Responding to query",
            source="affective_steering",
            priority=0.4
        )
        # Should retain the original focus
        assert schema.current_focus.source == "curiosity"
        assert schema.current_focus.content == "Analyzing the neural mesh"
        assert blocked_focus == schema.current_focus

        # Shift with same source is NOT blocked
        same_source_focus = await schema.set_focus(
            content="Deepening neural mesh analysis",
            source="curiosity",
            priority=0.1
        )
        assert schema.current_focus.source == "curiosity"
        assert schema.current_focus.content == "Deepening neural mesh analysis"

        # Shift with extremely high priority (0.7) exceeding threshold (0.62) should be allowed
        high_priority_focus = await schema.set_focus(
            content="Emergency shutdown response",
            source="safety_governor",
            priority=0.7
        )
        assert schema.current_focus.source == "safety_governor"
        assert schema.current_focus.content == "Emergency shutdown response"
