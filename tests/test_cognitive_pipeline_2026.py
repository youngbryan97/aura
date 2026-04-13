################################################################################

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agency_core import AgencyCore, AgencyState, EngagementMode
from core.cognitive_integration_layer import CognitiveIntegrationLayer
from core.container import ServiceContainer
from core.memory.memory_facade import MemoryFacade


@pytest.fixture(autouse=True)
def cleanup_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()

@pytest.mark.asyncio
async def test_memory_facade_hardening():
    """Verify MemoryFacade Pydantic status and setup."""
    # Mock sub-systems
    mock_episodic = MagicMock()
    ServiceContainer.register_instance("episodic_memory", mock_episodic)
    
    facade = MemoryFacade()
    facade.setup()
    
    assert facade.episodic == mock_episodic
    
    status = facade.get_status()
    assert status["episodic"] is True
    assert status["semantic"] is False
    assert "last_commit" in status

@pytest.mark.asyncio
async def test_agency_core_pydantic_state():
    """Verify AgencyCore Pydantic state and sync."""
    orch = MagicMock()
    # Mock liquid_state properly to avoid Pydantic float warnings
    mock_ls = MagicMock()
    mock_ls.current.energy = 0.9
    mock_ls.current.curiosity = 0.5
    mock_ls.current.frustration = 0.0
    orch.liquid_state = mock_ls
    
    orch.personality_engine = MagicMock()
    orch.personality_engine.traits = {"extraversion": 0.8}
    
    agency = AgencyCore(orchestrator=orch)
    assert isinstance(agency.state, AgencyState)
    
    # Test Sync
    agency._sync_from_orchestrator()
    assert abs(agency.state.initiative_energy - 0.9) < 0.001
    assert agency.state.social_hunger > 0.3
    
    # Test Telemetry
    status = agency.get_status()
    assert status["engagement_mode"] == EngagementMode.ATTENTIVE_IDLE.value
    assert "pathways_active" in status

@pytest.mark.asyncio
async def test_cognitive_integration_segments():
    """Verify CognitiveIntegrationLayer initialization and greeting fast-path."""
    orch = MagicMock()
    orch.cognitive_engine = MagicMock()
    
    # Register dependencies that initialize() resolves
    mock_kernel = AsyncMock()
    mock_monologue = AsyncMock()
    mock_language = AsyncMock()
    mock_synth = AsyncMock()
    
    # Setup mock_kernel.evaluate to return a CognitiveBrief
    from core.cognitive_kernel import CognitiveBrief
    mock_kernel.evaluate = AsyncMock(return_value=CognitiveBrief(
        key_points=["Hello."],
        conviction=0.5
    ))
    
    # Setup mock_monologue.think to return a ThoughtPacket
    from core.inner_monologue import ThoughtPacket
    mock_monologue.think = AsyncMock(return_value=ThoughtPacket(
        stance="Hello.",
        primary_points=["Hello."]
    ))
    
    # Setup mock_language.express to return a string
    mock_language.express = AsyncMock(return_value="Hello.")
    
    ServiceContainer.register_instance("cognitive_kernel", mock_kernel)
    ServiceContainer.register_instance("inner_monologue", mock_monologue)
    ServiceContainer.register_instance("language_center", mock_language)
    ServiceContainer.register_instance("memory_synthesizer", mock_synth)
    
    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()
    
    assert cognition.is_active is True
    assert cognition.kernel == mock_kernel
    
    # Test greeting fast-path (doesn't need LLM)
    response = await cognition.process_turn("hello")
    assert response == "Hello."


@pytest.mark.asyncio
async def test_cognitive_integration_threads_history_into_reasoning_pipeline(monkeypatch):
    orch = MagicMock()
    orch.cognitive_engine = MagicMock()

    from core.cognitive_kernel import CognitiveBrief
    from core.inner_monologue import ThoughtPacket

    mock_kernel = AsyncMock()
    mock_kernel.evaluate = AsyncMock(return_value=CognitiveBrief(key_points=["Depth."], conviction=0.7))
    mock_monologue = AsyncMock()
    mock_monologue.think = AsyncMock(return_value=ThoughtPacket(stance="Depth.", primary_points=["Depth."]))
    mock_language = AsyncMock()
    mock_language.express = AsyncMock(return_value="Depth.")

    ServiceContainer.register_instance("cognitive_kernel", mock_kernel)
    ServiceContainer.register_instance("inner_monologue", mock_monologue)
    ServiceContainer.register_instance("language_center", mock_language)

    monkeypatch.setattr("core.cognitive_integration_layer.get_reflex", lambda: MagicMock(process=lambda _msg: None))

    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()

    context = {
        "history": [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Later"},
        ]
    }
    response = await cognition.process_turn("Let's go deeper.", context=context)

    assert response == "Depth."
    assert mock_kernel.evaluate.await_args.kwargs["history"] == context["history"]
    assert mock_monologue.think.await_args.kwargs["history"] == context["history"]
    assert mock_language.express.await_args.kwargs["history"] == context["history"]

if __name__ == "__main__":
    pytest.main([__file__])


##
