import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from core.brain.multimodal_orchestrator import MultimodalOrchestrator
from core.container import ServiceContainer

@pytest.mark.asyncio
async def test_multimodal_render_sync():
    # Setup mocks
    mock_voice = MagicMock()
    mock_voice.speak = AsyncMock()
    
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()
    
    # Register in ServiceContainer
    ServiceContainer.register_instance("voice_engine", mock_voice)
    ServiceContainer.register_instance("input_bus", mock_bus)
    
    orchestrator = MultimodalOrchestrator()
    
    # Test rendering a "happy" message
    await orchestrator.render("I am so happy to see you!", metadata={"voice": True})
    
    # Allow background tasks to run
    await asyncio.sleep(0.1)
    
    # Verify voice was called
    mock_voice.speak.assert_called_once_with("I am so happy to see you!")
    
    # Verify expression was pulsed
    args, kwargs = mock_bus.publish.call_args_list[0]
    assert args[0] == "aura/expression"
    assert args[1]["expression"] == "joy"

@pytest.mark.asyncio
async def test_heuristic_expressions():
    orchestrator = MultimodalOrchestrator()
    assert orchestrator._heuristic_expression("I am happy") == "joy"
    assert orchestrator._heuristic_expression("This is an error") == "alert"
    assert orchestrator._heuristic_expression("I am sorry") == "sad"
    assert orchestrator._heuristic_expression("I am neutral") == "neutral"
