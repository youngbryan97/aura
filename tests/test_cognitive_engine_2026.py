################################################################################

"""tests/test_cognitive_engine_2026.py
A-Tier verification for the hardened Cognitive Engine.
Updated to match the modular-phase facade API.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from core.brain.cognitive_engine import CognitiveEngine
from core.brain.types import ThinkingMode, Thought
from core.exceptions import AuraError

class MockBackend:
    """Mock implementation of CognitiveBackend protocol."""
    async def generate(self, prompt, system_prompt, options=None):
        return '{"content": "I am thinking clearly.", "reasoning": ["Step 1"], "confidence": 0.9}'

    async def chat_stream_async(self, messages):
        yield "Thinking..."
        yield "Done."

    async def check_health_async(self):
        return True

@pytest.fixture
def engine():
    backend = MockBackend()
    engine = CognitiveEngine(backend=backend)
    # Mocking internal components to avoid full system dependency
    engine.knowledge_graph = MagicMock()
    engine.knowledge_graph.search_knowledge.return_value = []
    engine.autonomous_brain = MagicMock()
    return engine

@pytest.mark.asyncio
async def test_engine_initialization(engine):
    assert engine.backend is not None
    assert len(engine.thoughts) == 0

@pytest.mark.asyncio
async def test_engine_setup(engine):
    # Mock ServiceContainer to prevent full system boot
    with patch("core.container.ServiceContainer") as mock_container_cls:
        mock_container = mock_container_cls.return_value
        mock_container.get.return_value = None
        
        engine.setup()
        assert engine.autonomous_brain is not None

@pytest.mark.asyncio
async def test_engine_think_no_response(engine):
    """Test think() when no assistant response is generated."""
    mock_state = MagicMock()
    mock_state.cognition.working_memory = []
    mock_state.derive.return_value = mock_state

    mock_repo = AsyncMock()
    mock_repo.get_current.return_value = mock_state
    mock_repo.commit = AsyncMock()
    engine.state_repository = mock_repo
    engine._phases = []  # No phases — no response generated

    thought = await engine.think("Hello")
    assert isinstance(thought, Thought)
    assert thought.confidence == 0.5
    assert "processing" in thought.content.lower() or "modular" in thought.content.lower()

@pytest.mark.asyncio
async def test_engine_health_check(engine):
    health = await engine.check_health()
    assert health["status"] == "healthy"
    assert health["modular"] is True
    assert "phases_count" in health

@pytest.mark.asyncio
async def test_reactive_recovery_does_not_hold_lock_while_rollback_runs(engine):
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()

    async def _rollback(_reason):
        rollback_started.set()
        await release_rollback.wait()

    engine.state_repository = AsyncMock()
    engine.state_repository.rollback.side_effect = _rollback

    first_recovery = asyncio.create_task(
        engine._reactive_recovery("Hello", ThinkingMode.FAST, "api", "test-failure")
    )

    await asyncio.wait_for(rollback_started.wait(), timeout=1.0)

    second = await asyncio.wait_for(
        engine._reactive_recovery("Hello again", ThinkingMode.FAST, "api", "test-failure-2"),
        timeout=1.0,
    )

    assert "still recovering" in second.content.lower()

    release_rollback.set()
    first = await asyncio.wait_for(first_recovery, timeout=1.0)
    assert isinstance(first, Thought)

@pytest.mark.asyncio
async def test_engine_think_stream(engine):
    """Test think_stream() using mocked router."""
    mock_state = MagicMock()
    mock_state.cognition.working_memory = []

    mock_repo = AsyncMock()
    mock_repo.get_current.return_value = mock_state
    engine.state_repository = mock_repo

    async def fake_stream(**kwargs):
        for token in ["Thinking...", "Done."]:
            yield MagicMock(content=token)

    mock_router = MagicMock()
    mock_router.think_stream = fake_stream

    from core.container import ServiceContainer
    ServiceContainer.register_instance("llm_router", mock_router)
    try:
        with patch("core.brain.llm.context_assembler.ContextAssembler.build_messages",
                   return_value=[{"role": "user", "content": "Stream test"}]):
            tokens = []
            async for token in engine.think_stream("Stream test"):
                tokens.append(token)

            assert len(tokens) > 0
            assert "Thinking..." in tokens
    finally:
        try:
            ServiceContainer._services.pop("llm_router", None)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_record_interaction_prefers_context_manager():
    engine = CognitiveEngine()
    context_manager = SimpleNamespace(record_interaction=AsyncMock())
    learning_engine = SimpleNamespace(record_interaction=AsyncMock())
    fake_container = SimpleNamespace(
        get=lambda name, default=None: (
            context_manager if name == "context_manager"
            else learning_engine if name == "learning_engine"
            else default
        )
    )

    with patch("core.brain.cognitive_engine.get_container", return_value=fake_container):
        await engine.record_interaction("Hi", "Hey there", domain="relational")

    context_manager.record_interaction.assert_awaited_once_with("Hi", "Hey there", domain="relational")
    learning_engine.record_interaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_interaction_falls_back_to_learning_engine():
    engine = CognitiveEngine()
    context_manager = SimpleNamespace(record_interaction=AsyncMock(side_effect=RuntimeError("boom")))
    learning_engine = SimpleNamespace(record_interaction=AsyncMock())
    fake_container = SimpleNamespace(
        get=lambda name, default=None: (
            context_manager if name == "context_manager"
            else learning_engine if name == "learning_engine"
            else default
        )
    )

    with patch("core.brain.cognitive_engine.get_container", return_value=fake_container):
        await engine.record_interaction("Hi", "Hey there", domain="relational")

    learning_engine.record_interaction.assert_awaited_once_with(
        user_input="Hi",
        aura_response="Hey there",
        domain="relational",
    )
