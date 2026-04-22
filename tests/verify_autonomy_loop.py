import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.orchestrator import RobustOrchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAutonomy")


@pytest.mark.asyncio
async def test_autonomy_loop(monkeypatch):
    """Autonomous thought should fall back to the direct autonomous brain."""
    orchestrator = RobustOrchestrator()
    brain = AsyncMock()
    brain.think.return_value = {
        "content": "I am bored.",
        "tool_calls": [{"name": "web_search", "args": {"query": "news"}}],
    }
    orchestrator.cognitive_engine = SimpleNamespace(autonomous_brain=brain)
    orchestrator.status = SimpleNamespace(start_time=time.time() - 10.0)
    orchestrator._last_thought_time = time.time() - 5.0
    orchestrator._emit_thought_stream = MagicMock()
    orchestrator.execute_tool = AsyncMock()

    monkeypatch.setattr(
        "core.orchestrator.mixins.autonomy.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    await orchestrator._perform_autonomous_thought()

    brain.think.assert_awaited_once()
    kwargs = brain.think.await_args.kwargs
    assert kwargs["objective"] == "Reflect on current state."
    assert kwargs["context"]["boredom_level"] >= 5
    orchestrator.execute_tool.assert_awaited_once_with("web_search", {"query": "news"})
    orchestrator._emit_thought_stream.assert_any_call("...letting my mind wander...")
    orchestrator._emit_thought_stream.assert_any_call("I am bored.")
