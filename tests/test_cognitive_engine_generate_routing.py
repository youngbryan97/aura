from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.brain.cognitive_engine import CognitiveEngine


@pytest.mark.asyncio
async def test_generate_defaults_system_calls_to_background_tertiary(monkeypatch):
    router = SimpleNamespace(think=AsyncMock(return_value="ok"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    monkeypatch.setattr("core.brain.cognitive_engine.get_container", lambda: container)

    engine = CognitiveEngine()
    await engine.generate("Summarize this internal state.", use_strategies=False)

    _, kwargs = router.think.await_args
    assert kwargs["origin"] == "system"
    assert kwargs["is_background"] is True
    assert kwargs["prefer_tier"] == "tertiary"


@pytest.mark.asyncio
async def test_generate_preserves_user_facing_expression_calls(monkeypatch):
    router = SimpleNamespace(think=AsyncMock(return_value="ok"))
    container = SimpleNamespace(get=lambda name, default=None: router if name == "llm_router" else default)
    monkeypatch.setattr("core.brain.cognitive_engine.get_container", lambda: container)

    engine = CognitiveEngine()
    await engine.generate(
        "Reply to the user naturally.",
        use_strategies=False,
        purpose="expression",
    )

    _, kwargs = router.think.await_args
    assert kwargs["origin"] == "system"
    assert kwargs["is_background"] is False
    assert "prefer_tier" not in kwargs
