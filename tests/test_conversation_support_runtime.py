import asyncio

import pytest

from core.runtime import conversation_support
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_record_conversation_experience_prefers_memory_facade_commit(monkeypatch):
    state = AuraState.default()
    captured = {}

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            captured.update(kwargs)
            return "episode-1"

    def fake_optional_service(*names, default=None):
        if "memory_facade" in names:
            return DummyFacade()
        if "episodic_memory" in names:
            raise AssertionError("record_conversation_experience should not bypass memory_facade when it exists")
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    await conversation_support.record_conversation_experience(
        "Please explain this architecture in detail.",
        "Here is the grounded architectural breakdown.",
        state,
    )

    assert captured["action"] == "conversation_reply"
    assert captured["success"] is True
    assert captured["metadata"]["origin"] == "api"
    assert captured["metadata"]["domain"] == "conversation"
    assert captured["metadata"]["objective"] == "Please explain this architecture in detail."
    assert captured["metadata"]["semantic_mode"] == "technical"
