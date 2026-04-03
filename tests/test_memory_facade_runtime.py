from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.memory.memory_facade import MemoryFacade


@pytest.mark.asyncio
async def test_memory_facade_search_supports_sync_vector_and_graph():
    facade = MemoryFacade()
    facade._vector = SimpleNamespace(
        search_similar=lambda query, limit=5: [
            {"id": "vec-1", "content": f"vector memory about {query}", "metadata": {"source": "vector"}}
        ]
    )
    facade._graph = SimpleNamespace(
        search_knowledge=lambda query, limit=5: [
            {"id": "kg-1", "content": f"graph memory about {query}", "metadata": {"source": "graph"}}
        ]
    )

    results = await facade.search("Bryan", limit=5)

    assert any("vector memory about Bryan" == item["content"] for item in results)
    assert any("graph memory about Bryan" == item["content"] for item in results)


@pytest.mark.asyncio
async def test_memory_facade_commit_interaction_supports_sync_vector_and_ledger(monkeypatch):
    facade = MemoryFacade()
    facade._episodic = SimpleNamespace(record_episode_async=AsyncMock(return_value="episode-1"))
    vector_calls = []
    ledger_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda **kwargs: vector_calls.append(kwargs) or True)
    facade._ledger = SimpleNamespace(log_interaction=lambda *args: ledger_calls.append(args))

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(True, "ok"))
        ),
    )

    result = await facade.commit_interaction(
        context="Bryan asked about the story",
        action="conversation_reply",
        outcome="Aura answered from grounded context",
        success=True,
        importance=0.9,
    )

    assert result == "episode-1"
    assert vector_calls
    assert ledger_calls == [("conversation_reply", "Aura answered from grounded context", True)]
