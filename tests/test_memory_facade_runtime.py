import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.memory.black_hole_vault import BlackHoleVault
from core.memory.memory_facade import MemoryFacade
from core.skills.memory_ops import MemoryOpsInput, MemoryOpsSkill
from interface.routes.memory import api_memory_goals, api_memory_semantic


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


@pytest.mark.asyncio
async def test_memory_facade_commit_interaction_writes_semantic_for_user_facing_turn(monkeypatch):
    facade = MemoryFacade()
    facade._episodic = SimpleNamespace(record_episode_async=AsyncMock(return_value="episode-7"))
    semantic_calls = []
    facade._semantic = SimpleNamespace(
        add_memory=lambda text, metadata=None: semantic_calls.append({"text": text, "metadata": metadata}) or True
    )

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(True, "ok"))
        ),
    )

    result = await facade.commit_interaction(
        context="Bryan said his favorite theorem is Noether's theorem.",
        action="conversation_reply",
        outcome="I acknowledged it and will remember it.",
        success=True,
        importance=0.65,
        metadata={"origin": "api", "objective": "Remember Bryan's favorite theorem."},
    )

    assert result == "episode-7"
    assert semantic_calls
    assert "Bryan" in semantic_calls[0]["text"]
    assert semantic_calls[0]["metadata"]["episode_id"] == "episode-7"


@pytest.mark.asyncio
async def test_api_memory_semantic_reads_semantic_store(monkeypatch):
    class _FakeSemanticStore:
        def get(self, _ids=None, limit=None, include=None):
            return {
                "ids": ["mem-1"],
                "documents": ["Bryan likes Noether's theorem."],
                "metadatas": [{"source": "memory_ops"}],
            }

    monkeypatch.setattr(
        "interface.routes.memory.ServiceContainer.get",
        staticmethod(lambda name, default=None: _FakeSemanticStore() if name == "semantic_memory" else default),
    )

    response = await api_memory_semantic(limit=10, offset=0, _=None, __=None)
    payload = json.loads(response.body)

    assert payload["items"][0]["content"] == "Bryan likes Noether's theorem."
    assert payload["items"][0]["metadata"]["source"] == "memory_ops"


@pytest.mark.asyncio
async def test_api_memory_goals_prefers_canonical_goal_engine(monkeypatch):
    fake_goal_engine = SimpleNamespace(
        build_snapshot=lambda limit, include_external=True: {
            "items": [
                {
                    "id": "goal-1",
                    "objective": "Keep the runtime stable.",
                    "status": "in_progress",
                    "horizon": "short_term",
                    "source": "goal_engine",
                },
                {
                    "id": "goal-2",
                    "objective": "Preserve long-term planner continuity.",
                    "status": "completed",
                    "horizon": "long_term",
                    "source": "strategic_planner",
                },
            ],
            "summary": {"active_count": 1, "completed_count": 1},
        }
    )

    monkeypatch.setattr(
        "interface.routes.memory.ServiceContainer.get",
        staticmethod(lambda name, default=None: fake_goal_engine if name == "goal_engine" else default),
    )

    response = await api_memory_goals(limit=10, _=None)
    payload = json.loads(response.body)

    assert payload["summary"]["active_count"] == 1
    assert payload["summary"]["completed_count"] == 1
    assert payload["items"][0]["source"] == "goal_engine"


def test_black_hole_vault_get_prefers_most_recent_items_when_limited():
    vault = BlackHoleVault.__new__(BlackHoleVault)
    vault.memories = [
        {"created": 1, "text": "oldest", "metadata": {"rank": 1}},
        {"created": 2, "text": "older", "metadata": {"rank": 2}},
        {"created": 3, "text": "newer", "metadata": {"rank": 3}},
        {"created": 4, "text": "newest", "metadata": {"rank": 4}},
    ]

    payload = BlackHoleVault.get(vault, None, limit=2)

    assert payload["ids"] == ["3", "4"]
    assert payload["documents"] == ["newer", "newest"]


def test_memory_ops_derives_structured_fact_from_future_session_request():
    key, value = MemoryOpsSkill._derive_structured_fact(
        "Remember for future sessions that my verification codename is glass orchard."
    )

    assert key == "verification_codename"
    assert value == "glass orchard"


def test_memory_ops_coerce_input_derives_keys_for_pydantic_model_instances():
    skill = MemoryOpsSkill()
    request = skill._coerce_input(
        MemoryOpsInput(
            action="remember",
            content="Remember for future sessions that my verification codename is glass orchard.",
        ),
        {},
    )

    assert request.key == "verification_codename"
    assert request.value == "glass orchard"


@pytest.mark.asyncio
async def test_memory_ops_remember_uses_derived_key_with_pydantic_model_input():
    skill = MemoryOpsSkill()
    memory_store = SimpleNamespace(update_semantic_async=AsyncMock(return_value=True))

    result = await skill.execute(
        MemoryOpsInput(
            action="remember",
            content="Remember for future sessions that my verification codename is glass orchard.",
        ),
        {"memory_store": memory_store},
    )

    assert result["ok"] is True
    assert result["summary"] == "Stored fact: verification_codename."
    memory_store.update_semantic_async.assert_awaited_once_with("verification_codename", "glass orchard")


@pytest.mark.asyncio
async def test_memory_ops_recall_uses_derived_key_from_question():
    skill = MemoryOpsSkill()
    memory_store = SimpleNamespace(get_semantic_async=AsyncMock(return_value="glass orchard"))

    result = await skill.execute(
        {"action": "recall", "query": "What do you remember about my verification codename?"},
        {"memory_store": memory_store},
    )

    assert result["ok"] is True
    assert result["result"] == "glass orchard"
    memory_store.get_semantic_async.assert_awaited_once_with("verification_codename", None)


@pytest.mark.asyncio
async def test_memory_facade_add_memory_records_rejection_reason(monkeypatch):
    facade = MemoryFacade()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "substrate_blocked:neurochemical_cortisol_crisis"))
        ),
    )

    ok = await facade.add_memory(
        "Remember that my verification codename is glass orchard.",
        metadata={"origin": "user", "explicit_memory_request": True},
    )

    assert ok is False
    assert facade._last_add_memory_status["reason"] == "substrate_blocked:neurochemical_cortisol_crisis"


@pytest.mark.asyncio
async def test_memory_facade_add_memory_treats_none_returning_vector_backend_as_success(monkeypatch):
    facade = MemoryFacade()
    vector_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda text, metadata=None: vector_calls.append((text, metadata)))

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(True, "ok"))
        ),
    )

    ok = await facade.add_memory(
        "Remember that my verification codename is glass orchard.",
        metadata={"origin": "user", "explicit_memory_request": True},
    )

    assert ok is True
    assert vector_calls
    assert facade._last_add_memory_status["reason"] == "stored_via_vector"


@pytest.mark.asyncio
async def test_memory_ops_does_not_bypass_memory_facade_rejection():
    semantic_calls = []
    skill = MemoryOpsSkill()

    result = await skill.execute(
        {
            "action": "remember",
            "content": "Remember that my verification codename is glass orchard.",
        },
        {
            "memory_facade": SimpleNamespace(
                add_memory=AsyncMock(return_value=False),
                _last_add_memory_status={"ok": False, "reason": "substrate_blocked:neurochemical_cortisol_crisis"},
            ),
            "semantic_memory": SimpleNamespace(
                add_memory=lambda text, metadata=None: semantic_calls.append((text, metadata))
            ),
            "intent_source": "user",
        },
    )

    assert result["ok"] is False
    assert "substrate_blocked:neurochemical_cortisol_crisis" in result["error"]
    assert semantic_calls == []
