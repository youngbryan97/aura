import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.memory.black_hole import BlackHoleDecodeError, decode_payload
from core.memory.black_hole_vault import BlackHoleVault
from core.memory.memory_facade import MemoryFacade
from core.runtime.atomic_writer import atomic_write_json
from core.skills.memory_ops import MemoryOpsInput, MemoryOpsSkill
from interface.routes.memory import (
    _build_episodic_memory_response,
    api_memory_goals,
    api_memory_semantic,
)


class AsyncCallFixture:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


class MemoryWriteGatewayFixture:
    def __init__(self):
        self.requests = []
        self.quarantines = []

    async def write(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            record_id=f"record-{len(self.requests)}",
            receipt_id=f"receipt-{len(self.requests)}",
            bytes_written=len(request.content.encode("utf-8")),
            schema_version=1,
        )

    async def quarantine(self, record_id, reason):
        self.quarantines.append((record_id, reason))


def install_memory_gateway_fixture(monkeypatch):
    gateway = MemoryWriteGatewayFixture()
    monkeypatch.setattr(
        "core.memory.memory_write_gateway.get_memory_write_gateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        "core.runtime.action_executor.get_memory_write_gateway",
        lambda: gateway,
    )
    return gateway


def test_memory_facade_ready_requires_actual_memory_backend():
    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "state_repository",
            SimpleNamespace(is_initialized=lambda: True),
        )
        facade = MemoryFacade()

        assert facade.is_ready() is False

        ServiceContainer.register_instance(
            "vector_memory",
            SimpleNamespace(
                is_ready=lambda: True,
                add_memory=lambda *_args, **_kwargs: True,
                search=lambda *_args, **_kwargs: [],
            ),
        )

        assert facade.is_ready() is True
    finally:
        ServiceContainer.clear()


def test_memory_facade_ready_rejects_unhealthy_memory_backend():
    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "vector_memory",
            SimpleNamespace(
                is_ready=lambda: False,
                add_memory=lambda *_args, **_kwargs: True,
                search=lambda *_args, **_kwargs: [],
            ),
        )

        assert MemoryFacade().is_ready() is False
    finally:
        ServiceContainer.clear()


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
async def test_memory_facade_scopes_personal_recall_but_keeps_general_knowledge():
    facade = MemoryFacade()
    facade._vector = SimpleNamespace(
        search_similar=lambda _query, limit=5: [
            {
                "id": "general",
                "content": "Octopuses have distributed neural control.",
                "metadata": {"source": "encyclopedia"},
            },
            {
                "id": "owner",
                "content": "owner private conversation",
                "metadata": {
                    "conversation_lane": True,
                    "principal_id": "bryan",
                    "principal_surface": "owner",
                },
            },
            {
                "id": "paired-a",
                "content": "paired A private conversation",
                "metadata": {
                    "conversation_lane": True,
                    "principal_id": "paired-device:a",
                    "principal_surface": "paired_device",
                },
            },
            {
                "id": "legacy",
                "content": "legacy unbound personal conversation",
                "metadata": {"conversation_lane": True},
            },
            {
                "id": "profile-b",
                "content": "paired B profile preference",
                "metadata": {
                    "user_id": "paired-device:b",
                    "principal_surface": "paired_device",
                    "source": "profile_learning",
                },
            },
        ]
    )

    paired = await facade.search(
        "conversation",
        limit=10,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )
    assert {item["id"] for item in paired} == {"general", "paired-a"}

    owner = await facade.search(
        "conversation",
        limit=10,
        principal_id="bryan",
        principal_surface="owner",
    )
    assert {item["id"] for item in owner} == {"general", "owner", "legacy"}


@pytest.mark.asyncio
async def test_scoped_memory_search_overfetches_past_foreign_ranked_results():
    observed_limits = []
    foreign = [
        {
            "id": f"foreign-{index}",
            "content": f"foreign private record {index}",
            "metadata": {
                "conversation_lane": True,
                "principal_id": "paired-device:b",
                "principal_surface": "paired_device",
            },
        }
        for index in range(8)
    ]
    own = {
        "id": "own-after-foreign",
        "content": "authorized record behind foreign ranks",
        "metadata": {
            "conversation_lane": True,
            "principal_id": "paired-device:a",
            "principal_surface": "paired_device",
        },
    }

    def _search(_query, limit=5):
        observed_limits.append(limit)
        return (foreign + [own])[:limit]

    facade = MemoryFacade()
    facade._vector = SimpleNamespace(search_similar=_search)
    results = await facade.search(
        "private record",
        limit=1,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )

    assert observed_limits == [32]
    assert [item["id"] for item in results] == ["own-after-foreign"]


@pytest.mark.asyncio
async def test_memory_facade_search_reads_strict_gateway_records(monkeypatch, tmp_path):
    record_path = tmp_path / "episodic" / "session-pin.json"
    atomic_write_json(
        record_path,
        {
            "content": "Bryan explicitly asked Aura to remember the blue lantern phrase.",
            "metadata": {
                "source": "chat_api",
                "session_memory_pin": True,
                "explicit_memory_request": True,
            },
            "cause": "memory_facade.add_memory",
            "written_at": 12345.0,
        },
        schema_version=1,
        schema_name="memory.episodic",
    )
    monkeypatch.setattr(
        "core.memory.memory_write_gateway.get_memory_write_gateway",
        lambda: SimpleNamespace(root=tmp_path),
    )

    # The gateway record index serves cold searches empty by design (freshness
    # never costs event-loop time). Bind the process-wide index to this root
    # and warm it before searching, mirroring live behavior after boot.
    import core.memory.gateway_record_index as _gri

    monkeypatch.setattr(_gri, "_INDEX", None)
    _index = _gri.get_gateway_record_index(tmp_path)
    _index.search("warmup", limit=1)
    _deadline = time.monotonic() + 5.0
    while not _index._built and time.monotonic() < _deadline:
        await asyncio.sleep(0.01)
    assert _index._built, "gateway record index failed to warm"

    results = await MemoryFacade().search("blue lantern phrase", limit=5)

    assert any("blue lantern phrase" in item["content"] for item in results)
    matched = next(item for item in results if "blue lantern phrase" in item["content"])
    assert matched["metadata"]["session_memory_pin"] is True
    assert matched["metadata"]["verification_state"] == "not_applicable"


@pytest.mark.asyncio
async def test_memory_facade_compat_search_accepts_top_k_and_sync(monkeypatch, tmp_path):
    record_path = tmp_path / "episodic" / "compat-pin.json"
    atomic_write_json(
        record_path,
        {
            "content": "Aura should recall the compatibility blue lantern memory.",
            "metadata": {
                "source": "chat_api",
                "session_memory_pin": True,
                "explicit_memory_request": True,
            },
            "cause": "memory_facade.add_memory",
            "written_at": 12346.0,
        },
        schema_version=1,
        schema_name="memory.episodic",
    )
    monkeypatch.setattr(
        "core.memory.memory_write_gateway.get_memory_write_gateway",
        lambda: SimpleNamespace(root=tmp_path),
    )

    # The gateway record index serves cold searches empty by design (freshness
    # never costs event-loop time). Bind the process-wide index to this root
    # and warm it before searching, mirroring live behavior after boot.
    import core.memory.gateway_record_index as _gri

    monkeypatch.setattr(_gri, "_INDEX", None)
    _index = _gri.get_gateway_record_index(tmp_path)
    _index.search("warmup", limit=1)
    _deadline = time.monotonic() + 5.0
    while not _index._built and time.monotonic() < _deadline:
        await asyncio.sleep(0.01)
    assert _index._built, "gateway record index failed to warm"

    facade = MemoryFacade()

    async_results = await facade.search_memories("compatibility blue lantern", top_k=1)
    unified_results = await facade.retrieve_unified_context(
        "compatibility blue lantern",
        top_k=1,
        future_arg_is_ignored=True,
    )
    sync_results = facade.search_sync("compatibility blue lantern", top_k=1)

    assert len(async_results) == 1
    assert len(unified_results) == 1
    assert len(sync_results) == 1
    assert async_results[0]["content"] == sync_results[0]["content"]


@pytest.mark.asyncio
async def test_memory_facade_commit_interaction_supports_sync_vector_and_ledger(monkeypatch):
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "0")
    # These exercise the direct episodic and semantic writes, which is
    # the ungoverned path. It used to be reached by relaxing the global
    # strict flag; it has its own name now, so a test that wants it says
    # so where a reader can see it.
    monkeypatch.setenv("AURA_MEMORY_WRITES_UNGOVERNED", "1")
    facade = MemoryFacade()
    facade._episodic = SimpleNamespace(record_episode_async=AsyncCallFixture(return_value="episode-1"))
    vector_calls = []
    ledger_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda **kwargs: vector_calls.append(kwargs) or True)
    facade._ledger = SimpleNamespace(log_interaction=lambda *args: ledger_calls.append(args))

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
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
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "0")
    # These exercise the direct episodic and semantic writes, which is
    # the ungoverned path. It used to be reached by relaxing the global
    # strict flag; it has its own name now, so a test that wants it says
    # so where a reader can see it.
    monkeypatch.setenv("AURA_MEMORY_WRITES_UNGOVERNED", "1")
    facade = MemoryFacade()
    facade._episodic = SimpleNamespace(record_episode_async=AsyncCallFixture(return_value="episode-7"))
    semantic_calls = []
    facade._semantic = SimpleNamespace(
        add_memory=lambda text, metadata=None: semantic_calls.append({"text": text, "metadata": metadata}) or True
    )

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
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
async def test_memory_facade_commit_interaction_defaults_to_gateway(monkeypatch):
    gateway = install_memory_gateway_fixture(monkeypatch)
    facade = MemoryFacade()
    vector_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda **kwargs: vector_calls.append(kwargs) or True)

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
        ),
    )

    result = await facade.commit_interaction(
        context="Bryan asked about canonical runtime memory",
        action="conversation_reply",
        outcome="Aura routed memory through the gateway",
        success=True,
        importance=0.9,
    )

    assert result == "gateway-receipt"
    assert len(gateway.requests) == 1
    assert gateway.requests[0].cause == "memory_facade.commit_interaction"
    assert "canonical runtime memory" in gateway.requests[0].content
    assert vector_calls == []


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
async def test_api_memory_semantic_reports_degraded_backend_failure(monkeypatch):
    class _FailingSemanticStore:
        def __init__(self):
            self.called = False

        def get(self, _ids=None, limit=None, include=None):
            self.called = True
            raise RuntimeError("semantic store unavailable")

    store = _FailingSemanticStore()
    monkeypatch.setattr(
        "interface.routes.memory.ServiceContainer.get",
        staticmethod(lambda name, default=None: store if name == "semantic_memory" else default),
    )

    response = await api_memory_semantic(limit=10, offset=0, _=None, __=None)
    payload = json.loads(response.body)

    assert store.called is True
    assert payload["items"] == []
    assert payload["degraded"] is True
    assert "Semantic memory failed" in payload["degradation_reasons"]


@pytest.mark.asyncio
async def test_api_memory_episodic_reports_degraded_recall_failure(monkeypatch):
    class _FailingMemoryManager:
        def __init__(self):
            self.called = False

        async def recall(self, *_args, **_kwargs):
            self.called = True
            raise RuntimeError("episodic recall unavailable")

    manager = _FailingMemoryManager()

    def _get_service(name, default=None):
        if name == "memory_manager":
            return manager
        return default

    monkeypatch.setattr(
        "interface.routes.memory.ServiceContainer.get",
        staticmethod(_get_service),
    )

    response = await _build_episodic_memory_response(limit=10, offset=0)
    payload = json.loads(response.body)

    assert manager.called is True
    assert payload["items"] == []
    assert payload["degraded"] is True
    assert "Memory manager recall failed" in payload["degradation_reasons"]


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


@pytest.mark.asyncio
async def test_api_memory_goals_reports_degraded_planner_failure(monkeypatch):
    class _FailingPlannerStore:
        def __init__(self):
            self.called = False

        def get_active_projects(self):
            self.called = True
            raise RuntimeError("planner store unavailable")

    planner_store = _FailingPlannerStore()
    planner = SimpleNamespace(store=planner_store)

    monkeypatch.setattr(
        "interface.routes.memory.ServiceContainer.get",
        staticmethod(lambda name, default=None: planner if name == "strategic_planner" else default),
    )

    response = await api_memory_goals(limit=10, _=None)
    payload = json.loads(response.body)

    assert planner_store.called is True
    assert payload["degraded"] is True
    assert "Strategic planner goals failed" in payload["degradation_reasons"]
    assert payload["summary"]["active_count"] == 0


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


def test_black_hole_vault_delete_memories_supports_episode_metadata(monkeypatch):
    vault = BlackHoleVault.__new__(BlackHoleVault)
    vault.memories = [
        {"created": 1, "text": "keep", "metadata": {"episode_id": "keep"}},
        {"created": 2, "text": "drop-a", "metadata": {"episode_id": "drop-a"}},
        {"created": 3, "text": "drop-b", "metadata": {"episode_id": "drop-b"}},
    ]
    vault._dirty = False
    saves = []
    monkeypatch.setattr(vault, "_save_vault", lambda: saves.append(True))

    deleted = BlackHoleVault.delete_memories(
        vault,
        filter_metadata={"episode_id": ["drop-a", "drop-b"]},
    )

    assert deleted == 2
    assert [memory["text"] for memory in vault.memories] == ["keep"]
    assert vault._dirty is True
    assert saves == [True]


def test_black_hole_strict_decode_raises_without_runtime_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.memory import black_hole

    degradations = []
    monkeypatch.setattr(black_hole, "record_degradation", lambda *args, **kwargs: degradations.append(args))

    assert decode_payload("not-valid-ciphertext", "active-test-key") == ""
    assert degradations == []

    with pytest.raises(BlackHoleDecodeError):
        decode_payload("not-valid-ciphertext", "active-test-key", strict=True)

    assert degradations == []


def test_black_hole_vault_quarantines_unreadable_encrypted_file(tmp_path) -> None:
    vault = BlackHoleVault.__new__(BlackHoleVault)
    vault.data_dir = str(tmp_path)
    vault.memories_file = str(tmp_path / "event_horizon.json")
    vault.key = "active-test-key"
    vault.memories = [{"text": "stale"}]
    vault._fallback_mode = False

    (tmp_path / "event_horizon.json").write_text("not-valid-ciphertext", encoding="utf-8")

    BlackHoleVault._load_vault(vault)

    assert vault.memories == []
    assert vault._fallback_mode is True
    assert not (tmp_path / "event_horizon.json").exists()
    quarantined = list(tmp_path.glob("event_horizon.json.quarantine-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "not-valid-ciphertext"


@pytest.mark.asyncio
async def test_memory_ops_core_append_writes_to_block(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.config.paths", SimpleNamespace(base_dir=str(tmp_path), home_dir=tmp_path, data_dir=tmp_path))
    skill = MemoryOpsSkill()

    result = await skill.execute(
        MemoryOpsInput(action="core_append", block="user", content="verification_codename: glass orchard"),
        {},
    )

    assert result["ok"] is True
    assert "user" in result["summary"]
    block_text = (skill.mem_fs_dir / "user.txt").read_text()
    assert "verification_codename: glass orchard" in block_text
    assert result["effect_verified"] is True
    assert result["block"] == "user"
    assert result["sha256"] == hashlib.sha256(block_text.encode("utf-8")).hexdigest()
    assert result["criteria_results"]["core memory appended"] is True


@pytest.mark.asyncio
async def test_memory_ops_archival_search_uses_facade():
    skill = MemoryOpsSkill.__new__(MemoryOpsSkill)
    search_memories = AsyncCallFixture(return_value=[
        {"score": 0.95, "content": "glass orchard"},
    ])
    memory_facade = SimpleNamespace(
        search_memories=search_memories
    )

    result = await skill.execute(
        {"action": "archival_search", "query": "verification codename"},
        {"memory_facade": memory_facade},
    )

    assert result["ok"] is True
    assert any("glass orchard" in r for r in result["results"])
    assert search_memories.calls == [(("verification codename",), {"limit": 5})]


@pytest.mark.asyncio
async def test_memory_ops_remember_alias_uses_archival_insert(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.config.paths", SimpleNamespace(base_dir=str(tmp_path), home_dir=tmp_path, data_dir=tmp_path))
    gateway = install_memory_gateway_fixture(monkeypatch)
    skill = MemoryOpsSkill()
    facade = MemoryFacade()
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
        ),
    )

    result = await skill.execute(
        {
            "action": "remember",
            "content": "Remember for future sessions that my verification codename is glass orchard.",
        },
        {"memory_facade": facade},
    )

    assert result["ok"] is True
    assert result["summary"] == "Committed to archival storage."
    assert result["effect_verified"] is True
    assert result["record_id"] == "record-1"
    assert result["memory_receipt_id"] == "receipt-1"
    assert result["bytes_written"] > 0
    assert result["criteria_results"]["archival memory stored"] is True
    assert len(gateway.requests) == 1
    assert gateway.requests[0].content == "Remember for future sessions that my verification codename is glass orchard."
    assert gateway.requests[0].metadata["source"] == "archival_insert"
    assert "provenance" in gateway.requests[0].metadata


@pytest.mark.asyncio
async def test_memory_ops_recall_alias_uses_facade_search(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.config.paths", SimpleNamespace(base_dir=str(tmp_path), home_dir=tmp_path, data_dir=tmp_path))
    skill = MemoryOpsSkill()
    search = AsyncCallFixture(return_value=[
        {"score": 0.95, "content": "glass orchard"},
    ])

    result = await skill.execute(
        {"action": "recall", "query": "verification codename"},
        {"memory_facade": SimpleNamespace(search=search)},
    )

    assert result["ok"] is True
    assert any("glass orchard" in item for item in result["results"])
    assert search.calls == [(("verification codename",), {"limit": 5})]


@pytest.mark.asyncio
async def test_memory_facade_add_memory_records_rejection_reason(monkeypatch):
    facade = MemoryFacade()

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(False, "substrate_blocked:neurochemical_cortisol_crisis"))
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
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "0")
    # These exercise the direct episodic and semantic writes, which is
    # the ungoverned path. It used to be reached by relaxing the global
    # strict flag; it has its own name now, so a test that wants it says
    # so where a reader can see it.
    monkeypatch.setenv("AURA_MEMORY_WRITES_UNGOVERNED", "1")
    facade = MemoryFacade()
    vector_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda text, metadata=None: vector_calls.append((text, metadata)))

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
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
async def test_memory_facade_add_memory_degrades_open_for_legacy_non_user_writes(monkeypatch):
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "0")
    # These exercise the direct episodic and semantic writes, which is
    # the ungoverned path. It used to be reached by relaxing the global
    # strict flag; it has its own name now, so a test that wants it says
    # so where a reader can see it.
    monkeypatch.setenv("AURA_MEMORY_WRITES_UNGOVERNED", "1")
    facade = MemoryFacade()
    vector_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda text, metadata=None: vector_calls.append((text, metadata)) or True)

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(False, "self_model_required"))
        ),
    )

    ok = await facade.add_memory(
        "Journal line",
        metadata={"type": "narrative_journal", "timestamp": 10.0},
    )

    assert ok is True
    assert vector_calls
    assert facade._last_add_memory_status["reason"] == "stored_via_vector"


@pytest.mark.asyncio
async def test_memory_facade_add_memory_defaults_to_gateway(monkeypatch):
    gateway = install_memory_gateway_fixture(monkeypatch)
    facade = MemoryFacade()
    vector_calls = []
    facade._vector = SimpleNamespace(add_memory=lambda text, metadata=None: vector_calls.append((text, metadata)) or True)

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
        ),
    )

    ok = await facade.add_memory(
        "Remember that my verification codename is glass orchard.",
        metadata={"origin": "user", "explicit_memory_request": True},
    )

    assert ok is True
    assert facade._last_add_memory_status["reason"] == "stored_via_gateway"
    assert facade._last_add_memory_status["record_id"] == "record-1"
    assert facade._last_add_memory_status["receipt_id"] == "receipt-1"
    assert facade._last_add_memory_status["bytes_written"] > 0
    assert len(gateway.requests) == 1
    assert gateway.requests[0].cause == "memory_facade.add_memory"
    assert vector_calls == []


def test_memory_facade_treats_session_memory_pin_as_user_facing_source():
    facade = MemoryFacade()

    assert (
        facade._resolve_memory_write_source(
            {"source": "session_memory_pin", "explicit_memory_request": True}
        )
        == "session_memory_pin"
    )


@pytest.mark.asyncio
async def test_memory_ops_archival_insert_calls_gateway(monkeypatch):
    gateway = install_memory_gateway_fixture(monkeypatch)
    skill = MemoryOpsSkill.__new__(MemoryOpsSkill)
    memory_facade = MemoryFacade()
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncCallFixture(return_value=(True, "ok"))
        ),
    )

    result = await skill.execute(
        {
            "action": "archival_insert",
            "content": "My verification codename is glass orchard.",
        },
        {"memory_facade": memory_facade},
    )

    assert result["ok"] is True
    assert result["summary"] == "Committed to archival storage."
    assert result["effect_verified"] is True
    assert result["record_id"] == "record-1"
    assert result["memory_receipt_id"] == "receipt-1"
    assert len(gateway.requests) == 1
    assert gateway.requests[0].content == "My verification codename is glass orchard."
    assert gateway.requests[0].metadata["source"] == "archival_insert"
    assert "provenance" in gateway.requests[0].metadata


@pytest.mark.asyncio
async def test_memory_ops_archival_insert_rejects_backend_false_acknowledgement():
    skill = MemoryOpsSkill.__new__(MemoryOpsSkill)
    facade = SimpleNamespace(
        add_memory=AsyncCallFixture(return_value=False),
        last_add_memory_status=lambda: {
            "ok": False,
            "reason": "vector_backend_returned_false",
            "backend": "vector",
        },
    )

    result = await skill.execute(
        {"action": "archival_insert", "content": "Do not claim this was stored."},
        {"memory_facade": facade, "origin": "user"},
    )

    assert result["ok"] is False
    assert "vector_backend_returned_false" in result["error"]


def test_gateway_index_refresh_parses_in_bounded_passes(tmp_path, monkeypatch):
    """GIL discipline: a cold refresh parses at most MAX_PARSE_PER_PASS files
    per pass (top Jul 8 stall fingerprint was this refresher monopolizing the
    GIL); the cache carries across passes until the view is complete."""
    import json as _json

    from core.memory.gateway_record_index import GatewayRecordIndex

    root = tmp_path / "records"
    sub = root / "bucket"
    sub.mkdir(parents=True)
    total = 30
    for i in range(total):
        (sub / f"r{i:03d}.json").write_text(
            _json.dumps({"payload": {"content": f"record {i}", "metadata": {}}}),
            encoding="utf-8",
        )

    index = GatewayRecordIndex(root)
    monkeypatch.setattr(GatewayRecordIndex, "MAX_PARSE_PER_PASS", 10)
    monkeypatch.setattr(GatewayRecordIndex, "PARSE_YIELD_S", 0.0)

    index._refresh_running.acquire()
    index._do_refresh()
    first_pass = len(index._entries)
    assert first_pass == 10, "cold pass must stop at the parse budget"

    for _ in range(3):
        index._refresh_running.acquire()
        index._do_refresh()
    assert len(index._entries) == total, "later passes complete the view via the cache"
