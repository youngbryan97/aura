from types import SimpleNamespace

import pytest

from core.phases import memory_retrieval as memory_module
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_memory_retrieval_skips_background_queue_tuple_persisted_as_user_string():
    dual_memory = SimpleNamespace(retrieve_context=None)
    memory_manager = SimpleNamespace(dual_memory=dual_memory)
    knowledge_graph = SimpleNamespace(
        search_knowledge=lambda query, limit=3: [{"type": "fact", "content": query}]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: (
            memory_manager
            if name == "memory_manager"
            else knowledge_graph
            if name == "knowledge_graph"
            else default
        )
    )
    phase = MemoryRetrievalPhase(container)

    state = AuraState.default()
    state.cognition.working_memory.append(
        {
            "role": "user",
            "content": "(15, 1234.5, 7, {'content': 'Impulse: reflect on my recent interactions.', 'origin': 'autonomous_thought'}, 'autonomous_thought')",
        }
    )

    new_state = await phase.execute(state)

    assert new_state is state
    assert new_state.cognition.long_term_memory == []


@pytest.mark.asyncio
async def test_memory_retrieval_includes_memory_facade_results():
    memory_manager = SimpleNamespace()
    knowledge_graph = SimpleNamespace(search_knowledge=lambda query, limit=3: [])

    async def _search(query, limit=5):
        return [{"content": f"Stored memory about {query}"}]

    async def _get_hot_memory(limit=3):
        return {"recent_episodes": ["Bryan said this song reminds him of Aura."]}

    memory_facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    container = SimpleNamespace(
        get=lambda name, default=None: (
            memory_manager
            if name == "memory_manager"
            else knowledge_graph
            if name == "knowledge_graph"
            else memory_facade
            if name == "memory_facade"
            else default
        )
    )
    phase = MemoryRetrievalPhase(container)

    state = AuraState.default()
    state.cognition.working_memory.append(
        {
            "role": "user",
            "content": "Tell me about Beautiful Mind again.",
        }
    )

    new_state = await phase.execute(state)

    assert new_state is not state
    assert any(
        "Stored memory about Tell me about Beautiful Mind again." in item
        for item in new_state.cognition.long_term_memory
    )
    assert any(
        "Bryan said this song reminds him of Aura." in item
        for item in new_state.cognition.long_term_memory
    )


@pytest.mark.asyncio
async def test_memory_retrieval_prioritizes_affect_aligned_memories():
    memory_manager = SimpleNamespace()
    knowledge_graph = SimpleNamespace(search_knowledge=lambda query, limit=3: [])

    async def _search(query, limit=5):
        return [
            {
                "content": "Memory of a warm, connective conversation with Bryan.",
                "metadata": {"emotional_valence": 0.7, "importance": 0.9},
                "score": 0.4,
            },
            {
                "content": "Memory of a cold and dissonant exchange.",
                "metadata": {"emotional_valence": -0.8, "importance": 0.6},
                "score": 0.6,
            },
        ]

    async def _get_hot_memory(limit=3):
        return {"recent_episodes": []}

    memory_facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    container = SimpleNamespace(
        get=lambda name, default=None: (
            memory_manager
            if name == "memory_manager"
            else knowledge_graph
            if name == "knowledge_graph"
            else memory_facade
            if name == "memory_facade"
            else default
        )
    )
    phase = MemoryRetrievalPhase(container)

    state = AuraState.default()
    state.affect.valence = 0.8
    state.affect.arousal = 0.9
    state.affect.emotions["joy"] = 0.8
    state.affect.emotions["trust"] = 0.7
    state.cognition.working_memory.append(
        {
            "role": "user",
            "content": "Tell me what this reminds you of between us.",
        }
    )

    new_state = await phase.execute(state)

    assert "warm, connective conversation" in new_state.cognition.long_term_memory[0]
    assert new_state.response_modifiers["memory_retrieval_signature"]["retrieval_limit"] >= 6


@pytest.mark.asyncio
async def test_memory_retrieval_keeps_defaults_when_substrate_modulation_fails(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()

    def service_get(name, default=None):
        service_get.calls.append(name)
        if name == "attention_schema":
            raise RuntimeError("attention schema unavailable")
        return default

    service_get.calls = []
    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(service_get))
    knowledge_graph = SimpleNamespace(
        search_knowledge=lambda query, limit=5: [{"type": "fact", "content": f"Known: {query}"}]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: knowledge_graph if name == "knowledge_graph" else default
    )
    phase = MemoryRetrievalPhase(container)
    state = AuraState.default()
    state.cognition.working_memory.append(
        {"role": "user", "content": "Tell me what we know about the release path."}
    )

    new_state = await phase.execute(state)

    assert new_state is not state
    assert any(
        "Known: Tell me what we know" in item for item in new_state.cognition.long_term_memory
    )
    assert any(
        "default retrieval limits" in record.action
        for record in tracker.recent(subsystem="memory_retrieval")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_memory_retrieval_uses_remaining_sources_when_facade_fails():
    tracker = get_degradation_tracker()
    tracker.reset()

    async def failing_search(_query, limit=5):
        failing_search.calls += 1
        raise RuntimeError("facade unavailable")

    failing_search.calls = 0
    memory_facade = SimpleNamespace(search=failing_search)
    knowledge_graph = SimpleNamespace(
        search_knowledge=lambda _query, limit=5: [
            {"type": "fact", "content": "Knowledge graph memory still arrived."}
        ]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: (
            memory_facade
            if name == "memory_facade"
            else knowledge_graph
            if name == "knowledge_graph"
            else default
        )
    )
    phase = MemoryRetrievalPhase(container)
    state = AuraState.default()
    state.cognition.working_memory.append(
        {"role": "user", "content": "Find the resilient memory path."}
    )

    new_state = await phase.execute(state)

    assert failing_search.calls == 1
    assert any("Knowledge graph memory" in item for item in new_state.cognition.long_term_memory)
    assert any(
        "without memory-facade context" in record.action
        for record in tracker.recent(subsystem="memory_retrieval")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_memory_retrieval_keeps_memories_when_affect_scheduling_fails(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()

    class AffectEngine:
        def __init__(self):
            self.calls = 0

        def modify(self, **_kwargs):
            self.calls += 1

            async def apply_shift():
                return None

            return apply_shift()

    class TaskTrackerUnavailable:
        def __init__(self):
            self.names = []

        def create_task(self, _coroutine, name=None):
            self.names.append(name)
            raise RuntimeError("task tracker unavailable")

    affect_engine = AffectEngine()
    task_tracker = TaskTrackerUnavailable()
    monkeypatch.setattr(memory_module, "get_task_tracker", lambda: task_tracker)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: affect_engine if name == "affect_engine" else default
        ),
    )
    memory_facade = SimpleNamespace(
        search=lambda _query, limit=5: [
            {
                "content": "A warm memory with enough emotional valence to steer future tone.",
                "metadata": {"emotional_valence": 0.8, "importance": 0.9},
                "score": 0.7,
            }
        ],
        get_hot_memory=lambda limit=3: {"recent_episodes": []},
    )
    container = SimpleNamespace(
        get=lambda name, default=None: memory_facade if name == "memory_facade" else default
    )
    phase = MemoryRetrievalPhase(container)
    state = AuraState.default()
    state.affect.valence = 0.7
    state.cognition.working_memory.append(
        {"role": "user", "content": "Recall the emotional memory safely."}
    )

    new_state = await phase.execute(state)

    assert affect_engine.calls == 1
    assert task_tracker.names == ["memory_retrieval.affective_hit"]
    assert any("warm memory" in item for item in new_state.cognition.long_term_memory)
    assert any(
        "affect scheduling failed" in record.action
        for record in tracker.recent(subsystem="memory_retrieval")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_memory_retrieval_handles_legacy_string_working_memory_entry():
    knowledge_graph = SimpleNamespace(
        search_knowledge=lambda query, limit=5: [{"type": "fact", "content": f"Recovered: {query}"}]
    )
    container = SimpleNamespace(
        get=lambda name, default=None: knowledge_graph if name == "knowledge_graph" else default
    )
    phase = MemoryRetrievalPhase(container)
    state = AuraState.default()
    state.cognition.working_memory.append("Tell me about the legacy working memory entry.")

    new_state = await phase.execute(state)

    assert new_state is not state
    assert any(
        "legacy working memory entry" in item for item in new_state.cognition.long_term_memory
    )
