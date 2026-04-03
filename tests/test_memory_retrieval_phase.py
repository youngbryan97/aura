from types import SimpleNamespace

import pytest

from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_memory_retrieval_skips_background_queue_tuple_persisted_as_user_string():
    dual_memory = SimpleNamespace(retrieve_context=None)
    memory_manager = SimpleNamespace(dual_memory=dual_memory)
    knowledge_graph = SimpleNamespace(search_knowledge=lambda query, limit=3: [{"type": "fact", "content": query}])
    container = SimpleNamespace(
        get=lambda name, default=None: (
            memory_manager if name == "memory_manager" else knowledge_graph if name == "knowledge_graph" else default
        )
    )
    phase = MemoryRetrievalPhase(container)

    state = AuraState.default()
    state.cognition.working_memory.append({
        "role": "user",
        "content": "(15, 1234.5, 7, {'content': 'Impulse: reflect on my recent interactions.', 'origin': 'autonomous_thought'}, 'autonomous_thought')",
    })

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
            memory_manager if name == "memory_manager"
            else knowledge_graph if name == "knowledge_graph"
            else memory_facade if name == "memory_facade"
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
    assert any("Stored memory about Tell me about Beautiful Mind again." in item for item in new_state.cognition.long_term_memory)
    assert any("Bryan said this song reminds him of Aura." in item for item in new_state.cognition.long_term_memory)


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
            memory_manager if name == "memory_manager"
            else knowledge_graph if name == "knowledge_graph"
            else memory_facade if name == "memory_facade"
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
