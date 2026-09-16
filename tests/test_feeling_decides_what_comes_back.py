"""Her feeling decides which memories come back, not only the order they come back in.

Retrieval weighs each candidate by importance, how its stored feeling matches
hers and memory's salience, then keeps five. It searched for five, so when
stored texts were alike the search's cut decided recall and her state only
reordered what the cut had picked: on the content run's store, twenty-two
memories sharing one text, every percept class recalled the same set. It now
searches the facade's candidate pool and keeps five of those.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.memory_facade import MemoryFacade
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState

pytestmark = pytest.mark.unit

VALENCES = [-0.1 + 0.7 * index / 21 for index in range(22)]


def _phase(captured: dict) -> MemoryRetrievalPhase:
    async def _search(query, limit=5):
        captured["limit"] = limit
        return [
            {
                "content": f"We talked about the day. [{index:02d}]",
                "score": 0.8,
                "metadata": {"emotional_valence": valence, "importance": 0.7},
            }
            for index, valence in enumerate(VALENCES)
        ][:limit]

    async def _get_hot_memory(limit=3):
        return {"recent_episodes": []}

    facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    knowledge_graph = SimpleNamespace(search_knowledge=lambda query, limit=3: [])
    container = SimpleNamespace(
        get=lambda name, default=None: (
            SimpleNamespace() if name == "memory_manager"
            else knowledge_graph if name == "knowledge_graph"
            else facade if name == "memory_facade"
            else default
        )
    )
    return MemoryRetrievalPhase(container)


async def _recall_at(valence: float) -> tuple[list[str], dict]:
    captured: dict = {}
    phase = _phase(captured)
    state = AuraState.default()
    state.affect.valence = valence
    state.cognition.working_memory.append({"role": "user", "content": "What did we talk about today?"})
    new_state = await phase.execute(state)
    return [text for text in new_state.cognition.long_term_memory if "We talked about the day." in text], captured


def _valence_of(text: str) -> float:
    return VALENCES[int(text.rsplit("[", 1)[1].rstrip("]"))]


@pytest.mark.asyncio
async def test_the_search_is_asked_for_a_pool_larger_than_the_answer() -> None:
    _, captured = await _recall_at(0.5)
    assert captured["limit"] == MemoryFacade.candidate_pool(5)
    assert captured["limit"] > 5


@pytest.mark.asyncio
async def test_a_warm_moment_and_a_cold_one_bring_back_different_memories() -> None:
    warm, _ = await _recall_at(0.6)
    cold, _ = await _recall_at(-0.1)
    assert warm and cold
    assert set(warm) != set(cold)
    assert sum(map(_valence_of, warm)) / len(warm) > sum(map(_valence_of, cold)) / len(cold)


def test_the_scoped_overfetch_and_recall_share_one_pool_rule() -> None:
    assert MemoryFacade.candidate_pool(5) == 32
    assert MemoryFacade.candidate_pool(10) == 60
    assert MemoryFacade.candidate_pool(40) == 100
