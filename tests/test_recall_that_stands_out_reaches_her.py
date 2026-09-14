"""Reliving reaches recall, the state and the subject core.

The ledger's tests pin the two readings. These pin that the retrieval phase
uses them: asking the same question again searches deeper instead of returning
early, every recall says whether it was relived or looked up, and the columns
that read it are held by the lesion clamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory import reliving
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _fresh_ledgers():
    reliving.reset_for_test()
    yield
    reliving.reset_for_test()


def _phase() -> MemoryRetrievalPhase:
    async def _search(query, limit=5):
        return [{"content": f"Stored memory about {query}"}]

    async def _get_hot_memory(limit=3):
        return {"recent_episodes": ["He said the record reminded him of her."]}

    memory_facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    knowledge_graph = SimpleNamespace(search_knowledge=lambda query, limit=3: [])
    container = SimpleNamespace(
        get=lambda name, default=None: (
            SimpleNamespace()
            if name == "memory_manager"
            else knowledge_graph
            if name == "knowledge_graph"
            else memory_facade
            if name == "memory_facade"
            else default
        )
    )
    return MemoryRetrievalPhase(container)


def _asked(text: str = "Tell me about that night again.") -> AuraState:
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": text})
    return state


@pytest.mark.asyncio
async def test_asking_the_same_question_again_looks_deeper() -> None:
    phase = _phase()
    first = await phase.execute(_asked())
    second = await phase.execute(first)

    depth = lambda s: s.response_modifiers["memory_retrieval_signature"]["retrieval_limit"]
    assert second is not first, "the repeat returned early instead of going deeper"
    assert depth(second) == depth(first) + 1


@pytest.mark.asyncio
async def test_every_recall_says_whether_it_was_relived_or_looked_up() -> None:
    state = await _phase().execute(_asked())
    reading = state.cognition.relived
    assert reading, "recall said nothing about what it brought back"
    assert reading["returns"] == 0
    assert "relived" in reading and "intensity" in reading


@pytest.mark.asyncio
async def test_the_recall_percept_carries_which_kind_it_was() -> None:
    state = await _phase().execute(_asked())
    replays = [p for p in state.world.recent_percepts if p.get("type") == "memory_replay"]
    assert replays, "recall reached the state but not her feeling"
    assert "relived" in replays[-1]


def test_the_columns_that_read_it_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["M"].sources)
    for field in (
        "cognition.relived.relived",
        "cognition.relived.intensity",
        "cognition.relived.returns",
    ):
        assert field in sources, field
    assert "cognition.relived" in CLAMPED_FIELDS["M"]
