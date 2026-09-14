"""A percept is a retrieval cue.

Retrieval was asked what someone said or what she was working on, so what she
perceived reached affect, the workspace and the world model and never memory.
Every percept class in the content runs brought back the same memories. These
pin the cue: what arrives joins the question when nobody has spoken, the most
salient percept is the one asked about, and each percept cues one recall.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept


def _phase(captured: dict) -> MemoryRetrievalPhase:
    async def _search(query, limit=5):  # noqa: ANN001, ANN202
        captured.setdefault("queries", []).append(query)
        return [{"content": f"a memory recalled for: {query}"}]

    async def _get_hot_memory(limit=3):  # noqa: ANN001, ANN202
        return {"recent_episodes": []}

    facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    return MemoryRetrievalPhase(
        SimpleNamespace(get=lambda name, default=None: facade if name == "memory_facade" else default)
    )


def _thinking_state() -> AuraState:
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "assistant", "content": "I will look into it."})
    state.cognition.current_objective = "keep the workshop in order"
    return state


@pytest.mark.asyncio
async def test_what_arrives_joins_the_question_when_nobody_has_spoken() -> None:
    captured: dict = {}
    phase = _phase(captured)
    heron = _thinking_state()
    emit_percept(heron.world, "discovery", content="a heron standing in the flooded field", intensity=0.6)
    kettle = _thinking_state()
    emit_percept(kettle.world, "error", content="the kettle boiled dry on the stove", intensity=0.6)

    await phase.execute(heron)
    await phase.execute(kettle)
    first, second = captured["queries"]
    assert "heron" in first and "kettle" not in first
    assert "kettle" in second and "heron" not in second


@pytest.mark.asyncio
async def test_a_percept_alone_still_asks() -> None:
    captured: dict = {}
    state = AuraState.default()
    emit_percept(state.world, "discovery", content="a heron standing in the flooded field", intensity=0.6)
    after = await _phase(captured).execute(state)
    assert captured["queries"] == ["a heron standing in the flooded field"]
    assert after is not state


@pytest.mark.asyncio
async def test_the_most_salient_percept_is_the_one_asked_about() -> None:
    captured: dict = {}
    state = _thinking_state()
    emit_percept(state.world, "discovery", content="a leaf moving on the sill", intensity=0.2)
    emit_percept(state.world, "error", content="smoke coming from the workshop door", intensity=0.9)
    await _phase(captured).execute(state)
    (query,) = captured["queries"]
    assert "smoke coming from the workshop door" in query
    assert "leaf" not in query


@pytest.mark.asyncio
async def test_what_someone_said_is_asked_without_what_she_perceives() -> None:
    captured: dict = {}
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "What did we say about the concert?"})
    emit_percept(state.world, "discovery", content="a heron standing in the flooded field", intensity=0.6)
    await _phase(captured).execute(state)
    assert captured["queries"] == ["What did we say about the concert?"]


@pytest.mark.asyncio
async def test_one_percept_cues_one_recall() -> None:
    captured: dict = {}
    phase = _phase(captured)
    state = _thinking_state()
    emit_percept(state.world, "discovery", content="a heron standing in the flooded field", intensity=0.6)
    after = await phase.execute(state)
    await phase.execute(after)
    assert "heron" in captured["queries"][0]
    assert all("heron" not in query for query in captured["queries"][1:])


@pytest.mark.asyncio
async def test_nothing_perceived_and_nothing_in_mind_asks_nothing() -> None:
    captured: dict = {}
    state = AuraState.default()
    assert await _phase(captured).execute(state) is state
    assert "queries" not in captured
