"""What she remembers shapes what she perceives, and what she perceives rereads what she remembers.

Recall reached affect, the workspace and deliberation and never reached which
percept she took as meant; perception reached recall only as a search cue, and
never changed what a recollection was worth once it came back. P12.6 and P12.7
ask for both directions. These pin them: a recollection primes the percepts it
bears on, in proportion to its strength; a percept raises the recollections that
carry it, in proportion to its salience; and neither touches what it has no
words in common with.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept, prime_stream, read_percept

pytestmark = pytest.mark.unit


def test_a_recollection_primes_the_percept_it_bears_on() -> None:
    state = AuraState.default()
    heron = emit_percept(state.world, "discovery", content="a heron standing in the flooded field", intensity=0.4)
    kettle = emit_percept(state.world, "error", content="kettle boiled dry", intensity=0.4)
    moved = prime_stream(state.world, [("[memory] we watched a heron standing in the flooded field", 0.8)])
    assert moved == 1
    assert read_percept(heron).salience > 0.4
    assert read_percept(kettle).salience == pytest.approx(0.4)


def test_a_weaker_recollection_primes_less() -> None:
    strong, weak = AuraState.default(), AuraState.default()
    first = emit_percept(strong.world, "discovery", content="a heron in the field", intensity=0.3)
    second = emit_percept(weak.world, "discovery", content="a heron in the field", intensity=0.3)
    prime_stream(strong.world, [("a heron in the field", 0.9)])
    prime_stream(weak.world, [("a heron in the field", 0.2)])
    assert read_percept(first).salience > read_percept(second).salience > 0.3


def test_what_recall_put_in_the_stream_is_not_primed_by_recall() -> None:
    state = AuraState.default()
    replay = emit_percept(state.world, "memory_replay", content="a heron in the field", intensity=0.3, source="memory_retrieval")
    assert prime_stream(state.world, [("a heron in the field", 0.9)]) == 0
    assert read_percept(replay).salience == pytest.approx(0.3)


def _phase(recollections: list[str]) -> MemoryRetrievalPhase:
    async def _search(query, limit=5):  # noqa: ANN001, ANN202
        return [{"content": text, "score": 0.5, "metadata": {"importance": 0.5}} for text in recollections]

    async def _get_hot_memory(limit=3):  # noqa: ANN001, ANN202
        return {"recent_episodes": []}

    facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    return MemoryRetrievalPhase(
        SimpleNamespace(get=lambda name, default=None: facade if name == "memory_facade" else default)
    )


@pytest.mark.asyncio
async def test_a_percept_raises_the_recollections_that_carry_it() -> None:
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "assistant", "content": "I will look into it."})
    state.cognition.current_objective = "keep the workshop in order"
    emit_percept(state.world, "error", content="smoke coming from the workshop door", intensity=0.9)
    unrelated = "we reorganised the shelves by colour"
    related = "last winter smoke came from the workshop door when the stove failed"
    after = await _phase([unrelated, related]).execute(state)
    ranked = [str(text) for text in after.cognition.long_term_memory]
    assert "workshop door" in ranked[0]
    assert after.cognition.memory_scores[0] > after.cognition.memory_scores[1]
    assert f"score={after.cognition.memory_scores[0]:.3f}" in ranked[0]


@pytest.mark.asyncio
async def test_what_she_recalls_primes_what_she_perceives_within_the_turn() -> None:
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "assistant", "content": "I will look into it."})
    state.cognition.current_objective = "keep the workshop in order"
    emit_percept(state.world, "discovery", content="a heron standing in the flooded field", intensity=0.5)
    after = await _phase(["the heron standing in the flooded field last spring"]).execute(state)
    heron = [read_percept(item) for item in after.world.recent_percepts if "heron" in str(item.get("content"))]
    assert heron and heron[0].salience > 0.5
