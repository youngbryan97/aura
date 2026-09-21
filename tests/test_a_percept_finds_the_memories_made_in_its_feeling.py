"""A percept finds the memories that were made in the feeling it moves.

In the content runs a threat, an error and a disconnection brought back exactly
the same memories from the same fork: recall joined the percept to the question
as one appended word, nothing she had stored used any of those words, and a
memory kept only its valence, so nothing else about the percept could reach the
search. These pin the other route: every memory keeps what she felt above her
own baseline when it was made, and a percept's appraisal finds the memories
whose feeling lies on the emotions it moves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.felt_at_encoding import carries, decode, distinctive, encode, felt_now, stamp
from core.phases.memory_retrieval import MemoryRetrievalPhase
from core.state.aura_state import AuraState
from core.state.percepts import PERCEPT_EMOTIONS, emit_percept

pytestmark = pytest.mark.unit


def test_what_she_felt_is_what_stood_above_her_own_baseline() -> None:
    state = AuraState.default()
    assert felt_now(state.affect) == {}, "at rest nothing was felt, so a memory made at rest carries nothing"
    state.affect.emotions["dread"] = state.affect.mood_baselines["dread"] + 0.3
    state.affect.emotions["hope"] = state.affect.mood_baselines["hope"] - 0.05
    felt = felt_now(state.affect)
    assert felt == {"dread": pytest.approx(0.3)}


def test_the_stamp_is_flat_text_and_reads_back() -> None:
    felt = {"dread": 0.3, "fear": 0.1}
    text = encode(felt)
    assert text == "dread:0.300 fear:0.100"
    assert decode(text) == pytest.approx(felt)
    assert decode("not a stamp") == {}
    assert decode(None) == {}


def test_a_stamp_already_written_is_left_as_its_writer_made_it() -> None:
    state = AuraState.default()
    state.affect.emotions["joy"] = 0.8
    metadata = {"felt": "dread:0.300"}
    stamp(metadata, state.affect)
    assert metadata["felt"] == "dread:0.300"
    fresh: dict = {}
    stamp(fresh, state.affect)
    assert "joy" in decode(fresh["felt"])


def test_carries_is_the_share_of_the_feeling_on_the_named_emotions() -> None:
    threat = PERCEPT_EMOTIONS["threat_detected"]
    assert carries(threat, "dread:0.300 fear:0.100") == pytest.approx(1.0)
    assert carries(threat, "dread:0.200 joy:0.200") == pytest.approx(0.5)
    assert carries(threat, "joy:0.400") == 0.0
    assert carries(threat, "") == 0.0


def test_every_facade_write_carries_the_feeling_it_was_made_in(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.memory.memory_facade import MemoryFacade

    state = AuraState.default()
    state.affect.emotions["fear"] = 0.6
    repo = SimpleNamespace(_current=state)
    real_get = ServiceContainer.get
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: repo if name == "state_repository" else real_get(name, default=default)),
    )
    facade = MemoryFacade.__new__(MemoryFacade)
    written = facade._stamp_felt({})
    assert decode(written["felt"])["fear"] == pytest.approx(0.6)


def _phase(memories: list[dict]) -> MemoryRetrievalPhase:
    async def _search(query, limit=5):  # noqa: ANN001, ANN202
        return [dict(item) for item in memories]

    async def _get_hot_memory(limit=3):  # noqa: ANN001, ANN202
        return {"recent_episodes": []}

    facade = SimpleNamespace(search=_search, get_hot_memory=_get_hot_memory)
    return MemoryRetrievalPhase(
        SimpleNamespace(get=lambda name, default=None: facade if name == "memory_facade" else default)
    )


def _memories() -> list[dict]:
    """Twelve memories alike in every way the search can see, made in two feelings."""
    out = []
    for index in range(6):
        out.append(
            {
                "content": f"the morning the shed door stuck, part {index}",
                "score": 0.5,
                "metadata": {"importance": 0.5, "emotional_valence": 0.0, "felt": "dread:0.300 fear:0.200"},
            }
        )
        out.append(
            {
                "content": f"the afternoon the letters came, part {index}",
                "score": 0.5,
                "metadata": {"importance": 0.5, "emotional_valence": 0.0, "felt": "joy:0.300 pride:0.200"},
            }
        )
    return out


async def _recall(kind: str) -> list[str]:
    # A question asked again goes one deeper, which is a difference between
    # the two recalls that has nothing to do with the percept.
    from core.memory.reliving import reset_for_test

    reset_for_test()
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "assistant", "content": "I will look into it."})
    state.cognition.current_objective = "keep the workshop in order"
    emit_percept(state.world, kind, content=kind.replace("_", " "), intensity=1.0)
    after = await _phase(_memories()).execute(state)
    return list(after.cognition.long_term_memory)


@pytest.mark.asyncio
async def test_a_threat_and_an_achievement_bring_back_the_memories_made_in_their_feelings() -> None:
    threat = await _recall("threat_detected")
    achieved = await _recall("goal_achieved")
    assert threat and achieved
    assert all("shed door" in text for text in threat)
    assert all("letters came" in text for text in achieved)
    assert set(threat) != set(achieved)


@pytest.mark.asyncio
async def test_without_the_stamp_the_two_percepts_recall_the_same() -> None:
    """The control: the words alone cannot tell these percepts apart."""
    bare = [{**item, "metadata": {k: v for k, v in item["metadata"].items() if k != "felt"}} for item in _memories()]

    async def recall(kind: str) -> set[str]:
        state = AuraState.default()
        state.cognition.working_memory.append({"role": "assistant", "content": "I will look into it."})
        state.cognition.current_objective = "keep the workshop in order"
        emit_percept(state.world, kind, content=kind.replace("_", " "), intensity=1.0)
        from core.memory.reliving import reset_for_test

        reset_for_test()
        after = await _phase(bare).execute(state)
        return {text.split("] ", 1)[1] for text in after.cognition.long_term_memory}

    assert await recall("threat_detected") == await recall("goal_achieved")


def test_a_feeling_every_memory_shares_does_not_choose_between_them() -> None:
    """Cue overload: what is common to the pool is taken off before the percept looks."""
    common = {"trust": 0.8, "joy": 0.8, "fear": 0.1, "dread": 0.1}
    shed = {**common, "dread": 0.3}
    letters = {**common, "joy": 0.9}
    raw_gap = carries(PERCEPT_EMOTIONS["threat_detected"], shed) - carries(PERCEPT_EMOTIONS["threat_detected"], letters)
    apart = distinctive({"shed": encode(shed), "letters": encode(letters)})
    assert apart["shed"] == {"dread": pytest.approx(0.1)}
    assert apart["letters"] == {"joy": pytest.approx(0.05)}
    set_gap = carries(PERCEPT_EMOTIONS["threat_detected"], apart["shed"]) - carries(PERCEPT_EMOTIONS["threat_detected"], apart["letters"])
    assert raw_gap < 0.15, "on the whole profile the two memories are nearly the same to a threat"
    assert set_gap == pytest.approx(1.0), "on what sets them apart the threat finds only the shed"


def test_a_pool_of_one_keeps_its_feeling_whole() -> None:
    assert distinctive({"only": "dread:0.300"}) == {"only": {"dread": pytest.approx(0.3)}}
    assert distinctive({"none": ""}) == {"none": {}}
