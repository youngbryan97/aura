"""What she holds in mind settles on the feeling a percept left her with.

Recall runs before the affect phase, so the memories she held through a turn
were cued by how she met a percept and never by what it made her feel. On the
seed 31 and 73 content runs at 572f3dc60 her affect moved furthest under a
displacement and what she recalled followed it least. Recall now keeps the pool
it weighed, and the affect phase weighs it again by the feeling she has once she
has felt what arrived (`memory_retrieval.settle_on_what_she_feels`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.phases import memory_retrieval as recall

pytestmark = pytest.mark.unit

BASELINES = {"joy": 0.1, "fear": 0.1, "calm": 0.1}


def _her(feeling: dict[str, float], pool: dict, held: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        affect=SimpleNamespace(emotions=dict(feeling), mood_baselines=dict(BASELINES)),
        cognition=SimpleNamespace(recall_pool=pool, long_term_memory=list(held), memory_scores=[0.4] * len(held)),
    )


def _pool() -> dict:
    """Two memories the words found equally, one made in joy and one in fear, and one made at rest."""
    return {
        "salience": 1.0,
        "candidates": [
            {"text": "[memory score=0.400] a day at the sea", "score": 0.4, "felt": "joy:0.600"},
            {"text": "[memory score=0.400] the night the power went", "score": 0.4, "felt": "fear:0.600"},
            {"text": "[memory score=0.400] an ordinary errand", "score": 0.4, "felt": "calm:0.300"},
        ],
    }


def test_joy_after_the_percept_settles_her_on_the_joyful_memory() -> None:
    her = _her({"joy": 0.7, "fear": 0.1, "calm": 0.1}, _pool(), ["[memory score=0.400] the night the power went"])
    assert recall.settle_on_what_she_feels(her) is True
    assert her.cognition.long_term_memory[0].endswith("a day at the sea")
    assert her.cognition.memory_scores[0] > 0.4


def test_fear_after_the_same_percept_settles_her_on_the_frightening_one() -> None:
    her = _her({"joy": 0.1, "fear": 0.7, "calm": 0.1}, _pool(), ["[memory score=0.400] a day at the sea"])
    recall.settle_on_what_she_feels(her)
    assert her.cognition.long_term_memory[0].endswith("the night the power went")


def test_she_holds_as_many_as_recall_kept() -> None:
    held = ["[memory score=0.400] a day at the sea", "[memory score=0.400] an ordinary errand"]
    her = _her({"joy": 0.1, "fear": 0.7, "calm": 0.1}, _pool(), held)
    recall.settle_on_what_she_feels(her)
    assert len(her.cognition.long_term_memory) == 2
    assert len(her.cognition.memory_scores) == 2


def test_the_pool_is_used_once() -> None:
    her = _her({"joy": 0.7, "fear": 0.1, "calm": 0.1}, _pool(), ["[memory score=0.400] an ordinary errand"])
    recall.settle_on_what_she_feels(her)
    assert her.cognition.recall_pool == {}
    before = list(her.cognition.long_term_memory)
    her.affect.emotions = {"joy": 0.1, "fear": 0.7, "calm": 0.1}
    assert recall.settle_on_what_she_feels(her) is False
    assert her.cognition.long_term_memory == before


def test_an_ordinary_mood_leaves_the_words_in_charge() -> None:
    held = ["[memory score=0.400] the night the power went"]
    her = _her({"joy": 0.1, "fear": 0.1, "calm": 0.1}, _pool(), held)
    assert recall.settle_on_what_she_feels(her) is False
    assert her.cognition.long_term_memory == held


def test_no_pool_and_no_recall_change_nothing() -> None:
    her = _her({"joy": 0.7}, {}, ["[memory score=0.400] an ordinary errand"])
    assert recall.settle_on_what_she_feels(her) is False
    empty = _her({"joy": 0.7}, _pool(), [])
    assert recall.settle_on_what_she_feels(empty) is False


def test_what_she_holds_brings_its_feeling_back(monkeypatch) -> None:
    """Holding the joyful memory moves joy toward her baseline plus what it carried, at the recall gain."""
    from core.consciousness import substrate_gates

    monkeypatch.setattr(substrate_gates.get_substrate_gates(), "multiplier", lambda name: 0.5)
    her = _her({"joy": 0.7, "fear": 0.1, "calm": 0.1}, _pool(), ["[memory score=0.400] the night the power went"])
    recall.settle_on_what_she_feels(her)
    assert her.cognition.long_term_memory[0].endswith("a day at the sea")
    # Baseline 0.1 plus the 0.6 the memory carried is 0.7, so joy, already at 0.7, stays there.
    assert her.affect.emotions["joy"] == pytest.approx(0.7)


def test_an_emotion_no_held_memory_carried_eases_toward_her_baseline(monkeypatch) -> None:
    from core.consciousness import substrate_gates

    monkeypatch.setattr(substrate_gates.get_substrate_gates(), "multiplier", lambda name: 0.5)
    pool = {"salience": 1.0, "candidates": [{"text": "[memory score=0.400] a day at the sea", "score": 0.4, "felt": "joy:0.600"}]}
    her = _her({"joy": 0.1, "fear": 0.5, "calm": 0.1}, pool, ["[memory score=0.400] a day at the sea"])
    recall.settle_on_what_she_feels(her)
    # Gain 0.5: joy goes half way from 0.1 to 0.7, fear half way from 0.5 to its baseline 0.1.
    assert her.affect.emotions["joy"] == pytest.approx(0.4)
    assert her.affect.emotions["fear"] == pytest.approx(0.3)


def test_memories_that_carry_no_feeling_leave_her_feeling_alone() -> None:
    pool = {"salience": 1.0, "candidates": [{"text": "[memory score=0.400] an errand", "score": 0.4, "felt": None}]}
    her = _her({"joy": 0.7, "fear": 0.1, "calm": 0.1}, pool, ["[memory score=0.400] an errand"])
    before = dict(her.affect.emotions)
    recall.settle_on_what_she_feels(her)
    assert her.affect.emotions == before
