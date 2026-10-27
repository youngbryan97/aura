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
