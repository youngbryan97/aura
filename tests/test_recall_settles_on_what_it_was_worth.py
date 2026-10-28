"""What she holds in mind settles on what the turn turned out to be worth.

Each memory is stamped with the latest reading of her nine channel errors, and
once a turn's worth is read the recall she held through it is weighed again by
how near that reading lies to each memory's, in level and in shape
(core/memory/worth_at_encoding.py, `memory_retrieval.settle_on_what_it_was_worth`).
On seed 73 a nudge towards feeling good turned every percept into a payoff,
her field learned by the sign of it, and what she recalled followed neither.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.affect import what_it_was_worth as worth
from core.affect.what_it_was_worth import CHANNELS, Worth
from core.memory import worth_at_encoding as stamps
from core.phases import memory_retrieval as recall

pytestmark = pytest.mark.unit

GOOD = stamps.encode(np.full(len(CHANNELS), 2.0))
BAD = stamps.encode(np.full(len(CHANNELS), -2.0))
MILD = stamps.encode(np.full(len(CHANNELS), 0.5))


@pytest.fixture(autouse=True)
def fresh():
    worth.reset_for_test()
    yield
    worth.reset_for_test()


def _now(monkeypatch, value: float) -> None:
    monkeypatch.setattr(stamps, "worth_now", lambda: np.full(len(CHANNELS), value))


def _her(pool: dict, held: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        affect=SimpleNamespace(emotions={"joy": 0.1}, mood_baselines={"joy": 0.1}),
        cognition=SimpleNamespace(recall_pool=pool, long_term_memory=list(held), memory_scores=[0.4] * len(held)),
    )


def _pool(*entries: tuple[str, str | None]) -> dict:
    return {
        "salience": 1.0,
        "candidates": [
            {"text": f"[memory score=0.400] {text}", "score": 0.4, "felt": None, "worth": stamp}
            for text, stamp in entries
        ],
    }


def test_a_good_turn_settles_her_on_what_was_made_in_a_good_one(monkeypatch) -> None:
    _now(monkeypatch, 2.0)
    her = _her(_pool(("the prize", GOOD), ("the loss", BAD)), ["[memory score=0.400] the loss"])
    assert recall.settle_on_what_it_was_worth(her) is True
    assert her.cognition.long_term_memory[0].endswith("the prize")
    assert her.cognition.memory_scores[0] > 0.4


def test_a_bad_turn_settles_her_on_what_was_made_in_a_bad_one(monkeypatch) -> None:
    _now(monkeypatch, -2.0)
    her = _her(_pool(("the prize", GOOD), ("the loss", BAD)), ["[memory score=0.400] the prize"])
    recall.settle_on_what_it_was_worth(her)
    assert her.cognition.long_term_memory[0].endswith("the loss")


def test_the_level_counts_and_not_only_the_shape(monkeypatch) -> None:
    """Two stamps of one shape: the one made at this turn's level comes back, as a centred cosine could not tell."""
    _now(monkeypatch, 2.0)
    her = _her(_pool(("a small win", MILD), ("the prize", GOOD)), ["[memory score=0.400] a small win"])
    recall.settle_on_what_it_was_worth(her)
    assert her.cognition.long_term_memory[0].endswith("the prize")


def test_the_pool_is_used_once(monkeypatch) -> None:
    _now(monkeypatch, 2.0)
    her = _her(_pool(("the prize", GOOD), ("the loss", BAD)), ["[memory score=0.400] the loss"])
    recall.settle_on_what_it_was_worth(her)
    assert her.cognition.recall_pool == {}
    before = list(her.cognition.long_term_memory)
    _now(monkeypatch, -2.0)
    assert recall.settle_on_what_it_was_worth(her) is False
    assert her.cognition.long_term_memory == before


def test_an_unmeasured_turn_leaves_what_she_holds(monkeypatch) -> None:
    monkeypatch.setattr(stamps, "worth_now", lambda: None)
    held = ["[memory score=0.400] the loss"]
    her = _her(_pool(("the prize", GOOD), ("the loss", BAD)), held)
    assert recall.settle_on_what_it_was_worth(her) is False
    assert her.cognition.long_term_memory == held
    assert her.cognition.recall_pool == {}


def test_the_feeling_settle_hands_the_pool_on_at_the_scores_it_settled(monkeypatch) -> None:
    pool = {
        "salience": 1.0,
        "candidates": [
            {"text": "[memory score=0.400] a day at the sea", "score": 0.4, "felt": "joy:0.600", "worth": GOOD},
            {"text": "[memory score=0.400] the night the power went", "score": 0.4, "felt": "fear:0.600", "worth": BAD},
        ],
    }
    her = SimpleNamespace(
        affect=SimpleNamespace(emotions={"joy": 0.7, "fear": 0.1}, mood_baselines={"joy": 0.1, "fear": 0.1}),
        cognition=SimpleNamespace(recall_pool=pool, long_term_memory=["[memory score=0.400] the night the power went"], memory_scores=[0.4]),
    )
    recall.settle_on_what_she_feels(her)
    handed = her.cognition.recall_pool
    assert handed.get("felt") is True
    scores = {entry["text"].split("] ", 1)[1]: entry["score"] for entry in handed["candidates"]}
    assert scores["a day at the sea"] > scores["the night the power went"]
    assert recall.settle_on_what_she_feels(her) is False
    _now(monkeypatch, -2.0)
    recall.settle_on_what_it_was_worth(her)
    assert her.cognition.recall_pool == {}


def test_a_stamp_keeps_the_latest_measured_reading_and_nothing_else(monkeypatch) -> None:
    reading = Worth(error={name: 1.5 for name in CHANNELS}, worth=13.5, size=0.9, measured=True)
    monkeypatch.setattr(worth.get_worth_ledger(), "_last", reading)
    written = stamps.stamp({})
    assert np.allclose(stamps.decode(written[stamps.WORTH_KEY]), 1.5)
    assert stamps.stamp({stamps.WORTH_KEY: BAD})[stamps.WORTH_KEY] == BAD
    monkeypatch.setattr(worth.get_worth_ledger(), "_last", Worth())
    assert stamps.WORTH_KEY not in stamps.stamp({})


def test_nearness_is_one_at_the_same_reading_and_one_over_e_at_an_unrelated_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(CHANNELS))
    assert stamps.near(a, a) == pytest.approx(1.0)
    pairs = [stamps.near(rng.normal(size=len(CHANNELS)), rng.normal(size=len(CHANNELS))) for _ in range(4000)]
    assert np.mean([-math.log(p) for p in pairs]) == pytest.approx(1.0, abs=0.05)
    assert stamps.decode("not,a,number") is None


def test_the_end_of_a_turn_settles_recall_on_its_worth(monkeypatch) -> None:
    from core.phases.learning_phase import read_what_the_turn_was_worth
    from core.state.aura_state import AuraState

    settled: list[object] = []
    monkeypatch.setattr(recall, "settle_on_what_it_was_worth", lambda state: settled.append(state) or False)
    state = AuraState()
    state.cognition.last_response = ""
    asyncio.run(read_what_the_turn_was_worth(state))
    assert settled == [state]


def test_the_control_cuts_the_worth_settle_too(monkeypatch) -> None:
    import tools.run_subject_core_content as content
    from core.memory.memory_facade import MemoryFacade

    for name in ("_cued_by_the_percept", "_cued_by_what_she_feels", "settle_on_what_she_feels", "settle_on_what_it_was_worth"):
        monkeypatch.setattr(recall, name, getattr(recall, name))
    monkeypatch.setattr(MemoryFacade, "search_by_feeling", MemoryFacade.search_by_feeling)
    content._cut_feeling_from_recall()
    _now(monkeypatch, 2.0)
    held = ["[memory score=0.400] the loss"]
    her = _her(_pool(("the prize", GOOD), ("the loss", BAD)), held)
    assert recall.settle_on_what_it_was_worth(her) is False
    assert her.cognition.long_term_memory == held


def test_worth_is_added_so_a_high_scoring_memory_can_still_be_passed(monkeypatch) -> None:
    """Closing the gap, a candidate at 0.95 and one at 0.6 could barely trade places; added, worth decides."""
    _now(monkeypatch, 2.0)
    pool = {
        "salience": 1.0,
        "candidates": [
            {"text": "[memory score=0.950] the loss", "score": 0.95, "felt": None, "worth": BAD},
            {"text": "[memory score=0.600] the prize", "score": 0.6, "felt": None, "worth": GOOD},
        ],
    }
    her = _her(pool, ["[memory score=0.950] the loss"])
    assert recall.settle_on_what_it_was_worth(her) is True
    assert her.cognition.long_term_memory[0].endswith("the prize")
    assert all(0.0 <= score <= 1.0 for score in her.cognition.memory_scores)
