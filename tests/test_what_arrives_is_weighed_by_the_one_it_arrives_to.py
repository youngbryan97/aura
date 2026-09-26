"""Salience is hers, not the producer's.

`read_percept` falls back to intensity when nobody sets salience apart, and
nobody did, so the perception domain carried the environment's own number twice
and nothing of her. The workspace's perception bid, the temporal binding weight
and the world's "worth keeping" filter all read it. On the seed-7 run of 24
September the cheapest bipartition was perception against the other nine
domains: a domain that is a function of the world alone is severable from a
mind by construction.
"""

from __future__ import annotations

import random

import pytest

from core.perception.salience import (
    ENOUGH,
    RankLedger,
    attention_of,
    reset_for_test,
    salience_of,
)
from core.state.percepts import read_percept


@pytest.fixture(autouse=True)
def _fresh():
    reset_for_test()
    yield
    reset_for_test()


# ── the ledger ───────────────────────────────────────────────────────────


def test_the_middle_until_there_is_something_to_rank_against():
    book = RankLedger()
    assert all(book.note(value) == 0.5 for value in range(ENOUGH))


def test_a_reading_above_everything_before_it_ranks_at_the_top():
    book = RankLedger()
    for _ in range(ENOUGH + 4):
        book.note(0.1)
    assert book.note(0.9) == pytest.approx(1.0)


def test_a_run_of_identical_readings_sits_in_the_middle():
    book = RankLedger()
    for _ in range(ENOUGH + 8):
        book.note(0.4)
    assert book.note(0.4) == pytest.approx(0.5)


def test_a_reading_that_is_not_a_number_is_the_middle():
    book = RankLedger()
    assert book.note("loud") == 0.5
    assert book.note(float("nan")) == 0.5


def test_the_window_bounds_what_is_remembered():
    book = RankLedger(window=16)
    for _ in range(200):
        book.note(random.random())
    assert book.seen() == 16


# ── salience ─────────────────────────────────────────────────────────────


def _warm(rng: random.Random, turns: int = 60) -> None:
    for _ in range(turns):
        salience_of(rng.random(), 0.3 + 0.4 * rng.random())


def test_the_same_arrival_is_worth_more_when_she_is_more_awake():
    rng = random.Random(5)
    _warm(rng)
    calm = salience_of(0.6, 0.05)
    _warm(rng)
    awake = salience_of(0.6, 0.99)
    assert awake > calm


def test_a_rank_cannot_saturate():
    rng = random.Random(7)
    seen = [salience_of(rng.random(), 0.2 + 0.6 * rng.random()) for _ in range(400)]
    settled = seen[100:]
    assert max(settled) - min(settled) > 0.5
    at_the_top = sum(1 for value in settled if value >= 0.999)
    assert at_the_top / len(settled) < 0.05


def test_half_her_arrivals_sit_above_a_half():
    """The filters that test against a half keep the meaning they were written with."""
    rng = random.Random(11)
    seen = [salience_of(rng.random(), 0.2 + 0.6 * rng.random()) for _ in range(600)]
    settled = seen[100:]
    above = sum(1 for value in settled if value > 0.5) / len(settled)
    assert 0.35 < above < 0.65


def test_salience_stops_being_a_copy_of_intensity():
    rng = random.Random(13)
    pairs = []
    for _ in range(200):
        strength = rng.random()
        pairs.append((strength, salience_of(strength, 0.2 + 0.6 * rng.random())))
    settled = pairs[60:]
    assert sum(1 for a, b in settled if abs(a - b) > 1e-9) > 0.9 * len(settled)


def test_attention_is_her_arousal_against_her_own():
    book = RankLedger()
    for _ in range(ENOUGH + 4):
        attention_of(0.2, ledger=book)
    assert attention_of(0.9, ledger=book) > 0.9


# ── what the loop does with it ───────────────────────────────────────────


def _loop():
    from core.phases.proprioceptive_loop import ProprioceptiveLoop

    return ProprioceptiveLoop(None)


def test_the_loop_weighs_what_arrived():
    from core.state.aura_state import AuraState

    state = AuraState.default()
    rng = random.Random(17)
    for _ in range(ENOUGH + 4):
        salience_of(rng.random(), 0.5)
    state.world.recent_percepts = [
        {"type": "interaction", "content": "a message", "intensity": 0.9}
    ]
    state.affect.arousal = 0.95
    _loop()._weigh_what_arrived(state)
    item = state.world.recent_percepts[0]
    assert "salience" in item
    assert read_percept(item).salience == item["salience"]


def test_a_producer_that_set_salience_on_purpose_is_left_alone():
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.world.recent_percepts = [
        {"type": "vision", "content": "the screen", "intensity": 0.2, "salience": 0.88}
    ]
    _loop()._weigh_what_arrived(state)
    assert state.world.recent_percepts[0]["salience"] == 0.88


def test_a_percept_that_is_not_a_mapping_does_not_stop_the_rest():
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.world.recent_percepts = [
        "a bare string",
        {"type": "interaction", "content": "a message", "intensity": 0.5},
    ]
    _loop()._weigh_what_arrived(state)
    assert "salience" in state.world.recent_percepts[1]
