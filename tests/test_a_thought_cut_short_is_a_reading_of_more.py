"""A private channel the decoder closed before the model had is a reading of "more than this", not of what it cost.

LIVE 27 November 2026: 18 of the cortex's 128 readings of what its private channel cost were 360 to 372 tokens,
generations closed at a 368-token bound and counted as finished, and the reserve stood at 371. The channel was closed at
368 tokens 75 times that day, with the room for it added on top of the answer and unused, and a step of hers that ends
mid-thought writes no call.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_worker, thinking_reserve

pytestmark = pytest.mark.unit

MODEL = "a-27b-for-this-test"


@pytest.fixture(autouse=True)
def nothing_measured():
    # forget, not reset_for_test: a save folds the stored readings back in, and an earlier test's are on disk.
    thinking_reserve.forget()
    yield
    thinking_reserve.forget()


def _finished(tokens: int) -> None:
    thinking_reserve.record_reasoning_cost(reasoning_chars=tokens, surface_chars=0, generated_tokens=tokens, model=MODEL)


def test_with_nothing_cut_the_reserve_is_the_window_percentile_as_before():
    for tokens in range(1, 101):
        _finished(tokens)
    assert thinking_reserve.reserve_tokens(MODEL) == 91


def test_readings_cut_at_the_bound_do_not_set_the_reserve_at_the_bound():
    for _ in range(100):
        _finished(100)
    for _ in range(30):
        thinking_reserve.record_channel_cut_short(budget_tokens=368, model=MODEL)
        _finished(368)  # the cut generation's own spend, which is not what reasoning cost
    assert thinking_reserve.observations(MODEL) == 100, "a cut generation's spend is not read as a finished one"
    assert thinking_reserve.reserve_tokens(MODEL) == 736, "past every reading: twice the largest"


def test_an_exact_reading_past_the_cuts_is_where_the_percentile_falls():
    for _ in range(80):
        _finished(100)
    for _ in range(10):
        thinking_reserve.record_channel_cut_short(budget_tokens=368, model=MODEL)
        _finished(368)
    for tokens in (500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400):
        _finished(tokens)
    # The product-limit estimate worked out here, step by step, rather than trusted.
    at_risk, surviving, quantile = 100, 1.0, None
    surviving *= 1 - 80 / at_risk
    at_risk -= 80 + 10
    for tokens in (500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400):
        surviving *= 1 - 1 / at_risk
        at_risk -= 1
        if surviving <= 0.1 + 1e-12:
            quantile = tokens
            break
    assert thinking_reserve.reserve_tokens(MODEL) == quantile


def test_the_channel_may_take_the_room_added_for_it_on_top_of_the_answer(monkeypatch):
    monkeypatch.setattr(mlx_worker, "_reasoning_reserve_tokens", lambda model="": 736)
    monkeypatch.setattr(mlx_worker, "_the_private_channel_budget",
                        lambda job, max_tokens: min(int(max_tokens), 1500))  # the clock allows up to 1,500
    channel, room = mlx_worker._channel_and_room({}, 384, MODEL)
    assert room == 384 + 736 and channel == 736, "the answer keeps its own 384"
    monkeypatch.setattr(mlx_worker, "_the_private_channel_budget", lambda job, max_tokens: min(int(max_tokens), 500))
    channel, room = mlx_worker._channel_and_room({}, 384, MODEL)
    assert channel == 500 and room == 384 + 736, "as far as the clock allows"


def test_a_runaways_proof_does_not_size_every_channel_once_a_smaller_budget_was_enough():
    """LIVE 27 November 2026, 15:52: read raw, one runaway's proof of 7,955 sized every thinking channel."""
    for tokens in range(1, 101):
        _finished(tokens)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=7955, model=MODEL)
    thinking_reserve.record_budget_that_finished_thinking(budget_tokens=32, model=MODEL)
    assert thinking_reserve.reserve_tokens(MODEL) == 91, "the measured cost, not the runaway"
