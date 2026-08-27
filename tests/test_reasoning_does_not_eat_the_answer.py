"""A thinking model's budget is spent twice; the ceiling has to know that.

LIVE, 2026-08-27: a question about a number sequence planned at
``max_tokens=1536`` served 1,469 characters and stopped mid-paragraph, before
the part the person had asked for. Under one character per token is not prose;
the reasoning channel had taken the rest of the budget.
"""

from __future__ import annotations

import pytest

from core.brain.llm import thinking_reserve


@pytest.fixture(autouse=True)
def _clean() -> None:
    thinking_reserve.forget()
    yield
    thinking_reserve.forget()


def test_nothing_is_reserved_before_anything_is_measured() -> None:
    assert thinking_reserve.reserve_tokens() == 0
    thinking_reserve.record_reasoning_cost(
        reasoning_chars=4000, surface_chars=400, generated_tokens=1200
    )
    assert thinking_reserve.reserve_tokens() == 0


def test_the_reserve_is_what_reasoning_has_cost() -> None:
    # Four fifths of every generation went to the private channel.
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=4000, surface_chars=1000, generated_tokens=1000
        )
    assert thinking_reserve.reserve_tokens() == 800


def test_the_ratio_comes_from_the_generation_not_from_a_constant() -> None:
    # Same character split, half the tokens: a denser tokenizer must halve the
    # reserve without an edit anywhere.
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=4000, surface_chars=1000, generated_tokens=500
        )
    assert thinking_reserve.reserve_tokens() == 400


def test_a_generation_that_did_not_think_costs_nothing() -> None:
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=0, surface_chars=1500, generated_tokens=400
        )
    assert thinking_reserve.reserve_tokens() == 0


def test_the_window_forgets_an_older_model() -> None:
    for _ in range(200):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=9000, surface_chars=1000, generated_tokens=1000
        )
    for _ in range(200):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=1000, surface_chars=9000, generated_tokens=1000
        )
    assert thinking_reserve.reserve_tokens() == 100


def test_rubbish_readings_are_dropped_rather_than_counted() -> None:
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=-1, surface_chars=0, generated_tokens=0
        )
        thinking_reserve.record_reasoning_cost(
            reasoning_chars="x", surface_chars=None, generated_tokens=1
        )
    assert thinking_reserve.observations() == 0
    assert thinking_reserve.reserve_tokens() == 0


def test_the_worker_widens_the_ceiling_only_while_thinking() -> None:
    from core.brain.llm.mlx_worker import _reasoning_reserve_tokens

    assert _reasoning_reserve_tokens() == 0
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=2000, surface_chars=2000, generated_tokens=800
        )
    assert _reasoning_reserve_tokens() == 400
