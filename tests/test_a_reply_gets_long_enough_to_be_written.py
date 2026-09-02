"""A budget and a length that fit each other, and fit what the caller waits.

Given fifty seconds for a prompt that takes longer than that to read and
answer, the decode stops part way and a real reply is thrown away for an
apology. Measured live 2026-08-26: "Request deadline reached at token 207",
and what the person got was "I couldn't get a clear enough answer together."

Lifting the clock alone is the other half of the same mistake. Asked for
everything it wanted, the same question came back with a budget of five
hundred and seventy-eight seconds — longer than anyone waits, and longer than
the caller itself would wait, so the answer never arrived at all.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import (
    A_REAL_ANSWER,
    InferenceGate,
    fit_the_answer_to_the_time,
)
from core.brain.llm import mlx_client


@pytest.fixture(autouse=True)
def measured(monkeypatch):
    monkeypatch.setitem(mlx_client._HOST_RATES, "prefill", 200.0)
    monkeypatch.setitem(mlx_client._HOST_RATES, "decode", 9.0)


def test_a_question_that_already_fits_is_left_alone():
    assert fit_the_answer_to_the_time("hi", 64, 50.0) == (50.0, 64)


def test_what_does_not_fit_comes_out_of_the_length_not_the_clock():
    budget, tokens = fit_the_answer_to_the_time("x" * 6000, 512, 50.0)
    assert budget == 50.0
    assert tokens < 512


def test_and_what_is_left_is_still_worth_reading():
    _budget, tokens = fit_the_answer_to_the_time("x" * 6000, 512, 50.0)
    assert tokens >= A_REAL_ANSWER


def test_reading_the_question_is_not_optional_so_the_clock_gives_way_to_it():
    """Where reading alone outlasts the budget, there is nothing to trim."""
    budget, tokens = fit_the_answer_to_the_time("x" * 60_000, 2000, 50.0)
    assert budget > 50.0
    assert tokens == A_REAL_ANSWER


def test_nobody_waits_past_the_ceiling_the_gate_already_enforces():
    budget, _tokens = fit_the_answer_to_the_time("x" * 9_000_000, 4000, 50.0)
    assert budget <= InferenceGate._MAX_REQUEST_TIMEOUT_S


def test_a_caller_that_allowed_more_keeps_it():
    assert fit_the_answer_to_the_time("hi", 8, 120.0) == (120.0, 8)


def test_a_machine_that_cannot_be_asked_changes_nothing(monkeypatch):
    monkeypatch.delattr(mlx_client, "time_a_prompt_needs")
    assert fit_the_answer_to_the_time("x" * 60_000, 2000, 50.0) == (50.0, 2000)


def test_asking_for_no_answer_at_all_is_left_as_it_is():
    assert fit_the_answer_to_the_time("x" * 6000, 0, 50.0) == (50.0, 0)


def test_it_is_applied_where_the_prompt_and_the_length_are_both_known():
    import inspect

    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    assert (
        "settled_timeout_val, settled_max_tokens = fit_the_answer_to_the_time("
        in source
    )


def test_it_prices_the_prompt_after_final_compaction():
    """Discarded rich context cannot set the clock for compacted bytes."""

    import inspect

    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    compacted = source.index("system_prompt, messages = self._fit_prompt_to_window(")
    priced = source.index(
        "settled_timeout_val, settled_max_tokens = fit_the_answer_to_the_time("
    )
    assert priced > compacted
    window = source[priced : priced + 1200]
    assert "dispatched_prompt_text" in window
    assert "request_deadline.with_timeout(timeout_val)" in window


def test_a_protected_length_buys_time_instead_of_being_shortened():
    """The protected capability lane overrides the resource envelope to get a
    budget, and fitting the answer to the clock used to undo that override
    without knowing it existed."""
    prompt = "x" * 6000
    _budget, cut = fit_the_answer_to_the_time(prompt, 384, 20.0)
    assert cut < 384, "this prompt and clock have to force a cut, or the test is empty"

    budget, kept = fit_the_answer_to_the_time(prompt, 384, 20.0, floor=384)
    assert kept == 384, "the protected floor was cut anyway"
    assert budget > 20.0, "the clock did not stretch to afford the floor"


def test_a_floor_the_longest_clock_cannot_afford_is_cut_and_said_out_loud(caplog):
    """A protection that quietly did nothing is worse than a measured limit."""
    import logging

    with caplog.at_level(logging.WARNING):
        _budget, kept = fit_the_answer_to_the_time(
            "x" * 400_000, 4_000_000, 20.0, floor=4_000_000
        )
    assert kept < 4_000_000
    assert any("protected answer floor" in r.message for r in caplog.records), (
        "the floor was cut and nothing said so"
    )


def test_no_floor_leaves_the_old_behaviour_exactly_as_it_was():
    prompt = "x" * 6000
    assert fit_the_answer_to_the_time(prompt, 512, 50.0) == fit_the_answer_to_the_time(
        prompt, 512, 50.0, floor=0
    )
