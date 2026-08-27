"""A budget at least as long as this machine has been measured needing.

A question's deadline belongs to the question. Given fifty seconds for a
prompt that takes longer than that to read and answer, the decode stops
politely part way and a real reply is thrown away for an apology. Measured
live 2026-08-26: "Request deadline reached at token 207", and what the person
got was "I couldn't get a clear enough answer together."
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate, _long_enough_to_answer
from core.brain.llm import mlx_client


@pytest.fixture(autouse=True)
def measured(monkeypatch):
    monkeypatch.setitem(mlx_client._HOST_RATES, "prefill", 200.0)
    monkeypatch.setitem(mlx_client._HOST_RATES, "decode", 9.0)


def test_a_question_that_needs_longer_gets_longer():
    assert _long_enough_to_answer("x" * 6000, 512, 50.0) > 50.0


def test_a_question_that_fits_is_left_alone():
    assert _long_enough_to_answer("hi", 64, 50.0) == 50.0


def test_a_caller_that_allowed_more_keeps_it():
    assert _long_enough_to_answer("hi", 8, 120.0) == 120.0


def test_nobody_waits_past_the_ceiling_the_gate_already_enforces():
    asked = _long_enough_to_answer("x" * 900_000, 4000, 50.0)
    assert asked == InferenceGate._MAX_REQUEST_TIMEOUT_S


def test_writing_the_answer_is_charged_as_well_as_reading_the_question():
    reading_only = _long_enough_to_answer("x" * 6000, 0, 1.0)
    with_answer = _long_enough_to_answer("x" * 6000, 512, 1.0)
    assert with_answer - reading_only == pytest.approx(512 / 9.0, abs=1.0)


def test_a_machine_that_cannot_be_asked_leaves_the_budget_alone(monkeypatch):
    monkeypatch.delattr(mlx_client, "time_a_prompt_needs")
    assert _long_enough_to_answer("x" * 900_000, 4000, 50.0) == 50.0


def test_it_is_applied_where_the_prompt_and_the_token_budget_are_both_known():
    import inspect

    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    assert "timeout_val = _long_enough_to_answer(prompt, max_tokens, timeout_val)" in source
