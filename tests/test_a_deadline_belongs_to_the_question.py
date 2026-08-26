"""A question's deadline is sized by the question, not by a constant.

Measured live on 2026-08-26: every approach question in a game of 2048 timed
out at the eight seconds that suit a one-word move, so she played the whole
game with no plan held and nothing anywhere said why.
"""

from __future__ import annotations

import pytest

from core.agency.her_reasoning import time_this_question_needs
from core.brain.llm import mlx_client


@pytest.fixture(autouse=True)
def measured(monkeypatch):
    monkeypatch.setitem(mlx_client._HOST_RATES, "prefill", 700.0)
    monkeypatch.setitem(mlx_client._HOST_RATES, "decode", 20.0)


def test_a_long_question_wanting_a_long_answer_gets_longer_than_a_short_one():
    short = time_this_question_needs("pick a move", 8, 8.0)
    long = time_this_question_needs("x" * 12000, 400, 8.0)
    assert long > short


def test_a_caller_s_budget_is_never_shortened():
    assert time_this_question_needs("pick a move", 8, 45.0) == 45.0


def test_reading_the_prompt_is_charged_at_the_measured_rate():
    # 14000 chars is 3500 tokens; at 700 tok/s that is 5s of reading, and the
    # headroom the client applies makes it 15s.
    allowed = time_this_question_needs("x" * 14000, 0, 0.0)
    assert 14.0 < allowed < 16.0


def test_writing_the_answer_is_charged_too():
    reading_only = time_this_question_needs("x" * 14000, 0, 0.0)
    with_answer = time_this_question_needs("x" * 14000, 400, 0.0)
    assert with_answer - reading_only == pytest.approx(20.0, abs=0.5)


def test_an_unmeasured_host_is_still_given_a_number():
    mlx_client._HOST_RATES["prefill"] = 0.0
    mlx_client._HOST_RATES["decode"] = 0.0
    assert time_this_question_needs("x" * 4000, 200, 0.0) > 0.0


def test_a_mind_that_cannot_be_asked_leaves_the_budget_alone(monkeypatch):
    monkeypatch.delattr(mlx_client, "time_a_prompt_needs")
    assert time_this_question_needs("x" * 40000, 900, 8.0) == 8.0
