"""A turn's answer budget comes from the time it has, not the time a turn may have.

LIVE 2026-08-30: 4,167 tokens budgeted for a turn with 166.9s actually left.
Generation was aborted at the deadline and every token produced in those 166
seconds was discarded, and the person got "I couldn't get to an answer I'd
stand behind" after five minutes.

The budget was sized against USER_FACING_COMPLETION_DEADLINE_MAX_S — the most a
turn may EVER be given — while generation was handed the remainder. The gap
between them is exactly what the turn had already spent, and on a second
attempt that is most of the clock.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate
from core.brain.llm import thinking_reserve
from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

_allowed = InferenceGate._tokens_the_turn_is_allowed_to_take


@pytest.fixture(autouse=True)
def _a_measured_machine(monkeypatch):
    """A rate on the record, because with none the budget is always nothing.

    Twelve tokens a second, which is what this 27B actually decodes at. The
    point of the whole mechanism is that the number comes from the machine, so
    a test that runs without one measures the fallback and not the fix.
    """
    from collections import deque

    measured = deque(
        [(length, 12.0) for length in (512, 1024, 2048, 4096, 8192)] * 4, maxlen=64
    )
    monkeypatch.setattr(thinking_reserve, "_rates", measured)
    monkeypatch.setattr(thinking_reserve, "_restore_once", lambda: None)


def test_less_time_buys_fewer_tokens():
    assert _allowed(seconds=60.0) < _allowed(seconds=180.0)


def test_a_turn_with_almost_no_clock_left_is_not_promised_a_long_answer():
    assert _allowed(seconds=5.0) < _allowed(seconds=300.0)


def test_it_never_exceeds_what_a_turn_may_ever_be_given():
    """A caller handing in a wild number cannot buy more than the policy allows."""
    assert _allowed(seconds=10_000.0) == _allowed(
        seconds=float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
    )


def test_no_time_given_falls_back_to_the_policy_maximum():
    """The old behaviour, kept for callers that genuinely do not know."""
    assert _allowed(seconds=0.0) == _allowed(
        seconds=float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
    )


@pytest.mark.parametrize("seconds", [30.0, 90.0, 150.0, 240.0])
def test_the_budget_is_monotone_in_the_clock(seconds):
    assert _allowed(seconds=seconds) <= _allowed(seconds=seconds + 30.0)


def test_a_model_ceiling_still_caps_it():
    assert _allowed(1024, seconds=float(USER_FACING_COMPLETION_DEADLINE_MAX_S)) <= 1024


# --- and the variable that decides it exists on every path ----------------


def test_whether_anybody_is_waiting_is_decided_before_the_branch():
    """It was assigned inside one branch and used outside it.

    A turn whose clock was blocked — a benchmark, a resource-stakes hold, a
    desktop execution contract — reached the use having never made the
    assignment, and raised UnboundLocalError in the middle of generating.
    """
    import ast
    import inspect
    from pathlib import Path

    from core.brain import inference_gate

    source = Path(inspect.getfile(inference_gate)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned_at = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_is_user_facing"
                for target in child.targets
            )
        ]
        if not assigned_at:
            continue
        used_at = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and child.id == "_is_user_facing"
            and isinstance(child.ctx, ast.Load)
        ]
        assert min(assigned_at) < min(used_at), (
            "_is_user_facing is read before every path has assigned it"
        )


# --- reading the prompt comes out of the same clock -----------------------


def test_a_long_prompt_leaves_less_room_for_the_answer():
    """Only the writing was ever counted against the clock.

    A ten-thousand-character prompt takes about twenty-six seconds to read at
    the measured rate, on a turn whose first-token ceiling alone was a hundred
    and twenty seconds.
    """
    short = _allowed(seconds=167.0, prompt_chars=2000)
    long = _allowed(seconds=167.0, prompt_chars=10_364)
    assert long < short


def test_the_prompt_is_free_when_nobody_says_how_big_it_is():
    """Callers that do not know keep the behaviour they had."""
    assert _allowed(seconds=167.0, prompt_chars=0) >= _allowed(
        seconds=167.0, prompt_chars=2000
    )


def test_a_prompt_that_eats_the_whole_clock_buys_no_answer():
    assert _allowed(seconds=10.0, prompt_chars=200_000) == 0
