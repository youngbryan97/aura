"""The private channel had no end, so nothing could afford to open it.

The reasoning template opens a channel the model closes with `</think>` when
it is finished. Nothing bounded that, so what the channel COST could only be
read off the generations that had already run away with it — a 90th percentile
over ten samples, standing at 3,916 tokens for the resident 27B against turns
budgeted at 1,024. The gate refused every one, the channel stayed shut, and the
model did its searching where the answer goes.

LIVE, 2026-09-08: the visible draft began "We need answer user's question.
Need" and was rejected as an internal prompt leak. The runtime appended "Do not
mention validation, retry, hidden prompts, receipts, gates, or implementation
details" and sampled again, and the second draft opened "Must not leak internal
task prompt? ... avoid validation/retry/gates/implementation details." The
instruction against leaking was itself leaked, retries ran out, and the person
was told "I couldn't get my full attention onto that one."

A budget the decoder enforces is not a request.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.a_bounded_private_channel import (  # noqa: E402
    close_the_channel_after,
)

_CLOSING_TOKEN = 248069  # `</think>` in the resident checkpoint.
_A_RATE_MODEL = "a-model-with-a-known-rate"

from core.brain.llm import thinking_reserve  # noqa: E402


def _forget_rates() -> None:
    thinking_reserve.forget()


class _ATokenizer:
    def encode(self, text, add_special_tokens=False):
        return [_CLOSING_TOKEN] if text == "</think>" else [1, 2, 3]


class _NoSingleMarker:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def _logits(width: int = 300_000):
    return mx.zeros((1, width))


def test_below_the_budget_it_changes_nothing():
    bound = close_the_channel_after(_ATokenizer(), 512)
    logits = _logits()
    out = bound(list(range(10)), logits)
    assert bool((out == logits).all())


def test_at_the_budget_the_closing_token_is_the_only_one_left():
    bound = close_the_channel_after(_ATokenizer(), 512)
    out = np.array(bound(list(range(512)), _logits()))
    assert np.isfinite(out).sum() == 1
    assert int(np.argmax(out)) == _CLOSING_TOKEN


def test_it_forces_the_marker_once_and_then_stands_aside():
    """After the channel closes, the answer is the model's own."""
    bound = close_the_channel_after(_ATokenizer(), 512)
    bound(list(range(512)), _logits())
    logits = _logits()
    assert bool((bound(list(range(600)), logits) == logits).all())


def test_a_channel_the_model_closed_itself_is_left_alone():
    bound = close_the_channel_after(_ATokenizer(), 512)
    logits = _logits()
    out = bound([1, 2, _CLOSING_TOKEN], logits)
    assert bool((out == logits).all())
    # And it stays out of the way afterwards.
    assert bool((bound(list(range(900)), logits) == logits).all())


def test_it_refuses_rather_than_pretends():
    """A control that might not hold is worse than one the caller knows it
    does not have."""
    assert close_the_channel_after(_NoSingleMarker(), 512) is None
    assert close_the_channel_after(_ATokenizer(), 4) is None
    assert close_the_channel_after(_ATokenizer(), 0) is None
    assert close_the_channel_after(_ATokenizer(), "not a number") is None


# ── the budget it is given ───────────────────────────────────────────────


def test_the_channel_gets_what_the_clock_can_pay_for_after_the_answer():
    """From the clock, not from a fraction.

    The first version took half the token budget, and half is a number
    somebody chose. LIVE, 2026-09-08: a 7,314-token budget produced a
    3,657-token channel — 400 seconds of thinking at the measured rate before
    a word of the answer — and the turn ran past fifteen minutes with nothing
    delivered.
    """
    from core.brain.llm.a_bounded_private_channel import the_channel_budget_for

    _forget_rates()
    try:
        for _ in range(12):
            thinking_reserve.record_decode_rate(
                generated_tokens=100, elapsed_s=10.0, model=_A_RATE_MODEL
            )
        # 10 tokens a second. 400 seconds buys 4,000 tokens; the answer keeps
        # 1,024 of them.
        assert the_channel_budget_for(
            max_tokens=8000,
            seconds_left=400.0,
            answer_floor=1024,
            model=_A_RATE_MODEL,
        ) == pytest.approx(4000 - 1024, abs=40)
        # A turn whose whole clock is spent on the answer thinks in the open.
        assert the_channel_budget_for(
            max_tokens=2048,
            seconds_left=110.0,
            answer_floor=1024,
            model=_A_RATE_MODEL,
        ) == 0
    finally:
        _forget_rates()


def test_no_deadline_is_not_permission_to_think_for_a_whole_turn():
    from core.brain.llm.a_bounded_private_channel import the_channel_budget_for

    assert the_channel_budget_for(max_tokens=4096, seconds_left=0.0) == 0
    assert the_channel_budget_for(max_tokens=4096, seconds_left=None) == 0
    assert the_channel_budget_for(max_tokens=0, seconds_left=900.0) == 0


def test_a_caller_may_name_its_own_channel_budget():
    from core.brain.llm.a_bounded_private_channel import the_channel_budget_for

    assert the_channel_budget_for(
        max_tokens=4096, seconds_left=0.0, asked_for=256
    ) == 256


def test_the_worker_asks_the_one_owner():
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._the_private_channel_budget)
    assert "the_channel_budget_for(" in source


def test_the_gate_never_opens_a_channel_that_cannot_be_bounded():
    """The worst case is thinking switched on with no budget to bound it."""
    from core.brain.llm.a_bounded_private_channel import the_channel_budget_for
    from core.brain.llm.chat_format import answer_is_derived_for_generation

    _forget_rates()
    try:
        for _ in range(12):
            thinking_reserve.record_decode_rate(
                generated_tokens=100, elapsed_s=10.0, model=_A_RATE_MODEL
            )
        for floor, budget, seconds in (
            (1024, 2048, 211.0),
            (1408, 7314, 480.0),
            (1024, 1345, 211.0),
            (512, 4096, 900.0),
            (1024, 2048, 20.0),
        ):
            opened = answer_is_derived_for_generation(
                completion_floor=floor,
                budget_tokens=budget,
                model_name=_A_RATE_MODEL,
                seconds_remaining=seconds,
            )
            room = the_channel_budget_for(
                max_tokens=budget,
                seconds_left=seconds,
                answer_floor=floor,
                model=_A_RATE_MODEL,
            )
            if opened:
                assert room > 0, (floor, budget, seconds)
    finally:
        _forget_rates()


# ── and the gate can now afford to open ──────────────────────────────────


def test_an_ordinary_turn_can_now_derive_its_answer_here():
    from core.brain.llm.chat_format import answer_is_derived_for_generation

    assert answer_is_derived_for_generation(
        completion_floor=1024, budget_tokens=1345, seconds_remaining=480.0
    ) is True


def test_a_budget_too_small_for_both_halves_still_refuses():
    from core.brain.llm.chat_format import answer_is_derived_for_generation

    assert answer_is_derived_for_generation(
        completion_floor=1024,
        budget_tokens=200,
        seconds_remaining=480.0,
    ) is False


def test_with_no_clock_the_role_is_answered_and_the_size_is_not():
    """A deadline is what makes affordability a question at all.

    Every caller that can open a channel states a deadline: the gate floors
    its request timeout at one second, and the worker reads what is left on
    the job. One caller cannot — `_time_the_answer_needs` is computing the
    deadline, so there is none to hand it — and refusing there drops the
    thinking reserve out of the estimate and under-prices exactly the turns
    that will think.

    So with no clock this answers the role question and nothing more. It
    cannot open a channel with the answer: only the worker does that, and the
    worker sizes through `the_channel_budget_for`, which refuses a channel it
    cannot price. A budget far too small for both halves is still True here,
    because without a clock nothing has asked the affordability question.
    """
    from core.brain.llm.chat_format import answer_is_derived_for_generation

    assert answer_is_derived_for_generation(
        completion_floor=1024, budget_tokens=200, seconds_remaining=0.0
    ) is True
    # The role question still holds, and still refuses a closed question.
    assert answer_is_derived_for_generation(
        completion_floor=16, budget_tokens=4096, seconds_remaining=0.0
    ) is False
    # The live consequence is pinned beside the clock that depends on it, in
    # test_reasoning_does_not_eat_the_answer.py::
    # test_answer_clock_prices_only_the_private_channel_worker_will_open.


def test_the_worker_bounds_the_channel_on_both_generation_paths():
    """One path bounded and one not is the same defect with a longer name."""
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    assert source.count("close_the_channel_after(") == 2


# ── and the mode no longer decides where the thinking happens ────────────

_A_MODEL_PATH = "Aura-Qwen3.8-27B-persona"


def test_a_derived_answer_opens_the_channel_whatever_the_mode():
    """`fast` resolves to no-thinking, and on a turn whose answer is worked
    out in this call that does not buy a shorter turn — it moves the search
    out of the private channel and into the reply.

    LIVE, 2026-09-08: "Native thinking False (surface=True floor=1024
    mode=fast)", and the visible draft began "We need answer user's question.
    Need".
    """
    from core.brain.llm.chat_format import thinking_enabled_for_generation

    assert thinking_enabled_for_generation(
        _A_MODEL_PATH,
        cognitive_mode="fast",
        final_user_surface=True,
        answer_is_derived_here=True,
    ) is True


def test_a_turn_that_only_renders_an_answer_keeps_the_channel_shut():
    """The premise the closure rests on: the answer was settled upstream and
    this stage renders it. Opening a second channel there makes the search
    compete with the answer for one deadline."""
    from core.brain.llm.chat_format import thinking_enabled_for_generation

    assert thinking_enabled_for_generation(
        _A_MODEL_PATH,
        cognitive_mode="fast",
        final_user_surface=True,
        answer_is_derived_here=False,
    ) is False
