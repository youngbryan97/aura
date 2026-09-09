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
    THE_CHANNEL_FITS_IN,
    close_the_channel_after,
)

_CLOSING_TOKEN = 248069  # `</think>` in the resident checkpoint.


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


def test_the_channel_takes_a_share_and_leaves_the_answer_the_rest():
    from core.brain.llm.mlx_worker import _the_private_channel_budget

    assert _the_private_channel_budget({}, 1024) == 512
    assert _the_private_channel_budget({}, 2048) == 1024
    # Never all of it: the answer keeps room to be written.
    assert _the_private_channel_budget({}, 1024) < 1024
    # Too small to think in.
    assert _the_private_channel_budget({}, 128) == 0
    assert _the_private_channel_budget({}, 0) == 0


def test_a_job_may_name_its_own_channel_budget():
    from core.brain.llm.mlx_worker import _the_private_channel_budget

    assert _the_private_channel_budget({"private_channel_budget": 256}, 4096) == 256


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
        budget_tokens=THE_CHANNEL_FITS_IN - 1,
        seconds_remaining=480.0,
    ) is False


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
