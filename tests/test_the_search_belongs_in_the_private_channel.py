"""Where the answer gets worked out, the working belongs off the surface.

The render stage closes native thinking because the reasoning owners upstream
have already settled the answer. That premise fails when no phase settled it —
a rule to infer, an order to work out, a quantity nothing computed. A reasoning
model reasons either way, so closing the channel there does not save the
deadline; it moves the search into the reply.

LIVE, 2026-08-27: "45 becomes 15. 28 becomes 14. 66 becomes 22. What does 91
become?" was answered four times, each with the search visible and each cut off
before a conclusion.
"""

from __future__ import annotations

import pytest
from mlx_source import worker_source

from pathlib import Path

from core.brain.llm.chat_format import thinking_enabled_for_generation
from core.brain.llm.mlx_worker import _answer_is_derived_here
from core.runtime.structured_input import (
    A_CLOSED_QUESTIONS_FLOOR,
    answer_surface_token_floor,
)

_A_DERIVATION = (
    "Another sequence for you. 45 becomes 15. 28 becomes 14. 66 becomes 22. "
    "What am I doing, what does 91 become, and is three examples enough to pin "
    "it down or could a different rule fit these too?"
)


def test_a_closed_question_carries_the_base_floor() -> None:
    assert answer_surface_token_floor("what time is it") == A_CLOSED_QUESTIONS_FLOOR


def test_the_live_question_carries_more_than_the_base() -> None:
    assert answer_surface_token_floor(_A_DERIVATION) > A_CLOSED_QUESTIONS_FLOOR


def test_a_floor_above_the_base_says_the_answer_is_made_here() -> None:
    assert _answer_is_derived_here(
        {"user_surface_completion_floor": answer_surface_token_floor(_A_DERIVATION)}
    )


def test_the_base_floor_is_a_render_not_a_derivation() -> None:
    assert not _answer_is_derived_here(
        {"user_surface_completion_floor": A_CLOSED_QUESTIONS_FLOOR}
    )
    assert not _answer_is_derived_here({})
    assert not _answer_is_derived_here({"user_surface_completion_floor": "nonsense"})


def test_the_render_stage_still_closes_the_channel_by_default() -> None:
    assert (
        thinking_enabled_for_generation(
            "qwen-thinking", final_user_surface=True, answer_is_derived_here=False
        )
        is False
    )


def test_a_derived_answer_is_not_forced_closed() -> None:
    assert (
        thinking_enabled_for_generation(
            "qwen-thinking", final_user_surface=True, answer_is_derived_here=True
        )
        is not False
    )


def test_an_unclosed_boundary_is_written_down() -> None:
    from pathlib import Path

    body = worker_source()
    assert "Generation ended inside the private channel" in body
    assert "and not native_channels.boundary_closed" in body


_CORTEX = f"{Path.home()}/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83"
_BRAINSTEM = "models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c0"


def test_a_derived_answer_asks_explicitly_rather_than_shrugging() -> None:
    """None means "the artifact's default", and every reader treats it as no.

    LIVE, 2026-08-27: the channel resolved to None, the splitter was told there
    was no private channel, the budget ran out before the closing marker, and
    the whole of "We need answer user's puzzle. Need use tool?" was handed over
    as the answer. The surface validator rejected it, correctly.
    """

    assert (
        thinking_enabled_for_generation(
            _CORTEX, final_user_surface=True, answer_is_derived_here=True
        )
        is True
    )


def test_a_render_on_the_same_model_stays_closed() -> None:
    assert (
        thinking_enabled_for_generation(
            _CORTEX, final_user_surface=True, answer_is_derived_here=False
        )
        is False
    )


def test_a_pinned_fast_lane_stays_closed_even_for_a_derivation() -> None:
    assert (
        thinking_enabled_for_generation(
            _BRAINSTEM, final_user_surface=True, answer_is_derived_here=True
        )
        is False
    )


def test_a_generation_that_is_not_the_surface_is_left_alone() -> None:
    assert (
        thinking_enabled_for_generation(_CORTEX, final_user_surface=False) is None
    )


def test_a_budget_that_cannot_hold_both_halves_opens_a_bounded_channel() -> None:
    """Nothing served is worse than partial working served — and the bound is
    the decoder's now, not a veto.

    LIVE, 2026-08-27: three attempts in a row ended inside the channel, the
    last after 127 seconds and 3,411 characters of reasoning, and the turn
    served nothing each time. That was an UNBOUNDED channel; the decoder now
    closes it at its budget (a_bounded_private_channel, 2026-09-08), so an
    attempt cannot end inside it.

    LIVE, 2026-09-15: the veto this test used to assert kept the channel
    shut on a "walk me through" question whose floor was its whole budget.
    Shut, the model reasoned in the reply — 4,530 characters beginning "The
    user is asking about", every token of the budget, no answer — and the
    person was told to ask again. The channel opens, at the smallest size
    worth opening, and the decoder closes it there.
    """

    import time as _time

    from core.brain.llm import thinking_reserve
    from core.brain.llm.a_bounded_private_channel import TOO_SMALL_TO_THINK_IN
    from core.brain.llm.mlx_worker import _the_private_channel_budget

    budget_is_the_whole_answer = {
        "user_surface_completion_floor": 896,
        "max_tokens": 896,
        "deadline_unix": _time.time() + 127,
    }
    thinking_reserve.forget()
    try:
        # Six tokens a second, as on 2026-08-27: 127 seconds buys 762, and
        # the answer alone needs 896.
        for _ in range(20):
            thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=150.0)
        assert _answer_is_derived_here(budget_is_the_whole_answer)
        assert (
            _the_private_channel_budget(budget_is_the_whole_answer, 896)
            == TOO_SMALL_TO_THINK_IN
        )
    finally:
        thinking_reserve.forget()


def test_a_job_with_no_budget_is_left_to_the_floor_alone() -> None:
    assert _answer_is_derived_here({"user_surface_completion_floor": 896})


def test_the_clock_sizes_the_channel_and_no_longer_vetoes_it() -> None:
    """The clock spent proving it will not close is spent for nothing — so
    the decoder closes it, and the clock decides where.

    LIVE, 2026-08-27: the first attempt burned 98 of a 148-second turn
    discovering the channel would not close. The bound is enforced now; what
    the clock still decides is how much of the turn the channel may take.
    """

    import time as _time

    from core.brain.llm import thinking_reserve
    from core.brain.llm.a_bounded_private_channel import TOO_SMALL_TO_THINK_IN
    from core.brain.llm.mlx_worker import _the_private_channel_budget

    thinking_reserve.forget()
    try:
        def job(seconds_left: float) -> dict[str, object]:
            return {
                "user_surface_completion_floor": 896,
                "max_tokens": 4096,
                "deadline_unix": _time.time() + seconds_left,
            }

        # An unmeasured rate cannot refuse anything.
        assert _answer_is_derived_here(job(30))
        # Runs of a comparable length, at six tokens a second.
        for _ in range(20):
            thinking_reserve.record_decode_rate(
                generated_tokens=900, elapsed_s=150.0
            )
        # Thirty seconds at six a second is 180 tokens, and the answer alone
        # needs 896 of them: the role stands, the channel is the smallest one.
        assert _answer_is_derived_here(job(30))
        assert _the_private_channel_budget(job(30), 4096) == TOO_SMALL_TO_THINK_IN
        # Ten minutes buys 3,600 tokens; the answer keeps 896 of them.
        assert _answer_is_derived_here(job(600))
        assert _the_private_channel_budget(job(600), 4096) == pytest.approx(
            3600 - 896, abs=40
        )
        # A job that states no deadline is left to the other tests.
        assert _answer_is_derived_here(
            {"user_surface_completion_floor": 896, "max_tokens": 896}
        )
    finally:
        thinking_reserve.forget()
