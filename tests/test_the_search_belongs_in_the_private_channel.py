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

    body = Path("core/brain/llm/mlx_worker.py").read_text()
    assert "Generation ended inside the private channel" in body
    start = body.index("Generation ended inside the private channel")
    window = body[start - 700 : start + 400]
    assert "not native_channels.boundary_closed" in window


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


def test_a_budget_proved_unable_to_close_the_channel_keeps_it_shut() -> None:
    """Nothing served is worse than partial working served.

    LIVE, 2026-08-27: three attempts in a row ended inside the channel, the
    last after 127 seconds and 3,411 characters of reasoning, and the turn
    served nothing each time. The same question with the channel closed had
    served a real partial derivation.
    """

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    try:
        job = {"user_surface_completion_floor": 896, "max_tokens": 896}
        assert _answer_is_derived_here(job)
        thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=896)
        assert not _answer_is_derived_here(job)
        # A budget bigger than the one that failed is worth trying again.
        assert _answer_is_derived_here(
            {"user_surface_completion_floor": 896, "max_tokens": 1792}
        )
    finally:
        thinking_reserve.forget()


def test_a_job_with_no_budget_is_left_to_the_floor_alone() -> None:
    assert _answer_is_derived_here({"user_surface_completion_floor": 896})


def test_a_channel_there_is_no_time_to_close_is_not_opened() -> None:
    """The clock spent proving it will not close is spent for nothing.

    LIVE, 2026-08-27: the first attempt burned 98 of a 148-second turn
    discovering the channel would not close, and the retry that did answer had
    50 seconds and produced 85 characters.
    """

    import time as _time

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    try:
        def job(seconds_left: float) -> dict[str, object]:
            return {
                "user_surface_completion_floor": 896,
                "max_tokens": 896,
                "deadline_unix": _time.time() + seconds_left,
            }

        # An unmeasured rate cannot refuse anything.
        assert _answer_is_derived_here(job(30))
        # Runs of a comparable length, at six tokens a second.
        for _ in range(20):
            thinking_reserve.record_decode_rate(
                generated_tokens=900, elapsed_s=150.0
            )
        # 896 tokens at six a second is about 149 seconds.
        assert not _answer_is_derived_here(job(30))
        assert _answer_is_derived_here(job(300))
        # A job that states no deadline is left to the other tests.
        assert _answer_is_derived_here(
            {"user_surface_completion_floor": 896, "max_tokens": 896}
        )
    finally:
        thinking_reserve.forget()
