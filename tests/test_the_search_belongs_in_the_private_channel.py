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
