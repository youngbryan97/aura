"""Affect shapes her voice; it must not bend an answer that has a right one.

Substrate steering pushes an affect direction into the residual stream so her
replies sound like her. On a turn whose answer is a fact rather than a feeling,
that push is pure distortion — and the floor in `_live_mind_generation_controls`
was `max(0.20, ...)`, so steering could never stand down, on any turn, ever.

Measured live on the desktop surface 2026-07-26. "What is 17 minus 8, and then
times 3?" — which the bare 32B answers without effort — came back twice from a
healthy resident cortex:

    "Not too broad. Some skills serve me better than others.Did you pay
     attention in class? Hey, look at this - ätze! I got chocolate on my shirt."

    "Five thousand: So first you break it down. Mental operations can generate
     digits well outside the world's population. Imagination defaults to scalar
     scaling when pushed into math without boundaries."

The second is the diagnosis: not noise, but the answer pulled toward what the
affect vector encodes — discussing scaling and imagination rather than
subtracting eight from seventeen.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_engine import (
    _live_mind_generation_controls,
    _turn_needs_undistorted_computation,
)


def _ready_context(*, distress: float = 0.1, curiosity: float = 0.7) -> dict:
    """A live-mind context whose snapshot is ready enough to yield controls."""
    return {
        "mind_snapshot_quality": {"ready": True},
        "mind_snapshot": {
            "affect_grounding": {"dominant": {"label": "curiosity", "intensity": curiosity}},
            "drive_integration": {"drives": {"curiosity": {"activation": curiosity}}},
            "nociception": {"nociceptive_pressure": distress},
            "phenomenal_engine": {"integration": 0.7, "self_presence": 0.6},
            "global_workspace": {"ignited": True},
        },
    }


@pytest.mark.parametrize(
    "question",
    [
        "What is 17 minus 8, and then times 3?",
        "How much is 20 percent of 50?",
        "Calculate 144 divided by 12",
    ],
)
def test_steering_stands_down_for_a_determinate_question(question: str) -> None:
    assert _turn_needs_undistorted_computation(question) is True
    controls = _live_mind_generation_controls(_ready_context(), user_message=question)
    assert controls["clean_user_surface_steering_alpha"] == 0.0, (
        "steering must not sit in the path of a computation"
    )
    assert controls["clean_user_surface_recurrent_loops"] == 1
    assert controls["temperature"] <= 0.30


@pytest.mark.parametrize(
    "question",
    [
        "How are you feeling right now?",
        "Tell me about your day.",
        "What is love?",
        "",
    ],
)
def test_expressive_turns_do_not_require_unvalidated_residual_injection(question: str) -> None:
    assert _turn_needs_undistorted_computation(question) is False
    controls = _live_mind_generation_controls(_ready_context(), user_message=question)
    assert controls["clean_user_surface_steering_alpha"] == 0.0
    assert controls["temperature"] > 0.0


def test_user_visible_residual_steering_is_neutral_by_default() -> None:
    controls = _live_mind_generation_controls(_ready_context())
    assert controls["clean_user_surface_steering_alpha"] == 0.0


def test_the_predicate_fails_closed_on_junk() -> None:
    for value in (None, 123, object()):
        assert _turn_needs_undistorted_computation(value) is False


def test_controls_are_still_empty_without_a_ready_snapshot() -> None:
    """The determinate branch must not invent controls from nothing."""
    assert _live_mind_generation_controls({}, user_message="What is 2 plus 2?") == {}
    assert (
        _live_mind_generation_controls(
            {"mind_snapshot_quality": {"ready": False}},
            user_message="What is 2 plus 2?",
        )
        == {}
    )
