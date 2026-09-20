"""A turn is graded against its own question, never a previous one.

LIVE DEFECT, 2026-08-10. Asked "look at my screen and tell me what app is in
front right now" (128 chars), every draft was rejected with::

    Rejected live user-surface draft reasons=arithmetic_answer_missing
        validation_chars=31 excerpt='50864799'

31 characters is "what's 7919 multiplied by 6421?" — the previous turn. The
screen question was being graded against the arithmetic question's expected
answer, which no reply about windows can contain, so the turn died and the
person got a refusal about their screen.
"""
from __future__ import annotations

from core.conversation.user_surface_contract import (
    bind_user_surface_prompt,
    resolve_user_surface_prompt,
)

ARITHMETIC = "what's 7919 multiplied by 6421?"
SCREEN = (
    "look at my screen and tell me what app is in front right now. if the "
    "capture doesn't work, say exactly that instead of guessing."
)


def test_a_binding_from_an_earlier_turn_is_detected_as_stale():
    """The condition the gate now checks for."""
    context: dict[str, object] = {}
    bind_user_surface_prompt(context, ARITHMETIC, source="turn-1", overwrite=True)

    bound = resolve_user_surface_prompt(context)
    assert bound.bound
    assert str(bound.prompt).strip() == ARITHMETIC
    # Turn 2 arrives with its own question; the bound one is not it.
    assert str(bound.prompt).strip() != SCREEN


def test_rebinding_replaces_the_previous_turns_question():
    context: dict[str, object] = {}
    bind_user_surface_prompt(context, ARITHMETIC, source="turn-1", overwrite=True)
    bind_user_surface_prompt(context, SCREEN, source="turn-2", overwrite=True)

    resolved = resolve_user_surface_prompt(context)
    assert str(resolved.prompt).strip() == SCREEN
    assert context["visible_user_message"] == SCREEN
    assert context["user_surface_validation_prompt"] == SCREEN


def test_the_gate_rebinds_when_the_bound_prompt_is_not_this_turns():
    """The guard itself, read from the source it protects."""

    from source_contract import family_text

    from core.brain import inference_gate

    source = family_text(inference_gate)  # the gate and the modules lifted out of it
    assert "stale_binding" in source
    assert "explicit_visible_user_prompt" in source


def test_a_screen_question_expects_no_arithmetic():
    """Why the mismatch was fatal rather than merely wrong."""
    from core.conversation.response_reliability import requested_arithmetic_result

    assert requested_arithmetic_result(SCREEN) is None
    assert requested_arithmetic_result(ARITHMETIC) is not None
