"""The gate invented the user's question when nobody supplied one.

LIVE 2026-08-19. Deciding a move on a 2048 board, she answered:

    "right — the board is mostly open on the right side, sliding right
     consolidates the smaller numbers and creates space for new ones"

The live user-surface quality gate rejected it as arithmetic_answer_missing,
retried twice, then returned no text at all. The pursuit reported that she
had named no available move, which was the one thing that had not happened.

Nothing was answering a person. The gate reads the user's question out of
the call, and when none is supplied it fell back to the last user-role
message in the envelope — which for an internal call is the model prompt.
That prompt carried a screen reading full of numbers, so the reply was
required to contain a total, and a correct one-word move could not pass.

A chat turn supplies the question. Every path that produces a visible reply
passes visible_user_message or user_surface_validation_prompt. The fallback
survives only where the origin really is a person talking.
"""
from __future__ import annotations

import inspect

from core.brain.inference_gate import InferenceGate


def _generate_source() -> str:
    """The generation body, followed through the delegation that holds it.

    ``generate`` now does one thing: it calls
    ``_generate_with_metadata_sink``. Reading only ``generate`` found none of
    the binding logic and three tests failed about code that had not moved off
    the path at all.
    """
    entry = inspect.getsource(InferenceGate.generate)
    assert "_generate_with_metadata_sink(" in entry, (
        "generate no longer delegates; the body may have moved somewhere else"
    )
    return entry + inspect.getsource(InferenceGate._generate_with_metadata_sink)


def test_the_fallback_is_reached_only_for_an_origin_that_is_a_person():
    source = _generate_source()
    assert "derivable" in source, "the derivation is unconditional again"
    where = source.index("derivable")
    window = source[where : where + 500]
    assert "_origin_is_user_facing" in window
    assert "_visible_user_prompt_from_messages" in window


def test_nothing_is_bound_when_there_is_no_question():
    source = _generate_source()
    assert "elif not initial_visible_user_prompt and not surface_prompt.bound:" in source


def test_an_internal_call_binds_nothing_at_all():
    source = _generate_source()
    assert "internal_inference_call" in source
    where = source.index("internal_inference_call = bool")
    window = source[where : where + 700]
    assert "surface_prompt.bound" in window, "the internal branch must sit on the binding"


def test_a_deliberation_origin_is_not_treated_as_a_person_talking():
    """agency_next_move is not a chat surface, so nothing it sends is a question."""
    assert InferenceGate._origin_is_user_facing("agency_next_move") is False
    assert InferenceGate._origin_is_user_facing("action_deliberation") is False


def test_a_real_chat_origin_still_is():
    assert InferenceGate._origin_is_user_facing("user") is True
