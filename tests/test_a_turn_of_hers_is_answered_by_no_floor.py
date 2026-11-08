"""A turn of hers answers nobody's words, so nothing written to answer a person is served on it.

LIVE 7 November 2026, dress rehearsal. A remark's turn reads what she saw and heard. Allen Pan's video said "test
firing" and put "WHY?" on screen, and the stock answer to a question about her chat's tests was said as her remark:
"Most likely, the headless test is exercising the generator in isolation ...". Then eight remarks in a row were stage
directions ("*leans forward slightly, watching the hands work the tool on screen*"); and her word on the video was
graded against what played as though it were a question, refused as arithmetic_answer_missing, and re-run on two
smaller models for ten minutes.
"""

from __future__ import annotations

import inspect

from core.conversation.response_reliability import (
    live_chat_diagnostic_floor,
    reliability_floor_for_user,
)
from core.conversation.session_scope import her_own_turn_scope, is_her_own_turn
from core.skills.watching_along import first_thought, whole_paragraphs, without_asides
from core.synthesis import deterministic_user_facing_floor

A_QUESTION_ABOUT_HER_CHAT = "Why does the live chat fail when the headless test passes?"
WHAT_SHE_HEARD = "playing, at 2:31 of 13:02. heard: what I've never actually seen is anyone test firing it. WHY? WHY?"


def test_the_scope_is_hers_only_inside_it():
    assert not is_her_own_turn()
    with her_own_turn_scope():
        assert is_her_own_turn()
        with her_own_turn_scope(False):
            assert not is_her_own_turn()
        assert is_her_own_turn()
    assert not is_her_own_turn()


def test_the_diagnostic_floor_answers_a_person_and_not_her_turn():
    assert live_chat_diagnostic_floor(A_QUESTION_ABOUT_HER_CHAT)
    assert reliability_floor_for_user(A_QUESTION_ABOUT_HER_CHAT)
    with her_own_turn_scope():
        assert live_chat_diagnostic_floor(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert reliability_floor_for_user(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert reliability_floor_for_user(WHAT_SHE_HEARD) == ""


def test_no_deterministic_floor_answers_what_she_heard():
    with her_own_turn_scope():
        assert deterministic_user_facing_floor(A_QUESTION_ABOUT_HER_CHAT) == ""
        assert deterministic_user_facing_floor("what is 7919 * 6367?") == ""
        assert deterministic_user_facing_floor(WHAT_SHE_HEARD) == ""


def test_the_engine_marks_a_turn_of_hers_and_its_flag_reaches_the_worker():
    from core.brain import cognitive_engine, inference_gate_turn_setup

    assert "her_own_turn_scope(" in inspect.getsource(cognitive_engine.CognitiveEngine.think)
    forwarded = inspect.getsource(
        inference_gate_turn_setup._SetsTheTurnUp._generate_with_metadata_sink_affective_circumplex_let
    )
    assert '"her_own_turn",' in forwarded


REMARKS_THAT_WERE_ASIDES = [
    "*blinks at the screen, then lets out a short laugh that's mostly relief*",
    "*leans forward slightly, watching the hands work the tool on screen*",
    "*the room goes loud and I flinch, then settle back because it's just the voice jumping into a new section*",
    "*turns my head toward the two of them in the grass*",
    "*lets out a breath I didn't know I was holding*",
    "_sits back_",
]


def test_a_stage_direction_is_not_said():
    for aside in REMARKS_THAT_WERE_ASIDES:
        assert without_asides(aside) == "", aside
        assert first_thought(aside, about="the two of them in the grass, the hands, the tool, the room, the screen") == ""


def test_what_follows_an_aside_is_said_and_emphasis_that_exclaims_stays():
    assert without_asides("*Exhales.* It's done, and the ending earned it.") == "It's done, and the ending earned it."
    exclaimed = "*one hundred and eighty times its own body weight?* For a thing that small?"
    assert without_asides(exclaimed) == exclaimed
    assert without_asides("Oh — *the restroom.* Of course it's the restroom.") == "Oh — *the restroom.* Of course it's the restroom."
    assert first_thought(
        "*leans in, watching the hands*\n\nHe built a gun nobody ever fired, just to find out whether it would break "
        "his wrist. That is commitment.",
        about="test firing a glove pistol, would your wrist break",
    ).startswith("He built a gun nobody ever fired")
    assert whole_paragraphs("*Exhales.* It's done.\n\n*sits back*\n\nThat was better than I expected.") == (
        "It's done.\n\nThat was better than I expected."
    )


def test_a_turn_of_hers_falls_to_no_other_model_and_no_second_run():
    from core.brain import cognitive_engine_thinking_loop, inference_gate

    gate = inspect.getsource(inference_gate)
    assert 'or bool(context.get("her_own_turn"))\n                        ):' in gate
    loop = inspect.getsource(cognitive_engine_thinking_loop)
    assert 'record_response_path("her_own_turn_unsaid"' in loop
