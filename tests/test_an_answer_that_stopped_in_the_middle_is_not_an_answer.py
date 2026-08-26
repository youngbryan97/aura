"""Two things arrive where an answer should be and are not answers.

Measured live on 2026-08-26: the model's own warm-up was held as her plan for
a whole game of 2048, and a passage that stopped mid-sentence became the
question a deeper pass then went off and answered.
"""

from __future__ import annotations

import pytest

from core.utils.an_answer import adds_nothing_to, content_words, was_cut_off

SCAFFOLDING = (
    'We need answer user\'s request. Need choose move from up/down/left/right '
    'based on 2048 board? User gives: "Given what'
)


def test_a_passage_that_stops_mid_sentence_was_cut_off():
    assert was_cut_off(SCAFFOLDING)


def test_an_unclosed_quote_is_proof_of_a_cut():
    assert was_cut_off('She said "keep the big tile in the corner')


def test_an_unclosed_bracket_is_proof_of_a_cut():
    assert was_cut_off("Take the left column (it keeps the row above clear")


def test_a_finished_passage_was_not_cut_off():
    assert not was_cut_off(
        "Build the big tiles in the bottom-left corner. That keeps a clear row above."
    )


def test_a_one_word_reply_was_never_a_sentence():
    assert not was_cut_off("left")
    assert not was_cut_off("bottom-left corner")


def test_nothing_at_all_is_not_a_cut():
    assert not was_cut_off("")
    assert not was_cut_off("   ")


@pytest.mark.parametrize("ending", [".", "!", "?", "…"])
def test_anything_that_finishes_is_finished(ending):
    assert not was_cut_off(f"Keep the 64 in the corner. Then feed it from the left{ending}")


def test_the_question_handed_back_adds_nothing():
    asked = "Decide how to play toward this goal, not just the next move."
    assert adds_nothing_to("Decide how to play toward the goal, not just the next move", asked)


def test_a_real_answer_adds_something():
    asked = "Decide how to play toward this goal, not just the next move."
    assert not adds_nothing_to(
        "Keep the largest tile in the bottom-left corner and feed it along the bottom row",
        asked,
    )


def test_with_nothing_asked_the_check_stands_aside():
    assert not adds_nothing_to("anything at all", "")


def test_content_words_drop_the_short_ones():
    assert content_words("Go to the big one") == {"the", "big", "one"}


def test_a_quote_that_was_never_opened_is_still_unbalanced():
    assert was_cut_off('Keep the 64 in the corner. Then feed it from the left"')
