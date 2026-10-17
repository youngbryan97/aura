"""MMLU-Pro's knowledge questions run the knowledge arms: looked up by the question alone, graded by its letter."""

from __future__ import annotations

from tools.run_knowledge_with_corpus import agree_letter, grade_letter


def test_the_chosen_letter_is_graded_and_a_reply_that_does_not_know_declines():
    assert grade_letter("Step by step ... so the answer is (C).", "C") == "correct"
    assert grade_letter("the answer is (B)", "C") == "wrong"
    assert grade_letter("I do not know.", "C") == "declined"
    assert grade_letter("It is hard to say which.", "C") == "wrong"


def test_two_answers_agree_only_on_the_same_letter():
    assert agree_letter("the answer is (C)", "Answer: C")
    assert not agree_letter("the answer is (C)", "the answer is (D)")
    assert not agree_letter("no letter here", "none here either")
