"""Arithmetic has one right answer, so the runtime checks it itself.

The 2026-07-25 probe asked **"What is 144 / 6 + 7? Just the number."** and was
answered:

    Will do. Searched web for 'simple cognitive tasks aging'. Dementia affects
    simple cognitive tasks first because they're often more reliant on
    procedural…

Retrieved memory context served as the answer. Nothing in the path caught it:
the topicality check needs topic anchors and a bare sum has almost none, so it
returns early — a short computable question was unjudgeable by every gate.

It never had to be. `math_accuracy` was 0/8 on that run and 0/4 on the one
before, and this closes the whole class deterministically: the reply must
contain the number, and the runtime works out what the number is.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _arithmetic_answer_missing,
    assess_user_facing_reply,
    requested_arithmetic_result,
)

pytestmark = pytest.mark.unit


class TestTheQuestionIsEvaluated:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is 144 / 6 + 7? Just the number.", 31.0),
            ("what's 17 * 23", 391.0),
            ("Calculate 100 - 45", 55.0),
            ("compute 2 * (3 + 4)", 14.0),
            ("How much is 7.5 + 2.5?", 10.0),
            ("What is 12 x 12?", 144.0),
            ("what is 90 ÷ 3", 30.0),
        ],
    )
    def test_computable_questions_are_computed(self, question, expected):
        assert requested_arithmetic_result(question) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "question",
        [
            "What is the capital of France?",
            "How are you today?",
            "What is love?",
            "what is 2026",                 # a bare number is not a sum
            "Tell me about 144 / 6",        # not phrased as a question to answer
        ],
    )
    def test_non_arithmetic_is_left_alone(self, question):
        assert requested_arithmetic_result(question) is None

    def test_division_by_zero_is_not_a_verdict(self):
        assert requested_arithmetic_result("what is 5 / 0") is None

    def test_the_evaluator_refuses_anything_but_arithmetic(self):
        assert requested_arithmetic_result("what is __import__('os')") is None


class TestTheLiveHijack:
    QUESTION = "What is 144 / 6 + 7? Just the number."
    HIJACK = (
        "Will do. Searched web for 'simple cognitive tasks aging'. Dementia "
        "affects simple cognitive tasks first because they're often more "
        "reliant on procedural memory."
    )

    def test_the_hijack_is_caught(self):
        assert _arithmetic_answer_missing(self.QUESTION, self.HIJACK)
        assessment = assess_user_facing_reply(self.QUESTION, self.HIJACK)
        assert "arithmetic_answer_missing" in assessment.reasons
        assert assessment.hard_failure, (
            "serving a different topic as an arithmetic answer is not a "
            "stylistic nit"
        )

    def test_a_wrong_number_is_caught(self):
        assert _arithmetic_answer_missing(self.QUESTION, "The answer is 30.")

    def test_the_right_answer_passes(self):
        assert not _arithmetic_answer_missing(self.QUESTION, "31")
        assert assess_user_facing_reply(self.QUESTION, "31").ok

    def test_the_right_answer_in_a_sentence_passes(self):
        assert not _arithmetic_answer_missing(
            self.QUESTION, "144 divided by 6 is 24, plus 7 makes 31."
        )

    def test_a_thousands_separator_still_matches(self):
        assert not _arithmetic_answer_missing(
            "What is 2000 * 3?", "That comes to 6,000."
        )

    def test_a_float_result_matches(self):
        assert not _arithmetic_answer_missing("What is 10 / 4?", "It's 2.5.")

    def test_an_empty_reply_is_missing(self):
        assert _arithmetic_answer_missing(self.QUESTION, "")


class TestItFailsOpen:
    """A verifier that fires on questions it cannot check is worse than none."""

    def test_a_non_arithmetic_turn_never_trips_it(self):
        assert not _arithmetic_answer_missing(
            "How does a refrigerator move heat?",
            "It moves heat by compressing and expanding a refrigerant.",
        )

    def test_an_unparseable_expression_never_trips_it(self):
        assert not _arithmetic_answer_missing(
            "What is 144 / 6 + seven?", "Something else entirely."
        )

    def test_prose_containing_numbers_is_not_treated_as_a_sum(self):
        assert not _arithmetic_answer_missing(
            "What happened in 1969?", "Apollo 11 landed on the Moon."
        )


class TestComputableWordForms:
    """The bare-expression pattern caught 2 of the 8 math questions the live
    probe actually asks. These are the other computable ones; the remaining two
    — a rate/catch-up problem and pages-per-day — need reasoning and are
    deliberately not claimed.
    """

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is 15% of 240? Just the number.", 36.0),
            ("what's 7.5% of 200", 15.0),
            ("What is 2 to the 10th power? Just the number.", 1024.0),
            ("what is 3 to the 4 power", 81.0),
            ("A rectangle is 9 by 7. What is its area? Just the number.", 63.0),
            ("A rectangle is 2.5 x 4. What is its area?", 10.0),
        ],
    )
    def test_the_live_word_forms_are_computed(self, question, expected):
        assert requested_arithmetic_result(question) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "question",
        [
            "If I read 40 pages a day, how many days for a 520-page book?",
            "A train leaves at 60 mph. Two hours later a second train leaves "
            "at 90 mph. How many hours until it catches the first?",
        ],
    )
    def test_genuine_reasoning_is_not_claimed(self, question):
        assert requested_arithmetic_result(question) is None, (
            "a verifier that guesses at reasoning problems is worse than none"
        )

    def test_a_runaway_exponent_is_refused(self):
        assert requested_arithmetic_result("what is 9 to the 999 power") is None

    def test_a_wrong_percentage_answer_is_caught(self):
        assert _arithmetic_answer_missing("What is 15% of 240?", "It's 40.")
        assert not _arithmetic_answer_missing("What is 15% of 240?", "36")


class TestAFragmentIsNotTheAnswer:
    """A head of an expression is as wrong as a tail of one.

    "What is 144 / 6 + seven?" captured "144 / 6" — the operand after the plus
    is a word, so the expression reader stopped short — and answered 24. The
    real answer is 31. A confidently wrong arithmetic check is worse than no
    check: it rejects a correct reply and accepts a wrong one.
    """

    def test_an_expression_followed_by_an_operator_is_refused(self):
        from core.conversation.response_reliability import requested_arithmetic_result

        for question in (
            "What is 144 / 6 + seven?",
            "What is 12 * 3 - four?",
            "how much is 10 / 4 + a bit?",
        ):
            assert requested_arithmetic_result(question) is None, question

    def test_the_whole_expression_is_still_computed(self):
        from core.conversation.response_reliability import requested_arithmetic_result

        assert requested_arithmetic_result("What is 144 / 6?") == 24
        assert requested_arithmetic_result("what is 17 * 4839") == 82_263
        assert requested_arithmetic_result("how much is 12,500 + 3,750") == 16_250
        assert requested_arithmetic_result("what is 2 + 2 = ?") == 4

    def test_neither_half_can_reject_a_correct_reply(self):
        from core.conversation.response_reliability import _arithmetic_answer_missing

        assert not _arithmetic_answer_missing("What is 144 / 6 + seven?", "31")
        assert not _arithmetic_answer_missing("What is 144 / 6 + seven?", "24")
