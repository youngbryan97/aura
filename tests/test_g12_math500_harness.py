"""MATH-500 answers are read from the last box in the public reply, braces matched."""

from __future__ import annotations

from tools.run_g12_math500 import grade, last_boxed


def test_the_last_box_is_the_answer_and_nested_braces_hold() -> None:
    text = "First \\boxed{3}. On reflection the answer is \\boxed{\\frac{\\sqrt{2}}{2}}."
    assert last_boxed(text) == "\\frac{\\sqrt{2}}{2}"
    assert last_boxed("\\fbox{12}") == "12"


def test_no_box_or_an_unclosed_one_is_no_answer() -> None:
    assert last_boxed("the answer is 7") is None
    assert last_boxed("\\boxed{\\frac{1}{2}") is None


def test_no_answer_is_wrong_and_a_grader_failure_is_named() -> None:
    assert grade(lambda given, truth: True, None, "7", seconds=5) == (False, "no_boxed_answer")

    def broken(given: str, truth: str) -> bool:
        raise ValueError("unparseable")

    assert grade(broken, "x", "7", seconds=5) == (False, "grader_error:ValueError")
    assert grade(lambda given, truth: given == truth, "7", "7", seconds=5) == (True, "")
