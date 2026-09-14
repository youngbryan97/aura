import json

import pytest

from core.learning.semantic_task_grading import (
    LEGACY_GRADING_POLICY,
    SEMANTIC_GRADING_POLICY,
    grade_semantic_task,
)
from core.reasoning.asymptotic import growth_class, same_growth_class
from tools.run_semantic_neural_decode_canary import _task_cohort


@pytest.mark.parametrize(("left", "right"), [
    ("Theta(n^2)", "O(n**2)"), ("O(n*n)", "Theta(3*n**2+n+1)"),
    ("O(n*log(n))", "Theta(4*n*ln(n)+n)"), ("O(1)", "Theta(42)"),
    ("O(log(n)**3)", "Theta(log(n)^3+log(n))"),
    ("\u0398(n^4)", "O(n*n*n*n)"),
])
def test_mathematical_growth_equivalence(left, right):
    assert same_growth_class(left, right)


@pytest.mark.parametrize("value", [
    "O(n^3)", "O(n)", "O(n^2*log(n))", "o(n^2)", "Omega(n^2)",
    "not O(n^2)", "O(n^2) or O(n)", "O(__import__('os').system('id'))",
    "O(n^2-n^2)", "O((n+1)**100000)", "O(n**True)", None,
])
def test_wrong_or_unmeasured_complexity_is_not_equivalent(value):
    assert not same_growth_class(value, "O(n^2)")


def test_unknown_growth_is_not_equal_to_unknown_growth():
    assert growth_class("O(2**n)") is None
    assert not same_growth_class("O(2**n)", "O(2**n)")


def test_new_grading_accepts_equivalent_notation_without_rewriting_legacy():
    task = _task_cohort(("coding",), 2, seed=2026091306, surface_profile="canonical")[0]
    expected = task.grade("")["expected"]
    answer = {**expected, "time_complexity": "Theta(n^2)"}
    response = "FINAL_ANSWER: " + json.dumps(answer)
    assert not grade_semantic_task(task, response, policy=LEGACY_GRADING_POLICY)["correct"]
    assert grade_semantic_task(task, response, policy=SEMANTIC_GRADING_POLICY)["correct"]
    assert task.grade(response)["correct"] is False
    answer["returns"][0]["pressure"][0] += 1
    wrong = "FINAL_ANSWER: " + json.dumps(answer)
    assert not grade_semantic_task(task, wrong, policy=SEMANTIC_GRADING_POLICY)["correct"]
