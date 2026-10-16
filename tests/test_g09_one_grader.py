"""Both arms of a G09 request are graded by one function: a right answer in LaTeX is right, a wrong one wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MATH500 = Path("~/.aura/benchmarks/math500").expanduser()
pytestmark = pytest.mark.skipif(not (MATH500 / "grading" / "grader.py").exists(), reason="MATH-500's grader absent")


@pytest.fixture(autouse=True)
def _math500_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(MATH500))
    yield
    sys.modules.pop("grading", None)


@pytest.mark.parametrize(
    ("domain", "text", "truth", "right"),
    [
        ("dm_probability__swr_p_sequence", "so \\boxed{\\frac{1}{72}}", "1/72", True),
        ("dm_probability__swr_p_sequence", "1/72", "1/72", True),
        ("dm_algebra__sequence_nth_term", "\\boxed{231w + 10205}", "231*w + 10205", True),
        ("dm_numbers__is_prime", "\\boxed{\\text{Yes}}", "True", True),
        ("dm_numbers__is_prime", "\\boxed{\\text{Yes}}", "False", False),
        ("dm_numbers__list_prime_factors", "\\boxed{2,\\ 3,\\ 17}", "2, 3, 17", True),
        ("dm_numbers__list_prime_factors", "2, 3", "2, 3, 17", False),
        ("bbh_web_of_lies", "Step by step ... So the answer is Yes.", "Yes", True),
        ("bbh_web_of_lies", "Yes", "Yes", True),
        ("bbh_web_of_lies", "No", "Yes", False),
    ],
)
def test_one_function_grades_a_draft_and_a_kept_answer_alike(domain, text, truth, right):
    from tools.g09_one_grader import one_grade

    assert one_grade(domain, text, truth) is right
