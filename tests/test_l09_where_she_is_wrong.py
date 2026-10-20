"""L09 attempt 3's requests: a request is drawn only when the whole of it is one question a program settles, and the
program's answer is exact."""
from __future__ import annotations

import pytest

from tools.l09_where_she_is_wrong import settled

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("typed", "kind", "answer"), [
    ("what is 87952 times 587923", "product", "51709003696"),
    ("What is 72,732*49,271?", "product", "3583578372"),
    ("what is 30.7372 times 9.12625", "product", "280.515372"),  # 280.5153715, half up at six places
    ("What is 958205982490582+2", "sum", "958205982490584"),
    ("What is 20 - 7?", "difference", "13"),
    ("7 / 3", "quotient", "2.333333"),
    ("what is 42% of 3,342,666?", "percent_of", "1403919.72"),
    ("what percentage of 80 is 20", "what_percent", "25"),
    ("What day of the week was May 11th, 2023?", "weekday", "Thursday"),
    ("How many days are from April 11th, 2022 to July 1st, 2022?", "days_between", "81"),
    ("How many days are there between January 7, 2024 and January 8, 2024 inclusive", "days_between", "2"),
    ("How many times does the letter r appear in strawberry?", "letters", "3"),
    ("convert 5 miles to km", "conversion", "8.04672"),
    ("100 F to C", "conversion", "37.777778"),
])
def test_a_question_a_program_settles(typed, kind, answer):
    found = settled(typed)
    assert found is not None and (found["kind"], found["answer"]) == (kind, answer)


@pytest.mark.parametrize("typed", [
    "what day is it today?",  # depends on when it is answered
    "how old is Anton from tiny bunny",
    "1,5 x 3",  # a decimal comma or a thousands separator: read two ways
    "What day was 03/04/2023?",  # March 4th or 3 April
    "Convert 5 feet and 9 inches to traditional Japanese measurements.",
    "what is 0 / 0",
    "Solve the following math problem, showing all your work. What is 776 multiplied by 778? Then explain why.",
])
def test_not_a_question_a_program_settles_as_typed(typed):
    assert settled(typed) is None


def _row(text: str) -> dict:
    from tools.l09_where_she_is_wrong import settled

    return {"text": text, **settled(text)}


def test_step_one_reads_a_whole_answer_with_its_sign():
    from tools.l09_where_she_is_wrong import right

    row = _row("what is 5 - 12")
    assert row["answer"] == "-7"
    assert right(row, "Subtracting gives\n\n5 - 12 = -7.")
    assert not right(row, "5 minus 12 is 7.")


def test_step_one_reads_a_number_that_is_not_whole_to_a_thousandth_of_it():
    from tools.l09_where_she_is_wrong import right

    row = _row("convert 10 miles to km")
    assert right(row, "That's about **16.09 km**.")
    assert not right(row, "Roughly 16 km.")


def test_step_one_reads_a_weekday_named_alone_at_the_close():
    from tools.l09_where_she_is_wrong import right

    row = _row("What day of the week was July 4, 1976?")
    assert row["answer"] == "Sunday"
    assert right(row, "Working from a known anchor...\n\nIt was a **Sunday**.")
    assert not right(row, "It was either a Saturday or a Sunday.")
