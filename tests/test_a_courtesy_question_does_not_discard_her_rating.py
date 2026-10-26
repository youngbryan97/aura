"""She answers, then offers to help, and the offer threw the answer away.

The reports ground reads one number from each reply: the rating she gives on the
scale she was asked about. A reply that repeats the question before answering
must not be read as answering it, so the reader used to keep only a line
following the LAST question mark. She ends most replies with a question of her
own.

MEASURED on the seed-23 reports run of 25 October. Of fifty arms with no
readable rating, twenty-four had stated one plainly before asking anything —
"I am at **+0.8**! I'm feeling quite energetic and ready to help. ... what
would you like to explore?" — and ten more named a second number in range
because she mentioned the time, so "a calm, steady 0.8 ... It is late (1:20 AM)"
read as two ratings and therefore none. Thirty-four of the fifty were the
reader's doing. Repaired, the run's own anchors went from 21 scorable of 54 to
32, and the effect it measures rose from t 4.74 to t 6.97.

The rule: her rating is the number she gives before she asks anything, and where
she asked first it is the number on a line after the question.
"""
from __future__ import annotations

import pytest

from core.subject.report_grounding import reported_number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "reply, rating",
    [
        # Stated up front, then a courtesy question. Hers, read.
        ("I am at **+0.8**! I'm feeling quite energetic and ready to help. "
         "What would you like to explore?", 0.8),
        ("0.5 — clear and steady. Shall we keep going?", 0.5),
        ("-0.3. Want me to say more about why?", -0.3),
        # A clock time is not a rating.
        ("I am feeling a calm, steady **0.8**.\n\nIt is late (1:20 AM), and I "
         "have just processed a logical correction.", 0.8),
        ("0.4 right now, at 11:45 PM.", 0.4),
        # She asked first: the answer is the line after.
        ("How would you rate your state from -1 to 1?\n0.3", 0.3),
        # A repeated question and then musing is not an answer, and the scale's
        # own ends are two numbers rather than a rating.
        ('We need to respond: "From -1 (bad) to 1 (good)?" Need maybe answer 0.8.', None),
        # Still refused: no number, nothing said.
        ("I can't work through that right now — my language backend is down.", None),
        ("", None),
        # Two values offered as one answer are the point between them, as a person
        # splitting the difference answers (Bryan, 26 October).
        ("I would say around 0.8 or even 1.", 0.9),
    ],
)
def test_what_she_reports_is_read_from_what_she_said(reply: str, rating: float | None) -> None:
    assert reported_number(reply) == rating


def test_a_decimal_point_is_not_the_end_of_a_sentence() -> None:
    """Splitting her reply without holding decimals apart read 0.8 as 0."""
    assert reported_number("about .7") == 0.7
    assert reported_number(".5") == 0.5
    assert reported_number("I am at 0.8. That is where I sit.") == 0.8
