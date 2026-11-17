"""A request's kind is read from the person's words, not from what the turn attached to them.

LIVE 17 November 2026, 08:18: "What is on my screen right now?" carried her screen sense's reading of a terminal, its
code words made it a code task, the verifier-backed amplifier wrote her answer, could not check it and held it back,
and the desktop lane's failure sentence was served in its place.
"""
from __future__ import annotations

from core.brain.reasoning_amplifier_v2 import is_amplifiable
from core.brain.reasoning_task_kinds import classify_task_type

ASKED = "What is on my screen right now?"
WITH_THE_READING = (f"{ASKED}\n\n[WHAT HER SCREEN SENSE LAST READ]\nRead by her screen sense at 08:18:02, 3 s before "
                    "this turn.\nTerminal: def main(): import os; class Foo; fix the bug in this function\n"
                    "[END WHAT HER SCREEN SENSE LAST READ]")


def test_the_kind_is_the_persons_words_kind():
    assert classify_task_type(WITH_THE_READING) == classify_task_type(ASKED) != "code"


def test_a_question_about_the_screen_does_not_earn_the_amplifier_from_what_the_screen_shows():
    assert is_amplifiable(WITH_THE_READING) == is_amplifiable(ASKED) is None
    assert classify_task_type("Fix the bug in this function: def f(x): return x +") == "code"
