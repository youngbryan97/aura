"""Writing is set in the form of its kind: each part found where its form puts it, no word changed, nothing made up.

LIVE 2026-10-06 a letter written in a program she built began "Dear Future Self, I am writing this on a quiet
Tuesday evening" on one line. The forms are data (core/language/the_form_of_a_kind.py): a letter, an email, a
memo; a kind she does not know the form of is left as it was written.
"""
from __future__ import annotations

import pytest

from core.language.the_form_of_a_kind import fitted, the_form_for

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("task", "written", "set_as", "missing"), [
    ("write a one-page letter and export it", ["Dear Future Self, I am writing this on a quiet evening.", "I will see you soon. With all my warmth, Aura"],
     ["Dear Future Self,", "I am writing this on a quiet evening.", "I will see you soon.", "With all my warmth,", "Aura"], []),
    ("a letter to my landlord", ["Dear Sam,", "Body here.", "Warmly,", "Aura"], ["Dear Sam,", "Body here.", "Warmly,", "Aura"], []),
    ("write a letter to my sister", ["Dear Ana, I miss you.", "Come visit soon."], ["Dear Ana,", "I miss you.", "Come visit soon."],
     ["closing and name"]),
    ("draft an email to the team", ["Hi Ms. Rivera, thank you for the note.", "Thanks, Jordan Lee"],
     ["Hi Ms. Rivera,", "thank you for the note.", "Thanks,", "Jordan Lee"], []),
    ("an email to say I'm late", ["Running ten minutes late."], ["Running ten minutes late."], []),  # an email's greeting is not expected
    ("write a memo about the move", ["To: All staff From: Facilities Subject: The move", "The office moves on Friday."],
     ["To: All staff", "From: Facilities", "Subject: The move", "The office moves on Friday."], []),
])
def test_writing_is_set_in_the_form_of_its_kind(task, written, set_as, missing):
    fit = fitted(written, the_form_for(task))
    assert fit.paragraphs == set_as and fit.missing == missing


@pytest.mark.parametrize(("task", "written"), [
    ("write a letter", ["Hello, world is the first thing a programmer prints.", "Meeting notes. Thanks to everyone who came."]),
    ("write a short story", ["Dear reader, this is a story.", "The end."]),  # a kind whose form she does not know is left as written
])
def test_what_is_not_a_part_of_the_form_is_left_as_it_was(task, written):
    assert fitted(written, the_form_for(task)).paragraphs == written
