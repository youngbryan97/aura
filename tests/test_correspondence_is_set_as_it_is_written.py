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


@pytest.mark.parametrize(("given", "taken"), [
    ("Warmly, Aura", True),
    ("With love,\nAura", True),
    ("The rest will still be there tomorrow.", False),  # not a sign-off: not taken
    ("", False),
])
def test_a_missing_part_is_taken_from_the_writer_only_when_it_is_that_part(given, taken):
    from core.language.the_form_of_a_kind import with_the_part

    letter = the_form_for("a letter")
    fit = with_the_part(["Dear Sam,", "See you soon."], letter, "closing and name", given)
    assert (fit is not None) is taken
    if taken:
        assert fit.paragraphs[-1] == "Aura" and fit.paragraphs[-2].endswith(",") and not fit.missing


@pytest.mark.slow
def test_a_letter_left_unsigned_is_signed_by_its_writer_when_asked(tmp_path):
    """The writer is asked for exactly the part its form is missing; the program then saves the whole letter."""
    import asyncio

    from core.rebuilding.checks_a_person_makes import _what_a_file_says
    from core.rebuilding.parts_a_maker_knows import the_page_for, what_she_knows_how_to_make
    from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt
    from core.rebuilding.using_it_by_its_controls import _Part, _Writing, write_it_in
    from core.rebuilding.what_a_program_does import Feature, Genome

    genome = Genome(name="Q", work="A white Letter page.", features=[Feature(name="Export as PDF")])
    page = tmp_path / "q.html"
    ProgramAsBuilt("Q").with_part(the_page_for(genome)).with_part(Part("save as", what_she_knows_how_to_make(genome).given["Export as PDF"][0].code)).write(page)
    asked: list[str] = []

    async def ask(prompt, schema, n):
        asked.append(schema.__name__)
        if schema is _Writing:
            return _Writing(title="To Sam", paragraphs=["Dear Sam, it was good to see you.", "Come back soon."])
        return _Part(text="Warmly, Aura")

    said: list[str] = []
    used = asyncio.run(write_it_in(page, "write a one-page letter and export it as a PDF to my Desktop", tmp_path / "out", ask, tell=said.append))
    assert used is not None and used.ok, used and used.why_not
    text = " ".join(_what_a_file_says(used.files[0].read_bytes()).split())
    assert text.startswith("Dear Sam, it was good to see you.") and text.endswith("Warmly, Aura")
    assert asked == ["_Writing", "_Part"] and any("I added the one I would end it with" in line for line in said)
