"""A haiku is freeform writing, so it reaches the writer.

Asked to "write a haiku to a file called poem.txt on my desktop", the plan
composed the file body with the deterministic fallback:

    Notes on the requested subject: The requested subject is the focus of this
    note. The important part is to describe the subject clearly, ground it in
    concrete details, and preserve enough context that the note is useful
    after the moment of writing has passed.

Correctly created, correctly saved, and empty of content — the exact failure
the synthesis path was built to end. It happened because the freeform-writing
predicate listed paragraph, note, document, essay, summary, report and journal
entry, and no creative form: a poem, a story, a joke and a letter were all
outside the enumeration, so they fell to the composer that describes what a
note should contain instead of containing it.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


@pytest.mark.parametrize(
    "objective",
    [
        "write a haiku to a file called poem.txt on my desktop",
        "write a short story about orcas into story.md on my desktop",
        "compose a limerick and save it to fun.txt",
        "draft a letter to my landlord and put it in letter.txt",
        "write a note about the meeting to notes.txt",
    ],
)
def test_a_writing_request_reaches_the_writer(objective: str) -> None:
    assert DesktopTaskSkill._objective_requests_freeform_written_content(objective)


@pytest.mark.parametrize(
    "objective",
    [
        "create a file called data.csv on my desktop",
        "count the python files in core/runtime",
        "open Notes",
    ],
)
def test_a_non_writing_request_is_not_claimed(objective: str) -> None:
    assert not DesktopTaskSkill._objective_requests_freeform_written_content(objective)


def test_a_creative_request_is_a_written_artifact() -> None:
    """The artifact predicate gates whether the model is asked to write at all."""
    assert DesktopTaskSkill._objective_requests_written_artifact(
        "write a haiku to a file called poem.txt on my desktop"
    )


@pytest.mark.parametrize(
    "objective",
    [
        "make a file on my Desktop called aura_note.txt with one sentence in it "
        "about what you did tonight",
        "save one sentence about tonight to a file",
        "put two lines about the weather in weather.txt",
        "add a short note about the meeting to notes.txt",
        "jot a line about this in scratch.txt",
    ],
)
def test_a_verb_of_production_is_enough_to_reach_the_writer(objective):
    """The verb list held "make up" and not "make".

    LIVE 2026-08-26: "make a file on my Desktop with one sentence in it about
    what you did tonight" fell through to the deterministic composer and the
    file held "Notes on the requested subject: The requested subject is the
    focus of this note." Correctly created, correctly saved, and empty of
    content — for the third time, reached by a phrasing nobody had listed.

    Verbs of production are a small closed class and the same one every
    request uses. Literary forms are open, and there is always another.
    """
    assert DesktopTaskSkill._objective_requests_freeform_written_content(objective)


@pytest.mark.parametrize(
    "objective",
    [
        "open the notes app",
        "delete the file called old.txt",
        "take a screenshot",
        "close the window",
    ],
)
def test_a_request_that_authors_nothing_is_not_freeform_writing(objective):
    assert not DesktopTaskSkill._objective_requests_freeform_written_content(objective)
