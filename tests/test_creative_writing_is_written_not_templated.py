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
