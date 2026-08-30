"""A request with a subject in it is not thereby a request to write a document.

"about" is a preposition. Counting it as a literary form turned every request
that mentioned what it was about into an authoring task — LIVE 2026-08-30,
"play 2048 for me ... tell me what you learn about the game" was routed to the
writer, failed to author, and reported that no file had been created, to a
request that never mentioned a file.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


@pytest.mark.parametrize(
    "asked",
    [
        "Please play 2048 for me. Open it in the browser, work out how the board "
        "behaves, and play it as well as you can. Tell me what you learn about "
        "the game as you go.",
        "play 2048 and tell me about your strategy",
        "Tell me about the weather",
        "Open Safari and tell me what you see about the page layout",
        "Make a chess move and tell me about the position",
    ],
)
def test_a_request_that_merely_names_its_subject_is_not_authoring(asked):
    assert not DesktopTaskSkill._objective_needs_authored_content(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "Write a note with three sentences about orcas",
        "write about the orcas we saw",
        "Write a haiku to a file called poem.txt",
        "Create a note on the Desktop with one sentence in it about what you did tonight",
        "Draft an email explaining the delay",
        "Compose a short piece describing the harbour",
    ],
)
def test_a_real_writing_request_still_reaches_the_writer(asked):
    assert DesktopTaskSkill._objective_needs_authored_content(asked)


def test_a_subject_counts_only_behind_a_verb_that_means_writing():
    """The whole distinction, stated once."""
    assert DesktopTaskSkill._objective_needs_authored_content("write about the harbour")
    assert not DesktopTaskSkill._objective_needs_authored_content("tell me about the harbour")
