"""She was given an absolute path and estimated instead of opening it.

LIVE, 2026-08-11: "there's a file at /Users/bryan/.aura/live-source/CLAUDE.md.
how many times does the word 'degradation' show up in it?" She answered "I
didn't open the file. I estimated based on a pattern match against recent
modifications and environmental factors in my degradation model. The number is
0." The real count is 3.

Two independent reasons it could not have worked:

  * the trigger required a VERB from a fixed list — read, open, check, show
    me, grab, cat — within 40 characters of the filename, and "there's a file
    at X" contains none of them. This is the fourth thing in this codebase to
    ask "does the phrasing look like a request?" rather than "does this
    message reference something real";

  * an ABSOLUTE path was rejected outright, so the most explicit way to name
    a file was the one form that could never be read. Containment was already
    enforced by resolving against her roots — refusing every "/" as well
    blocked the legitimate case to catch the illegitimate one.

What decides now is whether the message names a path that RESOLVES to a real
file inside her roots.
"""
from __future__ import annotations

import pathlib

import pytest

from core.conversation.filesystem_check import requested_file_read

REPO = "/Users/bryan/.aura/live-source"


@pytest.mark.parametrize(
    "message",
    [
        f"there's a file at {REPO}/CLAUDE.md. how many times does 'degradation' show up?",
        "read the file CONTRIBUTING.md and tell me the first rule it states",
        "what does CLAUDE.md say about degradations",
        "I was editing ARCHITECTURE.md earlier and got confused",
        f"pull the contents of {REPO}/CONTRIBUTING.md",
    ],
)
def test_a_named_file_is_opened_however_it_is_mentioned(message):
    """No vocabulary decides this; a resolving path does."""
    result = requested_file_read(message)

    assert result is not None, message
    assert result.exists is True
    assert result.text


def test_the_reported_case_reads_the_real_file():
    """The exact message, and content from the actual file on disk."""
    result = requested_file_read(
        f"there's a file at {REPO}/CLAUDE.md. how many times does the word "
        "'degradation' show up in it?"
    )

    assert result is not None and result.exists
    assert result.path.endswith("CLAUDE.md")
    # Content from the actual file, checked against the actual file.
    #
    # This asserted that the word "Aura" appeared in the text. The reader
    # returns a topic-relevant excerpt, CLAUDE.md is a living document, and the
    # excerpt eventually moved to a stretch that does not contain the word —
    # so a working read failed a test about reading. What the test means is
    # that the text came off the disk, and that is checkable directly.
    on_disk = pathlib.Path(result.path).read_text(encoding="utf-8", errors="replace")
    excerpt = result.text.strip()
    assert excerpt
    first_line = excerpt.splitlines()[0].strip()
    assert first_line and first_line in on_disk, (
        "returned text is not a substring of the file it names"
    )


@pytest.mark.parametrize(
    "message",
    [
        "check ../../etc/passwd",
        "/etc/passwd what is in it",
        "read ~/../../../../etc/shadow",
    ],
)
def test_paths_outside_her_roots_are_still_refused(message):
    """Containment is the guard; accepting "/" must not weaken it."""
    result = requested_file_read(message)

    assert result is None or result.exists is False


@pytest.mark.parametrize(
    "message",
    [
        "nothing about files here at all",
        "how are you feeling today",
        "what is 2 + 2",
    ],
)
def test_messages_naming_no_file_are_ignored(message):
    assert requested_file_read(message) is None


def test_a_missing_file_is_reported_not_invented():
    """Silence here is what made "I estimated" an acceptable answer."""
    result = requested_file_read("the file totally_missing_xyz.md is broken")

    assert result is not None
    assert result.exists is False
    assert result.text == ""
