"""A full stop is not part of the path.

LIVE, 2026-08-25: "Something weird is happening in a little project of mine at
/private/tmp/.../invoice-tools. There's no error and no failing test, but the
second invoice comes out with the first one's lines in it."

The reply was a capability inventory — "79 registered entries; 79 entries
explicitly marked available" — ending in a declaration that no tool would be
run. The path had been captured as `invoice-tools.` with the sentence's full
stop attached, so it did not resolve, so nothing pointed at anything real, so
no tool was selected, so the model was asked to answer from nothing and its
attempts were exhausted into a canned catalogue.

Six regexes in this tree read a path, each written where it was needed. These
hold the shared reader they now go through, and check the readers that still
have their own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.language.named_paths import (
    first_existing_path,
    named_paths,
    trim_sentence_punctuation,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "invoice.py").write_text("def add_line(item, price, lines=[]):\n    pass\n")
    return tmp_path


def test_a_directory_at_the_end_of_a_sentence_resolves(project: Path) -> None:
    asked = f"Something weird is happening in a project of mine at {project}. No error."
    assert first_existing_path(asked) == project


@pytest.mark.parametrize("mark", [".", ",", ";", ":", "!", "?", ")", "]", '"', "'"])
def test_every_closing_mark_is_trimmed(project: Path, mark: str) -> None:
    assert first_existing_path(f"look at {project}{mark} thanks") == project


def test_an_opening_bracket_is_trimmed_too(project: Path) -> None:
    assert first_existing_path(f"look at ({project}) please") == project


def test_the_raw_capture_is_kept_as_well(project: Path) -> None:
    """A file really can be called `notes.`, and the disk settles it."""
    odd = project / "notes."
    odd.write_text("x")
    assert first_existing_path(f"open {odd}") == odd


def test_a_path_that_does_not_exist_resolves_to_nothing() -> None:
    assert first_existing_path("what about /nope/does/not/exist.") is None
    assert named_paths("what about /nope/does/not/exist.")


def test_prose_with_no_path_finds_none() -> None:
    assert named_paths("what is the capital of Peru") == ()
    assert first_existing_path("I think our http/2 support is fine") is None


def test_trimming_leaves_a_bare_path_alone() -> None:
    assert trim_sentence_punctuation("/etc/hosts") == "/etc/hosts"
    assert trim_sentence_punctuation("/etc/hosts.") == "/etc/hosts"


def test_the_request_points_at_something_real_again(project: Path) -> None:
    """The site the live failure ran through."""
    from core.intent.capability_selection import _points_at_something_real

    asked = (
        f"Something weird is happening in a little project of mine at {project}. "
        "There's no error and no failing test, but the second invoice comes out "
        "with the first one's lines in it. What's actually going on?"
    )
    assert _points_at_something_real(asked) is True
    assert _points_at_something_real("what is the capital of Peru") is False
