"""Containment is not absence, and a path the person named is the grant.

LIVE, 2026-08-19. Asked to read a file by its full path, the reading came back:

    No file exists at /private/tmp/.../ledger/README.md.

The file existed. It sat outside her two allowed roots — the repo and her
state directory — and a containment failure was reported as nonexistence. She
was told the file was not there, so she told the person she had no filesystem
access and asked them to paste the contents. Every task that begins by reading
something the person names was impossible: a paper, a spreadsheet, an
unfamiliar repository.

Two separate faults. A false report of absence, which is the "absence of a
check reported as a passed check" shape one level down; and a reach that could
never include a file the person spelled out in their own request.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.conversation.filesystem_check import requested_file_read


def test_a_file_the_person_named_by_absolute_path_is_read(tmp_path: Path):
    target = tmp_path / "notes.md"
    target.write_text("the invariant is that trial_balance is zero\n")
    read = requested_file_read(f"read {target} and tell me what it says")
    assert read is not None
    assert read.exists
    assert not read.refusal
    assert "trial_balance" in read.text


def test_a_file_that_is_genuinely_absent_is_reported_as_absent(tmp_path: Path):
    read = requested_file_read(f"read {tmp_path / 'nothing.txt'} and tell me what it says")
    assert read is not None
    assert not read.exists
    assert not read.refusal


def test_a_refusal_is_not_reported_as_absence(tmp_path: Path):
    """The whole defect: she said a file that exists was not there."""
    secret = tmp_path / "credentials.json"
    secret.write_text('{"api_key": "hunter2"}\n')
    read = requested_file_read(f"read {secret} and tell me what it says")
    assert read is not None
    assert read.refusal
    assert read.text == ""
    # And it must not claim the file is missing, which is what taught the model
    # it could not read files at all.
    assert read.exists


@pytest.mark.parametrize(
    "name",
    ["credentials.json", "secrets.yaml", "cookies.sqlite", "aws_credentials.txt"],
)
def test_sensitive_names_are_refused_when_named(tmp_path: Path, name: str):
    target = tmp_path / name
    target.write_text("sensitive marker 4242\n")
    read = requested_file_read(f"read {target} and tell me what it says")
    assert read is not None
    assert read.refusal
    assert "4242" not in (read.text or "")


@pytest.mark.parametrize("name", [".env", "id_rsa", ".netrc"])
def test_names_that_are_not_even_file_shaped_read_nothing(tmp_path: Path, name: str):
    """A second, independent reason these never reach a read.

    The extractor does not treat them as file-shaped, so they are refused by
    not being found at all. Recorded here because the deny list would look
    untested otherwise, and because a change to the extractor must not
    silently make them reachable.
    """
    target = tmp_path / name
    target.write_text("sensitive marker 4242\n")
    read = requested_file_read(f"read {target} and tell me what it says")
    assert read is None or not (read.text or "").strip()


def test_the_reading_says_which_of_the_two_it_was(tmp_path: Path):
    from core.brain.observable_registry import _read_file

    secret = tmp_path / "credentials.json"
    secret.write_text('{"api_key": "hunter2"}\n')
    refused = asyncio.run(_read_file(f"read {secret} and tell me what it says"))
    assert "do not read from" in refused
    assert "hunter2" not in refused

    missing = asyncio.run(_read_file(f"read {tmp_path / 'gone.txt'} and tell me what it says"))
    assert "No file exists" in missing


def test_a_relative_path_still_resolves_inside_her_roots():
    """The original containment behaviour is unchanged for unnamed roots."""
    read = requested_file_read("read CONTRIBUTING.md and tell me the first rule")
    assert read is not None
    assert read.exists
