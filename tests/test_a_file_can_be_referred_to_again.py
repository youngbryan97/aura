"""The second question about a file is the normal case, not the exception.

LIVE, 2026-08-19. She read accounts.py by full path and answered correctly.
The very next turn — "in that accounts.py you just read, the close() method has
a sign error, which line?" — named the file by BASENAME, which resolves
nowhere outside her two roots. Nothing was read, and she invented an
implementation: a close() iterating DebitEntry and CreditEntry objects that
appear nowhere in the file, then confidently proposed fixing signs in code
that does not exist.

A file that has been read is a file she is entitled to read again, and a
follow-up naming it by its short name means the one she just opened. Every
multi-step task on a real artifact depends on this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import (
    files_already_read,
    remember_file_read,
    requested_file_read,
)


@pytest.fixture
def opened(tmp_path: Path) -> Path:
    target = tmp_path / "accounts.py"
    target.write_text(
        "def close(self, name):\n"
        "    acct = self.account(name)\n"
        "    self.account('retained').post(-amount)\n"
    )
    read = requested_file_read(f"read {target} and tell me what it does")
    assert read is not None and read.exists
    return target


def test_a_follow_up_by_bare_name_reaches_the_same_file(opened: Path):
    read = requested_file_read("in that accounts.py you just read, which line is wrong?")
    assert read is not None
    assert read.exists
    assert read.path == str(opened)
    assert "retained" in read.text


def test_reading_a_file_records_it(opened: Path):
    assert str(opened) in files_already_read()


def test_a_bare_name_never_read_resolves_to_nothing(tmp_path: Path):
    """Memory of a read is the grant; a name alone is not."""
    stranger = tmp_path / "unseen.py"
    stranger.write_text("x = 1\n")
    read = requested_file_read("what does unseen.py say?")
    assert read is None or not read.exists


def test_the_history_is_bounded():
    from core.conversation.filesystem_check import _READ_HISTORY_LIMIT

    for index in range(_READ_HISTORY_LIMIT + 5):
        remember_file_read(f"/tmp/file{index}.py")
    assert len(files_already_read()) <= _READ_HISTORY_LIMIT


def test_the_newest_read_is_first():
    remember_file_read("/tmp/one.py")
    remember_file_read("/tmp/two.py")
    assert files_already_read()[0] == "/tmp/two.py"


def test_re_reading_does_not_duplicate():
    remember_file_read("/tmp/same.py")
    remember_file_read("/tmp/same.py")
    assert files_already_read().count("/tmp/same.py") == 1


def test_a_path_is_not_treated_as_a_bare_name(opened: Path):
    """Only a short name refers back; a full path is resolved on its own terms."""
    read = requested_file_read("read /tmp/definitely-absent-9999/accounts.py")
    assert read is None or read.path != str(opened)
