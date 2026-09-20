"""A file named by its short name is found in her tree, not declared absent.

LIVE 2026-09-16: two turns after a directory count had listed phi_core.py
as the largest file under core/consciousness, a repair turn that named it
by that short name read "No file exists at phi_core.py" off the grounding
channel and told the person the file was gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation import filesystem_check as fc


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / "core" / "consciousness").mkdir(parents=True)
    (tmp_path / "core" / "consciousness" / "phi_core.py").write_text("PHI = 1\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "twin.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b" / "twin.py").write_text("B = 2\n", encoding="utf-8")
    monkeypatch.setattr(fc, "_allowed_roots", lambda: [tmp_path])
    fc._NAME_INDEX.clear()
    fc._READ_HISTORY.set(())
    return tmp_path


def test_a_unique_short_name_is_the_file(tree: Path) -> None:
    read = fc.requested_file_read("what is in phi_core.py?")

    assert read is not None and read.exists and not read.refusal
    assert Path(read.path) == tree / "core" / "consciousness" / "phi_core.py"
    assert "PHI = 1" in read.text


def test_an_ambiguous_short_name_asks_which_rather_than_denying(tree: Path) -> None:
    read = fc.requested_file_read("open twin.py for me")

    assert read is not None and read.exists
    assert "2 files" in read.refusal and "a/twin.py" in read.refusal and "b/twin.py" in read.refusal


def test_a_name_that_is_nowhere_is_still_reported_missing(tree: Path) -> None:
    read = fc.requested_file_read("read the file nowhere_at_all.py please")

    assert read is not None and not read.exists
