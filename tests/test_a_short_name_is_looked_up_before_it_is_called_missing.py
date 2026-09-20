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


@pytest.mark.asyncio
async def test_the_index_is_never_built_on_the_loop_thread(tree: Path, monkeypatch) -> None:
    """LIVE 2026-09-19: a 5.9s loop stall inside the routing phase, the walk
    of the tree running on the loop. On the loop the first look serves an
    empty index and queues the walk; the walk, once done, serves the file."""
    import asyncio

    walked: list[str] = []
    real_walk = fc._walk_names

    def spy(root):
        walked.append(__import__("threading").current_thread().name)
        return real_walk(root)

    monkeypatch.setattr(fc, "_walk_names", spy)
    loop_thread = __import__("threading").current_thread().name

    first = fc._found_by_name("phi_core.py")
    assert first == (), "the loop thread was served a walk"
    for _ in range(50):
        await asyncio.sleep(0.05)
        if fc._found_by_name("phi_core.py"):
            break
    assert fc._found_by_name("phi_core.py") == (str(tree / "core" / "consciousness" / "phi_core.py"),)
    assert walked and all(name != loop_thread for name in walked)
