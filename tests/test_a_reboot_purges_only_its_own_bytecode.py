"""A reboot clears this checkout's bytecode and nobody else's.

LIVE 2026-09-24: the purge walked everything under the project root, the
virtualenv and 61 worktrees of other sessions included. A reboot spent 29.9 s
there, every dependency imported cold afterwards, and bytecode was deleted
under other checkouts' running jobs.
"""
from __future__ import annotations

from pathlib import Path

import aura_main


def _pyc(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


def test_only_this_checkouts_own_bytecode_is_purged(tmp_path):
    own = _pyc(tmp_path / "core" / "__pycache__" / "engine.cpython-312.pyc")
    stray = _pyc(tmp_path / "core" / "legacy.pyc")
    venv = _pyc(tmp_path / ".venv" / "lib" / "sqlalchemy" / "__pycache__" / "orm.cpython-312.pyc")
    worktree = _pyc(tmp_path / ".claude" / "worktrees" / "other" / "core" / "__pycache__" / "x.pyc")
    nested = tmp_path / "vendored_checkout"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: elsewhere")
    theirs = _pyc(nested / "pkg" / "__pycache__" / "y.pyc")

    aura_main.clean_artifacts(tmp_path)

    assert not own.exists() and not own.parent.exists()
    assert not stray.exists()
    assert venv.exists(), "installed dependencies keep their bytecode"
    assert worktree.exists(), "another session's worktree is not this checkout"
    assert theirs.exists(), "a checkout nested here keeps its own bytecode"
