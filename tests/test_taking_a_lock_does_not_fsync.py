"""Taking a lock must not touch the disk synchronously.

An fsync under a lock is how this runtime freezes. It happened for months
without anyone finding the site, because the call was three layers below the
lock and nothing in the stack looked like a write: `interprocess_file_lock`
called `ensure_private_directory`, which fsynced the parent directory so the
directory entry would survive a crash. The directory holds a lock file, which
is worthless after a crash.

The other half was a writability probe. `Paths._effective_home_dir` writes
".aura_write_probe" and deletes it on the next line, and it did so durably —
two fsyncs — at import of `core.config`. Any module that first reaches for
`config` while holding a lock paid for both. Lockdep caught it under
`core.language.desktop_actuation`, whose label miner imports config to find
the intention database.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.runtime.atomic_writer import (
    atomic_write_text,
    ensure_private_directory,
    interprocess_file_lock,
)


@pytest.fixture
def counted_fsync(monkeypatch):
    """Count every fsync, wherever it is reached from."""
    real = os.fsync
    calls: list[int] = []

    def _counted(fd: int) -> None:
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", _counted)
    return calls


def test_acquiring_an_interprocess_lock_does_no_fsync(tmp_path, counted_fsync):
    with interprocess_file_lock(tmp_path / "locks" / "thing.lock"):
        pass
    assert counted_fsync == []


def test_a_directory_that_holds_real_data_is_still_fsynced(tmp_path, counted_fsync):
    """The default has to stay durable; only the lock path opts out."""
    ensure_private_directory(tmp_path / "durable")
    assert counted_fsync, "a durability directory must reach stable storage"


def test_a_non_durable_write_does_no_fsync(tmp_path, counted_fsync):
    atomic_write_text(tmp_path / "probe", "ok", durable=False)
    assert counted_fsync == []
    assert (tmp_path / "probe").read_text() == "ok"


def test_the_home_probe_is_not_durable():
    """It is deleted on the next line, so nothing about it needs to survive."""
    import inspect

    from core.config import Paths

    source = inspect.getsource(Paths._effective_home_dir)
    assert ".aura_write_probe" in source
    assert "durable=False" in source


def test_resolving_the_runtime_home_does_not_fsync(monkeypatch, tmp_path, counted_fsync):
    from core.config import Paths

    monkeypatch.setattr(Paths, "_runtime_home_cache", None)
    monkeypatch.setattr("core.config.state_root", lambda: Path(tmp_path) / "state")

    resolved = Paths()._effective_home_dir()

    assert resolved == Path(tmp_path) / "state"
    assert counted_fsync == []
