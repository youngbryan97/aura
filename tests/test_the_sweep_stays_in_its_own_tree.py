"""The metabolism sweep cleans this checkout and nobody else's.

LIVE 2026-09-15, during shutdown: "Metabolism: 103 files, 2368 dirs removed
(1270.80 MB reclaimed) in 80.2s". The sweep walked every worktree under
.claude/ — thirty-six of them, three with campaigns running — and deleted
their caches and any .tmp file they had open. A directory with its own .git
is somebody else's checkout whatever it is called.
"""

from __future__ import annotations

from core.systems.metabolism import MetabolismEngine


def test_worktrees_and_nested_checkouts_are_not_entered(tmp_path):
    (tmp_path / "core" / "__pycache__").mkdir(parents=True)
    (tmp_path / "core" / "__pycache__" / "a.pyc").write_bytes(b"x")
    (tmp_path / "scratch.tmp").write_bytes(b"x")

    other = tmp_path / ".claude" / "worktrees" / "theirs" / "core" / "__pycache__"
    other.mkdir(parents=True)
    (other / "b.pyc").write_bytes(b"x")
    (tmp_path / ".claude" / "worktrees" / "theirs" / "run.tmp").write_bytes(b"x")

    vendored = tmp_path / "vendor" / "checkout"
    (vendored / "__pycache__").mkdir(parents=True)
    (vendored / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    (vendored / "work.tmp").write_bytes(b"x")

    report = MetabolismEngine(root_dir=tmp_path)._scan_and_purge_sync()

    assert not (tmp_path / "core" / "__pycache__").exists()
    assert not (tmp_path / "scratch.tmp").exists()
    assert other.exists() and (other / "b.pyc").exists()
    assert (tmp_path / ".claude" / "worktrees" / "theirs" / "run.tmp").exists()
    assert (vendored / "__pycache__").exists() and (vendored / "work.tmp").exists()
    assert report.dirs_removed == 1 and report.files_removed == 1


def test_a_log_in_the_tree_is_a_record_not_waste(tmp_path):
    """artifacts/closeout/ holds the runner and detached logs of every sealed
    campaign, tracked in git. The sweep deleted 103 of them, 21 tracked."""
    import os
    import time

    evidence = tmp_path / "artifacts" / "closeout" / "campaign"
    evidence.mkdir(parents=True)
    old_log = evidence / "runner.log"
    old_log.write_text("what the campaign did\n", encoding="utf-8")
    stale = time.time() - 30 * 86400
    os.utime(old_log, (stale, stale))

    report = MetabolismEngine(root_dir=tmp_path, days_threshold=7)._scan_and_purge_sync()

    assert old_log.exists()
    assert report.files_removed == 0
