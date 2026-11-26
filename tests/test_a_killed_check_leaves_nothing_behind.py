"""What a run of the outside check left when it was killed is taken away by the next run (26 November 2026).

A run takes its two checkouts away when it ends, and a run killed by her restart or by her gate's time running out
cannot. On 26 November fifty of them held 48 GB in the temporary folder. Each run now names its process beside its
checkouts, and a run takes away what runs whose process is gone left, and what runs from before that left, once older
than any run lasts.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from outside_bar import verify_change

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                   cwd=repo, check=True, capture_output=True)


def _a_run_left(repo: Path, owner: str | None, age_s: float = 0.0) -> Path:
    work = Path(tempfile.mkdtemp())
    checkout = work / "a1b2c3d4e5f6" / repo.name
    _git(repo, "worktree", "add", "-q", "--detach", str(checkout), "HEAD")
    if owner is not None:
        (work / verify_change._OWNER).write_text(owner)
    if age_s:
        old = time.time() - age_s
        os.utime(work, (old, old))
    return checkout.resolve()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a")
    return repo


def test_what_a_killed_run_left_is_taken_away_and_a_running_one_is_not(repo):
    dead = subprocess.Popen(["true"])
    dead.wait()
    killed = _a_run_left(repo, str(dead.pid))
    running = _a_run_left(repo, str(os.getpid()))
    before_owners_old = _a_run_left(repo, None, age_s=2 * verify_change._LEFT_BEFORE_OWNERS_S)
    before_owners_new = _a_run_left(repo, None)
    try:
        assert set(verify_change.what_killed_runs_left(repo)) == {killed, before_owners_old}
        verify_change._take_away_what_killed_runs_left(repo)
        assert not killed.parent.parent.exists() and not before_owners_old.parent.parent.exists()
        assert running.exists() and before_owners_new.exists()
        assert verify_change.what_killed_runs_left(repo) == []
    finally:
        for checkout in (running, before_owners_new):
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(checkout)], capture_output=True)
