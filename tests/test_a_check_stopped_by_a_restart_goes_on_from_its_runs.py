"""A test run of the outside check that finished whole is kept, and a check of the same commit goes on from it (F16).

LIVE 27 November 2026 her own yes's outside check of one draft ran half an hour four times, each stopped by a restart
and begun again from nothing. Bryan: "her processes should be able to pause and continue from a reset so that progress
isnt lost".
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar import verify_change

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "sums.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "tests" / "test_sums.py").write_text("from app.sums import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "sums")
    return root


def _checkout(repo: Path, where: Path, commit: str = "HEAD") -> Path:
    _git(repo, "worktree", "add", "-q", "--detach", str(where), commit)
    return where


def test_a_run_that_finished_is_taken_back_on_a_fresh_checkout_of_the_same_commit(repo, tmp_path, monkeypatch):
    first = _checkout(repo, tmp_path / "first")
    work = tmp_path / "work"
    work.mkdir()
    functions = {"app/sums.py": ["add"]}
    ran = verify_change.run_tests(first, ["tests/test_sums.py"], work, functions)
    assert ran["finished"] and ran["whole"] and ran["verdicts"], ran

    def never(*_args, **_kwargs):
        raise AssertionError("taken back from where it was kept, not run again")

    monkeypatch.setattr(verify_change, "_run_the_tests", never)
    second = _checkout(repo, tmp_path / "second")
    again = verify_change.run_tests(second, ["tests/test_sums.py"], work, functions)
    assert again["verdicts"] == ran["verdicts"]
    assert all(str(first.resolve()) not in line for line in again.get("ran", [])), "its paths are the new checkout's"
    assert again.get("ran") == [line.replace(str(first.resolve()), str(second.resolve())) for line in ran.get("ran", [])]


def test_a_run_of_another_commit_or_of_a_changed_checkout_is_run_again(repo, tmp_path, monkeypatch):
    first = _checkout(repo, tmp_path / "first")
    work = tmp_path / "work"
    work.mkdir()
    verify_change.run_tests(first, ["tests/test_sums.py"], work)
    runs: list[Path] = []
    real = verify_change._run_the_tests

    def counted(checkout, *args, **kwargs):
        runs.append(checkout)
        return real(checkout, *args, **kwargs)

    monkeypatch.setattr(verify_change, "_run_the_tests", counted)
    (repo / "app" / "sums.py").write_text("def add(a, b):\n    return b + a\n")
    _git(repo, "commit", "-qam", "commuted")
    verify_change.run_tests(_checkout(repo, tmp_path / "later"), ["tests/test_sums.py"], work)
    changed = _checkout(repo, tmp_path / "changed", "HEAD~1")
    (changed / "app" / "sums.py").write_text("def add(a, b):\n    return a - b\n")
    verify_change.run_tests(changed, ["tests/test_sums.py"], work)
    assert len(runs) == 2, "a new commit, and a checkout no longer its commit, are each run"


def test_a_run_that_did_not_finish_is_not_kept(repo, tmp_path):
    checkout = _checkout(repo, tmp_path / "first")
    where = verify_change._the_run_kept_as(checkout, ["tests/test_sums.py"], None, None, 0, False)
    verify_change._keep_run(where, checkout, {"finished": False, "whole": True, "verdicts": {"t": True}})
    assert not where.exists()


def test_kept_runs_are_swept_once_no_check_can_take_them_back(repo, tmp_path):
    import os
    import time

    checkout = _checkout(repo, tmp_path / "first")
    where = verify_change._the_run_kept_as(checkout, ["tests/test_sums.py"], None, None, 0, False)
    verify_change._keep_run(where, checkout, {"finished": True, "whole": True, "verdicts": {"t": True}})
    assert where.exists() and where in verify_change._RUNS_THIS_CHECK
    verify_change._sweep_kept_runs(repo)
    assert where.exists(), "a fresh one stays for the check that may go on from it"
    old = time.time() - verify_change._KEPT_RUNS_LAST_S - 60
    os.utime(where, (old, old))
    verify_change._sweep_kept_runs(repo)
    assert not where.exists()
    verify_change._RUNS_THIS_CHECK.clear()
