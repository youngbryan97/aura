"""A register cannot compare a failed batch with a different source revision."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_test_chunks as runner

NODE = "tests/test_example.py::test_example"


@pytest.fixture
def committed_tree(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
             "-c", "commit.gpgsign=false", *args], cwd=repo, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()

    git("init", "-q")
    (repo / "state.py").write_text("value = 1\n")
    git("add", "-f", "state.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    return repo, git("rev-parse", "--short", "HEAD")


def test_the_runner_can_name_the_tree_it_is_reading(committed_tree, monkeypatch):
    repo, revision = committed_tree
    monkeypatch.setattr(runner, "ROOT", repo)
    assert runner.working_tree_revision() == revision


def test_an_uncommitted_edit_counts_as_a_different_tree(committed_tree, monkeypatch):
    repo, revision = committed_tree
    monkeypatch.setattr(runner, "ROOT", repo)
    (repo / "state.py").write_text("value = 2\n")
    assert runner.working_tree_revision() == f"{revision}-dirty"


def test_the_probe_reads_the_repository_not_the_cwd(committed_tree, tmp_path, monkeypatch):
    repo, revision = committed_tree
    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.chdir(tmp_path)
    assert runner.working_tree_revision() == revision


@pytest.mark.parametrize("error", [OSError("git missing"), subprocess.TimeoutExpired("git", 1), ValueError("bad input")])
def test_the_revision_probe_never_raises(monkeypatch, error):
    def cannot_read(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner.subprocess, "run", cannot_read)
    assert runner.working_tree_revision() == ""


@pytest.fixture
def exercise(monkeypatch, tmp_path):
    def run(*, chunks=1, failed=False, after_chunk=None, after_retry=None):
        revision = ["revision-a"]
        files = [Path(f"test_{index}.py") for index in range(chunks)]
        monkeypatch.setattr(runner, "discover_test_files", lambda _: files)
        monkeypatch.setattr(runner, "split_chunks", lambda *a: [[file] for file in files])
        monkeypatch.setattr(runner, "working_tree_revision", lambda: revision[0])
        monkeypatch.setattr(runner, "free_memory_gb", lambda: 8.)
        retries = []

        def chunk(index, *args, **kwargs):
            if after_chunk:
                revision[0] = after_chunk[index-1]
            return not failed, f"chunk {index}", [NODE] if failed else []

        def retry(*args, **kwargs):
            retries.append(True)
            if after_retry is not None:
                revision[0] = after_retry
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(runner, "run_chunk", chunk)
        monkeypatch.setattr(runner.subprocess, "run", retry)
        register = tmp_path / "register.json"
        progress = tmp_path / "progress.log"
        code = runner.main([
            "--continue-on-failure", "--defect-register", str(register),
            "--progress-file", str(progress),
        ])
        return code, json.loads(register.read_text()), progress.read_text(), retries

    return run


@pytest.mark.parametrize("after_chunk", [["revision-b", "revision-b"], ["revision-b", "revision-a"]])
def test_a_moving_tree_keeps_the_run_red(exercise, after_chunk, capsys):
    code, register, progress, _ = exercise(chunks=2, after_chunk=after_chunk)
    assert code == 1
    source = register["source_revision"]
    assert source["started"] == "revision-a" and source["changed"]
    assert source["finished"] == after_chunk[-1]
    assert "MOVED" in progress and "THE TREE MOVED" in capsys.readouterr().out


def test_a_move_during_the_last_chunk_is_observed(exercise):
    code, register, _, _ = exercise(after_chunk=["revision-b"])
    assert code == 1
    assert register["source_revision"]["changed"]


def test_a_retry_cannot_call_a_changed_source_an_order_defect(exercise):
    code, register, _, retries = exercise(failed=True, after_chunk=["revision-b"])
    assert code == 1 and not retries
    assert register["order_dependent"] == register["real_failures"] == []
    assert "source" in register["retry_errors"][NODE]


@pytest.mark.parametrize("after_retry", ["revision-b", ""])
def test_a_source_change_during_retry_is_unmeasured(exercise, after_retry):
    code, register, _, retries = exercise(failed=True, after_retry=after_retry)
    assert code == 1 and len(retries) == 1
    assert register["order_dependent"] == register["real_failures"] == []
    assert "source" in register["retry_errors"][NODE]


def test_a_run_on_one_tree_says_that_too(exercise, capsys):
    code, register, progress, _ = exercise()
    assert code == 0
    source = register["source_revision"]
    assert source["started"] == source["finished"] == "revision-a"
    assert not source["changed"]
    assert "RUN begins revision=revision-a" in progress
    assert source["measured"]
    assert "unchanged at every source check" in capsys.readouterr().out


def test_an_unknown_starting_revision_starts_no_test_work(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "discover_test_files", lambda _: [Path("test_unused.py")])
    monkeypatch.setattr(runner, "working_tree_revision", lambda: "")

    def forbidden(*args, **kwargs):
        raise AssertionError("unknown source started test work")

    monkeypatch.setattr(runner, "run_chunk", forbidden)
    register = tmp_path / "register.json"
    assert runner.main([
        "--defect-register", str(register), "--progress-file", str(tmp_path / "progress.log"),
    ]) == 1
    source = json.loads(register.read_text())["source_revision"]
    assert not source["measured"] and "start" in source["errors"]


def test_the_runner_still_starts():
    done = subprocess.run(
        [sys.executable, str(Path(runner.__file__)), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert done.returncode == 0, done.stderr[-1500:]
