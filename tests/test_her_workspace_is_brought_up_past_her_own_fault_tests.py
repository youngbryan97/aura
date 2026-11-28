"""A workspace with a commit of hers on it is brought up to her source where the only change in it is a test her loop wrote of
a fault (F15).

LIVE 27 November 2026, 22:54: "My workspace could not be brought up to my source (39e5952e1): error: cannot rebase: You have
unstaged changes." The change was her fault's test, written again at a hearing with its count changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as goals

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "tests" / "her_faults").mkdir(parents=True)
    (source / "app.py").write_text("def f():\n    return 1\n")
    (source / "tests" / "her_faults" / "test_fault_f.py").write_text('"""Heard 2 time(s)."""\n')
    _git(source, "init", "-q", "-b", "main")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "base")
    base = _git(source, "rev-parse", "HEAD")
    work = tmp_path / "her_goals" / "mend-f"
    _git(source, "worktree", "add", "-q", "-b", "her/mend-f", str(work), base)
    (work / "app.py").write_text("def f():\n    return 2\n")
    _git(work, "commit", "-qam", "her draft")
    (source / "notes.txt").write_text("moved on\n")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "her source moved")
    monkeypatch.setattr(goals, "source_head", lambda: _git(source, "rev-parse", "HEAD"))
    monkeypatch.setattr(goals, "workspaces_dir", lambda: tmp_path / "her_goals")
    record = goals.GoalRecord(goal="mend f", slug="mend-f", base=base)
    return record, work, source


def test_a_fault_test_her_loop_wrote_again_does_not_keep_her_workspace_back(workspace):
    record, work, source = workspace
    (work / "tests" / "her_faults" / "test_fault_f.py").write_text('"""Heard 4 time(s)."""\n')
    goals._bring_it_up_to_the_source(record)
    assert record.base == _git(source, "rev-parse", "HEAD"), record.said
    assert (work / "app.py").read_text().endswith("return 2\n"), "her draft is carried onto the new source"
    assert (work / "notes.txt").exists()


def test_an_edit_of_hers_beside_it_keeps_the_workspace_where_it_is_and_she_is_told(workspace):
    record, work, source = workspace
    (work / "tests" / "her_faults" / "test_fault_f.py").write_text('"""Heard 4 time(s)."""\n')
    (work / "app.py").write_text("def f():\n    return 3\n")
    old = record.base
    goals._bring_it_up_to_the_source(record)
    assert record.base == old and "could not be brought up" in record.said[-1]["words"]
    assert (work / "app.py").read_text().endswith("return 3\n"), "an edit of hers is never put back"
