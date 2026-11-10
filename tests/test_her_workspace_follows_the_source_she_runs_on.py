"""Her workspace is brought up to the source she runs on before each pass, keeping her work.

A workspace was made once, and her own yes takes a draft only by fast-forward, so a draft on a base her source had
moved past could never be taken: her workspaces of 9 November 2026 sat under a source that moved forty commits on
10 November.
"""
from __future__ import annotations

from core.agency import her_goal_at_work as work
from tests.test_her_goal_is_worked_by_her import GOAL, _git, her_source  # noqa: F401

_COMMIT = ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am"]


def _a_workspace(her_source):
    record = work.GoalRecord(goal=GOAL, slug=work.slug_of(GOAL))
    work.ensure_workspace(record)
    return record


def _the_source_moves(her_source, text="def note_count():\n    return 1\n"):
    (her_source / "core" / "notes.py").write_text(text)
    _git(*_COMMIT, "the source moves", cwd=her_source)
    return _git("rev-parse", "HEAD", cwd=her_source)


def test_a_clean_workspace_is_brought_up_and_its_old_check_no_longer_counts(her_source):
    record = _a_workspace(her_source)
    record.last_check = {"passed": True}
    head = _the_source_moves(her_source)
    work.ensure_workspace(record)
    assert _git("rev-parse", "HEAD", cwd=record.workspace) == head and record.base == head
    assert record.last_check == {}


def test_her_uncommitted_edit_is_carried_up(her_source):
    record = _a_workspace(her_source)
    (record.workspace / "core" / "more.py").write_text("MORE = []\n")
    head = _the_source_moves(her_source)
    work.ensure_workspace(record)
    assert record.base == head and (record.workspace / "core" / "more.py").read_text() == "MORE = []\n"
    assert "return 1" in (record.workspace / "core" / "notes.py").read_text()


def test_a_declined_draft_on_her_branch_is_replayed_not_dropped(her_source):
    record = _a_workspace(her_source)
    (record.workspace / "core" / "kept.py").write_text("KEPT = 1\n")
    _git("add", "core/kept.py", cwd=record.workspace)
    _git(*_COMMIT, "her declined draft", cwd=record.workspace)
    head = _the_source_moves(her_source)
    work.ensure_workspace(record)
    assert record.base == head
    assert (record.workspace / "core" / "kept.py").read_text() == "KEPT = 1\n"
    assert _git("log", "-1", "--format=%s", cwd=record.workspace) == "her declined draft"


def test_an_edit_that_cannot_be_carried_leaves_it_where_it_was_and_says_why(her_source):
    record = _a_workspace(her_source)
    base = record.base
    (record.workspace / "core" / "notes.py").write_text("def note_count():\n    return 2\n")
    _the_source_moves(her_source)
    work.ensure_workspace(record)
    assert record.base == base and "return 2" in (record.workspace / "core" / "notes.py").read_text()
    assert "could not be brought up" in record.said[-1]["words"]
