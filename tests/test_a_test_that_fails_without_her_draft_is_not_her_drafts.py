"""A test that fails on her source without her draft too is not her draft's to answer for (Master Todo F16, 26 November
2026). Her own yes runs the tests that name what a draft changed and stopped at any failure, so a red already on her
source held back every draft of hers that touched it."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as goals

pytestmark = pytest.mark.unit

_TESTS = ("from app import double\n\n\ndef test_double():\n    assert double(2) == 4\n\n\n"
          "def test_already_red():\n    assert False, 'red on her source before any draft'\n")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def draft(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "app.py").write_text("def double(x):\n    return x * 2\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text(_TESTS)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "her source")
    head = _git(repo, "rev-parse", "HEAD")
    record = goals.GoalRecord(goal="g", slug="g")
    monkeypatch.setattr(goals.GoalRecord, "workspace", property(lambda self: repo))
    monkeypatch.setattr(goals, "source_head", lambda: head)

    def drafted(body: str) -> goals.GoalRecord:
        (repo / "app.py").write_text(f"def double(x):\n    return {body}\n")
        _git(repo, "commit", "-qam", "her draft")
        record.offer = {"commit": _git(repo, "rev-parse", "HEAD")}
        return record
    return drafted


def test_a_red_already_on_her_source_is_not_counted_against_her_draft(draft):
    record = draft("x + x")
    said = "FAILED tests/test_app.py::test_already_red - AssertionError: red on her source\n1 failed, 1 passed"
    assert goals._failing_without_the_draft(record, said, {}) == ["tests/test_app.py::test_already_red"]


def test_a_test_her_draft_broke_is_hers(draft):
    record = draft("x * 3")
    said = ("FAILED tests/test_app.py::test_already_red - AssertionError\nFAILED tests/test_app.py::test_double - "
            "assert 6 == 4\n2 failed")
    assert goals._failing_without_the_draft(record, said, {}) == [], "test_double passes without the draft"
