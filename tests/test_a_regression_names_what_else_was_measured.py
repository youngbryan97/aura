"""A change on probation is settled with what else the deciding comparison measured beside it (F16-08).

A comparison measures her source as it stands when it runs, so a change of hers on probation is measured beside
whatever landed since the baseline. LIVE 19 November 2026 a commit of mine took MATH-500 from 9 of 12 to 2 between two
baselines. A change of hers on probation then would have been taken back "because the outside benchmarks fell", with
nothing said of the other changes measured with it. The other changes to what she runs now go with every settlement,
and a fall measured beside them is said to be unattributed.
"""
from __future__ import annotations

import pytest

from core.agency import her_goal_at_work as work
from core.agency.the_outside_bar import other_changes_measured_with
from tests.test_her_goal_is_worked_by_her import her_source  # noqa: F401 - a fixture, found by its own name
from tests.test_her_real_calls_are_tried_on_her_changes import _on_probation
from tests.test_probation_is_settled_on_evidence_about_the_change import _git, _ledger

pytestmark = pytest.mark.unit


@pytest.fixture
def landed_after(tmp_path):
    """Her source: before the change, the change, then a change to her code, a note in her docs and a test."""
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "tests").mkdir()
    _git(root, "init", "-q", "-b", "main")
    made = {}
    for name, path in (("before", "core/a.py"), ("change", "core/b.py"), ("other code", "core/c.py"),
                       ("a note", "docs/note.md"), ("a test", "tests/test_c.py")):
        (root / path).write_text(f"# {name}\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", name)
        made[name] = _git(root, "rev-parse", "HEAD")
    return root, made


def test_the_other_changes_to_what_she_runs_are_named(landed_after, tmp_path):
    root, made = landed_after
    ledger = _ledger(tmp_path / "bar", {"base": made["before"], "with": made["a test"]}, [("base", "with", "REGRESSED")])
    assert other_changes_measured_with(made["change"], root, ledger) == [made["other code"]]


def test_a_comparison_of_the_change_alone_names_nothing_else(landed_after, tmp_path):
    root, made = landed_after
    ledger = _ledger(tmp_path / "bar", {"base": made["before"], "with": made["change"]}, [("base", "with", "REGRESSED")])
    assert other_changes_measured_with(made["change"], root, ledger) == []


def test_a_fall_measured_beside_other_changes_is_taken_back_as_unattributed(her_source, monkeypatch):  # noqa: F811
    record, _tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["REGRESSED"])
    monkeypatch.setattr(work, "other_changes_measured_with", lambda _commit, _root: ["9ae22ef33aaaa", "198b29221bbbb"])
    assert work.settle_probation()[0]["status"] == "taken back"
    why = taken[0][1]
    assert "beside 2 other change(s)" in why and "9ae22ef33" in why and "not known" in why


def test_a_settlement_keeps_what_else_was_measured(her_source, monkeypatch):  # noqa: F811
    record, _tried, _taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    monkeypatch.setattr(work, "other_changes_measured_with", lambda _commit, _root: ["9ae22ef33aaaa"])
    assert work.settle_probation()[0]["status"] == "kept after probation"
    assert work.load(record.slug).offer["settled_on"]["measured_beside"] == ["9ae22ef33"]
