"""A change on probation is settled only by evidence about that change, and missing evidence settles nothing.

Found 19 November 2026 in review. The outside benchmarks' verdicts were taken by time: any comparison recorded after a
change landed counted, though a run started before it measured the source without it, and a comparison of two runs
that both hold it says nothing about it. And her real calls answered "nothing to say" alike when no call of hers had
reached the change, when her calls could not be read, and when the check could not reach a verdict; a HELD from the
benchmarks then kept the change. Now a verdict counts only from a candidate run that holds the change against a
baseline that does not, and a try that could not be read or judged keeps the change on probation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as work
from core.agency.the_outside_bar import verdicts_on
from core.self_modification import her_real_calls as real
from tests.test_her_goal_is_worked_by_her import her_source  # noqa: F401 - a fixture, found by its own name
from tests.test_her_real_calls_are_tried_on_her_changes import _on_probation


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history(tmp_path):
    """Her source with three commits: before the change, the change, and one after it."""
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    made = {}
    for name in ("before", "change", "after"):
        (root / f"{name}.txt").write_text(name)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", name)
        made[name] = _git(root, "rev-parse", "HEAD")
    return root, made


def _ledger(home: Path, runs: dict[str, str], comparisons: list[tuple[str, str, str]]) -> Path:
    for run, commit in runs.items():
        (home / "runs" / run).mkdir(parents=True, exist_ok=True)
        (home / "runs" / run / "meta.json").write_text(json.dumps({"commit": commit}))
    lines, before = [], ""
    for baseline, candidate, verdict in comparisons:
        line = json.dumps({"kind": "compare", "baseline": baseline, "candidate": candidate, "verdict": verdict,
                           "at": "2026-11-19T12:00:00-0800", "previous": before}, sort_keys=True)
        before = hashlib.sha256(line.encode("utf-8")).hexdigest()
        lines.append(line)
    (home / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    return home / "ledger.jsonl"


def test_a_verdict_counts_only_from_a_run_that_holds_the_change_against_one_that_does_not(history, tmp_path):
    root, made = history
    runs = {"base": made["before"], "with": made["after"], "with-too": made["change"], "without": made["before"]}
    ledger = _ledger(tmp_path / "bar", runs, [
        ("base", "with", "HELD"),            # measured it: counts
        ("base", "without", "REGRESSED"),    # started before it landed: says nothing about it
        ("with-too", "with", "REGRESSED"),   # both hold it: says nothing about it
        ("base", "with", "INCOMPLETE"),      # not decided
        ("base", "unknown-run", "HELD"),     # no record of what it measured
    ])
    assert verdicts_on(made["change"], root, ledger) == ["HELD"]


def test_a_verdict_on_another_commit_is_not_a_verdict_on_this_one(history, tmp_path):
    root, made = history
    ledger = _ledger(tmp_path / "bar", {"base": made["before"], "with": made["change"]}, [("base", "with", "HELD")])
    assert verdicts_on(made["after"], root, ledger) == [], "the run measured source without the later commit"
    assert verdicts_on("", root, ledger) == []


def test_her_calls_that_cannot_be_read_keep_the_change_on_probation(her_source, monkeypatch):  # noqa: F811
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    (her_source / "kept_calls.json").write_text("{not json")
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    assert work.settle_probation() == [] and work.change_on_probation() is not None and not tried


def test_a_try_the_check_could_not_judge_is_tried_again_and_never_settles_it(her_source, monkeypatch):  # noqa: F811
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "", "could_not_be_run": "no clone"})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    assert work.settle_probation() == [] and work.settle_probation() == []
    assert tried == [1, 1], "the same sample is tried again while it has no verdict"
    assert work.load(record.slug).offer["real_calls_verdict"]["disposition"] == "inconclusive"


def test_with_no_call_of_hers_reaching_the_change_the_benchmarks_settle_it_and_say_so(her_source, monkeypatch):  # noqa: F811
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    monkeypatch.setattr(real, "functions_a_commit_changed", lambda _repo, _commit: [("core/elsewhere.py", "g")])
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    assert work.settle_probation()[0]["status"] == "kept after probation" and not tried
    assert work.load(record.slug).offer["settled_on"]["real_calls"] == "not_observed"


def test_a_pass_on_her_calls_and_a_held_verdict_settle_it_and_say_on_what(her_source, monkeypatch):  # noqa: F811
    record, tried, taken = _on_probation(her_source, monkeypatch, {"verdict": "HELD", "calls": 1})
    monkeypatch.setattr(work, "verdicts_on", lambda _commit, _root: ["HELD"])
    assert work.settle_probation()[0]["status"] == "kept after probation"
    settled = work.load(record.slug).offer["settled_on"]
    assert settled["benchmarks"] == "HELD" and settled["real_calls"] == "passed"
