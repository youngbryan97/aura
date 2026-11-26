"""Every fault she heard gets its turn, is kept with a call wherever one can be made, and counts as made again only
where it was made (Master Todo F15, 26 November 2026).

Read from her running on 25 and 26 November: her loop tried one fault a cycle, the most frequent first, and the most
frequent was one raised in what governs her, which is never hers to change. It came first every cycle, was said
stalled 85 times, and the eight faults behind it were never tried again. Of the rest she heard, 4 could not be made
again only because an argument was a path, and 6 because the error was recorded without being raised.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path, PosixPath

import pytest

from core.agency import her_goal_at_work as goals
from core.agency import mending_a_fault_of_hers as loop
from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import FailedCall, _written_back
from core.self_modification import faults_she_can_reproduce as heard
from core.self_modification.faults_she_can_reproduce import made_again

pytestmark = pytest.mark.unit


def _fault(file: str, qualname: str = "f") -> heard.HeardFault:
    call = FailedCall(module=file[:-3].replace("/", "."), qualname=qualname, file=file, line=3,
                      error_type="builtins.ValueError", error_message="no")
    return heard.HeardFault(call, "planted", 0.0, 0.0)


def test_a_fault_in_what_governs_her_does_not_stand_in_front_of_the_others(monkeypatch):
    governing, hers = _fault("core/governance_context.py", "require_governance"), _fault("core/notes.py", "count")
    stalled = goals.GoalRecord(goal="g", slug="g", status=goals.STALLED, stalled_on="old")

    async def nothing(*_args, **_kwargs):
        return None
    tried: list[str] = []

    async def mend(found, **_kwargs):
        tried.append(found.call.file)
        return {"slug": found.call.file}
    monkeypatch.setattr(heard, "keep_what_was_heard", nothing)
    monkeypatch.setattr(heard, "mends_undone", lambda: [])
    monkeypatch.setattr(heard, "faults_heard", lambda: [governing, hers])
    monkeypatch.setattr(goals, "mends_undone_on_record", lambda: [])
    monkeypatch.setattr(goals, "settle_probation", lambda: [])
    monkeypatch.setattr(goals, "source_head", lambda: "new")
    monkeypatch.setattr(goals, "load", lambda slug: stalled if "governance" in slug else None)
    monkeypatch.setattr(loop, "_what_it_stalled_on_has_moved", lambda record, file, head: True)
    for name in ("_keep_her_real_calls_of_what_she_may_change", "_keep_her_recent_turns",
                 "_start_an_outside_comparison_when_due"):
        monkeypatch.setattr(loop, name, nothing)
    monkeypatch.setattr(loop, "mend", mend)
    assert asyncio.run(loop.mend_the_faults_she_heard(seconds=1)) == [{"slug": "core/notes.py"}]
    assert tried == ["core/notes.py"]


def test_a_path_is_written_back_as_the_path_it_is():
    for value in (PosixPath("/tmp/her state/notes.db"), Path("relative/x.json")):
        source = _written_back(value)
        assert source is not None and eval(source) == value and type(eval(source)) is type(value)  # noqa: S307


def test_a_fault_recorded_without_being_raised_is_kept_as_the_call_that_recorded_it(monkeypatch, tmp_path):
    package = tmp_path / "recording_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "planted.py").write_text(
        "from core.runtime.errors import record_degradation\n\n\n"
        "def check(limit, path):\n"
        "    if limit < 0:\n"
        "        return record_degradation('planted', RuntimeError('limit below zero'), severity='warning',\n"
        "                                  enforce_failure_policy=False)\n"
        "    return None\n")
    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    planted = importlib.import_module("recording_pkg.planted")
    record = planted.check(-1, PosixPath("/tmp/x"))
    call = record.failed_call
    assert isinstance(call, FailedCall), call
    assert (call.qualname, call.error_type) == ("check", "builtins.RuntimeError")
    assert call.call_source("check") == "check(limit=-1, path=__import__('pathlib').PosixPath('/tmp/x'))"
    assert record.raised_in == ("recording_pkg.planted", "check")


def test_a_fault_is_made_again_only_where_she_made_it():
    call = _fault("core/capabilities/browser_controller.py", "BrowserController.extract_article_text").call
    elsewhere = "core/runtime/network_gateway.py:90: in request\n    raise ValueError('no network here')\nE   ValueError"
    there = ("core/capabilities/browser_controller.py:947: in extract_article_text\n    raise ValueError(...)\n"
             "E   ValueError: no")
    assert not made_again(call, elsewhere), "the same kind of error, raised for another reason somewhere else"
    assert made_again(call, there)
    assert made_again(call, "AssertionError: it recorded ValueError again")
    assert not made_again(call, "E   KeyError: 'x'")


def test_the_written_test_checks_the_place_a_recorded_fault_came_from():
    source = heard.test_source(_fault("core/notes.py", "count"))
    assert "where in (None, ('core.notes', 'count'))" in source
    namespace: dict = {}
    exec(compile(source.replace("from core.notes import count\n", ""), "kept", "exec"), namespace)  # noqa: S102
    assert isinstance(namespace["_recorded_since"], types.FunctionType)


def test_a_draft_a_judge_refused_is_asked_again_once_the_judge_has_changed(monkeypatch, tmp_path):
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    for name, text in (("outside_bar/verify_change.py", "A = 1\n"), ("core/notes.py", "B = 1\n")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    git("add", "-A")
    git("commit", "-q", "-m", "asked on this")
    asked_on = git("rev-parse", "HEAD")
    (tmp_path / "core/notes.py").write_text("B = 2\n")
    git("commit", "-qam", "another change")
    elsewhere = git("rev-parse", "HEAD")
    (tmp_path / "outside_bar/verify_change.py").write_text("A = 2\n")
    git("commit", "-qam", "the judge mended")
    mended = git("rev-parse", "HEAD")
    monkeypatch.setattr(goals, "source_root", lambda: tmp_path)
    refused = {"outside-check": {"passed": False, "finished": True, "verdict": "REFUSED"}}
    record = goals.GoalRecord(goal="g", slug="g", status=goals.STALLED,
                              offer={"commit": "abc", "asked_on": asked_on, "gates": refused})
    assert loop.refused_by_a_judge_since_changed(record, mended)
    assert not loop.refused_by_a_judge_since_changed(record, elsewhere), "nothing that judged it changed"
    stopped = {"outside-check": {"passed": False, "finished": False}}
    assert not loop.refused_by_a_judge_since_changed(
        goals.GoalRecord(goal="g", slug="g", status=goals.STALLED, offer={"commit": "abc", "asked_on": asked_on,
                                                                          "gates": stopped}), mended), \
        "a gate stopped before it finished judged nothing, and is asked again another way"
    tests_failed = {"tests": {"passed": False, "finished": True}}
    assert not loop.refused_by_a_judge_since_changed(
        goals.GoalRecord(goal="g", slug="g", status=goals.STALLED, offer={"commit": "abc", "asked_on": asked_on,
                                                                          "gates": tests_failed}), mended)


def test_the_call_that_recorded_a_fault_is_never_looked_for_past_her_source(monkeypatch, tmp_path):
    """From a frame of hers recorded through something not hers (an event loop, a thread's start), the walk stops: past
    it lies the machinery that ran her, and a test making that call would start her whole runtime."""
    import sys as interpreter

    from core.runtime.a_call_that_failed import the_call_that_recorded

    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    said = the_call_that_recorded(RuntimeError("x"), interpreter._getframe())
    assert said == "it was recorded without being raised, from outside her source"


def test_a_draft_asked_again_is_carried_onto_her_source_as_it_is_now(monkeypatch, tmp_path):
    """LIVE 26 November 2026: her young collection was asked again on 429b8199c, five days behind her source, and her
    yes said "the source she runs on moved since her draft"."""
    import subprocess

    source, workspace = tmp_path / "source", tmp_path / "workspace"
    source.mkdir()

    def git(where, *args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=where, check=True,
                              capture_output=True, text=True).stdout.strip()
    git(source, "init", "-q", "-b", "main")
    (source / "pauses.py").write_text("FULL = True\n")
    (source / "other.py").write_text("A = 1\n")
    git(source, "add", "-A")
    git(source, "commit", "-q", "-m", "her source")
    base = git(source, "rev-parse", "HEAD")
    git(source, "worktree", "add", "-q", "-b", "her-goal", str(workspace), base)
    (workspace / "pauses.py").write_text("FULL = False\n")
    git(workspace, "commit", "-qam", "her draft")
    (source / "other.py").write_text("A = 2\n")
    git(source, "commit", "-qam", "her source moves on")
    head = git(source, "rev-parse", "HEAD")
    record = goals.GoalRecord(goal="g", slug="g", status=goals.OFFERED, base=base,
                              offer={"commit": git(workspace, "rev-parse", "HEAD")})
    monkeypatch.setattr(goals.GoalRecord, "workspace", property(lambda self: workspace))
    monkeypatch.setattr(goals, "source_head", lambda: head)
    monkeypatch.setattr(goals, "source_root", lambda: source)
    replayed = loop._brought_up(record, head)
    assert replayed and replayed != record.offer["commit"] and record.status == goals.OFFERED
    assert git(workspace, "merge-base", "--is-ancestor", head, replayed) == ""
    assert (workspace / "pauses.py").read_text() == "FULL = False\n" and (workspace / "other.py").read_text() == "A = 2\n"
