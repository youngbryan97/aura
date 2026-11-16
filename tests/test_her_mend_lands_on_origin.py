"""A mend her own yes takes lands on origin's main first, then in the source she runs on, and can be taken back.

Master Todo F15, 16 November 2026: her own yes merged a mend into her checkout on disk and nowhere else, so the next
deploy, which fast-forwards that checkout to origin's main, would find the two parted. A bare repository stands in
for origin here, with a second clone for whoever else pushes to it; her workspace, her fault's test, repair by
selection, the rebase, the push and the revert all run for real. Only her gates are stood in for.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap

import pytest

from core.agency import her_goal_at_work as goals
from core.agency.mending_a_fault_of_hers import mend
from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import the_call_that_failed
from core.self_modification import faults_she_can_reproduce
from core.self_modification.faults_she_can_reproduce import HeardFault

pytestmark = pytest.mark.unit

_ROWS = textwrap.dedent('''
    def first(rows):
        if len(rows) > 0:
            raise ValueError("there are no rows")
        return rows[0]


    def count(rows):
        return len(rows)
''')
_GUARD = "from planted_pkg.rows import count\n\n\ndef test_count():\n    assert count([1, 2]) == 2\n"


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "core.fsmonitor=false", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def with_origin(tmp_path, monkeypatch):
    origin, source, other = tmp_path / "origin.git", tmp_path / "source", tmp_path / "other"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    (source / "planted_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "planted_pkg" / "__init__.py").write_text("")
    (source / "planted_pkg" / "rows.py").write_text(_ROWS)
    (source / "tests" / "test_rows.py").write_text(_GUARD)
    _git("init", "-q", "-b", "main", cwd=source)
    _git("add", "-A", cwd=source)
    _git("commit", "-q", "-m", "planted", cwd=source)
    _git("remote", "add", "origin", str(origin), cwd=source)
    _git("push", "-q", "origin", "main", cwd=source)
    _git("clone", "-q", str(origin), str(other), cwd=tmp_path)
    monkeypatch.setattr(goals, "source_root", lambda: source)
    monkeypatch.setattr(goals, "workspaces_dir", lambda: tmp_path / "workspaces")
    monkeypatch.setattr(goals, "records_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(goals, "MEASURING_HOLD", tmp_path / "no-hold")
    monkeypatch.setattr(goals, "the_gates_her_yes_asks", lambda record: {"stand-in": {"passed": True, "tail": ""}})
    monkeypatch.setattr(a_call_that_failed, "_ROOT", source.resolve())
    monkeypatch.setattr(faults_she_can_reproduce, "where_heard_faults_are_kept", lambda: tmp_path / "heard.json")
    monkeypatch.syspath_prepend(str(source))
    for name in ("planted_pkg", "planted_pkg.rows"):
        sys.modules.pop(name, None)
    faults_she_can_reproduce.forget_for_test()
    yield origin, source, other
    from core.self_modification.into_the_running_process import undo_every_load_for_test

    undo_every_load_for_test()
    faults_she_can_reproduce.forget_for_test()


def _origin_main(origin):
    return _git("rev-parse", "main", cwd=origin)


def _someone_else_pushes(other, path, text):
    (other / path).write_text(text)
    _git("add", "-A", cwd=other)
    _git("commit", "-q", "-m", f"someone else changed {path}", cwd=other)
    _git("push", "-q", "origin", "main", cwd=other)


def _the_fault():
    import planted_pkg.rows as rows

    try:
        rows.first([1, 2])
    except ValueError as exc:
        return HeardFault(the_call_that_failed(exc), "planted", 0.0, 0.0)
    raise AssertionError("the planted fault did not raise")


def test_with_origin_unmoved_her_mend_lands_there_and_on_her_disk(with_origin):
    origin, source, _other = with_origin
    done = asyncio.run(mend(_the_fault(), proposals=80, seconds=900))
    assert done["status"] == "merged", done
    assert _origin_main(origin) == _git("rev-parse", "HEAD", cwd=source) == done["taken"]["commit"]
    assert done["taken"]["on_origin"] and done["taken"]["on_disk"]


def test_with_origin_moved_her_mend_is_replayed_onto_it_and_her_disk_waits(with_origin):
    origin, source, other = with_origin
    _someone_else_pushes(other, "NOTES.md", "a note\n")
    before = _git("rev-parse", "HEAD", cwd=source)
    done = asyncio.run(mend(_the_fault(), proposals=80, seconds=900))
    assert done["status"] == "merged", done
    landed = _origin_main(origin)
    assert _git("log", "--format=%s", "-2", landed, cwd=source).splitlines() == [
        _git("log", "--format=%s", "-1", done["commit"], cwd=source), "someone else changed NOTES.md"]
    assert "if not (len(rows) > 0):" in _git("show", f"{landed}:planted_pkg/rows.py", cwd=source)
    assert _git("rev-parse", "HEAD", cwd=source) == before, "her disk does not take someone else's commit"
    assert done["taken"]["on_origin"] and not done["taken"]["on_disk"]


def test_a_mend_that_no_longer_applies_on_origin_lands_nowhere(with_origin):
    origin, source, other = with_origin
    _someone_else_pushes(other, "planted_pkg/rows.py", _ROWS.replace("if len(rows) > 0:", "if len(rows) >= 1:"))
    landed_before, before = _origin_main(origin), _git("rev-parse", "HEAD", cwd=source)
    done = asyncio.run(mend(_the_fault(), proposals=80, seconds=900))
    assert done["status"] == goals.OFFERED and "does not apply on origin" in done["taken"]["why"], done
    assert _origin_main(origin) == landed_before and _git("rev-parse", "HEAD", cwd=source) == before


def test_a_mend_taken_back_is_reverted_on_origin_and_on_her_disk(with_origin):
    origin, source, _other = with_origin
    done = asyncio.run(mend(_the_fault(), proposals=80, seconds=900))
    undone = goals.take_it_back(done["slug"], "it raised TypeError after it was loaded")
    assert undone["ok"], undone
    assert _origin_main(origin) == _git("rev-parse", "HEAD", cwd=source) == undone["revert"]
    assert "if len(rows) > 0:" in (source / "planted_pkg" / "rows.py").read_text()
    assert goals.load(done["slug"]).status == goals.STALLED


def test_the_whole_loop_heard_mended_landed_loaded_undone_and_taken_back(with_origin):
    """A fault raised in this process is mended, lands on origin and on disk, and is loaded into this process: the
    module imported before the mend runs the mended body. When the loaded body then faults in a way it does not
    mean, it is undone here, and her next cycle takes it back out of origin and her disk."""
    import planted_pkg.rows as rows

    from core.agency.mending_a_fault_of_hers import mend_the_faults_she_heard
    from core.runtime.errors import record_degradation

    fault = _the_fault()
    done = asyncio.run(mend(fault, proposals=80, seconds=900))
    assert done["status"] == "merged" and done["loaded"]["functions"] == ["first"], done
    assert rows.first([1, 2]) == 1, "the process that raised the fault runs the mend"
    with pytest.raises(ValueError, match="no rows"):
        rows.first([])

    try:
        rows.first(5)                    # 'int' object is not subscriptable: the mended body faulting
    except TypeError as exc:
        faults_she_can_reproduce.hear(record_degradation("planted", exc, severity="warning", action="planted",
                                                         enforce_failure_policy=False))
    with pytest.raises(ValueError):
        rows.first([1, 2])               # undone in this process

    (outcome,) = asyncio.run(mend_the_faults_she_heard(seconds=900))
    origin, source, _other = with_origin
    assert outcome["status"] == "taken back", outcome
    assert _origin_main(origin) == _git("rev-parse", "HEAD", cwd=source)
    assert "if len(rows) > 0:" in (source / "planted_pkg" / "rows.py").read_text()
