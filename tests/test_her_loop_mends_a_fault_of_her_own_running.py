"""A fault raised while she ran is made again, mended by her own Python, and taken into her source by her own yes.

Master Todo F15, 14 November 2026. A git repository of its own stands in for her source: a planted fault her fix
templates can reach, and a test that passed before it and must hold after. Her goal workspace, the fault's test, repair
by selection, the commit, the offer and her own yes all run for real; only her gates are stood in for, since they run
the repository's make targets, which a stand-in tree has none of.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap

import pytest

from core.agency import her_goal_at_work as goals
from core.agency.mending_a_fault_of_hers import mend, the_lines_it_ran
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
_GUARD = textwrap.dedent('''
    from planted_pkg.rows import count


    def test_count():
        assert count([1, 2]) == 2
''')
_UNGUARDED = "def half(n):\n    return n / 0\n"
_PEOPLE = "def label(person):\n    return person.name.title()\n"
_PEOPLE_GUARD = ("from planted_pkg.people import label\n\n\nclass _Person:\n    name = 'ada'\n\n\n"
                 "def test_label():\n    assert label(_Person()) == 'Ada'\n")


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def her_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "planted_pkg").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "planted_pkg" / "__init__.py").write_text("")
    (source / "planted_pkg" / "rows.py").write_text(_ROWS)
    (source / "planted_pkg" / "alone.py").write_text(_UNGUARDED)
    (source / "planted_pkg" / "people.py").write_text(_PEOPLE)
    (source / "tests" / "test_rows.py").write_text(_GUARD)
    (source / "tests" / "test_people.py").write_text(_PEOPLE_GUARD)
    _git("init", "-q", "-b", "main", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "planted", cwd=source)
    monkeypatch.setattr(goals, "source_root", lambda: source)
    monkeypatch.setattr(goals, "workspaces_dir", lambda: tmp_path / "workspaces")
    monkeypatch.setattr(goals, "records_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(goals, "MEASURING_HOLD", tmp_path / "no-hold")
    monkeypatch.setattr(goals, "the_gates_her_yes_asks", lambda record: {"stand-in": {"passed": True, "tail": ""}})
    monkeypatch.setattr(a_call_that_failed, "_ROOT", source.resolve())
    monkeypatch.setattr(faults_she_can_reproduce, "where_heard_faults_are_kept", lambda: tmp_path / "heard.json")
    monkeypatch.syspath_prepend(str(source))
    for name in ("planted_pkg", "planted_pkg.rows", "planted_pkg.alone", "planted_pkg.people"):
        sys.modules.pop(name, None)
    yield source
    from core.self_modification.into_the_running_process import undo_every_load_for_test

    undo_every_load_for_test()


def _fault(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the planted fault is what is caught
        kept = the_call_that_failed(exc)
        assert not isinstance(kept, str), kept
        return HeardFault(kept, "planted", 0.0, 0.0)
    raise AssertionError("the planted call did not raise")


def test_the_lines_ranked_are_the_raise_first_then_its_function():
    ranked = the_lines_it_ran(_ROWS, "planted_pkg/rows.py", 4)
    assert ranked[0] == (("planted_pkg/rows.py", 4), 1.0)
    assert {line for (_path, line), _score in ranked} == {3, 4, 5}


def test_a_fault_of_her_running_is_mended_and_taken_into_her_source(her_source):
    import planted_pkg.rows as rows

    done = asyncio.run(mend(_fault(lambda: rows.first([1, 2])), proposals=80, seconds=900))
    assert done["status"] == "merged", done
    assert done["origin"].startswith(("template:", "looks_wrong:"))
    assert done["loaded"] == {"functions": ["first"], "waits": []}
    assert rows.first([1, 2]) == 1, "the process she is running runs the mend without a restart"
    mended = (her_source / "planted_pkg" / "rows.py").read_text()
    namespace: dict = {}
    exec(compile(mended, "rows.py", "exec"), namespace)  # noqa: S102 - the merged source, run to see it mends
    assert namespace["first"]([1, 2]) == 1 and namespace["count"]([1, 2]) == 2
    # The check was mended, not deleted: what it refuses, it still refuses.
    with pytest.raises(ValueError, match="no rows"):
        namespace["first"]([])
    log = _git("log", "--format=%s", "-1", cwd=her_source)
    assert log.startswith("Mend planted_pkg.rows.first")
    shipped = _git("show", "--name-only", "--format=", "HEAD", cwd=her_source).split()
    assert "planted_pkg/rows.py" in shipped and any(name.startswith("tests/her_faults/") for name in shipped)


def test_a_fault_no_test_guards_is_not_mended_and_says_so(her_source):
    import planted_pkg.alone as alone

    done = asyncio.run(mend(_fault(lambda: alone.half(4)), proposals=40, seconds=600))
    assert done["status"] == goals.STALLED
    assert "guards" in done["why"] or "no candidate" in done["why"] or "nothing proposed" in done["why"], done
    assert _git("log", "--format=%s", "-1", cwd=her_source) == "planted\n"


def test_her_cycle_tries_each_fault_once_on_a_source_and_mends_the_next(her_source, monkeypatch):
    """One a cycle: the fault no test guards stalls and is not tried again on the same source; the next is mended."""
    import planted_pkg.alone as alone
    import planted_pkg.rows as rows

    from core.agency import mending_a_fault_of_hers

    heard = [_fault(lambda: alone.half(4)), _fault(lambda: rows.first([1, 2]))]
    monkeypatch.setattr(faults_she_can_reproduce, "faults_heard", lambda: heard)
    (first,) = asyncio.run(mending_a_fault_of_hers.mend_the_faults_she_heard(seconds=900))
    assert first["status"] == goals.STALLED
    (second,) = asyncio.run(mending_a_fault_of_hers.mend_the_faults_she_heard(seconds=900))
    assert second["status"] == "merged", second
    # Her mend moved her source, so the stalled fault is tried once more on it, and stalls again.
    (third,) = asyncio.run(mending_a_fault_of_hers.mend_the_faults_she_heard(seconds=900))
    assert third["status"] == goals.STALLED and "alone" in third["slug"]
    assert asyncio.run(mending_a_fault_of_hers.mend_the_faults_she_heard(seconds=900)) == []


def test_a_fault_heard_for_the_first_time_wakes_whoever_waits():
    from core.runtime.a_call_that_failed import FailedCall
    from core.runtime.errors import DegradationRecord
    from core.self_modification import faults_she_can_reproduce as faults

    woken = []
    faults.forget_for_test()
    faults.when_a_fault_is_first_heard(lambda: woken.append(1))
    call = FailedCall("m", "f", "m.py", 1, "builtins.ValueError", "no")
    record = DegradationRecord("s", "warning", "ValueError", "no", "", 1.0, failed_call=call)
    faults.hear(record)
    faults.hear(record)
    assert woken == [1]
    faults.forget_for_test()


def test_a_fault_raised_in_what_governs_her_is_not_drafted(her_source):
    """LIVE 16 November 2026: require_governance refused an ungoverned save, and her loop searched until its deadline
    for a change to the check. A mend there loosens the check, and her own yes refuses the file."""
    from core.runtime.a_call_that_failed import FailedCall

    call = FailedCall("core.governance_context", "require_governance", "core/governance_context.py", 548,
                      "core.governance_context.GovernanceViolationError", "called outside governed context")
    done = asyncio.run(mend(HeardFault(call, "causal_world_model", 0.0, 0.0), proposals=10, seconds=60))
    assert done["status"] == goals.STALLED and "which governs her" in done["why"], done
    assert not (her_source.parent / "workspaces").exists(), "no workspace is made for a mend it will not draft"


def test_a_none_her_function_was_handed_is_mended_by_the_checker_its_exception_asks_for(her_source):
    """AttributeError on None: her templates had no checker, so a fault of this, the commonest kind, had no mend."""
    import planted_pkg.people as people

    done = asyncio.run(mend(_fault(lambda: people.label(None)), proposals=80, seconds=900))
    assert done["status"] == "merged" and done["origin"] == "template:guard_none_value", done
    assert people.label(None) is None, "the running process runs the guarded body"
    assert "if person is not None else None" in (her_source / "planted_pkg" / "people.py").read_text()


def test_a_stalled_fault_taken_up_again_runs_on_her_source_as_it_now_is(her_source):
    """LIVE 16 November 2026: her drill's retry ran in the workspace of the try before, on the old source, so a guard
    fixed in between still failed there."""
    import planted_pkg.alone as alone

    first = asyncio.run(mend(_fault(lambda: alone.half(4)), proposals=40, seconds=600))
    assert first["status"] == goals.STALLED
    (her_source / "NOTES.md").write_text("moved on\n")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=her_source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "moved", cwd=her_source)
    asyncio.run(mend(_fault(lambda: alone.half(4)), proposals=40, seconds=600))
    workspace = goals.load(first["slug"]).workspace
    assert _git("rev-parse", "HEAD", cwd=workspace).strip() == _git("rev-parse", "HEAD", cwd=her_source).strip()


def test_her_mend_outlives_a_bound_put_on_the_cycle_that_started_it(monkeypatch):
    """LIVE 16 November 2026: her repair executor ran the cycle under a 90 s bound, which cancelled her drill's mend
    after its gates passed and it landed on origin, so it was never loaded into her running process."""
    from types import SimpleNamespace

    from core.agency import mending_a_fault_of_hers
    from core.self_modification.self_modification_engine import (
        AutonomousSelfModificationEngine as Engine,
    )

    finished = []

    async def a_slow_mend(**_kwargs):
        await asyncio.sleep(0.3)
        finished.append("mended")
        return []

    monkeypatch.setattr(mending_a_fault_of_hers, "mend_the_faults_she_heard", a_slow_mend)
    engine = SimpleNamespace()

    async def run():
        async def the_cycle():
            assert Engine._start_mending(engine) is True
            assert Engine._start_mending(engine) is False, "one mend at a time"
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(the_cycle(), timeout=0.05)
        await engine._mending

    asyncio.run(run())
    assert finished == ["mended"]
