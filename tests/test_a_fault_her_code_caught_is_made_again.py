"""A fault her code caught and recorded is made again when the call records it again, and a fault SQLite named as a
held database is made again with the database held (Master Todo F16, from her own goals, 22 November 2026).

Of her 12 stalled goals, 5 stalled on "Not mended: its test did not raise what she raised". Every fault she hears comes
from a degradation record, and her code records most of them where it catches them, so a test that only waited for a
raise passed on each: OutcomeLedger._write_rows caught "database is locked" and recorded it. And a locked database is
the state of another connection, which no argument of the call carries.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import textwrap

import pytest

from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import the_call_that_failed
from core.self_modification import faults_she_can_reproduce as faults

pytestmark = pytest.mark.unit

#: Her runtime's degradation tracker as a workspace of hers has it, for a tree of its own.
_ERRORS = textwrap.dedent('''
    class _Record:
        def __init__(self, error_type):
            self.error_type = error_type


    class _Tracker:
        def __init__(self):
            self.records = []

        def recent(self, *, limit=20):
            return self.records[-limit:]


    _TRACKER = _Tracker()


    def get_degradation_tracker():
        return _TRACKER


    def record_degradation(subsystem, error, **_):
        _TRACKER.records.append(_Record(type(error).__qualname__))
''')
_PLANTED = textwrap.dedent('''
    import sqlite3

    from core.runtime.errors import record_degradation


    def count_in(text):
        try:
            return int(text)
        except ValueError as exc:
            record_degradation("planted", exc)
            return None


    def save(path, text, wait=0.2):
        con = sqlite3.connect(path, timeout=wait)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS notes (body TEXT)")
            con.execute("INSERT INTO notes VALUES (?)", (text,))
            con.commit()
        finally:
            con.close()


    def fetch(url):
        try:
            raise urllib_error()
        except OSError as exc:
            raise RuntimeError(f"could not fetch {url}") from exc


    def urllib_error():
        import urllib.error
        return urllib.error.URLError("refused")
''')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A tree holding the planted module and a stand-in for her runtime's errors, which the written test imports; this
    process keeps its own core."""
    tree = tmp_path / "tree"
    (tree / "planted_pkg").mkdir(parents=True)
    (tree / "planted_pkg" / "__init__.py").write_text("")
    (tree / "planted_pkg" / "planted.py").write_text(_PLANTED)
    (tree / "core" / "runtime").mkdir(parents=True)
    for name in ("core/__init__.py", "core/runtime/__init__.py"):
        (tree / name).write_text("")
    (tree / "core" / "runtime" / "errors.py").write_text(_ERRORS)
    monkeypatch.setattr(a_call_that_failed, "_ROOT", tree.resolve())
    monkeypatch.syspath_prepend(str(tree))
    for name in ("planted_pkg.planted", "planted_pkg"):
        sys.modules.pop(name, None)
    import planted_pkg.planted as module

    monkeypatch.setattr(faults, "where_heard_faults_are_kept", lambda: tmp_path / "state" / "heard.json")
    faults.forget_for_test()
    yield tree, module
    faults.forget_for_test()


def _raised(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the planted faults are what is being caught
        return exc
    raise AssertionError("the planted call did not raise")


def test_a_fault_her_code_caught_and_recorded_is_made_again_when_it_records_it_again(workspace):
    tree, planted = workspace
    assert planted.count_in("many") is None, "it raises nothing: it catches the fault and records it"
    # The call as her degradation record keeps it: raised in count_in, where int() failed and was caught.
    call = a_call_that_failed.FailedCall(module="planted_pkg.planted", qualname="count_in", file="planted_pkg/planted.py",
                                         line=8, error_type="builtins.ValueError", error_message="invalid literal",
                                         keywords=(("text", "'many'"),))
    heard = faults.HeardFault(call, "planted", 0.0, 0.0)
    reproduced = asyncio.run(faults.reproduce(heard, tree))
    assert reproduced.reproduced and "it recorded ValueError again" in reproduced.output, reproduced.output
    (tree / "planted_pkg" / "planted.py").write_text(
        _PLANTED.replace("return int(text)", "return int(text) if text.isdigit() else None"))
    assert not asyncio.run(faults.reproduce(heard, tree)).reproduced, "mended, it records nothing and passes"


def test_a_database_another_connection_held_is_made_again_with_it_held(workspace):
    tree, planted = workspace
    database = str(tree / "notes.db")
    holder = sqlite3.connect(database, isolation_level=None)
    holder.execute("CREATE TABLE IF NOT EXISTS notes (body TEXT)")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        kept = the_call_that_failed(_raised(lambda: planted.save(database, "hello")))
    finally:
        holder.close()
    assert ("sqlite_errorname", "SQLITE_BUSY") in kept.codes and ("sqlite_errorcode", "5") in kept.codes, kept
    # The extended codes are the same thing held: SQLITE_BUSY_SNAPSHOT is 517, whose low byte is 5.
    assert faults._a_held_database((("sqlite_errorcode", "517"),)) and faults._a_held_database(
        (("sqlite_errorname", "SQLITE_LOCKED_SHAREDCACHE"),))
    assert not faults._a_held_database((("sqlite_errorcode", "1"), ("sqlite_errorname", "SQLITE_ERROR")))
    heard = faults.HeardFault(kept, "planted", 0.0, 0.0)
    assert "_held_by_another" in faults.test_source(heard)
    reproduced = asyncio.run(faults.reproduce(heard, tree))
    assert reproduced.reproduced, reproduced.output
    # Mended to wait past what another writer holds it for, the same test, which passes the same wait, passes.
    (tree / "planted_pkg" / "planted.py").write_text(_PLANTED.replace("timeout=wait", "timeout=max(wait, 10.0)"))
    assert not asyncio.run(faults.reproduce(heard, tree)).reproduced


def test_a_fault_raised_while_handling_the_worlds_is_the_worlds(workspace):
    _, planted = workspace
    kept = the_call_that_failed(_raised(lambda: planted.fetch("https://example.org")))
    assert kept.error_type == "builtins.RuntimeError" and "urllib.error.URLError" in kept.causes
    assert faults._raised_by_the_world(kept.error_type, kept.causes)
    assert not faults._raised_by_the_world(kept.error_type), "read from its own class alone, it is her code's"
