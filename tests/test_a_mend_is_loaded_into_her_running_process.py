"""A mend of hers is loaded into the process she is running: the functions keep their identity and take the new body.

Master Todo F15, 16 November 2026: a mend her own yes took went into her source on disk, and the process she was
running kept the old code until she restarted. A module planted in a tree of its own is imported here for real, held
by references made before the mend, and mended.
"""
from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

from core.self_modification.into_the_running_process import (
    load_into_this_process,
    the_load_holding,
    undo_every_load_for_test,
    what_changed,
)

pytestmark = pytest.mark.unit

_BEFORE = textwrap.dedent('''
    import functools


    def first(rows):
        if len(rows) > 0:
            raise ValueError("there are no rows")
        return rows[0]


    def wrapped(f):
        @functools.wraps(f)
        def inner(*a, **k):
            return f(*a, **k)
        return inner


    class Ledger:
        def total(self):
            return sum(self.rows) - 1

        @staticmethod
        def half(n):
            return n // 3

        @property
        def size(self):
            return len(self.rows) + 1

        @wrapped
        def named(self):
            return "old"


    async def later():
        return "old"


    def after():
        raise RuntimeError("where am I")
''')


def _mended(source: str) -> str:
    return (source.replace("if len(rows) > 0:", "if not rows:")
            .replace("sum(self.rows) - 1", "sum(self.rows)")
            .replace("n // 3", "n // 2")
            .replace("len(self.rows) + 1", "len(self.rows)")
            .replace('return "old"', 'return "new"'))


@pytest.fixture
def planted(tmp_path, monkeypatch):
    package = tmp_path / "loaded_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "rows.py").write_text(_BEFORE)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("loaded_pkg", "loaded_pkg.rows"):
        sys.modules.pop(name, None)
    import loaded_pkg.rows as module

    yield module
    undo_every_load_for_test()
    for name in ("loaded_pkg", "loaded_pkg.rows"):
        sys.modules.pop(name, None)


def test_every_reference_made_before_the_mend_runs_the_mended_body(planted):
    held_first = planted.first                       # an import elsewhere: `from loaded_pkg.rows import first`
    ledger = planted.Ledger()                        # an instance made before
    ledger.rows = [2, 3]
    bound_total = ledger.total                       # a bound method held by a callback
    with pytest.raises(ValueError):
        held_first([1, 2])

    loaded = load_into_this_process("loaded_pkg.rows", _BEFORE, _mended(_BEFORE))

    assert not loaded.waits, loaded.waits
    assert held_first([1, 2]) == 1 and held_first is planted.first
    with pytest.raises(ValueError, match="no rows"):
        held_first([])
    assert bound_total() == 5 and planted.Ledger.half(9) == 4 and ledger.size == 2
    assert ledger.named() == "new" and asyncio.run(planted.later()) == "new"
    assert the_load_holding("loaded_pkg.rows", "first") is loaded


def test_a_function_that_moved_in_the_file_says_its_new_line(planted):
    """A mend that adds a line shifts what follows; her next fault must point at the line in the file."""
    shifted = _BEFORE.replace("    return rows[0]", "    # the first row\n    return rows[0]")
    assert shifted != _BEFORE
    load_into_this_process("loaded_pkg.rows", _BEFORE, shifted)
    with pytest.raises(RuntimeError) as raised:
        planted.after()
    line = raised.traceback[-1].lineno + 1
    assert shifted.splitlines()[line - 1].strip() == 'raise RuntimeError("where am I")'


def test_undoing_a_load_puts_back_the_code_it_replaced(planted):
    loaded = load_into_this_process("loaded_pkg.rows", _BEFORE, _mended(_BEFORE))
    assert planted.first([1]) == 1
    assert "first" in loaded.undo()
    with pytest.raises(ValueError):
        planted.first([1])
    assert the_load_holding("loaded_pkg.rows", "first") is None


@pytest.mark.parametrize("change, why", [
    (lambda s: s.replace("def first(rows):", "def first(rows, default=None):"), "outside function bodies"),
    (lambda s: s.replace("    return rows[0]", "    return _helper(rows)") + "\n\ndef _helper(rows):\n"
                                                                           "    return rows[0]\n",
     "outside function bodies"),
    (lambda s: s.replace("        return sum(self.rows) - 1", "        return super().__hash__()"),
     "different closure"),
    (lambda s: s.replace('async def later():\n    return "old"', 'async def later():\n    yield "old"'),
     "different closure or kind"),
])
def test_a_change_that_is_not_only_a_body_loads_nothing_and_says_it_waits(planted, change, why):
    assert change(_BEFORE) != _BEFORE
    loaded = load_into_this_process("loaded_pkg.rows", _BEFORE, change(_BEFORE))
    assert loaded.swaps == [] and any(why in said for said in loaded.waits), loaded.waits
    with pytest.raises(ValueError):
        planted.first([1, 2])


def test_a_module_this_process_never_imported_waits_for_its_first_import():
    loaded = load_into_this_process("loaded_pkg.never_imported", "x = 1\n", "x = 2\n")
    assert loaded.swaps == [] and "first imported" in loaded.waits[0]


def test_only_bodies_and_moves_are_counted_as_changed():
    names, why = what_changed(_BEFORE, _mended(_BEFORE))
    assert not why and set(names) == {"first", "Ledger.total", "Ledger.half", "Ledger.size", "Ledger.named", "later"}


def test_a_source_with_no_body_changed_says_there_is_nothing_to_load(planted):
    loaded = load_into_this_process("loaded_pkg.rows", _BEFORE, _BEFORE + "\n# a comment\n")
    assert loaded.swaps == [] and "nothing to load" in loaded.waits[0]


def _recorded(planted_call):
    from core.runtime.errors import record_degradation

    try:
        planted_call()
    except Exception as exc:  # noqa: BLE001 - the planted fault is what is caught
        return record_degradation("loaded_pkg", exc, severity="warning", action="planted",
                                  enforce_failure_policy=False)
    raise AssertionError("the planted call did not raise")


@pytest.fixture
def heard_as_hers(planted, tmp_path, monkeypatch):
    from core.runtime import a_call_that_failed
    from core.self_modification import faults_she_can_reproduce as faults

    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    monkeypatch.setattr(faults, "where_heard_faults_are_kept", lambda: tmp_path / "heard.json")
    faults.forget_for_test()
    yield planted, faults
    faults.forget_for_test()


@pytest.mark.parametrize("call, undone", [
    (lambda m: m.first(5), True),        # TypeError: a new error from the mended body, which it does not raise
    (lambda m: m.first([]), False),      # ValueError("there are no rows"): the mended check answering, on purpose
])
def test_a_loaded_mend_is_undone_when_its_function_faults_in_a_way_it_does_not_mean(heard_as_hers, call, undone):
    planted, faults = heard_as_hers
    mended_call = _recorded(lambda: planted.first([1, 2])).failed_call
    load_into_this_process("loaded_pkg.rows", _BEFORE, _mended(_BEFORE), goal="mend-first", mended_call=mended_call)
    record = _recorded(lambda: call(planted))
    assert record.raised_in == ("loaded_pkg.rows", "first")
    faults.hear(record)
    assert (the_load_holding("loaded_pkg.rows", "first") is None) is undone
    assert faults.mends_undone() == (["mend-first"] if undone else [])


def test_the_fault_a_mend_was_for_coming_back_undoes_it(heard_as_hers):
    """A mend that did not mend: the same call raises the same ValueError, by name, as before it was loaded."""
    planted, faults = heard_as_hers
    mended_call = _recorded(lambda: planted.first([1, 2, 3])).failed_call
    load_into_this_process("loaded_pkg.rows", _BEFORE, _BEFORE.replace("if len(rows) > 0:", "if len(rows) > 1:"),
                           goal="mend-first", mended_call=mended_call)
    faults.hear(_recorded(lambda: planted.first([1, 2, 3])))
    assert the_load_holding("loaded_pkg.rows", "first") is None
    with pytest.raises(ValueError):
        planted.first([1])


def test_a_load_outside_a_governed_scope_is_refused_while_her_runtime_lives(planted, service_container):
    from core.governance_context import GovernanceViolation, local_internal_governed_scope

    service_container.register_instance("executive_core", object(), required=False)
    service_container.lock_registration()
    with pytest.raises(GovernanceViolation):
        load_into_this_process("loaded_pkg.rows", _BEFORE, _mended(_BEFORE))
    with pytest.raises(ValueError):
        planted.first([1, 2])
    with local_internal_governed_scope("into_the_running_process.test", domain="state_mutation",
                                       receipt_prefix="test-load", constraints={"operation": "load"}):
        loaded = load_into_this_process("loaded_pkg.rows", _BEFORE, _mended(_BEFORE))
    assert loaded.swaps and planted.first([1, 2]) == 1
