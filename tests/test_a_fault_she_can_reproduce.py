"""A fault from her own running is kept as the call it was raised in, and made again as a test that fails the same way.

Master Todo F15, 14 November 2026: her degradation records kept three frames of traceback and nothing a fault could be
reproduced from, and her fault-to-patch pipeline had no caller. A planted module in a tree of its own stands in for
her source here, so a real call raises, is kept, is written as a test and run in a real pytest process.
"""
from __future__ import annotations

import ast
import asyncio
import sys
import textwrap

import pytest

from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import FailedCall, the_call_that_failed
from core.self_modification import faults_she_can_reproduce as faults

pytestmark = pytest.mark.unit

_PLANTED = textwrap.dedent('''
    def share(total, parts, *, rounding=2):
        return round(total / parts, rounding)


    def nested():
        def inner(x):
            return 1 / x
        return inner(0)


    class Ledger:
        def __init__(self):
            self.rows = []

        def last(self):
            return self.rows[-1]

        @staticmethod
        def ratio(a, b):
            return a / b

        @classmethod
        def opened(cls, name):
            return {"name": name}[name]


    def takes_anything(thing):
        raise TypeError("no")
''')


@pytest.fixture
def planted(tmp_path, monkeypatch):
    package = tmp_path / "planted_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "planted.py").write_text(_PLANTED)
    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("planted_pkg.planted", None)
    sys.modules.pop("planted_pkg", None)
    import planted_pkg.planted as module

    faults.forget_for_test()
    yield module
    faults.forget_for_test()


def _raised(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the planted faults are what is being caught
        return exc
    raise AssertionError("the planted call did not raise")


def test_a_function_and_its_plain_arguments_are_kept(planted):
    kept = the_call_that_failed(_raised(lambda: planted.share(10, 0, rounding=3)))
    assert isinstance(kept, FailedCall)
    assert (kept.module, kept.qualname, kept.file) == ("planted_pkg.planted", "share", "planted_pkg/planted.py")
    assert kept.error_type == "builtins.ZeroDivisionError"
    assert kept.call_source("share") == "share(total=10, parts=0, rounding=3)"


@pytest.mark.parametrize("call, why", [
    (lambda m: m.Ledger().last(), "runs on an instance it cannot rebuild"),
    (lambda m: m.nested(), "is defined inside another function"),
    (lambda m: m.takes_anything(object()), "is a object, not a plain value"),
])
def test_a_call_that_cannot_be_made_again_says_why(planted, call, why):
    kept = the_call_that_failed(_raised(lambda: call(planted)))
    assert isinstance(kept, str) and why in kept


def test_a_static_and_a_class_method_are_made_again_through_their_class(planted):
    ratio = the_call_that_failed(_raised(lambda: planted.Ledger.ratio(1, 0)))
    opened = the_call_that_failed(_raised(lambda: planted.Ledger.opened("box")))
    assert ratio.call_source("Ledger.ratio") == "Ledger.ratio(a=1, b=0)"
    assert opened.qualname == "Ledger.opened" and opened.call_source("Ledger.opened") == "Ledger.opened(name='box')"


def test_a_fault_raised_outside_her_source_is_not_kept(planted):
    assert the_call_that_failed(_raised(lambda: int("x"))) == "raised outside her source"


def test_an_argument_too_long_to_write_back_is_not_walked_whole(planted):
    kept = the_call_that_failed(_raised(lambda: planted.share("x" * 10_000_000, 0)))
    assert isinstance(kept, str) and "not a plain value" in kept


def test_every_degradation_record_carries_its_call(planted):
    from core.runtime.errors import record_degradation

    record = record_degradation("planted", _raised(lambda: planted.share(3, 0)), severity="debug",
                                action="kept the call", enforce_failure_policy=False)
    assert isinstance(record.failed_call, FailedCall) and record.failed_call.qualname == "share"


def test_each_fault_is_heard_once_however_often_it_recurs(planted):
    from core.runtime.errors import DegradationRecord

    def record_of(exc, at):
        return DegradationRecord("planted", "warning", type(exc).__name__, str(exc), "", at,
                                 failed_call=the_call_that_failed(exc))

    faults.hear(record_of(_raised(lambda: planted.share(1, 0)), 1.0))
    faults.hear(record_of(_raised(lambda: planted.share(2, 0)), 2.0))
    faults.hear(record_of(_raised(lambda: planted.Ledger().last()), 3.0))     # cannot be made again: not kept
    (heard,) = faults.faults_heard()
    assert heard.call.qualname == "share" and heard.times == 2 and heard.last_at == 2.0


def test_the_test_written_for_a_fault_makes_the_same_call(planted):
    heard = faults.HeardFault(the_call_that_failed(_raised(lambda: planted.share(10, 0))), "planted", 0.0, 0.0)
    source = faults.test_source(heard)
    ast.parse(source)
    assert "from planted_pkg.planted import share" in source
    assert "    share(total=10, parts=0, rounding=2)" in source


def test_a_fault_is_reproduced_when_its_test_raises_what_she_raised(planted, tmp_path):
    """Run for real: the written test in a pytest process of its own, in the tree that holds the planted module."""
    heard = faults.HeardFault(the_call_that_failed(_raised(lambda: planted.share(10, 0))), "planted", 0.0, 0.0)
    reproduced = asyncio.run(faults.reproduce(heard, tmp_path))
    assert reproduced.reproduced, reproduced.output
    defect = faults.as_defect(reproduced)
    assert defect.tests == (reproduced.test,) and defect.files == ("planted_pkg/planted.py",)

    # Mended, the same test passes, and it is no longer a defect.
    (tmp_path / "planted_pkg" / "planted.py").write_text(_PLANTED.replace("total / parts", "total / (parts or 1)"))
    mended = asyncio.run(faults.reproduce(heard, tmp_path))
    assert not mended.reproduced
    with pytest.raises(ValueError):
        faults.as_defect(mended)


def test_why_a_fault_cannot_be_made_again_is_said_once(planted, caplog):
    import logging

    from core.runtime.errors import DegradationRecord

    exc = _raised(lambda: planted.Ledger().last())
    record = DegradationRecord("planted", "warning", "IndexError", "", "", 1.0, failed_call=the_call_that_failed(exc))
    with caplog.at_level(logging.INFO, logger="Aura.SelfModification.Faults"):
        faults.hear(record)
        faults.hear(record)
    said = [r.getMessage() for r in caplog.records if "Not a fault she can make again" in r.getMessage()]
    assert len(said) == 1 and "runs on an instance it cannot rebuild" in said[0]
