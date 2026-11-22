"""A fault the outside world raised is not one she can make again, and is not hers to mend.

LIVE 20 and 21 November 2026: her mend of NetworkGateway.request, which raised "URLError: <urlopen error timed out>",
stalled four times on "Not mended: its test did not raise what she raised". A test in her workspace runs without the
network, so a timeout, a refused or reset connection, or a name that did not answer cannot be made again there.
"""
from __future__ import annotations

import logging
import sys
import textwrap

import pytest

from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import the_call_that_failed
from core.runtime.errors import DegradationRecord
from core.self_modification import faults_she_can_reproduce as faults

pytestmark = pytest.mark.unit

_PLANTED = textwrap.dedent('''
    import urllib.error


    def fetches(url):
        raise urllib.error.URLError("timed out")


    def resets(host):
        raise ConnectionResetError(f"{host} reset the connection")


    def divides(a, b):
        return a / b
''')


@pytest.fixture
def planted(tmp_path, monkeypatch):
    package = tmp_path / "world_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "planted.py").write_text(_PLANTED)
    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("world_pkg.planted", "world_pkg"):
        sys.modules.pop(name, None)
    import world_pkg.planted as module

    monkeypatch.setattr(faults, "where_heard_faults_are_kept", lambda: tmp_path / "state" / "heard.json")
    faults.forget_for_test()
    yield module
    faults.forget_for_test()


def _heard_from(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the planted faults are what is caught
        return DegradationRecord("planted", "warning", type(exc).__name__, str(exc), "", 1.0,
                                 failed_call=the_call_that_failed(exc))
    raise AssertionError("the planted call did not raise")


@pytest.mark.parametrize("error_type, worlds", [
    ("urllib.error.URLError", True), ("builtins.TimeoutError", True), ("builtins.ConnectionResetError", True),
    ("socket.gaierror", True), ("builtins.KeyError", False), ("builtins.OSError", False),
    ("builtins.FileNotFoundError", False), ("no.such.module.Error", False),
])
def test_the_worlds_exceptions_are_known_by_their_class(error_type, worlds):
    assert faults._raised_by_the_world(error_type) is worlds


def test_a_fault_the_world_raised_is_said_once_and_not_kept(planted, caplog):
    with caplog.at_level(logging.INFO, logger=faults.logger.name):
        faults.hear(_heard_from(lambda: planted.fetches("https://example.org")))
        faults.hear(_heard_from(lambda: planted.resets("example.org")))
        faults.hear(_heard_from(lambda: planted.divides(1, 0)))
    (heard,) = faults.faults_heard()
    assert heard.call.qualname == "divides", "her own fault is still hers"
    assert any("URLError is the world's" in record.getMessage() for record in caplog.records)


def test_a_fault_the_world_raised_kept_from_her_last_run_is_not_recalled(planted):
    import asyncio

    faults.hear(_heard_from(lambda: planted.divides(1, 0)))
    assert asyncio.run(faults.keep_what_was_heard())
    path = faults.where_heard_faults_are_kept()
    import json

    rows = json.loads(path.read_text())
    world = json.loads(json.dumps(rows[0]))
    world["call"]["qualname"], world["call"]["error_type"] = "fetches", "urllib.error.URLError"
    path.write_text(json.dumps([*rows, world]))
    faults.forget_for_test()
    assert faults.recall() == 1
    assert [fault.call.qualname for fault in faults.faults_heard()] == ["divides"]
