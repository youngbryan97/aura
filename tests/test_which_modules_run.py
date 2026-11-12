"""F12's first measure: which watched modules ran, from sys.monitoring, without paying for a function after its
first call."""
from __future__ import annotations

import sys
import types

import pytest

from core.observability.which_modules_run import ModuleActivity

pytestmark = pytest.mark.unit


def _module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    sys.modules[name] = module
    return module


@pytest.fixture
def two_modules():
    ran = _module("probe_pkg.ran", "class Organ:\n    def tick(self):\n        return 1\n    def idle(self):\n"
                                    "        return 2\ndef helper():\n    return 3\n")
    never = _module("probe_pkg.never", "def unused():\n    return 4\n")
    yield ran, never
    for name in ("probe_pkg.ran", "probe_pkg.never", "probe_pkg.late"):
        sys.modules.pop(name, None)


def test_a_module_whose_functions_run_is_told_from_one_that_never_does(two_modules):
    ran, _never = two_modules
    activity = ModuleActivity(prefix="probe_pkg.", uptake_s=0)
    assert activity.start()
    try:
        organ = ran.Organ()
        for _ in range(1000):
            organ.tick()
        report = activity.report()
    finally:
        activity.stop()
    row = report["modules"]["probe_pkg.ran"]
    assert row["functions"] == 3 and row["ran"] == 1
    assert [name for name, _at in row["first_calls"]] == ["Organ.tick"]
    assert report["modules_that_never_ran"] == ["probe_pkg.never"]


def test_a_module_imported_after_the_watch_began_is_taken_up(two_modules):
    activity = ModuleActivity(prefix="probe_pkg.", uptake_s=0)
    assert activity.start()
    try:
        late = _module("probe_pkg.late", "def first():\n    return 5\n")
        activity.report()
        late.first()
        assert activity.report()["modules"]["probe_pkg.late"]["ran"] == 1
    finally:
        activity.stop()


def test_the_watch_gives_back_its_monitoring_slot(two_modules):
    activity = ModuleActivity(prefix="probe_pkg.", uptake_s=0)
    assert activity.start()
    tool = activity._tool
    activity.stop()
    assert sys.monitoring.get_tool(tool) is None
