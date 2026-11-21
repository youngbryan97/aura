"""In the model suite a test of a claim that needs her encoder or a model fails when it cannot run (U07-09).

An outside review of 19 November 2026: the bug in her attribution surface survived green runs because the tests that
needed her encoder skipped where it was not loaded, and a skip read as a pass. ``make model-suite`` runs them with
her encoder and her record reader loaded, and there a skip is a failure. 20 November 2026: 66 passed, none skipped.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _driven(monkeypatch, *, suite: bool, marked: bool, skipped: bool = True):
    from tests import conftest

    if suite:
        monkeypatch.setenv("AURA_MODEL_SUITE", "1")
    else:
        monkeypatch.delenv("AURA_MODEL_SUITE", raising=False)
    report = SimpleNamespace(skipped=skipped, outcome="skipped" if skipped else "passed",
                             longrepr=("tests/x.py", 7, "Skipped: her encoder is not available here"))
    item = SimpleNamespace(nodeid="tests/x.py::test_y", get_closest_marker=lambda name: object() if marked else None)
    hook = conftest.pytest_runtest_makereport.__wrapped__ if hasattr(conftest.pytest_runtest_makereport,
                                                                     "__wrapped__") else conftest.pytest_runtest_makereport
    driven = hook(item, SimpleNamespace(when="setup"))
    next(driven)
    with pytest.raises(StopIteration):
        driven.send(SimpleNamespace(get_result=lambda: report))
    return report


def test_a_skipped_model_test_fails_in_the_model_suite(monkeypatch):
    report = _driven(monkeypatch, suite=True, marked=True)
    assert report.outcome == "failed"
    assert "did not run: Skipped: her encoder is not available here" in report.longrepr


@pytest.mark.parametrize(("suite", "marked"), [(False, True), (True, False)])
def test_elsewhere_or_unmarked_a_skip_stays_a_skip(monkeypatch, suite, marked):
    assert _driven(monkeypatch, suite=suite, marked=marked).outcome == "skipped"


def test_a_test_that_ran_is_left_as_it_ran(monkeypatch):
    assert _driven(monkeypatch, suite=True, marked=True, skipped=False).outcome == "passed"


def test_the_target_collects_only_the_marked_files_and_selects_them():
    from pathlib import Path

    makefile = Path(__file__).resolve().parents[1].joinpath("Makefile").read_text(encoding="utf-8")
    assert "AURA_MODEL_SUITE=1" in makefile and '-m "model and not live" $(MODEL_SUITE_FILES)' in makefile
