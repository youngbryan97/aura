"""What the outside check says a refused draft of hers lost becomes a test her next search keeps (Master Todo F16).

A draft her gates refused was asked of them again on the next cycle, unchanged. For a gate that failed for a reason
outside her draft that is right; for the outside check, which judged the draft itself, the same draft got the same
verdict every cycle and no other was looked for. Now the check's counterexamples are written as a test of what the
code before the draft gave, committed in her workspace in place of the draft, and her search runs again with it among
the passing tests it must keep.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from core.agency import her_goal_at_work as work
from core.agency import mending_a_fault_of_hers as mending
from core.self_modification.what_her_change_must_keep import (
    keepable,
    keeping_test_source,
    path_of_what_it_must_keep,
)
from outside_bar.verify_change import verify
from tests.test_her_goal_is_worked_by_her import (
    her_source,  # noqa: F401 - a fixture, found by its own name
)
from tests.test_outside_bar_verifies_a_change import (  # noqa: F401 - shop is a fixture, found by its own name
    _GUARD,
    _PRICES,
    _committed,
    _git,
    shop,
)

LOST_VALUE = {"function": "shop/prices.py::price", "input": {"args": {"table": "{'tea': 3}", "item": "'tea'"}},
              "before": "3", "after": "None", "kind": "another value", "counted": "lost"}
LOST_REFUSAL = {"function": "shop/prices.py::price", "input": {"args": {"table": "{'tea': 3}", "item": "'milk'"}},
                "before": "raised KeyError: 'milk'", "after": "None", "kind": "a refusal gone", "counted": "lost"}


@pytest.mark.parametrize(("one", "kept"), [
    (LOST_VALUE, True),
    (LOST_REFUSAL, True),
    ({**LOST_VALUE, "counted": "doubt"}, False),
    ({**LOST_VALUE, "as": "digests", "input": "1a6e,5d43"}, False),
    ({**LOST_VALUE, "before": "1; its arguments became {}"}, False),
    ({**LOST_VALUE, "function": "shop/prices.py::Shelf.row"}, False),
    ({**LOST_VALUE, "function": "tests/test_prices.py::price"}, False),
    ({**LOST_VALUE, "input": {"args": {"table": "<object>"}}}, False),
])
def test_only_a_lost_input_that_reads_back_as_source_is_kept(one, kept):
    assert keepable(one) is kept


def _run(source: str, repo: Path) -> subprocess.CompletedProcess:
    test = repo / "tests" / "test_kept_here.py"
    test.write_text(source)
    try:
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                               str(test)], cwd=repo, capture_output=True, text=True, check=False)
    finally:
        test.unlink()


def test_the_kept_inputs_pass_on_the_code_before_and_fail_on_the_draft_that_lost_them(shop):  # noqa: F811 - the fixture
    source = keeping_test_source([LOST_VALUE, LOST_REFUSAL], goal="Mend price for an unknown item")
    assert "assert price(table={'tea': 3}, item='tea') == 3" in source and "pytest.raises(KeyError)" in source
    assert _run(source, shop).returncode == 0, "the code before the draft gave these"
    (shop / "shop" / "prices.py").write_text(_PRICES.replace("return table[item]", "return None"))
    assert _run(source, shop).returncode != 0, "the draft that lost them fails them"


def test_the_overfit_draft_of_17_november_comes_back_with_what_it_lost(shop):  # noqa: F811 - the fixture
    """LIVE 17 November 2026, 052670e4a: ``if len(prices) <= 1: return None``; the price of a one-item table was lost."""
    (shop / "tests" / "test_prices.py").write_text(_GUARD.replace('price({"tea": 3}, "tea") == 3',
                                                                  'price({"tea": 3, "oats": 2}, "tea") == 3'))
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a guard with two items")
    overfit = _PRICES.replace("    return table[item]", "    if len(table) <= 1:\n        return None\n    return table[item]")
    result = verify(shop, _committed(shop, overfit))
    kept = [one for one in result["counterexamples"] if keepable(one)]
    assert any(one["before"] == "3" and one["after"] == "None" for one in kept), result["counterexamples"]


def _refused_draft(source: Path, must_keep: list) -> work.GoalRecord:
    record = work.GoalRecord(goal="Mend note_count", slug="mend-note-count")
    workspace = work.ensure_workspace(record)
    (workspace / "core" / "notes.py").write_text("def note_count():\n    return 1\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "her draft")
    record.offer = {"commit": _git(workspace, "rev-parse", "HEAD"), "files": ["core/notes.py"],
                    "gates": {"tests": {"passed": True}, "outside-check": {"passed": False, "tail": "did not hold",
                                                                          "must_keep": must_keep}}}
    work._write_now(record)
    return record


_NOTE_LOST = {"function": "core/notes.py::note_count", "input": {"args": {}}, "before": "0", "after": "1",
              "kind": "another value", "counted": "lost"}


def test_a_draft_the_outside_check_refused_is_set_aside_and_what_it_lost_is_committed(her_source):  # noqa: F811
    record = _refused_draft(her_source, [_NOTE_LOST])
    workspace = record.workspace
    again = asyncio.run(mending._her_draft_asked_again(record, workspace, fault=None))
    assert again is None, "searched again, not asked again"
    path = path_of_what_it_must_keep(record.slug)
    assert (workspace / path).read_text().count("assert note_count() == 0") == 1
    assert _git(workspace, "log", "--format=%s", "-1").startswith("What my change for this goal must keep")
    assert _git(workspace, "rev-parse", "HEAD^") == record.base, "her refused draft is set aside"
    kept = work.load(record.slug)
    assert kept.status == work.WORKING and kept.offer["must_keep"] == [_NOTE_LOST] and not kept.offer.get("commit")
    assert kept.offers_before[-1]["set_aside"] == "the outside check refused it"
    assert path in work.tests_that_touch(["core/notes.py"], workspace), "her next search keeps it among its guards"


def test_with_nothing_it_can_keep_the_goal_waits_for_her_source_to_change(her_source):  # noqa: F811 - the fixture
    record = _refused_draft(her_source, [])
    said = asyncio.run(mending._her_draft_asked_again(record, record.workspace, fault=None))
    assert said["status"] == work.STALLED and "nothing it said it lost can be kept" in said["why"]


def test_a_draft_another_gate_refused_is_still_asked_again(her_source, monkeypatch):  # noqa: F811 - the fixture
    record = _refused_draft(her_source, [])
    record.offer["gates"] = {"lint": {"passed": False, "tail": "main's lint fails"}}
    work._write_now(record)
    asked: list = []

    async def taken(record, fault, mended):
        asked.append(record.offer["commit"])
        return {"ok": False}

    monkeypatch.setattr(mending, "_taken_and_loaded", taken)
    fault = type("Fault", (), {"call": type("Call", (), {"file": "core/notes.py"})()})()
    said = asyncio.run(mending._her_draft_asked_again(record, record.workspace, fault))
    assert said["why"] == "her draft asked of her gates again" and asked
