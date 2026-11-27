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
    (LOST_REFUSAL, False),  # a KeyError from an operation may be the fault the goal mends (round seven, G3)
    ({**LOST_REFUSAL, "before": "raised ValueError by the code: no such item"}, True),
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
    assert "assert _same(price(table={'tea': 3}, item='tea'), 3)" in source
    assert "KeyError" not in source, "the crash this goal mends is not kept as something to keep"
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
    assert (workspace / path).read_text().count("assert _same(note_count(), 0)") == 1
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
    fault = type("Fault", (), {"call": type("Call", (), {"file": "core/notes.py", "qualname": "note_count", "line": 2,
                                                         "error_type": "builtins.ValueError"})()})()
    said = asyncio.run(mending._her_draft_asked_again(record, record.workspace, fault))
    assert said["why"] == "her draft asked of her gates again" and asked


# Round seven of the red team (22 November 2026) wrote three counterexamples the outside check really returned, each of
# which this writer turned into a test that failed on the code before the draft, or kept a crash as a requirement.
_ROUND_SEVEN = {
    "keywords": ("def price(base: int, **opts) -> int:\n    return base - opts.get('discount', 0)\n",
                 {"function": "mod.py::price", "input": {"args": {"base": "99"}, "keywords": "{'discount': 20}"},
                  "before": "79", "after": "-999", "kind": "another value", "counted": "lost"}),
    "positional only": ("def score(x: int, /) -> int:\n    return x * 2\n",
                        {"function": "mod.py::score", "input": {"args": {"x": "9"}}, "before": "18", "after": "-999",
                         "kind": "another value", "counted": "lost"}),
    # Round eight: what the check awaited or read out, and a definition behind a version guard.
    "a coroutine": ("async def async_double(x: int) -> int:\n    return x * 2\n",
                    {"function": "mod.py::async_double", "input": {"args": {"x": "9"}}, "before": "18", "after": "-999",
                     "kind": "another value", "counted": "lost"}),
    "a generator": ("def count_up(n: int):\n    for i in range(n):\n        yield i\n",
                    {"function": "mod.py::count_up", "input": {"args": {"n": "2"}}, "before": "[0, 1]", "after": "[-999]",
                     "kind": "another value", "counted": "lost"}),
    "behind a version guard": ("import sys\n\nif sys.version_info >= (3, 8):\n    def score(x: int, /) -> int:\n"
                               "        return x * 2\n",
                               {"function": "mod.py::score", "input": {"args": {"x": "20"}}, "before": "40",
                                "after": "-999", "kind": "another value", "counted": "lost"}),
}


@pytest.mark.parametrize("name", sorted(_ROUND_SEVEN))
def test_a_kept_input_is_called_the_way_the_check_called_it(name, tmp_path):
    source, lost = _ROUND_SEVEN[name]
    (tmp_path / "mod.py").write_text(source)
    (tmp_path / "conftest.py").write_text("")  # so pytest puts this folder on the path, as a repository's root
    (tmp_path / "tests").mkdir()
    kept = keeping_test_source([lost], goal="round seven", root=tmp_path)
    assert kept, "the input is writable"
    assert _run(kept, tmp_path).returncode == 0, ("the code before the draft passes what it gave", kept)


def test_a_crash_on_a_none_the_check_put_in_is_not_kept():
    crash = {"function": "mod.py::double", "input": {"args": {"x": "None"}},
             "before": "raised TypeError: unsupported operand type(s) for *: '<input>Type' and 'int'", "after": "0",
             "kind": "a refusal gone", "counted": "lost"}
    assert not keepable(crash)
    assert keeping_test_source([crash], goal="round seven") == ""
