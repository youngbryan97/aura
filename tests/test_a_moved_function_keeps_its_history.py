"""The method-size gate keyed on `path::name`, so a move looked like a birth.

A function that left one file for another read as a NEW god function in the
new file and "no longer over the threshold" in the old one — and a function
that moved AND grew was reported as new, with nothing to compare against.
Three chat functions moved on 2026-09-12 and grew by 292, 171 and 35 lines
behind the label NEW.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from lint_method_size import compare_with_baseline  # noqa: E402


def _f(lines: int, cc: int = 10) -> dict[str, int]:
    return {"lines": lines, "complexity": cc, "returns": 1}


def test_a_move_that_grew_is_growth_measured_against_where_it_came_from() -> None:
    previous = {"interface/routes/chat.py::_stabilize": _f(686, 190)}
    now = {"interface/routes/chat_reply_repair.py::_stabilize": _f(857, 199)}
    grew, appeared, shrank = compare_with_baseline(now, previous)
    assert appeared == []
    assert len(grew) == 1 and "moved from interface/routes/chat.py and grew 686 -> 857" in grew[0]
    assert not any("no longer over the threshold" in line for line in shrank)


def test_a_move_that_did_not_grow_is_not_a_finding() -> None:
    previous = {"a/old.py::big": _f(500)}
    now = {"a/new.py::big": _f(500)}
    grew, appeared, shrank = compare_with_baseline(now, previous)
    assert grew == [] and appeared == []
    assert any("moved from a/old.py, unchanged" in line for line in shrank)


def test_a_genuinely_new_function_is_still_new() -> None:
    previous = {"a/old.py::big": _f(500)}
    now = {"a/old.py::big": _f(500), "b/fresh.py::brand_new": _f(450)}
    grew, appeared, shrank = compare_with_baseline(now, previous)
    assert grew == [] and shrank == []
    assert appeared == ["b/fresh.py::brand_new: NEW at 450 lines, CC 10"]


def test_growth_in_place_is_reported_as_before() -> None:
    previous = {"a/x.py::f": _f(400, 50)}
    now = {"a/x.py::f": _f(420, 55)}
    grew, appeared, shrank = compare_with_baseline(now, previous)
    assert grew == ["a/x.py::f: 400 -> 420 lines (CC 50 -> 55)"]
