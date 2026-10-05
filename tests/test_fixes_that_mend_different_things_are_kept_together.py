"""Fixes found in one round that each mend something different, in different places, are kept together.

One fix a round meant five rounds for Pong's five faults, about a minute and a
half each, with the person watching and waiting.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.self_modification.code_that_looks_wrong import Edit, Suspicion
from core.self_modification.repairing_by_behaviour import _alongside, _Believed


def _fix(pattern: str, line: int, start: int, right: set[str], wrong: set[str] = frozenset()) -> tuple:
    return (Suspicion(pattern, line, pattern), [Edit(start, start + 4, "x")], SimpleNamespace(right=set(right), wrong=set(wrong), findings={}))


def test_fixes_for_different_faults_in_different_places_go_together():
    now = _Believed(wrong={"controls", "idle", "credited", "escaped"})
    best = _fix("direction against its name", 63, 100, {"controls", "credited"}, {"idle"})
    computer = _fix("always zero", 72, 300, {"idle", "credited"}, {"controls"})
    wall = _fix("one-sided boundary", 93, 500, {"escaped", "credited"})
    same_place = _fix("direction against its name", 64, 102, {"controls"})  # overlaps the best's edit
    nothing_new = _fix("same in both branches", 102, 700, {"credited"})  # mends only what the best mends
    taken = _alongside(best, [best, computer, wall, same_place, nothing_new], now)
    assert sorted(t[0].line for t in taken) == [72, 93]


def test_one_fix_alone_has_nothing_alongside():
    now = _Believed(wrong={"controls"})
    best = _fix("direction against its name", 63, 100, {"controls"})
    assert _alongside(best, [best], now) == []
