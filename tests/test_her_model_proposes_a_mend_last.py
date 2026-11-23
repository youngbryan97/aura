"""Her model proposes a mend of a fault of hers only last, and its proposal is a candidate like any other.

Bryan, 22 November 2026: "As a last resort proposer is ok so long as its checked and verified the same way." The
proposer (core/self_modification/a_mend_her_model_proposes.py) writes each edit her model proposes as a SEARCH/REPLACE
block that repair_by_selection reads, applies and tests as it does her own Python's; her automatic loop hands it to the
search as the proposer after her own.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.self_modification.a_mend_her_model_proposes import ORIGIN, her_model_as_a_last_resort
from core.self_modification.code_that_looks_wrong import Edit, Suspicion
from core.self_modification.repair_by_selection import applied, edits_in

_SOURCE = '''import math


def price(table, item):
    return table[item]


def area(r):
    return math.pi * r * r
'''
_DEFECT = SimpleNamespace(tests=("tests/test_prices.py::test_an_unknown_item_has_no_price",),
                          seen="KeyError: 'milk'")


def _her_model_says(old: str, new: str):
    async def proposes(function: str, seen_wrong: dict) -> list:
        start = function.index(old)
        return [Suspicion(pattern="her model", line=2, why="an unknown item has no price",
                          edits=[Edit(start, start + len(old), new)])]
    return proposes


def test_what_her_model_proposes_is_a_block_the_search_applies():
    propose = her_model_as_a_last_resort("shop/prices.py", 5,
                                         propose_edits=_her_model_says("table[item]", "table.get(item)"))
    (proposal,) = asyncio.run(propose(_DEFECT, {"shop/prices.py": _SOURCE}, None, 1))
    origin, text = proposal
    assert origin == ORIGIN
    mended, why = applied({"shop/prices.py": _SOURCE}, edits_in(text, ["shop/prices.py"]))
    assert not why and "return table.get(item)" in mended["shop/prices.py"]
    assert "math.pi * r * r" in mended["shop/prices.py"], "only the function the fault was raised in is changed"


def test_it_is_shown_only_the_function_the_fault_was_raised_in_and_what_failed():
    shown = {}

    async def proposes(function: str, seen_wrong: dict) -> list:
        shown.update(function=function, seen_wrong=seen_wrong)
        return []

    propose = her_model_as_a_last_resort("shop/prices.py", 5, propose_edits=proposes)
    assert asyncio.run(propose(_DEFECT, {"shop/prices.py": _SOURCE}, None, 1)) == []
    assert shown["function"].startswith("def price(") and "area" not in shown["function"]
    assert "KeyError" in shown["seen_wrong"]["the failing test"]


def test_nothing_is_proposed_outside_a_function_or_when_her_model_proposes_nothing():
    nothing = her_model_as_a_last_resort("shop/prices.py", 1, propose_edits=_her_model_says("table[item]", "x"))
    assert asyncio.run(nothing(_DEFECT, {"shop/prices.py": _SOURCE}, None, 1)) == []
    unchanged = her_model_as_a_last_resort("shop/prices.py", 5, propose_edits=_her_model_says("table[item]", "table[item]"))
    assert asyncio.run(unchanged(_DEFECT, {"shop/prices.py": _SOURCE}, None, 1)) == []


def test_her_automatic_loop_gives_the_search_her_model_last(monkeypatch):
    from core.agency import her_goal_at_work as goals
    from core.agency import mending_a_fault_of_hers as loop
    from core.self_modification import faults_she_can_reproduce as heard

    fault = SimpleNamespace(call=SimpleNamespace(file="shop/prices.py", line=5, module="shop.prices", qualname="price",
                                                 error_type="builtins.KeyError", error_message="'milk'"))
    given = {}

    async def nothing(*_args, **_kwargs):
        return None

    async def mend(found, **kwargs):
        given.update(kwargs)
        return {"slug": "x"}

    for name in ("keep_what_was_heard",):
        monkeypatch.setattr(heard, name, nothing)
    monkeypatch.setattr(heard, "mends_undone", lambda: [])
    monkeypatch.setattr(heard, "faults_heard", lambda: [fault])
    monkeypatch.setattr(goals, "mends_undone_on_record", lambda: [])
    monkeypatch.setattr(goals, "settle_probation", lambda: [])
    monkeypatch.setattr(goals, "source_head", lambda: "")
    monkeypatch.setattr(goals, "load", lambda slug: None)
    for name in ("_keep_her_real_calls_of_what_she_may_change", "_keep_her_recent_turns",
                 "_start_an_outside_comparison_when_due"):
        monkeypatch.setattr(loop, name, nothing)
    monkeypatch.setattr(loop, "mend", mend)
    assert asyncio.run(loop.mend_the_faults_she_heard(seconds=1)) == [{"slug": "x"}]
    assert callable(given["model"]), "the search is given her model's proposer"
