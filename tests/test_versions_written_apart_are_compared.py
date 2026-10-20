"""Versions of a library written apart: the one the others agree with is kept, and a test is doubted only where
the versions and her model's own working of the call agree on another value. The code and its runs here come from
stand-ins for her model and the page."""
from __future__ import annotations

import asyncio
import json

import pytest

from core.rebuilding.building_her_libraries import build_library
from core.rebuilding.her_libraries import FunctionPlan, LibraryPlan, LibraryTest, RunOfATest
from core.rebuilding.what_versions_agree_on import Version, agreed_value, same_value, the_best_of

pytestmark = pytest.mark.unit

_PLAN = LibraryPlan(
    name="Days", purpose="days between two dates", rests_on=["Gregorian calendar"],
    functions=[FunctionPlan(name="between", params=["a", "b"], does="whole days from a to b"),
               FunctionPlan(name="leap", params=["year"], does="whether the year is a leap year")],
    tests=[LibraryTest(call='Days.between("2024-02-28", "2024-03-01")', expect="1"),     # wrong: 2024 is a leap year
           LibraryTest(call="Days.leap(2024)", expect="true")])
_RIGHT = {'Days.between("2024-02-28", "2024-03-01")': "2", "Days.leap(2024)": "true"}


def _page(behaviours):
    """A stand-in for the page: each version's code names how it behaves on each call."""

    async def run(code, tests, *, browser=None):
        if code not in behaviours:
            return [RunOfATest(t, False, "the library did not load: SyntaxError") for t in tests]
        runs = []
        for t in tests:
            got = behaviours[code].get(t.call, "threw TypeError")
            held = not got.startswith("threw") and same_value(json.loads(got), json.loads(t.expect))
            runs.append(RunOfATest(t, held, got))
        return runs
    return run


def _her_model(codes, worked):
    asked = []

    async def ask(prompt, schema, _tokens):
        asked.append(prompt)
        if "code" in schema.model_fields:
            return schema(code=codes[min(sum(1 for a in asked if "Write the JavaScript" in a), len(codes)) - 1])
        return schema(value=worked)
    return ask, asked


def test_an_expectation_the_versions_and_her_working_agree_against_is_doubted_not_written_toward():
    ask, asked = _her_model(["one", "two"], worked="2")
    built = asyncio.run(build_library(_PLAN, ask, run=_page({"one": _RIGHT, "two": _RIGHT})))
    assert built.doubted == ['Days.between("2024-02-28", "2024-03-01")']
    assert built.held == ["Days.leap(2024)"] and built.failed == [], "a doubted test is neither held nor failed"
    assert built.usable == ["between", "leap"] and built.tries == 2
    assert not any("Its tests showed" in a for a in asked), "nothing left to write toward"


def test_versions_agreeing_is_not_enough_without_her_own_working():
    ask, asked = _her_model(["one", "two", "three", "four"], worked="1")
    built = asyncio.run(build_library(_PLAN, ask, run=_page({c: _RIGHT for c in ("one", "two", "three", "four")})))
    assert built.doubted == [] and built.failed == ['Days.between("2024-02-28", "2024-03-01")']
    repair = [a for a in asked if "Its tests showed" in a]
    assert len(repair) == 2 and 'wanted 1, got 2' in repair[0]


def test_a_doubted_test_is_left_out_of_what_a_repair_is_shown():
    plan = _PLAN.model_copy(update={"tests": [*_PLAN.tests, LibraryTest(call="Days.leap(1900)", expect="false")]})
    half = {**_RIGHT, "Days.leap(1900)": "true"}
    ask, asked = _her_model(["one", "two", "fixed"], worked="2")
    built = asyncio.run(build_library(plan, ask, run=_page({"one": half, "two": half,
                                                            "fixed": {**_RIGHT, "Days.leap(1900)": "false"}})))
    repair = [a for a in asked if "Its tests showed" in a]
    assert len(repair) == 1 and "Days.leap(1900): wanted false, got true" in repair[0]
    assert "Days.between" not in repair[0].split("Its tests showed:")[1]
    assert built.held == ["Days.leap(2024)", "Days.leap(1900)"] and built.tries == 3


def _version(code, gots, expects):
    tests = [LibraryTest(call=f"X.f({i})", expect=e) for i, e in enumerate(expects)]
    return Version(code, [RunOfATest(t, g == t.expect, g) for t, g in zip(tests, gots, strict=True)])


def test_the_version_kept_is_from_the_group_that_behaves_alike_and_passes_most():
    expects = ["1", "2", "3"]
    alone = _version("alone", ["1", "2", "0"], expects)
    pair = [_version(c, ["1", "2", "9"], expects) for c in ("a", "b")]
    right = _version("right", ["1", "2", "3"], expects)
    assert the_best_of([alone, right]).code == "right"
    assert the_best_of([alone, *pair]).code == "a", "two that agree outweigh one alone that passes as much"
    unloaded = Version("broken", [RunOfATest(LibraryTest(call="X.f(0)", expect="1"), False, "the library did not load: x")])
    assert the_best_of([unloaded, alone]).code == "alone"


def test_a_value_is_agreed_only_by_two_versions_and_never_an_error():
    expects = ["5"]
    assert agreed_value([_version("a", ["4"], expects), _version("b", ["4"], expects)], 0) == "4"
    assert agreed_value([_version("a", ["4"], expects), _version("b", ["6"], expects)], 0) is None
    assert agreed_value([_version("a", ["threw x"], expects), _version("b", ["threw x"], expects)], 0) is None
    assert agreed_value([_version("a", ["5"], expects), _version("b", ["4"], expects), _version("c", ["4"], expects)], 0) is None


def _plan(name, uses=(), call=None):
    return LibraryPlan(name=name, purpose=f"the {name.lower()} part", uses=list(uses),
                       functions=[FunctionPlan(name="f", params=["x"], does="a value")],
                       tests=[LibraryTest(call=call or f"{name}.f(1)", expect="1")])


def test_a_library_is_built_after_what_it_uses_and_tested_with_it_loaded_beneath():
    from core.rebuilding.building_her_libraries import BuiltLibrary, beneath, in_order

    grid, cells, sheet = _plan("Grid"), _plan("Cells", ["Grid"]), _plan("Sheet", ["Cells", "Unknown"])
    assert [p.name for p in in_order([sheet, cells, grid])] == ["Grid", "Cells", "Sheet"]
    looped = [_plan("A", ["B"]), _plan("B", ["A"])]
    assert sorted(p.name for p in in_order(looped)) == ["A", "B"], "a circle of uses still builds each once"
    built = [BuiltLibrary(grid, "GRID;"), BuiltLibrary(cells, "CELLS;")]
    assert [b.plan.name for b in beneath(sheet, built)] == ["Grid", "Cells"], "what it uses, and what that uses, first"

    loaded = []

    async def run(code, tests, *, browser=None):
        loaded.append(code)
        return [RunOfATest(t, True, "1") for t in tests]

    ask, asked = _her_model(["SHEET;"], worked="")
    library = asyncio.run(build_library(sheet, ask, run=run, beneath=beneath(sheet, built)))
    assert loaded[0] == "GRID;\nCELLS;\nSHEET;" and library.code == "SHEET;"
    assert "It may call these libraries, already in the page, each function tested:\n- Grid.f(x)" in asked[0]
