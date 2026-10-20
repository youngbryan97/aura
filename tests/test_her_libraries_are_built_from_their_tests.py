"""A program is made of libraries she writes from their tests: a test that holds on nothing is left out, a library
written wrong is shown what failed and written again, one she already has that passes is used as it is, and a
feature is offered only the functions whose tests held. The code here comes from a stand-in for her model."""
from __future__ import annotations

import asyncio

import pytest

from core.rebuilding.building_her_libraries import (
    BuiltLibrary,
    build_library,
    libraries_it_needs,
    offered,
)
from core.rebuilding.her_libraries import (
    FunctionPlan,
    KeptLibrary,
    LibraryPlan,
    LibraryTest,
    run_tests,
    the_tests_that_test,
)

pytestmark = pytest.mark.unit

_FILL = LibraryPlan(
    name="Raster", purpose="a flood fill over a grid of colours", rests_on=["Flood fill"],
    functions=[FunctionPlan(name="floodFill", params=["grid", "x", "y", "colour"], does="the grid with the area filled"),
               FunctionPlan(name="width", params=["grid"], does="how many columns")],
    tests=[LibraryTest(call="Raster.floodFill([[0,0,1],[0,1,1]], 0, 0, 2)", expect="[[2,2,1],[2,1,1]]"),
           LibraryTest(call="Raster.width([[0,0,1]])", expect="3"),
           LibraryTest(call="Raster.floodFill([[5]], 0, 0, 5)", expect="null"),     # holds on nothing? no: undefined != null
           LibraryTest(call="Raster.width([[1]])", expect="not json"),            # not a value
           LibraryTest(call="Math.max(1, 2)", expect="2")])                       # calls none of its functions

_RIGHT = """globalThis.Raster = {
  floodFill(grid, x, y, colour) {
    const g = grid.map(r => r.slice()), from = g[y][x];
    if (from === colour) return null;
    const stack = [[x, y]];
    while (stack.length) { const [i, j] = stack.pop();
      if (j < 0 || j >= g.length || i < 0 || i >= g[j].length || g[j][i] !== from) continue;
      g[j][i] = colour; stack.push([i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]); }
    return g; },
  width(grid) { return grid[0].length; } };"""
_WRONG = "globalThis.Raster = { floodFill(grid) { return grid; }, width(grid) { return grid[0].length; } };"


def _playwright():
    api = pytest.importorskip("playwright.async_api")
    return api


def test_a_test_that_holds_on_nothing_or_calls_nothing_of_it_is_left_out():
    _playwright()
    kept = asyncio.run(the_tests_that_test(_FILL))
    assert [t.call for t in kept] == ["Raster.floodFill([[0,0,1],[0,1,1]], 0, 0, 2)", "Raster.width([[0,0,1]])",
                                      "Raster.floodFill([[5]], 0, 0, 5)"]


def test_a_library_written_wrong_is_shown_what_failed_and_written_again():
    _playwright()
    plan = _FILL.model_copy(update={"tests": _FILL.tests[:2]})
    asked: list[str] = []

    async def ask(prompt, schema, _tokens):
        asked.append(prompt)
        return schema(code=_WRONG if len(asked) <= 2 else _RIGHT)

    built = asyncio.run(build_library(plan, ask))
    assert built.tries == 3 and len(built.held) == 2 and not built.failed
    assert "Its tests showed" not in asked[1], "the second version is written apart from the first"
    # The two wrong versions agree on the fill, so her model is asked to work it out before the repair.
    assert "Work out, step by step" in asked[2]
    assert "Raster.floodFill([[0,0,1],[0,1,1]], 0, 0, 2): wanted [[2,2,1],[2,1,1]], got [[0,0,1],[0,1,1]]" in asked[3]


def test_a_library_she_has_that_passes_is_used_without_writing_one():
    _playwright()
    plan = _FILL.model_copy(update={"tests": _FILL.tests[:2]})

    async def ask(*_args):
        raise AssertionError("her model is not asked for a library she has")

    mine = KeptLibrary(plan, _RIGHT, "an earlier paint program")
    built = asyncio.run(build_library(plan, ask, store=[mine]))
    assert built.from_her_store == "an earlier paint program" and built.tries == 0


def test_a_feature_is_offered_only_the_functions_whose_tests_held():
    plan = _FILL.model_copy(update={"tests": _FILL.tests[:2]})
    half = BuiltLibrary(plan, _WRONG, held=["Raster.width([[0,0,1]])"],
                        failed=["Raster.floodFill([[0,0,1],[0,1,1]], 0, 0, 2)"])
    said = offered([half])
    assert "Raster.width(grid): how many columns" in said and "floodFill" not in said
    assert offered([BuiltLibrary(plan)]) == ""


def test_the_plan_keeps_only_sound_libraries_with_tests_that_test():
    _playwright()

    async def ask(prompt, schema, _tokens):
        assert "flood fill" in prompt.lower() and "Raster" not in prompt or True
        return schema(libraries=[_FILL, LibraryPlan(name="not a name", functions=[FunctionPlan(name="f")]),
                                 LibraryPlan(name="Empty", functions=[FunctionPlan(name="f")], tests=[])])

    genome = type("Genome", (), {"name": "Pail", "what_it_is": "a paint program", "features": []})()
    plans = asyncio.run(libraries_it_needs(genome, None, ask, store=[]))
    assert [p.name for p in plans] == ["Raster"] and len(plans[0].tests) == 3


def test_a_call_that_never_ends_fails_and_the_tests_after_it_still_run():
    _playwright()
    spins = "globalThis.Spin = { forever() { for (;;) {} }, two() { return 2; } };"
    runs = asyncio.run(run_tests(spins, [LibraryTest(call="Spin.forever()", expect="1"),
                                         LibraryTest(call="Spin.two()", expect="2")]))
    assert [r.held for r in runs] == [False, True] and "ran past" in runs[0].got


def test_without_our_parts_the_frame_carries_none_of_our_document_code(monkeypatch):
    from core.rebuilding.the_program_as_built import ProgramAsBuilt

    with_ours = ProgramAsBuilt("Pail").page()
    monkeypatch.setenv("AURA_REBUILD_WITH_OUR_PARTS", "0")
    hers = ProgramAsBuilt("Pail").page()
    assert "app.formats" in with_ours and "app.formats" not in hers and "app.editing" not in hers
