"""The whole tree, counted by what each module decides."""
from __future__ import annotations

import pathlib

import pytest

from core.verify.how_much_of_it_governs import (
    THE_PRODUCTION_ROOTS,
    ACensus,
    how_much_of_it_governs,
    reached_by_name,
    what_no_way_in_reaches,
)


@pytest.fixture()
def a_small_tree(tmp_path: pathlib.Path) -> str:
    """Five modules, one of each state the census can report."""
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "core" / "reader.py").write_text("from core.read_by_someone import x\n")
    (tmp_path / "core" / "read_by_someone.py").write_text("x = 1\n")
    (tmp_path / "core" / "lonely.py").write_text("y = 1\n")
    (tmp_path / "core" / "loaded_by_name.py").write_text("z = 1\n")
    (tmp_path / "core" / "loader.py").write_text(
        'import importlib\n\nWHAT = "core.loaded_by_name"\n'
    )
    (tmp_path / "core" / "for_a_tool.py").write_text("w = 1\n")
    (tmp_path / "tools" / "inspect_it.py").write_text("import core.for_a_tool\n")
    (tmp_path / "tests" / "test_lonely.py").write_text("import core.lonely\n")
    return str(tmp_path)


def test_an_imported_module_governs(a_small_tree: str):
    census = how_much_of_it_governs(a_small_tree)
    assert census.by_state["governing"] == 1


def test_a_module_only_its_tests_import_is_a_proposal(a_small_tree: str):
    assert "core.lonely" in what_no_way_in_reaches(a_small_tree)


def test_a_module_only_a_tool_imports_is_reachable(a_small_tree: str):
    census = how_much_of_it_governs(a_small_tree)
    assert census.by_state.get("reachable") == 1
    assert "core.for_a_tool" not in census.proposals


def test_a_module_named_by_a_string_is_not_called_dead(a_small_tree: str):
    """The error this file exists to avoid: a live module counted as debt."""
    assert reached_by_name("core.loaded_by_name", a_small_tree) == (
        "a string in production code"
    )
    assert "core.loaded_by_name" not in what_no_way_in_reaches(a_small_tree)


def test_a_module_does_not_name_itself_into_being_reached(a_small_tree: str):
    """A string in the module's own source is not another module reaching it."""
    assert reached_by_name("core.loader", a_small_tree) == ""


def test_the_share_is_the_governing_count_over_every_module(a_small_tree: str):
    census = how_much_of_it_governs(a_small_tree)
    assert census.modules == 6
    assert census.share_that_governs == pytest.approx(census.governing / 6)


def test_a_proposal_is_filed_under_the_package_that_holds_it():
    made_up = ACensus(
        modules=1,
        by_state={"a proposal": 1},
        proposals=("core.brain.llm.a_thing",),
        by_package={"core.brain.llm": 1},
        how_each_is_reached_by_name={},
    )
    assert made_up.as_dict()["proposals_by_package"] == {"core.brain.llm": 1}


def test_the_production_roots_are_the_ones_layering_knows():
    """The same roots that carry a DEPS file, so the census covers the organism."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for top in THE_PRODUCTION_ROOTS:
        where = root / top
        if where.exists():
            assert where.is_dir()
    assert "core" in THE_PRODUCTION_ROOTS
    assert "tests" not in THE_PRODUCTION_ROOTS
    assert "tools" not in THE_PRODUCTION_ROOTS


def test_the_census_is_serialisable_and_names_its_totals(a_small_tree: str):
    read = how_much_of_it_governs(a_small_tree).as_dict()
    assert read["modules"] == sum(read["by_state"].values())
    assert read["proposals"] == read["by_state"].get("a proposal", 0)
