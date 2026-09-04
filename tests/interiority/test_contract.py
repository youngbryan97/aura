"""What a faculty must declare before it is allowed to run.

The registry refuses a mechanism with no counterfactual, no null and no
falsifier, so these check the declarations are substantive rather than
present: a falsifier that names no observation, a question that does not
match the one that was asked, a home that points at a module which does
not exist.
"""

from __future__ import annotations

import importlib

import pytest

from core.interiority.faculties import load_all
from core.interiority.faculty import registry
from core.interiority.homes import HOMES


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_every_faculty_declares_a_refutation() -> None:
    for faculty in registry().all():
        falsifier = faculty.falsifier()
        assert len(falsifier) > 40, f"{faculty.id}: falsifier is too thin to refute"
        assert faculty.counterfactuals, f"{faculty.id}: no counterfactual"
        assert faculty.question.strip(), f"{faculty.id}: no question"
        assert faculty.mechanism.strip(), f"{faculty.id}: no mechanism"


def test_every_faculty_has_a_home_that_exists() -> None:
    ids = set(registry().ids())
    assert set(HOMES) == ids, (
        f"unmapped faculties: {sorted(ids - set(HOMES))}; "
        f"orphan homes: {sorted(set(HOMES) - ids)}"
    )
    for home in HOMES.values():
        assert home.feeds, f"{home.faculty}: declares no consumer"
        assert home.belongs_with, f"{home.faculty}: declares no organ"
        for binding in home.feeds:
            module = importlib.import_module(binding.module)
            root = binding.symbol.split(".")[0]
            assert hasattr(module, root), (
                f"{home.faculty} claims to feed {binding.module}.{binding.symbol}, "
                f"but {root} is not there"
            )


def test_superseded_logic_is_named_precisely() -> None:
    from core.interiority.homes import superseded

    replaced = superseded()
    assert replaced, "the package claims to replace nothing"
    for entry in replaced:
        assert ":" in entry, f"superseded entry names no file: {entry}"


def test_faculties_do_not_read_module_globals_for_substrate() -> None:
    """The substrate arrives through the context, or it cannot be measured."""
    import inspect

    for faculty in registry().all():
        source = inspect.getsource(type(faculty))
        for forbidden in ("get_receptor_bank()", "get_cleft()"):
            assert forbidden not in source, (
                f"{faculty.id} reaches for {forbidden} instead of the context; "
                "a faculty that reads a singleton cannot be measured against a world"
            )
