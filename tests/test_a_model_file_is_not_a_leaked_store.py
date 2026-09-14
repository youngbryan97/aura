"""An asset is not a store, and a run must not refuse over where a model lives.

`state_leaks` exists to catch a module that kept a path into the root a run
moved away from, because such a module goes on writing where it started. Model
weights are the other thing: read at load time, byte-identical in every arm,
and tens of gigabytes, so a fork cannot move them and a run that refuses over
them is refusing over the location of a file nobody writes.

The v25 carrier run refused for exactly that reason and got no further than
building the organism.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from core.subject import isolation


@pytest.fixture
def _root(tmp_path, monkeypatch):
    default = tmp_path / "shared"
    (default / "models").mkdir(parents=True)
    own = tmp_path / "run_001" / "state"
    own.mkdir(parents=True)
    monkeypatch.setattr(isolation, "_DEFAULT_ROOT", str(default))
    monkeypatch.setenv("AURA_STATE_ROOT", str(own))
    return default


def _module(name: str, **paths: Path) -> ModuleType:
    module = ModuleType(name)
    for attribute, value in paths.items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    return module


def test_a_store_left_in_the_old_root_is_still_a_leak(_root, monkeypatch) -> None:
    _module("core.testing_leak_store", LEDGER=_root / "data" / "ledger.db")
    monkeypatch.delitem(sys.modules, "core.testing_leak_store", raising=False)
    _module("core.testing_leak_store", LEDGER=_root / "data" / "ledger.db")
    try:
        leaks = isolation.state_leaks()
    finally:
        sys.modules.pop("core.testing_leak_store", None)
    assert any("testing_leak_store.LEDGER" in leak for leak in leaks)


def test_a_model_path_named_as_an_asset_is_not_a_leak(_root) -> None:
    name = next(iter(isolation.NOT_STATE))
    module_name, _, attribute = name.rpartition(".")
    original = sys.modules.get(module_name)
    _module(module_name, **{attribute: _root / "models" / "a-model"})
    try:
        leaks = isolation.state_leaks()
    finally:
        if original is not None:
            sys.modules[module_name] = original
        else:
            sys.modules.pop(module_name, None)
    assert not any(name in leak for leak in leaks)


def test_every_named_asset_is_a_module_and_an_attribute() -> None:
    for name in isolation.NOT_STATE:
        module_name, _, attribute = name.rpartition(".")
        assert module_name.startswith("core."), name
        assert attribute, name


def test_the_model_paths_this_repository_has_are_the_ones_named() -> None:
    """A renamed global would silently start refusing runs again."""
    from core.brain.llm import model_paths, model_registry

    for name in isolation.NOT_STATE:
        module_name, _, attribute = name.rpartition(".")
        module = model_paths if module_name.endswith("model_paths") else model_registry
        assert hasattr(module, attribute), name
