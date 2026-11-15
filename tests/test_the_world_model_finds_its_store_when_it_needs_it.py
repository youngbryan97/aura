"""The world model asks for its store's path when it writes, not when it is imported.

A path computed at import holds whatever state root was current then. A run
isolates its root after its imports, so on 22 October the content run, the
paired-synergy probe and the hot-stove test each refused to start:
`state_leaks()` found the world model's directory still pointing into the shared
root, and a run writing there would have written into hers.
"""

from __future__ import annotations

from pathlib import PurePath

import pytest

import core.world_model.learned_world_model as world_model

pytestmark = pytest.mark.unit


def test_no_module_level_path_into_the_state_root():
    held = [
        name for name, value in vars(world_model).items()
        if isinstance(value, PurePath)
    ]
    assert not held, held


def test_the_path_follows_the_root_it_is_asked_under(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "one"))
    first = world_model._model_path()
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "two"))
    second = world_model._model_path()
    assert first != second
    assert str(tmp_path / "two") in str(second)


def test_save_and_load_ask_for_it_each_time(monkeypatch, tmp_path):
    """A model saved under one root and then another is written under each, and loads from the one it is under."""
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "one"))
    model = world_model.LearnedWorldModel()
    model._step_count = 7
    assert model.save()
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / "two"))
    model._step_count = 9
    assert model.save()
    assert world_model._model_path().exists() and str(tmp_path / "two") in str(world_model._model_path())
    for root, steps in (("one", 7), ("two", 9)):
        monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path / root))
        again = world_model.LearnedWorldModel()
        again._load()
        assert again._step_count == steps, root


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
