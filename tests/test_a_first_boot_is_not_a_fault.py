"""A shard that has not been written yet is a first boot.

Every fresh state root opened a degradation and a resilience incident before
anything had gone wrong, because the loader read a file that does not exist on
a first boot and recorded the FileNotFoundError as a failure. That is the same
shape as reporting a missing continuity record as a failure to read one, and
the campaign log carried both.
"""

from __future__ import annotations

import pytest


def _horcrux(tmp_path, monkeypatch, noted: list[str]):
    from core.memory import horcrux as module

    monkeypatch.setattr(
        module, "record_degradation", lambda subsystem, exc, **kw: noted.append(subsystem)
    )
    instance = module.HorcruxManager.__new__(module.HorcruxManager)
    instance.aura_dir = str(tmp_path)
    return instance


def test_a_missing_shard_on_a_fresh_root_records_nothing(tmp_path, monkeypatch) -> None:
    noted: list[str] = []
    instance = _horcrux(tmp_path, monkeypatch, noted)
    assert instance._load_file_sync() == (None, None)
    assert noted == [], "a first boot opened a degradation"


def test_a_missing_hint_on_a_fresh_root_records_nothing(tmp_path, monkeypatch) -> None:
    noted: list[str] = []
    instance = _horcrux(tmp_path, monkeypatch, noted)
    assert instance._load_hint_sync("anything") == (None, None)
    assert noted == []


def test_a_shard_that_is_there_and_corrupt_is_still_a_fault(tmp_path, monkeypatch) -> None:
    """The guard must not swallow the failure it was written for."""
    noted: list[str] = []
    instance = _horcrux(tmp_path, monkeypatch, noted)
    (tmp_path / ".core_seed").write_text("not base64 at all !!!")
    assert instance._load_file_sync() == (None, None)
    assert noted == ["horcrux"]


def test_a_hint_that_is_there_and_corrupt_is_still_a_fault(tmp_path, monkeypatch) -> None:
    noted: list[str] = []
    instance = _horcrux(tmp_path, monkeypatch, noted)
    (tmp_path / ".hint_seed").write_text("also not base64 !!!")
    assert instance._load_hint_sync("anything") == (None, None)
    assert noted == ["horcrux"]
