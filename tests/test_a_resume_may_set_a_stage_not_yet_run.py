"""A resumed campaign may change the settings of a stage that has not run, and nothing else."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_subject_core.py"
_spec = importlib.util.spec_from_file_location("run_subject_core_for_test", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_the_lesion_settings_may_change_while_the_lesion_has_not_run() -> None:
    assert tool._moved_only_ahead(["lesion"], ("record", "interventions", "pci", "agency"))


def test_not_once_the_lesion_has_run() -> None:
    assert not tool._moved_only_ahead(["lesion"], tool.RESUMABLE)


def test_nothing_measured_before_may_change() -> None:
    before = ("record", "interventions", "pci", "agency")
    assert not tool._moved_only_ahead(["intervention"], before)
    assert not tool._moved_only_ahead(["lesion", "recording"], before)
    assert not tool._moved_only_ahead([], before)
