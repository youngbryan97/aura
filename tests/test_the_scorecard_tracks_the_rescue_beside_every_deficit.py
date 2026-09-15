"""The scorecard tracks a rescue beside every lesion deficit it tracks.

A lesion's deficit was tracked across runs for irreducibility, spread and
synergy, and its rescue only for irreducibility. A rescue that brought the
partition back and left the other two where the cut put them read the same
across seeds as a complete one. P49.13 asks for rescue especially.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _scorecard():
    spec = importlib.util.spec_from_file_location("scorecard", REPO / "tools" / "subject_core_scorecard.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_tracked_deficit_has_its_rescue_tracked() -> None:
    tracked = dict(_scorecard().TRACKED)
    deficits = {path[-1] for name, path in tracked.items() if path[:2] == ("lesion", "deltas")}
    rescues = {path[-1] for name, path in tracked.items() if path[:2] == ("lesion", "rescue")}
    assert deficits, "no lesion deficit is tracked at all"
    assert deficits <= rescues, f"deficits with no rescue beside them: {sorted(deficits - rescues)}"
    assert ("lesion", "recovery_fraction") in tracked.values()
