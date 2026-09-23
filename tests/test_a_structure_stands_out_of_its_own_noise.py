"""The content run refuses a geometry only when the geometry as a whole is noise.

It used to refuse whenever the closest of all pairs of classes was no further
apart than the noisiest class was from itself: the minimum of 190 distances
against the maximum of 20 floors. That refuses any structure in which two
classes are alike, which is what the bridge's gauge cells are for
(core/subject/bridge.py `orbits`). The rule now asks whether the distances
between classes, as a set, exceed the floors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("run_subject_core_content", REPO / "tools" / "run_subject_core_content.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence(between: list[float], floors: list[float]) -> dict:
    classes = [{"name": f"c{i}"} for i in range(len(floors))]
    pairs = {f"{i}-{i + 1 + k}": value for k, value in enumerate(between) for i in [0]}
    return {
        "classes": classes,
        "agreement": {"measured": True},
        "moves_together": {"measured": True},
        "behavioural_coverage": {c["name"]: 1.0 for c in classes},
        "internal": pairs,
        "internal_floor": {f"{i}-{i}": value for i, value in enumerate(floors)},
    }


def test_a_structure_with_two_alike_classes_is_still_measured(tool) -> None:
    rng = np.random.default_rng(1)
    floors = list(rng.uniform(0.05, 0.15, 20))
    between = list(rng.uniform(0.4, 1.2, 189)) + [0.02]  # one pair closer than any floor
    blockers = tool._authority(_evidence(between, floors))["blockers"]
    assert not any("own noise" in b for b in blockers), blockers


def test_a_geometry_at_its_own_noise_is_refused(tool) -> None:
    rng = np.random.default_rng(2)
    floors = list(rng.uniform(0.2, 0.5, 20))
    between = list(rng.uniform(0.2, 0.5, 190))
    blockers = tool._authority(_evidence(between, floors))["blockers"]
    assert any("own noise" in b for b in blockers), blockers


def test_too_few_floors_to_test_is_refused(tool) -> None:
    assert tool._out_of_noise([1.0, 2.0, 3.0], [0.1]) == (False, 1.0)
