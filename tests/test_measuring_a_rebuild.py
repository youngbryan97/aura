"""The programs a rebuild is measured on are fixed, and split once by seed into a set to work against and a set only
ever counted; what one build came to is read from the build itself."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_the_sets_are_fixed_by_seed_and_share_nothing():
    from tools.measure_rebuilding import HELD_OUT, PROGRAMS, split

    first, again = split(), split()
    assert first == again and len(first["held_out"]) == HELD_OUT
    assert not set(first["held_out"]) & set(first["development"])
    assert sorted(first["held_out"] + first["development"]) == sorted(PROGRAMS) and len(set(PROGRAMS)) == len(PROGRAMS)


def test_what_a_build_came_to_is_read_from_it(tmp_path):
    from tools.measure_rebuilding import measured

    (tmp_path / "libraries.json").write_text(json.dumps([{
        "plan": {"name": "Raster", "tests": [{}, {}, {}]}, "held": ["a", "b"], "failed": ["c"], "tries": 2,
        "from_her_store": ""}]))
    outcomes = [SimpleNamespace(kept=True), SimpleNamespace(kept=False), SimpleNamespace(kept=True)]
    rebuilt = SimpleNamespace(program="Pail", built=SimpleNamespace(outcomes=outcomes), why_not="")
    row = measured(rebuilt, 600.0, tmp_path)
    assert (row["features"], row["working"], row["minutes"]) == (3, 2, 10.0)
    assert row["libraries"] == [{"name": "Raster", "tests": 3, "held": 2, "from_her_store": ""}]
