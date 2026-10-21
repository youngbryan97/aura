"""Two snapshots are compared number by number, so what one experience changed is written out."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.subject.snapshot_diff import diff_snapshots


@dataclass
class _Held:
    services: dict = field(default_factory=dict)
    turn: int = 0


def test_only_what_moved_is_named_and_the_largest_move_comes_first():
    a = _Held(services={"memory": {"trace": np.ones(8), "count": 3.0}, "body": {"heat": 0.5}}, turn=4)
    b = _Held(services={"memory": {"trace": np.ones(8) * 1.5, "count": 3.0}, "body": {"heat": 0.5}}, turn=4)
    diff = diff_snapshots(a, b)
    assert [row["path"] for row in diff["largest"]] == ["services.memory.trace"]
    assert diff["largest"][0]["relative_change"] == 0.5
    assert diff["numbers_changed"] == 1 and diff["numbers_compared"] == 4


def test_identical_snapshots_differ_nowhere():
    a = _Held(services={"x": {"y": np.arange(5.0)}})
    assert diff_snapshots(a, _Held(services={"x": {"y": np.arange(5.0)}}))["numbers_changed"] == 0
