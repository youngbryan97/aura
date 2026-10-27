"""A content run split across processes by anchor reads the same samples one process would.

The coordinator takes the anchors; each worker presents every class from its
share of them through all three passes; the merge puts each class's rows back in
the order one process takes them, and refuses a set of anchors that is partial
or doubled. Every comparison the geometry makes is within one anchor, so this
splits the work and not the measurement.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.content_runtime import ClassSamples
from tools.run_subject_core_content import (
    _globally_keyed,
    _load_class_samples,
    _merge_samples,
    _save_class_samples,
    _shard_of,
)

pytestmark = pytest.mark.unit


def _rows(name: str, anchors: list[int], conditions: int = 2) -> ClassSamples:
    keys = [(a, c) for a in anchors for c in range(conditions)]
    return ClassSamples(
        name=name,
        futures=[np.full(3, 10.0 * a + c) for a, c in keys],
        retrieved=[frozenset({f"m{a}{c}"}) for a, c in keys],
        context=[np.full(2, float(a)) for a, _c in keys],
        futures_b=[np.full(3, 10.0 * a + c + 0.5) for a, c in keys],
        retrieved_b=[frozenset({f"m{a}{c}b"}) for a, c in keys],
        keys=keys,
    )


def test_the_shards_cover_every_anchor_once() -> None:
    shards = [_shard_of(f"{k}/3", 8) for k in range(3)]
    assert sorted(i for shard in shards for i in shard) == list(range(8))
    assert shards[0] == [0, 3, 6]
    with pytest.raises(SystemExit):
        _shard_of("3/3", 8)


def test_a_worker_names_each_anchor_by_its_place_in_the_bank() -> None:
    samples = {"joy": _rows("joy", [0, 1])}
    keyed = _globally_keyed(samples, [1, 4])
    assert keyed["joy"].keys == [(1, 0), (1, 1), (4, 0), (4, 1)]


def test_the_merge_puts_rows_back_in_one_process_order() -> None:
    first = {"joy": _rows("joy", [0, 2])}
    second = {"joy": _rows("joy", [1, 3])}
    merged = _merge_samples([second, first], anchors=4)["joy"]
    assert merged.keys == [(a, c) for a in range(4) for c in range(2)]
    assert [float(f[0]) for f in merged.futures] == [10.0 * a + c for a in range(4) for c in range(2)]
    assert merged.retrieved_b[0] == frozenset({"m00b"})
    assert [float(c[0]) for c in merged.context] == [float(a) for a in range(4) for _c in range(2)]


def test_the_merge_refuses_a_missing_or_doubled_anchor() -> None:
    with pytest.raises(SystemExit, match="missing anchors"):
        _merge_samples([{"joy": _rows("joy", [0, 2])}], anchors=4)
    with pytest.raises(SystemExit, match="two shards"):
        _merge_samples([{"joy": _rows("joy", [0, 1])}, {"joy": _rows("joy", [1])}], anchors=2)


def test_samples_survive_the_file_with_both_arms(tmp_path) -> None:
    classes = [SimpleNamespace(name="joy"), SimpleNamespace(name="fear")]
    stages = {
        "base": {"joy": _rows("joy", [0, 1]), "fear": _rows("fear", [0, 1])},
        "moved": {"joy": _rows("joy", [0, 1]), "fear": _rows("fear", [0, 1])},
    }
    path = _save_class_samples(tmp_path, classes, stages, "samples_0_of_1.npz")
    back = _load_class_samples(path, classes)
    assert set(back) == {"base", "moved"}
    joy = back["base"]["joy"]
    assert joy.keys == stages["base"]["joy"].keys
    assert joy.retrieved_b == stages["base"]["joy"].retrieved_b
    assert np.array_equal(np.vstack(joy.futures_b), np.vstack(stages["base"]["joy"].futures_b))


def test_a_worker_holds_the_coordinators_host_and_not_its_own() -> None:
    from core.subject.driver_host import _HeldHost
    from tools.run_subject_core_content import _hold_the_host

    class Body(_HeldHost):
        def __init__(self, cpu: float) -> None:
            self.state = SimpleNamespace(
                soma=SimpleNamespace(hardware={"cpu_usage": cpu, "temperature": 40.0}, latency={"last_thought_ms": cpu})
            )

        def _hold_observer(self) -> None:
            return None

    declared = _hold_the_host(Body(18.0))
    worker = Body(12.0)
    worker.freeze_host(declared)
    assert worker.frozen_host["cpu_usage"] == 18.0
    assert worker.frozen_latency["last_thought_ms"] == 18.0
