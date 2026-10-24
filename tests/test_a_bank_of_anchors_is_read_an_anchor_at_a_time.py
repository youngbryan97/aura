"""A bank of anchors is read an anchor at a time, and restores what the whole bank did.

At 370 rounds an anchor holds about 170 MB of her state decoded, and a worker
that decoded all 128 held some 22 GB; on 23 October the carrier's three
processes came to 57 GB and free memory fell to 70 MB. The bank now writes
what every snapshot holds by reference once and each anchor's own state apart,
and a process decodes an anchor when it restores it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pytest

from core.subject.anchor_bank import AnchorBank, write_anchor_bank
from core.subject.anchor_travel import _Writer, dumps


@dataclass
class _Snapshot:
    module_state: dict
    organs: dict
    stores: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Anchor:
    snapshot: object
    current: np.ndarray
    history: np.ndarray
    source_condition: str


class _Module:
    """A module-level object every snapshot holds by reference."""

    def __init__(self, weights):
        self.weights = weights


def _anchors(count: int = 6):
    held = _Module(np.arange(1000.0))
    rng = np.random.default_rng(0)
    out = []
    for index in range(count):
        mine = {"trace": rng.normal(size=50).tolist()}
        snapshot = _Snapshot(
            module_state={"pkg.mod:_instance": (held, {"count": index})},
            organs={"mesh": mine, "mesh_again": mine},
        )
        out.append(_Anchor(snapshot, rng.normal(size=4), rng.normal(size=8), f"condition_{index % 3}"))
    return out, held


def _same_content(left, right) -> bool:
    return pickle.dumps(left.organs) == pickle.dumps(right.organs) and \
        left.module_state["pkg.mod:_instance"][1] == right.module_state["pkg.mod:_instance"][1]


def test_each_anchor_restores_what_it_was(tmp_path):
    anchors, _held = _anchors()
    path = tmp_path / "anchor_bank.bin"
    write_anchor_bank(path, anchors=anchors, doses={"A": 0.1}, scale=np.ones(3))
    bank = AnchorBank(path, keep=2)
    read = bank.anchors()
    assert len(read) == len(anchors)
    assert bank.fields["doses"] == {"A": 0.1} and np.array_equal(bank.fields["scale"], np.ones(3))
    for original, banked in zip(anchors, read, strict=True):
        assert np.array_equal(original.current, banked.current)
        assert np.array_equal(original.history, banked.history)
        assert original.source_condition == banked.source_condition
        assert _same_content(original.snapshot, banked.snapshot)


def test_what_every_anchor_holds_by_reference_is_written_once_and_read_as_one(tmp_path):
    anchors, _held = _anchors()
    path = tmp_path / "anchor_bank.bin"
    write_anchor_bank(path, anchors=anchors)
    alone = len(dumps(anchors[0].snapshot))
    assert path.stat().st_size < alone * len(anchors)
    bank = AnchorBank(path, keep=len(anchors))
    first, last = bank.anchors()[0].snapshot, bank.anchors()[-1].snapshot
    assert first.module_state["pkg.mod:_instance"][0] is last.module_state["pkg.mod:_instance"][0]
    assert first.organs["mesh"] is first.organs["mesh_again"], "an anchor's own sharing is kept"
    assert first.organs["mesh"] is not last.organs["mesh"]


def test_an_anchor_is_decoded_when_restored_and_the_last_few_are_kept(tmp_path):
    anchors, _held = _anchors()
    path = tmp_path / "anchor_bank.bin"
    write_anchor_bank(path, anchors=anchors)
    bank = AnchorBank(path, keep=2)
    read = bank.anchors()
    assert bank.decodes == 0
    _ = read[0].snapshot, read[0].snapshot, read[1].snapshot, read[0].snapshot
    assert bank.decodes == 2
    _ = read[2].snapshot  # pushes out anchor 1, the least recently restored
    _ = read[0].snapshot
    assert bank.decodes == 3
    _ = read[1].snapshot
    assert bank.decodes == 4


def test_a_reader_fills_the_snapshot_from_its_own(tmp_path):
    anchors, _held = _anchors()
    path = tmp_path / "anchor_bank.bin"
    write_anchor_bank(path, anchors=anchors)
    own_module = _Module(np.zeros(3))
    own = _Snapshot(module_state={"pkg.mod:_instance": (own_module, {"count": -1})}, organs={})
    restored = AnchorBank(path, own=own, keep=1).anchors()[3].snapshot
    held, captured = restored.module_state["pkg.mod:_instance"]
    assert held is own_module, "the reader's own object is kept"
    assert captured == {"count": 3}, "the writer's copy of what it held is what is restored"


def test_a_file_in_the_old_format_is_not_read_as_this_one(tmp_path):
    path = tmp_path / "anchor_bank.bin"
    buffer = open(path, "wb")
    _Writer(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump({"anchors": []})
    buffer.close()
    assert AnchorBank.is_one(path) is False
    with pytest.raises(ValueError, match="shared-section format"):
        AnchorBank(path)


def _carrier_tool():
    import importlib.util
    import sys
    from pathlib import Path

    name = "v25_bank_under_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "tools" / "run_subject_core_v25.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_a_process_keeps_as_many_anchors_as_a_cuts_second_look_draws():
    tool = _carrier_tool()
    assert tool._anchors_kept({"looks": [8, 16, 32, 64, 96, 128]}, 6, 128) == 16
    assert tool._anchors_kept({"looks": [8, 16, 32]}, 3, 10) == 10
    assert tool._anchors_kept({"looks": [8]}, 1, 128) == 8


def test_an_array_large_enough_to_travel_as_a_buffer_is_written_and_restored(tmp_path):
    """Protocol 5 hands a large array to the writer as a PickleBuffer, which has no length.

    The bank of 23 October failed on exactly that, its first real snapshot, on 24 October.
    """
    large = _Module(np.arange(1_000_000.0))
    rng = np.random.default_rng(1)
    anchors = [
        _Anchor(
            _Snapshot(module_state={"pkg.mod:_instance": (large, {"count": index})}, organs={"mesh": {"w": np.ones(200_000)}}),
            rng.normal(size=4), rng.normal(size=8), "condition_0",
        )
        for index in range(2)
    ]
    path = tmp_path / "bank.bin"
    assert write_anchor_bank(path, anchors=anchors, extra="x") > 0
    restored = AnchorBank(path, keep=2).anchors()[1].snapshot
    assert np.array_equal(restored.organs["mesh"]["w"], np.ones(200_000))
    assert np.array_equal(restored.module_state["pkg.mod:_instance"][0].weights, np.arange(1_000_000.0))
