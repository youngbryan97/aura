"""A run that has asked to shut down records nothing more.

The shutdown latch changes her: two of four moments rolled from their own
snapshots on seed 79 moved under it. A stage, a row or a report measured under
it is a measurement of something else, and seed 73's first campaign went on
measuring under it for seven hours before a git call at the end noticed
(26 October). What was recorded before the latch stays as it was.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runtime.shutdown_coordinator import request_shutdown, reset_shutdown_coordinator
from core.subject.latch import latch_reason, refuse_if_latched

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def latch(tmp_path, monkeypatch):
    """Sets the latch on demand, with the grace flag kept out of the real home."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reset_shutdown_coordinator()
    yield lambda: request_shutdown("sync_in_process_model_lane_fence_lost:test")
    reset_shutdown_coordinator()


def _carrier_tool():
    spec = importlib.util.spec_from_file_location(
        "_latched_run_carrier", ROOT / "tools" / "run_subject_core_v25.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unlatched_run_records(latch) -> None:
    assert latch_reason() is None
    refuse_if_latched("the lesion stage")


def test_a_latched_run_refuses_and_says_why(latch) -> None:
    latch()
    with pytest.raises(SystemExit, match="fence_lost:test"):
        refuse_if_latched("the lesion stage")


def test_the_campaign_keeps_the_checkpoint_it_had(latch, tmp_path) -> None:
    from tools.run_subject_core import _checkpoint

    args = SimpleNamespace(out=tmp_path / "run")
    args.out.mkdir()
    _checkpoint(args, "pci", {"pci": {"measured": True}})
    latch()
    with pytest.raises(SystemExit, match="the agency stage"):
        _checkpoint(args, "agency", {"pci": {"measured": True}, "agency": {}})
    assert json.loads((args.out / "checkpoint.json").read_text())["stage"] == "pci"


def test_a_carrier_shard_is_not_written_under_the_latch(latch, tmp_path) -> None:
    carrier = _carrier_tool()
    carrier._write_shard(tmp_path / "shards", (0, 3), {"cuts": []})
    latch()
    with pytest.raises(SystemExit, match="shard 1 of 3"):
        carrier._write_shard(tmp_path / "shards", (1, 3), {"cuts": []})
    assert sorted(path.name for path in (tmp_path / "shards").iterdir()) == ["shard_0_of_3.json"]
