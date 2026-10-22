"""A worker that dies stops the launch, and a worker on shared anchors recalls no recording.

On 21 October every worker of the carrier rehearsal went looking for a baseline
recording it had never made, a minute after the coordinator's anchors arrived,
and died. The launcher waited for its processes one at a time, so nothing
noticed, and the coordinator waited fourteen hours for their shards.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location("v25_sharded_under_test", REPO / "tools" / "run_subject_core_v25_sharded.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Child:
    def __init__(self, codes):
        self._codes = list(codes)
        self.terminated = False
        self.returncode = None

    def poll(self):
        if self.terminated:
            self.returncode = -15
        elif self._codes:
            self.returncode = self._codes.pop(0)
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.poll()


def test_a_worker_that_dies_stops_the_coordinator(tmp_path):
    coordinator = _Child([None] * 100)
    worker = _Child([None, 1])
    code = _launcher().supervise([coordinator, worker], ["coordinator", "worker_0"], tmp_path / "launcher.log", every=0.0)
    assert code == 1
    assert coordinator.terminated
    assert "worker_0" in (tmp_path / "launcher.log").read_text()


def test_a_launch_that_finishes_returns_nought(tmp_path):
    children = [_Child([None, 0]), _Child([0])]
    assert _launcher().supervise(children, ["coordinator", "worker_0"], tmp_path / "launcher.log", every=0.0) == 0


def test_a_worker_on_shared_anchors_does_not_recall_a_recording():
    source = (REPO / "tools" / "run_subject_core_v25.py").read_text()
    recall = source.index("from core.subject.recording import load_recording\n\n            recording = load_recording(run_dir)")
    guard = source.rfind("\n        el", 0, recall)
    assert source[guard:recall].strip().startswith("elif shared_bank is None:")
