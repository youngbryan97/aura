"""A lane holder that has died reads as dead, though the run holds its host still.

LIVE 2026-10-30, seed 101's reports run at aeefdca31: memory pressure ended her cortex's worker at 17:55. The run's
declared host (core/subject/declared_host.py) answers each question once and keeps the answer, so two arms read one
body, and the lane controller asked it whether the worker was still running. It kept answering yes. The ledger marked
the dead worker's heartbeat stale and the holder not evictable, and every reload of her cortex after 17:55 was refused
with `required_eviction_not_preemptible:in-process-sync:75676:embedding-engine:...`, every fifteen minutes. Read from
outside the run, the same ledger called the worker dead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.runtime.model_lane_control import ModelLaneController, ProcessLiveness
from core.subject.declared_host import _HeldObserver

pytestmark = pytest.mark.unit

PID = 75676
STARTED = 1793403323.168395


class _Machine:
    """A machine on which one process is running until it is not."""

    provenance = SimpleNamespace(source=SimpleNamespace(value="test"), scenario_id="lane-holder")

    def __init__(self) -> None:
        self.running = {PID: STARTED}

    def process(self, pid: int):
        started = self.running.get(int(pid))
        if started is None:
            return None
        return SimpleNamespace(pid=int(pid), ppid=1, create_time=started, status="running")

    def processes(self):
        return tuple(self.process(pid) for pid in self.running)


def _holder() -> dict:
    return {
        "process": {"pid": PID, "started_at": STARTED},
        "preemptible": True,
        "lease_expires_at": 0.0,
        "metadata": {"lease_mode": "heartbeat", "in_process_model_owner": True, "synchronous_loader": True},
    }


def _controller(tmp_path, machine: _Machine) -> ModelLaneController:
    return ModelLaneController(
        state_path=tmp_path / "model_lane_control.json",
        observer=_HeldObserver(machine),
        process_discovery=None,
    )


def test_the_held_face_keeps_its_first_answer_for_her(tmp_path):
    machine = _Machine()
    held = _HeldObserver(machine)
    assert held.process(PID) is not None
    machine.running.clear()
    assert held.process(PID) is not None  # what she reads stays still between arms, as it must


def test_the_lane_asks_the_machine_whether_its_holder_is_running(tmp_path):
    machine = _Machine()
    controller = _controller(tmp_path, machine)
    assert controller._owner_process_tree_liveness(_holder()) is ProcessLiveness.ALIVE
    machine.running.clear()
    assert controller._owner_process_tree_liveness(_holder()) is ProcessLiveness.DEAD


def test_a_dead_holder_whose_heartbeat_lapsed_is_reaped_not_fenced(tmp_path):
    machine = _Machine()
    controller = _controller(tmp_path, machine)
    owner = f"in-process-sync:{PID}:embedding-engine:27500176848"
    state = controller._empty_state()
    state["owners"][owner] = _holder()
    controller._prune_locked(state, now=STARTED + 3600.0)  # alive, lease lapsed: fenced, as before
    assert state["owners"][owner]["preemptible"] is False
    machine.running.clear()
    controller._prune_locked(state, now=STARTED + 4500.0)
    assert owner not in state["owners"]
    assert any(e["event"] == "owner_reaped" and e.get("owner_id") == owner for e in state["events"])
