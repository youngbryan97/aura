"""A slow read before a probe's write cannot leave the write ungoverned.

On 22 September at 18:34 a sixty-second disk stall held a read of the action
log inside the probe's governed scope. A governance token lives thirty seconds,
so the write that followed was refused, and the seed-19 campaign, its content
run and two sweep processes died of the same refusal. Each write now opens its
own scope at the call, and a refused write is counted as the harness failing
rather than ending the run.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.governance_context import GovernanceViolationError, is_governed
from core.state.aura_state import AuraState
from core.subject.driver import SubjectRuntime

pytestmark = pytest.mark.unit


def _runtime(room: Path, action: str) -> SubjectRuntime:
    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime._scratch = room
    runtime.turn = 1
    runtime.forced_action = action
    runtime.state = AuraState.default()
    runtime.rng = random.Random(0)
    runtime.organs = SimpleNamespace(self_model=None)
    runtime.actor = "self"
    runtime.failures = {}
    runtime.failure_notes = {}
    runtime._through_the_intention_loop = lambda *args, **kwargs: None
    runtime._roll_the_log_if_long = lambda kind: False
    runtime._remember_outcome = lambda kind, ok: None
    runtime._expects_to_succeed = lambda kind: True
    return runtime


@pytest.mark.parametrize("action", ["append_log", "finish_task"])
def test_the_probe_reads_what_it_will_write_over_outside_any_governed_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    runtime = _runtime(tmp_path, action)
    (tmp_path / "actions.log").write_text("before\n")
    (tmp_path / "task.txt").write_text("step 1: before\n")
    governed_reads: list[str] = []
    real_read = Path.read_text

    def watched(self: Path, *args, **kwargs):
        if self.name in {"actions.log", "task.txt"} and is_governed():
            governed_reads.append(self.name)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", watched)
    runtime._act("keep the workshop in order")
    assert governed_reads == [], f"read inside a governed scope: {governed_reads}"


def test_a_refused_write_is_counted_and_the_run_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path, "append_log")
    from core.runtime import file_write_gateway

    gateway = file_write_gateway.get_file_write_gateway()

    def refuse(*args, **kwargs):
        raise GovernanceViolationError("file_write_gateway.write_text:subject_core.action_probe called outside governed context")

    monkeypatch.setattr(gateway, "write_text", refuse)
    runtime._act("keep the workshop in order")
    assert runtime.failures["action_probe.governance"] == 1
    assert "outside governed context" in runtime.failure_notes["action_probe.governance"]
