"""What she does changes a real environment, and what changed comes back as perception.

Phase 25 asks for the loop to close through the world: she acts in a sandboxed
real environment, the environment changes, a real sensor detects it,
perception changes her state, and the change reaches later action. The action
arm writes to a scratch room through the governed file gateway and reads the
outcome back off the filesystem. These pin the pieces a unit can pin: the room
really changes, what is perceived is what the filesystem holds, and a later
action's outcome depends on what an earlier one left behind. The domain-level
half, perception reaching deliberation and back through interventionally kept
edges, is read off the battery's reports.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.state.aura_state import AuraState
from core.state.percepts import read_percept
from core.subject.driver import SubjectRuntime

pytestmark = pytest.mark.unit


def _runtime(room: Path) -> SubjectRuntime:
    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime._scratch = room
    runtime.turn = 1
    runtime.forced_action = None
    runtime.state = AuraState.default()
    runtime.rng = random.Random(0)
    runtime.organs = SimpleNamespace(self_model=None)
    runtime.actor = "self"
    runtime._through_the_intention_loop = lambda *args, **kwargs: None
    runtime._roll_the_log_if_long = lambda kind: False
    runtime._remember_outcome = lambda kind, ok: None
    runtime._expects_to_succeed = lambda kind: True
    return runtime


def _latest_from_the_filesystem(runtime: SubjectRuntime) -> str:
    readings = [read_percept(item) for item in runtime.state.world.recent_percepts]
    sensed = [reading for reading in readings if reading.raw.get("source") == "filesystem"]
    assert sensed, "the action left nothing for her to perceive"
    return sensed[-1].content


def test_acting_changes_the_room_and_she_perceives_what_it_now_holds(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.forced_action = "make_room"
    runtime._act("keep the workshop in order")
    assert (tmp_path / "room_0001").is_dir()
    assert _latest_from_the_filesystem(runtime) == "1 rooms exist"

    runtime.turn = 2
    runtime._act("keep the workshop in order")
    assert (tmp_path / "room_0002").is_dir()
    assert _latest_from_the_filesystem(runtime) == "2 rooms exist"
    assert runtime.last_action["verified"] is True


def test_what_an_earlier_action_left_behind_decides_a_later_one(tmp_path: Path) -> None:
    made = _runtime(tmp_path / "made")
    (tmp_path / "made").mkdir()
    made.forced_action = "make_room"
    made._act("keep the workshop in order")
    made.turn = 2
    made.forced_action = "read_room"
    made._act("keep the workshop in order")
    assert made.last_action["verified"] is True

    skipped = _runtime(tmp_path / "skipped")
    (tmp_path / "skipped").mkdir()
    skipped.forced_action = "append_log"
    skipped._act("keep the workshop in order")
    skipped.turn = 2
    skipped.forced_action = "read_room"
    skipped._act("keep the workshop in order")
    assert skipped.last_action["verified"] is False
    assert _latest_from_the_filesystem(skipped) == "there is no room room_0001"
