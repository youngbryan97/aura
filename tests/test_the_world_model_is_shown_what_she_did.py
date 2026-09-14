"""The world model was shown that she never acted.

Each cycle the world model is given an action read off
`world.facts["last_action"]`: whether there was one, whether it was verified,
and whether she did it. Only the subject-core driver wrote that fact. The live
response path dispatched her skills and recorded a failure as a percept, and
wrote nothing the world model reads, so what she did could not change what it
predicted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.phases.action_grounding import GroundingResult, remember_last_action
from core.world_model.observe_cycle import action_of

ROOT = Path(__file__).resolve().parents[1]


def _result(status: str) -> GroundingResult:
    return GroundingResult(
        grounded_text="",
        marker_hits=[{"skill": "write_notes", "status": status, "ok": status == "executed"}],
    )


def _state(world: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(world=world, cognition=SimpleNamespace(active_goals=[{"goal": "finish"}]))


def test_an_executed_skill_is_shown_to_the_world_model_as_hers() -> None:
    world = SimpleNamespace(facts={})
    record = remember_last_action(world, _result("executed"))
    assert record is not None and world.facts["last_action"] is record
    assert list(action_of(_state(world))) == [1.0, 1.0, 1.0, 1.0]


def test_a_failed_dispatch_is_an_action_that_did_not_verify() -> None:
    world = SimpleNamespace(facts={})
    remember_last_action(world, _result("executed_failed"))
    assert world.facts["last_action"]["verified"] is False
    assert world.facts["last_action"]["outcome"] == "failed"
    assert list(action_of(_state(world)))[:2] == [1.0, 0.0]


def test_a_marker_that_was_never_dispatched_is_not_an_action() -> None:
    world = SimpleNamespace(facts={})
    assert remember_last_action(world, _result("unverified")) is None
    assert "last_action" not in world.facts


def test_the_live_response_path_writes_it() -> None:
    source = (ROOT / "core/phases/response_generation_unitary.py").read_text(encoding="utf-8")
    perceived = source.index("perceive_failed_actions(new_state.world, grounding_res)")
    remembered = source.index("remember_last_action(new_state.world, grounding_res)")
    assert 0 < remembered - perceived < 200
