import time
from types import SimpleNamespace

import pytest

from core.agency_core import AgencyCore
from core.container import ServiceContainer
from core.volition import VolitionEngine


@pytest.fixture(autouse=True)
def clean_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()

@pytest.mark.asyncio
async def test_goal_completion_lifecycle_by_match(monkeypatch: pytest.MonkeyPatch):
    agency = AgencyCore(orchestrator=None)
    agency.state.pending_goals = []

    goal = {
        "id": "goal_123",
        "text": "Ensure Persistence (Uplink)",
        "priority": 0.8,
    }

    monkeypatch.setattr(agency, "_constitutional_runtime_live", lambda: False)
    added = agency.add_goal(goal)
    assert added is True
    assert len(agency.state.pending_goals) == 1
    assert agency.state.pending_goals[0]["status"] == "pending"

    success = agency.complete_goal_by_match({"id": "goal_123"})
    assert success is True
    assert agency.state.pending_goals[0]["status"] == "completed"

    agency.state.pending_goals[0]["status"] = "pending"

    success_text = agency.complete_goal_by_match({"text": "Ensure Persistence (Uplink)"})
    assert success_text is True
    assert agency.state.pending_goals[0]["status"] == "completed"

    agency.state.pending_goals[0]["status"] = "pending"
    action = {
        "type": "pursue_goal",
        "goal": {"id": "goal_123"},
    }

    await agency._commit_action_side_effects(action, time.time())
    assert agency.state.pending_goals[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_volition_engine_cooldown_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    orchestrator = SimpleNamespace(
        status=SimpleNamespace(running=True),
        cognitive_engine=SimpleNamespace(),
    )

    test_config = SimpleNamespace(
        paths=SimpleNamespace(
            brain_dir=tmp_path / "brain",
            data_dir=tmp_path / "data",
        )
    )
    monkeypatch.setattr("core.volition.config", test_config)

    engine = VolitionEngine(orchestrator)

    assert hasattr(engine, "_goal_cooldowns")
    assert engine._goal_cooldowns == {}

    potential_goals = [
        {"objective": "Explore new pathways", "origin": "intrinsic_curiosity"},
        {"objective": "Verify system health", "origin": "intrinsic_duty"},
    ]

    selected = engine._select_and_parse_goal(potential_goals)
    assert selected is not None
    objective = selected["objective"]
    assert objective in engine._goal_cooldowns

    selected_again = engine._select_and_parse_goal(potential_goals)
    assert selected_again is not None
    assert selected_again["objective"] != objective
    assert selected_again["objective"] in engine._goal_cooldowns

    selected_third = engine._select_and_parse_goal(potential_goals)
    assert selected_third is None
