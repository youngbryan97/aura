import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.consciousness.executive_authority import ExecutiveAuthority
from core.container import ServiceContainer
from core.state.aura_state import AuraState


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_no_direct_pending_initiative_appenders_remain():
    targets = [
        PROJECT_ROOT / "core" / "phases" / "initiative_generation.py",
        PROJECT_ROOT / "core" / "phases" / "motivation_update.py",
        PROJECT_ROOT / "core" / "phases" / "memory_consolidation.py",
        PROJECT_ROOT / "core" / "mind_tick.py",
        PROJECT_ROOT / "core" / "consciousness" / "executive_closure.py",
        PROJECT_ROOT / "core" / "autonomy" / "research_cycle.py",
        PROJECT_ROOT / "core" / "kernel" / "upgrades_10x.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "pending_initiatives.append(" not in text, f"direct initiative append still present in {path.name}"
        if path.name == "research_cycle.py":
            assert "state.cognition.pending_initiatives =" not in text, "research cycle still mutates initiative queue directly"


def test_active_seeders_route_through_proposal_governance_helper():
    targets = [
        PROJECT_ROOT / "core" / "phases" / "initiative_generation.py",
        PROJECT_ROOT / "core" / "phases" / "motivation_update.py",
        PROJECT_ROOT / "core" / "consciousness" / "executive_closure.py",
        PROJECT_ROOT / "core" / "kernel" / "upgrades_10x.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "propose_governed_initiative_to_state(" in text, f"{path.name} bypasses proposal governance"
        assert "get_executive_authority().propose_initiative_to_state(" not in text, f"{path.name} still calls executive authority directly"


def test_background_seeders_do_not_bind_objectives_directly():
    text = (PROJECT_ROOT / "core" / "kernel" / "upgrades_10x.py").read_text(encoding="utf-8")
    assert 'state.cognition.current_objective = (' not in text
    assert "state.cognition.current_objective = transcript" not in text


def test_kernel_pending_initiative_consumer_is_retired():
    text = (PROJECT_ROOT / "core" / "kernel" / "aura_kernel.py").read_text(encoding="utf-8")
    assert "await self._dispatch_pending_initiatives()" not in text
    assert "Retired compatibility hook." in text


@pytest.mark.asyncio
async def test_executive_authority_promotes_and_completes_initiatives():
    state = AuraState()
    authority = ExecutiveAuthority(SimpleNamespace())
    goal_engine_calls = {"add": [], "update": []}

    class _FakeGoalEngine:
        async def add_goal(self, name, objective=None, **kwargs):
            goal_engine_calls["add"].append((name, objective, kwargs))
            return {"id": "goal-1", "status": kwargs.get("status", "queued")}

        def get_goal(self, goal_id):
            assert goal_id == "goal-1"
            return {"id": goal_id, "status": "in_progress"}

        async def update_goal_status(self, goal_id, **kwargs):
            goal_engine_calls["update"].append((goal_id, kwargs))
            return {"id": goal_id, **kwargs}

    original_get = ServiceContainer.get
    ServiceContainer.get = staticmethod(
        lambda name, default=None: _FakeGoalEngine() if name == "goal_engine" else original_get(name, default)
    )

    try:
        state, _ = await authority.propose_initiative_to_state(
            state,
            "Investigate runtime drift",
            source="initiative_generation",
            urgency=0.7,
            triggered_by="curiosity",
        )
        state, _ = await authority.propose_initiative_to_state(
            state,
            "Stabilize thermal load",
            source="executive_closure",
            urgency=0.9,
            triggered_by="stability",
        )

        promoted_state, initiative, decision = await authority.promote_next_initiative(state, source="mind_tick")

        assert decision["action"] == "promoted"
        assert initiative is not None
        assert promoted_state.cognition.current_objective == "Stabilize thermal load"
        assert promoted_state.cognition.current_origin in {"executive_closure", "mind_tick"}
        assert len(promoted_state.cognition.pending_initiatives) == 1
        assert promoted_state.cognition.modifiers["current_objective_binding"]["goal_id"] == "goal-1"

        held_state, held = await authority.complete_current_objective(
            promoted_state,
            reason="tick_cycle_complete",
            source="mind_tick",
        )

        assert held["action"] == "held"
        assert held_state.cognition.current_objective == "Stabilize thermal load"

        completed_state, completion = await authority.complete_current_objective(
            promoted_state,
            reason="goal_completed",
            source="mind_tick",
        )

        assert completion["action"] == "completed"
        assert completed_state.cognition.current_objective is None
        assert all(item.get("goal") != "Stabilize thermal load" for item in completed_state.cognition.pending_initiatives)
        assert goal_engine_calls["add"]
        assert goal_engine_calls["update"][0][0] == "goal-1"
        assert goal_engine_calls["update"][0][1]["status"] == "completed"
    finally:
        ServiceContainer.get = original_get
