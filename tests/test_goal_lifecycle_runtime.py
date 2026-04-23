from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.goals.goal_engine import GoalEngine
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_goal_engine_persists_short_and_long_horizon_lifecycle(tmp_path):
    engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))

    await engine.add_goal(
        "Ship the strategic planner cleanup",
        horizon="long_term",
        priority=0.88,
        status="queued",
        success_criteria="Planner runtime is stable and durable.",
    )

    running_step = SimpleNamespace(tool="web_search")
    plan = SimpleNamespace(
        plan_id="plan-1",
        goal="Verify the live runtime pressure path",
        steps=[running_step, SimpleNamespace(tool="clock")],
        succeeded_steps=[running_step],
        trace_id="trace-1",
        status="running",
        final_result="",
        requires_approval=False,
        context={
            "task_id": "task-1",
            "source": "task_engine",
            "priority": 0.97,
            "horizon": "short_term",
            "quick_win": True,
        },
    )

    engine.sync_task_plan(plan)

    active = engine.get_active_goals(limit=10, include_external=False)
    assert any(item["task_id"] == "task-1" for item in active)
    assert any(item["horizon"] == "long_term" for item in active)

    updated = await engine.update_task_lifecycle(
        task_id="task-1",
        status="completed",
        summary="Verified under runtime pressure.",
    )

    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["progress"] == 1.0

    completed = engine.get_completed_goals(limit=10, include_external=False)
    assert any(item["task_id"] == "task-1" for item in completed)
    assert any("runtime pressure" in item["summary"].lower() for item in completed)

    settled = await engine.update_goal_status(
        updated["id"],
        status="completed",
        summary="Verified under runtime pressure and settled.",
    )
    assert settled is not None
    assert settled["completed_at"] is not None

    snapshot = engine.build_snapshot(limit=20, include_external=False)
    assert snapshot["summary"]["short_term_count"] >= 0
    assert snapshot["summary"]["long_term_count"] >= 1
    assert snapshot["summary"]["completed_count"] >= 1

    active_async = await engine.get_active_goals_async(limit=10, include_external=False)
    assert any(item["horizon"] == "long_term" for item in active_async)


@pytest.mark.asyncio
async def test_goal_engine_context_block_surfaces_long_horizon_and_recovery_pressure(tmp_path):
    engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))

    await engine.add_goal(
        "Finish the long-horizon runtime stabilization project",
        objective="Finish the long-horizon runtime stabilization project",
        horizon="long_term",
        priority=0.9,
        status="in_progress",
    )

    interrupted_step = SimpleNamespace(tool="sovereign_terminal")
    interrupted_plan = SimpleNamespace(
        plan_id="plan-recovery",
        goal="Repair the interrupted runtime verification loop",
        steps=[interrupted_step, SimpleNamespace(tool="read_file")],
        succeeded_steps=[],
        trace_id="trace-recovery",
        status="interrupted",
        final_result="",
        requires_approval=False,
        context={
            "task_id": "task-recovery",
            "source": "task_engine",
            "priority": 0.96,
            "horizon": "short_term",
            "quick_win": False,
            "error": "Interrupted during verification.",
        },
    )

    engine.sync_task_plan(interrupted_plan)
    block = engine.get_context_block("Keep going on the interrupted runtime project")

    assert "## GOAL EXECUTION STATE" in block
    assert "Immediate execution:" in block
    assert "Long-horizon anchors:" in block
    assert "Recovery pressure:" in block


@pytest.mark.asyncio
async def test_goal_engine_reuses_existing_active_record_for_same_objective_and_source(tmp_path):
    engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))

    first = await engine.add_goal(
        "Protect identity, memory integrity, and process continuity.",
        objective="Protect identity, memory integrity, and process continuity.",
        horizon="long_term",
        priority=0.98,
        source="executive_authority",
        status="in_progress",
    )
    second = await engine.add_goal(
        "Protect identity, memory integrity, and process continuity.",
        objective="Protect identity, memory integrity, and process continuity.",
        horizon="long_term",
        priority=0.98,
        source="executive_authority",
        status="in_progress",
    )

    assert first["id"] == second["id"]

    snapshot = engine.build_snapshot(limit=20, include_external=False)
    matching = [
        item
        for item in snapshot["items"]
        if item["objective"] == "Protect identity, memory integrity, and process continuity."
    ]
    assert len(matching) == 1
    assert matching[0]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_goal_engine_reconciles_stale_task_engine_plan_records(tmp_path, monkeypatch):
    engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))

    running_step = SimpleNamespace(tool="think")
    plan = SimpleNamespace(
        plan_id="plan-stale",
        goal="Protect identity, memory integrity, and process continuity.",
        steps=[running_step],
        succeeded_steps=[],
        trace_id="trace-stale",
        status="running",
        final_result="",
        requires_approval=False,
        context={
            "source": "task_engine",
            "priority": 0.75,
            "horizon": "short_term",
            "quick_win": True,
        },
    )
    engine.sync_task_plan(plan)

    stale_at = engine._now() - 120.0
    assert engine._conn is not None
    engine._conn.execute(
        "UPDATE goals SET created_at = ?, updated_at = ? WHERE plan_id = ?",
        (stale_at, stale_at, "plan-stale"),
    )
    engine._conn.commit()

    monkeypatch.setattr(
        "core.goals.goal_engine.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                SimpleNamespace(get_active_plans=lambda: []) if name == "task_engine" else default
            )
        ),
    )

    snapshot = engine.build_snapshot(limit=20, include_external=False)
    matching = [
        item
        for item in snapshot["items"]
        if item["objective"] == "Protect identity, memory integrity, and process continuity."
    ]

    assert matching
    assert matching[0]["status"] == "blocked"
    assert "interrupted" in str(matching[0]["summary"]).lower()


@pytest.mark.asyncio
async def test_goal_engine_state_sync_prefers_actionable_goals(service_container, tmp_path):
    state = AuraState()
    ServiceContainer.register_instance("state_repo", SimpleNamespace(_current=state), required=False)

    engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))

    await engine.add_goal(
        "Protect identity, memory integrity, and process continuity.",
        objective="Protect identity, memory integrity, and process continuity.",
        horizon="long_term",
        priority=0.98,
        source="executive_authority",
        status="in_progress",
    )

    running_step = SimpleNamespace(tool="read_file")
    plan = SimpleNamespace(
        plan_id="plan-actionable",
        goal="Investigate hierarchical phi event loop lag",
        steps=[running_step],
        succeeded_steps=[],
        trace_id="trace-actionable",
        status="running",
        final_result="",
        requires_approval=False,
        context={
            "task_id": "task-actionable",
            "source": "task_engine",
            "priority": 0.91,
            "horizon": "short_term",
            "quick_win": True,
        },
    )

    engine.sync_task_plan(plan)

    assert state.cognition.current_objective == "Investigate hierarchical phi event loop lag"
    assert state.cognition.active_goals
    assert all(
        goal.get("description") != "Protect identity, memory integrity, and process continuity."
        for goal in state.cognition.active_goals
        if isinstance(goal, dict)
    )
