from types import SimpleNamespace

import pytest

from core.goals.goal_engine import GoalEngine


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
