from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine


@pytest.mark.asyncio
async def test_task_engine_fallback_plan_survives_malformed_decomposition():
    llm = SimpleNamespace(think=AsyncMock(return_value='[{"description": "broken"'))
    kernel = SimpleNamespace(
        organs={"llm": SimpleNamespace(get_instance=lambda: llm)}
    )

    engine = AutonomousTaskEngine(kernel)

    plan = await engine._decompose_goal("Inspect runtime health", "plan_test", context=None)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "think"
    assert plan.steps[0].rollback_action is None
    llm.think.assert_awaited()


@pytest.mark.asyncio
async def test_task_engine_execute_alias_delegates_to_execute_goal():
    engine = AutonomousTaskEngine.__new__(AutonomousTaskEngine)
    engine.execute_goal = AsyncMock(return_value="ok")

    result = await AutonomousTaskEngine.execute(
        engine,
        "Keep the runtime stable",
        context={"task_id": "task-1"},
        is_shadow=True,
    )

    assert result == "ok"
    engine.execute_goal.assert_awaited_once_with(
        goal="Keep the runtime stable",
        context={"task_id": "task-1"},
        on_progress=None,
        is_shadow=True,
    )
