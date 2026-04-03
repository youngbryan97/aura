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
