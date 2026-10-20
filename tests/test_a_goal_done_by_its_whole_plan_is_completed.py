"""A goal whose whole objective ran as a plan with every step verified is completed, not left in progress.

Live on 18 October a plan for one goal completed (1 of 1 steps) and the overt action loop set the goal back to
in_progress, at most 95% done, so the arbiter chose it again: 163 times that day.
"""
from __future__ import annotations

import asyncio

from core.runtime.overt_action_loop import OvertActionLoop, OvertActionResult


class GoalEngine:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def get_goal(self, goal_id):
        return {"id": goal_id, "progress": 0.4, "evidence": []}

    async def update_goal_status(self, goal_id, **kwargs):
        self.updates.append({"goal_id": goal_id, **kwargs})


def _settle(result: OvertActionResult, raw: dict) -> dict:
    engine = GoalEngine()
    loop = OvertActionLoop(goal_engine=engine)

    async def run() -> None:
        loop._update_goal(result, raw)
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(run())
    return engine.updates[-1]


def test_a_whole_plan_verified_completes_the_goal():
    result = OvertActionResult(action_id="a1", status="verified", verified=True, goal_id="g1",
                               execution_mode="planned_goal")
    update = _settle(result, {"status": "completed_verified", "steps_completed": 1, "steps_total": 1})
    assert update["status"] == "completed" and update["progress"] == 1.0


def test_one_skills_step_leaves_the_goal_in_progress():
    result = OvertActionResult(action_id="a2", status="verified", verified=True, goal_id="g1", execution_mode="skill")
    update = _settle(result, {"ok": True})
    assert update["status"] == "in_progress" and update["progress"] < 1.0


def test_a_plan_that_did_not_complete_leaves_it_in_progress():
    result = OvertActionResult(action_id="a3", status="verified", verified=True, goal_id="g1",
                               execution_mode="planned_goal")
    update = _settle(result, {"status": "execution_not_completed"})
    assert update["status"] == "in_progress"
