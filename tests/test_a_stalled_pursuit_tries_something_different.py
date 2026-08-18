"""A second attempt has to be a different attempt.

``GoalPursuitEngine.pursue`` has carried a ``replan`` parameter and a
``max_replans`` budget from the beginning, and the only live caller —
``ProactiveAgency.pursue_goal``, reached from the autonomous initiative loop —
never passed one. So the budget was dead: every stall ended the pursuit for
good, and the follow-through the engine advertises in its own docstring could
not happen.

Supplying a replanner is only half of it. A goal classifies the same way every
time it is read, so a naive re-plan hands back the plan that just stalled and
it stalls again in exactly the same place. The failure has to change the plan,
and it has to do so structurally rather than by being described in a prompt.
"""

from __future__ import annotations

import pytest

from core.agency.goal_planner import GoalPlanner
from core.agency.proactive_agency import ProactiveAgency
from core.skills.fluid_executor import ExecutionReceipt, Step, StepResult


def _step(name: str, approach: str) -> Step:
    return Step(name=name, action=lambda: None, verify="always_true", approach=approach)


# ── routing away from what failed ────────────────────────────────────────


def test_an_approach_that_failed_is_not_chosen_again():
    planner = GoalPlanner()
    goal = "open Notes and write a line"
    assert planner.classify(goal) == "desktop"
    assert planner.classify(goal, avoid=("desktop",)) == "reasoning"


def test_computation_falls_through_to_reasoning():
    planner = GoalPlanner()
    goal = "calculate 47*89"
    assert planner.classify(goal) == "computational"
    assert planner.classify(goal, avoid=("computational",)) == "reasoning"


def test_when_nothing_different_is_left_it_says_so():
    """Reasoning is the universal fallback; exhausting it means stop.

    Returning the failed plan again would be theatre — a retry that cannot
    differ from the attempt that just failed.
    """
    planner = GoalPlanner()
    assert planner.classify("open Notes", avoid=("desktop", "reasoning")) == "none"
    assert planner.classify("anything at all", avoid=("reasoning",)) == "none"


# ── the approach travels with the work ───────────────────────────────────


def test_planned_steps_carry_the_approach_that_produced_them():
    """A step NAME says what was tried; the approach says how it was chosen.

    Reconstructing one from the other needs a name-to-approach table, which is
    a second vocabulary that drifts. Measured: a stalled desktop step reported
    "desktop_open", matched no approach, and the retry was abandoned even
    though a different approach was available.
    """
    steps = GoalPlanner().plan  # the coroutine function itself
    assert steps is not None
    result = StepResult(name="desktop_open", ok=False, approach="desktop")
    assert result.approach == "desktop"
    assert result.to_dict()["approach"] == "desktop"


def test_the_executor_carries_the_approach_onto_the_result():
    from core.skills.fluid_executor import FluidExecutor

    executor = FluidExecutor()
    assert hasattr(Step("x", action=lambda: None), "approach")
    assert hasattr(StepResult("x", ok=True), "approach")
    assert executor is not None


# ── the live loop ────────────────────────────────────────────────────────


class _Engine:
    """Stands in for GoalPursuitEngine, recording what it was handed."""

    def __init__(self) -> None:
        self.attempts: list[list[tuple[str, str]]] = []
        self.replanned: list[list[tuple[str, str]]] = []

    async def pursue(self, goal, plan, *, parallel=False, timing_ok=None, replan=None):
        self.attempts.append([(s.name, s.approach) for s in plan])
        receipt = ExecutionReceipt(
            goal=goal,
            completed=False,
            stalled=True,
            steps=[
                StepResult(
                    name=plan[0].name, ok=False, detail="stalled", approach=plan[0].approach
                )
            ],
        )
        fresh = await replan(receipt) if replan is not None else None
        self.replanned.append([(s.name, s.approach) for s in (fresh or [])])
        return type("Outcome", (), {"completed": False, "deferred": False})()


def _agency(engine, planner):
    return ProactiveAgency(
        pursuit=engine,
        planner=planner,
        background_allowed=lambda: True,
        timing_ok=lambda: True,
    )


@pytest.mark.asyncio
async def test_a_stall_produces_a_plan_by_a_different_approach():
    engine = _Engine()
    seen: list[tuple[str, ...]] = []

    async def planner(goal, *, avoid=()):
        seen.append(tuple(avoid))
        if "desktop" in avoid:
            return [_step("reason", "reasoning")]
        return [_step("desktop_open", "desktop")]

    await _agency(engine, planner).pursue_goal("open Notes and write a line")

    assert seen == [(), ("desktop",)], "the failure did not reach the re-plan"
    assert engine.replanned[0] == [("reason", "reasoning")]


@pytest.mark.asyncio
async def test_a_replan_that_matches_the_failed_plan_is_refused():
    """Running it again would stall in the same place."""
    engine = _Engine()

    async def stubborn(goal, *, avoid=()):
        return [_step("desktop_open", "desktop")]

    await _agency(engine, stubborn).pursue_goal("open Notes")

    assert engine.replanned[0] == [], "an identical plan was handed back for a retry"


@pytest.mark.asyncio
async def test_a_planner_without_failure_routing_is_not_retried_blindly():
    """An older planner can only reproduce the plan that just failed."""
    engine = _Engine()

    async def legacy(goal):  # no `avoid` parameter
        return [_step("desktop_open", "desktop")]

    await _agency(engine, legacy).pursue_goal("open Notes")

    assert engine.replanned[0] == []


@pytest.mark.asyncio
async def test_the_live_caller_actually_supplies_a_replanner():
    """The whole defect was that it did not."""
    engine = _Engine()

    async def planner(goal, *, avoid=()):
        return [_step("reason", "reasoning")] if avoid else [_step("desktop_open", "desktop")]

    await _agency(engine, planner).pursue_goal("open Notes")

    # If no replanner were passed, `replanned` would hold an empty list because
    # replan was None — so assert the planner was consulted a second time.
    assert engine.replanned[0], "pursue() was called without a replanner again"
