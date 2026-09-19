"""Tests for the proactive agency bridge (goal → planned → pursued, gated)."""
from __future__ import annotations

import pytest

from core.agency.goal_pursuit import GoalPursuitEngine
from core.agency.proactive_agency import ProactiveAgency
from core.skills.fluid_executor import FluidExecutor, Step


class _OkVerifier:
    async def verify(self, predicate, args=None):
        from types import SimpleNamespace

        return SimpleNamespace(success=True, detail="ok")


async def _noop():
    return None


def _engine():
    return GoalPursuitEngine(executor=FluidExecutor(verifier=_OkVerifier(), sleep=lambda _s: _an()))


async def _an():
    return None


async def _planner(goal):
    return [Step(f"do:{goal[:10]}", _noop)]


@pytest.mark.asyncio
async def test_no_planner_with_default_disabled_is_safe_noop():
    pa = ProactiveAgency(pursuit=_engine(), default_planner_enabled=False)
    assert await pa.pursue_goal("organize my notes") is None
    assert pa.status()["has_planner"] is False


@pytest.mark.asyncio
async def test_default_planner_pursues_open_ended_goal():
    """No explicit planner → the GoalPlanner default makes it plannable.

    Given something to reason WITH. The default planner's reasoning step
    deliberates over a generator, and with none supplied it falls back to
    `inference_gate` out of the container — absent here, so the step
    produced an empty answer and failed three times as "action did not
    complete". What this test is about is the default planner turning an
    open-ended goal into a pursuable plan, not about where the words come
    from.
    """
    from core.agency.goal_planner import GoalPlanner

    async def _reasons(prompt: str, temperature: float) -> str:
        return "Start by listing where notes already live, then pick one place."

    pa = ProactiveAgency(
        pursuit=_engine(),
        planner=GoalPlanner(generate=_reasons, deliberate_samples=1).plan,
    )
    out = await pa.pursue_goal("figure out the best approach to organizing notes")
    assert out is not None and out.completed


@pytest.mark.asyncio
async def test_pursues_goal_when_planner_registered():
    pa = ProactiveAgency(pursuit=_engine(), planner=_planner)
    out = await pa.pursue_goal("research the Knicks")
    assert out is not None and out.completed
    assert pa.status()["completed"] == 1


@pytest.mark.asyncio
async def test_background_gate_blocks_when_not_allowed():
    pa = ProactiveAgency(pursuit=_engine(), planner=_planner, background_allowed=lambda: False)
    assert await pa.pursue_goal("do a thing") is None
    assert pa.status()["pursued"] == 0                 # never even planned


@pytest.mark.asyncio
async def test_timing_gate_defers():
    pa = ProactiveAgency(pursuit=_engine(), planner=_planner, timing_ok=lambda: False)
    out = await pa.pursue_goal("ping the user")
    assert out is not None and out.deferred and not out.completed


@pytest.mark.asyncio
async def test_empty_goal_ignored():
    pa = ProactiveAgency(pursuit=_engine(), planner=_planner)
    assert await pa.pursue_goal("") is None


@pytest.mark.asyncio
async def test_empty_plan_does_not_pursue():
    async def _empty_planner(goal):
        return []

    pa = ProactiveAgency(pursuit=_engine(), planner=_empty_planner)
    assert await pa.pursue_goal("unplannable") is None
    assert pa.status()["pursued"] == 0
