"""A plan that stalls used to ask a lambda what to do about it.

GoalPursuitEngine re-plans when a run stalls, and took the new plan from an
injected callable. Nothing in the codebase supplied one, so in practice a
stalled plan was retried unchanged or abandoned.

A model cannot write a step, because a step carries a real callable. It can
choose among repairs to a plan that really exists, stuck at a step that
really failed — and because the choice goes through the same deliberation,
the repair arrives with a prediction that the next receipt grades.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from core.agency.replanning import (
    DROP,
    GO_AROUND,
    RETRY,
    START_OVER,
    apply_repair,
    failed_step,
    replan,
    replanner,
)


@dataclass
class _Step:
    name: str
    optional: bool = False


@dataclass
class _Result:
    name: str
    ok: bool = True
    verified: bool = True


@dataclass
class _Receipt:
    completed: bool = False
    steps: list = field(default_factory=list)
    verified_progress: int = 0


class _Store:
    def record(self, episode):
        return "ep_1"

    def resolve(self, episode_id, outcome):
        pass

    def query_consequences(self, action, params=None):
        return []

    def record_outcome(self, action, context, outcome, success):
        pass


def _thinks(reply):
    async def think(objective, evidence):
        think.seen = list(evidence)
        return reply

    think.seen = None
    return think


def _plan():
    return [_Step("open the page"), _Step("fill the form"), _Step("submit")]


def _stalled(at="fill the form", progress=1):
    return _Receipt(
        steps=[_Result("open the page"), _Result(at, ok=False, verified=False), _Result("submit")],
        verified_progress=progress,
    )


def test_the_step_a_run_got_stuck_on_is_read_from_the_receipt():
    assert failed_step(_stalled()) == "fill the form"
    assert failed_step(_Receipt()) == ""


@pytest.mark.asyncio
async def test_a_repair_names_a_step_that_really_failed():
    repair = await replan(
        "book the appointment",
        _plan(),
        _stalled(),
        think=_thinks("drop it and carry on"),
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert repair is not None
    assert repair.kind == DROP
    assert repair.step == "fill the form"
    assert [s.name for s in repair.plan] == ["open the page", "submit"]


@pytest.mark.asyncio
async def test_the_decision_sees_where_the_run_actually_got_to():
    think = _thinks("retry")
    await replan(
        "book the appointment",
        _plan(),
        _stalled(progress=1),
        think=think,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    assert any("stalled at 'fill the form'" in line for line in think.seen)
    assert any("1 verified step(s) of 3" in line for line in think.seen)


@pytest.mark.asyncio
async def test_no_repair_she_can_justify_stops_the_run_rather_than_repeating_it():
    async def unreachable(objective, evidence):
        raise RuntimeError("no model")

    repair = await replan(
        "book the appointment", _plan(), _stalled(), think=unreachable, lived=False, spine=_Store(), graph=_Store()
    )
    assert repair is None


@pytest.mark.asyncio
async def test_dropping_the_last_step_is_not_offered():
    """Dropping the step a plan ends on leaves a plan that cannot reach the goal."""
    think = _thinks("drop")
    await replan(
        "book the appointment",
        _plan(),
        _stalled(at="submit"),
        think=think,
        lived=False,
        spine=_Store(),
        graph=_Store(),
    )
    offered = [line for line in think.seen if line.startswith("Available move")]
    assert not any(line.startswith(f"Available move — {DROP}") for line in offered)
    assert any(RETRY in line for line in offered)


def test_a_repair_never_edits_the_plan_it_was_given():
    plan = _plan()
    repaired = apply_repair(DROP, plan, "fill the form")
    assert [s.name for s in plan] == ["open the page", "fill the form", "submit"]
    assert [s.name for s in repaired] == ["open the page", "submit"]


def test_going_around_a_step_makes_it_one_the_run_may_finish_without():
    repaired = apply_repair(GO_AROUND, _plan(), "fill the form")
    assert repaired[1].optional is True
    assert repaired[0].optional is False


def test_retrying_and_starting_over_keep_the_plan_whole():
    for kind in (RETRY, START_OVER):
        assert [s.name for s in apply_repair(kind, _plan(), "fill the form")] == [
            "open the page",
            "fill the form",
            "submit",
        ]


@pytest.mark.asyncio
async def test_a_second_stall_knows_the_first_repair_did_not_help():
    think = _thinks("retry")
    fix = replanner(
        "book the appointment", _plan(), think=think, lived=False, spine=_Store(), graph=_Store()
    )
    await fix(_stalled(progress=1))
    # The run got no further, so the repair that was tried is what failed.
    await fix(_stalled(progress=1))
    assert any(
        line.startswith("retry was expected") and "nothing changed" in line for line in think.seen
    ), think.seen


@pytest.mark.asyncio
async def test_a_plan_with_nothing_in_it_has_nothing_to_repair():
    assert await replan("goal", [], _Receipt(), think=_thinks("retry"), lived=False) is None
