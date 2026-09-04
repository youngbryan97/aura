"""A run says what it spent, so a ceiling could later be chosen from numbers.

The turn observer says it plainly: every ceiling in ``Budget`` is a guess
borrowed from someone else's system, and a limit chosen without knowing the
distribution is an arbitrary constant wearing a safety label. Numbers first.

The execution loop had none. A receipt said "failed after 3 attempts" and
could not tell a step that failed fast from one that burned a minute doing
it, and a slow run could not say whether it was slow at thinking or slow at
acting. No ceiling is imposed here, deliberately — this is the measurement
that would have to come before one.
"""

from __future__ import annotations

import asyncio

import pytest

from core.skills.fluid_executor import ExecutionReceipt, Step, StepResult


def _executor():
    from core.skills.fluid_executor import FluidExecutor

    runner = FluidExecutor()

    async def always(predicate, args):
        return True, "verified"

    async def approved(step):
        return True, ""

    runner._verify = always
    runner._approved = approved
    return runner


@pytest.mark.asyncio
async def test_a_step_reports_the_time_it_took():
    async def slow():
        await asyncio.sleep(0.05)

    result = await _executor().run_step(
        Step(name="a slow one", action=slow, verify="anything")
    )
    assert result.ok
    assert result.seconds >= 0.05


@pytest.mark.asyncio
async def test_a_blocked_step_still_reports_its_time():
    """A refusal costs something, and a receipt that hides it under-counts."""
    from core.skills.fluid_executor import FluidExecutor

    runner = FluidExecutor()

    async def refused(step):
        return False, "not allowed"

    runner._approved = refused
    result = await runner.run_step(
        Step(name="a refused one", action=lambda: None, verify="anything")
    )
    assert result.blocked
    assert result.seconds >= 0.0
    assert "seconds" in result.to_dict()


@pytest.mark.asyncio
async def test_a_failed_step_reports_what_the_retries_cost():
    """The number that separates a fast failure from an expensive one."""
    from core.skills.fluid_executor import FluidExecutor

    runner = FluidExecutor()

    async def never(predicate, args):
        return False, "did not verify"

    async def approved(step):
        return True, ""

    runner._verify = never
    runner._approved = approved
    runner._sleep = lambda seconds: asyncio.sleep(0.02)
    result = await runner.run_step(
        Step(name="one that never verifies", action=lambda: None, verify="anything",
             max_retries=2)
    )
    assert not result.ok
    assert result.attempts == 3
    assert result.seconds >= 0.04, result.seconds


def test_a_receipt_separates_doing_from_deciding():
    receipt = ExecutionReceipt(
        goal="g",
        completed=True,
        elapsed_s=10.0,
        steps=[
            StepResult("one", ok=True, attempts=1, seconds=3.0),
            StepResult("two", ok=True, attempts=2, seconds=4.0),
        ],
    )
    assert receipt.seconds_in_steps == 7.0
    assert receipt.seconds_deciding == 3.0
    assert receipt.attempts == 3, "attempts counts tries, not steps"


def test_deciding_time_is_never_negative():
    """A step timed across a boundary the run did not span is not a bug worth crashing on."""
    receipt = ExecutionReceipt(
        goal="g", completed=True, elapsed_s=1.0,
        steps=[StepResult("one", ok=True, seconds=5.0)],
    )
    assert receipt.seconds_deciding == 0.0


def test_the_numbers_reach_the_dict():
    receipt = ExecutionReceipt(
        goal="g", completed=True, elapsed_s=2.0,
        steps=[StepResult("one", ok=True, attempts=1, seconds=1.0)],
    )
    said = receipt.to_dict()
    assert said["attempts"] == 1
    assert said["seconds_in_steps"] == 1.0
    assert said["seconds_deciding"] == 1.0
