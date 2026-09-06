"""A batch nobody can edit, and a ledger that says what happened to each."""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.what_she_decided_to_do_at_once import (
    ABatch,
    AnAction,
    HowItWent,
    a_batch_of,
    forget_everything,
    how_the_batches_have_gone,
    run_the_batch,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


async def _returns_its_name(action: AnAction):
    return action.name


def test_a_batch_cannot_be_edited_by_whatever_runs_it() -> None:
    batch = a_batch_of([AnAction("read"), AnAction("write")])
    with pytest.raises((AttributeError, TypeError)):
        batch.actions = ()  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        batch.actions[0].name = "something else"  # type: ignore[misc]


def test_two_actions_sharing_an_id_is_refused_at_construction() -> None:
    """Results are joined by id; two the same means one loses its result."""
    one = AnAction("read")
    with pytest.raises(ValueError, match="share an id"):
        ABatch(actions=(one, one))


def test_a_batch_that_may_run_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="not a batch"):
        ABatch(actions=(AnAction("read"),), at_once=0)


def test_blocked_is_not_failed() -> None:
    """One never ran and the other ran and raised; asking again differs."""

    async def raises(action: AnAction):
        raise RuntimeError("disk full")

    batch = a_batch_of([AnAction("write"), AnAction("post")])
    led = asyncio.run(
        run_the_batch(
            batch,
            raises,
            may_it_run=lambda a: "governance refuses posting" if a.name == "post" else "",
        )
    )
    went = {one.action_id: one.went for one in led.in_the_order_decided()}
    assert went[batch.actions[0].id] is HowItWent.FAILED
    assert went[batch.actions[1].id] is HowItWent.BLOCKED
    assert led.what_was_blocked()[0].said == "governance refuses posting"
    assert len(led.what_ran()) == 1, "the blocked one did not run"


def test_results_come_back_in_the_order_decided_not_the_order_finished() -> None:
    """Completion order tells a different story every run from one input."""
    slow_first = a_batch_of(
        [AnAction("slow"), AnAction("quick"), AnAction("quicker")], at_once=3
    )

    async def do(action: AnAction):
        await asyncio.sleep(0.08 if action.name == "slow" else 0.0)
        return action.name

    led = asyncio.run(run_the_batch(slow_first, do))
    got = [one.value for one in led.in_the_order_decided()]
    assert got == ["slow", "quick", "quicker"]


def test_nothing_after_a_finishing_action_is_reached() -> None:
    ran: list[str] = []

    async def do(action: AnAction):
        ran.append(action.name)
        return action.name

    batch = a_batch_of(
        [AnAction("work"), AnAction("finish", finishes=True), AnAction("after")]
    )
    led = asyncio.run(run_the_batch(batch, do))
    assert ran == ["work", "finish"]
    assert led.outcome(batch.actions[2].id).went is HowItWent.NOT_REACHED
    assert "finishing action" in led.outcome(batch.actions[2].id).said


def test_no_more_than_the_declared_number_run_together() -> None:
    """Parallel tools share a filesystem, so the limit is the batch's own."""
    running = {"now": 0, "most": 0}

    async def do(action: AnAction):
        running["now"] += 1
        running["most"] = max(running["most"], running["now"])
        await asyncio.sleep(0.02)
        running["now"] -= 1
        return action.name

    batch = a_batch_of([AnAction(f"a{n}") for n in range(8)], at_once=3)
    asyncio.run(run_the_batch(batch, do))
    assert running["most"] <= 3


def test_a_cancelled_batch_records_what_never_ran_rather_than_losing_it() -> None:
    async def do(action: AnAction):
        await asyncio.sleep(5)
        return action.name

    batch = a_batch_of([AnAction("slow")], at_once=1)

    async def go():
        task = asyncio.create_task(run_the_batch(batch, do))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    seen = how_the_batches_have_gone()
    assert seen["batches"] == 1, "the ledger is kept however the batch ended"


def test_the_ledger_is_not_the_batch() -> None:
    batch = a_batch_of([AnAction("read", because="needs the file")])
    led = asyncio.run(run_the_batch(batch, _returns_its_name))
    assert led.outcome(batch.actions[0].id).value == "read"
    assert batch.actions[0].because == "needs the file", "the plan is unchanged"
    assert not hasattr(batch.actions[0], "went")


def test_the_report_counts_every_ending() -> None:
    batch = a_batch_of([AnAction("a"), AnAction("b")])
    led = asyncio.run(run_the_batch(batch, _returns_its_name))
    seen = led.report()
    assert seen["actions"] == 2
    assert seen["counted"] == {"did it": 2}
    assert [one["name"] for one in seen["in_order"]] == ["a", "b"]
