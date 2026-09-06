"""Giving up on an answer, and the work carrying on regardless."""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.cancelling_the_call_and_not_just_the_wait import (
    ACall,
    GaveUpWaiting,
    call,
    forget_everything,
    how_the_calls_ended,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_giving_up_on_the_wait_stops_the_work() -> None:
    """It used to keep the lane, write state, and answer into nothing."""
    finished = {"it did": False}

    async def slow(one: ACall):
        for _ in range(200):
            if one.should_stop():
                return "stopped"
            await asyncio.sleep(0.01)
        finished["it did"] = True
        return "done"

    async def go():
        with pytest.raises(GaveUpWaiting, match="stopped rather than left running"):
            await call("a slow answer", slow, by="turn-4", seconds=0.08)
        await asyncio.sleep(0.15)

    asyncio.run(go())
    assert not finished["it did"]


def test_giving_up_is_not_the_same_as_the_work_failing() -> None:
    """A caller that stopped waiting learned nothing about what was possible."""

    async def raises(one: ACall):
        raise RuntimeError("the capability is broken")

    async def go():
        with pytest.raises(RuntimeError, match="broken"):
            await call("a call that fails", raises, seconds=5.0)

    asyncio.run(go())
    assert not issubclass(RuntimeError, GaveUpWaiting)
    assert issubclass(GaveUpWaiting, asyncio.CancelledError), (
        "every `except CancelledError: raise` in the tree must keep working"
    )


def test_work_that_finishes_in_time_comes_back_normally() -> None:
    async def quick(one: ACall):
        return "here it is"

    assert asyncio.run(call("quick", quick, seconds=5.0)) == "here it is"
    assert how_the_calls_ended()["given_up_on"] == 0


def test_cancelling_the_caller_stops_the_work_too() -> None:
    """Otherwise it answers into a future nobody is holding."""
    stopped = {"it was": False}

    async def slow(one: ACall):
        for _ in range(200):
            if one.should_stop():
                stopped["it was"] = True
                return "stopped"
            await asyncio.sleep(0.01)
        return "done"

    async def go():
        task = asyncio.create_task(call("a slow answer", slow))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)

    asyncio.run(go())
    assert stopped["it was"], "the work was not told the caller had gone"


def test_work_that_never_checks_is_named_rather_than_assumed_to_have_stopped() -> None:
    """The honest version of 'cancellation did nothing'."""

    async def deaf(one: ACall):
        await asyncio.sleep(5)
        return "done"

    async def go():
        with pytest.raises(GaveUpWaiting):
            await call("a deaf handler", deaf, seconds=0.05)

    asyncio.run(go())
    seen = how_the_calls_ended()
    assert seen["given_up_on"] == 1
    assert seen["work_that_never_checked"] == 1
    assert seen["never_checked_what"] == ["a deaf handler"]


def test_a_handler_that_checks_gets_to_stop_itself_before_being_cancelled() -> None:
    """Cancelling first raises inside whatever await it is on, so it never
    reaches its own stop branch and "cleanly" becomes "wherever it was"."""
    how = {"it ended": ""}

    async def careful(one: ACall):
        try:
            for _ in range(200):
                if one.should_stop():
                    how["it ended"] = "it noticed"
                    return "stopped"
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            how["it ended"] = "it was cancelled"
            raise
        return "done"

    async def go():
        with pytest.raises(GaveUpWaiting):
            await call("a careful handler", careful, seconds=0.05)

    asyncio.run(go())
    assert how["it ended"] == "it noticed"


def test_the_report_says_how_often_work_outlived_its_caller() -> None:
    async def quick(one: ACall):
        return 1

    asyncio.run(call("a", quick, seconds=5.0))
    seen = how_the_calls_ended()
    assert seen["calls"] == 1
    assert seen["given_up_on"] == 0
    assert seen["longest"] >= 0.0
