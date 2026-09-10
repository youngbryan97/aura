"""A turn priced twice must not be waited on by the smaller of the two numbers.

The route sets a wall-clock SLA when the request is admitted, from the lane and
a guess at how long the prompt will be. The answer clock prices the same turn
again from the prompt that was actually built and the rates the worker just
measured. LIVE, 2026-09-10: "deadline 103s to 251s", the gate honoured 251, the
route gave up at its own budget with the cortex still generating, and the person
was told the answer took too long to finish cleanly.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
from core.runtime.the_turn_clock import (
    ask_the_turn_for,
    close_the_turn_clock,
    open_a_turn_clock,
    the_turn_clock,
)


@pytest.fixture(autouse=True)
def _no_clock_left_open():
    yield
    close_the_turn_clock()


def test_an_extension_is_visible_to_whoever_is_waiting() -> None:
    clock = open_a_turn_clock(103.0)
    assert clock.budget() == 103.0
    ask_the_turn_for(251.0, "the answer clock priced this turn at 251s")
    assert clock.budget() == 251.0
    assert clock.was_extended()
    assert "251" in clock.to_dict()["why"]


def test_the_turn_is_never_extended_past_the_ceiling() -> None:
    clock = open_a_turn_clock(103.0)
    ask_the_turn_for(9000.0, "greedy")
    assert clock.budget() == float(USER_FACING_COMPLETION_DEADLINE_MAX_S)


def test_a_smaller_number_does_not_shorten_the_turn() -> None:
    clock = open_a_turn_clock(200.0)
    ask_the_turn_for(30.0, "a cheaper estimate")
    assert clock.budget() == 200.0
    assert not clock.was_extended()


def test_asking_with_no_turn_open_changes_nothing() -> None:
    close_the_turn_clock()
    assert the_turn_clock() is None
    assert ask_the_turn_for(400.0, "background work") == 0.0


def test_a_task_started_in_the_turn_extends_the_turn_the_route_is_waiting_on() -> None:
    """The property that makes this work: down the tree, back up the object.

    A task copies the context, so a value set inside it cannot reach the
    caller. The clock is a mutable object in that context, and mutating it is
    visible on both sides — which is the whole reason it is an object.
    """

    async def turn() -> float:
        clock = open_a_turn_clock(103.0)

        async def the_gate() -> None:
            ask_the_turn_for(251.0, "measured")

        await asyncio.create_task(the_gate())
        return clock.budget()

    assert asyncio.run(turn()) == 251.0


def test_the_wait_reads_the_clock_rather_than_the_admitted_budget() -> None:
    """The route's own remaining-budget helper, in the source it is written in.

    A behavioural test would need the whole chat route; what matters here is
    that the helper consults the turn clock at all, since the defect was that
    nothing did.
    """
    body = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    helper = re.search(
        r"def _remaining_foreground_budget\(.*?\n(?:.*?\n)*?        return max\(2\.0,",
        body,
    )
    assert helper is not None, "the route's remaining-budget helper moved"
    assert "the_turn_clock" in helper.group(0)
