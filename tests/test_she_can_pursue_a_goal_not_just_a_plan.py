"""The executor could follow a list but could not chase a goal.

FluidExecutor is the closed perceive→act→verify→recover loop, and it already
had governance, effect verification, bounded retry, stall detection and
receipts. Its entry point was ``run(goal, steps)`` — a list written in advance.

That is the right shape when the steps are known: open this app, click that
button. It is the wrong shape for anything whose next move depends on what
just happened. A board that changes between moves, a page that loads at its
own pace, a drag that needs correcting mid-flight — none can be written down
beforehand, so none were reachable through the executor, even though every
part needed to reach them already existed. The gap was never a missing
capability; it was that the loop had no way to look before choosing.

``pursue`` is the same loop with the plan removed: ``decide`` sees the world
before choosing, and the run ends on a PREDICATE rather than on running out of
list. Everything still goes through run_step, so nothing about governance or
verification changes.

Nothing here knows about screens, browsers or games. A progress bar and a
changing board are the same problem.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.perception_demand import (
    perception_is_demanded,
    reset_perception_demand,
)
from core.skills.fluid_executor import FluidExecutor, Step


@pytest.fixture(autouse=True)
def _clean():
    reset_perception_demand()
    yield
    reset_perception_demand()


def _executor(**kw):
    return FluidExecutor(verifier=None, gateway=None, **kw)


def _counter_world():
    world = {"n": 0}

    async def observe():
        return world["n"]

    async def decide(_obs):
        async def act():
            world["n"] += 1
            return True

        return Step(name="increment", action=act)

    return world, observe, decide


def test_a_goal_is_reached_by_looking_between_moves():
    world, observe, decide = _counter_world()

    receipt = asyncio.run(
        _executor().pursue(
            "count to five",
            observe=observe,
            decide=decide,
            is_satisfied=lambda n: n >= 5,
        )
    )

    assert receipt.completed is True
    assert receipt.outcome == "goal_reached"
    assert world["n"] == 5
    assert receipt.verified_progress == 5


def test_the_predicate_ends_the_run_not_the_length_of_a_list():
    """The whole difference from run(): nobody said how many steps."""
    world, observe, decide = _counter_world()

    receipt = asyncio.run(
        _executor().pursue(
            "count to two", observe=observe, decide=decide, is_satisfied=lambda n: n >= 2
        )
    )

    assert world["n"] == 2
    assert receipt.cycles < 10


def test_an_async_predicate_works_too():
    """Deciding whether a goal is met often needs a look of its own."""
    world, observe, decide = _counter_world()

    async def satisfied(n):
        await asyncio.sleep(0)
        return n >= 3

    receipt = asyncio.run(
        _executor().pursue(
            "count to three", observe=observe, decide=decide, is_satisfied=satisfied
        )
    )

    assert receipt.completed is True
    assert world["n"] == 3


def test_an_unreachable_goal_is_bounded_by_cycles():
    """A loop with a goal and no bound is how a process eats a machine."""
    _, observe, decide = _counter_world()

    receipt = asyncio.run(
        _executor().pursue(
            "never satisfied",
            observe=observe,
            decide=decide,
            is_satisfied=lambda _n: False,
            max_cycles=12,
        )
    )

    assert receipt.completed is False
    assert receipt.outcome == "out_of_cycles"
    assert receipt.cycles == 12


def test_a_slow_goal_is_bounded_by_the_clock():
    async def observe():
        await asyncio.sleep(0.02)
        return 0

    async def decide(_obs):
        async def act():
            return True

        return Step(name="tick", action=act)

    receipt = asyncio.run(
        _executor().pursue(
            "too slow",
            observe=observe,
            decide=decide,
            is_satisfied=lambda _n: False,
            max_cycles=10_000,
            max_seconds=0.25,
        )
    )

    assert receipt.outcome == "out_of_time"
    assert receipt.elapsed_s < 5.0


def test_having_no_move_stalls_honestly_instead_of_spinning():
    """"nothing worth doing from here" is an answer, not a reason to loop."""

    async def observe():
        return 0

    async def decide(_obs):
        return None

    receipt = asyncio.run(
        _executor(stall_window=3).pursue(
            "no move exists",
            observe=observe,
            decide=decide,
            is_satisfied=lambda _n: False,
            max_cycles=500,
        )
    )

    assert receipt.stalled is True
    assert receipt.outcome == "no_move_available"
    assert receipt.cycles <= 4


def test_perception_is_held_open_for_the_whole_pursuit():
    """A loop acting on what it sees is exactly where sight was throttled."""
    seen: list[bool] = []

    async def observe():
        seen.append(perception_is_demanded())
        return len(seen)

    async def decide(_obs):
        async def act():
            return True

        return Step(name="step", action=act)

    asyncio.run(
        _executor().pursue(
            "watch while acting",
            observe=observe,
            decide=decide,
            is_satisfied=lambda n: n >= 3,
        )
    )

    assert all(seen), seen
    assert not perception_is_demanded(), "the claim must not outlive the run"


def test_perception_is_released_even_when_the_run_raises():
    async def observe():
        raise RuntimeError("perception exploded")

    async def decide(_obs):
        return None

    with pytest.raises(RuntimeError):
        asyncio.run(
            _executor().pursue(
                "explodes",
                observe=observe,
                decide=decide,
                is_satisfied=lambda _n: False,
            )
        )

    assert not perception_is_demanded()


def test_the_receipt_says_how_it_ended():
    """"It didn't work" is not a diagnosis; the outcome names the reason."""
    _, observe, decide = _counter_world()

    reached = asyncio.run(
        _executor().pursue(
            "reachable", observe=observe, decide=decide, is_satisfied=lambda n: n >= 1
        )
    )
    exhausted = asyncio.run(
        _executor().pursue(
            "unreachable",
            observe=observe,
            decide=decide,
            is_satisfied=lambda _n: False,
            max_cycles=3,
        )
    )

    assert reached.to_dict()["outcome"] == "goal_reached"
    assert exhausted.to_dict()["outcome"] == "out_of_cycles"
    assert "cycles" in reached.to_dict()


def test_plan_shaped_runs_are_unchanged():
    """run() must keep working exactly as it did."""

    async def act():
        return True

    receipt = asyncio.run(
        _executor().run("a written plan", [Step(name="one", action=act)])
    )

    assert receipt.completed is True
    assert receipt.outcome == ""
    assert receipt.cycles == 0
