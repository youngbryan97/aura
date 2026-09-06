"""One register for every scarce thing, in the order asked.

A lock cannot answer "who holds the screen", "how long did training wait",
"did anyone give up waiting". It is a boolean with an invisible queue whose
order the loop decides, which is not fairness: a caller that asks often can
starve one that asks once.
"""
from __future__ import annotations

import asyncio

import pytest

from core.runtime.what_stops_it import AnExecutionContext, Stopping
from core.runtime.who_gets_it_next import (
    THE_RESOURCES,
    GaveUp,
    claim,
    forget_everything,
    how_it_has_gone,
    observe_held,
    observe_released,
    who_holds_what,
    who_is_waiting,
)


@pytest.fixture(autouse=True)
def _a_clean_register():
    forget_everything()
    yield
    forget_everything()


# ------------------------------------------------------------- the table


def test_every_resource_says_why_it_is_scarce_and_who_grants_it():
    for name, row in THE_RESOURCES.items():
        assert row["scarce"], f"{name} does not say why it is scarce"
        assert row["granted"] in {"here", "elsewhere"}


def test_an_undeclared_resource_is_refused():
    async def go():
        async with claim("gpu_fan", "me"):
            pass

    with pytest.raises(KeyError, match="no such resource"):
        asyncio.run(go())


def test_a_resource_granted_elsewhere_cannot_be_claimed_here():
    """The model lane's own transaction evicts and compensates.

    A queue in front of it would take the decision away from the thing that
    can actually make it.
    """
    async def go():
        async with claim("model_lane", "me"):
            pass

    with pytest.raises(ValueError, match="granted elsewhere"):
        asyncio.run(go())


# --------------------------------------------------------------- holding


def test_a_claim_is_held_and_then_released():
    async def go():
        async with claim("screen", "an_actor"):
            held = who_holds_what()
            assert held["screen"]["by"] == "an_actor"
        assert who_holds_what() == {}

    asyncio.run(go())


def test_the_same_holder_re_entering_does_not_deadlock():
    """The one case a plain lock turns into a hang."""
    async def go():
        async with claim("screen", "same"):
            async with claim("screen", "same"):
                assert who_holds_what()["screen"]["depth"] == 2
            assert who_holds_what()["screen"]["depth"] == 1
        assert who_holds_what() == {}
        assert how_it_has_gone()["screen"]["reentered"] == 1

    asyncio.run(go())


def test_a_claim_is_released_even_when_the_work_raises():
    async def go():
        with pytest.raises(ZeroDivisionError):
            async with claim("screen", "unlucky"):
                raise ZeroDivisionError
        assert who_holds_what() == {}

    asyncio.run(go())


# ------------------------------------------------------------- fairness


def test_the_queue_is_served_in_the_order_it_was_asked():
    got: list[str] = []

    async def hold(name: str, for_s: float):
        async with claim("screen", name):
            got.append(name)
            await asyncio.sleep(for_s)

    async def go():
        first = asyncio.create_task(hold("A", 0.08))
        await asyncio.sleep(0.01)
        rest = []
        for name in ("B", "C", "D"):
            rest.append(asyncio.create_task(hold(name, 0.005)))
            await asyncio.sleep(0.003)
        assert [one["by"] for one in who_is_waiting()["screen"]] == ["B", "C", "D"]
        await asyncio.gather(first, *rest)

    asyncio.run(go())
    assert got == ["A", "B", "C", "D"]


def test_waiting_is_measured_not_guessed():
    async def hold():
        async with claim("training", "first"):
            await asyncio.sleep(0.05)

    async def go():
        held = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        async with claim("training", "second"):
            pass
        await held

    asyncio.run(go())
    record = how_it_has_gone()["training"]
    assert record["granted"] == 2
    assert record["waited_s_worst"] > 0.0


# --------------------------------------------------- the two ways to fail


def test_running_out_of_time_is_reported_as_running_out_of_time():
    async def hold():
        async with claim("training", "hog"):
            await asyncio.sleep(0.3)

    async def go():
        held = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        with pytest.raises(GaveUp) as caught:
            async with claim("training", "patient", seconds=0.03):
                pass
        assert caught.value.reason == "ran out of time"
        assert caught.value.resource == "training"
        held.cancel()

    asyncio.run(go())
    assert how_it_has_gone()["training"]["timed_out"] == 1
    assert how_it_has_gone()["training"]["stopped"] == 0


def test_being_stopped_while_queued_is_not_a_timeout():
    """They need different fixes, so they are counted separately.

    A timeout says the holder is too slow or the deadline too tight. A stop
    says the caller's own work was cancelled while it queued, which is not
    the resource's fault at all.
    """
    stopping = Stopping("a turn")

    async def hold():
        async with claim("screen", "hog"):
            await asyncio.sleep(0.2)

    async def go():
        held = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        context = AnExecutionContext(stopping=stopping, doing="a turn")
        waiting = asyncio.create_task(_wait(context))
        await asyncio.sleep(0.01)
        stopping.stop("the user left")
        held.cancel()
        with pytest.raises(asyncio.CancelledError):
            waiting.cancel()
            await waiting

    async def _wait(context):
        async with claim("screen", "polite", context=context):
            pass

    asyncio.run(go())
    assert how_it_has_gone()["screen"]["timed_out"] == 0


def test_the_deadline_comes_from_the_caller_not_from_here():
    """A number written in the claim manager is wrong for somebody."""
    async def hold():
        async with claim("training", "hog"):
            await asyncio.sleep(0.3)

    async def go():
        held = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        context = AnExecutionContext(doing="a short turn").under("waiting", seconds=0.02)
        with pytest.raises(GaveUp):
            async with claim("training", "brief", context=context):
                pass
        held.cancel()

    asyncio.run(go())


# ------------------------------------------------- granted somewhere else


def test_a_resource_granted_elsewhere_still_shows_who_holds_it():
    observe_held("model_lane", "mlx:1234", trace="receipt-9")
    held = who_holds_what()["model_lane"]
    assert held["by"] == "mlx:1234"
    assert held["trace"] == "receipt-9"
    observe_released("model_lane", "mlx:1234")
    assert "model_lane" not in who_holds_what()


def test_a_release_by_somebody_else_does_not_clear_the_holder():
    observe_held("model_lane", "mlx:1")
    observe_released("model_lane", "mlx:2")
    assert who_holds_what()["model_lane"]["by"] == "mlx:1"


def test_a_resource_granted_here_cannot_be_observed_instead():
    with pytest.raises(ValueError, match="granted here"):
        observe_held("screen", "somebody")


# ------------------------------------------------------------ the wiring


def test_the_screen_is_claimed_by_the_thing_that_moves_the_pointer():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "tools" / "computer_use.py"
    ).read_text("utf-8")
    assert 'claim("screen"' in source


def test_training_is_claimed_by_the_thing_that_holds_the_gpu():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "learning" / "lora_trainer.py"
    ).read_text("utf-8")
    assert 'claim("training"' in source


def test_the_register_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["who_holds_what"]
    assert set(block) == {"held", "waiting", "record"}
