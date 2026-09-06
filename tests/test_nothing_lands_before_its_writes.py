"""A checkpoint must not reach the disk before the writes that produced it."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.state.nothing_lands_before_its_writes import (
    TookTooLong,
    a_write_in_flight,
    forget_everything,
    how_the_drains_have_gone,
    still_in_flight,
    wait_for_the_writes,
    wait_for_the_writes_async,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


def test_a_write_is_in_flight_only_while_it_is_running() -> None:
    assert still_in_flight("turn") == ()
    with a_write_in_flight("turn", "channels.json"):
        assert len(still_in_flight("turn")) == 1
    assert still_in_flight("turn") == ()


def test_a_write_that_raised_is_not_left_in_flight() -> None:
    """A failed write left registered would block every checkpoint after it."""
    with pytest.raises(ZeroDivisionError):
        with a_write_in_flight("turn", "channels.json"):
            raise ZeroDivisionError
    assert still_in_flight("turn") == ()


def test_the_wait_returns_only_once_the_writes_have_landed() -> None:
    landed: list[str] = []

    def slow():
        with a_write_in_flight("turn-4", "channels.json"):
            time.sleep(0.2)
            landed.append("channels.json")

    worker = threading.Thread(target=slow)
    worker.start()
    time.sleep(0.02)
    wait_for_the_writes("turn-4")
    assert landed == ["channels.json"], "the checkpoint would have outrun it"
    worker.join()


def test_a_write_that_will_not_land_refuses_the_checkpoint_rather_than_hanging() -> None:
    stop = threading.Event()

    def stuck():
        with a_write_in_flight("turn-9", "wedged.json"):
            stop.wait(timeout=5.0)

    worker = threading.Thread(target=stuck)
    worker.start()
    try:
        time.sleep(0.02)
        started = time.monotonic()
        with pytest.raises(TookTooLong, match="wedged.json"):
            wait_for_the_writes("turn-9", seconds=0.3)
        assert time.monotonic() - started < 2.0, "it waited past its own deadline"
    finally:
        stop.set()
        worker.join()


def test_the_deadline_is_the_whole_wait_not_one_per_write() -> None:
    """Ten writes of a second each must not add up to ten deadlines."""
    stop = threading.Event()

    def stuck(name: str):
        with a_write_in_flight("many", name):
            stop.wait(timeout=5.0)

    workers = [threading.Thread(target=stuck, args=(f"w{n}.json",)) for n in range(6)]
    for one in workers:
        one.start()
    try:
        time.sleep(0.05)
        started = time.monotonic()
        with pytest.raises(TookTooLong):
            wait_for_the_writes("many", seconds=0.3)
        assert time.monotonic() - started < 1.5
    finally:
        stop.set()
        for one in workers:
            one.join()


def test_writes_for_another_scope_do_not_hold_this_checkpoint_up() -> None:
    stop = threading.Event()

    def stuck():
        with a_write_in_flight("someone else", "theirs.json"):
            stop.wait(timeout=5.0)

    worker = threading.Thread(target=stuck)
    worker.start()
    try:
        time.sleep(0.02)
        wait_for_the_writes("mine", seconds=0.5)
    finally:
        stop.set()
        worker.join()


def test_the_async_wait_does_not_block_the_loop() -> None:
    ticks: list[int] = []

    async def go():
        async def tick():
            for n in range(20):
                ticks.append(n)
                await asyncio.sleep(0.01)

        def slow():
            with a_write_in_flight("turn", "channels.json"):
                time.sleep(0.15)

        worker = threading.Thread(target=slow)
        worker.start()
        await asyncio.sleep(0.01)
        ticking = asyncio.create_task(tick())
        await wait_for_the_writes_async("turn")
        ticking.cancel()
        worker.join()

    asyncio.run(go())
    assert len(ticks) > 3, "the loop stopped while the drain ran"


def test_the_report_says_whether_a_checkpoint_ever_outran_its_writes() -> None:
    wait_for_the_writes("quiet")
    seen = how_the_drains_have_gone()
    assert seen["drains"] == 1
    assert seen["refused_checkpoints"] == 0
    assert "quiet" in seen["scopes"]
