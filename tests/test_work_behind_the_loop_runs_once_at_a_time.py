"""Synchronous work reached from the loop runs off it, one run at a time.

LIVE 2026-09-19: the experience stream saved on every committed frame, from
the loop, and its journal append fsynced on the event loop thread.
"""
from __future__ import annotations

import asyncio
import threading
import time

from core.runtime.executors import behind_the_loop


def test_off_the_loop_it_runs_here():
    ran: list[str] = []
    assert behind_the_loop("t-here", lambda: ran.append(threading.current_thread().name))
    assert ran == [threading.current_thread().name]


def test_on_the_loop_runs_never_overlap_and_the_last_ask_is_honoured():
    active = {"now": 0, "most": 0}
    runs: list[int] = []
    state = {"value": 0}

    def work() -> None:
        active["now"] += 1
        active["most"] = max(active["most"], active["now"])
        time.sleep(0.02)
        runs.append(state["value"])
        active["now"] -= 1

    async def burst() -> None:
        loop_thread = threading.current_thread().name
        for n in range(1, 11):
            state["value"] = n
            assert not behind_the_loop("t-burst", work)
        for _ in range(200):
            if runs and runs[-1] == 10:
                break
            await asyncio.sleep(0.01)
        assert loop_thread

    asyncio.run(burst())
    assert active["most"] == 1, "two runs for one key never overlap"
    assert runs[-1] == 10, "the last state asked for is saved"
    assert len(runs) < 10, "asks while one is queued collapse into one more run"


def test_when_the_lane_refuses_the_work_leaves_the_loop_anyway(monkeypatch):
    """LIVE 2026-09-20: the shutdown record fsynced on the loop.

    The first thing to ask for work behind the loop after the shutdown fence
    is set is the shutdown record itself, and the lane refuses new work once
    the fence is set. Paying for it inline put the write on the loop thread.
    """
    from core.runtime import executors

    def refuse(*_args, **_kwargs):
        raise RuntimeError("blocking lane closed for shutdown")

    monkeypatch.setattr(executors, "submit_blocking_io", refuse)
    ran: list[str] = []
    done = threading.Event()

    def work() -> None:
        ran.append(threading.current_thread().name)
        done.set()

    async def ask() -> str:
        assert not behind_the_loop("t-refused", work)
        return threading.current_thread().name

    loop_thread = asyncio.run(ask())
    assert done.wait(2.0), "the work never ran"
    assert ran and ran[0] != loop_thread, "the work ran on the loop thread"
    assert ran[0].startswith("behind_the_loop:t-refused")
