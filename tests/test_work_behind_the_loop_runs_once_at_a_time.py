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
