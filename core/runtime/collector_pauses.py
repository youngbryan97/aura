"""How long the garbage collector stops her, by generation, measured before anything about it is changed (R06).

LIVE 12 November 2026, 09:31: lockdep reported her event loop held 363 ms inside `the_record_of_her_own_work`, 356 ms
of it on the CPU, in a section that checks one flag and sets it. Nothing in that section costs a third of a second. A
full collection set off by an allocation inside it does, and lockdep charges it to whatever lock is held at the time.
Her main process keeps millions of long-lived objects, and nothing in her runtime measured or tuned the collector.

This times every collection through ``gc.callbacks`` and says in her log each one that held its thread longer than
lockdep's loop budget (``LOOP_BLOCKING_HOLD_S``), with its generation, what it collected, and whether that thread was
her event loop. ``collector_pauses()`` gives the counts, totals and worst pause per generation. Nothing here changes
when or how the collector runs.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any

from core.runtime.lockdep import LOOP_BLOCKING_HOLD_S
from core.runtime.which_thread_may_do_this import the_loop_thread

__all__ = ["collector_pauses", "install_collector_timing", "uninstall_collector_timing"]

logger = logging.getLogger("Aura.CollectorPauses")

# Written only from inside a collection's callbacks, which the collector never runs two of at once, and read under
# the GIL. A lock here could be held by the very thread whose allocation started the collection.
_BEGAN: dict[int, float] = {}
_BY_GENERATION: dict[int, list[float]] = {}  # generation -> [count, total seconds, worst seconds, over budget]


def _timed(phase: str, info: dict[str, Any]) -> None:
    thread = threading.get_ident()
    if phase == "start":
        _BEGAN[thread] = time.perf_counter()
        return
    began = _BEGAN.pop(thread, None)
    if began is None:
        return
    took = time.perf_counter() - began
    generation = int(info.get("generation", 0) or 0)
    row = _BY_GENERATION.setdefault(generation, [0, 0.0, 0.0, 0])
    row[0] += 1
    row[1] += took
    row[2] = max(row[2], took)
    if took > LOOP_BLOCKING_HOLD_S:
        row[3] += 1
        on_the_loop = thread == the_loop_thread()
        logger.info(
            "🗑️ a generation-%d collection held %s %.0f ms (collected %s, uncollectable %s)",
            generation,
            "her event loop" if on_the_loop else "a thread",
            took * 1000.0,
            info.get("collected", "?"),
            info.get("uncollectable", "?"),
        )


def install_collector_timing() -> bool:
    """Start timing collections; True when it was not already on."""
    if _timed in gc.callbacks:
        return False
    gc.callbacks.append(_timed)
    return True


def uninstall_collector_timing() -> None:
    """Stop timing collections and forget what was measured."""
    while _timed in gc.callbacks:
        gc.callbacks.remove(_timed)
    _BEGAN.clear()
    _BY_GENERATION.clear()


def collector_pauses() -> dict[str, Any]:
    """Per generation: how many collections, their total and worst seconds, and how many passed the loop budget."""
    return {
        "budget_s": LOOP_BLOCKING_HOLD_S,
        "generations": {
            str(generation): {
                "collections": int(row[0]),
                "total_s": round(row[1], 4),
                "worst_s": round(row[2], 4),
                "over_budget": int(row[3]),
            }
            for generation, row in sorted(_BY_GENERATION.items())
        },
    }
