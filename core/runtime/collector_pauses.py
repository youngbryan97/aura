"""How long the garbage collector stops her, by generation, measured before anything about it is changed (R06).

LIVE 12 November 2026, 09:31: lockdep reported her event loop held 363 ms inside `the_record_of_her_own_work`, 356 ms
of it on the CPU, in a section that checks one flag and sets it. Nothing in that section costs a third of a second. A
full collection set off by an allocation inside it does, and lockdep charges it to whatever lock is held at the time.
Her main process keeps millions of long-lived objects, and nothing in her runtime measured or tuned the collector.

This times every collection through ``gc.callbacks`` and says in her log each one that held its thread longer than
lockdep's loop budget (``LOOP_BLOCKING_HOLD_S``), with its generation, what it collected, and whether that thread was
her event loop, and where it came from: a line that asked for it (``gc.collect``), or the line whose allocation set it
off. ``collector_pauses()`` gives the counts, totals and worst pause per generation. Nothing here changes when or how
the collector runs.
"""
from __future__ import annotations

import gc
import linecache
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any

from core.runtime.lockdep import LOOP_BLOCKING_HOLD_S
from core.runtime.which_thread_may_do_this import the_loop_thread

__all__ = ["asked_for_collections", "collector_pauses", "freeze_what_boot_made", "install_collector_timing",
           "say_what_was_measured", "uninstall_collector_timing"]

logger = logging.getLogger("Aura.CollectorPauses")

# Written only from inside a collection's callbacks, which the collector never runs two of at once, and read under
# the GIL. A lock here could be held by the very thread whose allocation started the collection.
_BEGAN: dict[int, float] = {}
_BY_GENERATION: dict[int, list[float]] = {}  # generation -> [count, total seconds, worst seconds, over budget]
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASKS_FOR_ONE = re.compile(r"\bgc\.collect\(")
#: Each line that asked for collections: (when, seconds, generation, collected), the newest kept. Every one it asked
#: for, not only those past the loop budget, so that a change to what it asks for is measured on the same terms.
_ASKED_FOR: dict[str, list[tuple[float, float, int, int]]] = {}
_ASKED_FOR_KEPT = 512

#: A collection past the budget, waiting to be said. A collection runs on whatever thread allocated, inside whatever
#: that thread holds, so it is said from another thread: logging takes the handlers' locks, and a handler can take a
#: lock the collecting thread already holds. LIVE 17 November 2026, 14:07 local, a full collection set off inside
#: lockdep's own bookkeeping (core/runtime/lockdep.py, on_acquire, under its registry lock) logged from here; her log
#: handler took a checked lock, lockdep asked for its registry lock again on the same thread, and every thread that
#: took a checked lock after it, her event loop among them, waited for good. SimpleQueue.put is reentrant: Python
#: documents it as safe from __del__ and weakref callbacks, which run where this does.
_UNSAID: queue.SimpleQueue = queue.SimpleQueue()
_REPORTER: threading.Thread | None = None


def _asked_for_at(frame: Any) -> str:
    """``path:line`` of the line that asked for this collection, or "" when an allocation set it off."""
    if frame is None:
        return ""
    path = frame.f_code.co_filename
    if not _ASKS_FOR_ONE.search(linecache.getline(path, frame.f_lineno)):
        return ""
    shown = os.path.relpath(path, _ROOT) if path.startswith(_ROOT + os.sep) else os.path.join(*path.split(os.sep)[-2:])
    return f"{shown}:{frame.f_lineno}"


def _where_it_came_from(frame: Any) -> str:
    """The line that asked for this collection, or the line whose allocation set it off.

    LIVE 16 November 2026: 180 full collections past the loop budget in 40 minutes, one every 11 s, 19 s in all, each
    finding about 400 objects to free. Several of her modules ask for full collections on timers, and the collector
    also starts them itself; which of the two these were decides what to change, and nothing said.
    """
    if frame is None:
        return "from nowhere in Python"
    path = frame.f_code.co_filename
    shown = os.path.relpath(path, _ROOT) if path.startswith(_ROOT + os.sep) else os.path.join(
        *path.split(os.sep)[-2:])
    if _ASKS_FOR_ONE.search(linecache.getline(path, frame.f_lineno)):
        return f"asked for at {shown}:{frame.f_lineno}"
    return f"set off by an allocation at {shown}:{frame.f_lineno}"


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
    # A young collection runs every few thousand allocations; only an older one is looked up to see whether a line
    # asked for it, and a line asking for a young one says so in its argument.
    asked = _asked_for_at(sys._getframe(1)) if generation >= 1 else ""
    if asked:
        held = _ASKED_FOR.setdefault(asked, [])
        held.append((time.time(), took, generation, int(info.get("collected", 0) or 0)))
        del held[:-_ASKED_FOR_KEPT]
    if took > LOOP_BLOCKING_HOLD_S:
        row[3] += 1
        _UNSAID.put((generation, thread == the_loop_thread(), took, info.get("collected", "?"),
                     info.get("uncollectable", "?"), _where_it_came_from(sys._getframe(1))))


def _say(measured: tuple[Any, ...]) -> None:
    generation, on_the_loop, took, collected, uncollectable, where = measured
    logger.info(
        "🗑️ a generation-%d collection held %s %.0f ms (collected %s, uncollectable %s), %s",
        generation, "her event loop" if on_the_loop else "a thread", took * 1000.0, collected, uncollectable, where,
    )


def _report() -> None:
    while True:
        item = _UNSAID.get()
        if item is None:
            return
        if isinstance(item, threading.Event):
            item.set()
            continue
        _say(item)


def _say_here() -> None:
    while True:
        try:
            item = _UNSAID.get_nowait()
        # not a failure: an empty queue is everything already said.
        except queue.Empty:
            return
        if isinstance(item, threading.Event):
            item.set()
        elif item is not None:
            _say(item)


def say_what_was_measured(timeout_s: float = 5.0) -> None:
    """Every collection measured so far past the budget is in her log when this returns, for a reader that needs it
    now. Not to be called from inside a collection."""
    reporter = _REPORTER
    if reporter is None or not reporter.is_alive():
        _say_here()
        return
    said = threading.Event()
    _UNSAID.put(said)
    said.wait(timeout_s)


def install_collector_timing() -> bool:
    """Start timing collections; True when it was not already on."""
    global _REPORTER
    if _timed in gc.callbacks:
        return False
    if _REPORTER is None or not _REPORTER.is_alive():
        _REPORTER = threading.Thread(target=_report, name="aura-collector-pauses", daemon=True)
        _REPORTER.start()
    gc.callbacks.append(_timed)
    return True


def uninstall_collector_timing() -> None:
    """Stop timing collections, say what was measured, and forget it."""
    global _REPORTER
    while _timed in gc.callbacks:
        gc.callbacks.remove(_timed)
    reporter, _REPORTER = _REPORTER, None
    if reporter is not None and reporter.is_alive():
        _UNSAID.put(None)
        reporter.join(timeout=5.0)
    _say_here()
    _BEGAN.clear()
    _BY_GENERATION.clear()
    _ASKED_FOR.clear()


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


def asked_for_collections(since: float = 0.0, until: float = float("inf")) -> dict[str, dict[str, Any]]:
    """Per line that asked for collections, between ``since`` and ``until``: how many, what they held their thread in
    all and at the median, what they collected at the median, and of which generations."""
    out: dict[str, dict[str, Any]] = {}
    for site, held in list(_ASKED_FOR.items()):
        rows = [row for row in list(held) if since <= row[0] < until]
        if not rows:
            continue
        took = sorted(row[1] for row in rows)
        collected = sorted(row[3] for row in rows)
        out[site] = {"collections": len(rows), "total_s": round(sum(took), 4),
                     "median_s": round(took[len(took) // 2], 4), "median_collected": collected[len(collected) // 2],
                     "generations": sorted({row[2] for row in rows}), "first_at": rows[0][0], "last_at": rows[-1][0]}
    return out


def freeze_what_boot_made() -> int:
    """Leave everything her boot made out of every later collection; how many objects that is.

    LIVE 12 November 2026, three hours of her runtime timed by this module: 269 full (generation-2) collections, a
    median of about 0.4 s each and the worst 2.7 s, every one of them holding the interpreter lock, so her event loop
    stood still for each one wherever it ran, every 40 s or so. A full collection walks every container object the
    process holds, and almost all of hers were made at boot and live until it exits: services, registries, tables,
    loaded modules. Collected once and then frozen (gc.freeze), they sit in the permanent generation, and a later full
    collection walks only what was made since. What was frozen is not collected again: a cycle among boot objects that
    becomes garbage later stays, which for objects that live as long as the process is no loss.
    """
    gc.collect(1)
    gc.freeze()
    frozen = gc.get_freeze_count()
    logger.info("🗑️ %d objects made during boot are frozen out of every later collection", frozen)
    return frozen
