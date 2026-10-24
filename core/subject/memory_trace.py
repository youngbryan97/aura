"""What a long run's process holds, and what grew since it last looked.

On 23 October the carrier's workers grew by 6.5 GB in the hour after they took
their anchors, while they scored cuts, and nothing said what grew. With
AURA_MEMORY_TRACE_MINUTES set, a process logs its resident memory and the
object kinds whose counts grew most since the previous look, every that many
minutes. Counting every object takes a second or two of the process's time,
so it is off unless asked for.

The count runs on the thread that owns the run, between the steps of its
tasks, never on the trace's own thread. On 24 October at 06:12 a carrier worker
died fifteen seconds after a look taken from that thread: the list
gc.get_objects() returns holds a reference to every tracked object, one of them
a tuple the main thread was still building inside a deepcopy, and CPython
resizes a tuple under construction only while it holds the one reference
("Objects/tupleobject.c:909: bad argument to internal function").
"""

from __future__ import annotations

import asyncio
import collections
import gc
import os
import threading
from collections.abc import Callable

__all__ = ["MEMORY_TRACE_ENV", "count_kinds", "describe_growth", "start_memory_trace"]

MEMORY_TRACE_ENV = "AURA_MEMORY_TRACE_MINUTES"


def _resident_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except (ImportError, OSError):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def count_kinds() -> collections.Counter[str]:
    """How many live objects of each kind the collector tracks.

    Counted in C, so the list that holds every object lives for a few
    instructions; a thread that is not the run's own can still be handed the
    interpreter in that window, which is why the look runs on the run's thread.
    """
    counts: dict[type, int] = {}
    collections._count_elements(counts, map(type, gc.get_objects()))
    named: collections.Counter[str] = collections.Counter()
    for kind, count in counts.items():
        named[kind.__qualname__] += count
    return named


def describe_growth(before: collections.Counter[str], after: collections.Counter[str], top: int) -> str:
    grown = sorted(((after[kind] - before.get(kind, 0), kind) for kind in after), reverse=True)[:top]
    return ", ".join(f"{kind} +{count}" for count, kind in grown if count > 0) or "nothing grew"


def start_memory_trace(log: Callable[[str], None], *, top: int = 12) -> threading.Thread | None:
    """Start the trace if AURA_MEMORY_TRACE_MINUTES asks for one; the thread, or None.

    Called from the run's event loop, each look is handed to that loop. Called
    with no loop running, the trace reports resident memory only.
    """
    try:
        minutes = float(os.environ.get(MEMORY_TRACE_ENV, "") or 0.0)
    except ValueError:
        minutes = 0.0
    if minutes <= 0:
        return None
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    held = {"kinds": count_kinds() if loop is not None else None, "resident": _resident_gb()}
    log(f"[memory] {held['resident']:.2f} GB resident at the first look")

    def look() -> None:
        now = _resident_gb()
        if held["kinds"] is None:
            log(f"[memory] {now:.2f} GB resident ({now - held['resident']:+.2f})")
        else:
            current = count_kinds()
            log(f"[memory] {now:.2f} GB resident ({now - held['resident']:+.2f}); "
                f"grew: {describe_growth(held['kinds'], current, top)}")
            held["kinds"] = current
        held["resident"] = now

    def watch() -> None:
        while not stop.wait(minutes * 60.0):
            if loop is None:
                look()
                continue
            try:
                loop.call_soon_threadsafe(look)
            except RuntimeError:
                return

    stop = threading.Event()
    thread = threading.Thread(target=watch, name="subject-memory-trace", daemon=True)
    thread.stop = stop  # type: ignore[attr-defined]
    thread.start()
    return thread
