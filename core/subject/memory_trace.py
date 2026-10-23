"""What a long run's process holds, and what grew since it last looked.

On 23 October the carrier's workers grew by 6.5 GB in the hour after they took
their anchors, while they scored cuts, and nothing said what grew. With
AURA_MEMORY_TRACE_MINUTES set, a process logs its resident memory and the
object kinds whose counts grew most since the previous look, every that many
minutes. Counting every object takes a second or two of the process's time,
so it is off unless asked for.
"""

from __future__ import annotations

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
    """How many live objects of each kind the collector tracks."""
    return collections.Counter(type(obj).__qualname__ for obj in gc.get_objects())


def describe_growth(before: collections.Counter[str], after: collections.Counter[str], top: int) -> str:
    grown = sorted(((after[kind] - before.get(kind, 0), kind) for kind in after), reverse=True)[:top]
    return ", ".join(f"{kind} +{count}" for count, kind in grown if count > 0) or "nothing grew"


def start_memory_trace(log: Callable[[str], None], *, top: int = 12) -> threading.Thread | None:
    """Start the trace if AURA_MEMORY_TRACE_MINUTES asks for one; the thread, or None."""
    try:
        minutes = float(os.environ.get(MEMORY_TRACE_ENV, "") or 0.0)
    except ValueError:
        minutes = 0.0
    if minutes <= 0:
        return None

    def watch() -> None:
        previous, resident = count_kinds(), _resident_gb()
        log(f"[memory] {resident:.2f} GB resident at the first look")
        while not stop.wait(minutes * 60.0):
            current, now = count_kinds(), _resident_gb()
            log(f"[memory] {now:.2f} GB resident ({now - resident:+.2f}); grew: {describe_growth(previous, current, top)}")
            previous, resident = current, now

    stop = threading.Event()
    thread = threading.Thread(target=watch, name="subject-memory-trace", daemon=True)
    thread.stop = stop  # type: ignore[attr-defined]
    thread.start()
    return thread
