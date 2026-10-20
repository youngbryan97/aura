"""Work handed off the event loop to one long-lived thread, never a thread started per piece of work.

A thread started on the loop for each walk of the process tree put Thread.start on the loop thread: on
2026-10-18 at 16:00, with the host short of memory and processes being killed, her loop stood 86 s inside it.
Handing work to a thread already running costs a queue put.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger(__name__)

_WORK: queue.SimpleQueue = queue.SimpleQueue()
_WALKER: list[threading.Thread] = []
_WALKER_LOCK = checked_lock("walks_on_one_thread.walker", rank=LockRank.LEAF)


def done_by_the_walker(work: Callable[[], object], *, name: str = "children-rss-walker") -> None:
    """Queue ``work`` for the walker thread, starting the thread only when none is alive."""
    _WORK.put(work)
    if _WALKER and _WALKER[0].is_alive():
        return
    with _WALKER_LOCK:
        if _WALKER and _WALKER[0].is_alive():
            return
        walker = threading.Thread(target=_work_forever, name=name, daemon=True)
        _WALKER[:] = [walker]
        walker.start()


def _work_forever() -> None:
    while True:
        work = _WORK.get()
        try:
            work()
        except Exception as exc:  # noqa: BLE001 - one failed piece of work must not end the walker
            logger.debug("work on the walker thread failed: %s", exc)
