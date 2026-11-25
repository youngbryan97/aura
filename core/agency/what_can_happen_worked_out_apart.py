"""What each side can do on a board, worked out in a process of its own while the other side thinks
(core/agency/what_each_area_of_a_board_comes_to.py ``possible_on_a_board``).

Worked out in a thread beside her, it shared one interpreter lock with her eyes. A look that lets go of the lock twenty
times, as reading the screen does, took 219 ms beside it against 8.5 ms alone (measured 24 November 2026), so she saw
their move late, on her own clock. In a process of its own it shares nothing with her but what it is given and what it
gives back, and it runs on a core of its own.

One process is kept for as long as the one that asked for it lives, as spawning one costs seconds: what it has worked
out it keeps, under the rules each reading was worked out under. Each request carries the count of requests stopped so
far; stopping them all is one more count, and the work under way ends at its next step, keeping what it had finished.
"""
from __future__ import annotations

import atexit
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

__all__ = ["WorkedOutApart", "worked_out_apart"]

#: In the worker: the count of requests stopped, shared with the process that asks.
_STOPPED: Any = None


def _begin(stopped: Any) -> None:
    global _STOPPED
    _STOPPED = stopped


def _possible(family: Any, position: Any, worth: dict[str, float], plies_at_most: int, seconds: float,
              asked_at: int) -> Any:
    from core.agency.what_each_area_of_a_board_comes_to import possible_on_a_board

    stopped = _STOPPED
    return possible_on_a_board(family, position, worth, plies_at_most=plies_at_most, seconds=seconds,
                               stop=lambda: stopped is not None and stopped.value != asked_at)


class WorkedOutApart:
    """A process that works out what each side can do, asked one position at a time."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._stopped = self._context.Value("q", 0)
        self._pool: ProcessPoolExecutor | None = None

    def submit(self, family: Any, position: Any, worth: dict[str, float], *, plies_at_most: int,
               seconds: float) -> Future:
        """Work out what each side can do from ``position``, for at most ``seconds``, until stopped."""
        # A pool whose worker died stays broken for good, so it is replaced: a host under memory pressure can take one.
        if self._pool is None or getattr(self._pool, "_broken", False):
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=self._context, initializer=_begin,
                                             initargs=(self._stopped,))
        with self._stopped.get_lock():
            asked_at = self._stopped.value
        return self._pool.submit(_possible, family, position, worth, plies_at_most, seconds, asked_at)

    def stop(self) -> None:
        """Every request so far ends at its next step, keeping what it had finished."""
        with self._stopped.get_lock():
            self._stopped.value += 1

    def shutdown(self) -> None:
        self.stop()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


_APART: WorkedOutApart | None = None


def worked_out_apart() -> WorkedOutApart:
    """The one process kept for this one."""
    global _APART
    if _APART is None:
        _APART = WorkedOutApart()
        atexit.register(_APART.shutdown)
    return _APART
