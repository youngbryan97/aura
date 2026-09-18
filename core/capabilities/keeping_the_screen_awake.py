"""Keeping the display awake for as long as she is working on it.

A machine locks itself when nobody has touched it, and nobody has: she is
working with keys addressed to a process and a picture taken of a window, so
from the outside the machine looks idle. LIVE 2026-09-17: the screen locked on
move 226 of a game she had been asked to play and was winning, and everything
after that was a run waiting for somebody to come back.

Held only while she is actually doing something on the screen, and let go the
moment she stops, so a machine she is not using locks itself as it should. A
person locking their own screen still locks it: this prevents the idle timer,
not the person.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("Aura.KeepingTheScreenAwake")

__all__ = ["keeping_it_awake", "it_is_being_kept_awake"]

_LOCK = threading.Lock()
#: The holder, and how many things are relying on it.
_HOLDING: Any = None
_RELYING = 0


def it_is_being_kept_awake() -> bool:
    holder = _HOLDING
    return bool(holder is not None and holder.poll() is None)


def _take_hold(why: str) -> Any:
    """Ask the system to keep the display on. None when it will not."""
    if os.uname().sysname != "Darwin":  # pragma: no cover - one machine here
        return None
    try:
        # Tied to this process: if Aura goes, the hold goes with it rather
        # than leaving a machine that never sleeps again.
        return subprocess.Popen(  # noqa: S603 - a system tool, fixed arguments
            ["/usr/bin/caffeinate", "-d", "-i", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError) as exc:
        logger.info("the display will sleep as usual while %s: %s", why, exc)
        return None


def _let_go(holder: Any) -> None:
    if holder is None:
        return
    try:
        holder.terminate()
        holder.wait(timeout=2.0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        try:
            holder.kill()
        except OSError:
            pass


@contextmanager
def keeping_it_awake(why: str = "she is working on the screen") -> Iterator[bool]:
    """Keep the display awake for the length of this block.

    Yields whether the hold was taken, which is a fact worth saying out loud
    rather than assuming. Nested holds share one: the last one out lets go.
    """
    global _HOLDING, _RELYING
    with _LOCK:
        first = _RELYING == 0
        if first and not it_is_being_kept_awake():
            _HOLDING = _take_hold(why)
            if _HOLDING is not None:
                logger.info("keeping the display awake while %s", why)
        _RELYING += 1
        held = it_is_being_kept_awake()
    try:
        yield held
    finally:
        with _LOCK:
            _RELYING = max(0, _RELYING - 1)
            done = _RELYING == 0
            holder, _HOLDING = (_HOLDING, None) if done else (None, _HOLDING)
        if done and holder is not None:
            logger.info("letting the display sleep again")
            _let_go(holder)
