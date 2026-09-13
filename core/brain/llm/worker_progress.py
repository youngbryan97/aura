"""Request-bound inference activity that does not wait for the parent loop."""

from __future__ import annotations

import time
from typing import Any


def create_channel(context: Any) -> Any:
    """Allocate a fresh record for one worker incarnation."""
    return context.Array("Q", 3, lock=True)


def publish(channel: Any, request_seq: int, activity_seq: int) -> bool:
    """Publish real job activity without waiting on a reader."""
    if channel is None or request_seq <= 0:
        return False
    lock = channel.get_lock()
    if not lock.acquire(False):
        return False
    try:
        record = channel.get_obj()
        record[0] = request_seq
        record[1] = time.monotonic_ns()
        record[2] = activity_seq
    finally:
        lock.release()
    return True


def activity_age(channel: Any, request_seq: int) -> float | None:
    """Read this job's age; a held lock or another job gives no evidence.

    Nonblocking acquisition also covers a child dying during publication.
    Such a child cannot leave the parent waiting forever on its semaphore.
    """
    if channel is None or request_seq <= 0:
        return None
    lock = channel.get_lock()
    if not lock.acquire(False):
        return None
    try:
        record = channel.get_obj()
        observed_seq, at, count = (int(record[i]) for i in range(3))
    finally:
        lock.release()
    now = time.monotonic_ns()
    if observed_seq != request_seq or at <= 0 or at > now or count <= 0:
        return None
    return (now - at) / 1_000_000_000.0
