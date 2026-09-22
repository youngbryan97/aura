"""A value collected off the event loop and read from a snapshot on it.

The integrity block of the health report learned this the hard way (see
``core/runtime/health_contract.py``, 2026-07-29: a 103.8s stall from a
health report collecting inline on the loop). The health FRAGMENTS kept
collecting inline — a connector registry read from disk inside a health
pulse was a 5.8s loop stall on 2026-09-16 — so the discipline is one class
that any collector can wear:

- off the loop, collect behind a TTL, so a burst of readers costs one
  collection;
- on the loop, serve the last snapshot and ask one daemon thread to
  refresh it, so the loop pays for a dict copy and nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.OffLoopSnapshot")

T = TypeVar("T")


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    # not a failure: no running loop is the answer to the question asked.
    except RuntimeError:
        return False
    return True


class OffLoopSnapshot(Generic[T]):
    """One collector, one snapshot, one refresher thread at a time."""

    def __init__(
        self,
        name: str,
        collect: Callable[[], T],
        *,
        ttl_s: float,
        empty: Callable[[], T],
    ) -> None:
        self.name = str(name)
        self._collect = collect
        self._ttl_s = max(0.0, float(ttl_s))
        self._empty = empty
        self._lock = checked_lock(f"off_loop_snapshot.{self.name}", rank=LockRank.LEAF)
        # The collection lock is held across the collect and takes the state
        # lock inside it, so it ranks above the leaf.
        self._collection_lock = checked_lock(
            f"off_loop_snapshot.{self.name}.collection", rank=LockRank.RESOURCE
        )
        self._value: T | None = None
        self._at = 0.0
        self._refreshing = False
        self.collections = 0
        self.loop_serves = 0

    def reset_for_test(self) -> None:
        with self._lock:
            self._value = None
            self._at = 0.0
            self._refreshing = False
            self.collections = 0
            self.loop_serves = 0

    def _fresh_locked(self) -> bool:
        return self._value is not None and (time.monotonic() - self._at) < self._ttl_s

    def _collect_now(self) -> T:
        with self._collection_lock:
            with self._lock:
                if self._fresh_locked():
                    return self._value  # type: ignore[return-value]
            try:
                value = self._collect()
            except Exception as exc:  # noqa: BLE001 — a snapshot survives its collector
                logger.warning("%s: collection failed: %s", self.name, exc)
                value = self._empty()
            with self._lock:
                self._value = value
                self._at = time.monotonic()
                self.collections += 1
            return value

    def _refresh_off_loop(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True

        def _run() -> None:
            try:
                self._collect_now()
            finally:
                with self._lock:
                    self._refreshing = False

        threading.Thread(target=_run, name=f"snapshot-{self.name}", daemon=True).start()

    def read(self) -> tuple[T, dict[str, Any]]:
        """The value and how it was obtained.

        The second element says whether this was a fresh collection, the
        last snapshot (with its age), or nothing yet (``warming``).
        """
        with self._lock:
            fresh = self._fresh_locked()
            value = self._value
            age = (time.monotonic() - self._at) if value is not None else None
        if fresh:
            return value, {"collected": True, "stale": False, "age_s": round(age or 0.0, 3)}  # type: ignore[return-value]
        if not _on_event_loop():
            return self._collect_now(), {"collected": True, "stale": False, "age_s": 0.0}
        self._refresh_off_loop()
        with self._lock:
            self.loop_serves += 1
        if value is None:
            return self._empty(), {"collected": False, "warming": True, "age_s": None}
        return value, {"collected": True, "stale": True, "age_s": round(age or 0.0, 3)}


__all__ = ["OffLoopSnapshot"]
