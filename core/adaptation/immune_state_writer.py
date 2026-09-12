"""A single-slot background writer for state that is rebuilt, not appended.

The immune ecology is serialised under the lock that guards it — it has to
be, because the payload reads the cell population, the tissue field, the
lineage table and the expansion engine, and a snapshot taken while another
thread mutates them is a snapshot of nothing. The fsync that follows needs
none of that, and doing it there is how the runtime freezes: lockdep
reported ``fsync attempted while holding
['core.adaptation.adaptive_immunity._lock']`` during a subject-core run,
the same class as the on-loop fsync that once stopped the live event loop
for twenty minutes.

So the payload is handed here and the caller releases the lock. One slot,
replaced rather than queued: a newer snapshot of the same state makes the
older one worthless, and a queue would make a burst of observations cost a
burst of writes — which is the amplification the coalescing interval in
``_save_state`` already exists to stop. The slot is the same idea one
layer down, and it holds across the window where a write is in flight.

Durability is explicit. :meth:`flush` drains the slot and waits for the
write to finish, and it does the write itself when no thread is running,
so a caller that needs a durable point gets one without the writer having
to exist. Callers that hold the serialising lock must not call it: waiting
on a disk write with the lock held costs exactly what doing the write
there cost.

The thread retires after an idle period and restarts on the next deferred
submit. A test suite that constructs thousands of these would otherwise
accumulate one idle thread apiece for the life of the process.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.ImmuneStateWriter")

#: How long the writer waits with an empty slot before retiring. Long enough
#: that a conversation's worth of observations reuses one thread, short
#: enough that a process holding many writers does not hold many threads.
DEFAULT_IDLE_TIMEOUT_S = 30.0

#: Default ceiling on how long a durable caller waits for the slot to drain.
#: A write that has not landed in this long is a sick disk, and blocking
#: past it turns one slow write into a stalled subsystem.
DEFAULT_FLUSH_TIMEOUT_S = 5.0


class SingleSlotStateWriter:
    """Write the newest payload off the caller's thread, one at a time."""

    def __init__(
        self,
        name: str,
        write: Callable[[Any], None],
        *,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._name = name
        self._write = write
        self._idle_timeout_s = float(idle_timeout_s)
        self._on_error = on_error
        self._slot_lock = checked_lock(f"{name}.slot", rank=LockRank.LEAF)
        self._slot: Any | None = None
        self._writing = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._wakeup = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._submitted = 0
        self._written = 0
        self._replaced = 0
        self._failed = 0
        self._inline = 0
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, payload: Any, *, background: bool) -> None:
        """Take the payload as the one to write next.

        ``background=True`` is the locked caller's path: it cannot wait, so
        a thread is started to do the write. ``background=False`` leaves the
        payload in the slot for the :meth:`flush` the caller is about to
        make, which keeps a durable write on the caller's own thread and
        starts no thread at all.
        """
        with self._slot_lock:
            if self._slot is not None:
                self._replaced += 1
            self._slot = payload
            self._submitted += 1
            self._idle.clear()
            if background and not self._stopping:
                self._ensure_thread_locked()
        self._wakeup.set()

    def flush(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> bool:
        """Make the pending payload durable. True when the slot is empty.

        Never call this while holding the lock the payload was built under.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            inline: Any | None = None
            with self._slot_lock:
                if self._slot is None and not self._writing:
                    return True
                thread = self._thread
                if thread is None or not thread.is_alive():
                    if self._slot is None:
                        # A write in flight on a thread that has since died.
                        return False
                    inline = self._slot
                    self._slot = None
                    self._writing = True
            if inline is not None:
                self._inline += 1
                self._write_once(inline)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._wakeup.set()
            self._idle.wait(min(remaining, 0.25))

    def stop(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> bool:
        """Drain what is pending and retire the thread, inside one budget.

        The drain and the join share ``timeout`` rather than each taking
        it, because a shutdown hook is given a budget for the whole call
        and a hook that doubles it reads as a hung subsystem.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._slot_lock:
            self._stopping = True
            thread = self._thread
        self._wakeup.set()
        drained = self.flush(max(0.0, deadline - time.monotonic()))
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return drained

    def stats(self) -> dict[str, Any]:
        with self._slot_lock:
            pending = self._slot is not None
            writing = self._writing
            running = self._thread is not None and self._thread.is_alive()
        return {
            "name": self._name,
            "submitted": self._submitted,
            "written": self._written,
            "coalesced": self._replaced,
            "failed": self._failed,
            "inline_writes": self._inline,
            "pending": pending,
            "writing": writing,
            "thread_running": running,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_thread_locked(self) -> None:
        """Start the writer if none is running. Caller holds the slot lock."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        try:
            thread = threading.Thread(
                target=self._run,
                name=f"{self._name}-writer",
                daemon=True,
            )
            thread.start()
        except RuntimeError as exc:
            # Out of threads, or interpreter shutdown. The payload stays in
            # the slot; the next flush writes it on the caller's thread.
            self._thread = None
            self._last_error = f"thread_start_failed: {exc}"
            logger.debug("%s: writer thread would not start: %s", self._name, exc)
            return
        self._thread = thread

    def _run(self) -> None:
        while True:
            woke = self._wakeup.wait(self._idle_timeout_s)
            self._wakeup.clear()
            self._drain()
            with self._slot_lock:
                if self._slot is not None:
                    continue
                self._idle.set()
                if self._stopping or not woke:
                    self._thread = None
                    return

    def _drain(self) -> None:
        while True:
            with self._slot_lock:
                payload = self._slot
                if payload is None:
                    return
                self._slot = None
                self._writing = True
            self._write_once(payload)

    def _write_once(self, payload: Any) -> None:
        try:
            self._write(payload)
            self._written += 1
        except Exception as exc:  # noqa: BLE001 — the writer must survive
            self._failed += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:  # noqa: BLE001 — reporting must not kill it
                    logger.debug("%s: error handler failed", self._name, exc_info=True)
            else:
                logger.debug("%s: state write failed: %s", self._name, exc)
        finally:
            with self._slot_lock:
                self._writing = False
                if self._slot is None:
                    self._idle.set()


__all__ = [
    "DEFAULT_FLUSH_TIMEOUT_S",
    "DEFAULT_IDLE_TIMEOUT_S",
    "SingleSlotStateWriter",
]
