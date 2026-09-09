"""A write the governor put off, kept until it can be made.

The ontogeny organ decides admission at the executive control point, and
"deferred" is one of the three actions it is allowed to EXPLORE with. The
comment beside that choice says why it is safe: a rejection cancels work,
where "a deferral or a constrained approval only costs time."

That is true only where somebody comes back. Ten modules in this tree gate a
memory write through the constitutional core, and nine of them read
``approved == False`` and drop what they were holding. So the organ explores
by deferring, the record is destroyed, and the exploration's own premise —
that a deferral costs time — is false at the call site.

The tenth is ``core/world_model/acg.py``, which found it first: 108 causal
links dropped in one sampled window, systematically rather than randomly,
because the thing deferring is a policy and will defer again under the same
conditions.

This is that mechanism, shared, so a caller gets it by using it rather than
by reimplementing it.

    is_a_deferral(reason)       "not now" as against "no"
    DeferredWrites(...)         a bounded queue with replay

Bounded on purpose. A queue that grows while the organ keeps deferring is a
leak wearing a fix's clothes, and the oldest entry is shed first because the
newest is the one most likely to still describe what happened.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.DeferredWrites")

__all__ = [
    "DEFERRAL_MARKERS",
    "DeferredWrites",
    "is_a_deferral",
]

#: Outcomes that mean "not now" rather than "no". Extended from the set
#: core/world_model/acg.py arrived at by reading live reasons.
DEFERRAL_MARKERS: tuple[str, ...] = (
    "defer",
    "not_now",
    "not now",
    "capacity_full",
    "backpressure",
    "try_again",
    "retry_after",
)

#: How often a steady stream of deferrals says so. Every hold logged is how a
#: working mechanism becomes the loudest thing in the feed.
_SAY_EVERY = 25

T = TypeVar("T")


def is_a_deferral(reason: Any) -> bool:
    """Whether this refusal reason says "not yet" rather than "no".

    A rejection and a deferral arrive through the same boolean, and the only
    thing separating them is the reason string the governor wrote.
    """

    text = str(reason or "").lower()
    return any(marker in text for marker in DEFERRAL_MARKERS)


class DeferredWrites[T]:
    """What the governor put off, and the next chance to make it.

    ``retry`` is given one held item and answers whether it landed. Anything
    it raises counts as not landing, and the item stays queued: a replay that
    can kill its caller is worse than the loss it exists to prevent.
    """

    def __init__(
        self,
        lane: str,
        retry: Callable[[T], bool],
        *,
        limit: int = 256,
        per_replay: int = 4,
        interval_s: float = 5.0,
        identity: Callable[[T], Hashable] = id,
    ) -> None:
        self.lane = str(lane)
        self._retry = retry
        self._per_replay = max(1, int(per_replay))
        self._interval_s = max(0.0, float(interval_s))
        self._held: deque[T] = deque(maxlen=max(1, int(limit)))
        self._identity = identity
        self._owned: set[Hashable] = set()
        lock_name = f"memory.deferred_writes.{self.lane}.{id(self):x}"
        self._state_lock = checked_lock(f"{lock_name}.state", reentrant=True)
        # A retry can call the same store, and the store offers another replay
        # before writing. Concurrent callers do the same. Only one drain owns
        # the deque; nested and competing drains leave it to that owner.
        # A claim, not a lock held across the work.
        #
        # This was a `checked_lock` acquired non-blockingly for the whole
        # replay, which is two faults in one. The retry writes an episode, so
        # a blocking disk write happened under a process-wide lock — the shape
        # that freezes a runtime and the one lockdep exists to catch. And a
        # replay can reach `record_episode`, which replays: lockdep sees the
        # same thread reaching for a lock it already holds and reports a
        # self-deadlock before the non-blocking acquire can decline.
        #
        # LIVE, 2026-09-09, in a single boot: `LOCKDEP self_deadlock:
        # non-reentrant lock 'memory.deferred_writes.episodic_memory...replay'`
        # and `LOCKDEP blocking_op_under_lock: fsync attempted while holding
        # [...replay]`, and the runtime tainted for a lock-order violation.
        #
        # The claim is taken and released under `_state_lock`, which is held
        # only across the bookkeeping, so nothing is held while an episode is
        # written and a nested call finds the claim taken and leaves.
        self._replaying_on: int | None = None
        self._next_at = 0.0
        self._shed = 0
        self._landed = 0
        self._held_total = 0

    def __len__(self) -> int:
        with self._state_lock:
            return len(self._held)

    @property
    def shed(self) -> int:
        """How many were dropped because the queue was full. Never silent."""

        with self._state_lock:
            return self._shed

    @property
    def landed(self) -> int:
        with self._state_lock:
            return self._landed

    def hold(self, item: T, reason: str = "") -> None:
        """Keep a write the governor deferred."""

        key = self._identity(item)
        with self._state_lock:
            # The drain retains custody while the store retries. A renewed
            # deferral of that write must not enqueue another obligation.
            if key in self._owned:
                return
            if len(self._owned) == self._held.maxlen:
                self._shed += 1
                logger.warning(
                    "%s: deferred capacity full at %d; shedding one pending write (%d shed so far)",
                    self.lane,
                    self._held.maxlen,
                    self._shed,
                )
                if not self._held:
                    # The sole slot belongs to an in-flight write.
                    return
                self._owned.remove(self._identity(self._held.popleft()))
            self._held.append(item)
            self._owned.add(key)
            self._held_total += 1
            if self._next_at <= 0.0:
                self._next_at = time.monotonic() + self._interval_s
            held_total = self._held_total
            queued = len(self._held)
            landed = self._landed
        # The first, and then a line per _SAY_EVERY. A governor that defers
        # steadily makes this the most frequent line in the feed, and the
        # useful facts — that the queue exists, how deep it is, and that
        # things are landing — survive being said periodically. `state()`
        # carries the exact numbers for anything that wants them.
        if held_total == 1 or held_total % _SAY_EVERY == 0:
            logger.info(
                "%s: holding deferred writes (%d queued, %d held so far, "
                "%d landed): %s",
                self.lane,
                queued,
                held_total,
                landed,
                str(reason)[:120],
            )

    def replay(self) -> int:
        """Try the ones held. Returns how many landed."""

        me = threading.get_ident()
        with self._state_lock:
            if self._replaying_on is not None:
                # Somebody is already replaying, possibly this very call
                # further up the stack. Either way there is nothing to do.
                return 0
            self._replaying_on = me
        try:
            with self._state_lock:
                if not self._held:
                    self._next_at = 0.0
                    return 0
                now = time.monotonic()
                if now < self._next_at:
                    return 0
                attempts = min(self._per_replay, len(self._held))
            landed = 0
            for _ in range(attempts):
                with self._state_lock:
                    if not self._held:
                        break
                    item = self._held.popleft()
                try:
                    ok = bool(self._retry(item))
                except Exception as exc:  # noqa: BLE001 — a replay must not kill its caller
                    logger.debug("%s: deferred replay raised: %s", self.lane, exc)
                    ok = False
                with self._state_lock:
                    if ok:
                        landed += 1
                        self._landed += 1
                        self._owned.remove(self._identity(item))
                    else:
                        # Back where it came from, not onto the end. Appending
                        # would reorder the queue on every failed replay.
                        self._held.appendleft(item)
                        break
            with self._state_lock:
                self._next_at = now + self._interval_s if self._held else 0.0
            return landed
        finally:
            with self._state_lock:
                self._replaying_on = None

    def state(self) -> dict[str, Any]:
        """What is waiting, for the health surface."""

        with self._state_lock:
            return {
                "lane": self.lane,
                "queued": len(self._held),
                "capacity": self._held.maxlen,
                "shed": self._shed,
                "landed": self._landed,
                "held_total": self._held_total,
                "next_replay_in_s": max(0.0, self._next_at - time.monotonic())
                if self._next_at
                else 0.0,
            }
