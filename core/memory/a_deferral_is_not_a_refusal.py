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
import time
from collections import deque
from typing import Any, Callable, Generic, TypeVar

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

_T = TypeVar("_T")


def is_a_deferral(reason: Any) -> bool:
    """Whether this refusal reason says "not yet" rather than "no".

    A rejection and a deferral arrive through the same boolean, and the only
    thing separating them is the reason string the governor wrote.
    """

    text = str(reason or "").lower()
    return any(marker in text for marker in DEFERRAL_MARKERS)


class DeferredWrites(Generic[_T]):
    """What the governor put off, and the next chance to make it.

    ``retry`` is given one held item and answers whether it landed. Anything
    it raises counts as not landing, and the item stays queued: a replay that
    can kill its caller is worse than the loss it exists to prevent.
    """

    def __init__(
        self,
        lane: str,
        retry: Callable[[_T], bool],
        *,
        limit: int = 256,
        per_replay: int = 4,
        interval_s: float = 5.0,
    ) -> None:
        self.lane = str(lane)
        self._retry = retry
        self._per_replay = max(1, int(per_replay))
        self._interval_s = max(0.0, float(interval_s))
        self._held: deque[_T] = deque(maxlen=max(1, int(limit)))
        self._next_at = 0.0
        self._shed = 0
        self._landed = 0
        self._held_total = 0

    def __len__(self) -> int:
        return len(self._held)

    @property
    def shed(self) -> int:
        """How many were dropped because the queue was full. Never silent."""

        return self._shed

    @property
    def landed(self) -> int:
        return self._landed

    def hold(self, item: _T, reason: str = "") -> None:
        """Keep a write the governor deferred."""

        if len(self._held) == self._held.maxlen:
            self._shed += 1
            logger.warning(
                "%s: deferred queue full at %d; shedding the oldest (%d shed so far)",
                self.lane,
                self._held.maxlen,
                self._shed,
            )
        self._held.append(item)
        self._held_total += 1
        if self._next_at <= 0.0:
            self._next_at = time.monotonic() + self._interval_s
        # The first, and then a line per _SAY_EVERY. A governor that defers
        # steadily makes this the most frequent line in the feed, and the
        # useful facts — that the queue exists, how deep it is, and that
        # things are landing — survive being said periodically. `state()`
        # carries the exact numbers for anything that wants them.
        if self._held_total == 1 or self._held_total % _SAY_EVERY == 0:
            logger.info(
                "%s: holding deferred writes (%d queued, %d held so far, "
                "%d landed): %s",
                self.lane,
                len(self._held),
                self._held_total,
                self._landed,
                str(reason)[:120],
            )

    def replay(self) -> int:
        """Try the ones held. Returns how many landed."""

        if not self._held:
            self._next_at = 0.0
            return 0
        now = time.monotonic()
        if now < self._next_at:
            return 0
        landed = 0
        for _ in range(min(self._per_replay, len(self._held))):
            item = self._held.popleft()
            try:
                ok = bool(self._retry(item))
            except Exception as exc:  # noqa: BLE001 — a replay must not kill its caller
                logger.debug("%s: deferred replay raised: %s", self.lane, exc)
                ok = False
            if ok:
                landed += 1
                self._landed += 1
            else:
                # Back where it came from, not onto the end. Appending would
                # reorder the queue every time a replay failed, and the order
                # is the order the writes were made in.
                self._held.appendleft(item)
                break
        self._next_at = now + self._interval_s if self._held else 0.0
        return landed

    def state(self) -> dict[str, Any]:
        """What is waiting, for the health surface."""

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
