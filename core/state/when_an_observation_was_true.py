"""What a reading is a reading OF, and when it was true.

Soar draws an explicit input/output phase boundary, and the closure asked for
the same: every sensor and action channel declares a consistency mode —
sampled at tick, streaming, transactional — and a state snapshot records which
observation frontier it was taken at.

The reason is that a snapshot with a mixture is not a snapshot of anything. If
CPU load was read at the top of the tick, the screen halfway through and the
model's health from a five-minute-old cache, then "the state at 04:31" never
existed: no instant had all three of those values at once. A reader comparing
two such snapshots is comparing two mixtures.

Three modes, and they are answers to different questions:

* **sampled at tick** — read once at a boundary and held for the tick. Two
  readers in one tick see the same value, which is what makes a tick a
  moment.
* **streaming** — whatever arrived most recently. Two readers may differ, and
  the reading carries how old it is so a reader can refuse a stale one.
* **transactional** — read inside a transaction and consistent with the other
  reads in it, at the cost of holding something open.

A channel that declares none is the defect: nobody can tell whether its value
belongs to the snapshot it appears in.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhenAnObservationWasTrue")

__all__ = [
    "AChannelReading",
    "HowItIsRead",
    "declare_a_channel",
    "note_a_reading",
    "the_frontier",
    "what_does_not_say_how_it_is_read",
]


class HowItIsRead(StrEnum):
    """When a channel's value was true, relative to the tick."""

    SAMPLED_AT_TICK = "sampled at tick"
    STREAMING = "streaming"
    TRANSACTIONAL = "transactional"


@dataclass(frozen=True)
class AChannelReading:
    """One value, and when it was true."""

    channel: str
    at: float
    tick: int
    #: For a streaming channel: how old the value was when it was read.
    stale_by_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "at": self.at,
            "tick": self.tick,
            "stale_by_s": round(self.stale_by_s, 4),
        }


_HOW: dict[str, HowItIsRead] = {}
_LAST: dict[str, AChannelReading] = {}
_TICK = {"now": 0}
_LOCK = threading.RLock()


def declare_a_channel(channel: str, how: HowItIsRead) -> None:
    """Say when this channel's value is true. Declaring twice replaces."""
    with _LOCK:
        _HOW[str(channel)] = how


def note_a_reading(channel: str, *, measured_at: float | None = None) -> AChannelReading:
    """Record that this channel was read, and how old the value was.

    Refuses a channel that never declared how it is read: a reading nobody can
    place in time is worse than no reading, because it still appears in the
    snapshot.
    """
    name = str(channel)
    with _LOCK:
        how = _HOW.get(name)
        if how is None:
            raise KeyError(
                f"{name} has not said how it is read; declare it as one of "
                f"{', '.join(str(one) for one in HowItIsRead)}"
            )
        now = time.time()
        taken = float(measured_at if measured_at is not None else now)
        reading = AChannelReading(
            channel=name,
            at=taken,
            tick=_TICK["now"],
            stale_by_s=max(0.0, now - taken),
        )
        _LAST[name] = reading
        return reading


def a_tick_began() -> int:
    """Move the frontier. Everything sampled at tick is read again after this."""
    with _LOCK:
        _TICK["now"] += 1
        return _TICK["now"]


def the_frontier() -> dict[str, Any]:
    """Which observations a snapshot taken now would be made of.

    A snapshot carrying this can be compared with another; one without it is a
    mixture whose parts were true at different moments.
    """
    with _LOCK:
        tick = _TICK["now"]
        readings = {name: one.to_dict() for name, one in sorted(_LAST.items())}
        modes = {name: str(how) for name, how in sorted(_HOW.items())}
        behind = sorted(
            name
            for name, one in _LAST.items()
            if _HOW.get(name) is HowItIsRead.SAMPLED_AT_TICK and one.tick < tick
        )
    return {
        "tick": tick,
        "channels": len(modes),
        "how_each_is_read": modes,
        "last_read": readings,
        "sampled_channels_not_read_this_tick": behind,
        "consistent": not behind,
        "what_this_means": (
            "a snapshot whose parts were true at different moments is a "
            "snapshot of no moment"
        ),
    }


def what_does_not_say_how_it_is_read(channels: Any) -> list[str]:
    """Which of these channels never declared a consistency mode."""
    with _LOCK:
        known = set(_HOW)
    return sorted(str(one) for one in (channels or ()) if str(one) not in known)


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    with _LOCK:
        _HOW.clear()
        _LAST.clear()
        _TICK["now"] = 0
