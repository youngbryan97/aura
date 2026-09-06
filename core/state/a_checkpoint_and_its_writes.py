"""A checkpoint never gets ahead of the writes that produced it.

LangGraph beat Aura substantially under the maturity rubric, and the review
was clear that it was not for cleanliness — some of its internals are dense.
It was for semantics: the loop separates pending writes from committed
checkpoints, tracks channel versions and what each node has seen, and
explicitly prevents checkpoint durability from getting ahead of the writes
that produced that checkpoint.

That last property is the one worth copying exactly. A checkpoint recorded
before its writes are durable is a resume point that restores a state which
never existed: the reader comes back believing work was done that the disk
never received.

So there are three things here and they are deliberately separate.

* **Channels** hold values, each with a version that moves on every write.
* **Pending writes** are what a node produced and has not committed. They are
  not in the channel yet, so nothing reads them by accident.
* **A checkpoint** is a point every channel version agrees on, and it cannot
  be taken while writes are still pending.

What each node has seen is kept per node, so "has this node already acted on
this version" is a question with an answer rather than an inference from
timing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.ACheckpointAndItsWrites")

__all__ = [
    "AChannel",
    "ACheckpoint",
    "APendingWrite",
    "TheChannels",
    "WritesStillPending",
]


class WritesStillPending(RuntimeError):
    """A checkpoint was asked for while writes had not been committed.

    Raised rather than taken anyway. A checkpoint whose writes are not in is a
    resume point that restores a state which never existed, and the reader
    comes back believing work happened that the disk never received.
    """


@dataclass
class AChannel:
    """One value, and the version that moves whenever it does."""

    name: str
    value: Any = None
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "version": self.version}


@dataclass(frozen=True)
class APendingWrite:
    """A value a node produced, not yet in its channel."""

    channel: str
    value: Any
    by: str


@dataclass(frozen=True)
class ACheckpoint:
    """A point every channel agreed on, with the versions that made it."""

    name: str
    versions: dict[str, int]
    values: dict[str, Any]
    seen: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "versions": dict(self.versions),
            "values": dict(self.values),
            "seen": {who: dict(what) for who, what in self.seen.items()},
        }


class TheChannels:
    """Channels, their versions, the writes waiting, and what each node saw."""

    def __init__(self) -> None:
        self._lock = checked_lock("core.state.a_checkpoint_and_its_writes")
        self._channels: dict[str, AChannel] = {}
        self._pending: list[APendingWrite] = []
        self._seen: dict[str, dict[str, int]] = {}
        self._checkpoints: dict[str, ACheckpoint] = {}

    # ---------------------------------------------------------- channels

    def value(self, channel: str) -> Any:
        with self._lock:
            found = self._channels.get(channel)
            return found.value if found else None

    def version(self, channel: str) -> int:
        with self._lock:
            found = self._channels.get(channel)
            return found.version if found else 0

    def channels(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: one.to_dict() for name, one in sorted(self._channels.items())}

    # ----------------------------------------------------- pending writes

    def write(self, channel: str, value: Any, *, by: str) -> None:
        """Produce a value. It is not in the channel until it is committed.

        Nothing reads it in the meantime, which is the property that makes a
        half-finished node invisible to the node after it.
        """
        with self._lock:
            self._pending.append(
                APendingWrite(channel=str(channel), value=value, by=str(by))
            )

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"channel": one.channel, "by": one.by} for one in self._pending
            ]

    def commit(self) -> dict[str, int]:
        """Put every pending write into its channel and move the versions.

        Returns the channels that moved and their new versions. Last write to
        a channel wins within one commit — the order inside a commit is the
        order the writes were produced, which is the same rule the compiled
        cognition uses for a field several phases write.
        """
        with self._lock:
            moved: dict[str, int] = {}
            for one in self._pending:
                channel = self._channels.setdefault(
                    one.channel, AChannel(name=one.channel)
                )
                channel.value = one.value
                channel.version += 1
                moved[one.channel] = channel.version
            self._pending.clear()
            return moved

    def discard_pending(self) -> int:
        """Throw away what was produced but not committed. Returns how many."""
        with self._lock:
            how_many = len(self._pending)
            self._pending.clear()
            return how_many

    # ------------------------------------------------------- what was seen

    def mark_seen(self, who: str, channel: str) -> int:
        """Record that this node has acted on the channel's current version."""
        with self._lock:
            version = self._channels.get(channel, AChannel(name=channel)).version
            self._seen.setdefault(str(who), {})[str(channel)] = version
            return version

    def is_new_to(self, who: str, channel: str) -> bool:
        """Whether this node has yet to act on the channel's current version.

        A question with an answer, rather than an inference from timing.
        """
        with self._lock:
            current = self._channels.get(channel, AChannel(name=channel)).version
            return self._seen.get(str(who), {}).get(str(channel), 0) < current

    def seen_by(self, who: str) -> dict[str, int]:
        with self._lock:
            return dict(self._seen.get(str(who), {}))

    # --------------------------------------------------------- checkpoints

    def checkpoint(self, name: str) -> ACheckpoint:
        """Take a checkpoint. Refuses while anything is still pending."""
        with self._lock:
            if self._pending:
                raise WritesStillPending(
                    f"{len(self._pending)} write(s) are not committed: "
                    + ", ".join(sorted({one.channel for one in self._pending}))
                )
            taken = ACheckpoint(
                name=str(name),
                versions={
                    channel: one.version for channel, one in self._channels.items()
                },
                values={
                    channel: one.value for channel, one in self._channels.items()
                },
                seen={who: dict(what) for who, what in self._seen.items()},
            )
            self._checkpoints[taken.name] = taken
            return taken

    def restore(self, name: str) -> ACheckpoint:
        """Go back to a checkpoint. Anything pending is dropped, and said so.

        Dropped rather than kept: a write produced after the point being
        restored belongs to a future that is being abandoned, and carrying it
        across would put a value in a channel whose version does not account
        for it.
        """
        with self._lock:
            taken = self._checkpoints.get(str(name))
            if taken is None:
                raise KeyError(f"no checkpoint called {name!r}")
            if self._pending:
                logger.info(
                    "restoring %s dropped %d write(s) produced after it",
                    name, len(self._pending),
                )
                self._pending.clear()
            self._channels = {
                channel: AChannel(
                    name=channel,
                    value=taken.values.get(channel),
                    version=version,
                )
                for channel, version in taken.versions.items()
            }
            self._seen = {who: dict(what) for who, what in taken.seen.items()}
            return taken

    def checkpoints(self) -> list[str]:
        with self._lock:
            return sorted(self._checkpoints)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "channels": len(self._channels),
                "versions": {
                    name: one.version for name, one in sorted(self._channels.items())
                },
                "pending": len(self._pending),
                "checkpoints": sorted(self._checkpoints),
                "nodes_that_have_seen_something": sorted(self._seen),
            }
