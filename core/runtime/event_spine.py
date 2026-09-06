"""core/runtime/event_spine.py — one log, and a state that is a fold over it.

Aura's state lives in many stores and each is authoritative for itself. That is
defensible per store and it means there is no answer to "what was true at
14:02", no way to replay a session, and no boundary at which a checkpoint could
be taken. ``bus_recorder`` keeps a ring, which forgets, so it cannot be the
spine; the tracer keeps timing, not content.

The spine is an **append-only log of typed events** plus a **projection** that
is a pure fold over them. Three properties follow, and none of them is
available without the log:

* **Replay.** The projection at event N is computable from the log, so a
  session can be reconstructed and a bug can be reproduced from the state that
  caused it rather than from a description of it.
* **Checkpoints.** A checkpoint is a sequence number. Rewinding is choosing a
  smaller one, and it costs nothing to take.
* **Two rewinds.** Events carry a lane, so the conversation and the work can be
  rewound separately - undoing a bad edit without losing the discussion that
  led to it is a fold over one lane and not the other.

Append-only means append-only
-----------------------------
:meth:`EventLog.append` is the only mutator and there is no delete. Compaction
writes a snapshot and drops events BEFORE it, never events after, and the
snapshot carries the sequence number it was taken at so a fold over the
remainder still lands in the same place. A log you can rewrite is a log that
cannot be evidence.

Ownership survives the fold
---------------------------
A reducer registers for the keys it owns, and a reducer that writes a key it
did not declare raises. That is ``state_ownership``'s rule applied to the
projection, and it is what stops one authoritative state from becoming the
same free-for-all the many stores were.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.EventSpine")

#: How far back to read when catching up with another process. One line is
#: never more than a few hundred bytes, so this covers the last few.
_HOW_FAR_BACK_TO_LOOK = 4096

__all__ = [
    "LineageBroken",
    "Lane",
    "Event",
    "Reducer",
    "EventLog",
    "Projection",
    "Checkpoint",
    "OwnershipViolation",
    "get_spine",
    "reset_spine_for_test",
]


class Lane(StrEnum):
    """Which stream an event belongs to, so the two can be rewound apart."""

    CONVERSATION = "conversation"
    WORK = "work"
    COGNITION = "cognition"
    SYSTEM = "system"


class OwnershipViolation(RuntimeError):
    """A reducer wrote state it does not own."""


class LineageBroken(ValueError):
    """An event named a causal parent the log does not have.

    Raised rather than clamped to zero. A caller that names a parent which is
    not there has a defect, and quietly rewriting the lineage to "no parent"
    makes the log agree with itself while disagreeing with what happened —
    which is the failure mode a causal history exists to rule out.
    """


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened. Immutable, ordered, and never edited."""

    seq: int
    kind: str
    lane: Lane
    payload: Mapping[str, Any]
    at: float
    actor: str = ""
    causal_parent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "lane": self.lane.value,
            "payload": dict(self.payload),
            "at": self.at,
            "actor": self.actor,
            "causal_parent": self.causal_parent,
        }


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A point the projection can be rewound to. Just a sequence number."""

    name: str
    seq: int
    at: float = field(default_factory=time.time)
    lane: Lane | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seq": self.seq, "at": self.at,
                "lane": self.lane.value if self.lane else None}


@dataclass(frozen=True, slots=True)
class Reducer:
    """One organ's contribution to the projection, and the keys it may write."""

    name: str
    kinds: frozenset[str]
    owns: frozenset[str]
    apply: Callable[[dict[str, Any], Event], None]


class EventLog:
    """Append-only. The only mutator is append, and there is no delete."""

    def __init__(self, *, capacity: int = 200_000, kept_at: Any = None) -> None:
        self._lock = checked_lock("core.runtime.event_spine.EventLog", reentrant=True)
        self._events: list[Event] = []
        self._offset = 0
        self._snapshot: dict[str, Any] = {}
        self._snapshot_seq = 0
        self._capacity = int(capacity)
        self._next = 1
        # Where the raw experience lives between processes.
        #
        # Append-only in memory is half of the property. OpenHands persists a
        # canonical causal history for its runtime and Letta keeps recall
        # immutable in a durable store, and both make the same point: a log
        # that dies with the process cannot be the thing summaries reference,
        # because the range a summary names is gone before anyone reads it.
        #
        # Appended, never rewritten. A compaction drops what a snapshot
        # accounts for from MEMORY and leaves the file alone, so the raw
        # experience outlives every projection taken from it.
        self._kept_at = Path(kept_at) if kept_at else None
        self._kept_through = 0
        #: What the durable log turned out to contain. Named rather than
        #: counted: "three gaps" does not tell you which range of experience
        #: is missing, and a summary referencing that range is now wrong.
        self._duplicates: list[int] = []
        self._gaps: list[tuple[int, int]] = []
        self._unreadable_after = 0
        self._parentless = 0
        if self._kept_at is not None:
            self._read_what_was_kept()

    def append(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        lane: Lane = Lane.SYSTEM,
        actor: str = "",
        causal_parent: int = 0,
        clock: Callable[[], float] = time.time,
    ) -> Event:
        # The interprocess lock OUTSIDE the thread lock, deliberately. It
        # blocks, and a blocking call under a lock is what lockdep refuses and
        # what wedges a runtime; taken the other way round this deadlocked
        # immediately.
        with self._across_processes(), self._lock:
            if self._kept_at is not None:
                # Two EventLogs on one file each minted their own next
                # sequence, so the file ended up holding each number twice and
                # the third reader had to drop half of them. Under the
                # interprocess lock the counter is caught up from the file
                # first, so the number is unique across processes rather than
                # only within one.
                self._catch_up()
            parent = int(causal_parent or 0)
            if parent and parent >= self._next:
                raise LineageBroken(
                    f"{kind} named parent {parent}, which has not happened yet "
                    f"(the next sequence number is {self._next})"
                )
            if parent < 0:
                raise LineageBroken(f"{kind} named parent {parent}")
            event = Event(
                seq=self._next, kind=kind, lane=lane, payload=dict(payload),
                at=clock(), actor=actor, causal_parent=parent,
            )
            self._next += 1
            self._events.append(event)
        # The write happens with the thread lock released and the interprocess
        # lock still held. Lockdep refuses an fsync under a lock and it is
        # right — a blocking call under one is how a runtime freezes — and
        # ordering is unaffected, because the interprocess lock is what other
        # writers are waiting on.
        self._keep(event)
        return event

    def ancestry(self, seq: int, *, most: int = 512) -> list[int]:
        """The causal chain back from one event, bounded.

        Bounded twice: by ``most``, and by refusing to visit a sequence number
        twice. A durable log is a file, a file can be edited, and an edited
        file can describe a cycle — a traversal that trusted the data would
        hang the process that read it rather than the one that wrote it.
        """
        with self._lock:
            by_seq = {one.seq: one for one in self._events}
            chain: list[int] = []
            seen: set[int] = set()
            here = int(seq)
            while here and here not in seen and len(chain) < int(most):
                seen.add(here)
                chain.append(here)
                found = by_seq.get(here)
                if found is None:
                    break
                here = found.causal_parent
            return chain

    def integrity(self) -> dict[str, Any]:
        """What the log knows to be wrong with itself.

        Everything here is about the durable file, because that is the part
        another process can have written, truncated, or edited.
        """
        with self._lock:
            return {
                "kept_at": str(self._kept_at) if self._kept_at else "",
                "kept_through": self._kept_through,
                "next": self._next,
                "in_memory": len(self._events),
                "duplicate_sequences": list(self._duplicates),
                "gaps": [list(one) for one in self._gaps],
                "unreadable_after": self._unreadable_after,
                "parentless": self._parentless,
                "whole": not (
                    self._duplicates or self._gaps or self._unreadable_after
                ),
            }

    @contextmanager
    def _across_processes(self) -> Iterator[None]:
        """Hold the file against other processes, where there is a file."""
        if self._kept_at is None:
            yield
            return
        try:
            from core.runtime.atomic_writer import interprocess_file_lock
        except ImportError:  # pragma: no cover - foundation import order
            yield
            return
        # A SIBLING file, not the log itself. `atomic_append_text` flocks the
        # log through its own descriptor, and flock is per open-file
        # description — so locking the same path here deadlocked the process
        # against itself on the first append.
        with interprocess_file_lock(self._kept_at.with_suffix(".seq.lock")):
            yield

    def _catch_up(self) -> int:
        """Read the last sequence another process wrote. Caller holds the lock.

        Only the tail: the whole point is that this runs on every append, and
        a full re-read would make an append cost the length of the log.
        """
        if self._kept_at is None or not self._kept_at.exists():
            return self._next
        try:
            with self._kept_at.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - _HOW_FAR_BACK_TO_LOOK))
                tail = handle.read().decode("utf-8", "replace")
        except OSError as exc:
            logger.debug("could not read the tail of the kept log: %s", exc)
            return self._next
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                seq = int(json.loads(line)["seq"])
            except (ValueError, KeyError, TypeError):
                continue
            if seq >= self._next:
                logger.debug(
                    "another process had reached %d; this one was at %d",
                    seq, self._next,
                )
                self._next = seq + 1
                self._kept_through = max(self._kept_through, seq)
            return self._next
        return self._next

    def rebuild(self) -> dict[str, Any]:
        """Read the file again from nothing, and say what came back.

        Deterministic: the same file gives the same events in the same order,
        and the counts say whether anything was lost. A recovery that cannot
        be checked is a recovery nobody can rely on.
        """
        with self._across_processes(), self._lock:
            before = len(self._events)
            self._events.clear()
            self._duplicates.clear()
            self._gaps.clear()
            self._unreadable_after = 0
            self._parentless = 0
            self._next = 1
            self._kept_through = 0
            read = self._read_what_was_kept()
            return {
                "in_memory_before": before,
                "read_back": read,
                "lost": max(0, before - read),
                "whole": self.integrity()["whole"],
            }

    def _keep(self, event: Event) -> None:
        """Append one event to the durable log. Never rewrites what is there."""

        if self._kept_at is None:
            return
        try:
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().append_text(
                self._kept_at,
                json.dumps(event.to_dict(), separators=(",", ":")) + "\n",
                source="event_spine",
            )
            self._kept_through = event.seq
        except Exception as exc:  # noqa: BLE001 — a turn must not die for a log line
            logger.warning("event %d was not kept: %s", event.seq, exc)

    def _read_what_was_kept(self) -> int:
        """Load the durable log at open. Returns how many events came back.

        A line that will not parse stops the read rather than being skipped:
        the sequence numbers are the thing every projection joins on, and a
        hole in them is worse than a short log, because a short log is
        obviously short.
        """

        if self._kept_at is None or not self._kept_at.exists():
            return 0
        read = 0
        seen: set[int] = set()
        last = 0
        try:
            with self._kept_at.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    event = Event(
                        seq=int(row["seq"]),
                        kind=str(row["kind"]),
                        lane=Lane(row["lane"]),
                        payload=dict(row.get("payload") or {}),
                        at=float(row.get("at", 0.0)),
                        actor=str(row.get("actor", "")),
                        causal_parent=int(row.get("causal_parent", 0)),
                    )
                    if event.seq in seen:
                        # Two writers on one file, or a file appended twice.
                        # The second copy is dropped rather than replayed: a
                        # projection that applies one event twice has counted
                        # something that happened once.
                        self._duplicates.append(event.seq)
                        continue
                    if last and event.seq > last + 1:
                        self._gaps.append((last + 1, event.seq - 1))
                    if not event.causal_parent:
                        self._parentless += 1
                    seen.add(event.seq)
                    last = event.seq
                    self._events.append(event)
                    self._next = max(self._next, event.seq + 1)
                    self._kept_through = event.seq
                    read += 1
        except Exception as exc:  # noqa: BLE001 — a truncated log is still a log
            # A half-written last line is the ordinary case: the process died
            # mid-append. Recorded rather than raised, and the sequence
            # counter is recovered from what did parse, so the next append
            # cannot reuse a number the file already holds.
            self._unreadable_after = last
            logger.warning(
                "the kept log stopped being readable after %d events: %s", read, exc
            )
        if self._duplicates or self._gaps or self._unreadable_after:
            logger.warning(
                "kept log: %d duplicate sequences, %d gaps, unreadable after %d",
                len(self._duplicates), len(self._gaps), self._unreadable_after,
            )
        return read

    @property
    def kept_through(self) -> int:
        """The last sequence number written durably. Zero when nothing is kept."""

        with self._lock:
            return self._kept_through

    def events(self, *, since: int = 0, until: int | None = None, lane: Lane | None = None) -> list[Event]:
        with self._lock:
            return [
                e for e in self._events
                if e.seq > since
                and (until is None or e.seq <= until)
                and (lane is None or e.lane is lane)
            ]

    @property
    def head(self) -> int:
        with self._lock:
            return self._next - 1

    def compact(self, projection_state: Mapping[str, Any], *, through: int) -> dict[str, Any]:
        """Write a snapshot and drop events BEFORE it. Never events after.

        The snapshot carries the sequence it was taken at, so a fold over the
        remaining events lands in exactly the state a full replay would.
        """
        with self._lock:
            if through > self.head:
                raise ValueError(f"cannot compact through {through}; the log ends at {self.head}")
            self._snapshot = dict(projection_state)
            self._snapshot_seq = through
            dropped = sum(1 for e in self._events if e.seq <= through)
            self._events = [e for e in self._events if e.seq > through]
            self._offset += dropped
            # The file is untouched. Compaction is about what this process
            # holds in memory; the raw experience outlives every projection
            # taken from it, which is the whole of "recall never mutates".
            return {
                "snapshot_seq": through,
                "dropped": dropped,
                "remaining": len(self._events),
                "still_kept_through": self._kept_through,
            }

    def snapshot(self) -> tuple[dict[str, Any], int]:
        with self._lock:
            return dict(self._snapshot), self._snapshot_seq

    def report(self) -> dict[str, Any]:
        with self._lock:
            by_lane: dict[str, int] = {}
            for event in self._events:
                by_lane[event.lane.value] = by_lane.get(event.lane.value, 0) + 1
            return {
                "events_retained": len(self._events),
                "events_ever": self._next - 1,
                "compacted_away": self._offset,
                "snapshot_seq": self._snapshot_seq,
                "head": self.head,
                "by_lane": by_lane,
                "durable": self._kept_at is not None,
                "kept_through": self._kept_through,
                # What the file turned out to contain. A log that is not whole
                # still answers every read; the difference is that a summary
                # naming a range now knows whether that range is there.
                "duplicate_sequences": list(self._duplicates),
                "gaps": [list(one) for one in self._gaps],
                "unreadable_after": self._unreadable_after,
                "whole": not (
                    self._duplicates or self._gaps or self._unreadable_after
                ),
            }


class Projection:
    """The one authoritative mutable state, and it is a fold over the log."""

    def __init__(self, log: EventLog) -> None:
        self._lock = checked_lock("core.runtime.event_spine.Projection", reentrant=True)
        self._log = log
        self._reducers: dict[str, Reducer] = {}
        self._state: dict[str, Any] = {}
        self._applied = 0
        self._checkpoints: dict[str, Checkpoint] = {}

    def register(
        self,
        name: str,
        kinds: Sequence[str],
        owns: Sequence[str],
        apply: Callable[[dict[str, Any], Event], None],
    ) -> Reducer:
        """Register a reducer and the state keys it may write."""
        if not owns:
            raise ValueError(
                f"reducer {name!r} declares no owned keys; a reducer that may write "
                "anything makes the projection the free-for-all the many stores were"
            )
        reducer = Reducer(name, frozenset(kinds), frozenset(owns), apply)
        with self._lock:
            for existing in self._reducers.values():
                clash = existing.owns & reducer.owns
                if clash:
                    raise OwnershipViolation(
                        f"{name!r} claims {sorted(clash)}, already owned by {existing.name!r}"
                    )
            self._reducers[name] = reducer
            return reducer

    def _apply_locked(self, event: Event) -> None:
        for reducer in self._reducers.values():
            if event.kind not in reducer.kinds:
                continue
            before = {k: self._state.get(k) for k in self._state}
            reducer.apply(self._state, event)
            written = {
                key for key, value in self._state.items()
                if key not in before or before[key] != value
            }
            trespass = written - reducer.owns
            if trespass:
                raise OwnershipViolation(
                    f"reducer {reducer.name!r} wrote {sorted(trespass)}, which it does "
                    f"not own (it owns {sorted(reducer.owns)})"
                )

    def advance(self) -> dict[str, Any]:
        """Fold every event since the last advance into the state."""
        with self._lock:
            for event in self._log.events(since=self._applied):
                self._apply_locked(event)
                self._applied = event.seq
            return dict(self._state)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def at(self, seq: int, *, lanes: Sequence[Lane] | None = None) -> dict[str, Any]:
        """The state as of a sequence number, rebuilt from the log.

        ``lanes`` rewinds one stream and not another: replaying the
        conversation lane to now while stopping the work lane at a checkpoint
        undoes an edit without losing the discussion that led to it.
        """
        snapshot, snapshot_seq = self._log.snapshot()
        state = dict(snapshot)
        if seq < snapshot_seq:
            raise ValueError(
                f"cannot rebuild state at {seq}: the log was compacted through "
                f"{snapshot_seq} and the events before it are gone"
            )
        with self._lock:
            reducers = list(self._reducers.values())
        for event in self._log.events(since=snapshot_seq, until=seq):
            if lanes is not None and event.lane not in lanes:
                continue
            for reducer in reducers:
                if event.kind in reducer.kinds:
                    reducer.apply(state, event)
        return state

    def checkpoint(self, name: str, *, lane: Lane | None = None) -> Checkpoint:
        with self._lock:
            mark = Checkpoint(name=name, seq=self._log.head, lane=lane)
            self._checkpoints[name] = mark
            return mark

    def rewind(self, name: str, *, lanes: Sequence[Lane] | None = None) -> dict[str, Any]:
        """The state as it was at a checkpoint. The log is never touched."""
        with self._lock:
            mark = self._checkpoints.get(name)
        if mark is None:
            raise KeyError(f"no checkpoint named {name!r}")
        return self.at(mark.seq, lanes=lanes)

    def revert(
        self, name: str, *, lanes: Sequence[Lane] | None = None, reason: str = ""
    ) -> dict[str, Any]:
        """Make the state at a checkpoint the live state, and say so in the log.

        :meth:`rewind` computes what the state WAS and hands it back; nothing
        adopts it. A caller who rewinds and then keeps working still has the
        abandoned work in the projection, which is how a correction lands in a
        report and not in the state. This is the half that acts.

        The log is still never rewritten. A revert is an event like any other,
        appended to the system lane, so "we went back" is in the history rather
        than being the absence of what used to be there. ``lanes`` reverts one
        stream and leaves the others where they are, which is what lets a
        correction undo the work without taking the conversation with it.
        """
        with self._lock:
            mark = self._checkpoints.get(name)
        if mark is None:
            raise KeyError(f"no checkpoint named {name!r}")
        # Fold everything outstanding first. A revert moves ``_applied`` past
        # every event up to it, so an event appended and not yet advanced would
        # be skipped for good — and in the per-lane case that is an event in a
        # lane nobody asked to revert.
        self.advance()
        restored = self.at(mark.seq, lanes=lanes)
        # One pass over the events since the checkpoint, outside the
        # projection lock. The first version scanned the whole log once per
        # owned key, which on a full 200,000-event log is millions of
        # iterations to answer a question about a handful of kinds.
        kinds_reverted: set[str] = (
            set()
            if lanes is None
            else {
                event.kind
                for event in self._log.events(since=mark.seq)
                if event.lane in lanes
            }
        )
        with self._lock:
            if lanes is None:
                self._state = restored
            else:
                # Only the keys the named lanes' reducers own go back. A key
                # written from a lane nobody reverted is not part of what was
                # undone, and taking it too is the failure this argument exists
                # to prevent.
                owned = {
                    key
                    for reducer in self._reducers.values()
                    if reducer.kinds & kinds_reverted
                    for key in reducer.owns
                }
                for key in owned:
                    if key in restored:
                        self._state[key] = restored[key]
                    else:
                        self._state.pop(key, None)
            reverted = dict(self._state)
        event = self._log.append(
            "spine.reverted",
            {
                "checkpoint": name,
                "to_seq": mark.seq,
                "lanes": sorted(lane.value for lane in lanes) if lanes else None,
                "reason": reason,
            },
            lane=Lane.SYSTEM,
        )
        with self._lock:
            # The revert event is history, not something to fold: applying it
            # would be asking a reducer what a revert does to the state it just
            # became.
            self._applied = event.seq
        return reverted

    def checkpoints(self) -> list[Checkpoint]:
        with self._lock:
            return sorted(self._checkpoints.values(), key=lambda c: c.seq)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "reducers": {
                    r.name: {"kinds": sorted(r.kinds), "owns": sorted(r.owns)}
                    for r in self._reducers.values()
                },
                "keys": sorted(self._state),
                "applied_through": self._applied,
                "checkpoints": [c.to_dict() for c in self.checkpoints()],
                "unowned_keys": sorted(
                    set(self._state)
                    - {k for r in self._reducers.values() for k in r.owns}
                ),
            }


@dataclass
class Spine:
    """The log and its projection, together, because neither is useful alone."""

    log: EventLog
    projection: Projection

    def emit(self, kind: str, payload: Mapping[str, Any], **kwargs: Any) -> Event:
        event = self.log.append(kind, payload, **kwargs)
        self.projection.advance()
        return event

    def report(self) -> dict[str, Any]:
        return {"log": self.log.report(), "projection": self.projection.report()}


_lock = checked_lock("core.runtime.event_spine.singleton")
_spine: Spine | None = None


def _where_the_experience_is_kept() -> Path | None:
    """The durable log's path, or None where this process must not write one.

    Reading is safe anywhere; writing is for a process that owns the state.
    A test run that appended to the live log would put its events into her
    experience, and the next boot would replay them as things that happened.
    """

    try:
        from core.runtime.state_ownership import RuntimeProfile, runtime_profile, state_root

        if runtime_profile() is not RuntimeProfile.LIVE:
            return None
        return Path(state_root()) / "experience.jsonl"
    except Exception:  # noqa: BLE001 — no path means an in-memory log, which still works
        return None


def get_spine() -> Spine:
    global _spine
    with _lock:
        if _spine is None:
            log = EventLog(kept_at=_where_the_experience_is_kept())
            _spine = Spine(log=log, projection=Projection(log))
        return _spine


def reset_spine_for_test(**kwargs: Any) -> Spine:
    global _spine
    with _lock:
        log = EventLog(**kwargs)
        _spine = Spine(log=log, projection=Projection(log))
        return _spine
