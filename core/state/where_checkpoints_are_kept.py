"""What triggers a checkpoint, and the two places one can be kept.

CrewAI came out above Aura on engineering maturity, and its checkpoint layer
was named: typed trigger events spanning task, crew, agent, flow, LLM, tool
and memory execution; JSON and SQLite providers; and instances restored from
a checkpoint.

Two things are worth taking from that and one is worth doing differently.

The trigger taxonomy is worth taking because "why was this checkpoint made"
is a question a resume has to answer. A checkpoint taken after a tool call and
one taken because a turn ended are different kinds of resume point, and a
timestamp cannot tell them apart.

Two providers are worth taking because one provider is an implementation
pretending to be an interface. Both here pass the same suite.

What is done differently: a checkpoint carries the digest of what it holds, so
a store that hands back something other than what was put in says so. CrewAI's
restore trusts the row.

**Lineage.** A checkpoint names the one it came after and the branch it is on.
Without that a store is a bag of resume points: you can restore any of them
and you cannot ask what happened between two, or which of two divergent
attempts a state belongs to. A branch is what a retry actually is — the same
work from the same point, going somewhere else — and a store that cannot
represent it makes the second attempt overwrite the first.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhereCheckpointsAreKept")

__all__ = [
    "AKeptCheckpoint",
    "the_line_back_from",
    "what_is_on",
    "the_branches",
    "prune",
    "AnAsyncStore",
    "ATrigger",
    "InJson",
    "InSqlite",
    "AsJson",
    "HowAStateIsWritten",
    "THE_DEFAULT_WRITER",
    "TheCheckpointsAreKept",
    "WhatCameBackIsNotWhatWentIn",
    "what_a_checkpoint_store_promises",
]


class ATrigger(StrEnum):
    """Why a checkpoint was taken. A timestamp cannot tell these apart."""

    TURN_ENDED = "turn ended"
    TASK_FINISHED = "task finished"
    AGENT_DECIDED = "agent decided"
    FLOW_BRANCHED = "flow branched"
    MODEL_ANSWERED = "model answered"
    TOOL_RAN = "tool ran"
    MEMORY_WROTE = "memory wrote"
    ASKED_FOR = "asked for"


class WhatCameBackIsNotWhatWentIn(RuntimeError):
    """A store handed back something other than what was put in."""


@runtime_checkable
class HowAStateIsWritten(Protocol):
    """How a checkpoint's state becomes bytes and comes back.

    Declared so a store can be handed a different one. JSON is the default
    because a checkpoint a person cannot read is a resume point nobody can
    check, and that is worth more than the few bytes another format saves.
    """

    def dumps(self, state: dict[str, Any]) -> str:
        ...

    def loads(self, raw: str) -> dict[str, Any]:
        ...


class AsJson:
    """The default. Sorted keys, so one state has one digest."""

    def dumps(self, state: dict[str, Any]) -> str:
        return json.dumps(state, sort_keys=True, default=str, separators=(",", ":"))

    def loads(self, raw: str) -> dict[str, Any]:
        held = json.loads(raw)
        return dict(held) if isinstance(held, dict) else {}


#: What writes a state when nothing else was given.
THE_DEFAULT_WRITER: HowAStateIsWritten = AsJson()


def _digest(state: Any, writer: HowAStateIsWritten | None = None) -> str:
    body = (writer or THE_DEFAULT_WRITER).dumps(state)
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class AKeptCheckpoint:
    """One resume point, with why it was taken and a digest of what it holds."""

    name: str
    trigger: ATrigger
    state: dict[str, Any]
    at: float = field(default_factory=time.time)
    digest: str = ""
    #: The checkpoint this one came after. Empty for the first on a branch.
    after: str = ""
    #: Which line of attempts this belongs to. A retry from a point is a new
    #: branch, not a replacement: the first attempt is still there to compare
    #: against, and "it worked the second time" is a claim about two states.
    branch: str = "main"

    @classmethod
    def of(
        cls,
        name: str,
        state: dict[str, Any],
        *,
        trigger: ATrigger,
        after: str = "",
        branch: str = "main",
    ) -> "AKeptCheckpoint":
        return cls(
            name=str(name),
            trigger=trigger,
            state=dict(state),
            digest=_digest(state),
            after=str(after),
            branch=str(branch or "main"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trigger": str(self.trigger),
            "state": dict(self.state),
            "at": self.at,
            "digest": self.digest,
            "after": self.after,
            "branch": self.branch,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AKeptCheckpoint":
        return cls(
            name=str(row["name"]),
            trigger=ATrigger(row["trigger"]),
            state=dict(row.get("state") or {}),
            at=float(row.get("at", 0.0)),
            digest=str(row.get("digest", "")),
            after=str(row.get("after", "")),
            # A row written before branches existed is on the main line, which
            # is where everything was.
            branch=str(row.get("branch") or "main"),
        )


@runtime_checkable
class TheCheckpointsAreKept(Protocol):
    """What a checkpoint store is, whatever it is written on."""

    def put(self, checkpoint: AKeptCheckpoint) -> None:
        ...

    def get(self, name: str) -> AKeptCheckpoint | None:
        ...

    def names(self) -> list[str]:
        ...

    def forget(self, name: str) -> bool:
        ...


class AnAsyncStore:
    """The same four, awaitable, for callers on the event loop.

    Not a different store — the same one with the blocking part moved off the
    loop. Writing a checkpoint fsyncs, and an fsync on the loop froze this
    runtime for twenty minutes once, so an async caller that used the
    synchronous store would be repeating that.

    A mixin rather than a second class: two implementations of one contract
    drift, and the async pair drifting from the sync one is the drift nobody
    notices until a checkpoint is missing.
    """

    async def put_async(self, checkpoint: AKeptCheckpoint) -> None:
        import asyncio

        # The writes this checkpoint describes must be on disk before it is.
        # Otherwise a reader restores a state whose values were never written
        # and every check passes: the digest matches what the checkpoint
        # recorded, and what it recorded was a future that did not happen.
        from core.state.nothing_lands_before_its_writes import (
            wait_for_the_writes_async,
        )

        await wait_for_the_writes_async(checkpoint.name)
        await asyncio.to_thread(self.put, checkpoint)

    async def get_async(self, name: str) -> AKeptCheckpoint | None:
        import asyncio

        return await asyncio.to_thread(self.get, name)

    async def names_async(self) -> list[str]:
        import asyncio

        return await asyncio.to_thread(self.names)

    async def forget_async(self, name: str) -> bool:
        import asyncio

        return await asyncio.to_thread(self.forget, name)


class InJson(AnAsyncStore):
    """One file holding every checkpoint. Readable by a person."""

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        self._lock = checked_lock(f"checkpoints_in_json:{self._path.name}")
        #: Which version of the document is on disk. The write happens outside
        #: the lock — lockdep refuses an fsync held under one, and it is right:
        #: a blocking call under a lock stalls everyone waiting on it. The
        #: document is written whole, so the newest write contains every
        #: earlier change and a stale one may simply be skipped.
        self._generation = 0
        self._written = 0
        #: The document, once loaded. The file is read once and the memory is
        #: the truth after that: re-reading the file under the lock while
        #: writes happen outside it means a change whose generation was
        #: already taken may not be on disk yet, and the next reader builds
        #: from a document that is missing it.
        self._rows: dict[str, Any] | None = None

    def _read(self) -> dict[str, Any]:
        """The document. Caller holds the lock."""
        if self._rows is not None:
            return self._rows
        held: Any = {}
        if self._path.exists():
            try:
                held = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("checkpoint file unreadable: %s", exc)
                held = {}
        self._rows = held if isinstance(held, dict) else {}
        return self._rows

    def _write(self, rows: dict[str, Any]) -> None:
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        gateway.ensure_directory(self._path.parent, source="checkpoints")
        gateway.write_text(
            self._path,
            json.dumps(rows, indent=2, sort_keys=True, default=str),
            source="checkpoints",
        )

    def put(self, checkpoint: AKeptCheckpoint) -> None:
        rows, generation = self._change(
            lambda held: held.__setitem__(checkpoint.name, checkpoint.to_dict())
        )
        self._write_if_newest(rows, generation)

    def _change(self, do: Any) -> tuple[dict[str, Any], int]:
        """Apply a change to the document and take a generation number.

        The copy handed back is what gets written; the live document keeps
        taking changes while the write happens outside the lock.
        """
        with self._lock:
            rows = self._read()
            do(rows)
            self._generation += 1
            return dict(rows), self._generation

    def _write_if_newest(self, rows: dict[str, Any], generation: int) -> None:
        """Write outside the lock, skipping a write a newer one has overtaken."""
        with self._lock:
            if generation <= self._written:
                return
            self._written = generation
        self._write(rows)

    def get(self, name: str) -> AKeptCheckpoint | None:
        with self._lock:
            row = self._read().get(str(name))
        return _checked(row)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._read())

    def forget(self, name: str) -> bool:
        gone = []

        def drop(held: dict[str, Any]) -> None:
            gone.append(held.pop(str(name), None) is not None)

        rows, generation = self._change(drop)
        if not gone or not gone[0]:
            return False
        self._write_if_newest(rows, generation)
        return True


class InSqlite(AnAsyncStore):
    """A table of checkpoints. For when there are more than a person reads."""

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        self._lock = checked_lock(f"checkpoints_in_sqlite:{self._path.name}")
        # Through the gateway like every other consequential write, and outside
        # the lock: the gateway fsyncs, and an fsync under a lock is what
        # lockdep refuses and what froze this runtime once.
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().ensure_directory(
            self._path.parent, source="checkpoints"
        )
        with self._lock:
            with self._open() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoints ("
                    "name TEXT PRIMARY KEY, trigger TEXT NOT NULL, "
                    "state TEXT NOT NULL, at REAL NOT NULL, digest TEXT NOT NULL)"
                )
                # Added after the table shipped, so a database written by an
                # earlier build opens and reads rather than refusing. A row
                # from before branches existed is on the main line.
                for column, kind, default in (
                    ("after", "TEXT", "''"),
                    ("branch", "TEXT", "'main'"),
                ):
                    try:
                        db.execute(
                            f"ALTER TABLE checkpoints ADD COLUMN {column} {kind} "
                            f"NOT NULL DEFAULT {default}"
                        )
                    except sqlite3.OperationalError:
                        pass  # already there, which is the usual case

    def _open(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=5.0)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def put(self, checkpoint: AKeptCheckpoint) -> None:
        with self._lock, self._open() as db:
            db.execute(
                "INSERT INTO checkpoints "
                "(name, trigger, state, at, digest, after, branch) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                "trigger=excluded.trigger, state=excluded.state, "
                "at=excluded.at, digest=excluded.digest, "
                "after=excluded.after, branch=excluded.branch",
                (
                    checkpoint.name,
                    str(checkpoint.trigger),
                    json.dumps(checkpoint.state, sort_keys=True, default=str),
                    checkpoint.at,
                    checkpoint.digest,
                    checkpoint.after,
                    checkpoint.branch,
                ),
            )

    def get(self, name: str) -> AKeptCheckpoint | None:
        with self._lock, self._open() as db:
            found = db.execute(
                "SELECT name, trigger, state, at, digest, after, branch "
                "FROM checkpoints WHERE name = ?",
                (str(name),),
            ).fetchone()
        if found is None:
            return None
        return _checked(
            {
                "name": found[0],
                "trigger": found[1],
                "state": json.loads(found[2]),
                "at": found[3],
                "digest": found[4],
                "after": found[5],
                "branch": found[6],
            }
        )

    def names(self) -> list[str]:
        with self._lock, self._open() as db:
            return sorted(
                row[0] for row in db.execute("SELECT name FROM checkpoints")
            )

    def forget(self, name: str) -> bool:
        with self._lock, self._open() as db:
            return db.execute(
                "DELETE FROM checkpoints WHERE name = ?", (str(name),)
            ).rowcount > 0


def _checked(row: Any) -> AKeptCheckpoint | None:
    """Rebuild a checkpoint, refusing one whose digest does not match.

    CrewAI's restore trusts the row. A resume point that silently differs from
    what was saved is worse than no resume point, because the run continues.
    """
    if not isinstance(row, dict):
        return None
    checkpoint = AKeptCheckpoint.from_dict(row)
    if checkpoint.digest and checkpoint.digest != _digest(checkpoint.state):
        raise WhatCameBackIsNotWhatWentIn(
            f"{checkpoint.name} was saved as {checkpoint.digest} and came back "
            f"as {_digest(checkpoint.state)}"
        )
    return checkpoint


#: What every provider must do, whatever it is written on.
THE_PROMISES: tuple[str, ...] = (
    "what was put in comes back out",
    "an unknown name is nothing rather than a raise",
    "a name put twice keeps the second",
    "the names are what was put",
    "forgetting removes it and says whether there was anything to forget",
    "the trigger comes back as the trigger",
    "a state that was changed underneath is refused",
    "a parent and a branch survive the round trip",
)


def the_line_back_from(store: Any, name: str, *, most: int = 200) -> list[AKeptCheckpoint]:
    """This checkpoint and every one it came after, newest first.

    Bounded and cycle-safe. A store is a file other processes write, so a row
    naming itself as its own parent is a thing that can arrive, and a walk
    that trusted the chain would spin on it.
    """
    walked: list[AKeptCheckpoint] = []
    seen: set[str] = set()
    at = str(name)
    while at and at not in seen and len(walked) < most:
        seen.add(at)
        one = store.get(at)
        if one is None:
            break
        walked.append(one)
        at = one.after
    return walked


def what_is_on(store: Any, branch: str) -> list[AKeptCheckpoint]:
    """Every checkpoint on one branch, oldest first."""
    out = [
        one
        for one in (store.get(name) for name in store.names())
        if one is not None and one.branch == str(branch)
    ]
    return sorted(out, key=lambda one: one.at)


def the_branches(store: Any) -> dict[str, int]:
    """Every branch the store holds, and how many checkpoints are on it."""
    counted: dict[str, int] = {}
    for name in store.names():
        one = store.get(name)
        if one is not None:
            counted[one.branch] = counted.get(one.branch, 0) + 1
    return dict(sorted(counted.items()))


def prune(store: Any, *, keep: int = 50, branch: str = "") -> list[str]:
    """Forget the oldest, keeping the newest ``keep`` on each branch.

    Per branch rather than overall: pruning globally deletes a whole short
    branch to make room on a long one, which is exactly the branch somebody
    kept to compare against.

    Never prunes a checkpoint another one still names as its parent — a chain
    with a hole in it restores to a state whose history stops mid-sentence.
    """
    wanted = [branch] if branch else list(the_branches(store))
    forgotten: list[str] = []
    for one_branch in wanted:
        on_it = what_is_on(store, one_branch)
        if len(on_it) <= keep:
            continue
        still_needed = {
            found.after
            for name in store.names()
            for found in (store.get(name),)
            if found is not None and found.after
        }
        for old in on_it[: len(on_it) - keep]:
            if old.name in still_needed:
                continue
            if store.forget(old.name):
                forgotten.append(old.name)
    return forgotten


def what_a_checkpoint_store_promises(
    make: Any, *, called: str = ""
) -> dict[str, str]:
    """Run every promise against a provider, with a fresh store each time."""
    name = called or getattr(make, "__name__", "a store")
    kept: dict[str, str] = {}

    def _try(promise: str, check: Any) -> None:
        try:
            check(make())
            kept[promise] = "kept"
        except AssertionError as exc:
            kept[promise] = f"broken: {exc}"
        except Exception as exc:  # noqa: BLE001 — a raise is a broken promise
            kept[promise] = f"broken: {exc!r}"

    one = AKeptCheckpoint.of("one", {"a": 1}, trigger=ATrigger.TURN_ENDED)

    def _lineage_survives(store: Any) -> None:
        first = AKeptCheckpoint.of("first", {"n": 1}, trigger=ATrigger.TURN_ENDED)
        second = AKeptCheckpoint.of(
            "second", {"n": 2}, trigger=ATrigger.TURN_ENDED, after="first"
        )
        other = AKeptCheckpoint.of(
            "other", {"n": 3}, trigger=ATrigger.TURN_ENDED,
            after="first", branch="a retry",
        )
        for each in (first, second, other):
            store.put(each)
        back = store.get("second")
        assert back is not None and back.after == "first", "the parent was lost"
        assert store.get("other").branch == "a retry", "the branch was lost"
        walked = [step.name for step in the_line_back_from(store, "second")]
        assert walked == ["second", "first"], walked
        # What this put, not what the store holds: a provider may be handed a
        # store with rows already in it, and a promise that assumed an empty
        # one would be testing the harness.
        counted = the_branches(store)
        assert counted.get("a retry") == 1, counted
        assert counted.get("main", 0) >= 2, counted
        on_the_retry = [one.name for one in what_is_on(store, "a retry")]
        assert on_the_retry == ["other"], on_the_retry

    def _round_trip(store: Any) -> None:
        store.put(one)
        back = store.get("one")
        assert back is not None, "put a checkpoint, got nothing back"
        assert back.state == {"a": 1}, f"got {back.state}"

    def _unknown(store: Any) -> None:
        assert store.get("never put") is None

    def _second_wins(store: Any) -> None:
        store.put(one)
        store.put(AKeptCheckpoint.of("one", {"a": 2}, trigger=ATrigger.TOOL_RAN))
        assert store.get("one").state == {"a": 2}

    def _names(store: Any) -> None:
        store.put(one)
        store.put(AKeptCheckpoint.of("two", {}, trigger=ATrigger.TOOL_RAN))
        assert store.names() == ["one", "two"]

    def _forget(store: Any) -> None:
        store.put(one)
        assert store.forget("one") is True
        assert store.forget("one") is False
        assert store.get("one") is None

    def _trigger(store: Any) -> None:
        store.put(AKeptCheckpoint.of("t", {}, trigger=ATrigger.FLOW_BRANCHED))
        assert store.get("t").trigger is ATrigger.FLOW_BRANCHED

    def _tamper(store: Any) -> None:
        store.put(one)
        changed = AKeptCheckpoint(
            name="one", trigger=ATrigger.TURN_ENDED, state={"a": 99},
            digest=one.digest,
        )
        store.put(changed)
        try:
            store.get("one")
        except WhatCameBackIsNotWhatWentIn:
            return
        raise AssertionError("a changed state came back without complaint")

    for promise, check in zip(
        THE_PROMISES,
        (_round_trip, _unknown, _second_wins, _names, _forget, _trigger,
         _tamper, _lineage_survives),
        strict=True,
    ):
        _try(promise, check)
    logger.debug(
        "%s kept %d of %d", name,
        sum(1 for one_ in kept.values() if one_ == "kept"), len(THE_PROMISES),
    )
    return kept
