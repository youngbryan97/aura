"""`with sqlite3.connect(...)` does not close the connection.

It reads like resource management and it is transaction management. From the
stdlib docs: the connection context manager commits on success and rolls back
on an exception — and "does not implicitly close the connection". Measured:

    for i in range(5):
        with sqlite3.connect(path) as conn:
            conn.execute(...)
    # -> 5 open file descriptors on that database

They come back when the connection object is finally collected, which is at
best the end of the enclosing function and at worst whenever the cyclic
collector runs. WAL adds two more handles per connection (`-wal`, `-shm`), so
the real cost is threefold. The suite's hermetic-resource guard finds these as
"open_files" leaks at test teardown; a long-running desktop runtime finds them
as a slow climb nobody attributes to a mood update.

This module is the one-line fix, and it PRESERVES the transaction semantics
rather than replacing them — `contextlib.closing` alone would close the handle
and silently drop the commit-on-success behaviour every call site depends on.

Usage::

    from core.runtime.sqlite_support import connecting

    with connecting(sqlite3.connect(path)) as conn:
        conn.execute(...)          # commits on success, rolls back on error,
                                   # and closes either way

It takes an already-constructed connection so the call site keeps full control
of its own pragmas, timeouts, URI flags and row factories — the alternative
(wrapping `connect` itself) would have to re-expose every one of those.
"""

from __future__ import annotations

import sqlite3
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "TrackedConnection",
    "close_all_tracked",
    "connecting",
    "connection_is_open",
    "open_tracked",
    "track",
]


@contextmanager
def connecting(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit-or-rollback like ``with conn``, then always close.

    Exactly equivalent to::

        try:
            with connection:
                ...
        finally:
            connection.close()

    so converting a call site changes only whether the descriptor is returned,
    never when the transaction ends.
    """
    try:
        with connection:
            yield connection
    finally:
        try:
            connection.close()
        # not a failure: a connection that cannot close is already unusable,
        # and raising here would replace the caller's real exception with a
        # cleanup one.
        except sqlite3.Error:
            pass


# ── Long-lived connections ────────────────────────────────────────────────
#
# `connecting()` above solves the per-call case: open, use, close. It cannot
# help the other shape, which is a durable store that opens ONE connection in
# its constructor and keeps it — the audit log, the receipts ledger, the goal
# lifecycle store, the cognitive ledger. Those are correct designs; a store
# that reopened sqlite on every write would be worse. The problem is only that
# nothing can find them again.
#
# Measured 2026-08-06: four test files failing the hermetic resource guard on
# four databases, each a process-global store holding its handle past the test
# that made it, each blaming whichever test ran when the collector noticed.
#
# A registry that keeps nothing alive. `sqlite3.Connection` does not support
# weak references; a subclass does, which is the whole trick.
_TRACKED_LOCK = checked_lock("sqlite_support.tracked_connections")
_TRACKED: weakref.WeakSet[sqlite3.Connection] = weakref.WeakSet()


class TrackedConnection(sqlite3.Connection):
    """A connection a registry can hold weakly.

    Exists only so `weakref` works. A registry holding strong references would
    be a leak of its own rather than a fix for one.
    """


def open_tracked(database: Any, **kwargs: Any) -> sqlite3.Connection:
    """`sqlite3.connect` that a later `close_all_tracked()` can find.

    A drop-in for the durable-store case: same arguments, same semantics, and
    the connection joins a registry that does not extend its life.
    """
    kwargs.setdefault("factory", TrackedConnection)
    connection = sqlite3.connect(database, **kwargs)
    with _TRACKED_LOCK:
        _TRACKED.add(connection)
    return connection


def track(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Register an already-open connection; returns it so it can be chained.

    Does nothing for a plain `sqlite3.Connection`, which cannot be weakly
    referenced — so prefer `open_tracked`, and read a store that cannot be
    tracked as a store that must close itself.
    """
    try:
        with _TRACKED_LOCK:
            _TRACKED.add(connection)
    # not a failure: the docstring above says so — a plain sqlite3.Connection
    # cannot be weakly referenced, and such a store closes itself.
    except TypeError:
        pass
    return connection


def connection_is_open(connection: sqlite3.Connection) -> bool:
    """Can this connection still be used?

    A cached handle can be closed underneath its owner — by corruption
    recovery, by an explicit shutdown, by a test teardown. Asking is cheaper
    than the alternative, which is every later call raising "Cannot operate on
    a closed database" with no way back.
    """
    try:
        connection.execute("SELECT 1")
        return True
    # not a failure: "Cannot operate on a closed database" is the answer this
    # asks for, which the docstring above describes.
    except sqlite3.ProgrammingError:
        return False
    except sqlite3.Error:
        # Locked or busy is not closed: the handle is alive and someone else
        # is using the file. Reopening here would make contention look like
        # death and double the connections under load.
        return True


def close_all_tracked() -> dict[str, int]:
    """Close every tracked connection still alive. Returns a small report.

    Used between tests so the hermetic guard blames the test that opened a
    handle rather than the one running when it was noticed, and available at
    shutdown for the reason this exists at all: a durable store that is never
    asked to let go never does.

    Never raises. A connection that cannot close is already unusable, and this
    runs in teardown paths where an exception costs more than a handle.
    """
    with _TRACKED_LOCK:
        connections = list(_TRACKED)
        _TRACKED.clear()
    closed = failed = 0
    for connection in connections:
        try:
            connection.close()
            closed += 1
        except sqlite3.Error:
            failed += 1
    return {"closed": closed, "failed": failed}
