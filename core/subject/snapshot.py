"""Taking the organism's state away and putting it back, unchanged.

A fork is the whole method: the same organism runs each arm from the same
instant, so whatever one arm did must not be there when the next one starts.
That means reaching every place state lives — the organs, the services, the
process-wide singletons, the world on disk, the intention rows, the effort
counter, torch's generator — and carrying none of the process furniture, since
a copied lock is at best useless and at worst a deadlock.

It lived beside the driver that uses it until the file was 2,509 lines, which
is past the size at which anyone reads the whole of it. Nothing moved but the
line numbers: the driver imports these names and re-exports the four that
other modules already ask it for.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import enum
import importlib
import inspect
import logging
import os
import sqlite3
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from core.runtime.errors import record_degradation

if TYPE_CHECKING:  # a driver type used in annotations only, and this is imported by it
    from core.subject.driver import SubjectRuntime

logger = logging.getLogger("Aura.Subject.Snapshot")

__all__ = [
    "SUBSTRATE_BODY",
    "Snapshot",
    "reanchor",
    "restore_singletons",
    "singleton_state",
]


#: Attribute names never carried across a fork. Locks, events, tasks and
#: sockets are process furniture; copying one is at best useless and at worst a
#: deadlock, and none of them is state in the sense this battery measures.
#: Locks, events, tasks and sockets are process furniture; copying one is at
#: best useless and at worst a deadlock, and none of them is state in the sense
#: this battery measures.
#:
#: Decided by the type, never by the name. The first version matched substrings
#: of the attribute name, which excluded `_cached_connectivity_norm` because it
#: contains `_conn`, `total_collapse_events` because it contains `_event`, and
#: `_loop_failure_streak` because it contains `_loop` — three pieces of real
#: substrate state, dropped from every fork, and therefore three contributions
#: to a floor that no experiment could get below. A name is not a type.
#: How deep the furniture scan looks inside an object before giving up and
#: treating it as safe. Deep enough to find a lock a service keeps behind one
#: helper, shallow enough that the scan is bounded.
_FURNITURE_DEPTH: int = 3


def _furniture_types() -> tuple[type, ...]:
    import io
    import multiprocessing
    import socket as _socket_module
    import sqlite3
    import threading as _threading
    from concurrent.futures import Executor, Future

    return (
        _threading.Event,
        _threading.Barrier,
        _threading.Semaphore,
        _threading.Thread,
        type(_threading.Lock()),
        type(_threading.RLock()),
        _socket_module.socket,
        sqlite3.Connection,
        sqlite3.Cursor,
        io.IOBase,
        multiprocessing.process.BaseProcess,
        asyncio.Task,
        asyncio.Future,
        asyncio.Event,
        asyncio.Lock,
        asyncio.Condition,
        asyncio.Queue,
        asyncio.AbstractEventLoop,
        Executor,
        Future,
    )


_FURNITURE: tuple[type, ...] = ()


def _is_process_furniture(value: Any) -> bool:
    """Whether this value *is* a handle. Not whether it contains one."""
    global _FURNITURE
    if not _FURNITURE:
        _FURNITURE = _furniture_types()
    if isinstance(value, _FURNITURE):
        return True
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) <= 64:
        return any(_is_process_furniture(item) for item in value)
    if isinstance(value, dict) and len(value) <= 64:
        return any(_is_process_furniture(item) for item in value.values())
    return False


def _holds_furniture(value: Any, depth: int = 0) -> bool:
    """Whether copying this would duplicate a handle somewhere inside it.

    Different question from the one above, and the two must not be confused. A
    value that *is* a lock has nothing to carry and is skipped. A value that
    *holds* one — the world model's forward network keeps a stop event, the
    self model keeps a lock beside its beliefs — has plenty to carry and must
    be taken apart rather than skipped, which is how the network came to be
    absent from every fork.

    The scan matters because a deep copy of an open file duplicates the
    descriptor, and collecting the copy closes the original underneath the
    process still using it. That is not a floor in a measurement, it is a
    broken runtime.
    """
    if _is_process_furniture(value):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.ndarray)):
        return False
    if depth >= _FURNITURE_DEPTH:
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_furniture(item, depth + 1) for item in list(value)[:64])
    if isinstance(value, dict):
        return any(_holds_furniture(item, depth + 1) for item in list(value.values())[:64])
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return any(_holds_furniture(item, depth + 1) for item in list(fields.values())[:64])
    return False


#: How the capture marks a field it had to take apart rather than copy whole.
_NESTED = "__nested__"

#: How deep the capture goes into an organ's own objects. Two is enough to
#: reach the world model's forward network through its wrapper, and shallow
#: enough that a graph of references cannot turn a snapshot into a traversal.
_ORGAN_DEPTH: int = 2


def _organ_state(organ: Any, depth: int = 0, skip: frozenset[int] = frozenset()) -> dict[str, Any]:
    """A deep copy of the numbers an organ is carrying, and nothing else.

    Copied by value so that restoring one arm cannot hand the next arm a live
    reference it then mutates.

    A field that will not copy is taken apart instead of dropped. The world
    model's wrapper holds the forward network, the network holds a stop event
    and a trainer handle, and a lock cannot be deep-copied — so the whole
    network was silently skipped and its weights and hidden state drifted
    across every fork. Dropping a field that will not copy is how a shared
    variable becomes a floor that no experiment can get below.
    """
    if organ is None or not hasattr(organ, "__dict__"):
        return {}
    out: dict[str, Any] = {}
    for name, value in list(vars(organ).items()):
        if callable(value) or inspect.ismodule(value) or _is_process_furniture(value):
            continue
        # Already carried under its own name. A phase holds references to
        # services, and capturing them twice doubles the cost of every fork
        # and writes the same values through two paths.
        if id(value) in skip:
            continue
        if hasattr(value, "__dict__") and depth < _ORGAN_DEPTH and _holds_furniture(value):
            out[name] = (_NESTED, _organ_state(value, depth + 1, skip))
            continue
        try:
            out[name] = copy.deepcopy(value)
        except Exception:  # noqa: BLE001 - every way a copy fails has one answer
            # Take it apart instead. A multiprocessing queue raises RuntimeError
            # rather than TypeError, an executor raises something else again,
            # and enumerating the ways a copy can fail is how the world model
            # came to be dropped from every fork.
            if hasattr(value, "__dict__") and depth < _ORGAN_DEPTH:
                out[name] = (_NESTED, _organ_state(value, depth + 1))
    return out


#: Values no arm can mutate, so a restore can hand the same object to all three
#: without copying it. Most of what an organ carries is one of these, and
#: copying them was most of what a restore cost.
_ATOMIC: tuple[type, ...] = (str, bytes, int, float, bool, type(None))


def _place(value: Any) -> Any:
    """A value safe to hand to an arm that may mutate it.

    A tuple is only as immutable as what is inside it — a tuple of lists is
    shared state wearing an immutable type — so the check goes one level in
    rather than trusting the container.
    """
    if isinstance(value, _ATOMIC):
        return value
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, (tuple, frozenset)) and all(
        isinstance(item, _ATOMIC) for item in value
    ):
        return value
    return copy.deepcopy(value)


def _guards_its_own_writes(organ: Any) -> bool:
    """Whether this object polices what may be written to it.

    A class that defines `__setattr__` has an opinion about its own mutation,
    and a fork rewinding state is not the caller that opinion was written for.
    The mycelial topology refuses the write and logs a critical before it does,
    which is the guard working — so the fork stops asking.
    """
    return type(organ).__setattr__ is not object.__setattr__


#: A field an object does not have, as distinct from one that holds None.
_ABSENT = object()


def _restore_organ(organ: Any, saved: Mapping[str, Any]) -> None:
    if organ is None or _guards_its_own_writes(organ):
        return
    seen: set[int] = set()
    for name, value in saved.items():
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _NESTED:
            _restore_organ(getattr(organ, name, None), value[1])
            continue
        _put_back(organ, name, value, seen)


def _put_back(owner: Any, name: str, saved: Any, seen: set[int]) -> None:
    """Write one saved field back, into the object already there if it can be.

    Writing a copy over the field is right for a number and wrong for an object
    something else also holds. `ConversationalDynamicsPhase` keeps the engine
    that `get_dynamics_engine()` hands every other caller, and the first restore
    gave the phase a copy of its own. From then on the phase rewound every arm
    and the shared engine, which the response phase updates, never did: across
    four arms from one snapshot its message count read 18, 19, 20 and 21. So an
    object of the same type is restored field by field and keeps its identity,
    and so are the dicts, lists, deques and arrays it holds.
    """
    if _restore_into(getattr(owner, name, _ABSENT), saved, seen):
        return
    try:
        setattr(owner, name, _place(saved))
    except (
        ArithmeticError,
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        # A field that will not be written stays as it was. The kinds are
        # named so an interrupt still stops the restore.
        return


#: CPython's flag on a class made by a class statement rather than written in C.
_HEAP_TYPE = 1 << 9


def _state_is_its_dict(value: Any) -> bool:
    """Whether everything this object holds is in its `__dict__`.

    Only then can it be restored by writing its fields back. A tensor has a
    `__dict__` as well, and it is empty: the numbers live in the C base class.
    Restored field by field, a tensor held on a plain object kept the values an
    arm had written into it. So every class above `object` has to be one
    written in Python, and none may declare slots.
    """
    if not hasattr(value, "__dict__") or isinstance(value, type):
        return False
    return all(
        klass.__flags__ & _HEAP_TYPE and "__slots__" not in vars(klass)
        for klass in type(value).__mro__[:-1]
    )


def _restore_into(current: Any, saved: Any, seen: set[int]) -> bool:
    """Make `current` hold what `saved` holds without replacing it.

    False when that cannot be done in place, and the caller writes a copy over
    the field instead. Fields an arm added to an object are left alone: a copy
    made through `__getstate__` can leave out a lock the live object needs, and
    deleting whatever the copy lacks would delete the lock.
    """
    if current is _ABSENT or current is saved or type(current) is not type(saved):
        return False
    if isinstance(saved, _ATOMIC) or isinstance(current, enum.Enum):
        return False
    if isinstance(current, np.ndarray):
        if current.shape != saved.shape or current.dtype != saved.dtype or not current.flags.writeable:
            return False
        np.copyto(current, saved)
        return True
    if isinstance(current, dict):
        # Key by key, so an object filed under a key keeps its identity too.
        rebuilt = {}
        for key, value in saved.items():
            held = current.get(key, _ABSENT)
            rebuilt[key] = held if _restore_into(held, value, seen) else _place(value)
        current.clear()
        current.update(rebuilt)
        return True
    if isinstance(current, list):
        current[:] = _place(saved)
        return True
    if isinstance(current, deque):
        if current.maxlen != saved.maxlen:
            return False
        current.clear()
        current.extend(_place(saved))
        return True
    if isinstance(current, set):
        current.clear()
        current.update(_place(saved))
        return True
    if (
        not _state_is_its_dict(current)
        or inspect.ismodule(current)
        or callable(current)
        or _is_process_furniture(current)
        or _guards_its_own_writes(current)
    ):
        return False
    if id(saved) in seen:
        # A cycle: this object is already being restored further up.
        return True
    seen.add(id(saved))
    for key, value in vars(saved).items():
        _put_back(current, key, value, seen)
    return True


#: The reservoir attributes that make up an ontogenetic state's whole memory.
_RESERVOIR_FIELDS: tuple[str, ...] = (
    "h",
    "steps",
    "era",
    "_centre",
    "_scatter",
    "_centre_n",
    # How much it usually moves, which the relative displacement is read
    # against. A relative reading against a mean the other arm never saw is
    # not a relative reading.
    "_displacement_mean",
    "_displacement_seen",
)


#: Attributes holding a wall-clock instant that a later reader turns into an
#: elapsed time. Restoring a snapshot rewinds these along with everything else,
#: so the arm that runs second sees a longer interval than the arm that ran
#: first — which is the machine's own speed entering the measurement.
_CLOCK_ANCHORS: tuple[str, ...] = (
    "last_tick",
    "last_update",
    "_last_update",
    "last_thought_at",
    "_last_pulse_t",
    "_last_state_mutation_at",
    "start_time",
    "_last_disk_time",
    "_last_thought_time",
    "submitted_at",
)

#: Anything smaller than this is not a wall-clock instant. Epoch seconds passed
#: a billion in 2001; a duration, a count or a rate never reaches it. The
#: ceiling is the year 2100, above which a large number is something else.
_EPOCH_FLOOR: float = 1e9
_EPOCH_CEILING: float = 4.1e9

#: What a field holding an instant is called. The named list above is the set
#: the shallow pass knew about; a fork has to find the ones nobody listed,
#: because the one that mattered most — when a workspace bid was submitted —
#: is two objects deep inside the organ and was named none of them.
_INSTANT_WORDS: tuple[str, ...] = ("time", "_at", "stamp", "clock", "since", "when")

#: How far into an organ the rewind looks. The bid that decides a competition
#: sits at depth two: workspace -> last_winner -> submitted_at.
_ANCHOR_DEPTH: int = 3

#: How many entries of a container the rewind reads. A history buffer can hold
#: thousands and only the recent ones carry an instant anything still reads.
_ANCHOR_FANOUT: int = 64

#: How far into the object graph the search for wall-clock instants goes. The
#: workspace holds its last winner, the winner holds the instant it was
#: submitted, and the priority every consumer reads is that instant against the
#: clock — three hops from the organ.
_ANCHOR_DEPTH: int = 4

def _looks_like_an_instant(name: str, value: Any) -> bool:
    """Whether this field holds a wall-clock instant rather than a number.

    A field called `submitted_at` holding 0.3 is a duration, so the value has to
    land in the epoch window either way. Past that, a float there is taken as an
    instant whatever it is called, and an int only when the name says so.

    The two errors are not symmetric. Missing a stamp puts the machine's speed
    into the floor of whatever domain reads it, silently, and that is how
    `submitted_at` cost a session; shifting something that was not a stamp
    moves it by seconds in seventeen hundred million, which only a reader
    taking a difference could see, and a reader taking a difference is one this
    is for. So a float in the window is enough, and the words are what admits
    a count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not (_EPOCH_FLOOR < float(value) < _EPOCH_CEILING):
        return False
    if isinstance(value, float):
        return True
    lowered = str(name).lower()
    return any(word in lowered for word in _INSTANT_WORDS)


def _shift_anchors(obj: Any, shift: float, depth: int = 0, seen: set[int] | None = None) -> None:
    """Move every wall-clock instant inside this object forward by `shift`.

    A restore rewinds the state but not the clock, so the arm that runs second
    always sees more elapsed time than the arm that ran first. The named
    anchors below are shifted at the top level; everything else is found by
    walking the object, because the instant that decided the largest remaining
    floor term was `submitted_at` on the workspace's last winner, and a bid's
    priority is read as of now — so two arms reading the same winner a second
    apart priced it differently and the workspace domain moved before either
    had been displaced.
    """
    if obj is None or depth > _ANCHOR_DEPTH:
        return
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for name in _CLOCK_ANCHORS:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > _EPOCH_FLOOR:
            try:
                setattr(obj, name, float(value) + shift)
            except (AttributeError, TypeError, ValueError):
                continue
    fields = getattr(obj, "__dict__", None)
    if isinstance(fields, dict):
        for name, value in list(fields.items())[: _ANCHOR_FANOUT * 4]:
            if _looks_like_an_instant(name, value):
                try:
                    setattr(obj, name, float(value) + shift)
                except (AttributeError, TypeError, ValueError):
                    continue
                continue
            _shift_inside(value, shift, depth + 1, seen)
    elif isinstance(obj, (dict, list, tuple)):
        _shift_inside(obj, shift, depth, seen)


def _shift_inside(value: Any, shift: float, depth: int, seen: set[int]) -> None:
    """Follow a container or an ordinary object, and stop at anything else."""
    if depth > _ANCHOR_DEPTH:
        return
    if isinstance(value, (str, bytes, bytearray, np.ndarray, int, float, bool, type(None))):
        return
    if isinstance(value, dict):
        for key, item in list(value.items())[:_ANCHOR_FANOUT]:
            if _looks_like_an_instant(str(key), item):
                try:
                    value[key] = float(item) + shift
                except (TypeError, ValueError):
                    continue
                continue
            _shift_inside(item, shift, depth + 1, seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in list(value)[-_ANCHOR_FANOUT:]:
            _shift_inside(item, shift, depth + 1, seen)
        return
    if hasattr(value, "__dict__") and not _is_process_furniture(value):
        _shift_anchors(value, shift, depth, seen)


#: The environment an arm acts in. The strongest form of the experiment forks
#: (K, E) rather than K alone: she writes a file, reads it back, and what she
#: reads is the evidence her self-model learns efficacy from. Without this the
#: sham arm inherited whatever the displaced arm had just written — the log it
#: appended to, the room it made — and the one action that can fail, reading
#: back a room made on the previous turn, succeeded or failed according to what
#: another arm had done.


def _world_state(root: Path | None) -> dict[str, Any] | None:
    """Every byte under the scratch root, and where the directories are."""
    if root is None or not Path(root).exists():
        return None
    root = Path(root)
    files: dict[str, bytes] = {}
    directories: list[str] = []
    for item in sorted(root.rglob("*")):
        name = str(item.relative_to(root))
        if item.is_dir():
            directories.append(name)
        elif item.is_file():
            try:
                files[name] = item.read_bytes()
            except OSError:
                continue
    return {"files": files, "directories": directories}


def _restore_world(root: Path | None, saved: dict[str, Any] | None) -> None:
    """Put the scratch root back to the bytes it held, and nothing else.

    Through the write gateway, like every other consequential write in the
    tree. A fork that rewinds the filesystem is exactly the kind of write the
    gateway exists to govern, and reaching past it for `unlink`, `mkdir`,
    `rmtree` and `write_bytes` put five raw mutations into the governance
    ledger's regression list the first time the gate ran after it landed.
    """
    if root is None or saved is None:
        return
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    root = Path(root)
    gateway = get_file_write_gateway()
    keep = set(saved["files"]) | set(saved["directories"])
    with local_internal_governed_scope("subject_core.fork"):
        for item in sorted(root.rglob("*"), key=lambda path: len(str(path)), reverse=True):
            name = str(item.relative_to(root))
            if name in keep:
                continue
            try:
                gateway.delete_path(
                    item, recursive=item.is_dir(), source="subject_core.fork"
                )
            except OSError:
                continue
        for name in saved["directories"]:
            gateway.ensure_directory(root / name, source="subject_core.fork")
        for name, payload in saved["files"].items():
            target = root / name
            gateway.ensure_directory(target.parent, source="subject_core.fork")
            try:
                if not target.exists() or target.read_bytes() != payload:
                    gateway.write_bytes(target, payload, source="subject_core.fork")
                # An arm reading a modification time would be reading which arm
                # it is. Nothing in the probe does today; putting the stamp back
                # costs nothing and stops that from becoming true by accident.
                os.utime(target, (0, 0))
            except OSError:
                continue


#: Files a fork leaves to their owners. SQLite manages its own journals, and
#: logs, locks and keys are not state an arm reads back.
_STORE_SKIP_SUFFIXES: tuple[str, ...] = ("-wal", "-shm", "-journal", ".lock", ".log", ".pem")

#: What the fork holds in memory for one run's state root. A quick run's root
#: was 24 MB over 3,178 files, most of them episodic memories and their write
#: receipts; a root past this bound is not one run's own, and holding it would
#: be holding other runs' history in every arm.
STORE_BOUND_BYTES: int = 1024 * 1024 * 1024

#: The latest bytes of every file, keyed by path and by the size and
#: modification time they were read at. A snapshot stats every file and reads
#: only the ones that moved, and two snapshots share the bytes of every file
#: that did not change between them. A whole-root copy per arm scaled with the
#: root; this scales with what one arm changed.
_STORE_CACHE: dict[str, tuple[tuple[tuple[int, int], str], tuple[str, bytes]]] = {}


def _store_root() -> Path | None:
    """The run's own state root, or None when the process was not isolated.

    Only an injected root is forked. A process on the shared default root is
    already not a valid run, and forking its gigabytes of history into each
    arm would hide that behind a slow run.
    """
    injected = os.environ.get("AURA_STATE_ROOT", "")
    if not injected:
        return None
    root = Path(injected)
    return root if root.is_dir() else None


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _sqlite_stamp(path: Path) -> tuple[int, int] | None:
    """A database's stamp includes its write-ahead log, where fresh writes land."""
    own = _stamp(path)
    if own is None:
        return None
    wal = _stamp(path.with_name(path.name + "-wal")) or (0, 0)
    return (own[0] + wal[0], max(own[1], wal[1]))


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_sqlite(path: Path) -> bool:
    """A database by its header, not by whether opening it raised.

    Opening a database can fail for a reason that has nothing to do with what
    the file is, a lock held a moment too long. Deciding "not a database" from
    that downgraded a real store to a byte copy of its main file, whose stamp
    does not move while fresh writes sit in the write-ahead log.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def _sqlite_image(path: Path) -> tuple[bytes, bool]:
    """A consistent image of a database other connections may hold open.

    Through SQLite's backup API, which reads under SQLite's own locks, and not
    a byte copy of a file in write-ahead mode. Returns the image and whether the
    store was in write-ahead mode.

    A write-ahead database serializes with the file-format version bytes at
    offsets 18 and 19 set to 2, and an in-memory connection cannot open such an
    image: deserializing it fails with "unable to open database file". Every
    store a run keeps is in that format, so the image is kept with both bytes
    set to 1, and the restore puts the journal mode back.
    """
    source = sqlite3.connect(str(path), timeout=5.0)
    try:
        image = sqlite3.connect(":memory:")
        try:
            source.backup(image)
            raw = bytes(image.serialize())
        finally:
            image.close()
    finally:
        source.close()
    write_ahead = len(raw) > 19 and raw[18] == 2
    if write_ahead:
        raw = raw[:18] + b"\x01\x01" + raw[20:]
    return raw, write_ahead


def _sqlite_restore(path: Path, image: bytes, write_ahead: bool) -> None:
    """Write an image back through SQLite, so an open connection sees it whole."""
    held = sqlite3.connect(":memory:")
    try:
        held.deserialize(image)
        live = sqlite3.connect(str(path), timeout=5.0)
        try:
            held.backup(live)
            if write_ahead:
                live.execute("PRAGMA journal_mode=WAL")
        finally:
            live.close()
    finally:
        held.close()


def _cached(name: str, path: Path, stamp: tuple[int, int], is_database: bool) -> tuple[str, bytes]:
    """The bytes for this path at this stamp, read only when the stamp moved.

    A failure to read raises. A state file the fork could not capture is a
    file an arm can leak through, and skipping it silently is how the first
    version left every database unrestored.
    """
    entry = _STORE_CACHE.get(name)
    if entry is not None and entry[0][0] == stamp:
        return entry[1]
    if is_database:
        payload, write_ahead = _sqlite_image(path)
        kind = "database-wal" if write_ahead else "database"
    else:
        payload, kind = path.read_bytes(), "file"
    _STORE_CACHE[name] = ((stamp, kind), (kind, payload))
    return (kind, payload)


def _store_state() -> dict[str, Any] | None:
    """Every state file in the run's own root, read only where it moved."""
    root = _store_root()
    if root is None:
        return None
    entries: dict[str, tuple[str, tuple[int, int], bytes]] = {}
    directories: list[str] = []
    total = 0
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if "logs" in relative.parts:
            continue
        name = str(relative)
        if item.is_dir():
            directories.append(name)
            continue
        if not item.is_file() or name.endswith(_STORE_SKIP_SUFFIXES):
            continue
        is_database = _is_sqlite(item)
        stamp = _sqlite_stamp(item) if is_database else _stamp(item)
        if stamp is None:
            continue
        kind, payload = _cached(name, item, stamp, is_database)
        entries[name] = (kind, stamp, payload)
        total += len(payload)
        if total > STORE_BOUND_BYTES:
            raise RuntimeError(
                f"state root {root} holds more than {STORE_BOUND_BYTES} bytes; it is not "
                "one run's own, and forking it into every arm would copy other runs' history"
            )
    for name in list(_STORE_CACHE):
        if name not in entries:
            del _STORE_CACHE[name]
    return {"root": str(root), "entries": entries, "directories": directories}


def _restore_stores(saved: dict[str, Any] | None) -> None:
    """Put the run's state files back to what they held when the fork was taken."""
    if not saved:
        return
    root = _store_root()
    if root is None or str(root) != saved["root"]:
        # The root moved since the snapshot. Writing old bytes into a new root
        # would be writing one run's state into another's.
        return
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    entries: dict[str, tuple[str, tuple[int, int], bytes]] = saved["entries"]
    keep = set(entries) | set(saved["directories"])
    with local_internal_governed_scope("subject_core.fork"):
        for item in sorted(root.rglob("*"), key=lambda path: len(str(path)), reverse=True):
            relative = item.relative_to(root)
            name = str(relative)
            if "logs" in relative.parts or name in keep or name.endswith(_STORE_SKIP_SUFFIXES):
                continue
            try:
                gateway.delete_path(item, recursive=item.is_dir(), source="subject_core.fork")
            except OSError:
                continue
            _STORE_CACHE.pop(name, None)
        for name in saved["directories"]:
            gateway.ensure_directory(root / name, source="subject_core.fork")
        failures: list[str] = []
        for name, (kind, stamp, payload) in entries.items():
            target = root / name
            try:
                if kind.startswith("database"):
                    # Compared with the stamp that counts the write-ahead log,
                    # where a fresh write lands before it reaches the file.
                    if _sqlite_stamp(target) != stamp:
                        _sqlite_restore(target, payload, kind == "database-wal")
                        _STORE_CACHE[name] = ((_sqlite_stamp(target) or stamp, kind), (kind, payload))
                    continue
                if _stamp(target) == stamp:
                    continue
                gateway.ensure_directory(target.parent, source="subject_core.fork")
                gateway.write_bytes(target, payload, source="subject_core.fork")
                # Its old modification time back, so a service caching on the
                # stamp reloads the restored bytes, and the next snapshot finds
                # the cached copy still current instead of reading it again.
                os.utime(target, ns=(stamp[1], stamp[1]))
                _STORE_CACHE[name] = ((stamp, kind), (kind, payload))
            except (OSError, sqlite3.Error) as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        # Raised, not skipped. The first version caught these and carried on,
        # and an arm whose store was not put back contaminates every arm after.
        raise RuntimeError(f"the fork could not restore {len(failures)} state file(s): {failures[:6]}")


def _loop_lock(loop: Any) -> Any:
    """The intention loop's own lock, which every write it makes is taken under."""
    lock = getattr(loop, "_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


def _intentions_state(loop: Any) -> dict[str, Any] | None:
    """Every row the intention database holds, and the loop's memory of them.

    Efficacy, the capability beliefs the self model reads and the comparator's
    attributions are all computed from what is on disk, so an arm that acted
    taught the next arm what it had learned. The rows alone are half of it. The
    loop also keeps its open and recently finished intentions in memory, and a
    restore that rolled back only the rows left the nine intentions a probe arm
    had opened still open for the arm after it. The persist count travels too,
    because it decides when the table is next pruned.
    """
    connection = getattr(loop, "_conn", None)
    if connection is None:
        return None
    with _loop_lock(loop):
        try:
            rows = list(connection.execute("SELECT * FROM intentions"))
        except sqlite3.Error:
            # A database that will not read is not carried across the fork.
            return None
        return {
            "rows": rows,
            "active": copy.deepcopy(getattr(loop, "_active_intentions", None)),
            "completed": copy.deepcopy(getattr(loop, "_completed_intentions", None)),
            "persist_count": getattr(loop, "_persist_count", None),
        }


def _restore_intentions(loop: Any, saved: dict[str, Any] | None) -> None:
    """Roll the intention database and the loop's memory back to the snapshot."""
    connection = getattr(loop, "_conn", None)
    if connection is None or saved is None:
        return
    rows = saved["rows"]
    with _loop_lock(loop):
        try:
            with connection:
                connection.execute("DELETE FROM intentions")
                if rows:
                    marks = ",".join("?" for _ in rows[0])
                    connection.executemany(f"INSERT INTO intentions VALUES ({marks})", rows)
        except sqlite3.Error as exc:
            # The snapshot is still the truth; the arm that could not be rolled
            # back runs on whatever the database holds, and the run has to say so
            # rather than carry a silent difference between two arms.
            record_degradation(
                "subject_driver",
                exc,
                severity="warning",
                action="intentions were not rolled back to the snapshot; this arm starts from live rows",
            )
        # Filled in place, because a caller may hold the dict or the deque, and
        # from a fresh copy, because one snapshot starts many arms and an arm
        # that updates a record must not change the record the next arm gets.
        active = getattr(loop, "_active_intentions", None)
        if active is not None and saved["active"] is not None:
            active.clear()
            active.update(copy.deepcopy(saved["active"]))
        completed = getattr(loop, "_completed_intentions", None)
        if completed is not None and saved["completed"] is not None:
            completed.clear()
            completed.extend(copy.deepcopy(saved["completed"]))
        if saved["persist_count"] is not None:
            loop._persist_count = saved["persist_count"]


def _reanchor(runtime: SubjectRuntime, shift: float) -> None:
    """Move every wall-clock instant forward by the time the restore skipped.

    One shared record of what has been visited, because the state, the organs
    and the services reach many of the same objects and shifting one of them
    twice puts the skipped interval into the reading rather than taking it out.
    """
    if shift <= 0.0:
        return
    state = runtime.state
    for holder in (
        state,
        getattr(state, "motivation", None),
        getattr(state, "cognition", None),
        getattr(state, "soma", None),
    ):
        _shift_anchors(holder, shift)
    seen: set[int] = set()
    for name in runtime.ORGAN_FIELDS:
        _shift_anchors(getattr(runtime.organs, name, None), shift, seen=seen)
    # And the services, which the fork restores and never rewound. The unity
    # layer binds a mind-moment over a four-second window with a 1.2-second
    # half-life, both measured against the wall clock, so two arms binding the
    # same events at different instants scored the binding differently and
    # coherence and fragmentation were the largest floor terms left in the
    # workspace domain. A phase that reads elapsed time has to be handed the
    # same elapsed time in both arms.
    for name, instance in _built_services().items():
        if name in _UNFORKED_SERVICES:
            continue
        _shift_anchors(instance, shift, seen=seen)
    world = getattr(state, "world", None)
    percepts = getattr(world, "recent_percepts", None)
    if isinstance(percepts, list):
        for item in percepts:
            if isinstance(item, dict):
                stamp = item.get("timestamp")
                if isinstance(stamp, (int, float)) and stamp > _EPOCH_FLOOR:
                    item["timestamp"] = float(stamp) + shift


#: Services left alone by the fork. The vault owns the run's database and the
#: container owns the services themselves; rewinding either would break the
#: machinery the measurement runs on rather than the state it measures.
#:
#: There is no allowlist of what the fork considers. There was one — sixteen
#: names, chosen by which services somebody thought a domain read — and the
#: unity layer was not among them, so its four-second binding window and its
#: last mind-moment survived from one arm into the next and coherence and
#: fragmentation stayed the two largest terms in the floor after everything
#: else had been found. A hand-maintained list of what to carry falls behind
#: the tree by construction, and the calibration exists precisely so the
#: question does not have to be answered by hand: take a reading, live a turn
#: in each condition, take another, carry whatever moved.

_UNFORKED_SERVICES: frozenset[str] = frozenset(
    {
        "state_repository",
        "vault",
        "service_container",
        "file_write_gateway",
        # The mycelial topology is guarded against rebinding on purpose and is
        # not per-arm state: it is the wiring the arms both run on. Writing to
        # it raises, and the guard logs a critical before it does.
        "mycelium",
        "mycelial_network",
    }
)


def _built_services() -> dict[str, Any]:
    try:
        from core.container import ServiceContainer

        built = getattr(ServiceContainer, "_services", {}) or {}
    except (AttributeError, ImportError):
        return {}
    out: dict[str, Any] = {}
    for name in list(built):
        try:
            instance = built.get(name)
        except (KeyError, RuntimeError, TypeError):
            continue
        instance = getattr(instance, "instance", instance)
        if instance is not None and hasattr(instance, "__dict__"):
            out[name] = instance
    return out


def _differs(left: Any, right: Any) -> bool:
    """Whether two captures hold different numbers. Any doubt counts as yes."""
    if left is None or right is None:
        return left is not right
    try:
        return repr(left) != repr(right)
    except (RecursionError, TypeError, ValueError):
        # Unreadable counts as different: any doubt is a difference.
        return True


def _service_state(only: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """The mutable state of everything the container has already built."""
    out: dict[str, dict[str, Any]] = {}
    for name, instance in _built_services().items():
        if name in _UNFORKED_SERVICES:
            continue
        if only is not None and name not in only:
            continue
        try:
            captured = _organ_state(instance)
        except (
            ArithmeticError,
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            # A service that cannot be read is skipped, by kind rather than by
            # catching everything.
            continue
        if captured:
            out[name] = captured
    return out


def _restore_services(saved: Mapping[str, dict[str, Any]]) -> None:
    if not saved:
        return
    built = _built_services()
    for name, fields in saved.items():
        instance = built.get(name)
        if instance is not None:
            _restore_organ(instance, fields)


def _effort_state() -> dict[str, float] | None:
    try:
        from core.soma.effort import get_effort_ledger

        return dict(get_effort_ledger().peek())
    except (ImportError, RuntimeError):
        return None


def _restore_effort(saved: dict[str, float] | None) -> None:
    if saved is None:
        return
    try:
        from core.soma.effort import get_effort_ledger

        ledger = get_effort_ledger()
        ledger.drain()
        for kind, amount in saved.items():
            ledger.note(kind, amount)
    except (ImportError, RuntimeError):
        return


#: The substrate's own loop body, and how often each part of it comes round.
#:
#: Its loop steps the dynamics and settles the psych state every iteration,
#: computes the recurrent self-model every fifth and applies Hebbian plasticity
#: every hundredth. The harness called `_step_dynamics`, which is the first of
#: those and is marked deprecated in the substrate itself, so the energy channel
#: never regenerated, the integrated-information estimate never advanced and the
#: connectivity never learned — in any run.
#:
#: Persistence is deliberately absent. Writing the state to disk is not
#: cognition, and a save that lands in one arm and not the other is a difference
#: between the arms that nothing thought.
#:
#: Each entry is a method, how many of the substrate's own ticks apart it runs,
#: and whether it takes the step size. It is hashed into the campaign like every
#: other layer's body, because it decides what organism the run measured.
SUBSTRATE_BODY: tuple[tuple[str, int, bool], ...] = (
    ("_step_torch_math", 1, True),
    ("_stabilize_psych_state", 1, True),
    ("_recurrent_self_model", 5, True),
    ("_apply_plasticity", 100, False),
)


#: Module-level singletons the fork has to carry, named by where they live.
#:
#: A service in the container is carried under its name and a phase's own
#: attributes are carried with the phase. What is left is state that lives at
#: module scope and is reachable from neither: an accessor returns the one
#: object and nothing holds a reference to it. The interiority layer publishes
#: every faculty through a synaptic cleft got this way, and that cleft carries
#: facilitation per channel and a receptor bank that adapts — so an arm that
#: felt something left the medium more excitable for the arm that followed it,
#: under every domain the interiority layer touches.
#:
#: Each entry is a name and a zero-argument accessor. Adding one is the whole
#: cost of bringing a new module singleton into the fork.
_MODULE_SINGLETONS: tuple[tuple[str, str, str], ...] = (
    ("interiority.cleft", "core.interiority.cleft", "get_cleft"),
    ("interiority.receptors", "core.interiority.receptors", "get_receptor_bank"),
)


def _singleton_state() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, module_name, accessor in _MODULE_SINGLETONS:
        try:
            module = importlib.import_module(module_name)
            instance = getattr(module, accessor)()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            continue
        try:
            captured = _organ_state(instance)
        except (
            ArithmeticError, AttributeError, ImportError, LookupError,
            OSError, RuntimeError, TypeError, ValueError,
        ):
            continue
        if captured:
            out[name] = captured
    return out


def _restore_singletons(saved: Mapping[str, dict[str, Any]]) -> None:
    if not saved:
        return
    for name, module_name, accessor in _MODULE_SINGLETONS:
        fields = saved.get(name)
        if not fields:
            continue
        try:
            module = importlib.import_module(module_name)
            _restore_organ(getattr(module, accessor)(), fields)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            continue


def _torch_random_state() -> Any:
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch.get_rng_state().clone()
    except (AttributeError, RuntimeError):
        return None


def _restore_torch_random(saved: Any) -> None:
    if saved is None:
        return
    try:
        import torch

        torch.set_rng_state(saved)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return



def _lifetime_last() -> Any:
    """The lifetime module's own copy of the last step.

    `core.ontogeny.lifetime` publishes what the reservoir sensed through a
    module-level global rather than through the service, so nothing that forks
    a run could carry it by carrying the service. The driver reads it once per
    turn to fill N's novelty and displacement columns.
    """
    try:
        from core.ontogeny import lifetime

        with lifetime._lock:
            return lifetime._last
    except (ImportError, AttributeError):
        return None


def _restore_lifetime_last(saved: Any) -> None:
    try:
        from core.ontogeny import lifetime

        with lifetime._lock:
            lifetime._last = saved
    except (ImportError, AttributeError):
        return


def _moments_of(service: Any) -> dict[str, Any] | None:
    """Everything the ontogeny service mutates that is not in a snapshot yet.

    The running moments normalise the design row, so an arm that displaced a
    domain shifted the mean that the next arm was measured against. The last
    reading is what `novelty` reports. And the service's reservoir is a
    different object from the one the N domain reads — the affect phase steps
    that one every turn — so it drifts across the fork unless it is carried.
    """
    if service is None:
        return None
    saved: dict[str, Any] = {}
    moments = getattr(service, "_advance_moments", None)
    if moments is not None:
        saved["moments"] = (
            np.array(moments.count, copy=True),
            np.array(moments.mean, copy=True),
            np.array(moments.m2, copy=True),
        )
    reservoir = getattr(service, "_state", None)
    if reservoir is not None:
        saved["reservoir"] = {
            name: (
                np.array(value, copy=True)
                if isinstance(value := getattr(reservoir, name, None), np.ndarray)
                else value
            )
            for name in _RESERVOIR_FIELDS
        }
    return saved or None


def _restore_moments(service: Any, saved: dict[str, Any] | None) -> None:
    if service is None or not saved:
        return
    moments = getattr(service, "_advance_moments", None)
    columns = saved.get("moments")
    if moments is not None and columns is not None:
        moments.count = np.array(columns[0], copy=True)
        moments.mean = np.array(columns[1], copy=True)
        moments.m2 = np.array(columns[2], copy=True)
    reservoir = getattr(service, "_state", None)
    fields = saved.get("reservoir")
    if reservoir is not None and fields:
        for name, value in fields.items():
            setattr(
                reservoir, name, np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            )



class _HeldObserver:
    """The shared host observer, answering every question the same way.

    A thin memoising face over the real one rather than a second
    implementation of the protocol: the first call of each shape passes
    through and its answer stands until the hold is released. What the machine
    is doing is the environment, and an experiment that compares two arms has
    to hold the environment still or the comparison is between two machines.
    """


    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._answers: dict[tuple, Any] = {}

    @property
    def provenance(self) -> Any:
        return self._inner.provenance

    def _held(self, name: str, args: tuple, kwargs: dict) -> Any:
        key = (name, args, tuple(sorted(kwargs.items())))
        if key not in self._answers:
            self._answers[key] = getattr(self._inner, name)(*args, **kwargs)
        return self._answers[key]

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute
        return lambda *args, **kwargs: self._held(name, args, kwargs)


def _delegate_to_the_real_observer() -> None:
    """Give the held face a real method per protocol member.

    `isinstance` against a runtime-checkable protocol looks the members up
    statically, so `__getattr__` alone does not satisfy it and the installer
    refuses the object. Written out of the protocol's own attribute list rather
    than by hand, so the face cannot fall behind what the protocol asks for.
    """
    try:
        from core.runtime.resource_observation import ResourceObserver
    except (ImportError, AttributeError):  # pragma: no cover - shipped together
        return
    for name in getattr(ResourceObserver, "__protocol_attrs__", ()):  # type: ignore[attr-defined]
        if name == "provenance" or hasattr(_HeldObserver, name):
            continue

        def make(method: str):
            def call(self, *args: Any, **kwargs: Any) -> Any:
                return self._held(method, args, kwargs)

            call.__name__ = method
            call.__qualname__ = f"_HeldObserver.{method}"
            return call

        setattr(_HeldObserver, name, make(name))


_delegate_to_the_real_observer()


@dataclass
class Snapshot:
    """Everything a fork has to carry for two arms to start from one place."""

    state: Any
    hidden: np.ndarray
    steps: int
    era: int
    centre: np.ndarray
    scatter: np.ndarray
    centre_n: float
    turn: int
    rng_state: tuple
    #: The ontogeny service's own accumulators, which are not in the reservoir
    #: and are mutated by every step. The running moments normalise the design
    #: row, so an arm that displaced a domain shifted the mean the next arm was
    #: measured against; and the last reading is what `novelty` reports. Both
    #: are process-wide, so without carrying them the sham arm started from a
    #: state the displaced arm had already moved — a floor on N of nearly two
    #: standard deviations before a single phase had run.
    #: The scratch world, byte for byte, and the intention loop's rows and memory. The
    #: experiment forks (K, E): what she wrote in one arm is not there for the
    #: next, and what she learned from writing it is not either.
    world: dict[str, Any] | None = None
    intentions: dict[str, Any] | None = None
    #: The harness's frame count. Which free-running layers step on a given
    #: frame is a function of this number, so two arms that do not start from
    #: the same one are not running the same organism.
    frame_index: int = 0
    moments: dict[str, Any] | None = None
    last_reading: Any = None
    #: What the last step sensed, which is what the N domain reports as novelty
    #: and displacement. `_step_ontogeny` copies it onto the reservoir object
    #: and `core.ontogeny.lifetime` keeps its own copy behind a module-level
    #: global that no container holds, so neither travelled with the fork: two
    #: sham arms restored from one snapshot read different novelty on their
    #: opening frame, before a phase had run. Nine tenths of a standard
    #: deviation of the largest floor term in the battery.
    last_novelty: float = 0.5
    last_displacement: float = 0.0
    lifetime_last: Any = None
    #: The process-wide generators. Phases draw from `random` and `numpy.random`
    #: directly — the affect decay adds a Gaussian drift on every turn, and for
    #: an emotion that never otherwise moves that drift is the whole of the
    #: column's recorded spread. Two arms drawing different numbers from it
    #: differ by more than a standard deviation before anything has happened.
    #: Two arms have to see the same computation, and a generator the phases
    #: draw from is part of the computation.
    global_random: Any = None
    numpy_random: Any = None
    #: Every service the container has already built, captured the same way
    #: the organs are. The list of organs was a list of the ones already
    #: thought of: conversation dynamics accumulates topic anchors, the self
    #: model accumulates beliefs, and each of those left half a standard
    #: deviation in the floor of a domain that reads it. What the fork has to
    #: carry is everything that persists, not everything that was remembered.
    services: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: The phases' own accumulators. A phase is not stateless: it caches the
    #: engine it talks to, and those engines are module-level singletons that
    #: no container holds — conversation dynamics accumulates topic anchors
    #: behind one, and that left half a standard deviation in the workspace
    #: domain's floor with nothing in the container to carry.
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: The effort ledger's pending total. It is a process-wide singleton, and
    #: an arm that thought harder left its exertion on the counter for the next
    #: arm to drain — six tenths of a standard deviation of the body's newest
    #: channel, before either arm had been displaced.
    effort: dict[str, float] | None = None

    #: Module-level singletons the container does not hold. See
    #: `_MODULE_SINGLETONS`.
    singletons: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: Where the experiment clock stood. Restoring rewinds the state, and the
    #: clock the phases read is part of the state as far as they are concerned:
    #: a drive decays by `decay * dt` and a mind-moment binds over a window,
    #: both against `time.time`.
    clock_at: float | None = None
    #: Wall clock when the snapshot was taken. Restoring rewinds the state but
    #: not the clock, so the second arm of a trial always sees more elapsed
    #: time than the first — the motivation phase decays every drive by
    #: `now - last_tick`, so the arms differ by however long the first one took
    #: before either has been displaced.
    taken_at: float = 0.0
    #: Torch's global generator. The substrate draws its integration noise from
    #: `torch.randn` on every step, so two arms integrating the same state
    #: diverged by more than a standard deviation of the substrate's own
    #: spread — the largest single term left in the floor once the organs
    #: themselves were carried.
    torch_random: Any = None
    #: The organs are process-wide singletons. Without carrying them across the
    #: fork, whatever the displaced arm did to the workspace, the self model,
    #: the world model or the agency ledger was still there when the sham arm
    #: ran, and the comparison was between an untouched organism and one that
    #: had already been touched.
    organs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: What each kind of action did for her last time, which is what she
    #: predicts from. An arm that inherited the other arm's experience would
    #: predict from something that did not happen to it, and whether she was
    #: right about her own effect is one of the things the ownership experiment
    #: has to span.
    outcomes_by_kind: dict[str, bool] = field(default_factory=dict)
    #: The run's own state files. The services' Python state was forked and
    #: the files they write and read back were not, so an arm of three turns
    #: left seventeen changed files and three new episodic memories for the
    #: next arm to recall.
    stores: dict[str, Any] | None = None


#: The three names other modules ask the driver for. Kept as aliases rather
#: than renamed at every call site: the private spelling is what 29 imports
#: already use, and a rename is a separate change from a move.
reanchor = _reanchor
restore_singletons = _restore_singletons
singleton_state = _singleton_state
