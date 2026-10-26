"""How the fork copies what it carries, and how it tells two copies apart.

Split out of core/subject/snapshot.py when that module crossed the size
ceiling. Three things live here, all about the values a snapshot holds rather
than about which of the organism's parts it holds them for:

- which objects are process furniture (a lock, a socket, a task) that no
  snapshot carries or compares;
- `identical`, an exact comparison of two copies that treats any doubt as a
  difference;
- the sharing that lets consecutive snapshots hold one copy of a field that
  did not move, and the reducer that lets a read-only mapping be copied at all.
"""

from __future__ import annotations

import asyncio
import copyreg
import datetime
import decimal
import fractions
import itertools
import math
import random
import sys
import types
from collections import deque
from pathlib import PurePath
from typing import Any

import numpy as np

from core.subject.monotonic_instants import MonotonicInstant

#: A field an object does not have, as distinct from one that holds None.
_ABSENT = object()


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
    import ctypes
    import io
    import multiprocessing
    import multiprocessing.connection
    import multiprocessing.queues
    import multiprocessing.sharedctypes
    import multiprocessing.synchronize
    import queue as _queue_module
    import socket as _socket_module
    import sqlite3
    import threading as _threading
    from concurrent.futures import Executor, Future

    from core.memory.embedding_runtime import SharedEmbeddingLease, SharedEmbeddingRuntime
    from core.runtime.descriptor_owner import OwnsDescriptors

    return (
        # An object that holds OS descriptors as ints is a handle too. Copied,
        # it became a second owner of the same descriptor numbers, and the
        # whole report run of 23 September died of the double close (EXC_GUARD
        # on descriptor 39). See core/runtime/descriptor_owner.py.
        OwnsDescriptors,
        # The process's one embedding engine and the claims on it. The fork
        # carried the runtime as a module holding, so restoring a coordinator's
        # anchor in a carrier worker wrote the coordinator's list of open
        # leases over the worker's, every lease the worker held was released
        # under it, and all three workers of seed 73's carrier died the first
        # time they enumerated module state (26 October). Swapped between
        # anchors 16 times on seed 73, the runtime never moved K: it is the
        # machine's bookkeeping of a loaded model, not something she keeps.
        SharedEmbeddingLease,
        SharedEmbeddingRuntime,
        # A pipe, a queue, a lock or a value shared with a model worker. A copy
        # of a pipe closed the worker's descriptor when it was collected, even
        # from a copy that failed part-way, and the dry run of 23 September
        # 01:44 died of it with the report run's EXC_GUARD. A shared value
        # copied and restored into the client is private memory the worker
        # never reads. Raw ctypes values in this tree are that shared memory.
        multiprocessing.connection.Connection,
        multiprocessing.queues.Queue,
        multiprocessing.queues.SimpleQueue,
        multiprocessing.synchronize.SemLock,
        multiprocessing.synchronize.Condition,
        multiprocessing.synchronize.Event,
        multiprocessing.sharedctypes.SynchronizedBase,
        ctypes.Array,
        ctypes._SimpleCData,
        _threading.Event,
        _threading.Condition,
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
        # And the threads' own queue. Its lock and conditions were furniture
        # and its counts were not, so a restore rewound `unfinished_tasks`
        # under a consumer that had already taken an item: the state
        # registry's dispatcher died of "task_done() called too many times"
        # on the seed-7 run of 26 September.
        _queue_module.Queue,
        _queue_module.SimpleQueue,
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


#: Values no arm can mutate, so a restore can hand the same object to all three
#: without copying it. Most of what an organ carries is one of these, and
#: copying them was most of what a restore cost.
_ATOMIC: tuple[type, ...] = (str, bytes, int, float, bool, type(None))


#: CPython's flag on a class made by a class statement rather than written in C.
_HEAP_TYPE = 1 << 9


#: Whether a type's whole state is its `__dict__`, by type. The answer is a
#: property of the class, and a fork asks it thousands of times a run.
_WHOLE_STATE_IS_DICT: dict[type, bool] = {}


def _state_is_its_dict(value: Any) -> bool:
    """Whether everything this object holds is in its `__dict__`.

    Only then can it be restored by writing its fields back. A tensor has a
    `__dict__` as well, and it is empty: the numbers live in the C base class.
    Restored field by field, a tensor held on a plain object kept the values an
    arm had written into it. So every class above `object` has to be one
    written in Python, and none may declare slots.

    The heap-type flag used to carry that on its own and no longer does. CPython
    converted the standard library's C types to heap types, so `random.Random`
    passes it: its `__dict__` holds `gauss_next` and its Mersenne state lives in
    `_random.Random`. Restored field by field it had `gauss_next` written, the
    restore reported success, and the generator went on from wherever the last
    arm left it. That was a floor under every paired measurement — the workspace
    draws its somatic noise from one of these — and it is the same failure the
    paragraph above describes, arriving through a door that opened underneath it.

    So the object is also asked for its own account of its state.
    `__reduce_ex__` is what a copy is built from, and a class whose state is
    exactly its `__dict__` says so there. One that keeps numbers in a C base
    returns them instead, and is restored by writing a copy over the field.
    """
    if not hasattr(value, "__dict__") or isinstance(value, type):
        return False
    kind = type(value)
    known = _WHOLE_STATE_IS_DICT.get(kind)
    if known is not None:
        return known
    answer = all(
        klass.__flags__ & _HEAP_TYPE and "__slots__" not in vars(klass)
        for klass in kind.__mro__[:-1]
    ) and _reduces_to_its_dict(value)
    _WHOLE_STATE_IS_DICT[kind] = answer
    return answer


def _reduces_to_its_dict(value: Any) -> bool:
    """Whether the object's own account of its state is exactly its `__dict__`."""
    try:
        reduced = value.__reduce_ex__(2)
    except Exception:  # noqa: BLE001 - every way this fails has one answer
        # not a failure: an object that cannot say what its state is gets a copy
        # written over it, which carries whatever a deep copy carries.
        return False
    if not isinstance(reduced, tuple) or len(reduced) < 3:
        return True
    state = reduced[2]
    if state is None:
        return True
    if not isinstance(state, dict):
        return False
    return set(state) == set(vars(value))


def _rebuild_mapping_proxy(items: dict) -> types.MappingProxyType:
    return types.MappingProxyType(items)


def _reduce_mapping_proxy(proxy: types.MappingProxyType) -> tuple[Any, tuple[dict]]:
    return _rebuild_mapping_proxy, (dict(proxy),)


#: A read-only mapping copies as a read-only mapping over a copied dict. Without
#: this `deepcopy` refuses it, the object holding one is taken apart field by
#: field instead of copied, and a frozen dataclass taken apart cannot be written
#: back: the affect engine's last stimulus receipt, whose appraisal is one,
#: survived every restore holding the arm's event.
copyreg.pickle(types.MappingProxyType, _reduce_mapping_proxy)


#: The last copy taken of each field, by the owner's id and the field's name.
#: A snapshot is never mutated: a restore copies out of it (`_place`,
#: `np.copyto`, a rebuilt dict). So when a field holds exactly what it held at
#: the last snapshot, the new snapshot can hold the same copy. The substrate's
#: private subsystems and most weight matrices do not move between two anchors,
#: and copying them again for each of 128 anchors was half of every fork.
_LAST_COPY: dict[tuple[int, str], Any] = {}


def _unchanged_or(owner: int, name: str, fresh: Any) -> Any:
    """The copy the last snapshot kept, if `fresh` is identical to it; else `fresh`."""
    key = (owner, name)
    kept = _LAST_COPY.get(key, _ABSENT)
    if kept is not _ABSENT and identical(kept, fresh):
        return kept
    _LAST_COPY[key] = fresh
    return fresh


def identical(left: Any, right: Any) -> bool:
    """Whether two copies hold the same thing, exactly. Any doubt is no.

    Same type all the way down, the same bits in every float (a signed zero
    is not zero), NaN where the other has NaN, the same dtype and shape for an
    array, the same key order for a dict. A type this does not know how to
    read counts as different, so the worst a gap here costs is a copy.
    """
    try:
        return _identical(left, right, set())
    except (AttributeError, RecursionError, RuntimeError, TypeError, ValueError):
        # not a failure: a comparison that could not be finished is a doubt, and
        # a doubt is a copy (bfloat16 tensors will not become numpy arrays).
        return False


#: Immutable types whose equality is their whole content.
_VALUES: tuple[type, ...] = (
    str, bytes, bytearray, int, bool, type(None), complex, PurePath, datetime.date,
    datetime.time, datetime.timedelta, decimal.Decimal, fractions.Fraction, range,
)


#: A tagged monotonic instant is the float it was, so a snapshot that tagged a
#: field can still hand its copy to the next snapshot when nothing changed.
_FLOATS: tuple[type, ...] = (float, MonotonicInstant)


def _identical(left: Any, right: Any, pairs: set[tuple[int, int]]) -> bool:
    if left is right:
        return True
    if type(left) is not type(right) and not (type(left) in _FLOATS and type(right) in _FLOATS):
        return False
    if isinstance(left, float):
        return (left == right and math.copysign(1.0, left) == math.copysign(1.0, right)) or (
            left != left and right != right
        )
    if isinstance(left, _VALUES):
        return left == right
    if isinstance(left, random.Random):
        return left.getstate() == right.getstate()
    if isinstance(left, np.random.Generator):
        return _identical(left.bit_generator.state, right.bit_generator.state, set())
    if isinstance(left, np.random.RandomState):
        return _identical(left.get_state(legacy=False), right.get_state(legacy=False), set())
    if isinstance(left, np.random.BitGenerator):
        return _identical(left.state, right.state, set())
    readable = _read_foreign(left, right)
    if readable is not None:
        return readable
    if isinstance(left, itertools.count):
        return repr(left) == repr(right)
    if isinstance(left, types.MethodType):
        # A copied method is bound to a copy of its object: the same function,
        # and an object holding the same things.
        return left.__func__ is right.__func__ and _identical(left.__self__, right.__self__, pairs)
    if isinstance(left, (types.SimpleNamespace, types.MappingProxyType)):
        return _identical(dict(vars(left) if hasattr(left, "__dict__") else left), dict(vars(right) if hasattr(right, "__dict__") else right), pairs)
    if isinstance(left, types.BuiltinMethodType):
        # `dict.items` bound to a dict, as a view keeps it: the same method of
        # an object holding the same things.
        return left.__name__ == right.__name__ and _identical(left.__self__, right.__self__, pairs)
    if isinstance(left, _furniture()):
        # A lock or a queue handle has no state a restore puts back.
        return True
    pair = (id(left), id(right))
    if pair in pairs:
        # Already being compared further up a cycle.
        return True
    pairs.add(pair)
    if isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.dtype.hasobject:
            return all(
                _identical(a, b, pairs)
                for a, b in zip(left.ravel().tolist(), right.ravel().tolist(), strict=True)
            )
        # Byte for byte: a signed zero and a NaN's payload are both kept.
        return left.tobytes() == right.tobytes()
    if isinstance(left, dict):
        if getattr(left, "default_factory", None) is not getattr(right, "default_factory", None):
            return False
        if hasattr(left, "__dict__") and not _identical(vars(left), vars(right), pairs):
            return False
        return list(left) == list(right) and all(_identical(left[k], right[k], pairs) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _identical(a, b, pairs) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, deque):
        return (
            left.maxlen == right.maxlen
            and len(left) == len(right)
            and all(_identical(a, b, pairs) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, (set, frozenset)):
        return all(isinstance(item, _ATOMIC) for item in left) and left == right
    tensor = _tensor_type()
    if tensor is not None and isinstance(left, tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.requires_grad == right.requires_grad
            and left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes()
        )
    slots = _slot_names(type(left))
    if slots is None:
        return False
    if any(
        hasattr(left, name) != hasattr(right, name)
        or (hasattr(left, name) and not _identical(getattr(left, name), getattr(right, name), pairs))
        for name in slots
    ):
        return False
    if hasattr(left, "__dict__"):
        # An object's attributes, read as a set: the order they were assigned
        # in is not something a phase can see.
        mine, theirs = vars(left), vars(right)
        return set(mine) == set(theirs) and all(_identical(mine[k], theirs[k], pairs) for k in mine)
    return True


def _furniture() -> tuple[type, ...]:
    global _FURNITURE
    if not _FURNITURE:
        _FURNITURE = _furniture_types()
    return _FURNITURE


def _slot_names(klass: type) -> tuple[str, ...] | None:
    """Every slot a Python-written class declares, or None for a type this cannot read."""
    names: list[str] = []
    for base in klass.__mro__[:-1]:
        if not base.__flags__ & _HEAP_TYPE:
            return None
        declared = vars(base).get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        names.extend(name for name in declared if name not in ("__dict__", "__weakref__"))
    return tuple(names)


def _read_foreign(left: Any, right: Any) -> bool | None:
    """Exact comparison for the library types an organ holds; None for any other type.

    torch's device, dtype and generator, and scipy's sparse matrices. None of
    them keeps its content in a `__dict__`, so without this each counted as
    changed on every snapshot and was copied again.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        if isinstance(left, (torch.device, torch.dtype)):
            return bool(left == right)
        if isinstance(left, torch.Generator):
            return bool(torch.equal(left.get_state(), right.get_state()))
    sparse = sys.modules.get("scipy.sparse")
    if sparse is not None and sparse.issparse(left):
        if left.shape != right.shape or left.dtype != right.dtype or left.format != right.format:
            return False
        a, b = left.tocsr(copy=True), right.tocsr(copy=True)
        a.sort_indices()
        b.sort_indices()
        return (
            a.indptr.tobytes() == b.indptr.tobytes()
            and a.indices.tobytes() == b.indices.tobytes()
            and a.data.tobytes() == b.data.tobytes()
        )
    return None


def _tensor_type() -> type | None:
    # Read each time: torch may be imported after the first snapshot.
    torch = sys.modules.get("torch")
    return getattr(torch, "Tensor", None)
