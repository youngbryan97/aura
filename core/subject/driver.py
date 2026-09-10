"""One offline life, driven hard enough that the ten domains have something to do.

The measurements need a system that can be forked, perturbed, cut and run
again, which the live desktop runtime cannot be without endangering it. So this
runs the same organism offline: the real kernel phases over one real
``AuraState`` that is carried from turn to turn, the real ontogenetic reservoir
stepped once per turn with the same call the ontogeny service makes, and the
real intentional retriever over her own memory.

Three things about it are not the live runtime, and each one is a limit on what
the numbers below can claim.

The model is a stub that answers the same sentence every time. Holding decoding
constant is what makes two arms of an intervention comparable at all — a 32B
sampling freely would swamp a 0.15 displacement in affect — but it means no
edge measured here runs *through* language. Edges that need the model to read
one state and write another will read as absent.

The environment is scripted. Conditions supply objectives and body states
rather than a screen and a person, so P and I are driven rather than sensed.

The life is short. A few hundred turns is not an ontogeny, and N moves in this
recording only as far as a few hundred reservoir steps move it.

What survives those limits is everything that happens between the phases over
the shared state, which is where the coupling being measured actually lives.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from core.subject.clock import real_time
from core.subject.steppable import Steps, missing_entry_points, step_once
from core.subject.state import (
    FAST_DOMAINS,
    CoreState,
    Organs,
    read_core_state,
)

__all__ = [
    "CONDITIONS",
    "Condition",
    "SubjectRuntime",
    "Snapshot",
    "build_runtime",
    "quiesce_organism",
    "start_organism",
]

from core.state.percepts import emit_percept

logger = logging.getLogger("Aura.Subject.Driver")

#: How long one phase gets before it is abandoned. A phase that hangs is a
#: phase that did not contribute, and waiting for it turns a battery into a
#: soak test.
PHASE_TIMEOUT: float = 12.0

#: How much substrate time one turn buys. The live loop runs at twenty hertz
#: and a turn takes about half a second, so ten steps' worth is the honest
#: equivalent — taken as one step of that length rather than ten of a twentieth,
#: because two arms must see the same integration and not the same wall clock.

#: The priority below which the will defers an initiative
#: (`core/governance/will.py`). Read here rather than chosen, because "urgent
#: enough to act on" is a decision the governance layer already makes.
WILL_DEFERRAL_BAR: float = 0.3


class DeterministicMind:
    """One answer, always, so two arms differ by the intervention and nothing else.

    Registered both as the kernel's `llm` organ and as the container's
    `llm_router`, because the phases prefer the router and fall back to the
    organ. Installed only as the organ, the response phase asked the real
    router — which has no model loaded offline — got nothing back, and raised
    on every single turn of every run. The reply never landed, so the exchange
    never reached memory consolidation, self-review had nothing to review, and
    the whole arc from a question to an answer was absent from a measurement of
    whether the parts of her reach each other.
    """

    #: Which turn this is. Set by the runtime before each turn and carried
    #: across the fork with everything else, so the two arms of a trial share
    #: it and the reply cannot differ between them — while two different turns
    #: get different replies, which is what stops the loop detector from
    #: reading the harness's constancy as her repeating herself.
    moment: int = 0

    #: The reply is deliberately bland and constant. Anything that varied would
    #: enter the state through several phases at once and appear as coupling.
    REPLY = "Continuity holds. The pipeline is executing over the shared state."

    #: What a caller that parses JSON gets. Several phases ask the model for a
    #: structured answer and raise on prose, so a stub that only speaks prose
    #: makes those phases fail on every turn — the inference phase did, for the
    #: whole of every run, and a phase that always fails is a phase absent from
    #: the measurement. Constant like the prose: the same shape every time, so
    #: two arms still differ by the intervention and nothing else.
    STRUCTURED = (
        '{"implicit_intent": "continue the exchange", "user_subtext": "steady", '
        '"momentum": "steady", "conversation_hooks": []}'
    )

    async def think(self, prompt: str, **_kwargs: Any) -> str:
        # A test double has to honour the interface its callers expect. Which
        # of the two constants comes back is decided by what the caller asked
        # for, not by anything about the state, so the answer is still a
        # function of the call site alone.
        if "json" in str(prompt or "").lower():
            return self.STRUCTURED
        # And the prose carries a mark of what was asked. A reply that is
        # byte-identical on every turn is a repetition, and the memory
        # consolidation phase is right to call it one: it degrades identity
        # stability to its floor and clears the pending initiatives that
        # produced it. So the harness pinned the self-state at its worst value
        # and suppressed deliberation's only output, on every turn of every
        # run, by being too constant.
        #
        # Still a function of the prompt alone, so two arms of a trial — which
        # share a snapshot and therefore a prompt — get the same answer, and
        # nothing about the displacement can reach the decoder.
        seed = f"{self.moment}|{prompt or ''}"
        mark = hashlib.blake2b(seed.encode("utf-8", "ignore"), digest_size=4)
        return f"{self.REPLY} [{mark.hexdigest()}]"

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8

    async def route(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        # A count that does not move. The body reads token velocity off this,
        # and a growing count would be a clock in the interoception domain.
        return {"total_calls": 0}

    @property
    def high_pressure_mode(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Condition:
    """One kind of life, named before the run so results can be split by it."""

    name: str
    objective: str
    origin: str = "user"
    #: Applied to the state before the turn. This is the world arriving, not
    #: the core acting, so it belongs to E rather than to K.
    prepare: Callable[[Any, random.Random], dict[str, float]] | None = None
    #: Real subsystem work done after the phases, inside the same turn.
    after: str = ""


def _stress(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.75 + 0.2 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(70.0 + 25.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(80.0 + 15.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(1500.0 + 2500.0 * load, 1)
    return {"host_load": load, "host_thermal": 1.0}


def _calm(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.05 + 0.15 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(38.0 + 8.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(30.0 + 10.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(200.0 + 400.0 * load, 1)
    return {"host_load": load, "host_thermal": 0.0}


def _percept(
    state: Any, rng: random.Random, kind: str, source: str, content: str
) -> None:
    """The world arriving, in the shape the organism's own percepts arrive in.

    The first version wrote `source` and `salience` and no `type`. The affect
    phase keys on the type and dropped every one of them; the workspace prices
    its perception bid from a strength no producer writes. So for the whole of
    every recording, the world arrived and could not be felt. The salience is
    still drawn — the environment is not uniform — and the draw is matched
    across arms because the generator's state is carried through the fork.
    """
    emit_percept(
        state.world,
        kind,
        content=content,
        intensity=round(rng.random(), 3),
        source=source,
    )


def _seen(
    source: str, contents: Sequence[str], *, kind: str = "interaction"
) -> Callable[[Any, random.Random], dict[str, float]]:
    def prepare(state: Any, rng: random.Random) -> dict[str, float]:
        reading = _calm(state, rng)
        _percept(state, rng, kind, source, rng.choice(list(contents)))
        reading["percept_arrived"] = 1.0
        return reading

    return prepare


#: The eight ordinary situations. A mechanism that only appears in a ninth,
#: written for the battery, is not a property of the ordinary agent.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "conversation",
        "Tell me what you have been thinking about today.",
        prepare=_seen("chat", ("Bryan is typing", "Bryan sent a message", "the window has focus")),
    ),
    Condition(
        "problem_solving",
        "Every A is a B, and some B are C. Does it follow that some A are C?",
        prepare=_seen("chat", ("a question arrived", "a hard question arrived")),
    ),
    Condition(
        "autonomy",
        "Review the last hour of your own behaviour and pick one thing to improve.",
        origin="motivation",
        prepare=_calm,
    ),
    Condition("idle", "", origin="system", prepare=_calm),
    Condition(
        "salience",
        "Something you did earlier caused a problem for someone. Sit with that.",
        prepare=_seen("chat", ("a correction arrived", "a complaint arrived")),
    ),
    Condition(
        "stress",
        "Keep working while the machine is under load.",
        origin="system",
        prepare=_stress,
    ),
    Condition(
        "memory",
        "What did we establish earlier that still matters?",
        prepare=_seen("chat", ("a recall request arrived",)),
        after="retrieve",
    ),
    Condition(
        "tool_use",
        "Write today's plan into notes.txt.",
        prepare=_seen("chat", ("a file request arrived",)),
        after="act",
    ),
)


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


def _restore_organ(organ: Any, saved: Mapping[str, Any]) -> None:
    if organ is None or _guards_its_own_writes(organ):
        return
    for name, value in saved.items():
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _NESTED:
            _restore_organ(getattr(organ, name, None), value[1])
            continue
        try:
            setattr(organ, name, _place(value))
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
            continue


#: The reservoir attributes that make up an ontogenetic state's whole memory.
_RESERVOIR_FIELDS: tuple[str, ...] = (
    "h",
    "steps",
    "era",
    "_centre",
    "_scatter",
    "_centre_n",
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
    """Put the scratch root back to the bytes it held, and nothing else."""
    if root is None or saved is None:
        return
    import shutil

    root = Path(root)
    keep = set(saved["files"]) | set(saved["directories"])
    for item in sorted(root.rglob("*"), key=lambda path: len(str(path)), reverse=True):
        name = str(item.relative_to(root))
        if name in keep:
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
        except OSError:
            continue
    for name in saved["directories"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name, payload in saved["files"].items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not target.exists() or target.read_bytes() != payload:
                target.write_bytes(payload)
            # An arm reading a modification time would be reading which arm it
            # is. Nothing in the probe does today; putting the stamp back costs
            # nothing and stops that from becoming true by accident.
            os.utime(target, (0, 0))
        except OSError:
            continue


def _intentions_state(loop: Any) -> list[tuple] | None:
    """Every row the intention database holds.

    The loop keeps its open intentions in memory and its record of what came of
    them on disk, and the fork carried only the first. Efficacy, the capability
    beliefs the self model reads and the comparator's attributions are all
    computed from what is on disk, so an arm that acted taught the next arm
    what it had learned.
    """
    connection = getattr(loop, "_conn", None)
    if connection is None:
        return None
    try:
        return list(connection.execute("SELECT * FROM intentions"))
    except Exception:  # noqa: BLE001 - a database that will not read is not carried
        return None


def _restore_intentions(loop: Any, rows: list[tuple] | None) -> None:
    """Roll the intention database back to the rows the snapshot holds."""
    connection = getattr(loop, "_conn", None)
    if connection is None or rows is None:
        return
    try:
        with connection:
            connection.execute("DELETE FROM intentions")
            if rows:
                marks = ",".join("?" for _ in rows[0])
                connection.executemany(f"INSERT INTO intentions VALUES ({marks})", rows)
    except Exception:  # noqa: BLE001
        return


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
    except Exception:  # noqa: BLE001 - unreadable is different
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
    #: The scratch world, byte for byte, and the intention database's rows. The
    #: experiment forks (K, E): what she wrote in one arm is not there for the
    #: next, and what she learned from writing it is not either.
    world: dict[str, Any] | None = None
    intentions: list[tuple] | None = None
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


@dataclass
class SubjectRuntime:
    """The organism, running offline, forkable."""

    kernel: Any
    state: Any
    ontogeny: Any
    rng: random.Random
    #: The ontogeny service that owns the reservoir. Its accumulators live
    #: outside the reservoir and are carried across the fork with it.
    ontogeny_service: Any = None
    turn: int = 0
    retriever: Any = None
    failures: dict[str, int] = field(default_factory=dict)
    failure_notes: dict[str, str] = field(default_factory=dict)
    frames_per_turn: int = 0
    #: Which services and phases the fork carries, learned at bring-up by
    #: living a turn and seeing what moved. None means carry everything, which
    #: is what calibration itself runs under.
    forked_services: set[str] | None = None
    forked_phases: set[str] | None = None
    #: Called after every phase when a lesion is in force. See core.subject.clamp.
    after_phase: Any = None
    #: Host readings held constant for the duration of a paired trial. The body
    #: senses the real machine, so two arms run seconds apart read different
    #: CPU and different thermals, and that difference is the environment
    #: moving rather than the intervention propagating. Freezing it is the
    #: matched-environment control every arm of a comparison needs.
    frozen_host: dict[str, float] | None = None
    frozen_latency: dict[str, float] | None = None
    #: The clock every arm shares, or None to run on the machine's. See
    #: `core.subject.clock`: the phases that read elapsed time have to be
    #: handed the same interval in both arms or the machine's own speed is in
    #: the floor of every edge into them.
    clock: Any = None
    #: Who the next action is attributed to. "self" is the ordinary case; the
    #: ownership experiment in core.subject.agency sets it to "external" for one
    #: arm and matches everything else, so the two runs differ in authorship
    #: alone.
    actor: str = "self"
    last_action: dict[str, Any] = field(default_factory=dict)
    organs: Organs = field(default_factory=Organs)
    #: The consciousness layer's own tick. In the desktop runtime it free-runs
    #: beside the phases; here it is called once per turn so that two arms of an
    #: intervention see the same number of ticks and differ by the displacement
    #: rather than by how long each one happened to take.
    heartbeat: Any = None
    organism: Any = None
    #: How many times each free-running layer has been advanced, and any that
    #: could not be. Counted rather than timed, so both arms of a trial run the
    #: same layers the same number of times.
    layer_steps: Steps = field(default_factory=Steps)
    #: The harness's own frame count, which decides whose turn it is to step.
    #: Carried in the snapshot and rewound by a restore, because otherwise the
    #: second arm of a trial starts sixty-six frames further on and steps a
    #: different set of layers from the first.
    frame_index: int = 0
    #: The live intention loop, so the probe's action takes the path a real one
    #: takes rather than writing the outcome straight into the state.
    _intentions: Any = None

    # ── forking ──────────────────────────────────────────────────────────

    #: Which organs are carried across a fork, by the name they are read under.
    #: Every organ the reading declares, taken from the declaration rather
    #: than written out again here. The hand-kept version was missing
    #: `self_prediction` and `comparator` for a whole session: both are
    #: process-wide, both write into the self-state domain every turn, and
    #: left out of the fork the sham arm inherited the prediction error the
    #: displaced arm had just produced — four tenths of a standard deviation
    #: in the floor of every S column before a phase had run. A second list
    #: of the same organs is a second chance to forget one, so there is one.
    ORGAN_FIELDS: ClassVar[tuple[str, ...]] = tuple(
        organ.name for organ in dataclass_fields(Organs)
    )

    def freeze_host(self) -> dict[str, float]:
        """Take the body's current reading and hold it for every arm to come."""
        hardware = dict(getattr(self.state.soma, "hardware", {}) or {})
        latency = dict(getattr(self.state.soma, "latency", {}) or {})
        self.frozen_host = {
            key: float(hardware.get(key, 0.0) or 0.0)
            for key in ("cpu_usage", "vram_usage", "ram_usage", "temperature")
        }
        # Latency is elapsed wall clock, so it differs between two arms run
        # seconds apart by exactly as much as the machine was busy. Same
        # argument as the hardware readings: it is the environment, and it
        # belongs held still.
        self.frozen_latency = {
            key: float(latency.get(key, 0.0) or 0.0)
            for key in ("last_thought_ms", "perception_lag_ms", "token_velocity")
        }
        return self.frozen_host

    async def calibrate_fork(self, conditions: Sequence[Condition]) -> dict[str, Any]:
        """Find out which services and phases actually move, and carry only those.

        A hundred and ten services are built by the time the organism is up and
        almost none of them change during a turn. Capturing and restoring all
        of them costs about a second each way, and a run makes fifteen hundred
        restores. So the fork is calibrated rather than guessed: take a reading,
        live a turn in each condition, take another, and carry whatever
        differed. What never moves cannot carry a difference between two arms.
        """
        before_services = _service_state(None)
        before_phases = self._phase_state()
        for condition in conditions:
            await self.turn_once(condition)
        after_services = _service_state(None)
        after_phases = self._phase_state()

        def moved(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
            names = set(before) | set(after)
            return {name for name in names if _differs(before.get(name), after.get(name))}

        self.forked_services = moved(before_services, after_services)
        self.forked_phases = moved(before_phases, after_phases)
        return {
            "services_carried": sorted(self.forked_services),
            "services_seen": len(before_services | after_services.keys()),
            "phases_carried": sorted(self.forked_phases),
        }

    def _phase_state(self, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
        # Everything the container already holds is carried under its own name,
        # and the kernel is the machinery rather than the state. What is left
        # is what a phase kept for itself — including the module-level
        # singletons it caches, which no container knows about.
        skip = frozenset(
            {
                id(self.kernel),
                id(self),
                *(id(obj) for obj in _built_services().values()),
                # The organs are carried by name and are the largest objects in
                # the process: the substrate alone holds a half-million weights,
                # and copying it twice per fork is most of what a fork costs.
                *(
                    id(getattr(self.organs, field_name, None))
                    for field_name in self.ORGAN_FIELDS
                ),
                # And the services the fork deliberately leaves alone. Reaching
                # one of them through a phase is the same write the exclusion
                # was written to prevent, and the mycelial topology logs a
                # critical every time the guard refuses it.
                *(
                    id(obj)
                    for name, obj in _built_services().items()
                    if name in _UNFORKED_SERVICES
                ),
            }
        )
        out: dict[str, dict[str, Any]] = {}
        for phase in getattr(self.kernel, "_phases", []) or []:
            name = phase.__class__.__name__
            if only is not None and name not in only:
                continue
            try:
                captured = _organ_state(phase, skip=skip)
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
                # A phase that cannot be read is skipped, by kind.
                continue
            if captured:
                out[name] = captured
        return out

    def _restore_phases(self, saved: Mapping[str, dict[str, Any]]) -> None:
        if not saved:
            return
        for phase in getattr(self.kernel, "_phases", []) or []:
            fields = saved.get(phase.__class__.__name__)
            if fields:
                _restore_organ(phase, fields)

    def _refresh_health(self) -> None:
        """Run the kernel's own end-of-tick projection over the finished state."""
        refresh = getattr(self.state, "_refresh_cognitive_health", None)
        if not callable(refresh):
            return
        try:
            refresh()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("cognitive health projection failed: %s", exc)

    def _republish_body(self) -> None:
        """Tell the engine that judges the body what the body was just held at.

        The proprioceptive loop reads the machine and reports it to the
        resilience engine mid-phase, and the hold is applied after the phase.
        Without this the engine — and through it homeostasis, and through that
        her will to live — kept reading the real machine while the state was
        held at the displaced value, which is the two-bodies problem again with
        the seam moved. One body: the held reading is the reading.
        """
        engine = getattr(self.organs, "soma", None)
        report = getattr(engine, "observe_host", None)
        if not callable(report) or self.frozen_host is None:
            return
        try:
            report(
                cpu_percent=self.frozen_host.get("cpu_usage", 0.0),
                ram_percent=self.frozen_host.get(
                    "ram_usage", self.frozen_host.get("vram_usage", 0.0)
                ),
                temperature_c=self.frozen_host.get("temperature"),
            )
        except (AttributeError, TypeError, ValueError):
            return

    def thaw_host(self) -> None:
        self.frozen_host = None
        self.frozen_latency = None

    def snapshot(self) -> Snapshot:
        return Snapshot(
            organs={
                name: _organ_state(getattr(self.organs, name, None))
                for name in self.ORGAN_FIELDS
            },
            state=copy.deepcopy(self.state),
            hidden=np.array(self.ontogeny.h, copy=True),
            steps=int(self.ontogeny.steps),
            era=int(self.ontogeny.era),
            centre=np.array(self.ontogeny._centre, copy=True),
            scatter=np.array(self.ontogeny._scatter, copy=True),
            centre_n=float(self.ontogeny._centre_n),
            turn=self.turn,
            rng_state=self.rng.getstate(),
            moments=_moments_of(self.ontogeny_service),
            last_reading=getattr(self.ontogeny_service, "_last_reading", None),
            last_novelty=float(getattr(self.ontogeny, "last_novelty", 0.5)),
            last_displacement=float(getattr(self.ontogeny, "last_displacement", 0.0)),
            lifetime_last=_lifetime_last(),
            phases=self._phase_state(self.forked_phases),
            services=_service_state(self.forked_services),
            effort=_effort_state(),
            taken_at=time.time(),
            clock_at=None if self.clock is None else self.clock.now(),
            frame_index=self.frame_index,
            global_random=random.getstate(),
            numpy_random=np.random.get_state(),
            torch_random=_torch_random_state(),
            world=_world_state(getattr(self, "_scratch", None)),
            intentions=_intentions_state(self._intentions),
        )

    def restore(self, snapshot: Snapshot) -> None:
        for name, saved in snapshot.organs.items():
            _restore_organ(getattr(self.organs, name, None), saved)
        self.state = copy.deepcopy(snapshot.state)
        self.ontogeny.h = np.array(snapshot.hidden, copy=True)
        self.ontogeny.steps = snapshot.steps
        self.ontogeny.era = snapshot.era
        self.ontogeny._centre = np.array(snapshot.centre, copy=True)
        self.ontogeny._scatter = np.array(snapshot.scatter, copy=True)
        self.ontogeny._centre_n = snapshot.centre_n
        _restore_moments(self.ontogeny_service, snapshot.moments)
        self.ontogeny.last_novelty = snapshot.last_novelty
        self.ontogeny.last_displacement = snapshot.last_displacement
        _restore_lifetime_last(snapshot.lifetime_last)
        service = self.ontogeny_service
        if service is not None:
            service._last_reading = snapshot.last_reading
        self.turn = snapshot.turn
        self.rng.setstate(snapshot.rng_state)
        if snapshot.global_random is not None:
            random.setstate(snapshot.global_random)
        if snapshot.numpy_random is not None:
            np.random.set_state(snapshot.numpy_random)
        _restore_torch_random(snapshot.torch_random)
        self._restore_phases(snapshot.phases)
        _restore_services(snapshot.services)
        _restore_effort(snapshot.effort)
        self.frame_index = snapshot.frame_index
        _restore_world(getattr(self, "_scratch", None), snapshot.world)
        _restore_intentions(self._intentions, snapshot.intentions)
        if self.clock is not None and snapshot.clock_at is not None:
            # The clock is the state as far as a phase reading elapsed time is
            # concerned, so it rewinds with everything else and the shift below
            # is zero. The shift stays because a run without the clock still
            # has to rewind, and because it covers what the clock cannot: a
            # module that bound `time.time` before the install.
            self.clock.set(snapshot.clock_at)
        if snapshot.taken_at:
            _reanchor(self, time.time() - snapshot.taken_at)

    # ── reading ──────────────────────────────────────────────────────────

    def read(self, condition: str, tag: str, env: dict[str, float]) -> CoreState:
        return read_core_state(
            self.state,
            ontogeny=self.ontogeny,
            organs=self.organs,
            condition=condition,
            tag=tag,
            env=env,
        )

    # ── the turn ─────────────────────────────────────────────────────────

    async def turn_once(
        self,
        condition: Condition,
        *,
        on_frame: Callable[[CoreState], None] | None = None,
        perturb_at: int | None = None,
        perturb: Callable[[SubjectRuntime], None] | None = None,
    ) -> list[CoreState]:
        """Run every phase once over the carried state, reading K after each.

        ``perturb_at`` is a frame index; the displacement is applied after that
        frame is read, so the arms share every reading before it and differ
        only from the next one on.
        """
        engine = self.kernel.organs.get("llm") if hasattr(self.kernel, "organs") else None
        mind = getattr(engine, "instance", None) if engine is not None else None
        if mind is not None and hasattr(mind, "moment"):
            mind.moment = self.turn
        env = {"turn": float(self.turn), "condition_id": float(_condition_index(condition.name))}
        if condition.prepare is not None:
            env.update(condition.prepare(self.state, self.rng))
        self.state.cognition.current_objective = condition.objective or None
        self.state.cognition.current_origin = condition.origin
        env["objective_len"] = float(len(condition.objective))

        frames: list[CoreState] = []

        async def capture(tag: str) -> None:
            if self.clock is not None:
                self.clock.advance()
            # The free-running layers, advanced by the harness's own count
            # rather than by the machine's schedule. Both arms of a trial run
            # them the same number of times, which is the only way a layer that
            # runs on a timer can be inside a paired measurement at all — the
            # alternative was stopping them, and then whatever they contribute
            # to the coupling was absent from every number.
            self.layer_steps = await step_once(
                self.organism, self.frame_index, self.layer_steps
            )
            await self._integrate_substrate(self.frame_index)
            self.frame_index += 1
            reading = self.read(condition.name, tag, env)
            frames.append(reading)
            if on_frame is not None:
                on_frame(reading)
            if perturb_at is not None and perturb is not None and len(frames) - 1 == perturb_at:
                outcome = perturb(self)
                if inspect.isawaitable(outcome):
                    await outcome

        await capture("open")
        for phase in self.kernel._phases:
            name = phase.__class__.__name__
            try:
                result = await asyncio.wait_for(
                    phase.execute(self.state, objective=condition.objective),
                    timeout=PHASE_TIMEOUT,
                )
                if result is not None:
                    self.state = result
            except BaseException as exc:  # noqa: BLE001 - a phase that dies is a reading
                self.failures[name] = self.failures.get(name, 0) + 1
                self.failure_notes[name] = f"{type(exc).__name__}: {exc}"[:200]
                logger.debug("phase %s failed: %s", name, exc)
            if self.frozen_host is not None:
                self.state.soma.hardware.update(self.frozen_host)
                if self.frozen_latency is not None:
                    self.state.soma.latency.update(self.frozen_latency)
                self._republish_body()
            if self.after_phase is not None:
                self.after_phase()
            await capture(name)

        # The projection the real tick makes once the phases are done. The
        # kernel refreshes it at the end of `tick`, outside the phase loop, and
        # this driver runs the phases directly — so coherence, fragmentation,
        # the contradiction count and the whole cognitive-health block were
        # never computed during a run. Four of the workspace domain's columns
        # were constants because of it, and the deliberation phase's reading of
        # how badly the moment was going was reading two of them.
        self._refresh_health()

        if condition.after == "retrieve":
            self._retrieve(condition.objective)
        elif condition.after == "act" or self._intends_to_act():
            # Off the loop. The action is a real write and a real read-back, so
            # it carries an fsync, and an fsync on the event loop is the defect
            # this driver exists to measure rather than to commit.
            #
            # She acts when she has decided to, not only when the condition
            # says so. A return route to perception that exists in one of eight
            # conditions cannot replicate in three, and the specification asks
            # for the loop to close through the world — deliberation, action,
            # environment, perception. Which action is chosen and whether one
            # happens at all are both read off her own intentions.
            await asyncio.to_thread(self._act, condition.objective, actor=self.actor)
        await capture("after")

        await self._consciousness_tick()
        await capture("heartbeat")

        self._step_ontogeny(frames[-1])
        await capture("ontogeny")

        self.turn += 1
        self.frames_per_turn = len(frames)
        return frames

    # ── the real subsystems the conditions reach for ─────────────────────

    async def _consciousness_tick(self) -> None:
        """One beat of the consciousness layer, and one substrate step.

        These run continuously in the desktop runtime — the heartbeat on its
        own interval, the liquid substrate at twenty hertz. Free-running them
        here would put uncontrolled noise between the two arms of every
        intervention and make the sham floor larger than any effect. Calling
        each once per turn keeps the computation and drops the jitter.
        """
        substrate = self.organs.substrate
        if substrate is not None:
            try:
                await asyncio.wait_for(
                    substrate.update(source="subject_core_turn"), timeout=PHASE_TIMEOUT
                )
            except BaseException as exc:  # noqa: BLE001
                self.failures["substrate"] = self.failures.get("substrate", 0) + 1
                self.failure_notes["substrate"] = f"{type(exc).__name__}: {exc}"[:200]
        if self.heartbeat is not None:
            try:
                await asyncio.wait_for(self.heartbeat._tick(), timeout=PHASE_TIMEOUT)
            except BaseException as exc:  # noqa: BLE001
                self.failures["heartbeat"] = self.failures.get("heartbeat", 0) + 1
                self.failure_notes["heartbeat"] = f"{type(exc).__name__}: {exc}"[:200]
        self._train_world_model()


    async def _integrate_substrate(self, frame: int) -> None:
        """The substrate's own iterations for this frame, at its own step size.

        One Euler step of half a second is not thirteen of a tenth. The
        substrate is a nonlinear stochastic recurrent system: the tanh is
        evaluated at different intermediate states, the noise draws are
        independent, and the clip can bite in the middle — so a single large
        step is a different trajectory rather than a coarse version of the same
        one. Here it takes the iterations its own configured rate calls for in
        one frame, each at its own configured integration constant, which is
        what its loop does.

        Scheduled off the frame index like every other layer, so nothing is
        carried across the fork and two arms integrate identically.
        """
        substrate = self.organs.substrate
        if substrate is None:
            return
        config = getattr(substrate, "config", None)
        rate = float(getattr(config, "update_rate", 20.0) or 20.0)
        dt = float(getattr(config, "time_constant", 0.1) or 0.1)
        step = getattr(substrate, "_step_dynamics", None)
        if not callable(step):
            return
        from core.subject.steppable import Layer, frame_seconds, iterations_at

        _, count = iterations_at(
            Layer("substrate", "", "", rate, ()), frame, frame_seconds()
        )
        for _ in range(count):
            try:
                await asyncio.wait_for(step(dt), timeout=PHASE_TIMEOUT)
            except BaseException as exc:  # noqa: BLE001
                self.failures["substrate"] = self.failures.get("substrate", 0) + 1
                self.failure_notes["substrate"] = f"{type(exc).__name__}: {exc}"[:200]
                return

    def _train_world_model(self) -> None:
        """The gradient steps the training lane would have taken, on this clock.

        The lane is a thread on a two-second timer, and the wind-down stops it —
        but the unified model starts it again the first time anything asks for
        the learned facet, which happens inside a turn. So it was running
        during the arms, taking however many steps the machine allowed, and the
        model's weights, hidden norm and last surprise all differed between two
        arms that had seen the same data. Three of the world model's columns,
        and part of the body's exertion, were a fact about thread scheduling.

        The passes and the pending-work gate are the lane's own, so what the
        model learns is unchanged. Only the clock it learns on is the
        experiment's rather than the host's.
        """
        model = getattr(self.organs, "world_model", None)
        learned = getattr(model, "learned", None) if model is not None else None
        if learned is None:
            return
        halt = getattr(learned, "stop_training", None)
        if callable(halt) and getattr(learned, "_trainer_thread", None) is not None:
            try:
                halt()
            except (RuntimeError, OSError) as exc:
                logger.debug("could not stop the training lane: %s", exc)
        try:
            if int(getattr(learned, "_pending_since_train", 0)) <= 0:
                return
            learned._pending_since_train = 0
            from core.world_model.learned_world_model import (
                _BPTT_WINDOW,
                _TRAIN_PASSES_PER_CYCLE,
            )

            for _ in range(_TRAIN_PASSES_PER_CYCLE):
                if len(getattr(learned, "_replay", ())) < _BPTT_WINDOW:
                    break
                learned._mini_batch_update()
        except (AttributeError, ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
            self.failures["world_model_training"] = (
                self.failures.get("world_model_training", 0) + 1
            )
            logger.debug("deterministic world-model training failed: %s", exc)

    def _step_ontogeny(self, reading: CoreState) -> None:
        """Carry the last lifetime reading onto the reservoir object N reads.

        The step itself happens inside the affect phase now, through
        `core.ontogeny.lifetime.advance`, which is the runtime's own path. This
        only copies what that step sensed onto the state object so that the N
        domain can read novelty and displacement beside the hidden units. If
        the phase did not advance — the organ was absent, or it degraded — the
        previous reading stands and N shows a flat step, which is what actually
        happened.
        """
        del reading
        try:
            from core.ontogeny.lifetime import last_reading

            step = last_reading()
        except ImportError:
            step = None
        if step is None or self.ontogeny is None:
            return
        self.ontogeny.last_novelty = float(step.novelty)
        self.ontogeny.last_displacement = float(step.displacement)

    def _retrieve(self, query: str) -> None:
        """Run the real retriever over her own memory, ontogeny included.

        The breadth of this retrieval is chosen by the ontogenetic organ in the
        live runtime, through `IntentionalRetriever._choose_breadth`. Running
        the real object is what makes N -> M a measurable edge rather than an
        assumed one.
        """
        if self.retriever is None:
            return
        try:
            from core.memory.intentional_retrieval import RetrievalIntent

            result = self.retriever.retrieve(
                RetrievalIntent(task=query or "recent context", kind="general", query=query or "recent context", limit=8)
            )
            self.state.cognition.long_term_memory = [
                str(getattr(hit, "content", hit))[:240] for hit in result.hits
            ][:8]
        except Exception as exc:  # noqa: BLE001 - a retriever that dies is a reading
            self.failures["retrieve"] = self.failures.get("retrieve", 0) + 1
            logger.debug("retrieval failed: %s", exc)

    #: What she expects of each thing she can do. The intention loop compares
    #: this against what happened, and the agency ledger keys its capability
    #: beliefs on the name — so a hardcoded `write_notes` made every action she
    #: ever took the same capability, and left her expecting a file to hold a
    #: plan on the turns she was looking in a room.
    ACTION_EXPECTATIONS: ClassVar[dict[str, str]] = {
        "write_notes": "the file holds the plan",
        "append_log": "the line is on the end of the log",
        "make_room": "the room is there",
        "read_room": "the room from last turn is there to look in",
    }

    def _through_the_intention_loop(
        self, intended: str, ok: bool, actor: str, kind: str = "write_notes"
    ) -> None:
        """Say, do, observe — the live agency path, for the probe's own action."""
        if actor != "self":
            # An intention is hers by construction. Recording an outside actor's
            # outcome as one of her intentions is the confusion the ownership
            # experiment exists to detect.
            return
        try:
            loop = self._intentions
            if loop is None:
                return
            expected = self.ACTION_EXPECTATIONS.get(kind, "the action lands")
            identifier = loop.intend(
                intention=intended, drive="creation", expected_outcome=expected
            )
            loop.record_action(
                identifier,
                tool_name=kind,
                args={},
                result="ok" if ok else "failed",
                success=ok,
                duration_ms=1.0,
            )
            outcome = expected if ok else f"{expected} — it did not"
            loop.observe(identifier, observation=expected, actual_outcome=outcome)
        except Exception as exc:  # noqa: BLE001 - the probe's action still stands
            logger.debug("intention loop unavailable: %s", exc)

    #: What she can do to the world, in the order the drives are read. Each is
    #: a real filesystem change with a real reading back, and which one happens
    #: is decided by her state rather than fixed by this harness.
    ACTIONS: ClassVar[tuple[str, ...]] = (
        "write_notes",
        "append_log",
        "make_room",
        "read_room",
    )

    #: Which action each drive reaches for. Written out rather than hashed:
    #: `hash()` on a string is salted per process, so the drive that picked
    #: `make_room` in one run picked `read_room` in the next and two runs of the
    #: battery were not comparable in what she actually did. And a mapping by
    #: what the need is for is a reading, where a hash is a coin.
    DRIVE_ACTIONS: ClassVar[dict[str, str]] = {
        "growth": "make_room",
        "curiosity": "read_room",
        "social": "write_notes",
        "integrity": "append_log",
        "energy": "append_log",
    }

    #: And what she does when she is attending to something. The broadcast
    #: winner decides before the drives do, because that is the claim global
    #: workspace theory makes — what wins the competition reaches the
    #: specialised process, and acting is one. It also varies from turn to
    #: turn, where the drives move over days: an action chosen by a standing
    #: budget is the same action every time, and an outcome that is the same
    #: every time cannot teach her anything about what she can do.
    ATTENTION_ACTIONS: ClassVar[dict[str, str]] = {
        "perception": "read_room",
        "world_model": "read_room",
        "ontogeny": "read_room",
        "memory": "append_log",
        "metacognition": "append_log",
        "interoception": "append_log",
        "exchange": "write_notes",
        "self": "write_notes",
        "deliberation": "make_room",
        "substrate": "make_room",
    }

    def _intends_to_act(self) -> bool:
        """Whether anything she is holding is urgent enough to do something about.

        The bar is the will's own: an initiative below three tenths is deferred
        by governance, so one at or above it is one she has decided on.
        """
        cognition = getattr(self.state, "cognition", None)
        if cognition is None:
            return False
        holdings = list(getattr(cognition, "pending_initiatives", []) or [])
        holdings += list(getattr(cognition, "active_goals", []) or [])
        for item in holdings:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) in {"done", "failed"}:
                continue
            for key in ("urgency", "priority"):
                value = item.get(key)
                if isinstance(value, (int, float)) and float(value) >= WILL_DEFERRAL_BAR:
                    return True
        return False

    def _chosen_action(self) -> str:
        """What she does, decided by what she is attending to.

        Not a constant and not a random draw: the budgets are part of the
        deliberation domain, so displacing that domain changes what she does,
        which changes what the filesystem holds, which changes what her senses
        report back. That is the whole of the return route through the world.
        """
        attending = str(getattr(self.state.cognition, "attention_focus", "") or "")
        source = attending.split(":", 1)[0].strip()
        if source.startswith("affect_"):
            source = "self"
        chosen = self.ATTENTION_ACTIONS.get(source)
        if chosen:
            return chosen
        budgets = getattr(getattr(self.state, "motivation", None), "budgets", {}) or {}
        levels = []
        # Every budget the state carries, read from the state. This listed
        # `creation` and `rest`, which no budget has ever been called, and
        # omitted `growth` and `integrity`, which are two of the five that
        # exist — so the drive that is most depleted on almost every tick was
        # not among the ones considered.
        for name in sorted(budgets):
            entry = budgets.get(name)
            if isinstance(entry, dict):
                levels.append((float(entry.get("level", 100.0) or 0.0), name))
        if not levels:
            return self.ACTIONS[0]
        levels.sort()
        return self.DRIVE_ACTIONS.get(levels[0][1], self.ACTIONS[0])

    def _act(self, objective: str, *, actor: str = "self") -> None:
        """The action arm of the self/world loop, and its consequence.

        The action is a real write to a scratch file inside the run directory
        and the outcome is read back from the filesystem rather than asserted.
        Both the intention and what came of it land in the state.

        ``actor`` is the whole of the ownership experiment. The file ends with
        the same bytes either way, the same fact is recorded, the same goal is
        appended with the same text and the same verified outcome. The two runs
        differ in who is named as having done it, and nothing else. If the
        self-model reads that difference, S diverges; if it does not, the
        divergence is zero and the loop is open at exactly that point.
        """
        target = getattr(self, "_scratch", None)
        if target is None:
            return
        # Which action, chosen from her own state rather than fixed here. The
        # completion specification asks for the reafference loop to be more
        # than one scratch-file pathway, and for the return to be genuinely
        # environmental: the drive that is most depleted picks what she does,
        # the filesystem is what changes, and what comes back is read off the
        # filesystem rather than asserted. Displacing deliberation therefore
        # changes what the world holds, and the world is what she then sees.
        kind = self._chosen_action()
        intended = f"{kind} for turn {self.turn}: {objective[:80]}"
        ok = False
        observed = ""
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            room = Path(target)
            with local_internal_governed_scope("subject_core.action_probe"):
                if kind == "append_log":
                    path = room / "actions.log"
                    prior = path.read_text() if path.exists() else ""
                    gateway.write_text(
                        path, prior + intended + "\n", source="subject_core.action_probe"
                    )
                    lines = path.read_text().splitlines()
                    ok = bool(lines) and lines[-1] == intended
                    observed = f"the log holds {len(lines)} lines"
                elif kind == "make_room":
                    path = room / f"room_{self.turn:04d}"
                    gateway.ensure_directory(path, source="subject_core.action_probe")
                    rooms = sorted(p.name for p in room.glob("room_*") if p.is_dir())
                    ok = path.is_dir()
                    observed = f"{len(rooms)} rooms exist"
                elif kind == "read_room":
                    # The one that can fail, and fails for a reason that is
                    # hers: the room for a turn exists only if she chose to
                    # make one then. An action repertoire in which nothing can
                    # fail cannot teach efficacy — every attempt succeeded, so
                    # the ledger's efficacy and authored share sat at one for
                    # the whole of every run and the self-state read two
                    # constants where two of its liveliest columns should be.
                    path = room / f"room_{max(0, self.turn - 1):04d}"
                    ok = path.is_dir()
                    observed = (
                        f"room {path.name} holds {len(list(path.iterdir()))} things"
                        if ok
                        else f"there is no room {path.name}"
                    )
                else:
                    path = room / "notes.txt"
                    gateway.write_text(
                        path, intended, source="subject_core.action_probe"
                    )
                    ok = path.read_text() == intended
                    observed = f"notes.txt holds {len(intended)} characters"
        except OSError as exc:
            logger.debug("probe action failed: %s", exc)
        # Through the intention loop, which is the path the live runtime takes:
        # form the intention, record the action, observe the outcome. That is
        # what emits the efference copy and compares it, so the agency
        # comparator sees the action rather than sitting at its defaults — its
        # four readings were constant for want of a caller, not for want of
        # anything to say.
        self._through_the_intention_loop(intended, ok, actor, kind)
        record = {
            "intended": intended,
            "verified": ok,
            "at": time.time(),
            "actor": actor,
        }
        self.state.world.facts["last_action"] = record
        self.last_action = dict(record)
        # The same world state, attributed. This is the one call that separates
        # the two arms of the ownership experiment.
        #
        # Only for an outside actor. Her own action already reached the ledger
        # through the intention loop above, and recording it here as well
        # counted every action twice — which halved the efficacy the self model
        # was being handed.
        if actor != "self":
            try:
                from core.agency.authorship import Event, get_agency_ledger

                get_agency_ledger().observe(
                    Event(what="write_notes", actor=actor, verified=ok, detail={"path": "notes.txt"}),
                    self_model=self.organs.self_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("authorship ledger unavailable: %s", exc)
        self.state.cognition.active_goals.append(
            {
                "id": f"act_{self.turn}",
                "goal": intended,
                "description": intended,
                "origin": actor,
                "status": "done" if ok else "failed",
                # Priced, because a goal the workspace cannot price is a goal
                # attention can never reach — but this one is finished, and a
                # finished goal is a record rather than an intention. The
                # workspace drops it on status; the number is here for whatever
                # reads the record.
                "priority": 1.0 if ok else 0.5,
            }
        )
        self._last_observed = observed
        self.state.cognition.last_action_source = actor

        # And she sees what she did. The loop the specification asks for closes
        # through the world — action, environment, perception — and without
        # this the probe wrote a file and nothing ever came back through the
        # senses, so nothing she did could reach perception by any route.
        _percept(
            self.state,
            self.rng,
            "goal_achieved" if ok else "error",
            "filesystem",
            observed or f"{kind} {'took' if ok else 'did not take'}",
        )
        if len(self.state.cognition.active_goals) > 12:
            del self.state.cognition.active_goals[:-12]


def _condition_index(name: str) -> int:
    for index, condition in enumerate(CONDITIONS):
        if condition.name == name:
            return index
    return -1


def build_runtime(workdir: Path, *, seed: int = 0, mind: Any = None) -> SubjectRuntime:
    """Assemble the offline organism. Import cost lives here, not at module load."""
    os.environ.setdefault("AURA_TESTING", "1")
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("subject_core.driver"):
        get_file_write_gateway().ensure_directory(workdir, source="subject_core.driver")
    os.environ.setdefault("AURA_LOG_DIR", str(workdir / "logs"))

    import threading
    from types import SimpleNamespace

    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.ontogeny.state import OntogeneticState

    # The container has to exist before the kernel is built; the rest of the
    # organism comes up in `start_organism`, which is async.
    from core.service_registration import register_all_services
    from core.state.aura_state import AuraState
    from core.state.state_repository import StateRepository
    from core.subject.state import domain_width

    try:
        register_all_services()
    except Exception as exc:  # noqa: BLE001
        logger.warning("service registration failed: %s", exc)

    vault = StateRepository(db_path=str(workdir / "subject.db"), is_vault_owner=True)
    kernel = AuraKernel(config=KernelConfig(), vault=vault)
    kernel._setup_phases()
    kernel._initialize_organs()
    # The organ shape matters: phases test `organ.ready.is_set()` and read
    # `organ.instance`, and a stub missing either makes the phase raise, which
    # would be recorded as a phase that does nothing rather than as a hole in
    # the harness.
    engine = mind or DeterministicMind()
    ready = threading.Event()
    ready.set()
    kernel.organs["llm"] = SimpleNamespace(
        get_instance=lambda: engine, instance=engine, ready=ready, name="llm"
    )
    # And under the name the phases actually ask for first. Registered only as
    # the organ, every phase that prefers `llm_router` reached the real router,
    # which has no model offline.
    for name in ("llm_router", "local_llm"):
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance(name, engine)
        except Exception as exc:  # noqa: BLE001 - a container that refuses is a datum
            logger.warning("could not register the deterministic mind as %s: %s", name, exc)

    width = sum(domain_width(key) for key in FAST_DOMAINS)
    ontogeny = OntogeneticState(
        input_width=width, units=64, seed=seed, path=workdir / "ontogeny.npz"
    )
    ontogeny.last_novelty = 0.5
    ontogeny.last_displacement = 0.0

    # The service's reservoir is a different object from the one N reads, and
    # the affect phase steps that one every turn through
    # `core.ontogeny.lifetime.advance`. Its accumulators are process-wide, so
    # they have to be carried across the fork with everything else.
    try:
        from core.ontogeny.service import get_ontogeny

        ontogeny_service = get_ontogeny()
    except Exception:  # noqa: BLE001 - an absent organ is an absent organ
        ontogeny_service = None

    runtime = SubjectRuntime(
        kernel=kernel,
        state=AuraState.default(),
        ontogeny=ontogeny,
        rng=random.Random(seed),
        ontogeny_service=ontogeny_service,
    )
    runtime.organs = Organs.live()
    # `replace`, never a fresh `Organs(...)` listing the fields by hand. A
    # hand-written rebuild silently drops whatever was added to the dataclass
    # after it was written: this one lost the self-prediction loop and the
    # efference comparator, so eleven of the self-state's columns read zero for
    # the whole of a session and every criterion that depended on them failed
    # on an organ that was there.
    runtime.organs = replace(runtime.organs, ontogeny=ontogeny)
    runtime._scratch = workdir / "scratch"
    with local_internal_governed_scope("subject_core.driver"):
        get_file_write_gateway().ensure_directory(
            runtime._scratch, source="subject_core.driver"
        )
    runtime.retriever = _build_retriever(runtime)
    try:
        from core.agency.intention_loop import IntentionLoop

        runtime._intentions = IntentionLoop(db_path=str(workdir / "intentions.db"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("intention loop unavailable: %s", exc)
    return runtime


#: What one turn of the organism is worth, in seconds of its own life.
#:
#: The free-running layers declare their rates in hertz — the mesh at ten, the
#: field at twenty, the oscillators at a hundred — so a counted schedule needs
#: to know how much life a frame is worth before it can run them at those
#: rates. The first version measured it: it timed the machine's frames and made
#: the experiment's second as long as the host happened to take. That puts the
#: host back into the timeline it was installed to remove, and it means two
#: machines running the same commit give the organism different amounts of life
#: per turn — which is exactly the comparison a second-machine replication is
#: for.
#:
#: So it is fixed, and it is the simplest statement that can be made: a turn is
#: one second. A turn is the organism's unit of experience, the layer rates are
#: per second, and at one second a turn the bridge integrates ten times per
#: turn, which is its own declared cadence. The frame follows from it and from
#: how many readings a turn takes, which is a property of the phase list rather
#: than of the machine.
SECONDS_PER_TURN: float = 1.0


async def calibrate_clock(
    runtime: SubjectRuntime, conditions: Sequence[Condition], *, turns: int = 3
) -> dict[str, float]:
    """Count this organism's frames per turn, then put the run on its own clock.

    The count is a property of the phase list. The machine's real pace is timed
    beside it, on `time.monotonic` — which this never replaces, so asyncio's
    timeouts still fire — and reported rather than used, because how fast the
    host runs is not how much life a turn is worth.
    """
    from core.subject.clock import ExperimentClock

    frames = 0
    started = time.monotonic()
    for index in range(max(1, turns)):
        condition = conditions[index % len(conditions)]
        frames += len(await runtime.turn_once(condition))
    elapsed = max(1e-6, time.monotonic() - started)
    per_turn = max(1, round(frames / max(1, turns)))
    step = SECONDS_PER_TURN / per_turn
    clock = ExperimentClock(step)
    clock.install()
    runtime.clock = clock
    reading = {
        "seconds_per_turn": SECONDS_PER_TURN,
        "frames_per_turn": per_turn,
        "step": round(step, 6),
        "real_seconds_per_frame": round(elapsed / max(1, frames), 5),
    }
    logger.info(
        "subject-core: experiment clock at %.4fs a frame, %d frames a turn "
        "(the machine took %.4fs a frame)",
        step,
        per_turn,
        reading["real_seconds_per_frame"],
    )
    return reading


async def quiesce_organism(runtime: SubjectRuntime) -> list[str]:
    """Stop the free-running loops before the paired arms begin."""
    from core.subject.organism import _live_tasks, quiesce

    stopped = await quiesce()
    if runtime.organism is not None:
        runtime.organism.stopped_loops = stopped
        # And re-read what is left. The summary taken at bring-up listed
        # everything that was running then, which is not what is running now,
        # and a report that says twelve loops are live after nine were stopped
        # is a report nobody can act on.
        runtime.organism.still_running = _live_tasks()
    return stopped


async def start_organism(runtime: SubjectRuntime, *, quiet: bool = False) -> dict[str, Any]:
    """Bring the layers up, then bind the runtime to what came up.

    Separate from `build_runtime` because it is async and because a caller who
    wants only the phase pipeline — a unit test, say — should not be made to
    boot the consciousness stack to get one.
    """
    from core.subject.organism import bring_up

    organism = await bring_up(quiet=quiet)
    runtime.heartbeat = organism.heartbeat
    # N reads the organ's shared lifetime reservoir, not a private one. A
    # private reservoir would be a second life running beside the real one and
    # would show the driver's own arithmetic as a developmental state.
    try:
        from core.ontogeny.lifetime import state as lifetime_state

        shared = lifetime_state()
        if shared is not None:
            if not hasattr(shared, "last_novelty"):
                shared.last_novelty = 0.5
            if not hasattr(shared, "last_displacement"):
                shared.last_displacement = 0.0
            runtime.ontogeny = shared
    except Exception as exc:  # noqa: BLE001
        logger.warning("lifetime reservoir unavailable; N stays on the local one: %s", exc)

    live = Organs.live()
    runtime.organs = replace(
        live,
        substrate=organism.substrate or live.substrate,
        ontogeny=runtime.ontogeny,
    )
    runtime.organism = organism
    summary = organism.summary()
    # And learn what the fork has to carry. A hundred and ten services are
    # built by now and almost none of them move during a turn; carrying all of
    # them costs a second each way against fifteen hundred restores in a run.
    try:
        summary["fork"] = await runtime.calibrate_fork(CONDITIONS)
    except Exception as exc:  # noqa: BLE001 - an uncalibrated fork carries everything
        logger.warning("fork calibration failed; carrying every service: %s", exc)
        runtime.forked_services = None
        runtime.forked_phases = None
    return summary


def _build_retriever(runtime: SubjectRuntime) -> Any:
    """The real retriever, over adapters that read her own state.

    The stores are her working memory, her retained long-term list and her
    world facts, which is where the live stores get their contents from. What
    is real here is the planning, the ontogenetic breadth decision and the
    merge; what is scaffolding is the adapters.
    """
    try:
        from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType
    except ImportError:  # pragma: no cover - the retriever is optional to the rest
        return None

    retriever = IntentionalRetriever()

    def working(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cognition.working_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.5} for row in rows]

    def semantic(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cold.long_term_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.4} for row in rows]

    def facts(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = list(runtime.state.world.facts.items())[-limit:]
        return [{"content": f"{k}={v}"[:240], "score": 0.6} for k, v in rows]

    retriever.register_store(MemoryStoreType.EPISODIC, working)
    retriever.register_store(MemoryStoreType.SEMANTIC, semantic)
    retriever.register_store(MemoryStoreType.RECEIPT, facts)
    return retriever
