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
import inspect
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

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
SUBSTRATE_STEP_SECONDS: float = 0.5

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
        return self.REPLY

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


def _restore_organ(organ: Any, saved: Mapping[str, Any]) -> None:
    if organ is None:
        return
    for name, value in saved.items():
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _NESTED:
            _restore_organ(getattr(organ, name, None), value[1])
            continue
        try:
            setattr(organ, name, _place(value))
        except Exception:  # noqa: BLE001 - a field that will not be written stays
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
)

#: Anything smaller than this is not a wall-clock instant. Epoch seconds passed
#: a billion in 2001; a duration, a count or a rate never reaches it.
_EPOCH_FLOOR: float = 1e9


def _shift_anchors(obj: Any, shift: float) -> None:
    if obj is None:
        return
    for name in _CLOCK_ANCHORS:
        value = getattr(obj, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > _EPOCH_FLOOR:
            try:
                setattr(obj, name, float(value) + shift)
            except (AttributeError, TypeError, ValueError):
                continue


def _reanchor(runtime: SubjectRuntime, shift: float) -> None:
    """Move every wall-clock anchor forward by the time the restore skipped."""
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
    for name in runtime.ORGAN_FIELDS:
        _shift_anchors(getattr(runtime.organs, name, None), shift)
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
#: The services the fork considers at all. Everything the schema reads through
#: an organ is already carried by name; what is left worth carrying is what
#: those organs consult and what writes the state fields the schema reads. A
#: hundred and ten services are built by the time the organism is up and almost
#: all of them move during a turn, so "carry what moved" is no restriction —
#: carrying them all costs about a second and a half each way against fifteen
#: hundred restores in a run, which is half an hour of measuring nothing.
_CANDIDATE_SERVICES: set[str] = {
    "affect_grounding",
    "drive_engine",
    "executive_closure",
    "goal_engine",
    "homeostasis",
    "homeostatic_coupling",
    "inhibition_manager",
    "intention_loop",
    "memory_facade",
    "metacognition",
    "mind_model",
    "motivation_engine",
    "nociception",
    "predictive_engine",
    "soma_subsystem",
    "temporal_binding",
}

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
        except Exception:  # noqa: BLE001 - a service that cannot be read is skipped
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
    moments: dict[str, Any] | None = None
    last_reading: Any = None
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
    #: The live intention loop, so the probe's action takes the path a real one
    #: takes rather than writing the outcome straight into the state.
    _intentions: Any = None

    # ── forking ──────────────────────────────────────────────────────────

    #: Which organs are carried across a fork, by the name they are read under.
    ORGAN_FIELDS: ClassVar[tuple[str, ...]] = (
        "workspace",
        "substrate",
        "free_energy",
        "self_model",
        "world_model",
        "agency",
        "soma",
        # Both of these are process-wide and both write into the self-state
        # domain every turn. Left out of the fork, the sham arm inherited the
        # prediction error the displaced arm had just produced, which put four
        # tenths of a standard deviation into the floor of every S column
        # before a single phase had run.
        "self_prediction",
        "comparator",
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
        before_services = _service_state(_CANDIDATE_SERVICES)
        before_phases = self._phase_state()
        for condition in conditions:
            await self.turn_once(condition)
        after_services = _service_state(_CANDIDATE_SERVICES)
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
            except Exception:  # noqa: BLE001 - a phase that cannot be read is skipped
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
            phases=self._phase_state(self.forked_phases),
            services=_service_state(self.forked_services),
            effort=_effort_state(),
            taken_at=time.time(),
            global_random=random.getstate(),
            numpy_random=np.random.get_state(),
            torch_random=_torch_random_state(),
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
        env = {"turn": float(self.turn), "condition_id": float(_condition_index(condition.name))}
        if condition.prepare is not None:
            env.update(condition.prepare(self.state, self.rng))
        self.state.cognition.current_objective = condition.objective or None
        self.state.cognition.current_origin = condition.origin
        env["objective_len"] = float(len(condition.objective))

        frames: list[CoreState] = []

        async def capture(tag: str) -> None:
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
                # And integrate it. The dynamics step is what the free-running
                # loop does twenty times a second, and it is the only thing
                # that marks the state snapshot fresh — with the loop stopped,
                # every consumer that checks staleness sees an infinitely old
                # substrate and skips it, so the substrate would be present,
                # perturbable, and invisible to everything downstream. One step
                # per turn at a fixed interval keeps the computation and drops
                # the jitter.
                await asyncio.wait_for(
                    substrate._step_dynamics(SUBSTRATE_STEP_SECONDS), timeout=PHASE_TIMEOUT
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

    def _through_the_intention_loop(self, intended: str, ok: bool, actor: str) -> None:
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
            identifier = loop.intend(
                intention=intended, drive="creation", expected_outcome="the file holds the plan"
            )
            loop.record_action(
                identifier,
                tool_name="write_notes",
                args={},
                result="ok" if ok else "failed",
                success=ok,
                duration_ms=1.0,
            )
            outcome = "the file holds the plan" if ok else "the write did not land"
            loop.observe(identifier, observation="the file holds the plan", actual_outcome=outcome)
        except Exception as exc:  # noqa: BLE001 - the probe's action still stands
            logger.debug("intention loop unavailable: %s", exc)

    #: What she can do to the world, in the order the drives are read. Each is
    #: a real filesystem change with a real reading back, and which one happens
    #: is decided by her state rather than fixed by this harness.
    ACTIONS: ClassVar[tuple[str, ...]] = ("write_notes", "append_log", "make_room")

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
        """The action her most depleted drive picks.

        Not a constant and not a random draw: the budgets are part of the
        deliberation domain, so displacing that domain changes what she does,
        which changes what the filesystem holds, which changes what her senses
        report back. That is the whole of the return route through the world.
        """
        budgets = getattr(getattr(self.state, "motivation", None), "budgets", {}) or {}
        levels = []
        for name in ("social", "curiosity", "creation", "rest", "energy"):
            entry = budgets.get(name)
            if isinstance(entry, dict):
                levels.append((float(entry.get("level", 100.0) or 0.0), name))
        if not levels:
            return self.ACTIONS[0]
        levels.sort()
        return self.ACTIONS[hash(levels[0][1]) % len(self.ACTIONS)]

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
        self._through_the_intention_loop(intended, ok, actor)
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
                # attention can never reach.
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
