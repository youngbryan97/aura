"""core/cognition/cognitive_event.py — one causal timeline for a thought.

Aura can already answer "what happened" four ways and none of them is a causal
graph. ``turn_receipt`` records what a turn produced. ``bus_recorder`` keeps a
ring of events, which forgets. ``unified_action_log`` records actions.
``get_tracer()`` writes a Perfetto trace, which is timing. Reconstructing why
a consequential action happened means reading all four and joining them by
timestamp, which is how a root cause takes an afternoon.

A :class:`CognitiveEvent` is one step of cognition with three things those
four lack:

* a **monotonic id** that orders it against every other step, in both phase
  loops, without clock comparison;
* its **causal parents** — the events whose results this one used;
* its **read-dependencies** — the state it consulted, each tagged with what
  it found.

The third is the one that does the work, and it carries a distinction Aura's
learners have never made. Reading a field and finding ``False`` is not the
same as reading a field that was not there, and neither is the same as not
looking. :class:`Epistemic` separates them, and
:meth:`EventGraph.minimal_support` refuses to put an ``UNOBSERVED`` dependency
into a compiled rule's preconditions. Without that, a chunk learns "this works
when the sidebar is absent" from a run that never checked the sidebar, and
then fires confidently in the case it was never entitled to an opinion about.

What minimal support is for
---------------------------
Soar's explanation-based chunking asks: of everything true when this worked,
which parts actually supported the result? :meth:`EventGraph.minimal_support`
walks back from an event through causal parents and collects the
dependencies that were read on the surviving path. Context that was present
and never read does not enter, which is what makes a compiled rule fire under
paraphrase — it never depended on the wording it happened to see.

This module is pure, clock-injectable and bounded. It records; it decides
nothing.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "Phase",
    "Epistemic",
    "ReadDependency",
    "CognitiveEvent",
    "EventGraph",
    "get_event_graph",
    "reset_event_graph_for_test",
    "current_cycle",
    "cycle",
    "reads",
]


class Phase(StrEnum):
    """What kind of cognitive step this was.

    Deliberately coarse. A phase taxonomy that mirrors the module list ages
    with the module list; these are the steps any cognitive architecture has.
    """

    PERCEIVE = "perceive"
    RETRIEVE = "retrieve"
    ELABORATE = "elaborate"
    PREFER = "prefer"
    SELECT = "select"
    APPLY = "apply"
    VERIFY = "verify"
    LEARN = "learn"
    IMPASSE = "impasse"


class Epistemic(StrEnum):
    """What a read found. The three cases a boolean cannot hold."""

    #: The state was read and had a value.
    OBSERVED = "observed"
    #: The state was read and was definitely absent or false.
    OBSERVED_ABSENT = "observed_absent"
    #: Nobody looked. Never a precondition.
    UNOBSERVED = "unobserved"
    #: Looked and could not tell — the read failed, timed out, was refused.
    INACCESSIBLE = "inaccessible"


#: Statuses a compiled rule may rest a precondition on.
_SUPPORTABLE = frozenset({Epistemic.OBSERVED, Epistemic.OBSERVED_ABSENT})


@dataclass(frozen=True, slots=True)
class ReadDependency:
    """One piece of state an event consulted, and what it found."""

    key: str
    status: Epistemic = Epistemic.OBSERVED
    value_digest: str = ""
    owner: str = ""

    @property
    def supportable(self) -> bool:
        return self.status in _SUPPORTABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status.value,
            "value_digest": self.value_digest,
            "owner": self.owner,
        }


@dataclass(frozen=True, slots=True)
class CognitiveEvent:
    """One step, ordered, attributed and causally linked."""

    seq: int
    phase: Phase
    organ: str
    label: str
    loop: str = ""
    cycle_id: int = 0
    parents: tuple[int, ...] = ()
    reads: tuple[ReadDependency, ...] = ()
    #: Identities of CognitiveStateRefs this event produced, if any.
    produced: tuple[str, ...] = ()
    at: float = 0.0
    duration_s: float = 0.0
    outcome: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "phase": self.phase.value,
            "organ": self.organ,
            "label": self.label,
            "loop": self.loop,
            "cycle_id": self.cycle_id,
            "parents": list(self.parents),
            "reads": [r.to_dict() for r in self.reads],
            "produced": list(self.produced),
            "at": self.at,
            "duration_s": self.duration_s,
            "outcome": self.outcome,
            "detail": dict(self.detail),
        }


#: How many events the graph keeps. A causal timeline that forgets mid-turn is
#: no timeline; a causal timeline that never forgets is a leak in a process
#: that runs for weeks. Bounded, and the bound is reported.
_DEFAULT_CAPACITY = 20_000

_cycle_counter = itertools.count(1)
_cycle_local = threading.local()


def current_cycle() -> int:
    """The cognitive cycle this thread is inside, or 0 outside one."""
    return getattr(_cycle_local, "cycle_id", 0)


class _CycleScope:
    def __init__(self, label: str) -> None:
        self.label = label
        self.cycle_id = 0
        self._previous = 0

    def __enter__(self) -> _CycleScope:
        self._previous = current_cycle()
        self.cycle_id = next(_cycle_counter)
        _cycle_local.cycle_id = self.cycle_id
        return self

    def __exit__(self, *_exc: object) -> None:
        _cycle_local.cycle_id = self._previous


def cycle(label: str = "") -> _CycleScope:
    """Open a cognitive cycle. Every event recorded inside it carries its id.

    Nesting is allowed and the inner cycle wins for events inside it, which is
    what a substate needs: its events belong to it, and its parent's events
    still belong to the parent.
    """
    return _CycleScope(label)


class EventGraph:
    """The events, their causal edges, and the questions worth asking of them."""

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY, clock=time.time) -> None:
        self._lock = checked_lock("core.cognition.cognitive_event.EventGraph", reentrant=True)
        self._events: dict[int, CognitiveEvent] = {}
        self._order: list[int] = []
        self._children: dict[int, set[int]] = {}
        self._by_cycle: dict[int, list[int]] = {}
        self._seq = itertools.count(1)
        self._capacity = int(capacity)
        self._clock = clock
        self._dropped = 0

    # ── recording ─────────────────────────────────────────────────────

    def record(
        self,
        phase: Phase,
        organ: str,
        label: str,
        *,
        loop: str = "",
        parents: Sequence[int] = (),
        reads: Sequence[ReadDependency] = (),
        produced: Sequence[str] = (),
        duration_s: float = 0.0,
        outcome: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> CognitiveEvent:
        """Record one step and return it. The returned ``seq`` is its parent id."""
        with self._lock:
            event = CognitiveEvent(
                seq=next(self._seq),
                phase=phase,
                organ=organ,
                label=label,
                loop=loop,
                cycle_id=current_cycle(),
                parents=tuple(int(p) for p in parents if p in self._events),
                reads=tuple(reads),
                produced=tuple(produced),
                at=self._clock(),
                duration_s=float(duration_s),
                outcome=outcome,
                detail=dict(detail or {}),
            )
            self._events[event.seq] = event
            self._order.append(event.seq)
            for parent in event.parents:
                self._children.setdefault(parent, set()).add(event.seq)
            if event.cycle_id:
                self._by_cycle.setdefault(event.cycle_id, []).append(event.seq)
            self._evict_locked()
            return event

    def _evict_locked(self) -> None:
        while len(self._order) > self._capacity:
            oldest = self._order.pop(0)
            event = self._events.pop(oldest, None)
            self._children.pop(oldest, None)
            self._dropped += 1
            if event and event.cycle_id in self._by_cycle:
                bucket = self._by_cycle[event.cycle_id]
                if oldest in bucket:
                    bucket.remove(oldest)
                if not bucket:
                    del self._by_cycle[event.cycle_id]

    # ── reading ───────────────────────────────────────────────────────

    def get(self, seq: int) -> CognitiveEvent | None:
        with self._lock:
            return self._events.get(seq)

    def ancestors(self, seq: int, *, limit: int = 10_000) -> list[CognitiveEvent]:
        """Every event this one causally depends on, nearest first."""
        with self._lock:
            seen: set[int] = set()
            out: list[CognitiveEvent] = []
            frontier = [seq]
            while frontier and len(out) < limit:
                current = frontier.pop(0)
                event = self._events.get(current)
                if event is None:
                    continue
                for parent in event.parents:
                    if parent not in seen:
                        seen.add(parent)
                        parent_event = self._events.get(parent)
                        if parent_event is not None:
                            out.append(parent_event)
                            frontier.append(parent)
            return out

    def bundle(self, seq: int) -> dict[str, Any]:
        """Everything needed to reconstruct why this event happened.

        One call, one object. This is what replaces joining four logs by
        timestamp when an action has to be root-caused.
        """
        with self._lock:
            event = self._events.get(seq)
            if event is None:
                return {"found": False, "seq": seq}
            chain = self.ancestors(seq)
            support = self.minimal_support(seq)
            return {
                "found": True,
                "event": event.to_dict(),
                "cycle_id": event.cycle_id,
                "ancestors": [e.to_dict() for e in reversed(chain)],
                "minimal_support": [d.to_dict() for d in support],
                "unsupportable_reads": [
                    d.to_dict()
                    for e in [event, *chain]
                    for d in e.reads
                    if not d.supportable
                ],
                "phases": sorted({e.phase.value for e in [event, *chain]}),
                "organs": sorted({e.organ for e in [event, *chain]}),
                "span_s": (event.at - min(e.at for e in chain)) if chain else 0.0,
            }

    def minimal_support(self, seq: int) -> list[ReadDependency]:
        """The dependencies that actually supported this result.

        Walks back through causal parents and keeps the reads on that path.
        Anything ``UNOBSERVED`` or ``INACCESSIBLE`` is dropped: a rule may not
        rest on a fact nobody established, and a rule that does will fire in
        the case it was never entitled to an opinion about.

        Duplicates collapse by key, keeping the strongest status seen, so the
        same field read in three steps is one precondition.
        """
        with self._lock:
            event = self._events.get(seq)
            if event is None:
                return []
            best: dict[str, ReadDependency] = {}
            for step in [event, *self.ancestors(seq)]:
                for dependency in step.reads:
                    if not dependency.supportable:
                        continue
                    existing = best.get(dependency.key)
                    if existing is None or (
                        existing.status is Epistemic.OBSERVED_ABSENT
                        and dependency.status is Epistemic.OBSERVED
                    ):
                        best[dependency.key] = dependency
            return [best[k] for k in sorted(best)]

    def cycle_events(self, cycle_id: int) -> list[CognitiveEvent]:
        with self._lock:
            return [self._events[s] for s in self._by_cycle.get(cycle_id, []) if s in self._events]

    def phase_timing(self) -> dict[str, Any]:
        """Where the time went, by phase and by loop.

        Card 073's bar: a bottleneck localises to a phase. The two phase loops
        report side by side because they serve different traffic and comparing
        them by hand was how the last three latency defects hid.
        """
        with self._lock:
            by_phase: dict[str, dict[str, float]] = {}
            by_loop: dict[str, dict[str, float]] = {}
            for event in self._events.values():
                for bucket, key in ((by_phase, event.phase.value), (by_loop, event.loop or "unlabelled")):
                    row = bucket.setdefault(key, {"count": 0.0, "total_s": 0.0, "max_s": 0.0})
                    row["count"] += 1
                    row["total_s"] += event.duration_s
                    row["max_s"] = max(row["max_s"], event.duration_s)
            for bucket in (by_phase, by_loop):
                for row in bucket.values():
                    row["mean_s"] = row["total_s"] / row["count"] if row["count"] else 0.0
            return {"by_phase": by_phase, "by_loop": by_loop}

    def report(self) -> dict[str, Any]:
        with self._lock:
            total_reads = sum(len(e.reads) for e in self._events.values())
            unobserved = sum(
                1 for e in self._events.values() for r in e.reads if not r.supportable
            )
            rooted = sum(1 for e in self._events.values() if e.parents)
            return {
                "events": len(self._events),
                "dropped": self._dropped,
                "capacity": self._capacity,
                "cycles": len(self._by_cycle),
                "reads": total_reads,
                "unsupportable_reads": unobserved,
                "events_with_a_causal_parent": rooted,
                "causal_coverage": (rooted / len(self._events)) if self._events else None,
            }

    def __iter__(self) -> Iterator[CognitiveEvent]:
        with self._lock:
            return iter([self._events[s] for s in self._order if s in self._events])


_graph_lock = checked_lock("core.cognition.cognitive_event.singleton")
_graph: EventGraph | None = None


def get_event_graph() -> EventGraph:
    global _graph
    with _graph_lock:
        if _graph is None:
            _graph = EventGraph()
        return _graph


def reset_event_graph_for_test(**kwargs: Any) -> EventGraph:
    global _graph
    with _graph_lock:
        _graph = EventGraph(**kwargs)
        return _graph


def reads(pairs: Iterable[tuple[str, Any]], *, owner: str = "") -> tuple[ReadDependency, ...]:
    """Build read-dependencies from key/value pairs, typing absence correctly.

    ``None`` becomes ``OBSERVED_ABSENT`` rather than ``OBSERVED`` with a null,
    because "I looked and it was not there" is a fact and "I looked and got
    None" is usually the same fact wearing a disguise. A caller that did not
    look must not be in this list at all.
    """
    import hashlib

    out = []
    for key, value in pairs:
        status = Epistemic.OBSERVED_ABSENT if value is None or value is False else Epistemic.OBSERVED
        digest = hashlib.blake2s(repr(value).encode("utf-8"), digest_size=6).hexdigest()
        out.append(ReadDependency(key=key, status=status, value_digest=digest, owner=owner))
    return tuple(out)
