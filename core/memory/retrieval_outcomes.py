"""core/memory/retrieval_outcomes.py — memories that can be found harmful.

Clean-room adoption of Springdrift's case-based-reasoning usage stats
(AGPL; mechanism reimplemented from its design, no code taken). Their
``CbrUsageStats`` carries four counters — retrieved, succeeded, helpful,
**harmful** — and their case taxonomy has a first-class ``Pitfall``
category for knowledge learned from failure.

Aura's retrieval had neither. Every signal in the stack was positive or
neutral: relevance, recency, salience, plasticity reinforcement. A memory
that consistently *misled* the turn it was retrieved into could only ever
be diluted by better matches. It could never be demoted, because nothing
was recording that it had done harm, and so the one piece of evidence that
distinguishes a stale memory from a merely unlucky one was never collected.

The asymmetry matters more than it sounds. Positive-only feedback makes
retrieval monotonically more confident about whatever it has surfaced
before: a memory retrieved often looks important, and looking important
gets it retrieved. A harm counter is the only thing in the loop that can
push the other way.

Two things live here:

**Outcome tracking.** Per memory key: retrieved, helpful, harmful. From
those, a :meth:`OutcomeLedger.influence` multiplier that retrieval applies
to a hit's score. It is bounded on both sides, and — importantly — a
memory is never suppressed to zero. A memory that keeps being harmful is
evidence about something; making it unfindable would destroy the record
that says so, and would also make the harm unattributable next time.

**Pitfalls.** ``what NOT to do, learned from failure`` as a category
rather than a note buried in a success-shaped record. A pitfall is exactly
the memory that ordinary relevance ranking under-serves, because the
question that needs it ("should I do X?") does not lexically resemble the
episode where X went wrong.

Grading is explicit and never inferred from the absence of a complaint.
Silence after a retrieval is not evidence it helped, and a ledger that
counted it as such would drift confidently upward forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from core.runtime.lockdep import LockRank, checked_lock

__all__ = [
    "RetrievalVerdict",
    "MemoryCategory",
    "OutcomeStats",
    "OutcomeLedger",
    "get_outcome_ledger",
]


class RetrievalVerdict(StrEnum):
    """What a retrieved memory did for the turn it was retrieved into."""

    HELPFUL = "helpful"
    HARMFUL = "harmful"
    #: Retrieved, used, and it made no difference either way.
    NEUTRAL = "neutral"


class MemoryCategory(StrEnum):
    """What kind of knowledge a memory carries."""

    EPISODE = "episode"
    FACT = "fact"
    PROCEDURE = "procedure"
    #: What NOT to do, learned from a failure. Ranked deliberately, because
    #: ordinary relevance under-serves it: "should I do X?" does not look
    #: like the episode where X went wrong.
    PITFALL = "pitfall"


#: Bounds on the influence multiplier. A well-earned memory gets a lift; a
#: repeatedly harmful one is pushed down but never to zero, because an
#: unfindable memory is also an unattributable one.
_MAX_BOOST = 1.35
_MIN_INFLUENCE = 0.35

#: Gradings needed before outcomes move ranking at all. Below this the
#: sample is too small to separate a bad memory from an unlucky one.
_MIN_GRADED = 3

_MAX_TRACKED = 4096


@dataclass
class OutcomeStats:
    """The record for one memory key."""

    key: str
    retrieved: int = 0
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    category: MemoryCategory = MemoryCategory.EPISODE
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def graded(self) -> int:
        return self.helpful + self.harmful + self.neutral

    @property
    def harm_rate(self) -> float:
        return round(self.harmful / self.graded, 4) if self.graded else 0.0

    @property
    def help_rate(self) -> float:
        return round(self.helpful / self.graded, 4) if self.graded else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "retrieved": self.retrieved,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
            "graded": self.graded,
            "harm_rate": self.harm_rate,
            "help_rate": self.help_rate,
            "category": str(self.category),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class OutcomeLedger:
    """Tracks whether retrieving a memory helped or hurt."""

    def __init__(self) -> None:
        self._lock = checked_lock("retrieval_outcomes", rank=LockRank.LEAF)
        self._stats: dict[str, OutcomeStats] = {}

    # ------------------------------------------------------------------ write

    def note_retrieved(self, key: str, *, category: MemoryCategory | str | None = None) -> None:
        """Record that a memory was surfaced. Says nothing about whether it helped."""
        name = str(key or "").strip()
        if not name:
            return
        with self._lock:
            stats = self._stats.get(name)
            if stats is None:
                stats = OutcomeStats(key=name)
                self._stats[name] = stats
            stats.retrieved += 1
            stats.last_seen = time.time()
            if category is not None:
                try:
                    stats.category = MemoryCategory(category)
                except ValueError:
                    pass
            self._evict_locked()

    def grade(self, key: str, verdict: RetrievalVerdict | str) -> OutcomeStats | None:
        """Record what a retrieved memory actually did.

        Must be called explicitly. Silence after a retrieval is NOT evidence
        that it helped, and inferring one from the other is how a ledger
        drifts confidently upward forever.
        """
        name = str(key or "").strip()
        if not name:
            return None
        try:
            call = RetrievalVerdict(verdict)
        except ValueError:
            return None
        with self._lock:
            stats = self._stats.get(name)
            if stats is None:
                stats = OutcomeStats(key=name)
                self._stats[name] = stats
            if call is RetrievalVerdict.HELPFUL:
                stats.helpful += 1
            elif call is RetrievalVerdict.HARMFUL:
                stats.harmful += 1
            else:
                stats.neutral += 1
            stats.last_seen = time.time()
            self._evict_locked()
            return OutcomeStats(**{**stats.__dict__})

    def _evict_locked(self) -> None:
        if len(self._stats) <= _MAX_TRACKED:
            return
        # Evict the least informative first: nothing graded, seen longest ago.
        ordered = sorted(self._stats.values(), key=lambda s: (s.graded, s.last_seen))
        for stats in ordered[: len(self._stats) - _MAX_TRACKED]:
            self._stats.pop(stats.key, None)

    # ------------------------------------------------------------------- read

    def influence(self, key: str) -> float:
        """Score multiplier for this memory, from its own track record.

        Returns 1.0 — no opinion — until there are enough gradings to tell a
        bad memory from an unlucky one.
        """
        name = str(key or "").strip()
        if not name:
            return 1.0
        with self._lock:
            stats = self._stats.get(name)
            if stats is None or stats.graded < _MIN_GRADED:
                return 1.0
            help_rate = stats.helpful / stats.graded
            harm_rate = stats.harmful / stats.graded
        # Harm pulls harder than help lifts: being actively misled is worse
        # than being unhelpfully verbose, and the asymmetry is the point of
        # having a harm counter at all.
        raw = 1.0 + (help_rate * (_MAX_BOOST - 1.0)) - (harm_rate * 1.0)
        return round(max(_MIN_INFLUENCE, min(_MAX_BOOST, raw)), 4)

    def stats_for(self, key: str) -> OutcomeStats | None:
        with self._lock:
            stats = self._stats.get(str(key or "").strip())
            return OutcomeStats(**{**stats.__dict__}) if stats else None

    def harmful_memories(self, *, minimum_graded: int = _MIN_GRADED) -> list[dict[str, Any]]:
        """Memories doing more harm than good — the list nothing used to produce."""
        with self._lock:
            candidates = [
                s
                for s in self._stats.values()
                if s.graded >= minimum_graded and s.harmful > s.helpful
            ]
        return [s.to_dict() for s in sorted(candidates, key=lambda s: -s.harm_rate)]

    def pitfalls(self) -> list[dict[str, Any]]:
        with self._lock:
            found = [s for s in self._stats.values() if s.category is MemoryCategory.PITFALL]
        return [s.to_dict() for s in found]

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._stats)
            graded = sum(1 for s in self._stats.values() if s.graded)
            harmful = sum(1 for s in self._stats.values() if s.harmful)
            pitfalls = sum(
                1 for s in self._stats.values() if s.category is MemoryCategory.PITFALL
            )
        return {
            "tracked": total,
            "graded": graded,
            "with_harm_recorded": harmful,
            "pitfalls": pitfalls,
            "harmful_memories": self.harmful_memories(),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._stats.clear()


_LEDGER = OutcomeLedger()


def get_outcome_ledger() -> OutcomeLedger:
    return _LEDGER


def apply_influence(hits: Iterable[Any]) -> list[Any]:
    """Re-rank hits by their track record, in place-ish.

    Hits carry ``content``/``score``; the key is the hit's source when it
    has one, else its content. Never raises into retrieval — a ledger
    problem must degrade ranking, not break recall.
    """
    ledger = get_outcome_ledger()
    scored = list(hits)
    for hit in scored:
        try:
            key = str(getattr(hit, "source", "") or getattr(hit, "content", ""))[:200]
            if not key:
                continue
            multiplier = ledger.influence(key)
            if multiplier != 1.0:
                hit.score = _as_score(getattr(hit, "score", 0.0)) * multiplier
        except (AttributeError, TypeError, ValueError):
            continue
    # The sort key must be total. Reading a score is the same coercion that
    # is guarded above, and a malformed hit reaching the comparator would
    # raise out of retrieval itself — turning a ranking refinement into a
    # recall outage, which is the opposite of what this is for.
    scored.sort(key=_as_score_of, reverse=True)
    return scored


def _as_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_score_of(hit: Any) -> float:
    return _as_score(getattr(hit, "score", 0.0))


def _register_runtime_service() -> None:
    """Publish the ledger so the runtime health surface can read it.

    `core/runtime/health_contract.py` imported this module directly — a
    layering violation grandfathered in the baseline, so the gate passed while
    the foundation depended on memory to describe its own health. The rule
    exists so that report still works when memory is what failed, which an
    import cannot honour and a registry lookup can.
    """
    try:
        from core.runtime.service_registry import register_runtime_service

        register_runtime_service(
            "retrieval_outcome_ledger",
            get_outcome_ledger(),
            owner="core/memory/retrieval_outcomes.py",
            registered_by="core.memory.retrieval_outcomes._register_runtime_service",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        # The surface reports an unregistered ledger as missing, which is the
        # honest reading; nothing here should raise at import time.
        pass


_register_runtime_service()
