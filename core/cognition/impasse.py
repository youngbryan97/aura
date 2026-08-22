"""Soar's impasse-driven subgoaling and chunking, with the utility problem taken seriously.

Soar's central claim is that learning has one occasion: the architecture notices
it cannot decide, creates a substate to work out what to do, and compiles the
result into a rule so the same deadlock is never worked through twice. The
occasion for learning is an *impasse*.

Aura has decision points that deadlock and no vocabulary for it. The global
workspace sorts candidates and takes ``[0]``, so exactly-tied bids resolve by
whoever submitted first — a tie impasse settled by list order, invisibly.
Nothing anywhere records that a decision was arbitrary, so nothing can learn
from it and nothing can report how often it happens.

The four impasse types are Soar's, and they are exhaustive over "the decision
procedure did not produce a choice":

* :attr:`ImpasseType.TIE` — several candidates, nothing discriminates.
* :attr:`ImpasseType.CONFLICT` — preferences contradict each other.
* :attr:`ImpasseType.REJECTION` — every candidate was rejected, or none offered.
* :attr:`ImpasseType.NO_CHANGE` — a choice was made and nothing moved.

Why this is not just a chunk cache
-----------------------------------
The standard criticism of chunking, and the one Soar spent years on, is the
**utility problem**: learned rules are not free. Each one adds match cost to
every subsequent decision, so a system that learns indiscriminately gets slower
as it gets more experienced — and the slowdown is invisible, because each
individual chunk looks like a win.

So a chunk here is retained only while it *pays*, on a per-use expected value::

    EV = p_correct · cost_saved_per_use − match_cost

and :meth:`ChunkStore.prune` retracts everything with ``EV <= 0``. That is a
derived condition, not a tuned threshold: a chunk whose expected value is
negative is costing more than it saves, by definition. The accounting is what
makes chunking safe to leave switched on.

The second criticism is **over-generalisation** — a chunk that fires in
situations its originating episode did not cover, giving a confident wrong
answer. That is why :meth:`ChunkStore.record_outcome` exists and why
``p_correct`` is measured rather than assumed: a chunk that fires often and is
wrong has a negative EV and is retracted by the same rule that catches the
expensive ones.

This module is pure and clock-injectable. It decides nothing on its own; it
records deadlocks, learns from resolutions, and reports what it has learned.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from core.runtime.lockdep import checked_lock

__all__ = [
    "ImpasseType",
    "Impasse",
    "Chunk",
    "ChunkStore",
    "ImpasseLearner",
    "get_impasse_learner",
    "classify",
    "situation_signature",
]


class ImpasseType(StrEnum):
    """Why the decision procedure failed to produce a choice."""

    #: Several candidates are indistinguishable under the available preferences.
    TIE = "tie"
    #: Preferences are mutually contradictory (a > b and b > a).
    CONFLICT = "conflict"
    #: Nothing was proposed, or everything proposed was rejected.
    REJECTION = "rejection"
    #: Something was chosen and applying it changed nothing.
    NO_CHANGE = "no_change"
    #: Two assertions cannot both hold: several candidates are each required, or
    #: a required candidate is also prohibited. Soar's fifth type, and the one
    #: that matters most here — the others say a choice could not be made, this
    #: one says the constraints themselves are inconsistent, so no amount of
    #: further deliberation in a substate can resolve it. It is not resolvable
    #: by learning and must reach a human.
    CONSTRAINT_FAILURE = "constraint_failure"


@dataclass(frozen=True)
class Impasse:
    """One recorded failure to decide."""

    type: ImpasseType
    signature: str
    candidates: tuple[str, ...]
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.type.value} impasse on {list(self.candidates)} [{self.signature}]"


def situation_signature(context: Mapping[str, object], candidates: Iterable[str]) -> str:
    """A stable key for "this decision, in this situation".

    Both halves matter and leaving either out is a known way to get chunking
    wrong. Without the candidate set a chunk answers a question that was not
    asked; without the context it over-generalises across situations that
    happened to offer the same options — the exact failure mode chunking is
    criticised for.
    """
    ctx = ",".join(f"{k}={context[k]!r}" for k in sorted(context))
    opts = ",".join(sorted(candidates))
    return f"{{{ctx}}}|[{opts}]"


def classify(
    candidates: Sequence[str],
    *,
    scores: Mapping[str, float] | None = None,
    rejected: Iterable[str] = (),
    preferences: Iterable[tuple[str, str]] = (),
    changed: bool | None = None,
    tolerance: float = 0.0,
    context: Mapping[str, object] | None = None,
) -> Impasse | None:
    """Detect an impasse, or return None when the decision is well-formed.

    ``preferences`` are ``(better, worse)`` pairs. ``changed`` is whether
    applying the selection actually moved the state; pass it only once that is
    known, since a no-change impasse is detectable in hindsight alone.

    Order matters: rejection is checked before ties because an empty field is
    not a tie, and conflict before ties because contradictory preferences are a
    different and more serious failure than an absent one.
    """
    ctx = dict(context or {})
    rejected_set = set(rejected)
    live = [c for c in candidates if c not in rejected_set]
    signature = situation_signature(ctx, candidates)

    if not live:
        return Impasse(
            type=ImpasseType.REJECTION,
            signature=signature,
            candidates=tuple(candidates),
            detail=(
                "no candidates were proposed"
                if not candidates
                else "every candidate was rejected"
            ),
        )

    pref_pairs = {(a, b) for a, b in preferences}
    contradictions = sorted(
        {tuple(sorted((a, b))) for a, b in pref_pairs if (b, a) in pref_pairs}
    )
    if contradictions:
        return Impasse(
            type=ImpasseType.CONFLICT,
            signature=signature,
            candidates=tuple(live),
            detail=f"contradictory preferences: {contradictions}",
        )

    if changed is False:
        return Impasse(
            type=ImpasseType.NO_CHANGE,
            signature=signature,
            candidates=tuple(live),
            detail="the selected operator applied and nothing changed",
        )

    if scores is not None and len(live) > 1:
        ordered = sorted(live, key=lambda c: scores.get(c, 0.0), reverse=True)
        best = scores.get(ordered[0], 0.0)
        tied = [c for c in ordered if abs(scores.get(c, 0.0) - best) <= tolerance]
        if len(tied) > 1:
            return Impasse(
                type=ImpasseType.TIE,
                signature=situation_signature(ctx, tied),
                candidates=tuple(sorted(tied)),
                detail=f"{len(tied)} candidates within {tolerance:g} of the best bid",
            )

    return None


@dataclass
class Chunk:
    """A compiled resolution, with the accounting that decides whether it stays."""

    signature: str
    resolution: str
    impasse_type: ImpasseType
    #: Seconds of deliberation this chunk avoids each time it fires. Measured
    #: from the substate that produced it, not estimated.
    cost_saved_per_use: float
    #: What this chunk costs to match on every decision, whether it fires or
    #: not. The utility problem lives here.
    match_cost: float
    uses: int = 0
    correct: int = 0
    incorrect: int = 0

    @property
    def outcomes(self) -> int:
        return self.correct + self.incorrect

    @property
    def p_correct(self) -> float:
        """Measured accuracy, optimistic until there is evidence otherwise.

        An unjudged chunk is treated as correct so that a newly learned
        resolution gets the chance to be evaluated; the first wrong outcome
        starts pulling it down immediately.
        """
        if self.outcomes == 0:
            return 1.0
        return self.correct / self.outcomes

    @property
    def expected_value(self) -> float:
        """``p_correct · cost_saved − match_cost``, in seconds per decision.

        Positive means the chunk pays for its own match cost. This is the whole
        answer to the utility problem: not a cap on how much is learned, but a
        continuously re-evaluated question of whether each thing learned is
        still worth carrying.
        """
        return self.p_correct * self.cost_saved_per_use - self.match_cost


#: Hard ceiling on retained chunks and recorded impasses. A long-lived process
#: meets unboundedly many distinct situations, so a store that only grows is a
#: leak with a learning-shaped justification — NASA/JPL rule 3 applied to a
#: cognitive cache. Eviction is by lowest expected value, so the bound removes
#: what pays least rather than what arrived first.
_MAX_CHUNKS = 4096
_MAX_IMPASSE_LOG = 2048


@dataclass
class ChunkStore:
    """Learned impasse resolutions, retained only while they pay."""

    max_chunks: int = _MAX_CHUNKS
    max_impasse_log: int = _MAX_IMPASSE_LOG
    _chunks: dict[str, Chunk] = field(default_factory=dict)
    _impasses: list[Impasse] = field(default_factory=list)
    _retracted: list[tuple[str, str]] = field(default_factory=list)
    _evicted: int = 0

    # -- recording -------------------------------------------------------

    def record_impasse(self, impasse: Impasse) -> Impasse:
        """Note that a decision deadlocked, whether or not it gets chunked.

        Kept separately from the chunks because the impasse rate is a
        diagnostic in its own right: a subsystem that deadlocks constantly and
        resolves the same way every time is telling you its preferences are
        underspecified.
        """
        self._impasses.append(impasse)
        if len(self._impasses) > self.max_impasse_log:
            # Keep the recent tail: the rate is the diagnostic, and an
            # unbounded log of every deadlock a long-lived process ever hit is
            # a leak rather than a record.
            del self._impasses[: len(self._impasses) - self.max_impasse_log]
        return impasse

    def learn(
        self,
        impasse: Impasse,
        resolution: str,
        *,
        cost_saved_per_use: float,
        match_cost: float,
    ) -> Chunk:
        """Compile the substate's result into a rule for this situation.

        Re-learning an existing signature with the same resolution keeps the
        accumulated statistics — the whole point of the accounting is that it
        survives. A *different* resolution replaces the chunk and resets them,
        because the evidence was gathered about a different answer.
        """
        existing = self._chunks.get(impasse.signature)
        if existing is not None and existing.resolution == resolution:
            existing.cost_saved_per_use = cost_saved_per_use
            existing.match_cost = match_cost
            return existing
        chunk = Chunk(
            signature=impasse.signature,
            resolution=resolution,
            impasse_type=impasse.type,
            cost_saved_per_use=cost_saved_per_use,
            match_cost=match_cost,
        )
        self._chunks[impasse.signature] = chunk
        self._enforce_capacity()
        return chunk

    def _enforce_capacity(self) -> None:
        """Evict the least valuable chunks once the store is full.

        Evicting by expected value rather than by age is the point: a chunk
        that has been paying for months should outlive one learned five
        minutes ago that has never fired. Age-based eviction would make the
        cache forget precisely what it had learned best.
        """
        overflow = len(self._chunks) - self.max_chunks
        if overflow <= 0:
            return
        ranked = sorted(self._chunks.values(), key=lambda c: c.expected_value)
        for chunk in ranked[:overflow]:
            del self._chunks[chunk.signature]
            self._evicted += 1

    def recall(self, signature: str) -> Chunk | None:
        """The learned resolution for this exact situation, if one is retained."""
        chunk = self._chunks.get(signature)
        if chunk is not None:
            chunk.uses += 1
        return chunk

    def record_outcome(self, signature: str, *, correct: bool) -> None:
        """Judge a chunk's firing after the fact.

        Without this, ``p_correct`` is an assumption and an over-general chunk
        is immortal.
        """
        chunk = self._chunks.get(signature)
        if chunk is None:
            return
        if correct:
            chunk.correct += 1
        else:
            chunk.incorrect += 1

    # -- utility ---------------------------------------------------------

    def prune(self) -> list[Chunk]:
        """Retract every chunk that is not paying for itself.

        Returns what was removed. Two populations get caught by the one rule:
        the expensive chunk whose match cost exceeds what it saves, and the
        over-general chunk that is cheap but wrong often enough that its
        expected saving has gone negative.
        """
        doomed = [c for c in self._chunks.values() if c.expected_value <= 0.0]
        for chunk in doomed:
            del self._chunks[chunk.signature]
            self._retracted.append(
                (
                    chunk.signature,
                    f"EV={chunk.expected_value:+.4g}s "
                    f"(p_correct={chunk.p_correct:.2f}, "
                    f"saves={chunk.cost_saved_per_use:.4g}s, "
                    f"match={chunk.match_cost:.4g}s)",
                )
            )
        return doomed

    def total_match_cost(self) -> float:
        """What the learned set costs on every single decision.

        The number that makes the utility problem visible. If this grows faster
        than the savings, learning is making the system slower and the only way
        to know is to add it up.
        """
        return sum(c.match_cost for c in self._chunks.values())

    def net_value(self) -> float:
        """Summed expected value across everything retained."""
        return sum(c.expected_value for c in self._chunks.values())

    # -- reporting -------------------------------------------------------

    def chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda c: c.signature)

    def impasses(self) -> list[Impasse]:
        return list(self._impasses)

    def impasse_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {t.value: 0 for t in ImpasseType}
        for imp in self._impasses:
            counts[imp.type.value] += 1
        return counts

    def report(self) -> dict[str, object]:
        return {
            "chunks": len(self._chunks),
            "impasses": len(self._impasses),
            "impasse_counts": self.impasse_counts(),
            "total_match_cost_s": round(self.total_match_cost(), 6),
            "net_value_s": round(self.net_value(), 6),
            "evicted": self._evicted,
            "retracted": list(self._retracted[-16:]),
        }


class ImpasseLearner:
    """Process-wide chunk learning, safe to call from live cognition.

    :class:`ChunkStore` is the mechanism; this is the thing a running system
    actually holds. It adds the three properties a cache in a long-lived
    cognitive loop needs and a bare dict does not: a lock, a bound, and a
    pruning cadence that is tied to use rather than to a timer, so a store that
    is never touched never spends any time maintaining itself.
    """

    def __init__(self, store: ChunkStore | None = None) -> None:

        self._lock = checked_lock("core.cognition.impasse")
        self._store = store or ChunkStore()
        self._since_prune = 0
        self._hits = 0
        self._misses = 0

    #: Learn/recall operations between prunes. Pruning walks every chunk, so
    #: doing it per call would add the cost the accounting exists to avoid;
    #: doing it never lets non-paying chunks accumulate match cost unchecked.
    _PRUNE_INTERVAL = 64

    def recall(self, signature: str) -> Chunk | None:
        with self._lock:
            chunk = self._store.recall(signature)
            if chunk is None:
                self._misses += 1
            else:
                self._hits += 1
            return chunk

    def record_impasse(self, impasse: Impasse) -> None:
        with self._lock:
            self._store.record_impasse(impasse)

    def learn(
        self,
        impasse: Impasse,
        resolution: str,
        *,
        cost_saved_per_use: float,
        match_cost: float,
    ) -> Chunk:
        with self._lock:
            chunk = self._store.learn(
                impasse,
                resolution,
                cost_saved_per_use=cost_saved_per_use,
                match_cost=match_cost,
            )
            self._maybe_prune_locked()
            return chunk

    def record_outcome(self, signature: str, *, correct: bool) -> None:
        with self._lock:
            self._store.record_outcome(signature, correct=correct)
            self._maybe_prune_locked()

    def _maybe_prune_locked(self) -> None:
        self._since_prune += 1
        if self._since_prune >= self._PRUNE_INTERVAL:
            self._since_prune = 0
            self._store.prune()

    def prune_now(self) -> list[Chunk]:
        with self._lock:
            self._since_prune = 0
            return self._store.prune()

    def report(self) -> dict[str, object]:
        with self._lock:
            report = self._store.report()
            total = self._hits + self._misses
            report["hits"] = self._hits
            report["misses"] = self._misses
            report["hit_rate"] = round(self._hits / total, 4) if total else 0.0
            return report


_learner: ImpasseLearner | None = None


def get_impasse_learner() -> ImpasseLearner:
    """The process-wide learner. Created on first use, never reset implicitly."""
    global _learner
    if _learner is None:
        _learner = ImpasseLearner()
    return _learner
