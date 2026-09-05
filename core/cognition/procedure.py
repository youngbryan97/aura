"""core/cognition/procedure.py — one currency for everything Aura has learned to do.

Count the ways this repository represents "a thing she knows how to do":
a ``Chunk`` in impasse.py with an expected value; a ``GeneralizedRule`` in
procedural_generalization.py with a Wilson bound; a macro in the skill library;
a ``Doing`` AST in an_action_she_composed.py; a habit; a planner operator; an
RLC opcode sequence; a tool schema. Eight representations, eight value
functions, and no way for a procedure learned by one to be matched, priced or
composed against a procedure learned by another. Every one of them is right
for its own learner. None of them is a currency.

A :class:`Procedure` is the currency. It does not replace the eight: the
program stays whatever its backend made, and the backend executes it. What is
common is the part every learner needed and each invented separately —

* a **signature**: typed preconditions and effects, which is what makes two
  procedures composable without a human deciding they fit;
* a **value**: one :class:`ProceduralValue` all eight compute into, so a chunk
  and a generalized rule can lose to each other;
* an **origin**: the impasse or trace it came from, the minimal support it was
  compiled against, and the counterexamples that have narrowed it since;
* a **cost model**: what it costs to match, what it costs to run, and what it
  saves — because a procedure that is free to learn is not free to keep.

The value contract, and why it is subtraction
---------------------------------------------
Soar's utility problem is that learned rules add match cost to every later
decision, so a system that learns indiscriminately gets slower as it gets more
experienced, and each individual rule still looks like a win.
:attr:`ProceduralValue.net` is the arithmetic that stops it::

    net = p_success · value_when_it_works − match_cost − risk_cost

A procedure whose net turns negative is retired by the same rule that catches
the expensive ones, and ``impasse.ChunkStore`` already proved the shape works;
this generalises it so a rule from a different learner is retired by the same
condition. The three terms come from the backends unchanged — Wilson bounds
stay Wilson bounds, chunk EV stays chunk EV — and are reported side by side
rather than averaged, because two learners disagreeing about a procedure's
worth is information.

Matching at scale
-----------------
Eight learners feeding one store means the store grows for the lifetime of the
agent, and a linear scan over it is the utility problem wearing a different
hat. :class:`ProcedureIndex` discriminates on precondition keys first, so the
number of procedures actually compared grows with the number that could
possibly apply rather than with the number that exist.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.evidence.packet import EvidencePacket

__all__ = [
    "Backend",
    "Effect",
    "Precondition",
    "Signature",
    "ProceduralValue",
    "Origin",
    "Reversibility",
    "Procedure",
    "ProcedureIndex",
    "ProcedureRegistry",
    "get_procedure_registry",
    "reset_procedure_registry_for_test",
    "compose",
]

#: Decay on the recency-weighted success rate, read off a sweep rather than
#: chosen: tools/campaigns/procedure_lifetime.py --sweep-decay measures, for
#: each value, how long a rule keeps firing after the world moves and how many
#: still-working rules get retired when it does not. At 0.99 a shift is
#: noticed in 999 firings against 21,999 with no decay, and one rule in
#: thirty-two is retired that should not have been — the same as 0.98, so the
#: value sits on a flat stretch and not on an edge. The table is in
#: docs/evidence/procedure_lifetime_halflife.json.
_RECENT_DECAY = 0.99

#: How much effective sample the recent rate needs before retirement listens
#: to it rather than the lifetime average. Below this the recent estimate is a
#: short run, and retiring on a short run is how a system forgets what works.
_RECENT_WEIGHT_FLOOR = 30.0


def _kind_accepts_value(kind: str, value: Any) -> bool:
    """Interpret the structural kinds shared by procedure backends.

    Unknown kinds remain nominal labels and preserve the historical presence
    semantics.  The closed structural kinds below are different: callers use
    them to compose executable values, so accepting the wrong Python shape
    would make the type field decorative.
    """

    if kind == "any":
        return True
    if kind == "integer":
        return type(value) is int
    if kind == "integer_sequence":
        return isinstance(value, (list, tuple)) and all(type(item) is int for item in value)
    if kind == "boolean":
        return type(value) is bool
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "mapping":
        return isinstance(value, Mapping)
    if kind == "sequence":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    return True


def _kinds_compose(produced: str, required: str) -> bool:
    """Whether an effect of one structural kind can satisfy a later read."""

    return produced == "any" or required == "any" or produced == required


class Backend(StrEnum):
    """Which learner made this, and therefore what executes it."""

    CHUNK = "chunk"  # core/cognition/impasse.py
    GENERALIZED_RULE = "rule"  # core/cognition/procedural_generalization.py
    MACRO = "macro"  # core/agency/skill_library.py
    DOING = "doing"  # core/cognition/an_action_she_composed.py
    HABIT = "habit"
    PLANNER = "planner"
    RLC = "rlc"  # core/learning/semantic_neural_composition.py
    TOOL = "tool"
    NEURAL = "neural"  # distilled into learned tissue


class Reversibility(StrEnum):
    """Whether the world can be put back. Never a number; never averaged."""

    REVERSIBLE = "reversible"
    COSTLY = "costly"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Precondition:
    """One thing that must hold. ``kind`` is the type, not the value."""

    key: str
    kind: str = "any"
    #: When set, the precondition also requires this exact value.
    equals: Any = None
    #: Whether the precondition is satisfied by ABSENCE. Distinct from a
    #: missing observation, which never satisfies anything.
    negated: bool = False

    def satisfied_by(self, state: Mapping[str, Any]) -> bool:
        if self.key not in state:
            return False  # never observed is never a match
        value = state[self.key]
        present = value is not None and value is not False
        if self.negated:
            return not present
        if not present:
            return False
        return _kind_accepts_value(self.kind, value) and (
            self.equals is None or value == self.equals
        )


@dataclass(frozen=True, slots=True)
class Effect:
    """One thing that becomes true. The half a planner composes on."""

    key: str
    kind: str = "any"
    value: Any = None

    def applied_to(self, state: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out[self.key] = True if self.value is None else self.value
        return out


@dataclass(frozen=True, slots=True)
class Signature:
    """What a procedure needs and what it leaves behind."""

    preconditions: tuple[Precondition, ...] = ()
    effects: tuple[Effect, ...] = ()

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(p.key for p in self.preconditions)

    def matches(self, state: Mapping[str, Any]) -> bool:
        return all(p.satisfied_by(state) for p in self.preconditions)

    def apply(self, state: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(state)
        for effect in self.effects:
            out = effect.applied_to(out)
        return out

    def follows(self, other: Signature) -> bool:
        """Whether this can run after ``other`` — its effects meet these needs."""
        produced = {effect.key: effect.kind for effect in other.effects}
        return (
            any(
                precondition.key in produced
                and _kinds_compose(produced[precondition.key], precondition.kind)
                for precondition in self.preconditions
            )
            or not self.preconditions
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProceduralValue:
    """What a procedure is worth, in one arithmetic every backend computes into.

    Keyword-only on purpose. ``ProceduralValue(0.9, 2.0, 0.1)`` reads the same
    before and after a field is added in the middle and means something else,
    and a success rate that silently becomes a cost is not a thing a test can
    be relied on to catch.

    ``value_when_it_works`` is in whatever unit the caller uses consistently —
    seconds saved, tokens saved, tasks completed. ``match_cost`` and
    ``risk_cost`` must be in the same unit or the subtraction is theatre, and
    :meth:`net` is the only number that decides retention.
    """

    p_success: float = 0.5
    #: The same rate weighted toward what happened lately. ``p_success`` is a
    #: lifetime average, and over a long life it cannot be moved: a rule right
    #: a million times needs about ninety thousand misses before the average
    #: crosses the retirement threshold, which is ninety thousand wrong
    #: firings. This is the estimate retirement actually asks.
    recent_success: float = 0.5
    #: Effective sample behind ``recent_success``, so a two-use estimate is
    #: not trusted the way a two-hundred-use one is.
    recent_weight: float = 0.0
    value_when_it_works: float = 0.0
    #: What one wrong firing costs. Without this term a rule that fails four
    #: times in five still "pays", because only its wins are priced — which is
    #: the utility problem chunking systems are known for, expressed as
    #: arithmetic the registry could not previously represent.
    cost_when_it_fails: float = 0.0
    match_cost: float = 0.0
    risk_cost: float = 0.0
    uses: int = 0
    successes: int = 0
    last_used: float = 0.0
    #: How far from its origin this has been shown to work: same instance,
    #: same family, structural analogue, unrelated domain. Card 186's tiers.
    transfer_tier: str = "same_instance"

    @property
    def rate_that_decides(self) -> float:
        """Which success rate retirement should use, and why.

        The recent rate, once enough uses stand behind it to be a measurement
        rather than a run of luck; the lifetime rate before that. Retirement
        asks whether a rule is paying now, and a lifetime average answers a
        different question.
        """
        if self.recent_weight < _RECENT_WEIGHT_FLOOR:
            return self.p_success
        # The optimistic reading of the recent evidence. Retirement is the
        # decision whose expensive mistake is the false one: a rule retired on
        # an unlucky run has to be impassed and compiled again, and the states
        # it covered go unhandled until then.
        from core.cognition.procedural_generalization import wilson_upper_bound

        trials = round(self.recent_weight)
        return wilson_upper_bound(round(self.recent_success * trials), trials)

    @property
    def net(self) -> float:
        """Expected value per use: what it wins, less what it loses and costs.

        ``match_cost`` and ``risk_cost`` are charged on every use — they are
        what holding the rule costs whatever it does. ``cost_when_it_fails``
        is charged only on the firings that miss, which is the term that lets
        a rule stop paying as its success rate falls.
        """
        rate = self.rate_that_decides
        return (
            rate * self.value_when_it_works
            - (1.0 - rate) * self.cost_when_it_fails
            - self.match_cost
            - self.risk_cost
        )

    @property
    def pays(self) -> bool:
        return self.net > 0.0

    def observed(self, *, success: bool, at: float, value: float | None = None) -> ProceduralValue:
        """Fold in one use. ``p_success`` becomes measured rather than assumed.

        A reported value is averaged over the uses that worked, not written
        over the top of the old one. Last-write-wins let a single lucky run
        restate what the whole rule is worth, which is not a measurement of
        anything and made the retirement threshold depend on which use came
        last. A value reported on a failure is ignored: the field is what the
        procedure is worth *when it works*.
        """
        uses = self.uses + 1
        successes = self.successes + (1 if success else 0)
        # Exponentially weighted, so the estimate has a horizon instead of a
        # memory. The weight saturates at 1/(1-alpha), which is what makes
        # rate_that_decides able to say "enough recent evidence".
        alpha = _RECENT_DECAY
        weight = self.recent_weight * alpha + 1.0
        recent = (
            self.recent_success * self.recent_weight * alpha + (1.0 if success else 0.0)
        ) / weight
        worth = self.value_when_it_works
        if success and value is not None:
            worth = (
                (self.value_when_it_works * self.successes + value) / successes
                if successes
                else value
            )
        return replace(
            self,
            uses=uses,
            successes=successes,
            p_success=successes / uses,
            recent_success=recent,
            recent_weight=weight,
            value_when_it_works=worth,
            last_used=at,
        )

    def wilson_floor(self, *, z: float = 1.96) -> float:
        """The conservative reading of ``p_success`` given how few uses there are."""
        from core.cognition.procedural_generalization import wilson_lower_bound

        return wilson_lower_bound(self.successes, self.uses, z=z) if self.uses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_success": self.p_success,
            "recent_success": self.recent_success,
            "recent_weight": self.recent_weight,
            "rate_that_decides": self.rate_that_decides,
            "wilson_floor": self.wilson_floor(),
            "value_when_it_works": self.value_when_it_works,
            "cost_when_it_fails": self.cost_when_it_fails,
            "match_cost": self.match_cost,
            "risk_cost": self.risk_cost,
            "net": self.net,
            "uses": self.uses,
            "successes": self.successes,
            "transfer_tier": self.transfer_tier,
            "last_used": self.last_used,
        }


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a procedure came from, so it can explain itself.

    Card 022's bar: any procedure produces a minimal reproducible account of
    why it exists. That is these six fields, and none is optional for a
    procedure a learner compiled — a rule that cannot say what it was compiled
    from cannot be audited when it fires wrongly.
    """

    learner: str
    impasse_type: str = ""
    support_keys: tuple[str, ...] = ()
    causal_events: tuple[int, ...] = ()
    counterexamples: tuple[str, ...] = ()
    rejected_conditions: tuple[str, ...] = ()
    #: Conditions kept without evidence that they gate anything: present in
    #: every run the rule was compiled from, and holding a different value
    #: each time. These are the ones a witness can drop.
    provisional_conditions: tuple[str, ...] = ()
    #: Conditions dropped, each naming the run that succeeded without it.
    #: Written "key<-witness", so a widening can be audited the same way a
    #: narrowing can.
    generalisations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner": self.learner,
            "impasse_type": self.impasse_type,
            "support_keys": list(self.support_keys),
            "causal_events": list(self.causal_events),
            "counterexamples": list(self.counterexamples),
            "rejected_conditions": list(self.rejected_conditions),
            "provisional_conditions": list(self.provisional_conditions),
            "generalisations": list(self.generalisations),
        }


@dataclass(frozen=True, slots=True)
class Procedure:
    """One learned way of doing something, priced in the common currency."""

    procedure_id: str
    name: str
    backend: Backend
    signature: Signature
    #: Whatever the backend needs to run it. Opaque here on purpose.
    program: Any = None
    value: ProceduralValue = field(default_factory=ProceduralValue)
    origin: Origin | None = None
    evidence: EvidencePacket | None = None
    reversibility: Reversibility = Reversibility.UNKNOWN
    #: Procedures this was composed from, innermost last.
    parts: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    retired: bool = False
    retired_because: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "backend": self.backend.value,
            "preconditions": [
                {"key": p.key, "kind": p.kind, "negated": p.negated}
                for p in self.signature.preconditions
            ],
            "effects": [{"key": e.key, "kind": e.kind} for e in self.signature.effects],
            "value": self.value.to_dict(),
            "origin": self.origin.to_dict() if self.origin else None,
            "reversibility": self.reversibility.value,
            "parts": list(self.parts),
            "retired": self.retired,
            "retired_because": self.retired_because,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


class ProcedureIndex:
    """Discrimination on precondition keys, so matching does not scan.

    Ten times the procedures must cost much less than ten times the match, or
    learning makes the agent slower. The index buckets by precondition key and
    compares only procedures that could possibly apply; a procedure with no
    preconditions is always a candidate and lives in its own bucket, which is
    where the remaining linear cost is and where it belongs.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, set[str]] = {}
        self._unconditional: set[str] = set()
        self._comparisons = 0

    def add(self, procedure: Procedure) -> None:
        keys = procedure.signature.keys
        if not keys:
            self._unconditional.add(procedure.procedure_id)
            return
        for key in keys:
            self._by_key.setdefault(key, set()).add(procedure.procedure_id)

    def remove(self, procedure: Procedure) -> None:
        self._unconditional.discard(procedure.procedure_id)
        for key in procedure.signature.keys:
            bucket = self._by_key.get(key)
            if bucket:
                bucket.discard(procedure.procedure_id)
                if not bucket:
                    del self._by_key[key]

    def candidates(self, state: Mapping[str, Any]) -> set[str]:
        """Procedure ids that could possibly match this state."""
        out = set(self._unconditional)
        for key in state:
            out |= self._by_key.get(key, set())
        self._comparisons += len(out)
        return out

    @property
    def comparisons(self) -> int:
        return self._comparisons


class ProcedureRegistry:
    #: Seconds between pulls of the other learners' stores. A ranking wants
    #: the current stores; scanning them on every match would cost more than
    #: the ranking saves.
    _REFRESH_SECONDS: float = 30.0

    """Every learned procedure, priced and matched through one door."""

    def __init__(self, *, max_procedures: int = 20_000, clock=time.time) -> None:
        self._lock = checked_lock("core.cognition.procedure.ProcedureRegistry", reentrant=True)
        self._procedures: dict[str, Procedure] = {}
        self._interned: dict[tuple[Backend, str], tuple[str, str]] = {}
        self._intern_key_by_procedure: dict[str, tuple[Backend, str]] = {}
        self._index = ProcedureIndex()
        self._counter = 0
        self._max = int(max_procedures)
        self._clock = clock
        self._retired = 0

    # ── registration ──────────────────────────────────────────────────
        #: What pulls the other learners' stores in, and when it last ran.
        self._refresh: Callable[[], Any] | None = None
        self._refreshed_at: float = 0.0
        self._refresh_failed: str = ""

    def register(
        self,
        name: str,
        backend: Backend,
        signature: Signature,
        *,
        program: Any = None,
        value: ProceduralValue | None = None,
        origin: Origin | None = None,
        evidence: EvidencePacket | None = None,
        reversibility: Reversibility = Reversibility.UNKNOWN,
        parts: Sequence[str] = (),
    ) -> Procedure:
        """Add a procedure. A compiled one must be able to say where it came from."""
        if backend in (Backend.CHUNK, Backend.GENERALIZED_RULE, Backend.RLC) and origin is None:
            raise ValueError(
                f"{name!r} was compiled by {backend.value} and names no origin; a rule that "
                "cannot say what it was compiled from cannot be audited when it fires wrongly"
            )
        with self._lock:
            self._counter += 1
            procedure = Procedure(
                procedure_id=f"p{self._counter}",
                name=name,
                backend=backend,
                signature=signature,
                program=program,
                value=value or ProceduralValue(),
                origin=origin,
                evidence=evidence,
                reversibility=reversibility,
                parts=tuple(parts),
                created_at=self._clock(),
            )
            self._procedures[procedure.procedure_id] = procedure
            self._index.add(procedure)
            self._evict_locked()
            return procedure

    def intern(
        self,
        identity: str,
        contract_sha256: str,
        name: str,
        backend: Backend,
        signature: Signature,
        *,
        program: Any = None,
        value: ProceduralValue | None = None,
        origin: Origin | None = None,
        evidence: EvidencePacket | None = None,
        reversibility: Reversibility = Reversibility.UNKNOWN,
        parts: Sequence[str] = (),
    ) -> Procedure:
        """Register one executable contract once and accumulate its provenance.

        ``identity`` names equivalence in the backend's own vocabulary;
        ``contract_sha256`` prevents two different executions from claiming
        that name. Re-observing the same contract fuses independent evidence
        sources but does not count another use or create another match entry.
        """

        if not identity or not (
            isinstance(contract_sha256, str)
            and len(contract_sha256) == 64
            and all(character in "0123456789abcdef" for character in contract_sha256)
        ):
            raise ValueError("interned procedure identity or contract is invalid")
        key = (backend, identity)
        with self._lock:
            prior = self._interned.get(key)
            if prior is not None:
                procedure_id, prior_contract = prior
                if prior_contract != contract_sha256:
                    raise ValueError("interned procedure identity names a different contract")
                existing = self._procedures.get(procedure_id)
                if existing is None:
                    raise RuntimeError("interned procedure index is inconsistent")
                if evidence is not None:
                    from core.evidence.packet import fuse

                    combined = (
                        fuse((existing.evidence, evidence))
                        if existing.evidence is not None
                        else evidence
                    )
                    existing = replace(existing, evidence=combined)
                    self._procedures[procedure_id] = existing
                return existing
            procedure = self.register(
                name,
                backend,
                signature,
                program=program,
                value=value,
                origin=origin,
                evidence=evidence,
                reversibility=reversibility,
                parts=parts,
            )
            self._interned[key] = (procedure.procedure_id, contract_sha256)
            self._intern_key_by_procedure[procedure.procedure_id] = key
            return procedure

    def get(self, procedure_id: str) -> Procedure | None:
        with self._lock:
            return self._procedures.get(procedure_id)

    def _drop_interned_locked(self, procedure_id: str) -> None:
        key = self._intern_key_by_procedure.pop(procedure_id, None)
        if key is not None:
            self._interned.pop(key, None)

    # ── matching ──────────────────────────────────────────────────────

    def keep_current_with(self, refresh: Callable[[], Any] | None) -> None:
        """Name what pulls the learners' stores in before a ranking is asked for.

        The registry cannot go and get them itself — the adapters import this
        module, so this module must not import the adapters. Naming the
        refresher here keeps the direction right and gives the economy one
        place where it is kept current, instead of none.
        """
        self._refresh = refresh

    def _refresh_if_stale(self) -> None:
        """Pull the learners in, at most every ``_REFRESH_SECONDS``.

        Outside the lock: the refresher calls back into ``register``, and the
        lock is not reentrant.
        """
        if self._refresh is None:
            return
        now = self._clock()
        if now - self._refreshed_at < self._REFRESH_SECONDS:
            return
        self._refreshed_at = now
        try:
            self._refresh()
        except Exception as exc:  # noqa: BLE001 - a stale ranking beats no ranking
            # Held rather than logged. A refresh that keeps failing means the
            # ranking silently covers one backend, and that belongs in the
            # report somebody reads, not in a debug line nobody does.
            self._refresh_failed = f"{type(exc).__name__}: {exc}"
        else:
            self._refresh_failed = ""

    def match(self, state: Mapping[str, Any], *, limit: int = 10) -> list[Procedure]:
        """The procedures that apply here, best net value first.

        Backends compete directly: a chunk and a generalized rule are ranked by
        the same number. That was true of the arithmetic and false of the
        registry, because nothing ever put the other backends' procedures in
        it: the adapters had no importer anywhere in production while the
        claim ladder cited them as the wired evidence. The refresh is what
        makes the sentence above describe the running system.
        """
        self._refresh_if_stale()
        with self._lock:
            candidates = self._index.candidates(state)
            applicable = [
                p
                for pid in candidates
                if (p := self._procedures.get(pid)) is not None
                and not p.retired
                and p.signature.matches(state)
            ]
            applicable.sort(key=lambda p: p.value.net, reverse=True)
            return applicable[:limit]

    # ── value and lifecycle ───────────────────────────────────────────

    def record_use(
        self, procedure_id: str, *, success: bool, value: float | None = None
    ) -> Procedure | None:
        """Fold one use into the value, and retire it if it stopped paying."""
        with self._lock:
            procedure = self._procedures.get(procedure_id)
            if procedure is None:
                return None
            updated = replace(
                procedure,
                value=procedure.value.observed(success=success, at=self._clock(), value=value),
            )
            self._procedures[procedure_id] = updated
            return updated

    def prune(self, *, min_uses: int = 3) -> list[Procedure]:
        """Retire everything whose net value has turned negative.

        ``min_uses`` exists because one failure is not evidence a procedure is
        worthless, and retiring on the first miss is how a system forgets
        things that work. It is a sample-size floor, not a grace period.
        """
        with self._lock:
            retired = []
            for pid, procedure in list(self._procedures.items()):
                if procedure.retired or procedure.value.uses < min_uses:
                    continue
                if procedure.value.pays:
                    continue
                gone = replace(
                    procedure,
                    retired=True,
                    retired_because=(
                        f"net {procedure.value.net:.4g} after {procedure.value.uses} uses"
                    ),
                )
                self._procedures[pid] = gone
                self._index.remove(gone)
                self._drop_interned_locked(pid)
                self._retired += 1
                retired.append(gone)
            return retired

    def specialise(
        self, procedure_id: str, extra: Precondition, *, counterexample: str = ""
    ) -> Procedure:
        """Narrow a procedure after a counterexample, keeping the original.

        A rule that fired wrongly is not deleted: the case it got wrong is
        added as a condition and the original stays available for the cases it
        still covers. That is the lifecycle card 059 asks for, and it is why
        coverage can improve without the rule count running away — the
        specialised child inherits the parent's uses and must earn its own.
        """
        with self._lock:
            parent = self._procedures[procedure_id]
            origin = parent.origin or Origin(learner=parent.backend.value)
            child = self.register(
                f"{parent.name}+{extra.key}",
                parent.backend,
                Signature(
                    preconditions=(*parent.signature.preconditions, extra),
                    effects=parent.signature.effects,
                ),
                program=parent.program,
                value=ProceduralValue(
                    p_success=parent.value.p_success,
                    value_when_it_works=parent.value.value_when_it_works,
                    cost_when_it_fails=parent.value.cost_when_it_fails,
                    match_cost=parent.value.match_cost,
                    risk_cost=parent.value.risk_cost,
                    transfer_tier=parent.value.transfer_tier,
                ),
                origin=replace(
                    origin,
                    counterexamples=(*origin.counterexamples, counterexample)
                    if counterexample
                    else origin.counterexamples,
                ),
                evidence=parent.evidence,
                reversibility=parent.reversibility,
                parts=(parent.procedure_id,),
            )
            return child

    def generalise(self, procedure_id: str, drop: str, *, witness: str) -> Procedure | None:
        """Widen a procedure after a run succeeded without one of its conditions.

        The counterpart of :meth:`specialise`, and the reason it has to exist:
        a registry that can only ever add conditions gets monotonically more
        specific over a lifetime, which is how a compiler that learned the room
        instead of the task never finds out. Success traces alone cannot tell a
        needed read from an incidental one — both are present every time — so
        the evidence that drops a condition is a run that did without it.

        ``witness`` names that run. There is no unwitnessed generalisation:
        dropping a condition because it looks incidental is guessing, and the
        guess fires on every state the condition was keeping it out of.
        """
        if not witness:
            raise ValueError(
                f"generalising {procedure_id!r} by dropping {drop!r} needs a run that "
                "succeeded without it; a condition dropped on suspicion widens the "
                "rule over exactly the states it was excluding"
            )
        with self._lock:
            parent = self._procedures.get(procedure_id)
            if parent is None:
                return None
            kept = tuple(p for p in parent.signature.preconditions if p.key != drop)
            if len(kept) == len(parent.signature.preconditions):
                return None
            origin = parent.origin or Origin(learner=parent.backend.value)
            return self.register(
                f"{parent.name}-{drop}",
                parent.backend,
                Signature(preconditions=kept, effects=parent.signature.effects),
                program=parent.program,
                value=ProceduralValue(
                    p_success=parent.value.p_success,
                    value_when_it_works=parent.value.value_when_it_works,
                    cost_when_it_fails=parent.value.cost_when_it_fails,
                    # One fewer condition to check is one less to pay for.
                    match_cost=parent.value.match_cost
                    * (len(kept) / len(parent.signature.preconditions)),
                    risk_cost=parent.value.risk_cost,
                    transfer_tier=parent.value.transfer_tier,
                ),
                origin=replace(
                    origin,
                    support_keys=tuple(k for k in origin.support_keys if k != drop),
                    generalisations=(*origin.generalisations, f"{drop}<-{witness}"),
                ),
                evidence=parent.evidence,
                reversibility=parent.reversibility,
                parts=(parent.procedure_id,),
            )

    def merge(self, keep_id: str, absorb_id: str) -> Procedure:
        """Two procedures turned out to be one. Evidence adds; sources do not double count."""
        from core.evidence.packet import fuse

        with self._lock:
            keep, absorb = self._procedures[keep_id], self._procedures[absorb_id]
            evidence = None
            if keep.evidence and absorb.evidence:
                evidence = fuse([keep.evidence, absorb.evidence])
            elif keep.evidence or absorb.evidence:
                evidence = keep.evidence or absorb.evidence
            merged = replace(
                keep,
                value=replace(
                    keep.value,
                    uses=keep.value.uses + absorb.value.uses,
                    successes=keep.value.successes + absorb.value.successes,
                    p_success=(
                        (keep.value.successes + absorb.value.successes)
                        / max(1, keep.value.uses + absorb.value.uses)
                    ),
                ),
                evidence=evidence,
                parts=(*keep.parts, absorb_id),
            )
            self._procedures[keep_id] = merged
            self._procedures[absorb_id] = replace(
                absorb, retired=True, retired_because=f"merged into {keep_id}"
            )
            self._index.remove(absorb)
            self._drop_interned_locked(absorb_id)
            return merged

    def _evict_locked(self) -> None:
        if len(self._procedures) <= self._max:
            return
        worst = sorted(
            (p for p in self._procedures.values() if p.retired),
            key=lambda p: p.created_at,
        )
        for procedure in worst[: len(self._procedures) - self._max]:
            del self._procedures[procedure.procedure_id]
            self._drop_interned_locked(procedure.procedure_id)

    # ── reporting ─────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        with self._lock:
            live = [p for p in self._procedures.values() if not p.retired]
            by_backend: dict[str, int] = {}
            for procedure in live:
                by_backend[procedure.backend.value] = by_backend.get(procedure.backend.value, 0) + 1
            composed = [p for p in live if len(p.parts) > 1]
            cross_backend = [
                p
                for p in composed
                if len({self._procedures[q].backend for q in p.parts if q in self._procedures}) > 1
            ]
            return {
                "procedures": len(live),
                "interned": len(self._interned),
                "retired": self._retired,
                "by_backend": dict(sorted(by_backend.items())),
                "backends_competing": len(by_backend),
                # One backend competing is one backend, whatever the arithmetic
                # says it could do. This is the reading that caught the
                # adapters having no importer at all.
                "learners_installed": self._refresh is not None,
                "refresh_failed": self._refresh_failed,
                "composed": len(composed),
                "composed_across_backends": len(cross_backend),
                "index_comparisons": self._index.comparisons,
                "mean_net": (sum(p.value.net for p in live) / len(live)) if live else 0.0,
                "by_transfer_tier": {
                    tier: sum(1 for p in live if p.value.transfer_tier == tier)
                    for tier in sorted({p.value.transfer_tier for p in live})
                },
            }

    def procedures(self) -> list[Procedure]:
        with self._lock:
            return list(self._procedures.values())


def compose(
    registry: ProcedureRegistry,
    parts: Sequence[Procedure],
    *,
    name: str = "",
    backend: Backend = Backend.PLANNER,
) -> Procedure:
    """Chain procedures into one, computing the combined signature.

    The composed preconditions are the first part's, plus anything a later
    part needs that no earlier part produced. The composed effects are the
    union, later winning. The composed value is multiplicative in success and
    additive in cost, which is the honest arithmetic: a three-step procedure
    fails if any step does, and costs what all three cost.

    Parts from different backends compose exactly as parts from the same one.
    That is card 199's bar, and it is the reason the signature is typed.
    """
    if not parts:
        raise ValueError("nothing to compose")
    produced: dict[str, str] = {}
    preconditions: list[Precondition] = []
    effects: list[Effect] = []
    p_success = 1.0
    total_value = 0.0
    match_cost = 0.0
    risk_cost = 0.0
    reversibility = Reversibility.REVERSIBLE
    for part in parts:
        for precondition in part.signature.preconditions:
            if precondition.key not in produced:
                preconditions.append(precondition)
            elif not _kinds_compose(produced[precondition.key], precondition.kind):
                raise ValueError(
                    f"procedure composition writes {precondition.key!r} as "
                    f"{produced[precondition.key]!r} before it is read as "
                    f"{precondition.kind!r}"
                )
        for effect in part.signature.effects:
            produced[effect.key] = effect.kind
            effects = [e for e in effects if e.key != effect.key] + [effect]
        p_success *= part.value.p_success
        total_value += part.value.value_when_it_works
        match_cost += part.value.match_cost
        risk_cost += part.value.risk_cost
        if part.reversibility is Reversibility.IRREVERSIBLE:
            reversibility = Reversibility.IRREVERSIBLE
        elif (
            part.reversibility is Reversibility.COSTLY and reversibility is Reversibility.REVERSIBLE
        ):
            reversibility = Reversibility.COSTLY
        elif (
            part.reversibility is Reversibility.UNKNOWN
            and reversibility is not Reversibility.IRREVERSIBLE
        ):
            reversibility = Reversibility.UNKNOWN

    return registry.register(
        name or " then ".join(p.name for p in parts),
        backend,
        Signature(preconditions=tuple(preconditions), effects=tuple(effects)),
        program=tuple(p.procedure_id for p in parts),
        value=ProceduralValue(
            p_success=p_success,
            value_when_it_works=total_value,
            match_cost=match_cost,
            risk_cost=risk_cost,
        ),
        origin=Origin(learner="compose", support_keys=tuple(sorted(produced))),
        reversibility=reversibility,
        parts=tuple(p.procedure_id for p in parts),
    )


_registry_lock = checked_lock("core.cognition.procedure.singleton")
_registry: ProcedureRegistry | None = None


def get_procedure_registry() -> ProcedureRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ProcedureRegistry()
        return _registry


def reset_procedure_registry_for_test(**kwargs: Any) -> ProcedureRegistry:
    global _registry
    with _registry_lock:
        _registry = ProcedureRegistry(**kwargs)
        return _registry
