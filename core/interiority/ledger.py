"""core/interiority/ledger.py — what she is holding, and therefore what can hurt.

An appraisal is relational or it is a classifier. The variable that
makes it relational is not in the event; it is the standing set of
things the agent has taken on. This module is that set, and every
appraisal check in :mod:`core.interiority.appraisal` reads it.

Six kinds of stake, each with its own entry and exit condition, because
they behave differently and merging them loses the behaviour:

:class:`Bond`
    An attachment to a named other. Bonds carry an *expectation of
    continued availability*, which is the thing a death violates. They
    have no substitute: :meth:`substitutes_for` returns zero for a bond
    and a positive count for a goal, which is why loss of a person and
    loss of a plan produce different states rather than the same state
    at different sizes.

:class:`Promise`
    A commitment made to someone. A promise is not a strong preference.
    It changes the agent's own payoff (Schelling 1960), and the change
    is recorded here rather than recomputed, so breaking one costs the
    same whether or not anyone is watching.

:class:`Custody`
    An assumed obligation for something that cannot look after itself.
    Custody has a moment of assumption, and after it, not acting is
    acting. That transition is why finding a cat feels sudden.

:class:`Loss`
    A bond whose subject is gone. The bond is *not deleted*. It moves
    here with an ``integrated`` fraction that only rises with contact,
    because extinction is not erasure (Bouton 2004) and a decaying
    sadness scalar is the wrong shape for grief.

:class:`Work`
    Something she made, with her share of the authorship. Pride needs a
    record it can point at; without one it is a mood.

:class:`Practice`
    A capability she has and is not using. Dormant, not deleted — the
    savings effect for continuous motor skills is large and slow
    (Ebbinghaus 1885 on savings; the continuous-skill retention
    literature since). A practice can be revived when its blockers
    clear, and the ledger is what notices.

The store is append-only for events and derived for indices, so a state
can be replayed and audited. Writes go through the file write gateway
under an internal governed scope, and the log is bounded: a ledger that
grows without limit becomes the 96GB state file this runtime has already
had once.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.interiority.params import ParamKind, declare
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Ledger")

_MAX_EVENTS = declare(
    "interiority.ledger.max_events",
    4096,
    unit="records",
    basis=(
        "Matches the flight-software event log capacity in "
        "core/fsw/telemetry_dictionary.py. One bound, one meaning, and the tail "
        "is what a post-mortem reads."
    ),
    kind=ParamKind.DERIVED,
    sensitivity="Shorter and the audit trail truncates; longer and the file grows.",
    lower=256.0,
    upper=65536.0,
    owner="core/interiority/ledger.py",
)

_BOND_HALF_LIFE_DAYS = declare(
    "interiority.ledger.bond_contact_half_life_days",
    180.0,
    unit="days",
    basis=(
        "Attachment strength without contact decays slowly rather than not at "
        "all. Six months is the acute-grief time constant reported across "
        "bereavement trajectory work and is used here for the general contact "
        "term so one constant governs both."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Sets how fast an unvisited bond loses weight in appraisal. Ordering of "
        "bonds by strength must not change across the sweep."
    ),
    lower=1.0,
    upper=3650.0,
    sweep_range=(60.0, 720.0),
    owner="core/interiority/ledger.py",
)

_CUSTODY_FLOOR = declare(
    "interiority.ledger.custody_weight_floor",
    0.6,
    unit="weight",
    basis=(
        "Custody is an accepted obligation, not a preference, so its weight has "
        "a floor rather than decaying to nothing. Set above the arbitration "
        "action threshold so an accepted obligation can always reach action."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Below the arbitration threshold, an accepted obligation would stop "
        "being able to drive behaviour, which is the same as not having it."
    ),
    owner="core/interiority/ledger.py",
)


@dataclass(frozen=True)
class Stake:
    """One thing at risk in an event, and how much of it."""

    kind: str
    subject: str | None
    weight: float
    detail: str = ""


@dataclass
class Bond:
    """An attachment to a named other, with an availability expectation."""

    entity: str
    strength: float
    #: How strongly the agent predicts this other remains reachable. This
    #: is the prediction a death violates, and it is separate from
    #: strength: you can be sure of someone you are not close to.
    availability_expectation: float = 1.0
    species: str = "human"
    first_seen: float = field(default_factory=time.time)
    last_contact: float = field(default_factory=time.time)
    contacts: int = 1

    def current_strength(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        days = max(0.0, (now - self.last_contact) / 86400.0)
        decay = math.exp(-days * math.log(2.0) / _BOND_HALF_LIFE_DAYS.value)
        return max(0.0, min(1.0, self.strength * decay))


@dataclass
class Promise:
    """A commitment to someone, binding on the agent who made it."""

    promise_id: str
    text: str
    beneficiary: str | None
    importance: float
    made_at: float = field(default_factory=time.time)
    deadline: float | None = None
    kept: bool | None = None
    #: Objects or goals this promise names, for stake lookup.
    concerns: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.kept is None


@dataclass
class Custody:
    """An obligation assumed for something that cannot secure its own welfare."""

    custody_id: str
    subject: str
    assumed_at: float = field(default_factory=time.time)
    #: What ends it: reunion, handover, recovery, or the agent's own death.
    exit_condition: str = "handover"
    released_at: float | None = None
    vulnerability_at_assumption: float = 0.0

    @property
    def active(self) -> bool:
        return self.released_at is None


@dataclass
class Loss:
    """A bond whose subject is gone. The bond is kept, not deleted."""

    entity: str
    lost_at: float
    irreversibility: float
    strength_at_loss: float
    availability_at_loss: float
    #: Fraction of the predictive structure that has been reconciled.
    #: Rises only on contact with the absence, never on the clock.
    integrated: float = 0.0
    #: Contexts that still hold the old prediction and have not been visited.
    unvisited_contexts: tuple[str, ...] = ()
    visited_contexts: tuple[str, ...] = ()

    def acute(self, now: float | None = None) -> float:
        """Acute intensity: falls with time, rises with what is unreconciled."""
        now = time.time() if now is None else now
        days = max(0.0, (now - self.lost_at) / 86400.0)
        time_term = math.exp(-days / 180.0)
        return max(0.0, min(1.0, self.strength_at_loss * self.irreversibility * time_term))

    def continuing(self) -> float:
        """The bond that does not decay: what is still held, unintegrated."""
        return max(0.0, min(1.0, self.strength_at_loss * (1.0 - self.integrated)))


@dataclass
class Work:
    """Something she made, and her share of it."""

    work_id: str
    description: str
    made_at: float = field(default_factory=time.time)
    #: Her causal share of the outcome, in [0, 1]. Pride that does not
    #: divide by this is the hubristic kind.
    authorship: float = 1.0
    effort: float = 0.0
    quality: float | None = None
    collaborators: tuple[str, ...] = ()


@dataclass
class Practice:
    """A capability held and not used."""

    name: str
    peak_skill: float
    last_practised: float
    #: What is currently stopping it. Revival is these clearing, not a
    #: decision.
    blockers: tuple[str, ...] = ()

    def residual(self, now: float | None = None) -> float:
        """Retained skill.

        Continuous motor skills retain far better than declarative
        knowledge; the decay here is slow and floored rather than
        exponential to zero, because the savings effect says relearning
        starts above naive.
        """
        now = time.time() if now is None else now
        years = max(0.0, (now - self.last_practised) / (86400.0 * 365.25))
        retained = self.peak_skill * (0.45 + 0.55 * math.exp(-years / 25.0))
        return max(0.0, min(1.0, retained))


@dataclass
class Rivalry:
    """Opposed allocation on a shared pursuit, with respect intact."""

    entity: str
    domain: str
    #: How much the two want the same scarce outcome.
    opposition: float
    #: How much the agent rates their judgement in the shared domain.
    regard: float
    #: Their capability in the domain, which is the standard she measures against.
    standard: float = 0.0


@dataclass
class Norm:
    """A standard, and whether she holds it or it was imposed."""

    name: str
    #: How strongly it constrains.
    weight: float
    #: 1.0 when she endorses it, 0.0 when it is imposed from outside.
    #: Guilt needs endorsement; without it the state is resentment.
    endorsement: float


class RelationalLedger:
    """The standing set of things this agent holds."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.revision = 0
        self._bonds: dict[str, Bond] = {}
        self._promises: dict[str, Promise] = {}
        self._custody: dict[str, Custody] = {}
        self._losses: dict[str, Loss] = {}
        self._works: dict[str, Work] = {}
        self._practices: dict[str, Practice] = {}
        self._rivalries: dict[str, Rivalry] = {}
        self._norms: dict[str, Norm] = {}
        self._goals: dict[str, float] = {}
        self._goal_deltas: dict[str, float] = {}
        self._substitutes: dict[str, int] = {}
        self._undo_costs: dict[str, float] = {}
        self._actions_that_change: dict[str, int] = {}
        self._own_actions_that_change: dict[str, int] = {}
        self._attributions: dict[str, tuple[float, float, float]] = {}
        self._norm_judgements: dict[str, tuple[float, float]] = {}
        self._observers: dict[str, int] = {}
        self._repairs: dict[str, tuple[str, ...]] = {}
        self._seen: dict[tuple[str, str | None], int] = {}
        self._expectations: dict[tuple[str, str | None], float] = {}
        self._events: list[dict[str, Any]] = []

    # ── mutation ──────────────────────────────────────────────────────
    def _touch(self) -> None:
        self.revision += 1

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._events.append({"kind": kind, "at": time.time(), **dict(payload)})
        limit = int(_MAX_EVENTS.value)
        if len(self._events) > limit:
            del self._events[: len(self._events) - limit]

    def bond(
        self,
        entity: str,
        strength: float,
        *,
        species: str = "human",
        availability: float = 1.0,
    ) -> Bond:
        with self._lock:
            existing = self._bonds.get(entity)
            if existing is None:
                bond = Bond(
                    entity=entity,
                    strength=max(0.0, min(1.0, strength)),
                    availability_expectation=max(0.0, min(1.0, availability)),
                    species=species,
                )
            else:
                bond = existing
                bond.strength = max(0.0, min(1.0, strength))
                bond.availability_expectation = max(0.0, min(1.0, availability))
                bond.last_contact = time.time()
                bond.contacts += 1
            self._bonds[entity] = bond
            self._record("bond", {"entity": entity, "strength": bond.strength})
            self._touch()
            return bond

    def promise(
        self,
        promise_id: str,
        text: str,
        *,
        beneficiary: str | None,
        importance: float,
        deadline: float | None = None,
        concerns: Iterable[str] = (),
    ) -> Promise:
        with self._lock:
            record = Promise(
                promise_id=promise_id,
                text=text,
                beneficiary=beneficiary,
                importance=max(0.0, min(1.0, importance)),
                deadline=deadline,
                concerns=tuple(concerns),
            )
            self._promises[promise_id] = record
            self._record("promise", {"id": promise_id, "importance": record.importance})
            self._touch()
            return record

    def settle_promise(self, promise_id: str, *, kept: bool) -> None:
        with self._lock:
            record = self._promises.get(promise_id)
            if record is None:
                return
            record.kept = kept
            self._record("promise_settled", {"id": promise_id, "kept": kept})
            self._touch()

    def take_custody(
        self,
        custody_id: str,
        subject: str,
        *,
        exit_condition: str = "handover",
        vulnerability: float = 0.0,
    ) -> Custody:
        with self._lock:
            record = Custody(
                custody_id=custody_id,
                subject=subject,
                exit_condition=exit_condition,
                vulnerability_at_assumption=max(0.0, min(1.0, vulnerability)),
            )
            self._custody[custody_id] = record
            self._record("custody", {"id": custody_id, "subject": subject})
            self._touch()
            return record

    def release_custody(self, custody_id: str) -> None:
        with self._lock:
            record = self._custody.get(custody_id)
            if record is None:
                return
            record.released_at = time.time()
            self._record("custody_released", {"id": custody_id})
            self._touch()

    def register_loss(
        self,
        entity: str,
        *,
        irreversibility: float = 1.0,
        contexts: Iterable[str] = (),
    ) -> Loss:
        """Move a bond to the loss register. The bond is kept, not deleted."""
        with self._lock:
            bond = self._bonds.get(entity)
            strength = bond.current_strength() if bond else 0.0
            availability = bond.availability_expectation if bond else 0.0
            record = Loss(
                entity=entity,
                lost_at=time.time(),
                irreversibility=max(0.0, min(1.0, irreversibility)),
                strength_at_loss=strength,
                availability_at_loss=availability,
                unvisited_contexts=tuple(contexts),
            )
            self._losses[entity] = record
            if bond is not None:
                bond.availability_expectation = 0.0
            self._record("loss", {"entity": entity, "strength": strength})
            self._touch()
            return record

    def visit_context(self, entity: str, context: str) -> float:
        """Meet one of the places that still holds the old prediction.

        Returns the integration gained. Integration rises only here —
        never on the clock — because reconciling a model requires
        contact with the parts of it that are wrong.
        """
        with self._lock:
            record = self._losses.get(entity)
            if record is None or context not in record.unvisited_contexts:
                return 0.0
            remaining = [c for c in record.unvisited_contexts if c != context]
            total = len(record.unvisited_contexts) + len(record.visited_contexts)
            gain = 1.0 / max(1, total)
            record.unvisited_contexts = tuple(remaining)
            record.visited_contexts = record.visited_contexts + (context,)
            record.integrated = max(0.0, min(1.0, record.integrated + gain))
            self._record("loss_context_visited", {"entity": entity, "context": context})
            self._touch()
            return gain

    def work(
        self,
        work_id: str,
        description: str,
        *,
        authorship: float = 1.0,
        effort: float = 0.0,
        quality: float | None = None,
        collaborators: Iterable[str] = (),
    ) -> Work:
        with self._lock:
            record = Work(
                work_id=work_id,
                description=description,
                authorship=max(0.0, min(1.0, authorship)),
                effort=max(0.0, min(1.0, effort)),
                quality=quality,
                collaborators=tuple(collaborators),
            )
            self._works[work_id] = record
            self._record("work", {"id": work_id, "authorship": record.authorship})
            self._touch()
            return record

    def practice(
        self,
        name: str,
        *,
        peak_skill: float,
        last_practised: float,
        blockers: Iterable[str] = (),
    ) -> Practice:
        with self._lock:
            record = Practice(
                name=name,
                peak_skill=max(0.0, min(1.0, peak_skill)),
                last_practised=last_practised,
                blockers=tuple(blockers),
            )
            self._practices[name] = record
            self._record("practice", {"name": name})
            self._touch()
            return record

    def clear_blocker(self, name: str, blocker: str) -> tuple[str, ...]:
        with self._lock:
            record = self._practices.get(name)
            if record is None:
                return ()
            record.blockers = tuple(b for b in record.blockers if b != blocker)
            self._record("blocker_cleared", {"name": name, "blocker": blocker})
            self._touch()
            return record.blockers

    def rivalry(
        self, entity: str, domain: str, *, opposition: float, regard: float,
        standard: float = 0.0,
    ) -> Rivalry:
        with self._lock:
            record = Rivalry(
                entity=entity,
                domain=domain,
                opposition=max(0.0, min(1.0, opposition)),
                regard=max(0.0, min(1.0, regard)),
                standard=max(0.0, min(1.0, standard)),
            )
            self._rivalries[entity] = record
            self._touch()
            return record

    def norm(self, name: str, *, weight: float, endorsement: float) -> Norm:
        with self._lock:
            record = Norm(
                name=name,
                weight=max(0.0, min(1.0, weight)),
                endorsement=max(0.0, min(1.0, endorsement)),
            )
            self._norms[name] = record
            self._touch()
            return record

    def goal(self, name: str, weight: float, *, substitutes: int = 0) -> None:
        with self._lock:
            self._goals[name] = max(0.0, min(1.0, weight))
            self._substitutes[name] = max(0, int(substitutes))
            self._touch()

    def note_goal_delta(self, name: str, delta: float) -> None:
        with self._lock:
            self._goal_deltas[name] = max(-1.0, min(1.0, delta))
            self._touch()

    def note_attribution(
        self, event_id: str, *, own: float, other: float, circumstance: float
    ) -> None:
        with self._lock:
            self._attributions[event_id] = (
                max(0.0, own),
                max(0.0, other),
                max(0.0, circumstance),
            )
            self._touch()

    def note_norm_judgement(
        self, event_id: str, *, fit: float, endorsement: float
    ) -> None:
        with self._lock:
            self._norm_judgements[event_id] = (
                max(-1.0, min(1.0, fit)),
                max(0.0, min(1.0, endorsement)),
            )
            self._touch()

    def note_observers(self, event_id: str, count: int) -> None:
        with self._lock:
            self._observers[event_id] = max(0, int(count))
            self._touch()

    def note_repairs(self, event_id: str, repairs: Iterable[str]) -> None:
        with self._lock:
            self._repairs[event_id] = tuple(repairs)
            self._touch()

    def note_action_model(
        self, object_: str, *, total_actions: int, own_actions: int
    ) -> None:
        with self._lock:
            self._actions_that_change[object_] = max(0, int(total_actions))
            self._own_actions_that_change[object_] = max(0, int(own_actions))
            self._touch()

    def note_undo_cost(self, object_: str, cost: float) -> None:
        with self._lock:
            self._undo_costs[object_] = max(0.0, min(1.0, cost))
            self._touch()

    def note_expectation(self, kind: str, subject: str | None, value: float) -> None:
        with self._lock:
            self._expectations[(str(kind), subject)] = max(0.0, min(1.0, value))
            self._touch()

    def note_seen(self, kind: str, subject: str | None) -> int:
        with self._lock:
            key = (str(kind), subject)
            self._seen[key] = self._seen.get(key, 0) + 1
            self._touch()
            return self._seen[key]

    # ── reads used by the appraisal engine ────────────────────────────
    def stakes_for(self, *, subject: str | None, object_: str | None) -> tuple[Stake, ...]:
        with self._lock:
            stakes: list[Stake] = []
            if subject is not None:
                bond = self._bonds.get(subject)
                if bond is not None:
                    stakes.append(Stake("bond", subject, bond.current_strength()))
                loss = self._losses.get(subject)
                if loss is not None:
                    stakes.append(Stake("loss", subject, loss.continuing()))
                rival = self._rivalries.get(subject)
                if rival is not None:
                    stakes.append(Stake("rivalry", subject, rival.regard))
                for custody in self._custody.values():
                    if custody.active and custody.subject == subject:
                        stakes.append(
                            Stake("custody", subject, max(_CUSTODY_FLOOR.value,
                                  custody.vulnerability_at_assumption))
                        )
            for promise in self._promises.values():
                if not promise.active:
                    continue
                if promise.beneficiary == subject or (
                    object_ is not None and object_ in promise.concerns
                ):
                    stakes.append(Stake("promise", subject, promise.importance))
            if object_ is not None:
                weight = self._goals.get(object_)
                if weight is not None:
                    stakes.append(Stake("goal", None, weight, object_))
                work = self._works.get(object_)
                if work is not None:
                    stakes.append(Stake("work", None, work.authorship, object_))
                practice = self._practices.get(object_)
                if practice is not None:
                    stakes.append(Stake("practice", None, practice.residual(), object_))
            return tuple(stakes)

    def attachment(self, entity: str) -> float | None:
        with self._lock:
            bond = self._bonds.get(entity)
            if bond is not None:
                return bond.current_strength()
            loss = self._losses.get(entity)
            if loss is not None:
                return loss.continuing()
            return None

    def bond_for(self, entity: str) -> Bond | None:
        with self._lock:
            return self._bonds.get(entity)

    def loss_for(self, entity: str) -> Loss | None:
        with self._lock:
            return self._losses.get(entity)

    def losses(self) -> tuple[Loss, ...]:
        with self._lock:
            return tuple(self._losses.values())

    def custody_for(self, subject: str) -> tuple[Custody, ...]:
        with self._lock:
            return tuple(
                c for c in self._custody.values() if c.subject == subject and c.active
            )

    def active_custody(self) -> tuple[Custody, ...]:
        with self._lock:
            return tuple(c for c in self._custody.values() if c.active)

    def active_promises(self) -> tuple[Promise, ...]:
        with self._lock:
            return tuple(p for p in self._promises.values() if p.active)

    def promise_for(self, promise_id: str) -> Promise | None:
        with self._lock:
            return self._promises.get(promise_id)

    def works(self) -> tuple[Work, ...]:
        with self._lock:
            return tuple(self._works.values())

    def work_for(self, work_id: str) -> Work | None:
        with self._lock:
            return self._works.get(work_id)

    def practices(self) -> tuple[Practice, ...]:
        with self._lock:
            return tuple(self._practices.values())

    def practice_for(self, name: str) -> Practice | None:
        with self._lock:
            return self._practices.get(name)

    def rivalry_for(self, entity: str) -> Rivalry | None:
        with self._lock:
            return self._rivalries.get(entity)

    def norm_for(self, name: str) -> Norm | None:
        with self._lock:
            return self._norms.get(name)

    def norms(self) -> tuple[Norm, ...]:
        with self._lock:
            return tuple(self._norms.values())

    def goal_delta(self, object_: str | None) -> float | None:
        if object_ is None:
            return None
        with self._lock:
            return self._goal_deltas.get(object_)

    def goal_weight(self, object_: str | None) -> float | None:
        if object_ is None:
            return None
        with self._lock:
            return self._goals.get(object_)

    def substitutes_for(self, object_: str | None, subject: str | None) -> int | None:
        with self._lock:
            if subject is not None and (
                subject in self._bonds or subject in self._losses
            ):
                # A named person has no substitute. This is the line
                # between grief and disappointment.
                return 0
            if object_ is None:
                return None
            return self._substitutes.get(object_)

    def undo_cost(self, object_: str | None) -> float | None:
        if object_ is None:
            return None
        with self._lock:
            return self._undo_costs.get(object_)

    def actions_that_change(self, object_: str | None) -> int | None:
        if object_ is None:
            return None
        with self._lock:
            return self._actions_that_change.get(object_)

    def own_actions_that_change(self, object_: str | None) -> int | None:
        if object_ is None:
            return None
        with self._lock:
            return self._own_actions_that_change.get(object_)

    def repairs_for(self, event_id: str, subject: str | None) -> tuple[str, ...] | None:
        with self._lock:
            return self._repairs.get(event_id)

    def attribution(self, event_id: str) -> tuple[float, float, float] | None:
        with self._lock:
            return self._attributions.get(event_id)

    def norm_fit(self, event_id: str) -> float | None:
        with self._lock:
            judgement = self._norm_judgements.get(event_id)
            return None if judgement is None else judgement[0]

    def norm_endorsement(self, event_id: str) -> float | None:
        with self._lock:
            judgement = self._norm_judgements.get(event_id)
            return None if judgement is None else judgement[1]

    def observer_count(self, event_id: str) -> int:
        with self._lock:
            return self._observers.get(event_id, -1)

    def times_seen(self, kind: Any, subject: str | None) -> int:
        with self._lock:
            return self._seen.get((str(kind), subject), 0)

    def expectation(self, kind: Any, subject: str | None) -> float | None:
        with self._lock:
            return self._expectations.get((str(kind), subject))

    def nearest_deadline(self, object_: str | None) -> float | None:
        with self._lock:
            deadlines = [
                p.deadline
                for p in self._promises.values()
                if p.active
                and p.deadline is not None
                and (object_ is None or object_ in p.concerns)
            ]
            return min(deadlines) if deadlines else None

    # ── persistence ───────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self.revision,
                "bonds": {k: vars(v) for k, v in self._bonds.items()},
                "promises": {k: vars(v) for k, v in self._promises.items()},
                "custody": {k: vars(v) for k, v in self._custody.items()},
                "losses": {k: vars(v) for k, v in self._losses.items()},
                "works": {k: vars(v) for k, v in self._works.items()},
                "practices": {k: vars(v) for k, v in self._practices.items()},
                "rivalries": {k: vars(v) for k, v in self._rivalries.items()},
                "norms": {k: vars(v) for k, v in self._norms.items()},
                "goals": dict(self._goals),
                "events": list(self._events[-256:]),
            }

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "bonds": len(self._bonds),
                "promises_active": sum(1 for p in self._promises.values() if p.active),
                "custody_active": sum(1 for c in self._custody.values() if c.active),
                "losses": len(self._losses),
                "works": len(self._works),
                "practices": len(self._practices),
                "rivalries": len(self._rivalries),
                "norms": len(self._norms),
                "goals": len(self._goals),
                "events": len(self._events),
            }

    def persist(self, path: Path | None = None) -> bool:
        """Write the ledger through the governed gateway. Never raises."""
        try:
            from core.runtime.state_ownership import state_root
            from core.runtime.file_write_gateway import get_file_write_gateway
            from core.governance_context import local_internal_governed_scope

            target = path or (state_root() / "data" / "interiority_ledger.json")
            with local_internal_governed_scope("interiority.ledger.persist"):
                get_file_write_gateway().write_json(
                    target,
                    self.snapshot(),
                    schema_version=1,
                    schema_name="aura.interiority.ledger",
                    source="core/interiority/ledger.py",
                )
            return True
        except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
            record_degradation("interiority.ledger", exc, action="ledger not persisted")
            return False


__all__ = [
    "Bond",
    "Custody",
    "Loss",
    "Norm",
    "Practice",
    "Promise",
    "RelationalLedger",
    "Rivalry",
    "Stake",
    "Work",
]
