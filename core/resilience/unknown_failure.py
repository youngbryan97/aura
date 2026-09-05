"""core/resilience/unknown_failure.py — a failure the catalogue does not contain.

`fault_taxonomy.py` is an FMEA catalogue: every known failure mode with its
severity, its blast radius and its runbook. It is good, and it has the
property every catalogue has — ``record_fault`` takes a ``fault_id``, so a
failure that is not in it has to be forced into an entry that nearly fits or
dropped. Both lose the interesting case.

The interesting case is the hard version of self-repair, and it is a sequence
rather than a fix:

1. recognise that this is not any failure mode the system knows;
2. infer which invariant stopped holding;
3. localise where it stopped holding;
4. invent a repair;
5. verify the repair;
6. integrate the new failure concept, so next time it is known.

Step six is what makes it learning rather than handling. A system that
recovers from a novel failure and forgets it has to rediscover it every time.

Step one is the one that can go quietly wrong. Saying "this is not anything I
know" needs the known things described in the same terms as the new one, and a
catalogue entry is prose — a name, a description, a runbook. So the signature
of a known fault is *learned from its instances*: every time a fault is
recorded, its observable features go into that fault's record, and a new
failure is compared against what was actually seen rather than against what
somebody wrote down.

That makes the recogniser checkable, which matters more than it sounds. It has
a null: a repeat of a known failure must come back KNOWN. A recogniser that
calls everything novel is as useless as one that calls nothing novel, and only
one of the two looks like it is working.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Resilience.Unknown")

#: Instances of a known fault needed before its learned signature is worth
#: matching against. One instance is a point, not a signature.
MIN_INSTANCES = 3

#: How close a new failure has to be to a known fault's signature, as a
#: fraction of the typical distance between distinct known faults. Relative
#: for the same reason the model horizon's radius is: whether 0.3 is "close"
#: depends on how the features are scaled.
MATCH_FRACTION = 0.5

#: Known faults needed before "not any of these" means anything. With one
#: fault in the catalogue everything else is novel and the word is empty.
MIN_KNOWN_FAULTS = 3


class Recognition(StrEnum):
    """What the system can say about a failure it just saw."""

    #: It matches a fault whose signature has been seen enough times.
    KNOWN = "known"
    #: It matches nothing, and enough is known for that to mean something.
    NOVEL = "novel"
    #: Too little is catalogued for "not any of these" to be a finding.
    UNDECIDABLE = "undecidable"


@dataclass(frozen=True)
class Signature:
    """The observable shape of one failure, in features anything can produce."""

    #: Which subsystem reported it.
    subsystem: str
    #: The exception or condition type.
    kind: str
    #: Declared invariants that were violated around it.
    broken_invariants: tuple[str, ...] = ()
    #: Numeric observations at the time, by channel.
    observations: Mapping[str, float] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def vector(self, keys: Sequence[str]) -> tuple[float, ...]:
        return tuple(float(self.observations.get(key, 0.0)) for key in keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "kind": self.kind,
            "broken_invariants": list(self.broken_invariants),
            "observations": dict(self.observations),
            "at": self.at,
        }


@dataclass(frozen=True)
class Verdict:
    """What was recognised, and what follows from it."""

    recognition: Recognition
    #: The nearest known fault, when there is one.
    nearest: str = ""
    distance: float | None = None
    #: The invariant most likely to have broken, inferred rather than reported.
    broken_invariant: str = ""
    #: Where it broke.
    locus: str = ""
    because: str = ""

    @property
    def needs_a_new_concept(self) -> bool:
        return self.recognition is Recognition.NOVEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognition": str(self.recognition),
            "nearest": self.nearest,
            "distance": None if self.distance is None else round(self.distance, 4),
            "broken_invariant": self.broken_invariant,
            "locus": self.locus,
            "needs_a_new_concept": self.needs_a_new_concept,
            "because": self.because,
        }


@dataclass(frozen=True)
class Repair:
    """A proposed fix, and how to tell whether it worked."""

    action: str
    rationale: str
    #: The invariant that must hold again for this to count as repaired.
    restores: str
    #: How irreversible the action is, in [0, 1].
    irreversibility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "restores": self.restores,
            "irreversibility": self.irreversibility,
        }


@dataclass(frozen=True)
class RepairOutcome:
    """Whether a repair held, checked rather than assumed."""

    repair: Repair
    invariant_holds: bool
    signature_recurred: bool

    @property
    def worked(self) -> bool:
        """Both halves. An invariant that holds while the failure keeps
        happening means the invariant was not the one that broke."""
        return self.invariant_holds and not self.signature_recurred

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair": self.repair.to_dict(),
            "invariant_holds": self.invariant_holds,
            "signature_recurred": self.signature_recurred,
            "worked": self.worked,
        }


#: Per-channel (low, high) taken from everything the ontology has seen, so a
#: numeric feature is compared on its own scale.
Ranges = Mapping[str, tuple[float, float]]


def _normalise(value: float, span: tuple[float, float]) -> float:
    low, high = span
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _distance(
    left: Signature, right: Signature, keys: Sequence[str], ranges: Ranges
) -> float:
    """How unlike two failures are, over features either could produce.

    Every part lands in [0, 1] before they are averaged. Without normalising
    the numeric channels, a latency measured in milliseconds swamps every
    categorical difference: on the first run a consciousness-subsystem drift
    error came back as a known inference fault, because 2.5 looked small
    beside a typical separation of 5144 that was entirely made of latency.

    A channel only one of them carries is dropped rather than read as zero.
    Absence is not a measurement, and treating it as one does damage in both
    directions: a failure reported with no numbers at all looked like a
    failure whose every number was zero, so it was far from everything and
    always novel; and two failures neither of which was measured looked
    identical, so the second was always known. Where nothing is shared the
    comparison is the categorical one, which is a smaller claim honestly
    made.
    """
    parts: list[float] = []
    parts.append(0.0 if left.subsystem == right.subsystem else 1.0)
    parts.append(0.0 if left.kind == right.kind else 1.0)
    a, b = set(left.broken_invariants), set(right.broken_invariants)
    union = a | b
    parts.append(0.0 if not union else 1.0 - len(a & b) / len(union))
    shared = [
        key
        for key in keys
        if key in left.observations and key in right.observations
    ]
    if shared:
        total = 0.0
        for key in shared:
            span = ranges.get(key, (0.0, 1.0))
            difference = _normalise(
                float(left.observations[key]), span
            ) - _normalise(float(right.observations[key]), span)
            total += difference * difference
        parts.append(math.sqrt(total / len(shared)))
    return sum(parts) / len(parts)


class FailureOntology:
    """What the system knows how to recognise, and how it grows."""

    def __init__(self) -> None:
        self._instances: dict[str, list[Signature]] = {}
        self._invented: set[str] = set()
        self._lock = threading.RLock()

    # ── learning what the known ones look like ───────────────────────────

    def observe(self, fault_id: str, signature: Signature) -> None:
        """Record what one known fault actually looked like."""
        with self._lock:
            self._instances.setdefault(str(fault_id), []).append(signature)

    def _feature_keys(self) -> tuple[str, ...]:
        with self._lock:
            keys: set[str] = set()
            for instances in self._instances.values():
                for instance in instances:
                    keys |= set(instance.observations)
        return tuple(sorted(keys))

    def _ranges(self, keys: Sequence[str]) -> dict[str, tuple[float, float]]:
        """The span of every numeric channel across everything seen."""
        with self._lock:
            everything = [i for group in self._instances.values() for i in group]
        spans: dict[str, tuple[float, float]] = {}
        for key in keys:
            values = [
                float(i.observations[key]) for i in everything if key in i.observations
            ]
            spans[key] = (min(values), max(values)) if values else (0.0, 1.0)
        return spans

    def _learned(self) -> dict[str, list[Signature]]:
        with self._lock:
            return {
                fault: list(instances)
                for fault, instances in self._instances.items()
                if len(instances) >= MIN_INSTANCES
            }

    @property
    def known_faults(self) -> tuple[str, ...]:
        return tuple(sorted(self._learned()))

    # ── step 1: is this anything I know ──────────────────────────────────

    def recognise(self, signature: Signature) -> Verdict:
        learned = self._learned()
        if len(learned) < MIN_KNOWN_FAULTS:
            return Verdict(
                recognition=Recognition.UNDECIDABLE,
                because=(
                    f"{len(learned)} fault(s) have enough instances to have a "
                    f"signature; {MIN_KNOWN_FAULTS} are needed before "
                    '"not any of these" is a finding'
                ),
            )
        keys = self._feature_keys()
        ranges = self._ranges(keys)
        scored: list[tuple[float, str]] = []
        for fault, instances in learned.items():
            best = min(_distance(signature, i, keys, ranges) for i in instances)
            scored.append((best, fault))
        scored.sort(key=lambda pair: pair[0])
        nearest_distance, nearest = scored[0]

        typical = _typical_separation(learned, keys, ranges)
        threshold = typical * MATCH_FRACTION
        if nearest_distance <= threshold:
            return Verdict(
                recognition=Recognition.KNOWN,
                nearest=nearest,
                distance=nearest_distance,
                broken_invariant=_most_common_invariant(learned[nearest]),
                locus=signature.subsystem,
                because=(
                    f"{nearest_distance:.3f} from {nearest}, inside "
                    f"{threshold:.3f} — half the {typical:.3f} that separates "
                    "distinct known faults"
                ),
            )
        return Verdict(
            recognition=Recognition.NOVEL,
            nearest=nearest,
            distance=nearest_distance,
            broken_invariant=_infer_invariant(signature),
            locus=signature.subsystem,
            because=(
                f"nearest known fault is {nearest} at {nearest_distance:.3f}, "
                f"outside the {threshold:.3f} that would make it a match. This "
                "is not any failure mode the system has a concept for"
            ),
        )

    # ── step 6: give it a name so it is known next time ──────────────────

    def integrate(self, name: str, signature: Signature) -> str:
        """Mint a concept for a novel failure, so it stops being novel.

        The step that makes this learning rather than handling. Without it a
        system rediscovers the same unknown failure every time it happens and
        pays the full diagnostic cost each round.
        """
        fault_id = f"FAULT-NOVEL-{name}"
        with self._lock:
            self._invented.add(fault_id)
        self.observe(fault_id, signature)
        logger.info("Integrated a new failure concept: %s", fault_id)
        return fault_id

    @property
    def invented(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._invented))

    def snapshot(self) -> dict[str, Any]:
        learned = self._learned()
        with self._lock:
            total = sum(len(v) for v in self._instances.values())
        return {
            "faults_with_signatures": len(learned),
            "faults_seen": len(self._instances),
            "instances": total,
            "invented": list(self.invented),
        }

    def clear(self) -> None:
        with self._lock:
            self._instances.clear()
            self._invented.clear()


def _typical_separation(
    learned: Mapping[str, list[Signature]], keys: Sequence[str], ranges: Ranges
) -> float:
    """How far apart distinct known faults are, as the scale of "close"."""
    representatives = [instances[0] for instances in learned.values()]
    if len(representatives) < 2:
        return 1.0
    distances = [
        _distance(representatives[i], representatives[j], keys, ranges)
        for i in range(len(representatives))
        for j in range(i + 1, len(representatives))
    ]
    distances.sort()
    middle = len(distances) // 2
    if len(distances) % 2:
        return max(1e-9, distances[middle])
    return max(1e-9, (distances[middle - 1] + distances[middle]) / 2.0)


def _most_common_invariant(instances: Sequence[Signature]) -> str:
    counts: dict[str, int] = {}
    for instance in instances:
        for name in instance.broken_invariants:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return min(counts, key=lambda name: (-counts[name], name))


def _infer_invariant(signature: Signature) -> str:
    """Which invariant most likely stopped holding, for a novel failure.

    The reported ones when there are any. Otherwise nothing: guessing an
    invariant for a failure that broke none is how a repair gets verified
    against a condition that was never the problem.
    """
    if signature.broken_invariants:
        return sorted(signature.broken_invariants)[0]
    return ""


#: General repairs, in the order they should be tried: least irreversible
#: first. Not a list of fixes for particular faults — those are runbooks — but
#: the moves available when nobody knows what this is.
REPAIR_REPERTOIRE: tuple[Repair, ...] = (
    Repair(
        action="retry_with_backoff",
        rationale="a transient condition clears on its own and costs nothing to wait out",
        restores="",
        irreversibility=0.0,
    ),
    Repair(
        action="reset_component_state",
        rationale="a component holding bad state recovers when the state is dropped",
        restores="",
        irreversibility=0.2,
    ),
    Repair(
        action="isolate_and_degrade",
        rationale="the rest of the system keeps working while this part is out",
        restores="",
        irreversibility=0.3,
    ),
    Repair(
        action="restart_subsystem",
        rationale="everything the subsystem holds is rebuilt from durable state",
        restores="",
        irreversibility=0.6,
    ),
)


def propose_repairs(verdict: Verdict) -> tuple[Repair, ...]:
    """Repairs to try for a failure nobody has a runbook for, cheapest first.

    Each carries the invariant it has to restore, taken from the verdict, so
    verification checks the thing that actually broke rather than a general
    notion of health.
    """
    return tuple(
        Repair(
            action=repair.action,
            rationale=repair.rationale,
            restores=verdict.broken_invariant,
            irreversibility=repair.irreversibility,
        )
        for repair in REPAIR_REPERTOIRE
    )


# ── the part that was missing: this happening in a running system ────────
#
# Everything above was correct and nothing called it. A module that can
# recognise a failure nobody has a concept for, and which no failure ever
# reaches, is a description of a capability rather than the capability. The
# live ladder went failure → restart → localisation → governed reconstruction
# → receipt, and never once asked whether it had seen this before.
#
# Two seams, in the two places that can afford them:
#
#   learning what the known ones look like, from the fault registry's own
#   listener hook, which must be O(1) and is;
#
#   recognising a new one, at the escalation point in the healing ladder,
#   which is already async and already off every hot path.
#
# The ontology persists for the same reason the record of her own work does.
# A failure concept invented on Tuesday and forgotten at the next restart is
# rediscovered at full diagnostic cost on Wednesday, which is exactly the
# thing step six exists to prevent.


def _where_it_is_kept() -> Any:
    from core.runtime.state_ownership import state_root

    return state_root() / "what_failure_looks_like.json"


class _KeptOntology(FailureOntology):
    """A FailureOntology that survives the process it learned in."""

    def __init__(self) -> None:
        super().__init__()
        self._restored = False
        self._unwritten = 0

    def _restore_once(self) -> None:
        with self._lock:
            if self._restored:
                return
            self._restored = True
        try:
            import json

            held = json.loads(_where_it_is_kept().read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError, AttributeError):
            return
        if not isinstance(held, dict):
            return
        with self._lock:
            for fault_id, rows in (held.get("instances") or {}).items():
                for row in rows or ():
                    if not isinstance(row, dict):
                        continue
                    self._instances.setdefault(str(fault_id), []).append(
                        Signature(
                            subsystem=str(row.get("subsystem") or ""),
                            kind=str(row.get("kind") or ""),
                            broken_invariants=tuple(
                                str(one) for one in row.get("broken_invariants") or ()
                            ),
                            observations={
                                str(k): float(v)
                                for k, v in (row.get("observations") or {}).items()
                            },
                            at=float(row.get("at") or 0.0),
                        )
                    )
            self._invented |= {str(one) for one in held.get("invented") or ()}

    def keep(self) -> bool:
        """Write down what failure looks like, through the governed gateway."""
        import json

        with self._lock:
            body = {
                "instances": {
                    fault: [one.to_dict() for one in group[-HOW_MANY_KEPT_EACH:]]
                    for fault, group in self._instances.items()
                },
                "invented": sorted(self._invented),
            }
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "unknown_failure.keep", domain="state_mutation"
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(
                    _where_it_is_kept().parent, source="unknown_failure"
                )
                gateway.write_text(
                    _where_it_is_kept(), json.dumps(body), source="unknown_failure"
                )
            return True
        except (OSError, RuntimeError, TypeError, ValueError, ImportError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "unknown_failure", exc, severity="info",
                action="keep what failure looks like",
            )
            return False

    def observe(self, fault_id: str, signature: Signature) -> None:
        self._restore_once()
        super().observe(fault_id, signature)
        with self._lock:
            self._unwritten += 1
            due = self._unwritten >= HOW_OFTEN_IT_IS_WRITTEN
            if due:
                self._unwritten = 0
        if due:
            _ASKED_TO_WRITE.set()

    def recognise(self, signature: Signature) -> Verdict:
        self._restore_once()
        return super().recognise(signature)

    def _learned(self) -> dict[str, list[Signature]]:
        # Every question about what is known goes through here, so this is
        # the one place the restore has to happen. Putting it only on the
        # three public methods left `known_faults` answering from an empty
        # ontology after a restart, which is the persistence bug this class
        # exists to fix, one level down.
        self._restore_once()
        return super()._learned()

    @property
    def invented(self) -> tuple[str, ...]:
        self._restore_once()
        return super().invented

    def snapshot(self) -> dict[str, Any]:
        self._restore_once()
        return super().snapshot()

    def integrate(self, name: str, signature: Signature) -> str:
        self._restore_once()
        fault_id = super().integrate(name, signature)
        _ASKED_TO_WRITE.set()
        return fault_id


#: Instances kept per fault. The signature is a shape, and thirty instances of
#: one shape say what three hundred do at a tenth of the file.
HOW_MANY_KEPT_EACH = 30

#: Observations between write-backs.
HOW_OFTEN_IT_IS_WRITTEN = 8

_ONTOLOGY: _KeptOntology | None = None
_ONTOLOGY_LOCK = threading.Lock()
_ASKED_TO_WRITE = threading.Event()
_WRITER: threading.Thread | None = None
_ATTACHED = [False]


def get_failure_ontology() -> FailureOntology:
    """The one ontology, restoring itself and writing itself back."""
    global _ONTOLOGY, _WRITER
    with _ONTOLOGY_LOCK:
        if _ONTOLOGY is None:
            _ONTOLOGY = _KeptOntology()
        if _WRITER is None:
            _WRITER = threading.Thread(
                target=_write_when_asked,
                name="what-failure-looks-like",
                daemon=True,
            )
            _WRITER.start()
        return _ONTOLOGY


def _write_when_asked() -> None:
    while True:
        _ASKED_TO_WRITE.wait()
        _ASKED_TO_WRITE.clear()
        held = _ONTOLOGY
        if held is None:
            continue
        try:
            held.keep()
        except Exception as exc:  # noqa: BLE001 - a writer thread may not die
            logger.debug("could not keep what failure looks like: %s", exc)


def signature_of(record: Any, *, broken_invariants: Sequence[str] = ()) -> Signature:
    """The observable shape of one recorded fault.

    Features anything can produce, so a new failure is compared against what
    known faults were actually seen doing rather than against the prose in a
    catalogue entry.
    """
    severity = getattr(record, "severity", None)
    recovery = getattr(record, "recovery_time_s", None)
    return Signature(
        subsystem=str(getattr(record, "subsystem", "") or ""),
        kind=str(getattr(record, "error_type", "") or "")
        or str(getattr(record, "fault_id", "") or ""),
        broken_invariants=tuple(str(one) for one in broken_invariants),
        observations={
            "severity": float(int(severity)) if severity is not None else 0.0,
            "recovered": 1.0 if getattr(record, "recovered", False) else 0.0,
            "recovery_seconds": float(recovery) if recovery else 0.0,
            "message_length": float(len(str(getattr(record, "error_message", "")))),
        },
    )


def learn_from_fault(record: Any) -> None:
    """What one known fault looked like. The registry listener body: O(1)."""
    fault_id = str(getattr(record, "fault_id", "") or "")
    if not fault_id:
        return
    try:
        get_failure_ontology().observe(fault_id, signature_of(record))
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("could not learn from fault %s: %s", fault_id, exc)


def attach_to_the_fault_registry(registry: Any = None) -> bool:
    """Learn the shape of every fault the system records. Idempotent.

    The registry is passed in by the singleton that is building itself, so
    this never re-enters ``get_fault_registry`` from inside it.
    """
    with _ONTOLOGY_LOCK:
        if _ATTACHED[0]:
            return False
        _ATTACHED[0] = True
    try:
        if registry is None:
            from core.resilience.fault_taxonomy import get_fault_registry

            registry = get_fault_registry()
        return bool(registry.add_listener(learn_from_fault))
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("could not attach the failure ontology: %s", exc)
        with _ONTOLOGY_LOCK:
            _ATTACHED[0] = False
        return False


@dataclass(frozen=True)
class Diagnosis:
    """What was recognised and what to try, for one failure in flight."""

    verdict: Verdict
    signature: Signature
    repairs: tuple[Repair, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.to_dict(),
            "signature": self.signature.to_dict(),
            "repairs": [one.to_dict() for one in self.repairs],
        }


def look_at_this_failure(
    subsystem: str,
    kind: str,
    *,
    broken_invariants: Sequence[str] = (),
    observations: Mapping[str, float] | None = None,
) -> Diagnosis:
    """Steps one to four, at the point where the ladder is about to escalate.

    Not on any hot path. This runs where the watchdog has already decided a
    service is not coming back, which is the only place worth spending a
    diagnosis and the one place the ladder never asked.
    """
    signature = Signature(
        subsystem=str(subsystem),
        kind=str(kind),
        broken_invariants=tuple(str(one) for one in broken_invariants),
        observations=dict(observations or {}),
    )
    verdict = get_failure_ontology().recognise(signature)
    return Diagnosis(
        verdict=verdict,
        signature=signature,
        repairs=propose_repairs(verdict) if verdict.needs_a_new_concept else (),
    )


def a_repair_that_held(diagnosis: Diagnosis, *, called: str) -> str:
    """Step six. Give the novel failure a name, so it is known next time.

    Only for a repair that actually held. Integrating a concept for a failure
    that is still happening teaches the recogniser to expect the broken state,
    and the next occurrence comes back KNOWN with nothing known about it.
    """
    if not diagnosis.verdict.needs_a_new_concept:
        return ""
    return get_failure_ontology().integrate(called, diagnosis.signature)


__all__ = [
    "HOW_MANY_KEPT_EACH",
    "HOW_OFTEN_IT_IS_WRITTEN",
    "MATCH_FRACTION",
    "MIN_INSTANCES",
    "MIN_KNOWN_FAULTS",
    "REPAIR_REPERTOIRE",
    "Diagnosis",
    "FailureOntology",
    "Recognition",
    "Repair",
    "RepairOutcome",
    "Signature",
    "Verdict",
    "a_repair_that_held",
    "attach_to_the_fault_registry",
    "get_failure_ontology",
    "learn_from_fault",
    "look_at_this_failure",
    "propose_repairs",
    "signature_of",
]
