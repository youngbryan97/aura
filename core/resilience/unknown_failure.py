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
    """
    parts: list[float] = []
    parts.append(0.0 if left.subsystem == right.subsystem else 1.0)
    parts.append(0.0 if left.kind == right.kind else 1.0)
    a, b = set(left.broken_invariants), set(right.broken_invariants)
    union = a | b
    parts.append(0.0 if not union else 1.0 - len(a & b) / len(union))
    if keys:
        total = 0.0
        for key in keys:
            span = ranges.get(key, (0.0, 1.0))
            difference = _normalise(
                float(left.observations.get(key, 0.0)), span
            ) - _normalise(float(right.observations.get(key, 0.0)), span)
            total += difference * difference
        parts.append(math.sqrt(total / len(keys)))
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


__all__ = [
    "MATCH_FRACTION",
    "MIN_INSTANCES",
    "MIN_KNOWN_FAULTS",
    "REPAIR_REPERTOIRE",
    "FailureOntology",
    "Recognition",
    "Repair",
    "RepairOutcome",
    "Signature",
    "Verdict",
    "propose_repairs",
]
