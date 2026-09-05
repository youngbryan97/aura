"""core/evidence/packet.py — a belief that cannot forget what it rests on.

The defect is one line of arithmetic and it is reproducible in four lines::

    space = AtomSpace()
    for _ in range(10):
        space.add(concept("rain"), TruthValue(0.9, 4.0))
    space.get_tv(concept("rain")).confidence   # 0.9756

One observation, asserted ten times, moved confidence from 0.44 to 0.98.
:meth:`TruthValue.revise` is documented as merging "two independent estimates"
and has no way to tell whether they are independent, because a truth value is
a strength and a count with no memory of where the count came from. Every
store in the repository has the same shape under a different name: a float in
the world model, a Wilson bound in ontogeny, a probability from the cortex.
None of them can distinguish ten witnesses from one witness heard ten times.

That distinction is the whole of this module.

What a packet is
----------------
An :class:`EvidencePacket` is a belief plus the identities of the observations
supporting it. ``mass`` is evidence in the same units PLN counts in; ``sources``
is the set of observation identities that produced it. Fusion is a set union
over sources before it is arithmetic over mass, so:

* the same source twice contributes once — the union does not grow;
* two genuinely distinct sources contribute twice;
* a derivation contributes no more than the weakest premise it came through,
  because a conclusion drawn twice from one observation is still one
  observation.

The last point is why :class:`EvidenceKind` exists. An OBSERVATION carries its
own identity into the source set. A DERIVATION inherits its premises' sources
instead of minting a new one, so a chain of inference cannot manufacture
support by being long. This is the property NARS calls an evidential stamp,
built here from the published description of the idea rather than from any
implementation of it.

What this does not do
---------------------
It does not decide how much a source is worth, and it does not model partial
dependence between two sources that share an upstream cause. A shared prefix
in the source set is the signal available, and correlated-but-distinct
evidence still counts twice. Saying so is better than a dependence coefficient
nobody can calibrate.

Conversions in and out are lossy in a stated direction: :func:`from_wilson`
and :func:`from_probability` mint a single source because that is all their
inputs know, which means a Wilson bound converted and re-converted does not
gain independence it never had.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from core.knowledge.atomspace import TruthValue

__all__ = [
    "SCHEMA_VERSION",
    "EvidenceKind",
    "EvidenceSource",
    "EvidencePacket",
    "fuse",
    "independent_mass",
    "from_truth_value",
    "from_wilson",
    "from_probability",
]

#: Evidence-to-confidence lookahead. Matches ``atomspace._LOOKAHEAD`` so a
#: packet and a PLN truth value report the same confidence for the same mass;
#: a second constant here would silently fork the two scales.
#: Schema version for the packet as a cross-organ contract. Bump it when a
#: field is added, removed or changes meaning; a consumer reading an older
#: shape then knows it is reading an older shape.
SCHEMA_VERSION = "aura.evidence.packet.v1"

LOOKAHEAD = 1.0

#: Ceiling on accumulated mass, mirroring ``atomspace._MAX_COUNT``. Confidence
#: asymptotes anyway; the cap stops a runaway loop turning into an unbounded
#: float.
MAX_MASS = 1e9


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


class EvidenceKind(StrEnum):
    """Where a packet's support came from, which decides whether it mints an id.

    The two cases behave differently under fusion and that is the point:
    observing something twice is two pieces of evidence, concluding something
    twice from one observation is one.
    """

    #: Something was seen, measured or reported. Mints its own source identity.
    OBSERVATION = "observation"
    #: Something was concluded from other packets. Inherits their sources and
    #: mints none, so inference cannot manufacture support.
    DERIVATION = "derivation"
    #: Asserted by construction — a definition, a configured policy, an axiom.
    #: Mints an identity so it can be traced, but is never treated as an
    #: observation of the world.
    STIPULATION = "stipulation"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """One observation's identity.

    ``origin`` says which subsystem saw it, ``ref`` identifies the observation
    within that subsystem, and ``at`` is when. Identity is the pair
    ``(origin, ref)``: the same reading reported twice by the same subsystem is
    the same source however many times it is packaged.
    """

    origin: str
    ref: str
    at: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.origin}:{self.ref}"

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "ref": self.ref, "at": self.at}


def _source_key(source: EvidenceSource | str) -> str:
    return source if isinstance(source, str) else source.key


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """A belief, the mass of evidence behind it, and whose evidence it is.

    ``strength`` is the frequency reading — how much of the evidence supports
    the proposition. ``mass`` is how much evidence there is. ``sources`` is the
    set of observation identities that mass is drawn from, and it is what makes
    the arithmetic honest: two packets fuse by unioning sources first.
    """

    strength: float = 0.5
    mass: float = 0.0
    kind: EvidenceKind = EvidenceKind.OBSERVATION
    sources: frozenset[str] = field(default_factory=frozenset)
    #: Free-form label for the claim this is evidence *for*. Never part of
    #: identity; two packets about different subjects must not be fused, and
    #: this is what lets a caller notice.
    subject: str = ""
    #: What produced this packet, for a reader trying to find the code.
    produced_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _clamp01(self.strength))
        object.__setattr__(self, "mass", max(0.0, min(float(self.mass), MAX_MASS)))
        if not isinstance(self.sources, frozenset):
            object.__setattr__(self, "sources", frozenset(self.sources))
        if self.kind is EvidenceKind.DERIVATION and not self.sources:
            raise ValueError(
                "a derivation with no sources claims support it cannot name; "
                "derive from packets, or record it as a stipulation"
            )

    @property
    def confidence(self) -> float:
        """Confidence in the PLN sense: mass against the lookahead constant."""
        return self.mass / (self.mass + LOOKAHEAD)

    @property
    def independent_sources(self) -> int:
        return len(self.sources)

    def with_subject(self, subject: str) -> EvidencePacket:
        return replace(self, subject=subject)

    def fuse(self, other: EvidencePacket) -> EvidencePacket:
        """Merge another packet into this one, counting shared sources once.

        Mass is apportioned per source, so the shared part of two overlapping
        packets contributes the *stronger* of the two readings of it rather
        than their sum. Fusing a packet with itself is a fixed point, which is
        the property the AtomSpace lacked.
        """
        return fuse((self, other))

    def negated(self) -> EvidencePacket:
        return replace(self, strength=1.0 - self.strength)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "mass": self.mass,
            "confidence": self.confidence,
            "kind": self.kind.value,
            "sources": sorted(self.sources),
            "independent_sources": self.independent_sources,
            "subject": self.subject,
            "produced_by": self.produced_by,
        }


def observe(
    strength: float,
    *,
    origin: str,
    ref: str,
    mass: float = 1.0,
    at: float = 0.0,
    subject: str = "",
    produced_by: str = "",
) -> EvidencePacket:
    """Package one observation, minting its identity from origin and ref."""
    source = EvidenceSource(origin=origin, ref=ref, at=at)
    return EvidencePacket(
        strength=strength,
        mass=mass,
        kind=EvidenceKind.OBSERVATION,
        sources=frozenset({source.key}),
        subject=subject,
        produced_by=produced_by or origin,
    )


def independent_mass(packets: Iterable[EvidencePacket]) -> float:
    """Total mass after shared sources are counted once.

    Each source contributes the largest per-source mass any packet assigned it.
    A packet with no sources at all — a bare float from somewhere that has not
    been converted yet — contributes its own mass, because refusing to count it
    would silently discard evidence rather than deduplicate it.
    """
    best: dict[str, float] = {}
    unattributed = 0.0
    for packet in packets:
        if not packet.sources:
            unattributed += packet.mass
            continue
        share = packet.mass / len(packet.sources)
        for key in packet.sources:
            if share > best.get(key, 0.0):
                best[key] = share
    return min(sum(best.values()) + unattributed, MAX_MASS)


def fuse(packets: Sequence[EvidencePacket]) -> EvidencePacket:
    """Combine packets, counting each source once.

    Strength is the mass-weighted mean over the deduplicated mass, so a
    repeated observation neither raises confidence nor drags strength toward
    itself; a genuinely new observation does both.
    """
    packets = [p for p in packets if p is not None]
    if not packets:
        return EvidencePacket(kind=EvidenceKind.STIPULATION)
    if len(packets) == 1:
        return packets[0]

    subjects = {p.subject for p in packets if p.subject}
    if len(subjects) > 1:
        raise ValueError(
            f"refusing to fuse evidence about different subjects: {sorted(subjects)}; "
            "fusing across subjects is how a confident answer to one question "
            "becomes a confident answer to another"
        )

    total_mass = independent_mass(packets)
    sources = frozenset().union(*(p.sources for p in packets))

    # Strength is weighted by each packet's *new* contribution, so a duplicate
    # cannot pull the mean toward its own reading either.
    seen: set[str] = set()
    weighted = 0.0
    weight = 0.0
    for packet in sorted(packets, key=lambda p: -p.mass):
        fresh = packet.sources - seen if packet.sources else frozenset()
        if packet.sources:
            share = packet.mass * (len(fresh) / len(packet.sources))
            seen |= packet.sources
        else:
            share = packet.mass
        weighted += packet.strength * share
        weight += share
    strength = (weighted / weight) if weight > 0 else sum(p.strength for p in packets) / len(packets)

    kinds = {p.kind for p in packets}
    kind = EvidenceKind.OBSERVATION if EvidenceKind.OBSERVATION in kinds else next(iter(kinds))
    return EvidencePacket(
        strength=strength,
        mass=total_mass,
        kind=kind,
        sources=sources,
        subject=next(iter(subjects), ""),
        produced_by="core.evidence.fuse",
    )


def derive(
    strength: float,
    premises: Sequence[EvidencePacket],
    *,
    discount: float = 1.0,
    subject: str = "",
    produced_by: str = "",
) -> EvidencePacket:
    """Conclude something from premises, inheriting their sources.

    The conclusion's mass is the weakest premise's mass times ``discount`` and
    never more, and its sources are the union of the premises'. Both together
    are what stop a long chain from looking like a lot of evidence: deriving
    the same conclusion by two routes through one observation fuses back to
    that one observation.
    """
    if not premises:
        raise ValueError("a derivation needs premises; use observe() or a stipulation")
    sources = frozenset().union(*(p.sources for p in premises))
    if not sources:
        raise ValueError(
            "every premise is unattributed, so this conclusion cannot name its support; "
            "convert the premises with from_wilson/from_probability first"
        )
    mass = min(p.mass for p in premises) * max(0.0, min(1.0, discount))
    return EvidencePacket(
        strength=strength,
        mass=mass,
        kind=EvidenceKind.DERIVATION,
        sources=sources,
        subject=subject or next((p.subject for p in premises if p.subject), ""),
        produced_by=produced_by or "core.evidence.derive",
    )


# ── adapters: the other four uncertainty dialects in the repository ────────

def _mint(origin: str, payload: object) -> frozenset[str]:
    digest = hashlib.blake2s(repr(payload).encode("utf-8"), digest_size=8).hexdigest()
    return frozenset({f"{origin}:{digest}"})


def from_truth_value(
    tv: TruthValue, *, origin: str, ref: str = "", subject: str = ""
) -> EvidencePacket:
    """Lift a PLN truth value into a packet.

    A truth value has no lineage, so one is minted from ``ref`` when given and
    from the value itself otherwise. The second case is deliberately weak: two
    identical truth values from the same origin become the same source, which
    is the conservative reading when nothing better is known.
    """
    sources = frozenset({EvidenceSource(origin, ref).key}) if ref else _mint(origin, (tv.strength, tv.count))
    return EvidencePacket(
        strength=tv.strength,
        mass=tv.count,
        kind=EvidenceKind.OBSERVATION,
        sources=sources,
        subject=subject,
        produced_by="core.evidence.from_truth_value",
    )


def to_truth_value(packet: EvidencePacket) -> TruthValue:
    """Project back to PLN. Lineage is dropped, which is why fusion happens here."""
    from core.knowledge.atomspace import TruthValue

    return TruthValue(packet.strength, packet.mass)


def from_wilson(
    successes: int,
    trials: int,
    *,
    origin: str,
    ref: str = "",
    z: float = 1.96,
    subject: str = "",
) -> EvidencePacket:
    """Lift a success/trial count, using the Wilson lower bound as strength.

    ``trials`` is real evidence mass, so it transfers directly. The whole run
    is one source: a batch of trials reported once is one report, and treating
    each trial as independently attributable would recreate the defect this
    module exists to close.
    """
    from core.cognition.procedural_generalization import wilson_lower_bound

    sources = frozenset({EvidenceSource(origin, ref).key}) if ref else _mint(origin, (successes, trials))
    return EvidencePacket(
        strength=wilson_lower_bound(successes, trials, z=z),
        mass=float(trials),
        kind=EvidenceKind.OBSERVATION,
        sources=sources,
        subject=subject,
        produced_by="core.evidence.from_wilson",
    )


def from_probability(
    probability: float,
    *,
    origin: str,
    ref: str = "",
    mass: float = 1.0,
    subject: str = "",
) -> EvidencePacket:
    """Lift a bare probability — a model logit, a heuristic score, a float.

    ``mass`` defaults to 1.0 because a probability with no sample behind it is
    one opinion. Callers that know the sample size should pass it; callers that
    do not should leave it, and the resulting confidence of 0.5 is the correct
    reading of a number nobody counted.
    """
    sources = frozenset({EvidenceSource(origin, ref).key}) if ref else _mint(origin, (origin, probability))
    return EvidencePacket(
        strength=probability,
        mass=mass,
        kind=EvidenceKind.OBSERVATION,
        sources=sources,
        subject=subject,
        produced_by="core.evidence.from_probability",
    )
