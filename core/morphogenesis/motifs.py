"""core/morphogenesis/motifs.py — what development remembers.

A motif is not a saved graph. Replaying a stored topology onto a different
problem is template selection with extra steps, and it fails the moment the
environment differs from the one the template came from.

A motif is a *developmental prior*: which capabilities to seed, which bindings
to prefer, and when to stop growing. Applied to a new situation it produces
proposals, and those proposals go through the same governor as any other. So a
motif can be wrong, and being wrong costs it credit.

The rule this module exists to enforce: **a motif is not credited for being
used.** Credit requires the run it shaped to beat a run without it, measured on
the same workload with the same seed. A motif that has been applied fifty times
and never beaten its own absence has a score of zero, and
:meth:`MotifLibrary.select` will not return it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph import MorphGraph
from .proposal import MorphProposal, bind, grow
from .types import clamp01, json_safe, stable_digest


def demand_fingerprint(demand: Mapping[str, float]) -> tuple[str, ...]:
    """The shape of a demand, without its scale.

    Two workloads that want the same capabilities in the same proportions have
    the same fingerprint whatever their volume, which is what lets a motif
    learned on one transfer to another.
    """
    total = sum(max(0.0, float(v)) for v in demand.values())
    if total <= 0:
        return ()
    ranked = sorted(
        ((float(v) / total, str(k)) for k, v in demand.items() if float(v) > 0),
        reverse=True,
    )
    # Bucketed to a tenth, so a small difference in mix is the same shape.
    return tuple(f"{name}:{round(share, 1):.1f}" for share, name in ranked[:5])


def fingerprint_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """0.0 when identical, 1.0 when disjoint."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return 1.0 - (len(set_a & set_b) / len(union)) if union else 0.0


@dataclass
class MotifTrial:
    """One application of a motif, with the null it was measured against."""

    at: float
    with_motif: float
    without_motif: float
    scenario: str = ""
    seed: int = 0

    @property
    def gain(self) -> float:
        return self.with_motif - self.without_motif

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "with_motif": round(self.with_motif, 6),
            "without_motif": round(self.without_motif, 6),
            "gain": round(self.gain, 6),
            "scenario": self.scenario,
            "seed": self.seed,
        }


@dataclass
class MorphMotif:
    """A reusable developmental prior.

    ``seed_capabilities`` and ``seed_counts`` say what to have and how much of
    it. ``preferred_bindings`` says how to wire it, as
    ``(from_capability, to_capability)`` pairs rather than cell ids, because
    the cells in a new situation are different cells.
    ``stop_when_cells`` is the growth bound the motif itself carries, so a
    motif cannot be the thing that makes a population run away.
    """

    motif_id: str
    name: str
    fingerprint: tuple[str, ...]
    seed_capabilities: tuple[str, ...] = ()
    #: How many providers of each capability the successful shape held. A
    #: motif that records only *which* capabilities proposes nothing to a
    #: population that already has one of each, and transfers nothing.
    seed_counts: dict[str, int] = field(default_factory=dict)
    preferred_bindings: tuple[tuple[str, str], ...] = ()
    stop_when_cells: int = 12
    origin_scenario: str = ""
    created_at: float = field(default_factory=time.time)
    trials: list[MotifTrial] = field(default_factory=list)
    applications: int = 0

    @property
    def validated_trials(self) -> int:
        return len(self.trials)

    @property
    def mean_gain(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.gain for t in self.trials) / len(self.trials)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trials if t.gain > 0)

    @property
    def credit(self) -> float:
        """What the motif has actually earned.

        Zero until it has been tried against its own absence at least twice and
        won on balance. Applications do not appear in this number anywhere.
        """
        if len(self.trials) < 2:
            return 0.0
        gain = self.mean_gain
        if gain <= 0:
            return 0.0
        consistency = self.wins / len(self.trials)
        return clamp01(gain * consistency)

    @property
    def credited(self) -> bool:
        return self.credit > 0.0

    def record_trial(self, trial: MotifTrial) -> None:
        self.trials.append(trial)
        if len(self.trials) > 64:
            del self.trials[:-64]

    def matches(self, fingerprint: Sequence[str], *, tolerance: float = 0.5) -> bool:
        return fingerprint_distance(self.fingerprint, fingerprint) <= tolerance

    def develop(
        self,
        *,
        graph: MorphGraph,
        present_capabilities: Mapping[str, Sequence[str]],
        proposer: str,
        round_index: int = 0,
    ) -> list[MorphProposal]:
        """Turn the prior into proposals for the situation in front of it.

        It proposes only what is missing. A motif applied to a population that
        already has its shape proposes nothing, which is what stops repeated
        application from being repeated growth.
        """
        if len(present_capabilities) >= self.stop_when_cells:
            return []
        have: dict[str, list[str]] = {}
        for cell_id, capabilities in present_capabilities.items():
            for capability in capabilities:
                have.setdefault(str(capability), []).append(str(cell_id))

        out: list[MorphProposal] = []
        budget = max(0, self.stop_when_cells - len(present_capabilities))
        for capability in self.seed_capabilities:
            wanted = max(1, int(self.seed_counts.get(capability, 1)))
            held = len(have.get(capability, ()))
            for index in range(min(budget, max(0, wanted - held))):
                # Attach to a cell that already holds the capability where one
                # exists, so the new provider shares real traffic; otherwise to
                # the proposer, which at least makes it reachable.
                anchor = sorted(have.get(capability, ()) or [proposer])[0]
                if anchor == f"m_{capability}_{round_index}_{index}":
                    continue
                out.append(grow(
                    {
                        "name": f"m_{capability}_{round_index}_{index}",
                        "capabilities": [capability],
                        "subsystem": "sandbox",
                        "service_rate": 2,
                    },
                    cell_id=f"m_{capability}_{round_index}_{index}",
                    attach_from=anchor,
                    port=capability,
                    return_port=(
                        sorted(present_capabilities.get(anchor, ()) or (capability,))[0]
                    ),
                    proposer=proposer,
                    parent=proposer,
                    placement="local",
                    subsystem="sandbox",
                    benefit=0.6,
                    cost=0.6,
                    rationale=(
                        f"motif {self.name} wants {wanted} of {capability} and the "
                        f"population holds {held}"
                    ),
                    evidence={"motif_id": self.motif_id, "capability": capability},
                ))
                budget -= 1

        for source_capability, target_capability in self.preferred_bindings:
            sources = have.get(source_capability, [])
            targets = have.get(target_capability, [])
            if not sources or not targets:
                continue
            source = sorted(sources)[0]
            target = sorted(t for t in targets if t != source)
            if not target:
                continue
            destination = target[0]
            if any(
                e.port == target_capability
                for e in graph.out_edges(source)
            ):
                continue
            out.append(bind(
                source, destination, target_capability,
                proposer=proposer,
                subsystem="sandbox",
                benefit=0.55,
                cost=0.05,
                rationale=f"motif {self.name} prefers {source_capability}->{target_capability}",
                evidence={"motif_id": self.motif_id},
            ))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_id": self.motif_id,
            "name": self.name,
            "fingerprint": list(self.fingerprint),
            "seed_capabilities": list(self.seed_capabilities),
            "seed_counts": dict(sorted(self.seed_counts.items())),
            "preferred_bindings": [list(p) for p in self.preferred_bindings],
            "stop_when_cells": self.stop_when_cells,
            "origin_scenario": self.origin_scenario,
            "created_at": self.created_at,
            "applications": self.applications,
            "validated_trials": self.validated_trials,
            "wins": self.wins,
            "mean_gain": round(self.mean_gain, 6),
            "credit": round(self.credit, 6),
            "credited": self.credited,
            "trials": [t.to_dict() for t in self.trials[-16:]],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MorphMotif:
        payload = dict(data or {})
        motif = cls(
            motif_id=str(payload.get("motif_id", "")),
            name=str(payload.get("name", "")),
            fingerprint=tuple(str(f) for f in payload.get("fingerprint", ())),
            seed_capabilities=tuple(str(c) for c in payload.get("seed_capabilities", ())),
            seed_counts={str(k): int(v) for k, v in dict(payload.get("seed_counts", {})).items()},
            preferred_bindings=tuple(
                (str(p[0]), str(p[1])) for p in payload.get("preferred_bindings", ()) if len(p) >= 2
            ),
            stop_when_cells=int(payload.get("stop_when_cells", 12)),
            origin_scenario=str(payload.get("origin_scenario", "")),
            created_at=float(payload.get("created_at", time.time())),
            applications=int(payload.get("applications", 0)),
        )
        for raw in payload.get("trials", ()):
            motif.record_trial(MotifTrial(
                at=float(raw.get("at", 0.0)),
                with_motif=float(raw.get("with_motif", 0.0)),
                without_motif=float(raw.get("without_motif", 0.0)),
                scenario=str(raw.get("scenario", "")),
                seed=int(raw.get("seed", 0)),
            ))
        return motif


class MotifLibrary:
    """The motifs a population has kept, and what each has earned."""

    def __init__(self, *, capacity: int = 32):
        self.capacity = int(capacity)
        self._motifs: dict[str, MorphMotif] = {}

    def __len__(self) -> int:
        return len(self._motifs)

    def all(self) -> list[MorphMotif]:
        return [self._motifs[k] for k in sorted(self._motifs)]

    def get(self, motif_id: str) -> MorphMotif | None:
        return self._motifs.get(motif_id)

    def learn(
        self,
        *,
        name: str,
        demand: Mapping[str, float],
        graph: MorphGraph,
        capabilities: Mapping[str, Sequence[str]],
        baseline_capabilities: Mapping[str, Sequence[str]] | None = None,
        scenario: str = "",
    ) -> MorphMotif:
        """Compress what development produced into a prior that can be reapplied.

        ``baseline_capabilities`` is the founding population. What the motif
        keeps is the *difference*: the capabilities development added and the
        bindings it made. Recording the whole final population instead makes
        every motif from a given starting kit look alike, and two motifs learned
        from opposite demands come out identical — at which point the library
        cannot be wrong about anything, and cannot be right either.
        """
        fingerprint = demand_fingerprint(demand)
        capability_of: dict[str, str] = {}
        for cell_id, caps in capabilities.items():
            if caps:
                capability_of[str(cell_id)] = str(list(caps)[0])

        counts: dict[str, int] = {}
        for capability in capability_of.values():
            counts[capability] = counts.get(capability, 0) + 1

        base_counts: dict[str, int] = {}
        founders = set()
        for cell_id, caps in dict(baseline_capabilities or {}).items():
            founders.add(str(cell_id))
            for capability in caps:
                base_counts[str(capability)] = base_counts.get(str(capability), 0) + 1

        if baseline_capabilities:
            grown = {
                capability for capability, count in counts.items()
                if count > base_counts.get(capability, 0)
            }
            # A demand that needed no new capability still developed a shape;
            # fall back to what the demand itself names so the motif is never
            # empty.
            seeds = tuple(sorted(grown)) or tuple(
                sorted(k for k, v in demand.items() if v > 0)
            )
        else:
            seeds = tuple(sorted(set(capability_of.values())))

        bindings: set[tuple[str, str]] = set()
        for edge in graph.edges():
            if founders and edge.source in founders and edge.target in founders:
                # A binding both ends of which were already there is part of
                # the starting kit, not something this demand taught.
                continue
            source_capability = capability_of.get(edge.source)
            target_capability = capability_of.get(edge.target)
            if source_capability and target_capability and source_capability != target_capability:
                bindings.add((source_capability, target_capability))

        motif_id = "motif_" + stable_digest(name, *fingerprint, *seeds, length=14)
        existing = self._motifs.get(motif_id)
        if existing is not None:
            return existing

        motif = MorphMotif(
            motif_id=motif_id,
            name=name,
            fingerprint=fingerprint,
            seed_capabilities=seeds,
            seed_counts={c: counts.get(c, 1) for c in seeds},
            preferred_bindings=tuple(sorted(bindings)),
            stop_when_cells=max(4, len(capability_of) + 2),
            origin_scenario=scenario,
        )
        self._motifs[motif_id] = motif
        self._evict()
        return motif

    def select(
        self,
        demand: Mapping[str, float],
        *,
        tolerance: float = 0.5,
        require_credit: bool = True,
    ) -> MorphMotif | None:
        """The best-earning motif whose fingerprint fits.

        ``require_credit`` is on by default. An uncredited motif is a guess,
        and applying a guess as if it were experience is how a library of
        motifs makes a system worse the more of them it has.
        """
        fingerprint = demand_fingerprint(demand)
        if not fingerprint:
            return None
        candidates = [
            m for m in self._motifs.values()
            if m.matches(fingerprint, tolerance=tolerance)
            and (m.credited or not require_credit)
        ]
        if not candidates:
            return None
        # Distance first, then earned credit, then id. The id tiebreak keeps
        # the choice stable across processes.
        return min(
            candidates,
            key=lambda m: (
                round(fingerprint_distance(m.fingerprint, fingerprint), 4),
                -m.credit,
                m.motif_id,
            ),
        )

    def record_trial(
        self,
        motif_id: str,
        *,
        with_motif: float,
        without_motif: float,
        scenario: str = "",
        seed: int = 0,
    ) -> MorphMotif | None:
        motif = self._motifs.get(motif_id)
        if motif is None:
            return None
        motif.record_trial(MotifTrial(
            at=time.time(),
            with_motif=float(with_motif),
            without_motif=float(without_motif),
            scenario=scenario,
            seed=seed,
        ))
        return motif

    def note_application(self, motif_id: str) -> None:
        """Count a use. Deliberately separate from credit, and deliberately
        not an input to it."""
        motif = self._motifs.get(motif_id)
        if motif is not None:
            motif.applications += 1

    def prune_uncredited(self, *, min_trials: int = 4) -> list[str]:
        """Drop motifs that have had their chances and never won.

        A library that only grows is a library whose average member is
        untested, and :meth:`select` gets slower and worse as it fills.
        """
        dropped = [
            motif_id for motif_id, motif in self._motifs.items()
            if motif.validated_trials >= min_trials and not motif.credited
        ]
        for motif_id in dropped:
            self._motifs.pop(motif_id, None)
        return sorted(dropped)

    def _evict(self) -> None:
        if len(self._motifs) <= self.capacity:
            return
        ordered = sorted(self._motifs.values(), key=lambda m: (m.credit, m.created_at))
        for motif in ordered[: len(self._motifs) - self.capacity]:
            self._motifs.pop(motif.motif_id, None)

    def status(self) -> dict[str, Any]:
        motifs = self.all()
        return {
            "count": len(motifs),
            "credited": sum(1 for m in motifs if m.credited),
            "applications": sum(m.applications for m in motifs),
            "trials": sum(m.validated_trials for m in motifs),
            "best_credit": round(max((m.credit for m in motifs), default=0.0), 6),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "motifs": {mid: m.to_dict() for mid, m in sorted(self._motifs.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotifLibrary:
        payload = dict(data or {})
        library = cls(capacity=int(payload.get("capacity", 32)))
        for motif_id, raw in dict(payload.get("motifs", {})).items():
            library._motifs[str(motif_id)] = MorphMotif.from_dict(raw)
        return library

    def report(self) -> dict[str, Any]:
        return json_safe({"status": self.status(), "motifs": [m.to_dict() for m in self.all()]})


__all__ = [
    "MorphMotif",
    "MotifLibrary",
    "MotifTrial",
    "demand_fingerprint",
    "fingerprint_distance",
]
