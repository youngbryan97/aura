"""core/morphogenesis/substrate.py — where a cell physically is, and what it
costs to change that.

The point of a substrate boundary is that ``bind(a, b)`` should mean different
things to different hardware without the layer above knowing which. In
simulation it adds an edge. Across processes it opens a channel. On an FPGA it
would load a partial bitstream into a reconfigurable region. On docking robot
modules it would drive one module to another, align, latch, and negotiate
power and data.

Those substrates have one thing in common that an in-process graph edit does
not: **a transition takes time, can fail halfway, costs energy, and can leave
a cell unreachable while it happens.** Software written against instant,
infallible binding does not survive contact with any of them. So
:class:`SimulationSubstrate` models all four now, deterministically, and the
governor is written against the awkward version rather than the easy one.

Nothing here modifies source, restarts a process, or reaches the network. A
simulated migration is a simulated migration, and this module never describes
one as physical motion.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .graph import MorphEdge
from .types import json_safe

logger = logging.getLogger("Aura.Morphogenesis.Substrate")


class TransitionOutcome(StrEnum):
    OK = "ok"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REFUSED = "refused"


@dataclass(frozen=True)
class SubstrateResult:
    """What the substrate did.

    ``partial`` is the field that matters. A physical dock that latched
    mechanically and then failed its data handshake has changed the world
    without completing the transition, and the caller must roll back rather
    than retry. A substrate that can never report ``partial`` is a substrate
    that has not been asked to do anything hard yet.
    """

    outcome: TransitionOutcome
    duration_ms: float = 0.0
    energy_spent: float = 0.0
    detail: str = ""
    partial: bool = False
    unreachable_until: float = 0.0

    @property
    def ok(self) -> bool:
        return self.outcome is TransitionOutcome.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "duration_ms": round(float(self.duration_ms), 3),
            "energy_spent": round(float(self.energy_spent), 5),
            "detail": self.detail,
            "partial": self.partial,
            "unreachable_until": self.unreachable_until,
        }


@dataclass
class Placement:
    """Where a cell sits, in whatever coordinates the substrate uses.

    ``locus`` is a process, a host, an FPGA region, or a spot on a table. The
    layer above treats it as an opaque name and only cares that moving between
    two loci has a cost the substrate can quote.
    """

    cell_id: str
    locus: str = "local"
    coordinates: tuple[float, ...] = ()
    energy: float = 1.0
    reachable_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "locus": self.locus,
            "coordinates": list(self.coordinates),
            "energy": round(float(self.energy), 5),
            "reachable_at": self.reachable_at,
        }


@runtime_checkable
class SubstrateAdapter(Protocol):
    """The contract a future hardware module has to satisfy.

    Kept deliberately small. Everything the morphogenetic layer needs from a
    physical carrier is here, and nothing about Python objects, processes, or
    the current machine leaks into it.

    A hardware implementation would map these to: identity and neighbour
    discovery over a radio; capabilities and energy from the module's own
    telemetry; bind/unbind to docking and undocking; migrate to locomotion;
    checkpoint to writing the cell's state into the module's own storage
    before it moves.
    """

    name: str

    def describe(self) -> dict[str, Any]:
        """Identity, capabilities, and limits of this substrate."""

    def place(self, cell_id: str, *, locus: str = "") -> Placement: ...

    def placement(self, cell_id: str) -> Placement | None: ...

    def reachable(self, cell_id: str, *, now: float | None = None) -> bool:
        """False while a cell is mid-transition. A caller that skips this
        check works in simulation and deadlocks on hardware."""

    def bind(self, edge: MorphEdge) -> SubstrateResult: ...

    def unbind(self, edge: MorphEdge) -> SubstrateResult: ...

    def spawn(self, cell_id: str, manifest_data: Mapping[str, Any], *, locus: str = "") -> SubstrateResult: ...

    def retire(self, cell_id: str) -> SubstrateResult: ...

    def migrate(self, cell_id: str, destination: str) -> SubstrateResult: ...

    def checkpoint(self, cell_id: str, state: Mapping[str, Any]) -> SubstrateResult:
        """Persist a cell's state where it survives the cell moving."""

    def health(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


@dataclass
class SubstratePhysics:
    """The costs a substrate imposes.

    Defaults describe an in-process graph edit: instant, free, reliable. The
    scenarios override them with numbers that make transitions worth thinking
    about, which is the point of having the knobs at all.
    """

    bind_ms: float = 0.0
    unbind_ms: float = 0.0
    spawn_ms: float = 0.0
    retire_ms: float = 0.0
    migrate_ms: float = 0.0
    bind_energy: float = 0.0
    spawn_energy: float = 0.0
    migrate_energy: float = 0.0
    #: Probability a transition fails outright.
    failure_rate: float = 0.0
    #: Probability a failure leaves the world changed. Of the failures, this
    #: fraction are partial, and a partial failure is what rollback is for.
    partial_failure_share: float = 0.5
    #: Seconds a migrating cell is unreachable. Zero in-process; seconds on a
    #: robot that has to physically travel.
    migrate_blackout_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bind_ms": self.bind_ms,
            "unbind_ms": self.unbind_ms,
            "spawn_ms": self.spawn_ms,
            "retire_ms": self.retire_ms,
            "migrate_ms": self.migrate_ms,
            "bind_energy": self.bind_energy,
            "spawn_energy": self.spawn_energy,
            "migrate_energy": self.migrate_energy,
            "failure_rate": self.failure_rate,
            "partial_failure_share": self.partial_failure_share,
            "migrate_blackout_s": self.migrate_blackout_s,
        }


#: Physics roughly matching a set of docking tabletop modules. Used by the
#: scenarios to prove the governor survives slow, failable transitions. It is a
#: simulation of those costs and nothing in this file moves any matter.
PHYSICAL_LIKE = SubstratePhysics(
    bind_ms=1200.0,
    unbind_ms=400.0,
    spawn_ms=0.0,
    retire_ms=200.0,
    migrate_ms=4000.0,
    bind_energy=0.08,
    spawn_energy=0.25,
    migrate_energy=0.30,
    failure_rate=0.08,
    partial_failure_share=0.5,
    migrate_blackout_s=4.0,
)


class SimulationSubstrate:
    """A deterministic substrate with a clock the caller controls.

    Two things make it useful rather than a stub. It is seeded, so a scenario
    replays exactly. And its clock is virtual, so a four-second simulated dock
    costs no wall time and the blackout window is still enforced against the
    same clock the caller reads.
    """

    name = "simulation"

    def __init__(
        self,
        *,
        seed: int = 0,
        physics: SubstratePhysics | None = None,
        max_cells: int = 256,
    ):
        self.physics = physics or SubstratePhysics()
        self.max_cells = int(max_cells)
        self._rng = random.Random(seed)
        self._placements: dict[str, Placement] = {}
        self._bound: set[tuple[str, str, str, str]] = set()
        self._clock = 0.0
        self._events: list[dict[str, Any]] = []
        self._energy_spent = 0.0
        self._transitions = 0
        self._failures = 0
        self._partials = 0

    # ── virtual clock ───────────────────────────────────────────────────

    @property
    def now(self) -> float:
        return self._clock

    def advance(self, seconds: float) -> float:
        self._clock += max(0.0, float(seconds))
        return self._clock

    # ── identity ────────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "simulation",
            "physical": False,
            "max_cells": self.max_cells,
            "physics": self.physics.to_dict(),
            "supports": [
                "place", "bind", "unbind", "spawn", "retire", "migrate", "checkpoint",
            ],
        }

    def place(self, cell_id: str, *, locus: str = "") -> Placement:
        cell_id = str(cell_id)
        existing = self._placements.get(cell_id)
        if existing is not None:
            return existing
        placement = Placement(cell_id=cell_id, locus=locus or "local")
        self._placements[cell_id] = placement
        return placement

    def placement(self, cell_id: str) -> Placement | None:
        return self._placements.get(str(cell_id))

    def reachable(self, cell_id: str, *, now: float | None = None) -> bool:
        placement = self._placements.get(str(cell_id))
        if placement is None:
            return False
        clock = self._clock if now is None else float(now)
        return clock >= placement.reachable_at

    def loci(self) -> dict[str, list[str]]:
        """Which cells sit at which locus. A migration scenario is scored on
        how this distribution changes."""
        out: dict[str, list[str]] = {}
        for placement in self._placements.values():
            out.setdefault(placement.locus, []).append(placement.cell_id)
        return {k: sorted(v) for k, v in sorted(out.items())}

    # ── transitions ─────────────────────────────────────────────────────

    def _roll(self, duration_ms: float, energy: float, *, what: str) -> SubstrateResult:
        """One transition attempt against the declared physics."""
        self._transitions += 1
        self.advance(duration_ms / 1000.0)
        self._energy_spent += energy
        if self.physics.failure_rate > 0.0 and self._rng.random() < self.physics.failure_rate:
            self._failures += 1
            partial = self._rng.random() < self.physics.partial_failure_share
            if partial:
                self._partials += 1
            return SubstrateResult(
                outcome=TransitionOutcome.FAILED,
                duration_ms=duration_ms,
                energy_spent=energy,
                detail=f"{what} failed in the substrate",
                partial=partial,
            )
        return SubstrateResult(
            outcome=TransitionOutcome.OK,
            duration_ms=duration_ms,
            energy_spent=energy,
            detail=what,
        )

    def _record(self, kind: str, subject: str, result: SubstrateResult) -> SubstrateResult:
        self._events.append({
            "kind": kind,
            "subject": subject,
            "at": self._clock,
            **result.to_dict(),
        })
        if len(self._events) > 4096:
            del self._events[:-4096]
        return result

    def bind(self, edge: MorphEdge) -> SubstrateResult:
        if not self.reachable(edge.source) or not self.reachable(edge.target):
            return self._record("bind", edge.source, SubstrateResult(
                outcome=TransitionOutcome.REFUSED,
                detail="an endpoint is mid-transition and cannot be bound",
            ))
        result = self._roll(self.physics.bind_ms, self.physics.bind_energy, what="bind")
        if result.ok:
            self._bound.add(edge.key)
        elif result.partial:
            # Latched but never handshook. The edge is half-present in the
            # world, which is exactly what rollback has to clean up.
            self._bound.add(edge.key)
        return self._record("bind", f"{edge.source}->{edge.target}", result)

    def unbind(self, edge: MorphEdge) -> SubstrateResult:
        result = self._roll(self.physics.unbind_ms, 0.0, what="unbind")
        if result.ok or result.partial:
            self._bound.discard(edge.key)
        return self._record("unbind", f"{edge.source}->{edge.target}", result)

    def bound(self, edge: MorphEdge) -> bool:
        """Whether the substrate believes this binding physically exists.

        Compared against the graph by an invariant, because the two disagreeing
        is the signature of a partial failure nobody cleaned up.
        """
        return edge.key in self._bound

    def bound_keys(self) -> set[tuple[str, str, str, str]]:
        return set(self._bound)

    def spawn(self, cell_id: str, manifest_data: Mapping[str, Any], *, locus: str = "") -> SubstrateResult:
        if len(self._placements) >= self.max_cells:
            return self._record("spawn", cell_id, SubstrateResult(
                outcome=TransitionOutcome.REFUSED,
                detail=f"substrate is full at {self.max_cells} cells",
            ))
        result = self._roll(self.physics.spawn_ms, self.physics.spawn_energy, what="spawn")
        if result.ok:
            self.place(cell_id, locus=locus or "local")
        return self._record("spawn", cell_id, result)

    def retire(self, cell_id: str) -> SubstrateResult:
        result = self._roll(self.physics.retire_ms, 0.0, what="retire")
        if result.ok or result.partial:
            self._placements.pop(str(cell_id), None)
            for key in [k for k in self._bound if k[0] == cell_id or k[1] == cell_id]:
                self._bound.discard(key)
        return self._record("retire", cell_id, result)

    def migrate(self, cell_id: str, destination: str) -> SubstrateResult:
        placement = self._placements.get(str(cell_id))
        if placement is None:
            return self._record("migrate", cell_id, SubstrateResult(
                outcome=TransitionOutcome.REFUSED,
                detail="cannot migrate a cell the substrate does not hold",
            ))
        if not self.reachable(cell_id):
            return self._record("migrate", cell_id, SubstrateResult(
                outcome=TransitionOutcome.REFUSED,
                detail="cell is already mid-transition",
            ))
        result = self._roll(self.physics.migrate_ms, self.physics.migrate_energy, what="migrate")
        if result.ok:
            placement.locus = str(destination)
            placement.reachable_at = self._clock + self.physics.migrate_blackout_s
            placement.energy = max(0.0, placement.energy - self.physics.migrate_energy)
            return self._record("migrate", cell_id, SubstrateResult(
                outcome=result.outcome,
                duration_ms=result.duration_ms,
                energy_spent=result.energy_spent,
                detail=result.detail,
                unreachable_until=placement.reachable_at,
            ))
        if result.partial:
            # Left the dock, never arrived. Unreachable and at neither locus.
            placement.locus = "in_transit"
            placement.reachable_at = self._clock + self.physics.migrate_blackout_s
        return self._record("migrate", cell_id, result)

    def checkpoint(self, cell_id: str, state: Mapping[str, Any]) -> SubstrateResult:
        placement = self._placements.get(str(cell_id))
        if placement is None:
            return SubstrateResult(outcome=TransitionOutcome.REFUSED, detail="no such cell")
        return self._record("checkpoint", cell_id, SubstrateResult(
            outcome=TransitionOutcome.OK,
            detail=f"checkpointed {len(dict(state))} field(s)",
        ))

    # ── damage, for the lesion scenario ─────────────────────────────────

    def lesion(self, cell_ids: list[str]) -> list[str]:
        """Remove cells without warning, the way hardware fails.

        No signal, no apoptosis, no chance to hand off. Whatever the layer
        above notices, it notices from the consequences.
        """
        removed: list[str] = []
        for cell_id in cell_ids:
            if self._placements.pop(str(cell_id), None) is not None:
                removed.append(str(cell_id))
                for key in [k for k in self._bound if k[0] == cell_id or k[1] == cell_id]:
                    self._bound.discard(key)
        self._record("lesion", ",".join(removed[:8]), SubstrateResult(
            outcome=TransitionOutcome.OK,
            detail=f"{len(removed)} cell(s) removed without notice",
        ))
        return removed

    # ── reporting ───────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cells": len(self._placements),
            "bindings": len(self._bound),
            "clock": round(self._clock, 4),
            "transitions": self._transitions,
            "failures": self._failures,
            "partial_failures": self._partials,
            "energy_spent": round(self._energy_spent, 5),
            "unreachable": sorted(
                p.cell_id for p in self._placements.values() if p.reachable_at > self._clock
            ),
            "loci": self.loci(),
        }

    def events(self, *, limit: int = 128) -> list[dict[str, Any]]:
        return [json_safe(e) for e in self._events[-int(limit):]]

    def shutdown(self) -> None:
        self._placements.clear()
        self._bound.clear()


class LocalRuntimeSubstrate:
    """In-process placement for the live runtime.

    Every cell sits at one locus and transitions are instant, because in this
    process they are. It exists so the live path goes through the same contract
    the scenarios exercise, rather than through a special case that only the
    simulation ever tests.

    It refuses ``migrate`` to anywhere but its own locus. A cell cannot leave
    this process, and reporting a move that did not happen would put the graph
    and the world out of agreement.
    """

    name = "local_runtime"

    def __init__(self, *, locus: str = "aura_main", max_cells: int = 256):
        self.locus = str(locus)
        self.max_cells = int(max_cells)
        self._placements: dict[str, Placement] = {}
        self._bound: set[tuple[str, str, str, str]] = set()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "in_process",
            "physical": False,
            "locus": self.locus,
            "max_cells": self.max_cells,
            "supports": ["place", "bind", "unbind", "spawn", "retire", "checkpoint"],
            "refuses": ["migrate"],
        }

    def place(self, cell_id: str, *, locus: str = "") -> Placement:
        cell_id = str(cell_id)
        placement = self._placements.get(cell_id)
        if placement is None:
            placement = Placement(cell_id=cell_id, locus=self.locus)
            self._placements[cell_id] = placement
        return placement

    def placement(self, cell_id: str) -> Placement | None:
        return self._placements.get(str(cell_id))

    def reachable(self, cell_id: str, *, now: float | None = None) -> bool:
        return str(cell_id) in self._placements

    def bind(self, edge: MorphEdge) -> SubstrateResult:
        self._bound.add(edge.key)
        return SubstrateResult(outcome=TransitionOutcome.OK, detail="in-process binding")

    def unbind(self, edge: MorphEdge) -> SubstrateResult:
        self._bound.discard(edge.key)
        return SubstrateResult(outcome=TransitionOutcome.OK, detail="in-process unbinding")

    def bound(self, edge: MorphEdge) -> bool:
        return edge.key in self._bound

    def bound_keys(self) -> set[tuple[str, str, str, str]]:
        return set(self._bound)

    def spawn(self, cell_id: str, manifest_data: Mapping[str, Any], *, locus: str = "") -> SubstrateResult:
        if len(self._placements) >= self.max_cells:
            return SubstrateResult(
                outcome=TransitionOutcome.REFUSED,
                detail=f"in-process substrate is full at {self.max_cells} cells",
            )
        self.place(cell_id)
        return SubstrateResult(outcome=TransitionOutcome.OK, detail="in-process cell")

    def retire(self, cell_id: str) -> SubstrateResult:
        self._placements.pop(str(cell_id), None)
        for key in [k for k in self._bound if k[0] == cell_id or k[1] == cell_id]:
            self._bound.discard(key)
        return SubstrateResult(outcome=TransitionOutcome.OK, detail="in-process retirement")

    def migrate(self, cell_id: str, destination: str) -> SubstrateResult:
        if str(destination) == self.locus:
            return SubstrateResult(outcome=TransitionOutcome.OK, detail="already here")
        return SubstrateResult(
            outcome=TransitionOutcome.REFUSED,
            detail=f"{self.name} holds one locus ({self.locus}); a cell cannot leave this process",
        )

    def checkpoint(self, cell_id: str, state: Mapping[str, Any]) -> SubstrateResult:
        if str(cell_id) not in self._placements:
            return SubstrateResult(outcome=TransitionOutcome.REFUSED, detail="no such cell")
        return SubstrateResult(outcome=TransitionOutcome.OK, detail="state is already in this process")

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locus": self.locus,
            "cells": len(self._placements),
            "bindings": len(self._bound),
        }

    def events(self, *, limit: int = 128) -> list[dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        self._placements.clear()
        self._bound.clear()


__all__ = [
    "LocalRuntimeSubstrate",
    "PHYSICAL_LIKE",
    "Placement",
    "SimulationSubstrate",
    "SubstrateAdapter",
    "SubstratePhysics",
    "SubstrateResult",
    "TransitionOutcome",
]
