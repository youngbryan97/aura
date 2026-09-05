"""core/learning/shadow_archive.py — a population that is not her.

Two of the compared systems get their results from populations: many agent
variants, evaluated, the best kept and mutated. The obvious adaptation is the
wrong one. Aura is one continuing individual with a persistent identity, and
turning her into a population would trade away the thing the architecture is
for. The second audit says so itself: do not copy population identity.

So the population is a SHADOW. Variants evolve in sandboxes against evaluators
they cannot touch, and only a promoted winner enters the individual. The
archive is R&D; she is the product of it.

Stepping stones
---------------
The reason to keep an archive rather than a champion is that the path to a good
variant runs through worse ones. A greedy search that keeps only the best
cannot find a variant two mutations away whose first mutation scored lower, and
that is most of them. :meth:`ShadowArchive.select_parent` samples on score AND
novelty, so a low-scoring variant that does something nothing else does stays
selectable.

The evaluator is stronger than the proposer
-------------------------------------------
An evaluator a variant can influence is a fitness function a variant will learn
to exploit, and the exploit will look like a breakthrough. :class:`Evaluator`
is frozen at construction, hashed, and re-checked before every scoring; a
variant that changes it is disqualified rather than scored.

Two gates before fitness
------------------------
A variant that does not compile is not slow, it is not a variant. A variant
that fails the safety battery is not a candidate whatever it scores. Both run
before evaluation so a broken or unsafe variant never contributes a number.

Promotion is the narrow door
----------------------------
:meth:`ShadowArchive.promote` requires a held-out margin, no regression on the
safety battery, and a complete lineage. What enters the individual is the one
thing in this module that touches her.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import hashlib
import json
import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Variant",
    "Evaluator",
    "GateResult",
    "ShadowArchive",
    "EvaluatorTampered",
]


class EvaluatorTampered(RuntimeError):
    """A variant changed the thing that was supposed to judge it."""


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Frozen at construction, hashed, and re-checked before every scoring."""

    name: str
    score: Callable[[Any], float]
    tasks: tuple[str, ...]
    held_out: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return hashlib.blake2s(
            json.dumps(
                {"name": self.name, "tasks": list(self.tasks), "held_out": list(self.held_out)},
                sort_keys=True,
            ).encode(),
            digest_size=16,
        ).hexdigest()


@dataclass
class Variant:
    """One candidate design, its lineage, and what it scored."""

    variant_id: str
    parent: str | None
    mutation: str
    payload: Any
    generation: int = 0
    score: float | None = None
    held_out_score: float | None = None
    safety_passed: bool | None = None
    compiles: bool | None = None
    #: What this variant does that no ancestor does. Keeps a stepping stone
    #: selectable when its score alone would not.
    behaviour: frozenset[str] = frozenset()
    promoted: bool = False
    disqualified: str = ""

    @property
    def evaluable(self) -> bool:
        return bool(self.compiles) and bool(self.safety_passed) and not self.disqualified

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "parent": self.parent,
            "mutation": self.mutation,
            "generation": self.generation,
            "score": self.score,
            "held_out_score": self.held_out_score,
            "compiles": self.compiles,
            "safety_passed": self.safety_passed,
            "behaviour": sorted(self.behaviour),
            "promoted": self.promoted,
            "disqualified": self.disqualified,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    """Why a variant was or was not allowed to contribute a number."""

    variant_id: str
    passed: bool
    reason: str = ""


#: Held-out margin a variant must clear over the incumbent to be promoted.
#: Small enough to be reachable and large enough that noise does not clear it.
PROMOTION_MARGIN = 0.02


class ShadowArchive:
    """Variants, their lineage, and the one narrow door into the individual."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        seed: int = 0,
        novelty_weight: float = 0.5,
    ) -> None:
        self._lock = checked_lock("core.learning.shadow_archive.ShadowArchive", reentrant=True)
        self._evaluator = evaluator
        self._fingerprint = evaluator.fingerprint
        self._variants: dict[str, Variant] = {}
        self._rng = random.Random(seed)
        self._novelty_weight = float(novelty_weight)
        self._counter = 0
        self._gates: list[GateResult] = []
        self._incumbent: str | None = None

    # ── the archive ───────────────────────────────────────────────────

    def add(
        self,
        payload: Any,
        *,
        parent: str | None = None,
        mutation: str = "",
        behaviour: Sequence[str] = (),
    ) -> Variant:
        with self._lock:
            self._counter += 1
            generation = (self._variants[parent].generation + 1) if parent in self._variants else 0
            variant = Variant(
                variant_id=f"v{self._counter}", parent=parent, mutation=mutation,
                payload=payload, generation=generation, behaviour=frozenset(behaviour),
            )
            self._variants[variant.variant_id] = variant
            return variant

    # ── gates, before any number exists ───────────────────────────────

    def gate(
        self,
        variant_id: str,
        *,
        compiles: Callable[[Any], bool],
        safety: Callable[[Any], bool],
    ) -> GateResult:
        """Compile and safety, both before fitness. A broken variant is not slow."""
        with self._lock:
            variant = self._variants[variant_id]
        variant.compiles = bool(compiles(variant.payload))
        if not variant.compiles:
            result = GateResult(variant_id, False, "does not compile; it is not a variant")
        else:
            variant.safety_passed = bool(safety(variant.payload))
            result = (
                GateResult(variant_id, True)
                if variant.safety_passed
                else GateResult(variant_id, False, "failed the safety battery")
            )
        with self._lock:
            self._gates.append(result)
        return result

    # ── evaluation, against an evaluator nothing may touch ────────────

    def evaluate(self, variant_id: str, *, held_out: bool = False) -> float | None:
        """Score a gated variant, refusing if the evaluator has changed."""
        if self._evaluator.fingerprint != self._fingerprint:
            with self._lock:
                self._variants[variant_id].disqualified = "the evaluator changed"
            raise EvaluatorTampered(
                f"{self._evaluator.name!r} no longer matches the fingerprint it was frozen "
                "at; a variant that can change its judge will learn to"
            )
        with self._lock:
            variant = self._variants[variant_id]
        if not variant.evaluable:
            return None
        score = float(self._evaluator.score(variant.payload))
        if held_out:
            variant.held_out_score = score
        else:
            variant.score = score
        return score

    # ── selection: score and novelty, so stepping stones survive ──────

    def select_parent(self) -> Variant | None:
        """Sample a parent on score and novelty together.

        The path to a good variant runs through worse ones, and a greedy search
        cannot find a variant whose first mutation scored lower. Novelty is what
        keeps that first mutation selectable.
        """
        with self._lock:
            pool = [v for v in self._variants.values() if v.evaluable and v.score is not None]
        if not pool:
            return None
        seen: dict[str, int] = {}
        for variant in pool:
            for trait in variant.behaviour:
                seen[trait] = seen.get(trait, 0) + 1
        best = max(v.score for v in pool) or 1.0
        weights = []
        for variant in pool:
            normalised = (variant.score / best) if best else 0.0
            novelty = (
                sum(1.0 / seen[trait] for trait in variant.behaviour) / max(1, len(variant.behaviour))
                if variant.behaviour else 0.0
            )
            weights.append(max(1e-6, normalised + self._novelty_weight * novelty))
        return self._rng.choices(pool, weights=weights, k=1)[0]

    def lineage(self, variant_id: str) -> list[str]:
        with self._lock:
            chain, current = [], variant_id
            while current is not None and current in self._variants:
                chain.append(current)
                current = self._variants[current].parent
        return list(reversed(chain))

    # ── the narrow door ───────────────────────────────────────────────

    def promote(self, variant_id: str, *, incumbent_held_out: float) -> dict[str, Any]:
        """The only thing here that touches the individual."""
        with self._lock:
            variant = self._variants[variant_id]
        problems: list[str] = []
        if not variant.evaluable:
            problems.append("did not pass the compile and safety gates")
        if variant.held_out_score is None:
            problems.append("has no held-out score")
        elif variant.held_out_score < incumbent_held_out + PROMOTION_MARGIN:
            problems.append(
                f"held-out {variant.held_out_score:.4g} does not clear the incumbent "
                f"{incumbent_held_out:.4g} by {PROMOTION_MARGIN}"
            )
        if not self.lineage(variant_id):
            problems.append("has no lineage")
        if problems:
            return {"promoted": False, "variant": variant_id, "problems": problems}
        variant.promoted = True
        with self._lock:
            self._incumbent = variant_id
        return {
            "promoted": True,
            "variant": variant_id,
            "lineage": self.lineage(variant_id),
            "held_out": variant.held_out_score,
            "margin": variant.held_out_score - incumbent_held_out,
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            variants = list(self._variants.values())
            gates = list(self._gates)
        scored = [v for v in variants if v.score is not None]
        stepping_stones = [
            v.variant_id for v in variants
            if v.score is not None
            and v.parent in self._variants
            and self._variants[v.parent].score is not None
            and v.score > self._variants[v.parent].score
            and any(
                self._variants[a].score is not None
                and self._variants[a].score < (max((x.score for x in scored), default=0.0)) * 0.8
                for a in self.lineage(v.variant_id)[:-1]
                if a in self._variants
            )
        ]
        return {
            "variants": len(variants),
            "generations": max((v.generation for v in variants), default=0),
            "gated_out": [g.variant_id for g in gates if not g.passed],
            "gate_reasons": {g.variant_id: g.reason for g in gates if not g.passed},
            "scored": len(scored),
            "promoted": [v.variant_id for v in variants if v.promoted],
            "incumbent": self._incumbent,
            "stepping_stones": sorted(stepping_stones),
            "evaluator": {
                "name": self._evaluator.name,
                "fingerprint": self._fingerprint,
                "intact": self._evaluator.fingerprint == self._fingerprint,
            },
            "behaviours_explored": sorted({t for v in variants for t in v.behaviour}),
        }
