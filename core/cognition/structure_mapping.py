"""core/cognition/structure_mapping.py — the same shape, wearing different words.

Transfer in Aura is currently same-kind: a strategy moves to another task when
the surface vocabulary matches. That is a lookup with extra steps. The transfer
worth having is between domains that share a relational structure and share no
words at all - the reason a person who has understood one queueing system
understands another.

:func:`map_structures` matches two relation graphs on their relations rather
than their names. It follows the two constraints that make structure mapping
more than graph isomorphism:

* **One-to-one.** An object maps to one object. A mapping that lets two things
  in the source both become one thing in the target can align anything.
* **Systematicity.** A mapping supported by relations that are themselves
  arguments to higher relations beats a mapping supported by the same number of
  isolated ones. Deep structure is the point.

The shuffled control
--------------------
:func:`shuffled_null` is what makes a transfer result a result. Permute the
target's object labels and map again: if the shuffled mapping scores as well,
the alignment was arithmetic, not structure. Every claim made through this
module carries that number.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Relation", "Graph", "Alignment", "map_structures", "shuffled_null"]


@dataclass(frozen=True, slots=True)
class Relation:
    """One relation. ``predicate`` is the shape; ``args`` are the objects."""

    predicate: str
    args: tuple[str, ...]
    #: Relations that take this relation as an argument make it systematic.
    order: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args), "order": self.order}


@dataclass(frozen=True, slots=True)
class Graph:
    """A domain, as relations over objects."""

    name: str
    relations: tuple[Relation, ...]

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(sorted({a for r in self.relations for a in r.args}))

    def relations_with(self, predicate: str) -> tuple[Relation, ...]:
        return tuple(r for r in self.relations if r.predicate == predicate)


@dataclass(frozen=True, slots=True)
class Alignment:
    """One correspondence between two domains, and how well it holds up."""

    mapping: Mapping[str, str]
    matched: tuple[tuple[Relation, Relation], ...]
    score: float
    systematicity: float

    @property
    def shares_no_vocabulary(self) -> bool:
        """Whether the two domains name their objects differently throughout."""
        return all(source != target for source, target in self.mapping.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": dict(self.mapping),
            "matched_relations": len(self.matched),
            "score": self.score,
            "systematicity": self.systematicity,
            "shares_no_vocabulary": self.shares_no_vocabulary,
        }


def _score(source: Graph, target: Graph, mapping: Mapping[str, str]) -> tuple[float, float, list]:
    matched = []
    depth = 0.0
    for relation in source.relations:
        projected = tuple(mapping.get(a, a) for a in relation.args)
        for candidate in target.relations_with(relation.predicate):
            if candidate.args == projected:
                matched.append((relation, candidate))
                depth += relation.order
                break
    total = len(source.relations) or 1
    return len(matched) / total, depth / total, matched


def map_structures(
    source: Graph, target: Graph, *, max_objects: int = 7
) -> Alignment | None:
    """Find the correspondence that aligns the most relational structure.

    Exhaustive over object correspondences, which is why ``max_objects`` exists:
    the search is factorial and a domain with more objects than this needs a
    heuristic that is not written here. Refusing is better than a partial
    search whose failures look like "no analogy".
    """
    source_objects, target_objects = source.objects, target.objects
    if not source_objects or not target_objects:
        return None
    if len(source_objects) > max_objects or len(target_objects) > max_objects:
        raise ValueError(
            f"{len(source_objects)} and {len(target_objects)} objects exceed the "
            f"{max_objects} this exhaustive search will attempt; a bigger domain needs "
            "a heuristic search, and pretending to have found nothing would be worse"
        )
    best: Alignment | None = None
    for permutation in itertools.permutations(target_objects, len(source_objects)):
        mapping = dict(zip(source_objects, permutation, strict=True))
        score, systematicity, matched = _score(source, target, mapping)
        if best is None or (score, systematicity) > (best.score, best.systematicity):
            best = Alignment(
                mapping=mapping, matched=tuple(matched), score=score, systematicity=systematicity
            )
    return best


def shuffled_null(
    source: Graph, target: Graph, *, trials: int = 20, seed: int = 0, max_objects: int = 7
) -> dict[str, Any]:
    """Score the real alignment against a target whose STRUCTURE is scrambled.

    The obvious null - relabel the target's objects - is not one. The search is
    exhaustive over correspondences, so it simply undoes the relabelling and
    scores exactly as well; the first version of this control did that and
    reported every analogy as arithmetic. The structure has to be broken, not
    renamed: this permutes which objects fill which argument slots, keeping the
    predicates and the object set and destroying the relational pattern.

    If that scores as well, the alignment was arithmetic. This is the control
    every analogy claim needs and almost never has.
    """
    real = map_structures(source, target, max_objects=max_objects)
    if real is None:
        return {"measurable": False}
    rng = random.Random(seed)
    objects = list(target.objects)
    scores = []
    for _ in range(trials):
        scrambled = Graph(
            name=f"{target.name}_scrambled",
            relations=tuple(
                Relation(
                    r.predicate,
                    tuple(rng.choice(objects) for _ in r.args),
                    r.order,
                )
                for r in target.relations
            ),
        )
        alignment = map_structures(source, scrambled, max_objects=max_objects)
        if alignment is not None:
            scores.append(alignment.score)
    mean_null = sum(scores) / len(scores) if scores else 0.0
    return {
        "measurable": True,
        "score": real.score,
        "null_mean": mean_null,
        "separation": real.score - mean_null,
        "structural": real.score > mean_null,
        "alignment": real.to_dict(),
        "reading": (
            "the alignment tracks structure the shuffled control cannot reach"
            if real.score > mean_null
            else "shuffling the labels scores as well; this alignment is arithmetic"
        ),
    }
