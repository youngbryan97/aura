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
    #: How each source predicate was read in the target. Empty when the two
    #: domains happened to use the same words, which is the easy case and not
    #: the one the module is for.
    predicate_mapping: Mapping[str, str] = field(default_factory=dict)

    @property
    def shares_no_object_names(self) -> bool:
        """Whether the two domains name their objects differently throughout.

        The solar system and the atom share this: nothing is called `sun` in
        an atom. They do share their relation words — both say `attracts` —
        which is a different and easier thing, and the two used to be the same
        property under this name.
        """
        return all(source != target for source, target in self.mapping.items())

    @property
    def shares_no_vocabulary(self) -> bool:
        """Whether the two domains share no words at all — objects or relations.

        The strong claim, and the one the module's description makes. Queues
        say `waits_behind` and traffic says `follows`; matching predicates as
        strings scored that pair at exactly zero, so nothing could satisfy
        this until predicates could be read as one another.
        """
        if not self.predicate_mapping:
            return False
        predicates_differ = all(a != b for a, b in self.predicate_mapping.items())
        return self.shares_no_object_names and predicates_differ

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": dict(self.mapping),
            "matched_relations": len(self.matched),
            "score": self.score,
            "systematicity": self.systematicity,
            "shares_no_object_names": self.shares_no_object_names,
            "shares_no_vocabulary": self.shares_no_vocabulary,
            "predicate_mapping": dict(self.predicate_mapping),
        }


def _score(
    source: Graph,
    target: Graph,
    mapping: Mapping[str, str],
    predicates: Mapping[str, str] | None = None,
) -> tuple[float, float, list]:
    matched = []
    depth = 0.0
    for relation in source.relations:
        projected = tuple(mapping.get(a, a) for a in relation.args)
        read_as = (
            predicates.get(relation.predicate, relation.predicate)
            if predicates
            else relation.predicate
        )
        for candidate in target.relations_with(read_as):
            if candidate.args == projected:
                matched.append((relation, candidate))
                depth += relation.order
                break
    total = len(source.relations) or 1
    return len(matched) / total, depth / total, matched


def _predicate_candidates(source: Graph, target: Graph) -> list[dict[str, str]]:
    """Every way of reading the source's relation words as the target's.

    Only arity-compatible pairings: a two-place relation cannot be read as a
    one-place one whatever the words are. That is what keeps this from being a
    search over every possible renaming.
    """
    source_predicates = sorted({r.predicate for r in source.relations})
    by_arity: dict[int, list[str]] = {}
    for relation in target.relations:
        by_arity.setdefault(len(relation.args), []).append(relation.predicate)
    for arity in by_arity:
        by_arity[arity] = sorted(set(by_arity[arity]))

    arities = {
        predicate: {
            len(r.args) for r in source.relations if r.predicate == predicate
        }
        for predicate in source_predicates
    }
    options: list[list[str]] = []
    for predicate in source_predicates:
        allowed: set[str] = set()
        for arity in arities[predicate]:
            allowed |= set(by_arity.get(arity, ()))
        # None means "this relation has no counterpart here". Without it, a
        # target with fewer distinct relations than the source admits no
        # injective reading at all and the whole alignment returns None —
        # which reads as "no analogy" when the truth is "a partial one".
        options.append([*sorted(allowed), None])
    if not options:
        return [{}]
    total = 1
    for choices in options:
        total *= len(choices)
        if total > _MAX_PREDICATE_READINGS:
            # Too many readings to enumerate. Fall back to matching the words
            # exactly, which is the old behaviour, rather than searching a
            # fraction of the space and reporting the best of it as the best.
            return [{p: p for p in source_predicates}]
    readings = [
        {
            source: target
            for source, target in zip(source_predicates, combination, strict=True)
            if target is not None
        }
        for combination in itertools.product(*options)
        # One-to-one on predicates for the same reason it is one-to-one on
        # objects: a reading that lets two different source relations both
        # become the same target relation can align anything with anything.
        # Without it, an unrelated domain with two relations matched two
        # thirds of the solar system.
        if _injective(combination)
    ]
    # The identity reading first, so a pair of domains that happen to share
    # their vocabulary keeps the mapping it had before predicates could be
    # renamed. Renaming nothing is the better explanation when it scores the
    # same, and ties here are common.
    readings.sort(
        key=lambda reading: (
            # Most relations accounted for first: a reading that leaves a
            # source relation unmapped explains less than one that does not.
            -len(reading),
            sum(1 for k, v in reading.items() if k != v),
        )
    )
    return readings


def _injective(combination: Sequence[str | None]) -> bool:
    """One-to-one over the predicates that are mapped at all."""
    mapped = [name for name in combination if name is not None]
    return len(set(mapped)) == len(mapped)


#: Predicate readings enumerated before the search gives up and matches words
#: exactly. Bounded for the same reason `max_objects` is: a partial search
#: whose failures look like "no analogy" is worse than a refusal.
_MAX_PREDICATE_READINGS = 4096


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
    readings = _predicate_candidates(source, target)
    best: Alignment | None = None
    for permutation in itertools.permutations(target_objects, len(source_objects)):
        mapping = dict(zip(source_objects, permutation, strict=True))
        for reading in readings:
            score, systematicity, matched = _score(source, target, mapping, reading)
            # Strictly better only. Readings arrive with the fewest renamings
            # first, so an equal score keeps the more conservative one.
            if best is None or (score, systematicity) > (best.score, best.systematicity):
                best = Alignment(
                    mapping=mapping,
                    matched=tuple(matched),
                    score=score,
                    systematicity=systematicity,
                    predicate_mapping=dict(reading),
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
