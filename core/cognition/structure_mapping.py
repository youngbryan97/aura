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
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Relation", "Graph", "Alignment", "AlignmentAlternatives",
           "map_structures", "map_structures_alternatives", "scrambled",
           "shuffled_null"]


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
    #: Whether every correspondence was tried. False for a domain past
    #: ``max_objects``, which was searched by growing matches outward from
    #: its strongest ones and may have missed the best.
    exhaustive: bool = True

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
            "exhaustive": self.exhaustive,
        }


@dataclass(frozen=True, slots=True)
class AlignmentAlternatives:
    """Best tied readings from a bounded search, with truncation explicit."""

    readings: tuple[Alignment, ...]
    truncated: bool


def _score(
    source: Graph,
    target: Graph,
    mapping: Mapping[str, str],
    predicates: Mapping[str, str] | None = None,
) -> tuple[float, float, list]:
    matched = []
    depth = 0.0
    for relation in source.relations:
        if any(a not in mapping for a in relation.args):
            continue
        if predicates is not None and relation.predicate not in predicates:
            continue
        projected = tuple(mapping[a] for a in relation.args)
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


def _predicate_candidates(
    source: Graph, target: Graph, *, require_complete: bool = False,
) -> list[dict[str, str]]:
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
            if require_complete:
                raise ValueError(
                    "predicate reading search exceeds its exhaustive budget")
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


def _candidate_alignments(
    source: Graph, target: Graph, max_objects: int, *,
    require_complete: bool = False,
    readings: list[dict[str, str]] | None = None,
) -> Iterator[Alignment]:
    source_objects, target_objects = source.objects, target.objects
    if not source_objects or not target_objects:
        return
    if len(source_objects) > max_objects or len(target_objects) > max_objects:
        raise ValueError(
            f"{len(source_objects)} and {len(target_objects)} objects exceed the "
            f"{max_objects} this exhaustive search will attempt; a bigger domain needs "
            "a heuristic search, and pretending to have found nothing would be worse"
        )
    if readings is None:
        readings = _predicate_candidates(
            source, target, require_complete=require_complete)
    size = min(len(source_objects), len(target_objects))
    # A maximal partial injection can extend every smaller injection without
    # removing matches. Enumerate source subsets when the target is smaller.
    for subset in itertools.combinations(source_objects, size):
        for permutation in itertools.permutations(target_objects, size):
            mapping = dict(zip(subset, permutation, strict=True))
            for reading in readings:
                score, systematicity, matched = _score(source, target, mapping, reading)
                yield Alignment(mapping=dict(mapping), matched=tuple(matched),
                                score=score, systematicity=systematicity,
                                predicate_mapping=dict(reading))


def map_structures(
    source: Graph, target: Graph, *, max_objects: int = 7,
    same_vocabulary: bool = False,
) -> Alignment | None:
    """Find the correspondence that aligns the most relational structure.

    Exhaustive over object correspondences up to ``max_objects``, where the
    factorial search is affordable and its answer is the best there is. Past
    it, the search grows matches outward from the strongest local ones (see
    `_grown_alignment`) and says so in ``Alignment.exhaustive``: a refusal
    left every domain bigger than seven objects with no analogy at all, which
    is the same failure as a partial search hiding its misses, only certain.

    ``same_vocabulary`` reads every relation word as itself. Right when both
    graphs were written by one describer, whose words mean one thing each;
    renaming them there only finds coincidences.
    """
    if same_vocabulary:
        readings = [{r.predicate: r.predicate for r in source.relations}]
    else:
        readings = None
    source_objects, target_objects = source.objects, target.objects
    if len(source_objects) > max_objects or len(target_objects) > max_objects:
        return _grown_alignment(source, target, readings)
    best: Alignment | None = None
    for candidate in _candidate_alignments(
            source, target, max_objects, readings=readings):
        if best is None or (candidate.score, candidate.systematicity) > (
                best.score, best.systematicity):
            best = candidate
    return best


#: Readings of the relation words tried when the objects are too many for an
#: exhaustive search: the first of `_predicate_candidates`' order, which puts
#: the readings that rename least and leave least unmapped first.
_READINGS_WHEN_GROWING = 16

#: Strongest local matches each grown search starts from.
_SEEDS_WHEN_GROWING = 24


def _grown_alignment(
    source: Graph, target: Graph, readings: list[dict[str, str]] | None,
) -> Alignment | None:
    """The best correspondence found by growing matches outward, for a big domain.

    A match hypothesis pairs one source relation with one target relation it
    can be read as, and with them the objects in their places. Starting from
    each of the strongest — a higher-order relation first, then the one most
    other matches agree with — the mapping takes on every match consistent
    with it that shares an object with what it already holds, then any
    consistent match at all, until nothing more fits. This is the greedy merge
    of structure-mapping engines, and like it, it can miss the best.
    """
    if not source.objects or not target.objects:
        return None
    if readings is None:
        readings = _predicate_candidates(source, target)[:_READINGS_WHEN_GROWING]
    best: Alignment | None = None
    for reading in readings:
        matches: list[tuple[Relation, Relation, tuple[tuple[str, str], ...]]] = []
        for relation in source.relations:
            read_as = reading.get(relation.predicate)
            if read_as is None:
                continue
            for candidate in target.relations_with(read_as):
                if len(candidate.args) != len(relation.args):
                    continue
                pairs = tuple(zip(relation.args, candidate.args, strict=True))
                if _consistent({}, pairs) is None:
                    continue
                matches.append((relation, candidate, pairs))
        if not matches:
            continue
        support = {
            index: sum(
                1 for other, (_r, _c, pairs) in enumerate(matches)
                if other != index and set(pairs) & set(matches[index][2])
            )
            for index in range(len(matches))
        }
        order = sorted(
            range(len(matches)),
            key=lambda index: (-matches[index][0].order, -support[index], index),
        )
        for seed in order[:_SEEDS_WHEN_GROWING]:
            mapping = _consistent({}, matches[seed][2]) or {}
            grew = True
            while grew:
                grew = False
                for touching in (True, False):
                    for index in order:
                        pairs = matches[index][2]
                        if touching and not any(a in mapping for a, _b in pairs):
                            continue
                        merged = _consistent(mapping, pairs)
                        if merged is not None and merged != mapping:
                            mapping = merged
                            grew = True
                    if grew:
                        break
            score, systematicity, matched = _score(source, target, mapping, reading)
            candidate = Alignment(
                mapping=dict(mapping), matched=tuple(matched), score=score,
                systematicity=systematicity, predicate_mapping=dict(reading),
                exhaustive=False,
            )
            if best is None or (candidate.score, candidate.systematicity) > (
                    best.score, best.systematicity):
                best = candidate
    return best


def _consistent(
    mapping: Mapping[str, str], pairs: Sequence[tuple[str, str]],
) -> dict[str, str] | None:
    """``mapping`` with ``pairs`` added, or None where that breaks one-to-one."""
    merged = dict(mapping)
    taken = {target: source for source, target in merged.items()}
    for source, target in pairs:
        if merged.get(source, target) != target or taken.get(target, source) != source:
            return None
        merged[source] = target
        taken[target] = source
    return merged


def map_structures_alternatives(
    source: Graph, target: Graph, *, max_objects: int = 7,
    max_results: int = 16,
) -> AlignmentAlternatives:
    """Retain tied best correspondences instead of hiding ambiguity.

    The predicate candidate budget is the same as ``map_structures``. A tie
    describes that search, not every semantic interpretation of the scene.
    """
    if type(max_results) is not int or not 1 <= max_results <= 256:
        raise ValueError("alignment result budget must be between 1 and 256")
    best_key: tuple[float, float] | None = None
    ties: list[Alignment] = []
    truncated = False
    for candidate in _candidate_alignments(
            source, target, max_objects, require_complete=True):
        key = (candidate.score, candidate.systematicity)
        if best_key is None or key > best_key:
            best_key = key
            ties = [candidate]
            truncated = False
        elif key == best_key:
            if len(ties) < max_results:
                ties.append(candidate)
            else:
                truncated = True
    return AlignmentAlternatives(tuple(ties), truncated)


def scrambled(target: Graph, rng: random.Random) -> Graph:
    """``target`` with its structure broken and its names and words kept.

    Which object fills which argument slot is drawn afresh for every relation,
    from the target's own objects. Renaming would not do: an exhaustive search
    undoes a renaming exactly.
    """
    objects = list(target.objects)
    return Graph(
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
    scores = []
    for _ in range(trials):
        alignment = map_structures(
            source, scrambled(target, rng), max_objects=max_objects)
        if alignment is not None:
            scores.append(alignment.score)
    mean_null = sum(scores) / len(scores) if scores else 0.0
    ordered = sorted(scores)
    # The upper tail, and how often a scrambled copy did as well. A mean is
    # cleared by a distractor that shares a third of the structure: one
    # scored 0.67 against a mean of 0.50 here, and "structural" said yes.
    upper = (
        ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]
        if ordered else 1.0
    )
    as_well = sum(1 for score in scores if score >= real.score)
    return {
        "measurable": True,
        "score": real.score,
        "null_mean": mean_null,
        "null_upper": upper,
        "as_often_by_chance": (as_well + 1) / (len(scores) + 1),
        "beats_the_upper_tail": real.score > upper,
        "separation": real.score - mean_null,
        "structural": real.score > mean_null,
        "alignment": real.to_dict(),
        "reading": (
            "the alignment tracks structure the shuffled control cannot reach"
            if real.score > mean_null
            else "shuffling the labels scores as well; this alignment is arithmetic"
        ),
    }
