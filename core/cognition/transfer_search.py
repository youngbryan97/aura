"""core/cognition/transfer_search.py — finding the analogue nobody pointed at.

:mod:`core.cognition.structure_mapping` can tell whether two domains share a
relational structure. It has to be handed the pair. That is the easy half:
somebody who already suspects queues and traffic are the same thing has done
the interesting part of the work, and what remains is confirmation.

The transfer worth having is the other way round. Aura meets a new situation
and something she has already understood turns out to have the same shape,
without anyone having tagged the two as related — which is what happens when a
person who has understood one queueing system understands another, and they
were not told there was a queue involved.

Doing that needs a way to look up a domain by its shape rather than by its
words, because the words are exactly what will not match. A
:class:`Signature` is that key: the multiset of relation arities and orders,
the object count, and the degree profile. All of it survives renaming
everything, which is the property that makes it a structural index and not a
vocabulary one.

Retrieval narrows; it does not decide. Signature distance is a cheap filter
over what could possibly align, and every candidate it returns is then put
through the real mapping and its shuffled null. A domain that retrieves well
and maps badly is a near miss, and reporting it as a transfer is how a system
ends up believing everything is like everything.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.structure_mapping import Alignment, Graph, map_structures, shuffled_null

logger = logging.getLogger("Aura.Cognition.Transfer")

#: Candidates put through the full mapping. The mapping is factorial in the
#: object count, so retrieval has to narrow hard before verification runs.
VERIFY_TOP_K = 5

#: How far the real alignment must beat its shuffled null before a retrieved
#: domain counts as an analogue rather than a coincidence.
MIN_SEPARATION = 0.2

#: Alignment score below which nothing is a transfer whatever the null says.
MIN_SCORE = 0.5


@dataclass(frozen=True)
class Signature:
    """The shape of a domain, with every word removed.

    Everything here survives renaming every object and every relation, which
    is the point: the index has to find a domain whose words share nothing
    with the query's.
    """

    objects: int
    #: (arity, order) for each relation, sorted. The relational skeleton.
    shape: tuple[tuple[int, int], ...]
    #: How many relations each object takes part in, sorted descending.
    degrees: tuple[int, ...]

    @classmethod
    def of(cls, graph: Graph) -> Signature:
        counts: dict[str, int] = {}
        for relation in graph.relations:
            for argument in relation.args:
                counts[argument] = counts.get(argument, 0) + 1
        return cls(
            objects=len(graph.objects),
            shape=tuple(
                sorted((len(r.args), r.order) for r in graph.relations)
            ),
            degrees=tuple(sorted(counts.values(), reverse=True)),
        )

    def distance(self, other: Signature) -> float:
        """How unlike two shapes are, in [0, 1]."""
        parts = [
            _ratio_distance(self.objects, other.objects),
            _multiset_distance(self.shape, other.shape),
            _multiset_distance(self.degrees, other.degrees),
        ]
        return sum(parts) / len(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": self.objects,
            "shape": [list(pair) for pair in self.shape],
            "degrees": list(self.degrees),
        }


def _ratio_distance(left: int, right: int) -> float:
    high = max(left, right)
    return 0.0 if high == 0 else abs(left - right) / high


def _multiset_distance(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Jaccard over multisets: how much of the two do not correspond."""
    if not left and not right:
        return 0.0
    counts_left: dict[Any, int] = {}
    counts_right: dict[Any, int] = {}
    for item in left:
        counts_left[item] = counts_left.get(item, 0) + 1
    for item in right:
        counts_right[item] = counts_right.get(item, 0) + 1
    keys = set(counts_left) | set(counts_right)
    shared = sum(min(counts_left.get(k, 0), counts_right.get(k, 0)) for k in keys)
    total = sum(max(counts_left.get(k, 0), counts_right.get(k, 0)) for k in keys)
    return 0.0 if total == 0 else 1.0 - shared / total


@dataclass(frozen=True)
class Transfer:
    """A domain that turned out to have the same shape, and the evidence."""

    source: str
    target: str
    alignment: Alignment
    separation: float
    null_mean: float
    signature_distance: float

    @property
    def holds(self) -> bool:
        return (
            self.alignment.score >= MIN_SCORE and self.separation >= MIN_SEPARATION
        )

    @property
    def crosses_vocabularies(self) -> bool:
        return self.alignment.shares_no_vocabulary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "score": round(self.alignment.score, 4),
            "systematicity": round(self.alignment.systematicity, 4),
            "null_mean": round(self.null_mean, 4),
            "separation": round(self.separation, 4),
            "signature_distance": round(self.signature_distance, 4),
            "holds": self.holds,
            "crosses_vocabularies": self.crosses_vocabularies,
            "object_mapping": dict(self.alignment.mapping),
            "predicate_mapping": dict(self.alignment.predicate_mapping),
        }


class DomainIndex:
    """Everything understood so far, keyed by shape rather than by subject."""

    def __init__(self) -> None:
        self._graphs: dict[str, Graph] = {}
        self._signatures: dict[str, Signature] = {}

    def add(self, graph: Graph) -> None:
        self._graphs[graph.name] = graph
        self._signatures[graph.name] = Signature.of(graph)

    def extend(self, graphs: Iterable[Graph]) -> None:
        for graph in graphs:
            self.add(graph)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._graphs))

    def candidates(self, query: Graph, *, top_k: int = VERIFY_TOP_K) -> list[tuple[float, str]]:
        """Domains whose shape could align with this one, nearest first."""
        signature = Signature.of(query)
        scored = [
            (signature.distance(other), name)
            for name, other in self._signatures.items()
            if name != query.name
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        return scored[: max(0, top_k)]

    def find_analogues(
        self,
        query: Graph,
        *,
        top_k: int = VERIFY_TOP_K,
        max_objects: int = 7,
        trials: int = 20,
    ) -> tuple[Transfer, ...]:
        """Search for a domain with the same shape, and verify what turns up.

        Nobody names the pair. Retrieval narrows by shape, and every candidate
        is then put through the real mapping and its shuffled null — because a
        domain that retrieves well and maps badly is a near miss, and calling
        that a transfer is how a system comes to believe everything is like
        everything.
        """
        found: list[Transfer] = []
        for distance, name in self.candidates(query, top_k=top_k):
            target = self._graphs[name]
            if (
                len(query.objects) > max_objects
                or len(target.objects) > max_objects
            ):
                logger.debug("Skipping %s: too many objects to map exhaustively", name)
                continue
            alignment = map_structures(query, target, max_objects=max_objects)
            if alignment is None:
                continue
            null = shuffled_null(
                query, target, trials=trials, max_objects=max_objects
            )
            if not null.get("measurable"):
                continue
            found.append(
                Transfer(
                    source=query.name,
                    target=name,
                    alignment=alignment,
                    separation=float(null.get("separation", 0.0)),
                    null_mean=float(null.get("null_mean", 0.0)),
                    signature_distance=distance,
                )
            )
        found.sort(key=lambda t: (-t.alignment.score, -t.separation, t.target))
        return tuple(found)

    def best_analogue(self, query: Graph, **kwargs: Any) -> Transfer | None:
        """The one transfer that holds, or None. None is a real answer."""
        for transfer in self.find_analogues(query, **kwargs):
            if transfer.holds:
                return transfer
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "domains": len(self._graphs),
            "names": list(self.names),
        }


__all__ = [
    "MIN_SCORE",
    "MIN_SEPARATION",
    "VERIFY_TOP_K",
    "DomainIndex",
    "Signature",
    "Transfer",
]
