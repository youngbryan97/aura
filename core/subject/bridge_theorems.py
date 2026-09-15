"""The bridge's theorems, as exact computations a finite case can be checked against.

`core/subject/bridge.py` computes J* from what the runs measured and names the
theorems it rests on. A theorem stated in a docstring is a claim; this module
states each one as a function over finite structures, with exact rational
arithmetic wherever a probability appears, so that every instance small enough
to enumerate can be checked in full rather than sampled. The proofs are in
docs/BRIDGE_PROOFS.md. A check here does not replace a proof: it confirms, on
every case enumerated, that the definitions in this repository mean what the
proof says they mean.

Nothing here tests the postulates P1 to P6. Two of the theorems are about
exactly that limit: no likelihood separates two bridge laws on one causally
closed history, and no finite data set fixes a universal law.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import networkx as nx

__all__ = [
    "agreeing_law",
    "automorphisms",
    "causal_state_partition",
    "is_sufficient",
    "likelihood_ratio",
    "partitions",
    "refines",
    "relational_likelihood",
]


# ── non-identifiability ─────────────────────────────────────────────────────


def likelihood_ratio(
    observation_model: Callable[[Hashable, Hashable, Any], Fraction],
    history: Hashable,
    outcome: Hashable,
    law_a: Any,
    law_b: Any,
) -> Fraction | None:
    """P(O | U, J_a) / P(O | U, J_b), exactly, or None where the denominator is zero.

    The theorem: when phenomenology adds no physical effect, the observation
    model does not read the law at all, and this is one for every outcome.
    """
    numerator = Fraction(observation_model(history, outcome, law_a))
    denominator = Fraction(observation_model(history, outcome, law_b))
    if denominator == 0:
        return None
    return numerator / denominator


# ── finite evidence ─────────────────────────────────────────────────────────


def agreeing_law(
    law: Mapping[Hashable, Hashable],
    observed: Iterable[Hashable],
    alternatives: Mapping[Hashable, Sequence[Hashable]],
) -> dict[Hashable, Hashable] | None:
    """A second law equal to `law` on every observed state and different somewhere else.

    The construction in the proof: keep the law on the data, change it at one
    unobserved state to any other value that state admits. None when every
    state was observed or no unobserved state admits a second value, which is
    the only case in which the data leave no rival.
    """
    seen = set(observed)
    for state, value in law.items():
        if state in seen:
            continue
        for other in alternatives.get(state, ()):
            if other != value:
                rival = dict(law)
                rival[state] = other
                return rival
    return None


# ── the grain: minimal sufficiency of the interventional causal state ───────


Process = Mapping[Hashable, Mapping[Hashable, Mapping[Hashable, Fraction]]]
"""history -> action -> future -> probability, each action's row summing to one."""


def _signature(process: Process, history: Hashable) -> tuple[Any, ...]:
    rows = process[history]
    return tuple(
        (action, tuple(sorted((future, Fraction(p)) for future, p in rows[action].items() if Fraction(p) != 0)))
        for action in sorted(rows)
    )


def causal_state_partition(process: Process) -> frozenset[frozenset[Hashable]]:
    """Histories grouped by their future distribution under every intervention.

    h ~ h' exactly when, for every action, P(F | h, do(a)) = P(F | h', do(a)).
    """
    blocks: dict[tuple[Any, ...], set[Hashable]] = {}
    for history in process:
        blocks.setdefault(_signature(process, history), set()).add(history)
    return frozenset(frozenset(block) for block in blocks.values())


def is_sufficient(process: Process, partition: Iterable[Iterable[Hashable]]) -> bool:
    """Whether knowing a history's block fixes its future under every intervention."""
    for block in partition:
        members = list(block)
        if len({_signature(process, history) for history in members}) > 1:
            return False
    return True


def refines(finer: Iterable[Iterable[Hashable]], coarser: Iterable[Iterable[Hashable]]) -> bool:
    """Whether every block of `finer` sits inside one block of `coarser`, so coarser = f(finer)."""
    owner: dict[Hashable, int] = {}
    for index, block in enumerate(coarser):
        for item in block:
            owner[item] = index
    for block in finer:
        if len({owner.get(item) for item in block}) > 1:
            return False
    return True


def partitions(items: Sequence[Hashable]) -> Iterable[list[list[Hashable]]]:
    """Every set partition of a small sequence, each exactly once."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in partitions(rest):
        for index in range(len(smaller)):
            yield [*smaller[:index], [first, *smaller[index]], *smaller[index + 1 :]]
        yield [[first], *smaller]


# ── symmetry and gauge ──────────────────────────────────────────────────────


def automorphisms(weights: Mapping[tuple[Hashable, Hashable], Any]) -> list[dict[Hashable, Hashable]]:
    """Every permutation of the nodes that preserves every weighted, directed edge."""
    graph = nx.DiGraph()
    nodes = {node for edge in weights for node in edge}
    graph.add_nodes_from(nodes)
    for (source, target), weight in weights.items():
        graph.add_edge(source, target, weight=weight)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        graph, graph, edge_match=lambda left, right: left["weight"] == right["weight"]
    )
    return [dict(mapping) for mapping in matcher.isomorphisms_iter()]


def relational_likelihood(
    distances: Mapping[tuple[Hashable, Hashable], Any],
    labels: Sequence[Hashable],
    model: Callable[[tuple[Any, ...]], Fraction],
) -> Fraction:
    """A likelihood that sees a labelling only through the relations it induces.

    What a third person can record about content is how its classes relate,
    in the order they were reported. Relabelling the classes by a map that
    preserves every relation leaves this sequence, and so the likelihood,
    unchanged.
    """
    observed = tuple(distances[(left, right)] for left, right in itertools.combinations(labels, 2))
    return Fraction(model(observed))
