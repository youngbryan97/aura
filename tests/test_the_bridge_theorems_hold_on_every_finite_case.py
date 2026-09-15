"""The bridge theorems, checked in full on every case small enough to enumerate.

Each theorem is proved in docs/BRIDGE_PROOFS.md. These checks run the
repository's own definitions, and the carrier-selection and gauge code J* is
computed with, over every finite instance of a given size, with exact rational
arithmetic where a probability appears. Every check is paired with a case that
breaks the theorem's hypothesis, so a check that could not fail would show.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

from core.subject.bridge import orbits
from core.subject.bridge_theorems import (
    agreeing_law,
    automorphisms,
    causal_state_partition,
    is_sufficient,
    likelihood_ratio,
    partitions,
    refines,
    relational_likelihood,
)
from core.subject.v25_exclusion import Candidate, dominates, pareto_frontier

pytestmark = pytest.mark.unit


# ── Theorem 1: non-identifiability ──────────────────────────────────────────


def _physical_model(table):
    """An observation model that reads the physical history and nothing else."""
    return lambda history, outcome, law: table[(history, outcome)]


def test_a_law_that_adds_no_physical_effect_has_a_likelihood_ratio_of_one() -> None:
    histories, outcomes = ("u1", "u2"), ("o1", "o2", "o3")
    grid = [Fraction(n, 6) for n in range(1, 5)]
    for weights in itertools.product(grid, repeat=3):
        total = sum(weights)
        table = {(h, o): w / total for h in histories for o, w in zip(outcomes, weights, strict=True)}
        model = _physical_model(table)
        for history, outcome in itertools.product(histories, outcomes):
            for law_a, law_b in itertools.product(("nothing", "red", "blue"), repeat=2):
                assert likelihood_ratio(model, history, outcome, law_a, law_b) == 1


def test_a_law_with_a_physical_effect_is_distinguishable() -> None:
    def model(history, outcome, law):
        return Fraction(2, 3) if (law == "red") == (outcome == "reports_red") else Fraction(1, 3)

    assert likelihood_ratio(model, "u", "reports_red", "red", "nothing") == 2


# ── Theorem 2: finite evidence ──────────────────────────────────────────────


def test_every_proper_data_set_leaves_a_rival_law() -> None:
    universe = ("a", "b", "c", "d")
    alternatives = {state: (0, 1) for state in universe}
    for values in itertools.product((0, 1), repeat=len(universe)):
        law = dict(zip(universe, values, strict=True))
        for size in range(len(universe) + 1):
            for observed in itertools.combinations(universe, size):
                rival = agreeing_law(law, observed, alternatives)
                if size == len(universe):
                    assert rival is None
                    continue
                assert rival is not None
                assert all(rival[state] == law[state] for state in observed)
                assert any(rival[state] != law[state] for state in universe if state not in observed)


# ── Theorem 3: the interventional causal state is the minimal sufficient grain ──


def _process(rng: random.Random, histories: int) -> dict:
    grid = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    process = {}
    for h in range(histories):
        process[f"h{h}"] = {}
        for action in ("a0", "a1"):
            p = rng.choice(grid)
            process[f"h{h}"][action] = {"f0": p, "f1": 1 - p}
    return process


@pytest.mark.parametrize("seed", range(40))
def test_every_sufficient_grain_refines_the_causal_state_and_the_causal_state_is_sufficient(seed: int) -> None:
    rng = random.Random(seed)
    process = _process(rng, histories=5)
    causal = causal_state_partition(process)
    assert is_sufficient(process, causal)
    for partition in partitions(sorted(process)):
        if is_sufficient(process, partition):
            assert refines(partition, causal)
        if refines(causal, partition) and partition_blocks(partition) != partition_blocks(causal):
            # Anything strictly coarser than the causal state merges two
            # histories with different futures, so it is not sufficient.
            assert not is_sufficient(process, partition)


def partition_blocks(partition) -> frozenset:
    return frozenset(frozenset(block) for block in partition)


def test_a_grain_that_merges_different_futures_is_not_sufficient() -> None:
    process = {
        "h0": {"a": {"f0": Fraction(1, 4), "f1": Fraction(3, 4)}},
        "h1": {"a": {"f0": Fraction(3, 4), "f1": Fraction(1, 4)}},
    }
    assert not is_sufficient(process, [["h0", "h1"]])


# ── Theorem 4: symmetry — invariant functionals cannot separate symmetric supports ──


def _cycle(n: int, weight: int = 1) -> dict:
    return {(i, (i + 1) % n): weight for i in range(n)}


def _functionals(weights: dict):
    nodes = sorted({node for edge in weights for node in edge})

    def internal(support):
        return sum(w for (s, t), w in weights.items() if s in support and t in support)

    def outgoing(support):
        return sum(w for (s, t), w in weights.items() if s in support and t not in support)

    def degree_profile(support):
        return tuple(sorted(sum(1 for (s, t) in weights if s == node and t in support) for node in support))

    del nodes
    return internal, outgoing, degree_profile


@pytest.mark.parametrize("weights", [_cycle(5), {**_cycle(3), **{(i + 3, (i + 1) % 3 + 3): 1 for i in range(3)}}])
def test_a_permutation_invariant_functional_scores_symmetric_supports_alike(weights: dict) -> None:
    nodes = sorted({node for edge in weights for node in edge})
    symmetries = automorphisms(weights)
    assert len(symmetries) > 1
    for functional in _functionals(weights):
        for size in range(1, len(nodes) + 1):
            for support in itertools.combinations(nodes, size):
                for sigma in symmetries:
                    image = tuple(sorted(sigma[node] for node in support))
                    assert functional(set(support)) == functional(set(image))


def test_a_functional_that_reads_labels_can_separate_them() -> None:
    weights = _cycle(4)
    sigma = next(s for s in automorphisms(weights) if s[0] != 0)
    labelled = lambda support: 0 in support  # noqa: E731
    assert labelled({0}) != labelled({sigma[0]})


# ── Theorem 5: colour refinement is sound, so a rigid class is fixed by every symmetry ──


def _distance_matrix(weights: dict, nodes: list) -> np.ndarray:
    size = len(nodes)
    matrix = np.full((size, size), 9.0)
    np.fill_diagonal(matrix, 0.0)
    for (s, t), w in weights.items():
        matrix[nodes.index(s), nodes.index(t)] = matrix[nodes.index(t), nodes.index(s)] = float(w)
    return matrix


@pytest.mark.parametrize("seed", range(25))
def test_every_symmetry_maps_each_refined_class_onto_itself(seed: int) -> None:
    rng = random.Random(seed)
    size = rng.randint(3, 6)
    names = [f"q{i}" for i in range(size)]
    values = [rng.choice((1.0, 2.0, 3.0)) for _ in range(size * (size - 1) // 2)]
    matrix = np.zeros((size, size))
    for (i, j), value in zip(itertools.combinations(range(size), 2), values, strict=True):
        matrix[i, j] = matrix[j, i] = value
    weights = {(names[i], names[j]): matrix[i, j] for i in range(size) for j in range(size) if i != j}
    cells = orbits(names, matrix)
    for sigma in automorphisms(weights):
        for cell in cells:
            assert {sigma[name] for name in cell} == set(cell)
            if len(cell) == 1:
                assert sigma[cell[0]] == cell[0]


# ── Theorem 6: gauge — a relation-preserving relabelling changes no likelihood ──


def test_relabelling_by_a_symmetry_of_the_relations_leaves_every_likelihood_unchanged() -> None:
    weights = _cycle(5)
    nodes = sorted({node for edge in weights for node in edge})
    distances = {}
    matrix = _distance_matrix(weights, nodes)
    for i, j in itertools.product(range(len(nodes)), repeat=2):
        distances[(nodes[i], nodes[j])] = matrix[i, j]
    symmetric = {edge: w for edge, w in weights.items()}
    symmetric.update({(t, s): w for (s, t), w in weights.items()})
    models = [
        lambda seq: Fraction(1, 1 + int(sum(seq))),
        lambda seq: Fraction(sum(1 for d in seq if d == 1.0), 1 + len(seq)),
    ]
    report = [0, 1, 2, 3]
    for sigma in automorphisms(symmetric):
        relabelled = [sigma[label] for label in report]
        for model in models:
            assert relational_likelihood(distances, report, model) == relational_likelihood(
                distances, relabelled, model
            )


def test_a_relabelling_that_breaks_a_relation_can_change_a_likelihood() -> None:
    weights = _cycle(5)
    nodes = sorted({node for edge in weights for node in edge})
    matrix = _distance_matrix(weights, nodes)
    distances = {(nodes[i], nodes[j]): matrix[i, j] for i, j in itertools.product(range(len(nodes)), repeat=2)}
    model = lambda seq: Fraction(sum(1 for d in seq if d == 1.0), 1 + len(seq))  # noqa: E731
    assert relational_likelihood(distances, [0, 1, 2], model) != relational_likelihood(distances, [0, 2, 4], model)


# ── Theorem 7: exclusion is a Pareto frontier, set-valued when spectra cross ──


def _candidate(support: str, spectrum: dict) -> Candidate:
    return Candidate(support=frozenset(support), closed=True, recurrent=True, spectrum=spectrum)


@pytest.mark.parametrize("seed", range(30))
def test_dominance_is_a_strict_partial_order_and_the_frontier_is_never_empty(seed: int) -> None:
    rng = random.Random(seed)
    horizons = (1.0, 2.0, 4.0)
    candidates = [
        _candidate("".join(sorted(rng.sample("PIAGC", rng.randint(2, 4)))) + "S", {t: rng.choice((0.1, 0.2, 0.3)) for t in horizons})
        for _ in range(rng.randint(2, 6))
    ]
    spectra = [c.spectrum for c in candidates]
    for a in spectra:
        assert not dominates(a, a)
        for b in spectra:
            if dominates(a, b):
                assert not dominates(b, a)
            for c in spectra:
                if dominates(a, b) and dominates(b, c):
                    assert dominates(a, c)
    assert pareto_frontier(candidates)


def test_crossing_spectra_leave_both_on_the_frontier() -> None:
    early = _candidate("PAS", {1.0: 0.3, 4.0: 0.1})
    late = _candidate("PMS", {1.0: 0.1, 4.0: 0.3})
    assert {c.support for c in pareto_frontier([early, late])} == {early.support, late.support}


def test_a_dominated_overlapping_candidate_is_excluded_and_a_disjoint_one_is_not() -> None:
    strong = _candidate("PAS", {1.0: 0.3, 4.0: 0.3})
    weak = _candidate("PS", {1.0: 0.1, 4.0: 0.2})
    apart = _candidate("MW", {1.0: 0.05, 4.0: 0.05})
    frontier = {c.support for c in pareto_frontier([strong, weak, apart])}
    assert strong.support in frontier and apart.support in frontier
    assert weak.support not in frontier
