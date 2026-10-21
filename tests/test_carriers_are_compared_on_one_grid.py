"""Carriers are compared on one grid of horizons, so dominance cannot cycle.

Compared pairwise where each pair happened to overlap, three admissible
carriers could beat each other in a ring (A over B at one horizon, B over C at
a second, C over A at a third) and the frontier came out empty. A reviewer
built exactly that case on 21 October 2026.
"""
from __future__ import annotations

import itertools
import random

import pytest

from core.subject.v25_exclusion import (
    Candidate,
    ExclusionReport,
    dominates,
    frontier_grid,
    pareto_frontier,
    unresolved_pairs,
)


def _carrier(support: str, spectrum: dict[float, float]) -> Candidate:
    return Candidate(support=frozenset(support), closed=True, recurrent=True, spectrum=spectrum)


def _ring() -> list[Candidate]:
    return [
        _carrier("AB", {1.0: 2.0, 2.0: 1.0}),
        _carrier("BC", {2.0: 0.5, 3.0: 2.0}),
        _carrier("CA", {3.0: 1.0, 1.0: 3.0}),
    ]


def test_the_ring_that_emptied_the_frontier_now_leaves_every_carrier_standing_unresolved():
    ring = _ring()
    frontier = pareto_frontier(ring)
    assert {c.support for c in frontier} == {c.support for c in ring}
    assert len(unresolved_pairs(ring)) == 3


def test_the_report_says_the_evidence_did_not_decide():
    ring = _ring()
    report = ExclusionReport(candidates=ring, frontier=pareto_frontier(ring), unresolved=unresolved_pairs(ring))
    assert report.status == "UNRESOLVED"
    assert report.unique_carrier is None
    assert len(report.as_dict()["unresolved_pairs"]) == 3


def test_a_named_grid_leaves_out_of_the_comparison_a_carrier_not_measured_on_it():
    full = _carrier("AB", {1.0: 2.0, 2.0: 2.0, 4.0: 2.0})
    partial = _carrier("BC", {1.0: 1.0, 2.0: 1.0})
    grid = (1.0, 2.0, 4.0)
    assert not dominates(full.spectrum, partial.spectrum, grid)
    assert {c.support for c in pareto_frontier([full, partial], grid)} == {full.support, partial.support}
    assert unresolved_pairs([full, partial], grid) == (("AB", "BC"),)
    # On the horizons both have, the stronger one wins outright.
    assert [c.support for c in pareto_frontier([full, partial])] == [full.support]


@pytest.mark.parametrize("seed", range(40))
def test_on_one_grid_the_frontier_is_never_empty_whatever_was_measured(seed: int):
    rng = random.Random(seed)
    horizons = (0.5, 1.0, 2.0, 4.0, 8.0)
    candidates = []
    for _ in range(rng.randint(2, 7)):
        support = "".join(sorted(rng.sample("PIAGCSMWDN", rng.randint(2, 5))))
        measured = rng.sample(horizons, rng.randint(1, len(horizons)))
        candidates.append(_carrier(support, {tau: rng.choice((0.1, 0.2, 0.3)) for tau in measured}))
    for grid in (None, horizons, horizons[:2]):
        assert pareto_frontier(candidates, grid)
        on = frontier_grid(candidates, grid)
        for a, b, c in itertools.permutations(candidates, 3):
            if dominates(a.spectrum, b.spectrum, on) and dominates(b.spectrum, c.spectrum, on):
                assert dominates(a.spectrum, c.spectrum, on)
            assert not (dominates(a.spectrum, b.spectrum, on) and dominates(b.spectrum, a.spectrum, on))
