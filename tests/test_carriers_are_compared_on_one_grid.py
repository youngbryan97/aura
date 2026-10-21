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
    missing_horizons,
    outranks,
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
    # With no grid named, the grid is every horizon either has, so the same
    # pair is unresolved rather than decided on the two horizons they share.
    assert {c.support for c in pareto_frontier([full, partial])} == {full.support, partial.support}
    assert missing_horizons([full, partial]) == {"BC": (4.0,)}


def test_a_lead_the_intervals_do_not_separate_decides_nothing():
    high = _carrier("AB", {1.0: 0.30, 2.0: 0.30})
    low = _carrier("BC", {1.0: 0.20, 2.0: 0.20})
    assert [c.support for c in pareto_frontier([high, low])] == [high.support]
    high.bounds = {1.0: (0.15, 0.45), 2.0: (0.15, 0.45)}
    low.bounds = {1.0: (0.05, 0.35), 2.0: (0.05, 0.35)}
    assert {c.support for c in pareto_frontier([high, low])} == {high.support, low.support}
    high.bounds = {1.0: (0.26, 0.34), 2.0: (0.26, 0.34)}
    low.bounds = {1.0: (0.16, 0.24), 2.0: (0.16, 0.24)}
    assert [c.support for c in pareto_frontier([high, low])] == [high.support]


@pytest.mark.parametrize("seed", range(20))
def test_interval_dominance_never_empties_the_frontier(seed: int):
    rng = random.Random(seed)
    horizons = (1.0, 2.0, 4.0)
    candidates = []
    for _ in range(rng.randint(2, 7)):
        support = "".join(sorted(rng.sample("PIAGCSMWDN", rng.randint(2, 5))))
        centre = {tau: rng.choice((0.1, 0.2, 0.3, 0.4)) for tau in horizons}
        width = rng.choice((0.0, 0.04, 0.12))
        item = _carrier(support, centre)
        item.bounds = {tau: (value - width, value + width) for tau, value in centre.items()}
        candidates.append(item)
    assert pareto_frontier(candidates)
    for a, b, c in itertools.permutations(candidates, 3):
        if outranks(a, b, horizons) and outranks(b, c, horizons):
            assert outranks(a, c, horizons)


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
