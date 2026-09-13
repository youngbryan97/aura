"""A null's edges are decided by the rule the organism's edges are decided by.

The organism's graph keeps a pair only when the effect beats its sham floor on
a sign-flip test, survives a Benjamini-Hochberg correction across all ninety
pairs, reaches the effect bar and replicates across conditions. A null's graph
kept every pair whose mean peak displacement reached the effect bar and asked
nothing else. The comparison between them was between two rules.
"""

from __future__ import annotations

from collections import Counter

import pytest

from core.subject import causal
from core.subject.graph import analyse_graph
from core.subject.nulls import NULL_CONDITIONS, architecture, toy_edges, toy_interventions
from core.subject.state import DOMAINS


def test_a_null_gets_as_many_regimes_as_the_battery_has_conditions() -> None:
    from core.subject.driver import CONDITIONS

    assert NULL_CONDITIONS == len(CONDITIONS)


def test_every_source_is_tried_the_same_number_of_times_in_every_regime() -> None:
    results = toy_interventions(architecture("recurrent", seed=3), trials=3, conditions=2, seed=3)
    counts = Counter((trial.source, trial.condition) for trial in results.trials)
    assert set(counts.values()) == {3}
    assert len(counts) == len(DOMAINS) * 2


def test_the_sham_floor_is_read_as_the_real_run_reads_it() -> None:
    """Two sham arms from the same state and noise do not move apart."""
    results = toy_interventions(architecture("recurrent", seed=3), trials=2, conditions=1, seed=3)
    assert all(value == 0.0 for trial in results.trials for value in trial.floor.values())


def test_null_edges_go_through_the_real_edge_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}
    real = causal.build_edges

    def spy(results, **kwargs):
        seen["trials"] = len(results.trials)
        return real(results, **kwargs)

    monkeypatch.setattr(causal, "build_edges", spy)
    toy_edges(architecture("independent", seed=3), trials=2, conditions=3, seed=3)
    assert seen["trials"] == 2 * 3 * len(DOMAINS)


@pytest.mark.slow
def test_the_reference_still_comes_out_strongly_connected() -> None:
    """The positive control has to survive the stricter rule, or the rule is
    measuring its own power rather than the wiring."""
    edges = toy_edges(architecture("recurrent", seed=5), seed=5)
    assert analyse_graph(list(DOMAINS), edges).one_component, edges


@pytest.mark.slow
def test_independent_domains_leave_no_edge() -> None:
    assert toy_edges(architecture("independent", seed=5), seed=5) == []


def test_a_displacement_shows_no_earlier_than_it_lands() -> None:
    """Every frame before the injection reads exactly like the sham, whichever
    point in the arm the injection lands at."""
    results = toy_interventions(architecture("star", seed=3), trials=3, conditions=1, seed=3)
    assert {trial.injected_at for trial in results.trials} == {0, 8, 16}
    for trial in results.trials:
        for target, trace in trial.trace.items():
            assert all(value == 0.0 for value in trace[: trial.injected_at]), (
                trial.source, target, trial.injected_at, trace[: trial.injected_at + 1]
            )
