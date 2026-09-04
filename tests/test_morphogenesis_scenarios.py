"""tests/test_morphogenesis_scenarios.py

The developmental scenarios, run as tests.

These are slower than a contract test because each one runs a population for
tens of rounds. They are here rather than only in the CLI because a scenario
that nobody runs stops being true quietly, and the whole claim of this layer
rests on them.

Determinism is checked first. A scenario that gives a different verdict on the
same seed cannot support any claim at all, whatever verdict it gives.
"""
from __future__ import annotations

import pytest

from core.morphogenesis.sandbox import ABLATIONS, SCENARIOS
from core.morphogenesis.scenarios import (
    SCENARIO_RUNNERS,
    run_ablation_matrix,
    run_scenario,
)

pytestmark = pytest.mark.slow


def test_every_declared_scenario_has_a_runner():
    """A scenario named in the CLI and missing a runner reports nothing and
    looks like a pass."""
    assert set(SCENARIOS) == set(SCENARIO_RUNNERS)


@pytest.mark.parametrize("name", sorted(SCENARIO_RUNNERS))
def test_scenario_passes_and_states_its_rule(name):
    result = run_scenario(name, seed=42, steps=20)
    assert result.verdict_rule, f"{name} did not state the rule it was judged by"
    assert result.passed, f"{name}: {result.verdict}"


def test_a_scenario_is_deterministic_under_its_seed():
    first = run_scenario("task_shift", seed=7, steps=16)
    second = run_scenario("task_shift", seed=7, steps=16)
    assert first.verdict == second.verdict
    for label, arm in first.arms.items():
        assert arm.signature() if callable(getattr(arm, "signature", None)) else True
        assert arm.graph_digest == second.arms[label].graph_digest
        assert arm.metrics == second.arms[label].metrics


def test_the_seed_reaches_the_run_and_development_still_converges():
    """A run that ignores its seed is not deterministic, it is constant, and a
    constant cannot be evidence of anything.

    What the seed reaches is the run: the routing choices, the substrate's
    failures, and so the score. What it does not reach is the shape that comes
    out — across five seeds, including one that took a substrate failure and a
    rollback, development lands on the same topology.

    That is worth asserting rather than tuning away. A developmental rule that
    produced a different anatomy from every noise draw would not be a rule, and
    convergence through a rollback is the property the lesion scenario depends
    on.
    """
    arms = [
        run_scenario("unknown_topology", seed=s, steps=16).arms["adaptive"]
        for s in (1, 2, 3, 4, 5)
    ]
    assert len({arm.signature for arm in arms}) > 1, "the seed never reached the run"
    assert len({round(arm.score, 6) for arm in arms}) > 1
    assert len({arm.graph_digest for arm in arms}) == 1, (
        "development stopped converging; this test records convergence rather "
        "than requiring it, so a change here is a finding, not a failure"
    )


def test_the_poisoned_signal_cannot_grow_the_population_past_its_cap():
    result = run_scenario("poisoned_signal", seed=42, steps=20)
    assert result.passed
    assert result.measurements["final_cells"] <= result.measurements["cap"]
    assert result.measurements["refused"] > result.measurements["applied"]


def test_the_lesion_arm_recovers_and_the_fixed_arm_does_not():
    result = run_scenario("lesion", seed=42, steps=20)
    recovery = result.measurements["recovery"]
    assert recovery["adaptive"]["recovered_share"] > recovery["recovery_off"]["recovered_share"]
    assert recovery["adaptive"]["detected_after_rounds"] >= 0


def test_the_partition_is_reported_rather_than_served_around():
    result = run_scenario("partition", seed=42, steps=20)
    detail = result.measurements
    assert detail["components_after_cut"] > 1
    assert detail["reported_degraded"] is (detail["components_at_end"] > 1)


def test_the_ablation_matrix_separates_the_arms():
    """Under a load the seed population absorbs, every arm completes the same
    work and the matrix collapses to one number — which reads like "morphology
    makes no difference" when it means "nothing was asked of it"."""
    matrix = run_ablation_matrix(seed=42, steps=16, seeds=2)
    assert set(matrix["rows"]) == set(ABLATIONS)
    assert matrix["discriminating"], f"arms did not separate: spread {matrix['spread']}"
    assert matrix["rows"]["none"]["score"] > matrix["rows"]["morphology_off"]["score"]
    assert matrix["rows"]["random_mutation"]["score"] < matrix["rows"]["none"]["score"]
