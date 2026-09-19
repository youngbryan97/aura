"""Separate a score representation's feasible witnesses from optimizer loss."""

import numpy as np
import pytest

from core.learning.semantic_graph_margin import graph_scale_feasibility, fit_graph_score_scales


def test_feasible_scales_replay_every_inequality():
    differences = np.eye(3)
    offsets = np.array([-.9, -1.9, -2.9])
    receipt = graph_scale_feasibility(differences, offsets)
    assert receipt["status"] == "verified_witness"
    assert np.all(differences @ receipt["scales"] + offsets >= .1)
    assert not receipt["serving_authority"]
    assert not receipt["fresh_transfer_claim"]


def test_incompatible_constraints_are_not_called_an_optimizer_failure():
    from core.learning.score_capacity import verify_score_capacity

    receipt = graph_scale_feasibility([[1., 0., 0.], [-1., 0., 0.]], [0., 0.])
    assert receipt["status"] == "exactly_infeasible"
    assert receipt["scales"] is None
    assert receipt["exact_infeasibility_proven"]
    assert verify_score_capacity(receipt["capacity_certificate"])


def test_graph_scale_infeasibility_stays_numerical_without_kernel_proof(monkeypatch):
    monkeypatch.setattr("core.learning.score_capacity.find_farkas", lambda _: None)
    receipt = graph_scale_feasibility([[1., 0., 0.], [-1., 0., 0.]], [0., 0.])
    assert receipt["status"] == "numerically_infeasible"
    assert not receipt["exact_infeasibility_proven"]


def test_historical_factor_conflict_prevents_even_strict_ranking():
    from core.learning.score_capacity import assess_score_capacity, verify_score_capacity

    # Source-training contrasts 35, 144, 548, 551 from the 2026-09-15 factor run.
    differences = [[0., -2.3021699339151387, 0.],
        [-1.426961898803711, -2.8313596665508927, -5.8164209446680815],
        [.1068115234375, 5.771680553127132, 2.823387648594914],
        [.3038291931152344, 4.80948348767131, -2.860594416105453]]
    offsets = [4.440892098500626e-16, 3.841851115226743, -1.569742619991299, 1.121928840875622]
    receipt = assess_score_capacity(differences, offsets, margin=0., strict=True,
                                    lower_bounds=[1e-6] * 3)
    assert receipt["status"] == "infeasible" and verify_score_capacity(receipt)
    assert all(index < 4 for index, _ in receipt["certificate"]["multipliers"])


def test_existing_fit_reports_representability_without_replacing_learned_scales():
    scales, receipt = fit_graph_score_scales(np.eye(3), np.zeros(3), np.ones(3), np.ones(3))
    assert receipt["representability"]["status"] == "verified_witness"
    assert receipt["fitted_scales"] == scales.tolist()


@pytest.mark.parametrize("differences,offsets", [([], []), ([[1., 2.]], [0.]),
    ([[np.nan, 0., 0.]], [0.]), ([[1., 0., 0.]], [np.inf])])
def test_invalid_evidence_is_not_a_feasibility_result(differences, offsets):
    with pytest.raises(ValueError):
        graph_scale_feasibility(differences, offsets)
