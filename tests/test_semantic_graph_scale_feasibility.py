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
    receipt = graph_scale_feasibility([[1., 0., 0.], [-1., 0., 0.]], [0., 0.])
    assert receipt["status"] == "numerically_infeasible"
    assert receipt["scales"] is None
    assert not receipt["exact_infeasibility_proven"]


def test_existing_fit_reports_representability_without_replacing_learned_scales():
    scales, receipt = fit_graph_score_scales(np.eye(3), np.zeros(3), np.ones(3), np.ones(3))
    assert receipt["representability"]["status"] == "verified_witness"
    assert receipt["fitted_scales"] == scales.tolist()


@pytest.mark.parametrize("differences,offsets", [([], []), ([[1., 2.]], [0.]),
    ([[np.nan, 0., 0.]], [0.]), ([[1., 0., 0.]], [np.inf])])
def test_invalid_evidence_is_not_a_feasibility_result(differences, offsets):
    with pytest.raises(ValueError):
        graph_scale_feasibility(differences, offsets)
