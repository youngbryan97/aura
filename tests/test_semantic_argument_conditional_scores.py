"""Binary edge labels conditioned on exactly one selected argument mention."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_transducer_fitting import (
    _argument_semantic_evidence,
    _log_sigmoid,
)


@pytest.mark.parametrize("seed", range(6))
def test_log_odds_match_conditioned_joint_binary_likelihood(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(2, 7)) * 4
    scales = [1.0, 0.875]
    joint = []
    for chosen in range(7):
        joint.append(sum(
            scale * sum(_log_sigmoid(value if i == chosen else -value)
                        for i, value in enumerate(head))
            for head, scale in zip(logits, scales, strict=True)
        ))
    conditional = np.array([
        _argument_semantic_evidence(role, proposal, role_scale=scales[0],
            proposal_scale=scales[1], strategy="conditional_log_odds_v1")
        for role, proposal in zip(*logits, strict=True)
    ])
    assert np.allclose(conditional - conditional[0], np.array(joint) - joint[0], atol=1e-12)


@pytest.mark.parametrize("role,proposal", [(1000., 2000.), (-1000., -2000.), (4.6, -2.3)])
def test_legacy_positive_score_is_unchanged(role, proposal):
    assert _argument_semantic_evidence(role, proposal, role_scale=1., proposal_scale=.875,
        strategy="independent_positive_v1") == _log_sigmoid(role) + .875 * _log_sigmoid(proposal)


def test_conditional_scores_do_not_saturate_strong_competing_evidence():
    assert _argument_semantic_evidence(20., 20., role_scale=1., proposal_scale=1.,
        strategy="conditional_log_odds_v1") == 40.
    assert _argument_semantic_evidence(40., 40., role_scale=1., proposal_scale=1.,
        strategy="conditional_log_odds_v1") == 80.


def test_unknown_scoring_rule_is_not_silently_treated_as_legacy():
    with pytest.raises(ValueError, match="unknown semantic argument"):
        _argument_semantic_evidence(0., 0., role_scale=1., proposal_scale=1., strategy="typo")


def test_scoring_change_is_opt_in_and_receipt_bound():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_global_constraint_arguments().with_conditional_argument_scores()
    assert "argument_score_strategy" not in parent.training_receipt
    assert candidate.training_receipt["argument_score_strategy"] == "conditional_log_odds_v1"
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.receipt_sha256 == candidate.receipt_sha256
