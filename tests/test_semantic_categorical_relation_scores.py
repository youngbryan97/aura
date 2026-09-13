"""A categorical relation decision must preserve its trained log-odds."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_transducer_fitting import (
    _log_sigmoid,
    _mention_invariant_relation_evidence,
)


@pytest.mark.parametrize("seed", range(6))
def test_categorical_margins_match_training_cross_entropy_without_inflating_mentions(seed):
    rng = np.random.default_rng(seed)
    base = tuple(rng.normal(size=7) * 10)
    logits = rng.normal(size=7) * 10
    log_probabilities = logits - np.logaddexp.reduce(logits)
    evidence = np.asarray(_mention_invariant_relation_evidence(
        base, tuple(logits), strategy="categorical_log_margin_v1",
    ))
    assert np.allclose(evidence - evidence[0], log_probabilities - log_probabilities[0], atol=1e-12)
    assert np.max(evidence) == pytest.approx(max(map(_log_sigmoid, base)), abs=1e-12)
    assert evidence.argmax() == logits.argmax()


def test_strong_competing_relations_do_not_saturate_to_the_same_score():
    evidence = _mention_invariant_relation_evidence(
        (2., 2.), (40., 60.), strategy="categorical_log_margin_v1",
    )
    assert evidence[1] - evidence[0] == pytest.approx(20.)


def test_legacy_score_is_bitwise_unchanged():
    base, logits = (1., .5, -2.), (-2., 4., 1.)
    old = tuple(map(_log_sigmoid, logits))
    expected = tuple(max(map(_log_sigmoid, base)) + x - max(old) for x in old)
    assert _mention_invariant_relation_evidence(base, logits) == expected


def test_unknown_objective_cannot_be_treated_as_the_frozen_rule():
    with pytest.raises(ValueError, match="unknown semantic relation"):
        _mention_invariant_relation_evidence((0.,), (0.,), strategy="typo")


def test_candidate_identity_and_lesions_preserve_the_declared_objective():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_categorical_relation_scores()
    assert "relation_score_strategy" not in parent.training_receipt
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.coefficient_lesion().training_receipt["relation_score_strategy"] == "categorical_log_margin_v1"
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()
