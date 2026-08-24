from __future__ import annotations

from tools.adjudicate_semantic_neural_wow import (
    EXPECTED_FAMILIES,
    MODEL_BOUND_CLAIM,
    _sha,
)


def test_adjudication_receipt_hash_is_canonical():
    body = {
        "schema": "aura.rlc.semantic_neural_bounded_wow_adjudication.v1",
        "passed": True,
        "families": EXPECTED_FAMILIES,
    }
    assert _sha(body) == _sha(dict(reversed(tuple(body.items()))))


def test_expected_family_contract_covers_all_four_domains():
    assert EXPECTED_FAMILIES == (
        "frontier_coding",
        "frontier_calibration",
        "frontier_misleading_premise",
        "frontier_scientific_inference",
    )


def test_adjudication_claim_is_model_bound_without_relabeling_a_size():
    assert "resident-model" in MODEL_BOUND_CLAIM
    assert "32B" not in MODEL_BOUND_CLAIM
    assert "27B" not in MODEL_BOUND_CLAIM
