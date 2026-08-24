from __future__ import annotations

import hashlib
import json

import pytest

from core.evaluation.caa_causal_evaluation import (
    CAACausalEvaluationError,
    build_causal_evaluation,
    build_independent_verifier_evidence,
    canonical_sha256,
    replay_campaign,
)


def _campaign() -> dict:
    baseline = ["plain response"] * 30
    treatment = ["happy curious response"] * 30
    conditions = {
        "steered_black_box": treatment,
        "baseline": baseline,
        "baseline_replicate": baseline,
        "text_terse": baseline,
        "text_rich_adversarial": baseline,
        "zero_vector": baseline,
        "random_vector": baseline,
        "shuffled_layers": baseline,
    }
    scores = {
        name: [2.0 if name == "steered_black_box" else 0.0] * 30
        for name in conditions
    }
    return {
        "model_descriptor_sha256": "a" * 64,
        "n_trials_per_task": 6,
        "held_out_tasks": [f"task-{index}" for index in range(5)],
        "condition_outputs": conditions,
        "target_scores": scores,
    }


def _metadata(tmp_path) -> tuple[dict, object]:
    vector = tmp_path / "valence_positive_layer1.npz"
    vector.write_bytes(b"bound-vector")
    manifest = [
        {
            "name": vector.name,
            "size_bytes": vector.stat().st_size,
            "sha256": hashlib.sha256(vector.read_bytes()).hexdigest(),
        }
    ]
    extraction = {"extraction_contract_sha256": "b" * 64}
    generation = canonical_sha256(
        {
            "extraction_contract_sha256": extraction["extraction_contract_sha256"],
            "vector_files": manifest,
        }
    )
    return (
        {
            "model_identity": {"model_descriptor_sha256": "a" * 64},
            "extraction_contract": extraction,
            "vector_files": manifest,
            "generation_sha256": generation,
        },
        vector,
    )


def test_independent_replay_builds_strict_causal_evaluation(tmp_path):
    result = _campaign()
    metadata, _vector = _metadata(tmp_path)
    evidence = build_independent_verifier_evidence(
        result=result,
        result_sha256="c" * 64,
        metadata=metadata,
        metadata_sha256="d" * 64,
        generation_dir=tmp_path,
    )
    evidence_payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    evaluation = build_causal_evaluation(
        result=result,
        metadata=metadata,
        verifier_evidence=evidence,
        verifier_evidence_sha256=hashlib.sha256(
            evidence_payload.encode("utf-8")
        ).hexdigest(),
    )

    assert evidence["verified"] is True
    assert evaluation["qualified"] is True
    assert evaluation["treatment_successes"] == 30
    assert evaluation["matched_control_successes"] == 0
    assert set(evaluation["lesion_successes"]) == {
        "random_vector",
        "shuffled_layers",
        "zero_vector",
    }


def test_replay_rejects_producer_score_tampering():
    result = _campaign()
    result["target_scores"]["steered_black_box"][0] = -100.0

    with pytest.raises(CAACausalEvaluationError, match="recorded_scores_mismatch"):
        replay_campaign(result)


def test_replay_requires_all_specificity_lesions():
    result = _campaign()
    del result["condition_outputs"]["random_vector"]

    with pytest.raises(CAACausalEvaluationError, match="outputs_incomplete"):
        replay_campaign(result)
