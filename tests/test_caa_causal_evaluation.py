from __future__ import annotations

import hashlib
import json

import pytest

from core.brain.llm.public_channel_decode import PUBLIC_CHANNEL_SAMPLE_POLICY
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
        "max_tokens": 256,
        "generation_policy": PUBLIC_CHANNEL_SAMPLE_POLICY,
        "generation_receipts": {
            name: [{
                "policy": PUBLIC_CHANNEL_SAMPLE_POLICY,
                "public_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_sha256": "f" * 64, "token_ids_sha256": "d" * 64,
                "reasoning_sha256": hashlib.sha256(b"").hexdigest(), "reasoning_chars": 0,
                "max_tokens": 256, "generated_tokens": 10,
                "native_thinking": False, "boundary_closed": True, "stop_reason": "eos",
            } for text in values] for name, values in conditions.items()
        },
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


def test_legacy_raw_channel_replays_but_cannot_be_new_qualification(tmp_path):
    result = _campaign()
    del result["generation_policy"]
    del result["generation_receipts"]
    replay = replay_campaign(result)
    assert replay["causal_effect_positive"]
    assert not replay["public_generation"]["verified"]
    metadata, _ = _metadata(tmp_path)
    evidence = build_independent_verifier_evidence(result=result, result_sha256="c" * 64,
        metadata=metadata, metadata_sha256="d" * 64, generation_dir=tmp_path)
    assert not evidence["verified"]


@pytest.mark.parametrize("change, error", [
    ("text", "public_text_drift"), ("receipt", "receipts_shape"),
    ("allocation", "allocation_mismatch"), ("boundary", "boundary_invalid"),
])
def test_public_generation_evidence_cannot_drift(change, error):
    result = _campaign()
    if change == "text":
        result["condition_outputs"]["baseline"][0] = "different"
    elif change == "receipt":
        result["generation_receipts"]["baseline"].pop()
    elif change == "allocation":
        result["generation_receipts"]["baseline"][0]["max_tokens"] = 128
    else:
        result["generation_receipts"]["baseline"][0]["boundary_closed"] = False
    with pytest.raises(CAACausalEvaluationError, match=error):
        replay_campaign(result)


def test_incomplete_public_sample_is_retained_and_refuses_qualification(tmp_path):
    result = _campaign()
    receipt = result["generation_receipts"]["baseline"][0]
    receipt.update(stop_reason="token_limit", generated_tokens=256)
    replay = replay_campaign(result)
    assert replay["sample_count"] == 30
    assert replay["public_generation"]["completed_per_condition"]["baseline"] == 29
    metadata, _ = _metadata(tmp_path)
    evidence = build_independent_verifier_evidence(result=result, result_sha256="c" * 64,
        metadata=metadata, metadata_sha256="d" * 64, generation_dir=tmp_path)
    assert evidence["verified"] is False


def test_adjudication_rejects_a_changed_result_after_verification(tmp_path):
    result = _campaign()
    metadata, _ = _metadata(tmp_path)
    evidence = build_independent_verifier_evidence(result=result, result_sha256="c" * 64,
        metadata=metadata, metadata_sha256="d" * 64, generation_dir=tmp_path)
    result["max_tokens"] = 1000
    with pytest.raises(CAACausalEvaluationError, match="independent_verifier_invalid"):
        build_causal_evaluation(result=result, metadata=metadata, verifier_evidence=evidence,
            verifier_evidence_sha256="e" * 64)
