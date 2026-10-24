"""A greater total win count does not establish superiority to a matched control."""
from __future__ import annotations

import pytest

from core.evaluation.fitted_steering_evaluation import (
    ARMS, LESIONS, MATCHED_CONTROL, RESULT_SCHEMA, TREATMENT,
    PAIRED_PROTOCOL, FittedEvaluationError, paired_qualification_plan, replay, replay_control_contrasts,
)
from core.evaluation.caa_causal_evaluation import canonical_sha256


def _record(treatment, random, shuffled, count):
    rows = []
    winners = {TREATMENT: treatment, MATCHED_CONTROL: random, LESIONS[1]: shuffled}
    for index in range(count):
        values = {"plain": -12.0}
        for arm in ARMS:
            values[f"{arm}:other"] = -10.0
            values[f"{arm}:own"] = -10.0 if arm == LESIONS[0] else -9.0 if index in winners[arm] else -11.0
        rows.append({"index": index, "tokens": 4, "loglik": values})
    return {"schema": RESULT_SCHEMA, "replies": rows}


def test_counts_can_pass_while_the_paired_layer_control_does_not():
    record = _record(set(range(66)), set(range(28)) | set(range(66, 89)),
                     set(range(55)) | set(range(66, 71)), 110)
    assert replay(record)["causal_effect_positive"]
    result = replay_control_contrasts(record)
    layer = result["contrasts"][LESIONS[1]]
    assert (layer["treatment_only"], layer["control_only"]) == (11, 5)
    assert layer["one_sided_exact_p"] == pytest.approx(0.1050567626953125)
    assert not layer["rejects"] and not result["all_control_contrasts_significant"]
    assert not result["serving_authority"] and not result["general_task_quality_measured"]


def test_a_reproducible_paired_advantage_survives_the_control_family():
    record = _record(set(range(36)), set(range(12)), set(range(10)), 40)
    result = replay_control_contrasts(record)
    assert result["all_control_contrasts_significant"]
    assert all(row["holm_adjusted_p"] >= row["one_sided_exact_p"] for row in result["contrasts"].values())


def test_duplicate_indices_cannot_multiply_one_replys_evidence():
    record = _record(set(range(36)), set(range(12)), set(range(10)), 40)
    record["replies"][1]["index"] = 0
    with pytest.raises(FittedEvaluationError, match="unique_reply"):
        replay_control_contrasts(record)


def test_identical_arms_cannot_show_a_paired_advantage():
    record = _record(set(range(36)), set(range(36)), set(range(36)), 40)
    result = replay_control_contrasts(record)
    assert not result["all_control_contrasts_significant"]
    assert result["contrasts"][MATCHED_CONTROL]["one_sided_exact_p"] == 1.0


def _v2(record):
    record.update(model_descriptor_sha256="checkpoint", generation_sha256="directions",
                  held_out_replies_sha256="replies", alpha=0.5, seed=7,
                  qualification_protocol=PAIRED_PROTOCOL)
    record["analysis_plan"] = paired_qualification_plan(record)
    record["analysis_plan_sha256"] = canonical_sha256(record["analysis_plan"])
    return record


def test_new_protocol_refuses_the_weak_control_result_that_legacy_accepts():
    record = _record(set(range(66)), set(range(28)) | set(range(66, 89)),
                     set(range(55)) | set(range(66, 71)), 110)
    assert replay(record)["causal_effect_positive"]
    revised = replay(_v2(record))
    assert not revised["causal_effect_positive"]
    assert "paired_control_superiority_not_established" in revised["unmet_requirements"]


def test_new_protocol_accepts_a_strong_paired_result():
    assert replay(_v2(_record(set(range(36)), set(range(12)), set(range(10)), 40)))["causal_effect_positive"]


def test_analysis_cannot_silently_change_its_inputs_or_correction():
    record = _v2(_record(set(range(36)), set(range(12)), set(range(10)), 40))
    record["seed"] += 1
    with pytest.raises(FittedEvaluationError, match="analysis_plan_mismatch"):
        replay(record)
    record["seed"] -= 1
    record["analysis_plan"]["correction"] = "none"
    record["analysis_plan_sha256"] = canonical_sha256(record["analysis_plan"])
    with pytest.raises(FittedEvaluationError, match="analysis_plan_mismatch"):
        replay(record)


def test_unknown_protocol_cannot_fall_back_to_legacy():
    record = _record(set(range(36)), set(range(12)), set(range(10)), 40)
    record["qualification_protocol"] = "unknown"
    with pytest.raises(FittedEvaluationError, match="protocol_unknown"):
        replay(record)
