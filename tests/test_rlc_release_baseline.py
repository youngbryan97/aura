"""Recount the retained G01 evidence without granting new serving authority."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "docs/evidence/G01_RLC_BASELINE_2026-09-08.json"
BASELINE = json.loads(BASELINE_PATH.read_text())
ARTIFACTS = BASELINE["artifacts"]


def _artifact(name: str) -> dict:
    record = ARTIFACTS[name]
    payload = (ROOT / record["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], name
    value = json.loads(payload)
    assert value["schema"] == record["schema"], name
    return value


@pytest.mark.parametrize("name", tuple(ARTIFACTS))
def test_evidence_bytes_remain_the_frozen_record(name):
    _artifact(name)


def test_baseline_names_the_source_snapshot_without_licensing_current_code():
    assert BASELINE["schema"] == "aura.rlc.release_baseline.v1"
    assert BASELINE["purpose"] == "retained_evidence_baseline_not_new_benchmark"
    assert BASELINE["current_live_activation"] == "not_measured_by_this_baseline"
    for path, digest in BASELINE["source_sha256s"].items():
        payload = subprocess.run(
            ["git", "show", f"{BASELINE['source_commit']}:{path}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=20,
        ).stdout
        assert hashlib.sha256(payload).hexdigest() == digest, path


@pytest.mark.parametrize("lane", ("32b", "27b"))
def test_raw_decodes_regrade_to_the_frozen_five_arm_counts(lane):
    from tools.verify_semantic_neural_decode_canary import _expected_tasks

    report = _artifact(f"{lane}_result")
    expected = BASELINE["measurements"][lane]
    tasks = _expected_tasks(
        report["seed"],
        report["tasks_per_difficulty"],
        tuple(report["domains"]),
        tuple(report["difficulties"]),
        report["surface_profile"],
    )
    by_id = {task.task_id: task for task in tasks}
    assert len(by_id) == len(tasks) == expected["total"]
    pairs = [(row["task_id"], row["arm"]) for row in report["raw_outputs"]]
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == {(task_id, arm) for task_id in by_id for arm in expected["exact_by_arm"]}
    counts = Counter({arm: 0 for arm in expected["exact_by_arm"]})
    for row in report["raw_outputs"]:
        correct = by_id[row["task_id"]].grade(row["response"])["correct"]
        assert type(correct) is bool
        counts[row["arm"]] += int(correct)
    assert dict(counts) == expected["exact_by_arm"]
    assert {arm: values["exact"] for arm, values in report["arms"].items()} == counts
    assert _artifact(f"{lane}_adjudication")["verdict"] == "BOUNDED_WOW_SIGNAL"


def test_configured_model_observation_matches_27b_evidence_not_32b():
    current = BASELINE["configured_model"]
    evidence = _artifact("27b_adjudication")["model_identity"]
    assert current["path"] == evidence["path"]
    files = current["observed_file_sha256s"]
    assert files["config.json"] == evidence["config_sha256"]
    assert files["model.safetensors.index.json"] == evidence["weights_index_sha256"]
    assert current["path"] != _artifact("32b_result")["model_identity"]["path"]
    assert current["weight_bytes_rehashed"] is False


def test_withheld_family_rows_preserve_both_failures_and_lesions():
    family = _artifact("v14_family")
    assert family["held_out_family_was_available_to_fit"] is False
    assert family["source_fresh_example_overlap"] == 0
    assert family["source_fresh_text_overlap"] == 0
    report = _artifact("v14_endogenous")
    assert report["serving_authority"] is False
    expected = BASELINE["measurements"]["v14_endogenous"]
    cohort = report["cohorts"]["sequence_binary"]
    for arm, target in (
        ("treatment", expected["answer_exact"]),
        ("lesion", expected["lesion_answer_exact"]),
    ):
        rows = cohort[arm]["rows"]
        assert len(rows) == expected["total"]
        assert len({row["example_id"] for row in rows}) == len(rows)
        assert all(type(row["answer_exact"]) is bool for row in rows)
        assert sum(row["answer_exact"] for row in rows) == target
        assert cohort[arm]["answer_exact"] == target
    assert cohort["treatment"]["program_exact"] == expected["program_exact"]


def test_failed_replication_is_not_reclassified_by_lesion_significance():
    report = _artifact("v16_result")
    prereg = _artifact("v16_preregistration")
    expected = BASELINE["measurements"]["v16_natural"]
    for metric in ("total", "answer_exact", "program_exact"):
        assert report["treatment"][metric] == expected[metric]
    assert report["preregistered_minimum_answer_exact"] == expected["minimum_answer_exact"]
    assert (
        "frozen_v15_treatment has at least 48 exact answers of 96"
        in (prereg["evaluation"]["success_conditions"])
    )
    assert expected["answer_exact"] < expected["minimum_answer_exact"]
    assert report["verdict"] == expected["verdict"] == "FAIL_ABSOLUTE_CAPABILITY_FLOOR"
    assert report["ordinary_decode"]["status"] == expected["ordinary_decode_status"]
    assert report["ordinary_decode"]["model_load_or_decode_calls"] == 0
    assert report["serving_authority"] is False


def test_development_repair_recounts_but_does_not_become_fresh_replication():
    report = _artifact("v19_development")
    expected = BASELINE["measurements"]["v19_development"]
    for arm, target in (
        ("held_out_natural_development", expected["answer_exact"]),
        ("coefficient_lesion", expected["lesion_answer_exact"]),
    ):
        evaluation = report[arm]
        rows = [row for split in evaluation["splits"].values() for row in split["rows"]]
        assert len(rows) == expected["total"]
        assert len({row["source_text_sha256"] for row in rows}) == len(rows)
        assert all(type(row["answer_exact"]) is bool for row in rows)
        assert sum(row["answer_exact"] for row in rows) == target
        assert evaluation["held_out_answer_exact"] == target
    assert (
        report["held_out_natural_development"]["held_out_program_exact"]
        == expected["program_exact"]
    )
    assert expected["fresh_replication"] is False
    assert report["serving_authority"] is False


def test_baseline_cannot_grant_unmeasured_claims():
    claims = BASELINE["claim_status"]
    assert claims["serving_authority"] is False
    assert claims["new_model_benchmark_run"] is False
    for claim in ("broad_reasoning_gain", "frontier_performance", "static_rlc_weight_fusion"):
        assert claims[claim] == "not_established"
    assert claims["current_development_repair"] == "requires_fresh_replication"
