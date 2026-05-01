from __future__ import annotations

import json

from core.evaluation.behavioral_proof import (
    run_behavioral_proof_bundle,
    run_behavioral_proof_smoke,
    run_live_autonomy_loop_smoke,
)


def test_behavioral_proof_smoke_promotes_competent_solver(tmp_path):
    output = tmp_path / "behavioral.json"
    report = run_behavioral_proof_smoke(
        output_path=output,
        seed=12345,
        task_count=40,
    )

    assert report.passed is True
    assert report.candidate.result.score == 1.0
    assert report.baseline.result.score < report.candidate.result.score
    assert report.promotion.accepted is True
    assert report.leakage_controls["answer_hash_ok"] is True
    assert report.leakage_controls["public_manifest_excludes_answer_fields"] is True
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["candidate"]["result"]["passed"] == payload["task_count"]


def test_behavioral_proof_smoke_reproducible_for_same_seed():
    a = run_behavioral_proof_smoke(seed=6789, task_count=25)
    b = run_behavioral_proof_smoke(seed=6789, task_count=25)

    assert a.pack_id == b.pack_id
    assert a.manifest_hash == b.manifest_hash
    assert a.candidate.result.score == b.candidate.result.score
    assert a.baseline.result.score == b.baseline.result.score


def test_live_autonomy_loop_closes_goal_action_artifact_eval_memory(tmp_path):
    report = run_live_autonomy_loop_smoke(
        seed=2468,
        task_count=6,
        receipt_root=tmp_path / "receipts",
    )

    assert report.passed is True
    assert len(report.steps) == 6
    assert all(step.passed for step in report.steps)
    assert all(len(step.receipt_ids) == 4 for step in report.steps)
    assert report.loop_closure == {
        "internal_state_generated_goals": True,
        "actions_emitted_artifacts": True,
        "independent_evaluation_passed": True,
        "memory_updated_after_each_action": True,
        "future_policy_changed": True,
        "receipts_cover_each_step": True,
    }


def test_behavioral_proof_bundle_writes_smoke_and_live_loop(tmp_path):
    output = tmp_path / "bundle.json"
    bundle = run_behavioral_proof_bundle(
        output_path=output,
        smoke_seed=1357,
        live_loop_seed=2468,
        smoke_task_count=30,
        live_loop_task_count=6,
        receipt_root=tmp_path / "receipts",
    )

    assert bundle.passed is True
    assert bundle.smoke.passed is True
    assert bundle.live_loop.passed is True

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["smoke"]["passed"] is True
    assert payload["live_loop"]["passed"] is True
