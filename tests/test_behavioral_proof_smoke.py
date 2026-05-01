from __future__ import annotations

import json

from core.evaluation.behavioral_proof import run_behavioral_proof_smoke


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
