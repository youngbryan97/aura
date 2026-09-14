from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_semantic_neural_decode_canary import (
    REPO_ROOT,
    _sha,
    verify_canary,
)

ARTIFACT = REPO_ROOT / "artifacts/closeout/latent_cortex/cp550_semantic_decode_canary/result.json"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["model_identity"] = {
        "path": str(model.resolve()),
        "config_sha256": _file_sha(model / "config.json"),
        "weights_index_sha256": _file_sha(model / "model.safetensors.index.json"),
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact, model


def _write_journal(artifact: Path, destination: Path) -> None:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    previous = "0" * 64
    events = [
        {
            "event": "campaign_started",
            "source_commit": payload["source_commit"],
            "seed": payload["seed"],
            "tasks_per_difficulty": payload["tasks_per_difficulty"],
            "task_count": payload["task_count"],
            "arm_count": 5,
        }
    ]
    for field in ("domains", "difficulties", "surface_profile", "resident_manifest_identity",
                  "decode_policy", "grading_policy", "max_tokens"):
        if field in payload:
            events[0][field] = payload[field]
    for index, raw_output in enumerate(payload["raw_outputs"], start=1):
        events.append(
            {
                "event": "decode_committed",
                "completed": index,
                "total": len(payload["raw_outputs"]),
                "row": payload["rows"][index - 1] if "rows" in payload else {
                    "task_id": raw_output["task_id"],
                    "arm": raw_output["arm"],
                    "response_sha256": hashlib.sha256(raw_output["response"].encode()).hexdigest(),
                },
                "raw_output": raw_output,
            }
        )
    events.append(
        {
            "event": "campaign_completed",
            "admitted": payload["admitted"],
            "report_receipt_sha256": payload["receipt_sha256"],
        }
    )
    lines = []
    last_decode_receipt = ""
    for index, event in enumerate(events):
        body = {
            "schema": "aura.rlc.semantic_neural_decode_journal.v1",
            "previous_receipt_sha256": previous,
            **event,
        }
        receipt = _sha(body)
        lines.append(json.dumps({**body, "receipt_sha256": receipt}, sort_keys=True))
        previous = receipt
        if index == len(events) - 2:
            last_decode_receipt = receipt
    payload["journal_last_decode_receipt_sha256"] = last_decode_receipt
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    completed = json.loads(lines[-1])
    completed["report_receipt_sha256"] = payload["receipt_sha256"]
    completed_body = {key: value for key, value in completed.items() if key != "receipt_sha256"}
    completed["receipt_sha256"] = _sha(completed_body)
    lines[-1] = json.dumps(completed, sort_keys=True)
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _public_fixture(tmp_path, monkeypatch):
    """Synthetic receipts test verifier integrity; they are not model evidence."""
    from core.brain.llm.public_channel_decode import (
        PUBLIC_CHANNEL_DECODE_POLICY,
        PublicChannelDecode,
    )
    from core.learning.public_decode_evidence import public_decode_coverage
    from core.learning.semantic_task_grading import SEMANTIC_GRADING_POLICY
    from tools.run_semantic_neural_decode_canary import (
        PUBLIC_DECODE_SOURCE_PATHS,
        SOURCE_PATHS,
        _summary,
    )
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_bytes())
    payload.update(schema="aura.rlc.semantic_neural_decode_canary.v2",
                   decode_policy=PUBLIC_CHANNEL_DECODE_POLICY, grading_policy=SEMANTIC_GRADING_POLICY,
                   max_tokens=4096,
                   source_sha256s={path: "a" * 64 for path in (*SOURCE_PATHS, *PUBLIC_DECODE_SOURCE_PATHS)})
    monkeypatch.setattr("tools.verify_semantic_neural_decode_canary._git_blob_sha", lambda *_: "a" * 64)
    from tools.verify_semantic_neural_decode_canary import (
        LEGACY_DIFFICULTIES,
        LEGACY_DOMAINS,
        _expected_tasks,
    )
    tasks = {task.task_id: task for task in _expected_tasks(
        payload["seed"], payload["tasks_per_difficulty"], LEGACY_DOMAINS, LEGACY_DIFFICULTIES, "canonical",
    )}
    rows = []
    for output in payload["raw_outputs"]:
        text = output["response"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        decode = PublicChannelDecode(text, 32, 0, 0, 10, "eos", True, True, 0,
                                     hashlib.sha256(b"").hexdigest(), "a" * 64).receipt()
        output["attempts"] = [{"attempt": 1, "raw_response": text, "raw_response_sha256": digest,
                               "response": text, "response_sha256": digest, "prefill_tokens": 0,
                               "prompt_tokens": 16, "decode": decode}]
        grade = tasks[output["task_id"]].grade(text)
        rows.append({"task_id": output["task_id"], "arm": output["arm"], "response_sha256": digest,
                     "correct": grade["correct"], "parsed": grade["parsed"] is not None,
                     "decode_attempts": 1, "generated_tokens": 32, "prompt_tokens": 16,
                     "latency_ms": 10, "stopped": True})
    payload["rows"] = rows
    payload["arms"] = {arm: _summary(rows, arm) for arm in payload["arms"]}
    payload["decode_coverage"] = public_decode_coverage(payload["raw_outputs"], max_tokens=4096)
    _reseal(artifact, payload)
    journal = tmp_path / "journal.jsonl"
    _write_journal(artifact, journal)
    return artifact, model, journal


def _reseal(path, payload):
    payload["receipt_sha256"] = _sha({key: value for key, value in payload.items() if key != "receipt_sha256"})
    path.write_text(json.dumps(payload))


def test_public_verifier_checks_policy_and_all_attempts(tmp_path, monkeypatch):
    artifact, model, journal = _public_fixture(tmp_path, monkeypatch)
    report = verify_canary(artifact, model_path=model, journal_path=journal)
    assert report["verified"]
    assert report["decode_coverage"]["uncensored"]
    assert report["grading_policy"] == "semantic_values_v2"
    with pytest.raises(RuntimeError, match="policy, budget or journal"):
        verify_canary(artifact, model_path=model)


@pytest.mark.parametrize("mutation", ["private_predicate", "censor", "count", "policy", "downgrade", "journal_row"])
def test_public_verifier_rejects_resealed_bad_measurement(tmp_path, monkeypatch, mutation):
    artifact, model, journal = _public_fixture(tmp_path, monkeypatch)
    payload = json.loads(artifact.read_bytes())
    attempt = payload["raw_outputs"][0]["attempts"][0]
    if mutation == "private_predicate":
        attempt["decode"].update(boundary_closed=False, stop_reason="public_contract")
    elif mutation == "censor":
        attempt["decode"].update(stop_reason="token_limit", generated_tokens=4096)
        from core.learning.public_decode_evidence import public_decode_coverage
        payload["decode_coverage"] = public_decode_coverage(payload["raw_outputs"], max_tokens=4096)
        payload["rows"][0].update(generated_tokens=4096, stopped=False)
    elif mutation == "count":
        payload["rows"][0]["generated_tokens"] += 1
    elif mutation == "policy":
        payload["grading_policy"] = "exact_wire_v1"
    elif mutation == "downgrade":
        payload["schema"] = "aura.rlc.semantic_neural_decode_canary.v1"
    else:
        payload["rows"][0]["unrecorded"] = True
    _reseal(artifact, payload)
    if mutation != "journal_row":
        _write_journal(artifact, journal)
    with pytest.raises((ValueError, RuntimeError)):
        verify_canary(artifact, model_path=model, journal_path=journal)


def test_semantic_decode_verifier_regrades_and_replays_frozen_canary(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    report = verify_canary(artifact, model_path=model)
    assert report["verified"] is True
    assert report["independent_exact_by_arm"] == {
        "ordinary_base": 0,
        "matched_wire_base": 0,
        "treatment": 27,
        "coefficient_lesion": 9,
        "matched_wrong_state": 0,
    }
    assert report["gain_count"] == 27
    assert report["regression_count"] == 0
    assert report["treatment_state_replay_count"] == 27
    assert report["paired_discordant_count"] == 27
    assert report["paired_one_sided_exact_p"] == pytest.approx(2**-27)


def test_negative_integrity_audit_never_grants_qualification(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_bytes())
    treatment = {row["task_id"]: row["response"] for row in payload["raw_outputs"]
                 if row["arm"] == "treatment"}
    for row in payload["raw_outputs"]:
        if row["arm"] == "ordinary_base":
            row["response"] = treatment[row["task_id"]]
            row["response_sha256"] = hashlib.sha256(row["response"].encode()).hexdigest()
    payload["arms"]["ordinary_base"].update(exact=27, parsed=27, exact_accuracy=1.0, parsed_accuracy=1.0)
    summary = payload["arms"]["ordinary_base"]
    summary["receipt_sha256"] = _sha({key: value for key, value in summary.items() if key != "receipt_sha256"})
    payload.update(gain_count=0, gain_set_sha256=_sha([]), admitted=False)
    _reseal(artifact, payload)
    with pytest.raises(RuntimeError, match="admission failed"):
        verify_canary(artifact, model_path=model)
    journal = tmp_path / "journal.jsonl"
    _write_journal(artifact, journal)
    report = verify_canary(artifact, model_path=model, journal_path=journal, require_admission=False)
    assert report["integrity_verified"] is True
    assert report["verified"] is False
    assert report["admitted"] is False
    assert report["paired_discordant_count"] == 0
    assert report["paired_one_sided_exact_p"] == 1.0
    assert report["journal_decode_count"] == 135
    payload = json.loads(artifact.read_bytes())
    payload["admitted"] = True
    _reseal(artifact, payload)
    with pytest.raises(RuntimeError, match="claimed admission"):
        verify_canary(artifact, model_path=model, require_admission=False)


def test_semantic_decode_verifier_checks_receipt_chained_journal(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    journal = tmp_path / "journal.jsonl"
    _write_journal(artifact, journal)
    report = verify_canary(artifact, model_path=model, journal_path=journal)
    assert report["journal_event_count"] == 137
    assert report["journal_decode_count"] == 135

    lines = journal.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[20])
    tampered["row"]["task_id"] = "tampered"
    lines[20] = json.dumps(tampered, sort_keys=True)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt chain broke"):
        verify_canary(artifact, model_path=model, journal_path=journal)


def test_semantic_decode_verifier_binds_resident_manifest(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    manifest = tmp_path / "active.json"
    manifest_payload = {
        "active_model_path": str(model.resolve()),
        "base_model": "base",
        "fused_at": 1,
        "schema_version": 2,
        "tag": "resident-test",
    }
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["resident_manifest_identity"] = {
        "path": str(manifest.resolve()),
        "sha256": _file_sha(manifest),
        "active_model_path": str(model.resolve()),
        "schema_version": 2,
        "base_model": "base",
        "tag": "resident-test",
        "fused_at": 1,
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    report = verify_canary(
        artifact,
        model_path=model,
        resident_manifest_path=manifest,
    )
    assert report["resident_manifest_identity"]["tag"] == "resident-test"
    assert "resident model bound by" in report["claim_boundary"]
    assert report["producer_claim_boundary_legacy"] is True

    manifest_payload["tag"] = "changed"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resident manifest identity mismatch"):
        verify_canary(
            artifact,
            model_path=model,
            resident_manifest_path=manifest,
        )


def test_semantic_decode_verifier_rejects_resealed_response_tamper(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = next(row for row in payload["raw_outputs"] if row["arm"] == "treatment")
    row["response"] = "FINAL_ANSWER: {}"
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary disagrees with raw output"):
        verify_canary(artifact, model_path=model)


def test_semantic_decode_verifier_independently_checks_declared_lesions(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["coefficient_lesion_contract"] = {
        "frontier_coding": {
            "operation": "addition",
            "operation_index": 0,
            "coefficient_index": 1,
        },
        "frontier_calibration": {
            "operation": "multiplication",
            "operation_index": 1,
            "coefficient_index": 2,
        },
        "frontier_misleading_premise": {
            "operation": "multiplication",
            "operation_index": 1,
            "coefficient_index": 2,
        },
    }
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_canary(artifact, model_path=model)["coefficient_lesion_contract_verified"] is True

    payload["coefficient_lesion_contract"]["frontier_coding"]["coefficient_index"] = 2
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="coefficient lesion contract mismatch"):
        verify_canary(artifact, model_path=model)


def test_semantic_decode_verifier_rejects_resealed_undeclared_cohort(tmp_path):
    artifact, model = _portable_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["domains"] = ["coding", "coding"]
    payload["difficulties"] = [1, 2, 3]
    payload["receipt_sha256"] = _sha(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cohort identity is invalid"):
        verify_canary(artifact, model_path=model)
