"""Replay preserves source checkpoint selection and independently graded rows."""

import hashlib
import json

import pytest

from tools.evaluate_semantic_native_checkpoint import (
    digest,
    selected_checkpoint,
    verified_document,
    verify_replay_row,
)


def write(path, body, field="receipt_sha256"):
    path.write_text(json.dumps({**body, field: digest(body)}, sort_keys=True))


def campaign(root, *, baseline=True):
    plan = {"schema": "aura.semantic_native_fit_plan.v1", "steps": 4, "save_every": 2,
        "held_labels_used_for_fit_or_selection": False, "serving_authority": False,
        "qualification_evidence": False, "fit_ids": ["fit"], "calibration_ids": ["cal"],
        "held_ids": ["held"], "unfitted_checkpoint_eligible": baseline}
    write(root / "plan.json", plan, "plan_sha256")
    for step in ([0, 2, 4] if baseline else [2, 4]):
        weights = f"weights-{step}".encode()
        (root / f"checkpoint-{step}.safetensors").write_bytes(weights)
        write(root / f"checkpoint-{step}.json", {"plan_sha256": digest(plan), "step": step,
            "calibration_loss": {0: 1., 2: 2., 4: .5}[step],
            "weights_sha256": hashlib.sha256(weights).hexdigest()})
    return plan


@pytest.mark.parametrize("baseline", [False, True])
def test_complete_calibration_selects_without_reading_held_rows(tmp_path, baseline):
    campaign(tmp_path, baseline=baseline)
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "held.json").write_text("invalid held labels are never read")
    plan, selected = selected_checkpoint(tmp_path)
    assert selected["step"] == 4
    assert plan["held_ids"] == ["held"]


def test_unfitted_checkpoint_can_win_instead_of_forcing_an_update(tmp_path):
    campaign(tmp_path)
    row = verified_document(tmp_path / "checkpoint-4.json")
    body = {key: value for key, value in row.items() if key != "receipt_sha256"}
    body["calibration_loss"] = 3.
    write(tmp_path / "checkpoint-4.json", body)
    assert selected_checkpoint(tmp_path)[1]["step"] == 0


@pytest.mark.parametrize("defect", ["missing", "weights", "partition", "plan", "report"])
def test_replay_refuses_incomplete_or_substituted_training(tmp_path, defect):
    plan = campaign(tmp_path)
    if defect == "missing":
        (tmp_path / "checkpoint-4.json").unlink()
    elif defect == "weights":
        (tmp_path / "checkpoint-4.safetensors").write_bytes(b"different")
    elif defect == "partition":
        plan["held_ids"] = ["fit"]
        write(tmp_path / "plan.json", plan, "plan_sha256")
    elif defect == "plan":
        (tmp_path / "plan.json").write_text(json.dumps({**plan, "plan_sha256": "wrong"}))
    else:
        write(tmp_path / "report.json", {"plan_sha256": digest(plan), "selected_step": 2,
                                         "selected_calibration_loss": 2.})
    with pytest.raises(ValueError):
        selected_checkpoint(tmp_path)


def replay_row():
    body = {"source": "held", "plan_sha256": "plan", "program_sha256s": ["a", "b"],
        "scores": [-2., -1.], "pretrained_scores": [-1., -2.],
        "chosen_program_sha256": "b", "pretrained_program_sha256": "a",
        "labels_available_to_scorer": False, "incumbent_correct": False,
        "selected_correct": True, "pretrained_correct": False, "bank_reachable": True}
    return {**body, "receipt_sha256": digest(body)}


def test_resumed_rows_match_scores_and_independent_bank_labels():
    row = replay_row()
    assert verify_replay_row(row, source="held", plan_sha256="plan",
                             programs=("a", "b"), labels=(False, True)) == row
    with pytest.raises(ValueError, match="independent bank"):
        verify_replay_row(row, source="held", plan_sha256="plan",
                           programs=("a", "b"), labels=(True, False))


@pytest.mark.parametrize("field,value", [
    ("source", "other"), ("plan_sha256", "other"),
    ("chosen_program_sha256", "a"), ("pretrained_program_sha256", "b"),
    ("scores", [-1., -2.]), ("labels_available_to_scorer", True),
    ("program_sha256s", ["a", "a"])])
def test_replay_rejects_rehashed_score_identity_changes(field, value):
    row = replay_row()
    row[field] = value
    row["receipt_sha256"] = digest({key: item for key, item in row.items()
                                    if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        verify_replay_row(row, source="held", plan_sha256="plan")
