"""Replay preserves source checkpoint selection and independently graded rows."""

import hashlib
import json

import pytest

from tools.evaluate_semantic_native_checkpoint import (
    digest,
    observed_program_reach,
    selected_checkpoint,
    source_calibration_labels,
    verified_calibration_replay_basis,
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


def test_source_calibration_does_not_teach_that_unknown_means_wrong():
    comparisons = [{"program_sha256": key, "status": status} for key, status in
                   (("a", "equivalent"), ("b", "different"), ("c", "unknown"))]
    bank = {"diagnosis": {"comparisons": comparisons}}
    assert source_calibration_labels(bank) == {"a": True, "b": False, "c": None}
    for rows in ([*comparisons, comparisons[0]],
                 [{"program_sha256": "a", "status": "assumed_correct"}]):
        with pytest.raises(ValueError, match="comparison"):
            source_calibration_labels({"diagnosis": {"comparisons": rows}})


def test_unknown_programs_do_not_establish_absence_of_an_equivalent():
    assert observed_program_reach((False, None)) is None
    assert observed_program_reach((True, None)) is True
    assert observed_program_reach((False, False)) is False
    row = replay_row()
    row.update(selected_correct=None, bank_reachable=None)
    row["receipt_sha256"] = digest({key: value for key, value in row.items()
                                    if key != "receipt_sha256"})
    assert verify_replay_row(row, source="held", plan_sha256="plan",
                            programs=("a", "b"), labels=(False, None)) == row
    row["bank_reachable"] = False
    row["receipt_sha256"] = digest({key: value for key, value in row.items()
                                    if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="independent bank"):
        verify_replay_row(row, source="held", plan_sha256="plan",
                          programs=("a", "b"), labels=(False, None))


def test_resumed_calibration_keeps_an_unproved_incumbent_outcome_unknown():
    row = replay_row()
    row["incumbent_correct"] = row["pretrained_correct"] = None
    row["receipt_sha256"] = digest({key: value for key, value in row.items()
                                    if key != "receipt_sha256"})
    assert verify_replay_row(row, source="held", plan_sha256="plan",
                             programs=("a", "b"), labels=(None, True)) == row
    with pytest.raises(ValueError, match="independent bank"):
        verify_replay_row(row, source="held", plan_sha256="plan",
                          programs=("a", "b"), labels=(False, True))


def test_first_candidate_is_not_an_incumbent_when_ordinary_decode_is_missing():
    row = replay_row()
    row.update(incumbent_available=False, incumbent_correct=False, pretrained_correct=True)
    row["receipt_sha256"] = digest({key: value for key, value in row.items()
                                    if key != "receipt_sha256"})
    assert verify_replay_row(row, source="held", plan_sha256="plan",
        programs=("a", "b"), labels=(True, True), incumbent_available=False) == row
    with pytest.raises(ValueError, match="independent bank"):
        verify_replay_row(row, source="held", plan_sha256="plan",
                          programs=("a", "b"), labels=(True, True))


def test_unscored_calibration_records_no_native_score_or_success():
    body = {"source": "source", "plan_sha256": "plan", "program_sha256s": [],
        "scores": [], "pretrained_scores": [], "chosen_program_sha256": None,
        "pretrained_program_sha256": None, "incumbent_correct": False,
        "incumbent_available": False, "selected_correct": None, "pretrained_correct": None,
        "bank_reachable": None, "labels_available_to_scorer": False, "scored": False,
        "unrankable_reason": "ordinary_decode_unavailable"}
    row = {**body, "receipt_sha256": digest(body)}
    assert verify_replay_row(row, source="source", plan_sha256="plan") == row
    for key, value in (("selected_correct", True), ("pretrained_correct", False),
                       ("incumbent_available", True), ("scores", [-1.]),
                       ("unrankable_reason", "not_checked")):
        changed = {**body, key: value}
        with pytest.raises(ValueError, match="unscored"):
            verify_replay_row({**changed, "receipt_sha256": digest(changed)},
                              source="source", plan_sha256="plan")


def test_calibration_replay_requires_complete_population_and_original_candidate(tmp_path):
    origin = tmp_path / "origin"
    source = tmp_path / "source"
    origin.mkdir()
    source.mkdir()
    candidate = b"immutable proposer"
    (origin / "candidate.json").write_bytes(candidate)
    base = {"schema": "aura.semantic_proposer_crossfit_plan.v1",
        "fit_ids": ["fit"], "calibration_ids": ["cal-a", "cal-b"], "held_ids": ["held"],
        "source_report_sha256": "source", "parent_receipt_sha256": "parent",
        "folds_sha256": "folds", "fold": 0, "input_order_policy": "source",
        "heldout_axis": "wording"}
    origin_plan = {**base, "plan_sha256": digest(base)}
    origin_body = {"schema": "aura.semantic_proposer_crossfit.v1",
        "plan_sha256": origin_plan["plan_sha256"], "row_receipts": {"held": "h"},
        "candidate_receipt_sha256": "candidate"}
    origin_report = {**origin_body, "receipt_sha256": digest(origin_body)}
    plan = {**base, "schema": "aura.semantic_proposer_source_calibration_plan.v1",
        "bank_partition": "source_calibration", "evaluated_ids": ["cal-a", "cal-b"],
        "reused_candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "held_rows_evaluated": False, "proposer_fit_updates": 0,
        "serving_authority": False, "qualification_evidence": False}
    write(source / "plan.json", plan, "plan_sha256")
    report = {"schema": "aura.semantic_proposer_source_calibration.v1",
        "plan_sha256": digest(plan), "candidate_receipt_sha256": "candidate",
        "source_calibration_population": 2, "row_receipts": {"cal-a": "a", "cal-b": "b"},
        "held_rows_evaluated": False, "proposer_fit_updates": 0,
        "serving_authority": False, "qualification_evidence": False}
    kwargs = dict(origin_directory=origin, origin_plan=origin_plan, origin_report=origin_report)
    with pytest.raises(FileNotFoundError):
        verified_calibration_replay_basis(source, **kwargs)
    write(source / "report.json", {**report, "row_receipts": {"cal-a": "a"}})
    with pytest.raises(ValueError, match="source calibration bank"):
        verified_calibration_replay_basis(source, **kwargs)
    write(source / "report.json", report)
    assert verified_calibration_replay_basis(source, **kwargs)[2] == ("cal-a", "cal-b")
    (origin / "candidate.json").write_bytes(b"substituted proposer")
    with pytest.raises(ValueError, match="source calibration bank"):
        verified_calibration_replay_basis(source, **kwargs)
