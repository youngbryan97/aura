"""A proposer holdout excludes whole source constructions before fitting."""

import json
from types import SimpleNamespace

import pytest

from core.learning.semantic_construction_folds import (
    construction_folds,
    utterance_construction_folds,
)
from core.learning.semantic_program_campaign import _sha
from tools.audit_semantic_proposer_reach import audit_directory
from tools.probe_semantic_proposer_crossfit import (
    _digest,
    bank_measurement_population,
    crossfit_partition,
    nested_crossfit_partition,
    proposal_reach_profile,
    verify_source_calibration_bank,
)


def test_source_calibration_bank_keeps_source_partition_and_never_reads_held_targets():
    def row(identity):
        return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity))
    fit, cal, held = [row("fit")], [row("cal-b"), row("cal-a")], [row("held")]
    assert bank_measurement_population(fit, cal, held, partition="source_calibration") == cal[::-1]
    assert bank_measurement_population(fit, cal, held, partition="held") == held
    for parts in ((fit, fit, held), (fit, cal, cal), (fit, [], held),
                  (fit, [*cal, cal[0]], held)):
        with pytest.raises(ValueError, match="partitions"):
            bank_measurement_population(*parts, partition="source_calibration")
    with pytest.raises(ValueError, match="partition"):
        bank_measurement_population(fit, cal, held, partition="train")


def test_source_calibration_bank_cannot_be_relabelled_as_held_or_change_its_proposer():
    import copy

    def seal(body, field):
        return {**body, field: _digest(body)}
    origin = seal({"schema": "aura.semantic_proposer_crossfit_plan.v1",
                   "fit_ids": ["fit"], "calibration_ids": ["cal-a", "cal-b"],
                   "held_ids": ["held"], "source_report_sha256": "source",
                   "parent_receipt_sha256": "parent", "folds_sha256": "folds", "fold": 0,
                   "input_order_policy": "source", "heldout_axis": "wording"}, "plan_sha256")
    origin_report = seal({"schema": "aura.semantic_proposer_crossfit.v1",
                          "plan_sha256": origin["plan_sha256"],
                          "row_receipts": {"held": "h"},
                          "candidate_receipt_sha256": "candidate"}, "receipt_sha256")
    body = {**{key: value for key, value in origin.items() if key != "plan_sha256"},
            "schema": "aura.semantic_proposer_source_calibration_plan.v1",
            "bank_partition": "source_calibration", "evaluated_ids": ["cal-a", "cal-b"],
            "reused_candidate_sha256": "candidate-file", "held_rows_evaluated": False,
            "proposer_fit_updates": 0, "serving_authority": False, "qualification_evidence": False}
    plan = seal(body, "plan_sha256")
    rb = {"schema": "aura.semantic_proposer_source_calibration.v1",
          "plan_sha256": plan["plan_sha256"], "candidate_receipt_sha256": "candidate",
          "source_calibration_population": 2, "row_receipts": {"cal-a": "a", "cal-b": "b"},
          "held_rows_evaluated": False, "proposer_fit_updates": 0,
          "serving_authority": False, "qualification_evidence": False}
    kwargs = {"origin_plan": origin, "origin_report": origin_report,
              "candidate_sha256": "candidate-file"}
    assert verify_source_calibration_bank(plan, seal(rb, "receipt_sha256"), **kwargs)[0] == plan
    for field, value in (("held_rows_evaluated", True), ("proposer_fit_updates", 1),
                         ("proposer_fit_updates", False),
                         ("evaluated_ids", ["held"]), ("fit_ids", ["cal-a"]),
                         ("reused_candidate_sha256", "changed")):
        changed = copy.deepcopy(body)
        changed[field] = value
        changed_plan = seal(changed, "plan_sha256")
        changed_report = seal({**rb, "plan_sha256": changed_plan["plan_sha256"]}, "receipt_sha256")
        with pytest.raises(ValueError, match="source calibration bank"):
            verify_source_calibration_bank(changed_plan, changed_report, **kwargs)
    for field, value in (("candidate_receipt_sha256", "changed"),
                         ("held_population", 2), ("row_receipts", {"held": "a"}),
                         ("source_calibration_population", 1), ("serving_authority", True)):
        with pytest.raises(ValueError, match="source calibration bank"):
            verify_source_calibration_bank(plan, seal({**rb, field: value}, "receipt_sha256"), **kwargs)


def test_existing_transfer_bank_reader_refuses_source_calibration_evidence(tmp_path):
    from tools.train_nested_semantic_ranker import _verified_pair

    plan_body = {"schema": "aura.semantic_proposer_source_calibration_plan.v1",
                 "held_ids": ["held"], "evaluated_ids": ["cal"]}
    plan = {**plan_body, "plan_sha256": _digest(plan_body)}
    report_body = {"schema": "aura.semantic_proposer_source_calibration.v1",
                   "plan_sha256": plan["plan_sha256"], "row_receipts": {"cal": "row"}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "report.json").write_text(json.dumps(
        {**report_body, "receipt_sha256": _digest(report_body)}))
    with pytest.raises(ValueError, match="signed plan"):
        _verified_pair(tmp_path)


def _examples():
    return [SimpleNamespace(
        ir=SimpleNamespace(source_text_sha256=f"source-{index:02d}-{contrast}",
                           n_inputs=2 + index % 2, instructions=("step",)),
        construction_id=f"construction-{index:02d}",
        contrast_id="", split="train")
        for index in range(12) for contrast in range(2)]


def test_crossfit_excludes_held_sources_from_fit_and_calibration():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 2)
    groups = [{item.construction_id for item in part}
              for part in (fit, calibration, held)]
    assert all(groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(held) == len(groups[2])
    assert all(folds["assignments"][item.ir.source_text_sha256] == 2 for item in held)


def test_utterance_crossfit_replays_wording_split_and_preserves_family_fit():
    examples = [SimpleNamespace(
        ir=SimpleNamespace(source_text_sha256=f"{family}-{index}-{variant}",
                           n_inputs=inputs, instructions=("step",)),
        construction_id=f"{family}:wording-{index}", contrast_id=f"{family}-same",
        split="train")
        for family, inputs in (("first", 2), ("second", 3))
        for index in range(3) for variant in range(2)]
    folds = json.loads(json.dumps(utterance_construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 1, all_held=True)
    groups = [{item.construction_id for item in part}
              for part in (fit, calibration, held)]
    assert all(groups) and not groups[0] & groups[1]
    assert not groups[0] & groups[2] and not groups[1] & groups[2]
    assert {item.construction_id.partition(":")[0] for item in fit} == {"first", "second"}
    assert {item.construction_id.partition(":")[0] for item in held} == {"first", "second"}
    with pytest.raises(ValueError, match="contrast-closed"):
        nested_crossfit_partition(examples, folds, 1, 0)


def test_all_held_reuses_partition_without_omitting_contrasts():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 2, all_held=True)
    wanted = {item.ir.source_text_sha256 for item in examples
              if folds["assignments"][item.ir.source_text_sha256] == 2}
    assert {item.ir.source_text_sha256 for item in held} == wanted
    assert not wanted & {item.ir.source_text_sha256 for item in fit + calibration}


def test_fixed_per_construction_selection_is_label_blind_and_disjoint():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    fit, calibration, held = crossfit_partition(
        examples, folds, 2, per_construction=2)
    assert len(held) == 2 * len({item.construction_id for item in held})
    assert not {item.ir.source_text_sha256 for item in held} & {
        item.ir.source_text_sha256 for item in fit + calibration}
    with pytest.raises(ValueError, match="partition differs"):
        crossfit_partition(examples, folds, 2, per_construction=0)
    with pytest.raises(ValueError, match="partition differs"):
        crossfit_partition(examples, folds, 2, all_held=True, per_construction=2)


def test_nested_selector_bank_excludes_outer_and_inner_constructions():
    examples = _examples()
    outer = json.loads(json.dumps(construction_folds(examples)))
    inner, fit, calibration, held, outer_excluded = nested_crossfit_partition(
        examples, outer, 1, 0)
    assert inner["validation_used"] is False and inner["test_used"] is False
    assert set(inner["assignments"]) == {
        item.ir.source_text_sha256 for item in examples
        if outer["assignments"][item.ir.source_text_sha256] != 1}
    assert set(outer_excluded) == {
        item.ir.source_text_sha256 for item in examples
        if outer["assignments"][item.ir.source_text_sha256] == 1}
    groups = [{item.construction_id for item in part}
              for part in (fit, calibration, held)]
    outer_groups = {item.construction_id for item in examples
                    if item.ir.source_text_sha256 in outer_excluded}
    assert all(groups)
    assert not outer_groups & set.union(*groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert all(inner["assignments"][item.ir.source_text_sha256] == 0 for item in held)


def test_diagnostic_subset_is_ordered_and_confined_to_held_fold():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    wanted = [item.ir.source_text_sha256 for item in examples
              if folds["assignments"][item.ir.source_text_sha256] == 2][:2]
    _fit, _calibration, held = crossfit_partition(
        examples, folds, 2, all_held=True, held_source_ids=tuple(reversed(wanted)))
    assert [item.ir.source_text_sha256 for item in held] == list(reversed(wanted))
    outside = next(item.ir.source_text_sha256 for item in examples
                   if folds["assignments"][item.ir.source_text_sha256] != 2)
    with pytest.raises(ValueError, match="outside"):
        crossfit_partition(examples, folds, 2, all_held=True,
                           held_source_ids=(outside,))


def test_crossfit_rejects_relabelled_frozen_fold():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    source = examples[0].ir.source_text_sha256
    folds["assignments"][source] = (folds["assignments"][source] + 1) % folds["count"]
    body = {key: value for key, value in folds.items() if key != "receipt_sha256"}
    folds["receipt_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="construction group crosses"):
        crossfit_partition(examples, folds, 2)


def test_proposal_profile_counts_unique_programs_after_target_blind_generation():
    def row(candidates, statuses):
        return {"bank": {"candidates": [
            {"program_sha256": identity, "joint_score": score}
            for identity, score in candidates]},
            "diagnosis": {"comparisons": [
                {"program_sha256": identity, "status": status}
                for identity, status in statuses.items()]}}

    rows = [row([("wrong", 4.), ("wrong", 3.), ("right", 2.)],
                {"wrong": "different", "right": "equivalent"}),
            row([("miss", 1.)], {"miss": "different"}),
            row([("right", 5.)], {"right": "equivalent"})]
    assert proposal_reach_profile(rows) == {
        "population": 3, "distinct_generated_proposals": 4,
        "distinct_scored_proposals": 4, "observed_reachable": 2,
        "unscored_only_reachable": 0,
        "generation_recall_at_1": 1, "generation_recall_at_2": 2,
        "generation_recall_at_4": 2, "scored_recall_at_1": 1,
        "scored_recall_at_2": 2, "scored_recall_at_4": 2,
        "mean_first_correct_generation_index": 1.5,
        "mean_first_correct_scored_rank": 1.5}
    only_incumbent = row([("right", None), ("wrong", 1.)],
                         {"right": "equivalent", "wrong": "different"})
    assert proposal_reach_profile([only_incumbent])["unscored_only_reachable"] == 1
    with pytest.raises(ValueError, match="independent comparison"):
        proposal_reach_profile([row([("unknown", 1.)], {})])


def test_read_only_profile_requires_signed_rows_and_plan(tmp_path):
    from tools.probe_semantic_proposer_crossfit import _digest

    source = "a" * 64
    plan_body = {"schema": "aura.semantic_proposer_crossfit_plan.v1", "fold": 0,
                 "held_ids": [source]}
    plan = {**plan_body, "plan_sha256": _digest(plan_body)}
    row_body = {"source": source, "plan_sha256": plan["plan_sha256"],
                "bank": {"candidates": [
                    {"program_sha256": "right", "joint_score": 1.}]},
                "diagnosis": {"comparisons": [
                    {"program_sha256": "right", "status": "equivalent"}]}}
    row = {**row_body, "receipt_sha256": _digest(row_body)}
    report_body = {"schema": "aura.semantic_proposer_crossfit.v1",
                   "plan_sha256": plan["plan_sha256"], "held_population": 1,
                   "correct_reachable": 1,
                   "row_receipts": {source: row["receipt_sha256"]}}
    report = {**report_body, "receipt_sha256": _digest(report_body)}
    (tmp_path / "rows").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "report.json").write_text(json.dumps(report))
    path = tmp_path / "rows" / f"{source}.json"
    path.write_text(json.dumps(row))
    assert audit_directory(tmp_path)["profile"]["scored_recall_at_1"] == 1
    path.write_text(json.dumps({**row, "source": "b" * 64}))
    with pytest.raises(ValueError, match="signed report"):
        audit_directory(tmp_path)
