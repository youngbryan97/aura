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
    crossfit_partition,
    nested_crossfit_partition,
    proposal_reach_profile,
)


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
