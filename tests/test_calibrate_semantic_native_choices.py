"""Combined native choices use source evidence and independent admission."""

import copy
import hashlib
import json

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.calibrate_semantic_native_choices import (
    calibrate,
    candidate_relation_keys,
    combined_views,
    native_method_basis,
    partition_source_rows,
    select_combined,
    source_invariance_audit,
    source_observations,
    with_schema_support,
)
from tools.evaluate_semantic_native_checkpoint import digest


def fixture():
    programs = [Program(2, (Instruction(op, (0, 1)),)) for op in ("add", "sub")]
    keys = [program.sha() for program in programs]
    bank = {"bank": {"input_spans": [{}, {}], "selected_program_sha256": keys[0],
        "candidates": [{"program_sha256": key, "joint_score": score,
                        "program": {"instructions": [[program.instructions[0].op, [0, 1]]],
                                    "sha": key}}
                       for key, program, score in zip(keys, programs, (2., 1.), strict=True)]}}
    rows = [{"program_sha256s": keys, "scores": [-3., -1.], "pretrained_scores": [-1., -2.],
             "chosen_program_sha256": keys[1], "pretrained_program_sha256": keys[0]}]
    return bank, rows, keys


def test_combined_features_ignore_target_labels_and_construction_names():
    bank, rows, keys = fixture()
    before = combined_views(bank, rows)
    bank["diagnosis"] = {"comparisons": [{"status": "equivalent"}]}
    bank["construction"] = "unfamiliar-domain"
    rows[0].update(selected_correct=False, incumbent_correct=True, construction="different")
    assert combined_views(bank, rows) == before
    incumbent, choices, views = before
    assert incumbent == keys[0] and choices == tuple(keys)
    assert set(views[keys[0]]) == set(views[keys[1]])
    assert not any("correct" in name or "construction" in name for name in views[keys[0]])
    assert views[keys[0]]["relative_method_0_scores"] == 0.
    assert views[keys[1]]["relative_method_0_scores"] == 2.


def test_multiple_native_methods_keep_each_distinct_proposal():
    bank, rows, keys = fixture()
    alternate = {**rows[0], "scores": [-1., -3.], "chosen_program_sha256": keys[0]}
    incumbent, choices, views = combined_views(bank, [*rows, alternate])
    assert incumbent == keys[0] and set(choices) == set(keys)
    assert "method_1_scores_gap" in views[keys[0]]
    assert combined_views(bank, [*rows, alternate, alternate])[1] == choices


def test_joint_and_unfitted_successes_are_available_without_an_outcome_oracle():
    bank, rows, keys = fixture()
    program = Program(2, (Instruction("mul", (0, 1)),))
    key = program.sha()
    bank["bank"]["candidates"].append({"program_sha256": key, "joint_score": 4.,
        "program": {"instructions": [["mul", [0, 1]]], "sha": key}})
    rows[0]["program_sha256s"] = [*keys, key]
    rows[0]["scores"] = [-3., -1., -4.]
    rows[0]["pretrained_scores"] = [-1., -2., -3.]
    incumbent, choices, views = combined_views(bank, rows)
    assert incumbent == keys[0]
    assert set(choices) == {*keys, key}
    assert set(views) == set(choices)


def test_nonwinning_measured_program_remains_a_choice():
    bank, rows, keys = fixture()
    program = Program(2, (Instruction("mul", (0, 1)),))
    key = program.sha()
    bank["bank"]["candidates"].append({"program_sha256": key, "joint_score": -5.,
        "program": {"instructions": [["mul", [0, 1]]], "sha": key}})
    rows[0]["program_sha256s"] = [*keys, key]
    rows[0]["scores"] = [-3., -1., -4.]
    rows[0]["pretrained_scores"] = [-1., -2., -3.]
    assert combined_views(bank, rows)[1] == (*keys, key)


def test_source_relationships_are_candidate_evidence_not_diagnosis():
    bank, rows, keys = fixture()
    candidate = bank["bank"]["candidates"][1]
    candidate.update(operation_spans=[{"start": 2, "end": 5}],
                     argument_spans=[[{"start": 8, "end": 10},
                                      {"start": 12, "end": 14}]],
                     definition_spans=[[{"start": 8, "end": 10},
                                       {"start": 2, "end": 5}]],
                     definition_provenance="optimizer_selected")
    first = combined_views(bank, rows)[2][keys[1]]
    assert first["source_span_available"] == 1.
    assert first["source_relation_available"] == 1.
    assert first["source_relation_overlap"] == .5
    assert first["selected_definition_paths"] == 1.
    bank["diagnosis"] = {"comparisons": [{"program_sha256": keys[1],
                                             "status": "equivalent"}]}
    assert combined_views(bank, rows)[2][keys[1]] == first
    candidate["definition_spans"][0][1] = {"start": 12, "end": 14}
    assert combined_views(bank, rows)[2][keys[1]]["source_relation_overlap"] == 1.
    candidate["operation_spans"] = [{"start": -1, "end": 5}]
    with pytest.raises(ValueError):
        combined_views(bank, rows)


@pytest.mark.parametrize("defect", ["inventory", "incumbent", "nonfinite", "program"])
def test_combined_choice_refuses_substituted_or_unusable_evidence(defect):
    bank, rows, keys = fixture()
    if defect == "inventory":
        rows.append({**rows[0], "program_sha256s": keys[::-1]})
    elif defect == "incumbent":
        bank["bank"]["selected_program_sha256"] = keys[1]
    elif defect == "nonfinite":
        rows[0]["scores"][0] = float("nan")
    else:
        bank["bank"]["candidates"][0]["program"]["instructions"][0][1] = [0, 9]
    with pytest.raises(ValueError):
        combined_views(bank, rows)


def test_unavailable_incumbent_does_not_become_a_calibration_success():
    bank, rows, _keys = fixture()
    bank["bank"]["selected_program_sha256"] = None
    assert combined_views(bank, rows) is None
    rows[0]["scored"] = False
    assert combined_views(bank, rows) is None


def population():
    bank, native, keys = fixture()
    views = combined_views(bank, native)
    return [{"source": f"{group}-{index}", "construction": f"group-{group}",
             "views": views, "labels": {keys[0]: False, keys[1]: True}}
            for group in range(6) for index in range(24)]


def test_source_partitions_preserve_groups_and_exclude_checkpoint_selection_instances():
    rows = population()
    excluded = {"0-0", "1-0"}
    splits = partition_source_rows(rows, excluded_ids=excluded)
    assert splits == partition_source_rows(rows[::-1], excluded_ids=excluded)
    groups = [{row["construction"] for row in split} for split in splits]
    assert all(len(group) == 2 for group in groups)
    assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    assert {row["source"] for split in splits for row in split} == {
        row["source"] for row in rows} - excluded
    for bad in (rows + rows[:1], [{**row, "construction": "one"} for row in rows]):
        with pytest.raises(ValueError):
            partition_source_rows(bad, excluded_ids=set())
    with pytest.raises(ValueError):
        partition_source_rows(rows, excluded_ids={"absent"})


def test_calibration_reports_groups_lost_to_checkpoint_selection():
    rows = population()
    extra = copy.deepcopy([row for row in rows if row["construction"] == "group-5"])
    for row in extra:
        row["source"] = row["source"].replace("5-", "6-", 1)
        row["construction"] = "group-6"
    rows.extend(extra)
    excluded = {row["source"] for row in rows if row["construction"] == "group-6"}
    result = calibrate(rows, excluded_ids=excluded)
    coverage = result["construction_coverage"]
    assert coverage["source_groups"]["group-6"] == 24
    assert "group-6" not in coverage["eligible_groups"]
    assert coverage["excluded_groups"] == ["group-6"]
    assert {group for split in coverage["split_groups"] for group in split} == {
        f"group-{index}" for index in range(6)}


def test_relation_overlap_requires_verified_cross_group_support():
    rows = population()
    for row in rows:
        row["verified_relation_keys"] = ("shared" if row["construction"] in {
            "group-0", "group-1"} else row["construction"],)
    coverage = calibrate(rows, excluded_ids=set())["relation_coverage"]
    assert coverage == {"eligible_verified_relations": 5,
                        "cross_group_verified_relations": 1,
                        "eligible_groups_with_cross_group_relation": ["group-0", "group-1"]}
    assert calibrate(population(), excluded_ids=set())["relation_coverage"] is None


def test_invariance_audit_detects_source_family_shortcut_without_training_on_labels():
    rows = population()
    for row in rows:
        incumbent, choices, views = row["views"]
        row["views"] = (incumbent, choices, {
            key: {**values, "family_shortcut": float(row["construction"].split("-")[1])}
            for key, values in views.items()})
        row["candidate_relation_keys"] = {key: "common" for key in choices}
    audit = source_invariance_audit(rows, excluded_ids=set(), permutations=20)
    assert audit["nuisance_probe"]["balanced_accuracy"] > audit["nuisance_probe"]["permutation_mean"]
    assert audit["relation_geometry"]["same_relation_cross_group_pairs"] > 0
    assert calibrate(rows, excluded_ids=set())["invariance_audit"]["status"] == "measured"


def test_source_fit_schema_memory_excludes_its_own_group(monkeypatch):
    from core.evidence import calibrated_binary

    rows = population()
    for row in rows:
        row["verified_relation_keys"] = ("shared",)
        row["candidate_relation_keys"] = {key: "shared" for key in row["views"][1]}
    observed = {}
    def capture(fit, tune):
        observed["fit"] = fit
        observed["tune"] = tune
        return None, {"admitted": False}
    monkeypatch.setattr(calibrated_binary, "fit_calibrated_binary_scorer", capture)
    result = calibrate(rows, excluded_ids=set())
    assert result["schema_memory_counts"] == {"shared": 2}
    assert {dict(row.values)["schema_support_groups"] for row in observed["fit"]} == {1.}
    assert {dict(row.values)["schema_support_groups"] for row in observed["tune"]} == {2.}


def test_source_schema_support_is_typed_and_target_blind():
    bank, native, keys = fixture()
    relations = candidate_relation_keys(bank)
    before = with_schema_support(combined_views(bank, native), relations,
                                 {relations[keys[1]]: 3})
    assert before[2][keys[0]]["schema_support_groups"] == 0.
    assert before[2][keys[1]]["schema_support_groups"] == 3.
    bank["diagnosis"] = {"comparisons": [{"status": "different"}]}
    assert with_schema_support(combined_views(bank, native), relations,
                               {relations[keys[1]]: 3}) == before
    with pytest.raises(ValueError):
        with_schema_support(combined_views(bank, native), relations, {relations[keys[1]]: -1})


def test_existing_selector_can_admit_a_measured_combined_gain():
    from core.evidence.calibrated_candidate_selector import calibrated_candidate_selector_from_dict

    result = calibrate(population(), excluded_ids=set())
    assert result["selector"] is not None
    assert result["schema_memory_counts"] == {}
    assert all(all(group["gains"] == group["observations"]
                   for group in split.values()) for split in result["raw_rank_groups"])
    assert result["selector_report"]["admission_improvements"] == 48
    assert result["selector_report"]["admission_regressions"] == 0
    selector = calibrated_candidate_selector_from_dict(result["selector"])
    bank, native, keys = fixture()
    assert select_combined(selector, bank, native, source_ref="new-request",
                           schema_memory_counts=result["schema_memory_counts"]) == keys[1]
    bank["diagnosis"] = {"comparisons": "invalid and unavailable labels"}
    assert select_combined(selector, bank, native, source_ref="new-request",
                           schema_memory_counts=result["schema_memory_counts"]) == keys[1]
    native[0]["scored"] = False
    assert select_combined(selector, bank, native, source_ref="unscored-request",
                           schema_memory_counts=result["schema_memory_counts"]) == keys[0]


def test_unknown_evidence_cannot_supply_negative_training_labels():
    rows = copy.deepcopy(population())
    for row in rows:
        row["labels"] = {key: None for key in row["labels"]}
    result = calibrate(rows, excluded_ids=set())
    assert result["selector"] is None
    assert result["scorer_report"]["reason"] == "insufficient_fit_observations"


def test_best_incumbent_is_retained_without_a_self_comparison():
    from types import SimpleNamespace

    bank, native, keys = fixture()
    incumbent_values = combined_views(bank, native)[2][keys[0]]
    class Scorer:
        def predict(self, values):
            return float(values == incumbent_values)
    class Selector:
        scorer = Scorer()
        def select(self, **kwargs):
            raise AssertionError("an incumbent is not a distinct challenger")
    bank["diagnosis"] = SimpleNamespace(comparisons="unavailable")
    assert select_combined(Selector(), bank, native, source_ref="same-choice") == keys[0]


def durable_fixture(root):
    from core.learning.semantic_program_campaign import _sha

    origin, bank_dir, native_dir = (root / name for name in ("origin", "bank", "native"))
    for directory in (origin, bank_dir, native_dir):
        (directory / "rows").mkdir(parents=True)
    def save(path, body, field="receipt_sha256", hasher=digest):
        value = {**body, field: hasher(body)}
        path.write_text(json.dumps(value))
        return value
    candidate = b"frozen source proposer"
    (origin / "candidate.json").write_bytes(candidate)
    base = {"schema": "aura.semantic_proposer_crossfit_plan.v1", "fit_ids": ["fit"],
        "calibration_ids": ["cal-a", "cal-b"], "held_ids": ["held"],
        "source_report_sha256": "source", "parent_receipt_sha256": "parent",
        "folds_sha256": "folds", "fold": 0, "input_order_policy": "source",
        "heldout_axis": "wording"}
    origin_plan = save(origin / "plan.json", base, "plan_sha256")
    save(origin / "report.json", {"schema": "aura.semantic_proposer_crossfit.v1",
        "plan_sha256": origin_plan["plan_sha256"], "row_receipts": {"held": "h"},
        "candidate_receipt_sha256": "candidate"})
    plan = save(bank_dir / "plan.json", {**base,
        "schema": "aura.semantic_proposer_source_calibration_plan.v1",
        "bank_partition": "source_calibration", "evaluated_ids": ["cal-a", "cal-b"],
        "reused_candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "held_rows_evaluated": False, "proposer_fit_updates": 0,
        "serving_authority": False, "qualification_evidence": False}, "plan_sha256")
    receipts = {}
    for source in base["calibration_ids"]:
        bank, _native, keys = fixture()
        b = save(bank_dir / "temporary.json", {**bank["bank"],
            "source_text_sha256": source, "transducer_receipt_sha256": "candidate"}, hasher=_sha)
        diagnosis = save(bank_dir / "temporary.json", {"bank_receipt_sha256": b["receipt_sha256"],
            "split": "train", "comparisons": [
                {"program_sha256": keys[0], "status": "different"},
                {"program_sha256": keys[1], "status": "equivalent"}]}, hasher=_sha)
        row = save(bank_dir / "rows" / f"{source}.json", {"source": source,
            "plan_sha256": plan["plan_sha256"], "bank": b, "diagnosis": diagnosis, "mixed": None})
        receipts[source] = row["receipt_sha256"]
    report = save(bank_dir / "report.json", {
        "schema": "aura.semantic_proposer_source_calibration.v1",
        "plan_sha256": plan["plan_sha256"], "row_receipts": receipts,
        "candidate_receipt_sha256": "candidate", "source_calibration_population": 2,
        "held_rows_evaluated": False, "proposer_fit_updates": 0,
        "serving_authority": False, "qualification_evidence": False})
    native_plan = save(native_dir / "plan.json", {
        "schema": "aura.semantic_native_source_calibration_plan.v1",
        "evaluated_ids": base["calibration_ids"], "held_rows_evaluated": False,
        "source_calibration_plan_sha256": plan["plan_sha256"],
        "source_calibration_receipt_sha256": report["receipt_sha256"],
        "checkpoint_calibration_ids": ["cal-a"], "fit_updates": 0,
        "model_descriptor_sha256": "model", "pointer_sha256": "pointer",
        "training_plan_sha256": "training", "checkpoint_receipt_sha256": "checkpoint",
        "weights_sha256": "weights", "selected_step": 2,
        "semantic_decision_basis": "program_atoms_v1", "loss_scope": "semantic_decisions",
        "serving_authority": False, "qualification_evidence": False}, "plan_sha256")
    rows = []
    for source in base["calibration_ids"]:
        _bank, native, _keys = fixture()
        rows.append(save(native_dir / "rows" / f"{source}.json", {**native[0],
            "source": source, "construction": "construction",
            "plan_sha256": native_plan["plan_sha256"], "incumbent_correct": False,
            "incumbent_available": True, "selected_correct": True,
            "pretrained_correct": False, "bank_reachable": True,
            "labels_available_to_scorer": False}))
    save(native_dir / "report.json", {"schema": "aura.semantic_native_source_calibration.v1",
        "plan_sha256": native_plan["plan_sha256"], "rows": rows, "population": 2,
        "held_rows_evaluated": False, "fit_updates": 0,
        "serving_authority": False, "qualification_evidence": False})
    return origin, bank_dir, native_dir


def test_durable_source_observations_bind_native_scores_to_original_source_bank(tmp_path):
    origin, bank, native = durable_fixture(tmp_path)
    rows, excluded, _bank_report, _reports = source_observations(
        bank, [native], origin_directory=origin)
    assert len(rows) == 2 and excluded == {"cal-a"}
    assert all(row["views"] is not None for row in rows)
    with pytest.raises(ValueError, match="duplicated"):
        source_observations(bank, [native, native], origin_directory=origin)
    path = native / "rows" / "cal-b.json"
    body = json.loads(path.read_bytes())
    body["scores"] = body["scores"][::-1]
    body.pop("receipt_sha256")
    path.write_text(json.dumps({**body, "receipt_sha256": digest(body)}))
    with pytest.raises(ValueError, match="durable score row"):
        source_observations(bank, [native], origin_directory=origin)


def test_frozen_combined_replay_is_callable_and_refuses_calibration_leakage(tmp_path):
    from core.learning.semantic_program_campaign import _sha
    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.replay_semantic_native_choices import replay

    origin, bank, native = durable_fixture(tmp_path)
    def save(path, body, field="receipt_sha256", hasher=digest):
        value = {**body, field: hasher(body)}
        path.write_text(json.dumps(value))
        return value
    origin_plan = verified_document(origin / "plan.json", "plan_sha256")
    row = verified_document(bank / "rows" / "cal-a.json")
    b = {key: value for key, value in row["bank"].items() if key != "receipt_sha256"}
    b["source_text_sha256"] = "held"
    b = {**b, "receipt_sha256": _sha(b)}
    diagnosis = {key: value for key, value in row["diagnosis"].items() if key != "receipt_sha256"}
    diagnosis["bank_receipt_sha256"] = b["receipt_sha256"]
    diagnosis = {**diagnosis, "receipt_sha256": _sha(diagnosis)}
    held = save(origin / "rows" / "held.json", {"source": "held", "bank": b,
        "diagnosis": diagnosis, "mixed": None, "plan_sha256": origin_plan["plan_sha256"]})
    origin_report = save(origin / "report.json", {"schema": "aura.semantic_proposer_crossfit.v1",
        "plan_sha256": origin_plan["plan_sha256"], "row_receipts": {"held": held["receipt_sha256"]},
        "candidate_receipt_sha256": "candidate"})
    native_plan = verified_document(native / "plan.json", "plan_sha256")
    basis = native_method_basis(native_plan)
    held_dir = tmp_path / "held-native"
    (held_dir / "rows").mkdir(parents=True)
    held_plan = save(held_dir / "plan.json", {**basis,
        "schema": "aura.semantic_native_replay_plan.v1", "held_ids": ["held"],
        "bank_receipt_sha256": origin_report["receipt_sha256"], "fit_updates": 0,
        "held_labels_used_for_fit_or_selection": False,
        "serving_authority": False, "qualification_evidence": False}, "plan_sha256")
    original_row = verified_document(native / "rows" / "cal-a.json")
    held_row = save(held_dir / "rows" / "held.json", {
        **{key: value for key, value in original_row.items() if key != "receipt_sha256"},
        "source": "held", "plan_sha256": held_plan["plan_sha256"]})
    save(held_dir / "report.json", {"schema": "aura.semantic_native_replay.v1",
        "plan_sha256": held_plan["plan_sha256"], "population": 1, "rows": [held_row],
        "fit_updates": 0, "serving_authority": False, "qualification_evidence": False})
    policy = {"schema": "aura.semantic_native_combined_calibration.v1",
        **calibrate(population(), excluded_ids=set()),
        "origin_bank_receipt_sha256": origin_report["receipt_sha256"],
        "native_method_bases": [basis], "held_rows_evaluated": False,
        "serving_authority": False, "qualification_evidence": False}
    result = replay(policy, origin, [held_dir])
    assert (result["population"], result["selected_correct"], result["gains"]) == (1, 1, 1)
    with pytest.raises(ValueError, match="calibration instances"):
        replay({**policy, "excluded_checkpoint_calibration_ids": ["held"]}, origin, [held_dir])
    with pytest.raises(ValueError, match="scoring basis"):
        replay({**policy, "native_method_bases": [{**basis, "weights_sha256": "changed"}]},
               origin, [held_dir])
    with pytest.raises(ValueError, match="admission"):
        replay({**policy, "selector": None}, origin, [held_dir])
