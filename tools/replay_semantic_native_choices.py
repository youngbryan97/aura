#!/usr/bin/env python3
"""Replay a frozen source-admitted combined selector without fitting on held rows."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def replay(policy, bank_directory, native_directories):
    from core.evidence.calibrated_candidate_selector import calibrated_candidate_selector_from_dict
    from tools.calibrate_semantic_native_choices import (
        independently_graded,
        native_method_basis,
        select_combined,
    )
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.evaluate_semantic_native_checkpoint import verified_document, verify_replay_row
    from tools.train_nested_semantic_ranker import _verified_pair

    if (policy.get("schema") != "aura.semantic_native_combined_calibration.v1"
            or policy.get("held_rows_evaluated") is not False
            or policy.get("serving_authority") is not False
            or policy.get("qualification_evidence") is not False
            or policy.get("selector") is None
            or policy["selector_report"].get("admitted") is not True):
        raise ValueError("combined native selector lacks source-only admission")
    selector = calibrated_candidate_selector_from_dict(policy["selector"])
    plan, report = _verified_pair(bank_directory)
    if report["receipt_sha256"] != policy["origin_bank_receipt_sha256"]:
        raise ValueError("combined native replay changed its original proposer bank")
    source_ids = set(policy["excluded_checkpoint_calibration_ids"]).union(*(
        set(population) for population in policy["split_ids"]))
    if source_ids & set(plan["held_ids"]):
        raise ValueError("combined native replay contains calibration instances")
    if len(native_directories) != len(policy["native_method_bases"]):
        raise ValueError("combined native replay method count differs")
    methods = []
    for directory, basis in zip(native_directories, policy["native_method_bases"], strict=True):
        native_plan = verified_document(directory / "plan.json", "plan_sha256")
        native_report = verified_document(directory / "report.json")
        if (native_plan.get("schema") != "aura.semantic_native_replay_plan.v1"
                or native_report.get("schema") != "aura.semantic_native_replay.v1"
                or native_report["plan_sha256"] != native_plan["plan_sha256"]
                or native_method_basis(native_plan) != basis
                or native_plan["bank_receipt_sha256"] != report["receipt_sha256"]
                or native_plan["held_ids"] != plan["held_ids"]
                or type(native_plan.get("fit_updates")) is not int or native_plan["fit_updates"] != 0
                or type(native_report.get("fit_updates")) is not int or native_report["fit_updates"] != 0
                or native_plan.get("held_labels_used_for_fit_or_selection") is not False
                or any(document.get("serving_authority") is not False
                       or document.get("qualification_evidence") is not False
                       for document in (native_plan, native_report))):
            raise ValueError("combined native replay scoring basis differs")
        by_source = {row["source"]: row for row in native_report["rows"]}
        if (set(by_source) != set(plan["held_ids"])
                or len(by_source) != len(native_report["rows"])
                or native_report["population"] != len(by_source)):
            raise ValueError("combined native replay held population differs")
        methods.append((directory, native_plan, native_report, by_source))
    outcomes = []
    for source in plan["held_ids"]:
        bank = _read_bank(bank_directory / "rows" / f"{source}.json", source=source,
            plan_sha=plan["plan_sha256"], model_receipt=report["candidate_receipt_sha256"],
            expected_receipt=report["row_receipts"][source])
        labels = independently_graded(bank)
        rows = []
        for directory, native_plan, _native_report, by_source in methods:
            row = by_source[source]
            if verified_document(directory / "rows" / f"{source}.json") != row:
                raise ValueError("combined native replay durable score row differs")
            verify_replay_row(row, source=source, plan_sha256=native_plan["plan_sha256"],
                programs=row["program_sha256s"],
                labels=tuple(labels[key] for key in row["program_sha256s"]),
                incumbent_available=bank["bank"]["selected_program_sha256"] is not None)
            rows.append(row)
        selected = select_combined(selector, bank, rows, source_ref=source)
        incumbent = bank["bank"]["selected_program_sha256"]
        outcomes.append({"source": source, "selected_program_sha256": selected,
            "incumbent_program_sha256": incumbent,
            "selected_correct": labels[selected] if selected is not None else False,
            "incumbent_correct": labels[incumbent] if incumbent is not None else False})
    return {"population": len(outcomes), "rows": outcomes,
        "incumbent_correct": sum(row["incumbent_correct"] is True for row in outcomes),
        "selected_correct": sum(row["selected_correct"] is True for row in outcomes),
        "unresolved_selected": sum(row["selected_correct"] is None for row in outcomes),
        "gains": sum(row["selected_correct"] is True and row["incumbent_correct"] is False
                     for row in outcomes),
        "regressions": sum(row["incumbent_correct"] is True and row["selected_correct"] is False
                           for row in outcomes),
        "native_replay_receipt_sha256s": [method[2]["receipt_sha256"] for method in methods]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    policy = verified_document(args.policy)
    paths = [Path(__file__), ROOT / "tools/calibrate_semantic_native_choices.py",
             ROOT / "core/evidence/calibrated_candidate_selector.py",
             ROOT / "core/evidence/calibrated_binary.py",
             ROOT / "core/evidence/necessary_condition_selector.py"]
    implementation = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths}
    result = replay(policy, args.bank, args.native_replay)
    if (verified_document(args.policy) != policy or result != replay(policy, args.bank, args.native_replay)
            or any(hashlib.sha256(path.read_bytes()).hexdigest()
                   != implementation[str(path.relative_to(ROOT))] for path in paths)):
        raise ValueError("combined native replay inputs or implementation drifted")
    body = {"schema": "aura.semantic_native_combined_replay.v1", **result,
        "policy_receipt_sha256": policy["receipt_sha256"], "implementation": implementation,
        "fit_updates": 0, "held_labels_used_for_fit_or_selection": False,
        "serving_authority": False, "qualification_evidence": False}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
