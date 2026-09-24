#!/usr/bin/env python3
"""Measure source-program proposals with whole constructions absent from fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def crossfit_partition(examples: list, folds: dict, fold: int) -> tuple[list, list, list]:
    """Keep a group's examples together in fit, calibration, or holdout."""
    from core.learning.semantic_construction_folds import construction_folds
    from core.learning.semantic_program_campaign import _sha

    body = {key: value for key, value in folds.items() if key != "receipt_sha256"}
    rows = sorted((item.ir.source_text_sha256, item.construction_id, item.contrast_id)
                  for item in examples if item.split == "train")
    if (folds.get("schema") != "aura.semantic_construction_folds.v1"
            or folds.get("receipt_sha256") != _sha(body)
            or folds.get("population") != [list(row) for row in rows]
            or type(fold) is not int or not 0 <= fold < folds["count"]
            or set(folds["assignments"]) != {row[0] for row in rows}):
        raise ValueError("crossfit source partition differs from frozen folds")
    groups = {}
    for item in examples:
        if item.split == "train":
            groups.setdefault(item.construction_id, []).append(item)
    if any(len({folds["assignments"][item.ir.source_text_sha256] for item in items}) != 1
           for items in groups.values()):
        raise ValueError("construction group crosses a frozen fold")
    available = [item for item in examples if item.split == "train"
                 and folds["assignments"][item.ir.source_text_sha256] != fold]
    calibration_folds = construction_folds(available, count=5)
    possible = [index for index in range(calibration_folds["count"])
                if len({(item.ir.n_inputs, len(item.ir.instructions)) for item in available
                        if calibration_folds["assignments"][item.ir.source_text_sha256]
                        != index}) >= 2]
    if not possible:
        raise ValueError("no source-only calibration fold preserves fit geometries")
    calibration_index = min(possible, key=lambda index: (
        sum(calibration_folds["assignments"][item.ir.source_text_sha256] == index
            for item in available), index))
    fit = [item for item in available if calibration_folds["assignments"][
        item.ir.source_text_sha256] != calibration_index]
    calibration = [item for item in available if calibration_folds["assignments"][
        item.ir.source_text_sha256] == calibration_index]
    held = [min(items, key=lambda item: item.ir.source_text_sha256)
            for construction, items in sorted(groups.items())
            if folds["assignments"][items[0].ir.source_text_sha256] == fold]
    if (not fit or not calibration or not held
            or set(item.construction_id for item in fit)
            & set(item.construction_id for item in calibration + held)
            or set(item.construction_id for item in calibration)
            & set(item.construction_id for item in held)):
        raise ValueError("crossfit has no independent fit, calibration, and holdout groups")
    return fit, calibration, held


def _save_if_absent(path: Path, body: dict) -> None:
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    payload = json.dumps(body, sort_keys=True, allow_nan=False).encode()
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError(f"crossfit artifact changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-charts", type=int, default=4)
    parser.add_argument("--max-graphs", type=int, default=2)
    parser.add_argument("--solve-seconds", type=float, default=1.)
    args = parser.parse_args()
    if (args.max_charts < 1 or args.max_graphs < 1 or
            not 0 < args.solve_seconds <= 60):
        parser.error("bounded candidate search needs positive limits")

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.directory / "report.json")
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from core.learning.semantic_validation_checkpoint import validation_implementation_identity
    from tools.diagnose_semantic_gap_candidate_bank import _audit_row

    parent_raw, source_raw, folds_raw = (path.read_bytes() for path in
                                         (args.parent, args.source_report, args.folds))
    parent = compositional_semantic_program_transducer_from_dict(json.loads(parent_raw))
    source_report, folds = json.loads(source_raw), json.loads(folds_raw)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(parent, source_report, bundles)
    fit, calibration, held = crossfit_partition(examples, folds, args.fold)
    all_sources = sorted(item.ir.source_text_sha256 for item in examples if item.split == "train")
    if parent.training_receipt.get("training_example_ids_sha256") != _sha(all_sources):
        raise ValueError("parent training cohort cannot be established")
    plan_body = {"schema": "aura.semantic_proposer_crossfit_plan.v1",
                 "parent_receipt_sha256": parent.receipt_sha256,
                 "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
                 "folds_sha256": hashlib.sha256(folds_raw).hexdigest(),
                 "fold": args.fold,
                 "fit_ids": sorted(item.ir.source_text_sha256 for item in fit),
                 "calibration_ids": sorted(item.ir.source_text_sha256 for item in calibration),
                 "held_ids": sorted(item.ir.source_text_sha256 for item in held),
                 "max_charts": args.max_charts, "max_graphs_per_chart": args.max_graphs,
                 "solve_seconds": args.solve_seconds,
                 "implementation": validation_implementation_identity(),
                 "serving_authority": False, "qualification_evidence": False,
                 "original_validation_used_for_fit": False}
    plan = {**plan_body, "plan_sha256": _digest(plan_body)}
    args.directory.mkdir(parents=True, exist_ok=True)
    _save_if_absent(args.directory / "plan.json", plan)
    candidate_path = args.directory / "candidate.json"
    if candidate_path.exists():
        candidate = compositional_semantic_program_transducer_from_dict(
            json.loads(candidate_path.read_bytes()))
    else:
        candidate = fit_compositional_semantic_program_transducer(
            [*(replace(item, split="train") for item in fit),
             *(replace(item, split="validation") for item in calibration)],
            input_grounding=parent.input_grounding)
        candidate = (candidate.with_global_constraint_arguments()
                     .with_conditional_argument_scores()
                     .with_overlap_complete_mentions()
                     .with_atomic_literal_arguments()
                     .with_feasible_operation_charts()
                     .with_order_invariant_argument_graph()
                     .with_joint_definition_graph()
                     .with_categorical_relation_scores())
        _save_if_absent(candidate_path, candidate.to_dict())
    if (candidate.training_receipt["training_example_ids_sha256"] != _sha(plan["fit_ids"])
            or candidate.training_receipt["validation_example_ids_sha256"] != _sha(
                plan["calibration_ids"])):
        raise ValueError("crossfit model saw a held construction")
    (args.directory / "rows").mkdir(exist_ok=True)
    rows = []
    for item in held:
        source = item.ir.source_text_sha256
        path = args.directory / "rows" / f"{source}.json"
        if not path.exists():
            row_body = _audit_row(item, candidate, plan)
            _save_if_absent(path, {**row_body, "receipt_sha256": _digest(row_body)})
        row = json.loads(path.read_bytes())
        if (row["source"] != source or row["plan_sha256"] != plan["plan_sha256"]
                or row["receipt_sha256"] != _digest({
                    key: value for key, value in row.items() if key != "receipt_sha256"})):
            raise ValueError("crossfit held-row receipt differs")
        rows.append(row)
        print(json.dumps({"stage": "held", "done": len(rows), "total": len(held),
                          "source": source,
                          "reachable": row["diagnosis"]["correct_reachable"]}), flush=True)
    top_correct = 0
    for row in rows:
        statuses = {result["program_sha256"]: result["status"] for result in
                    row["diagnosis"]["comparisons"]}
        scored = [candidate for candidate in row["bank"]["candidates"]
                  if candidate["joint_score"] is not None]
        if scored:
            top = max(scored, key=lambda candidate: candidate["joint_score"])
            top_correct += statuses.get(top["program_sha256"]) == "equivalent"
    body = {"schema": "aura.semantic_proposer_crossfit.v1", "plan_sha256": plan["plan_sha256"],
            "candidate_receipt_sha256": candidate.receipt_sha256,
            "held_population": len(rows),
            "correct_reachable": sum(row["diagnosis"]["correct_reachable"] is True for row in rows),
            "ordinary_correct": sum(row["diagnosis"]["selected_semantic_status"] == "equivalent"
                                    for row in rows),
            "top_joint_score_correct": top_correct,
            "row_receipts": {row["source"]: row["receipt_sha256"] for row in rows},
            "serving_authority": False, "qualification_evidence": False}
    report = {**body, "receipt_sha256": _digest(body)}
    _save_if_absent(args.directory / "report.json", report)
    print(json.dumps({"stage": "complete", "held_population": len(rows),
                      "correct_reachable": body["correct_reachable"],
                      "ordinary_correct": body["ordinary_correct"],
                      "top_joint_score_correct": top_correct}), flush=True)


if __name__ == "__main__":
    main()
