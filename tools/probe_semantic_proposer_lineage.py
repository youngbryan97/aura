#!/usr/bin/env python3
"""Replay one frozen proposer on the same source identities as a cross-fit probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()

    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.directory / "report.json")
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.diagnose_semantic_gap_candidate_bank import _audit_row

    loader = compositional_semantic_program_transducer_from_dict(
        json.loads(args.loader.read_bytes()))
    candidate = compositional_semantic_program_transducer_from_dict(
        json.loads(args.candidate.read_bytes()))
    source_raw, plan_raw = args.source_report.read_bytes(), args.source_plan.read_bytes()
    source_report, source_plan = json.loads(source_raw), json.loads(plan_raw)
    if (source_plan.get("schema") != "aura.semantic_proposer_crossfit_plan.v1"
            or source_plan["parent_receipt_sha256"] != loader.receipt_sha256
            or source_plan["source_report_sha256"] != hashlib.sha256(source_raw).hexdigest()
            or source_plan["plan_sha256"] != _digest({
                key: value for key, value in source_plan.items() if key != "plan_sha256"})
            or candidate.model_basis_sha256 != loader.model_basis_sha256
            or candidate.input_grounding.contract_sha256 != loader.input_grounding.contract_sha256):
        raise ValueError("lineage probe model or source plan differs")
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(loader, source_report, bundles)
    items = {item.ir.source_text_sha256: item for item in examples if item.split == "train"}
    ids = source_plan["held_ids"]
    if len(set(ids)) != len(ids) or not set(ids) <= set(items):
        raise ValueError("lineage probe held sources differ")
    body = {"schema": "aura.semantic_proposer_lineage_plan.v1",
            "source_plan_sha256": source_plan["plan_sha256"],
            "candidate_receipt_sha256": candidate.receipt_sha256,
            "candidate_training_ids_sha256": candidate.training_receipt[
                "training_example_ids_sha256"],
            "candidate_trained_on_full_source": candidate.training_receipt[
                "training_example_ids_sha256"] == _sha(sorted(items)),
            "max_charts": source_plan["max_charts"],
            "max_graphs_per_chart": source_plan["max_graphs_per_chart"],
            "solve_seconds": source_plan["solve_seconds"],
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": _digest(body)}
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "rows").mkdir(exist_ok=True)
    rows = []
    for source in ids:
        path = args.directory / "rows" / f"{source}.json"
        if not path.exists():
            row_body = _audit_row(items[source], candidate, plan)
            payload = {**row_body, "receipt_sha256": _digest(row_body)}
            if not atomic_write_bytes_if_absent(path, json.dumps(
                    payload, sort_keys=True).encode(), mode=0o400):
                raise FileExistsError(path)
        row = json.loads(path.read_bytes())
        if (row["source"] != source or row["plan_sha256"] != plan["plan_sha256"]
                or row["receipt_sha256"] != _digest({
                    key: value for key, value in row.items() if key != "receipt_sha256"})):
            raise ValueError("lineage probe held row changed")
        rows.append(row)
        print(json.dumps({"stage": "held", "done": len(rows), "total": len(ids),
                          "source": source}), flush=True)
    report_body = {"schema": "aura.semantic_proposer_lineage_probe.v1",
                   "plan": plan, "population": len(rows),
                   "ordinary_correct": sum(row["diagnosis"]["selected_semantic_status"]
                                           == "equivalent" for row in rows),
                   "correct_reachable": sum(row["diagnosis"]["correct_reachable"] is True
                                            for row in rows),
                   "row_receipts": {row["source"]: row["receipt_sha256"] for row in rows},
                   "serving_authority": False, "qualification_evidence": False}
    report = {**report_body, "receipt_sha256": _digest(report_body)}
    path = args.directory / "report.json"
    payload = json.dumps(report, sort_keys=True).encode()
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError("lineage probe report changed")
    print(json.dumps({"stage": "complete", "population": len(rows),
                      "ordinary_correct": report_body["ordinary_correct"],
                      "correct_reachable": report_body["correct_reachable"]}), flush=True)


if __name__ == "__main__":
    main()
