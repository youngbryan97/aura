#!/usr/bin/env python3
"""Separate public operation proposal gaps from failed chart selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def operation_inventory_audit(model, item):
    # Freeze the public inventory before consulting any annotation.
    spans, _scores, _arguments, inventory = model._runtime_operation_charts(
        item.ir.source_token_ids, item.hidden_states, item.public_inputs,
        model.inference_step_limit(len(item.public_inputs)))
    charts = tuple(inventory)
    target = tuple((value.op, value.operation_span.start, value.operation_span.end)
        for value in sorted(item.ir.instructions, key=lambda value: (
            value.operation_span.start, value.operation_span.end)))
    records = []
    for index, chart in enumerate(charts):
        signature = tuple((node.operation, node.span.start, node.span.end) for node in chart)
        score = math.fsum(node.score for node in chart) - model.operation_length_penalty * len(chart)
        if not math.isfinite(score):
            raise ValueError("operation audit needs finite public chart scores")
        records.append({"index": index, "operations": signature, "operation_score": score,
            "target_count": len(chart) == len(target),
            "target_labels": tuple(row[0] for row in signature) == tuple(row[0] for row in target),
            "target_spans": signature == target})
    matching = [row for row in records if row["target_labels"]]
    return {"source_id": item.ir.source_text_sha256, "construction": item.construction_id,
        "input_anchors_match_annotation": set(spans) == set(item.ir.input_spans),
        "annotated_operations": target, "charts": records, "chart_count": len(charts),
        "correct_count_proposed": any(row["target_count"] for row in records),
        "correct_labels_proposed": bool(matching),
        "correct_spans_proposed": any(row["target_spans"] for row in records),
        "first_correct_label_index": matching[0]["index"] if matching else None,
        "best_correct_label_operation_margin": (
            max(row["operation_score"] for row in matching) - records[0]["operation_score"]
            if matching else None),
        "annotations_used_for_proposals": False, "backbone_loaded": False,
        "bounded_inventory_is_universal_coverage": False, "serving_authority": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "adjudication", "parent", "source-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--source-id", action="append", default=[])
    args = parser.parse_args(argv)
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.file_read_gateway import read_stable_bytes
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )
    from tools.run_semantic_grounded_handoff import publish
    from tools.semantic_grounded_development_archive import MAX_ROW_BYTES, digest
    from tools.verify_semantic_grounded_evaluation import document

    configure_refit_environment(args.output)
    if args.output.exists() or len(set(args.source_id)) != len(args.source_id):
        raise ValueError("operation audit needs a fresh output and unique source identities")
    report = document(json.loads(read_stable_bytes(args.report, max_bytes=MAX_ROW_BYTES)), "receipt_sha256")
    adjudication = document(json.loads(read_stable_bytes(args.adjudication, max_bytes=MAX_ROW_BYTES)), "receipt_sha256")
    source_bytes = read_stable_bytes(args.source_report, max_bytes=MAX_ROW_BYTES)
    if (adjudication["evaluation_report_receipt_sha256"] != report["receipt_sha256"]
            or adjudication["plan_sha256"] != report["plan"]["plan_sha256"]
            or adjudication["semantic_interpretations_independently_replayed"] is not True
            or report["plan"].get("population") != "source_validation"
            or hashlib.sha256(source_bytes).hexdigest() != report["plan"]["source_report_sha256"]):
        raise ValueError("operation audit differs from the independently regraded source population")
    parent = compositional_semantic_program_transducer_from_dict(json.loads(
        read_stable_bytes(args.parent, max_bytes=MAX_ROW_BYTES)))
    examples = {item.ir.source_text_sha256: item for item in load_source_examples(
        parent, json.loads(source_bytes), args.bundle)}
    identities = args.source_id or report["plan"]["held_ids"]
    if not identities or not set(identities) <= set(report["plan"]["held_ids"]):
        raise ValueError("operation audit cannot substitute its measured population")
    rows = []
    for identity in identities:
        row = operation_inventory_audit(parent, examples[identity])
        row["measured_program_outcomes"] = {arm: next(value for value in outcomes
            if value["source_id"] == identity) for arm, outcomes in adjudication["source_outcomes"].items()}
        rows.append(row)
        print(json.dumps({"stage": "operation_inventory_audited", "completed": len(rows),
            "total": len(identities), "source_id": identity, "chart_count": row["chart_count"],
            "correct_labels_proposed": row["correct_labels_proposed"],
            "first_correct_label_index": row["first_correct_label_index"]}), flush=True)
    body = {"schema": "aura.grounded_operation_inventory_audit.v1",
        "report_receipt_sha256": report["receipt_sha256"],
        "adjudication_receipt_sha256": adjudication["receipt_sha256"],
        "parent_receipt_sha256": parent.receipt_sha256, "source_ids": identities, "rows": rows,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "population": len(rows), "summary": {key: sum(row[key] for row in rows) for key in (
            "correct_count_proposed", "correct_labels_proposed", "correct_spans_proposed")},
        "held_labels_used_for_fit": False, "backbone_loaded": False,
        "g03_complete": False, "qualification_evidence": False, "serving_authority": False}
    result = publish(args.output, body)
    if result["receipt_sha256"] != digest(body):
        raise ValueError("operation audit publication differs from measured results")
    print(json.dumps({"stage": "operation_inventory_complete", "receipt_sha256": result["receipt_sha256"],
        "summary": result["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
