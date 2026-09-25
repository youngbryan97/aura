#!/usr/bin/env python3
"""Locate target operation loss inside a frozen source-only proposer.

Targets are read only after the candidate has generated its bank. This is an
offline diagnostic, never an input to candidate generation or serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_semantic_crossfit_misses import _digest, _load_verified
from tools.profile_semantic_proposal_gaps import classify_proposal_gap


def classify_inventory(item, ranked_nodes, complete_nodes) -> dict:
    ranked = {(node.span, node.operation) for node in ranked_nodes}
    complete = {(node.span, node.operation) for node in complete_nodes}
    complete_spans = {node.span for node in complete_nodes}
    ranked_spans = {node.span for node in ranked_nodes}
    losses = []
    for instruction in item.ir.instructions:
        pair = (instruction.operation_span, instruction.op)
        if pair in ranked:
            stage = "operation_node_retained"
        elif instruction.operation_span not in complete_spans:
            stage = "source_span_outside_declared_inventory"
        elif pair not in complete:
            stage = "target_operation_outside_learned_vocabulary"
        elif instruction.operation_span not in ranked_spans:
            stage = "source_span_pruned"
        else:
            stage = "operation_label_pruned"
        losses.append(stage)
    return {"operation_nodes": losses,
            "all_target_nodes_retained": all(stage == "operation_node_retained"
                                         for stage in losses)}


def inventory_nodes(model, item, *, complete: bool):
    from core.learning.semantic_candidate_bank import _hidden_array
    from core.learning.semantic_program_transducer_fitting import _operation_nodes

    hidden = _hidden_array(item.hidden_states, expected_width=model.hidden_size)
    input_spans, _, _ = model._runtime_input_grounding(
        tuple(item.ir.source_token_ids), hidden, tuple(item.public_inputs))
    return _operation_nodes(
        pointer=model.operation_pointer,
        classifier=model.operation_head,
        hidden=hidden,
        input_spans=input_spans,
        max_span_tokens=model.max_span_tokens,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
        label_limit=(len(model.operation_head.labels) if complete else
                     model.training_receipt.get("operation_label_limit", 1)),
        complete_inventory=complete,
        background_log_odds=(model.training_receipt.get("operation_background_fit", {}).get("score")
                             == "joint_operation_background_log_odds_v2"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import load_source_examples

    plan, observations = _load_verified(args.bank)
    source_raw = args.source_report.read_bytes()
    parent = compositional_semantic_program_transducer_from_dict(
        json.loads(args.parent.read_bytes()))
    model = compositional_semantic_program_transducer_from_dict(
        json.loads((args.bank / "candidate.json").read_bytes()))
    report = json.loads((args.bank / "report.json").read_bytes())
    if (plan["source_report_sha256"] != hashlib.sha256(source_raw).hexdigest()
            or plan["parent_receipt_sha256"] != parent.receipt_sha256
            or report["candidate_receipt_sha256"] != model.receipt_sha256):
        raise ValueError("operation inventory model or source differs from proposal bank")
    source = json.loads(source_raw)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source["representation_compatibility"]["source_feature_manifest_sha256s"]]
    items = {item.ir.source_text_sha256: item for item in
             load_source_examples(parent, source, bundles) if item.split == "train"}
    rows = []
    for row in observations:
        item = items[row["source"]]
        gap = classify_proposal_gap(item, row)
        if gap["stage"] == "correct_candidate_observed":
            continue
        inventory = classify_inventory(item, inventory_nodes(model, item, complete=False),
                                       inventory_nodes(model, item, complete=True))
        rows.append({"source": row["source"], "construction": item.construction_id,
                     "proposal_stage": gap["stage"], **inventory})
    by_stage = Counter(stage for row in rows for stage in row["operation_nodes"])
    body = {"schema": "aura.semantic_operation_inventory_profile.v1",
            "bank_report_sha256": hashlib.sha256((args.bank / "report.json").read_bytes()).hexdigest(),
            "candidate_receipt_sha256": model.receipt_sha256,
            "population": len(rows), "by_operation_stage": dict(sorted(by_stage.items())),
            "all_target_nodes_retained": sum(row["all_target_nodes_retained"] for row in rows),
            "rows": sorted(rows, key=lambda row: row["source"]),
            "development_only": True, "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _digest(body)}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("operation inventory profile already exists with different content")
    args.output.write_bytes(payload)
    print(json.dumps({"population": body["population"],
                      "by_operation_stage": body["by_operation_stage"],
                      "all_target_nodes_retained": body["all_target_nodes_retained"]}, sort_keys=True))


if __name__ == "__main__":
    main()
