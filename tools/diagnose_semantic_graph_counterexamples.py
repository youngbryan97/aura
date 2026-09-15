#!/usr/bin/env python3
"""Recheck source graph contrasts with semantic equivalence and floor witnesses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--capacity-report", type=Path,
                        help="diagnose only the source comparisons in this capacity witness")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-graphs", type=int, default=128)
    parser.add_argument("--solve-time-limit", type=float, default=20.)
    args = parser.parse_args()
    from core.governance_context import local_internal_governed_scope
    from core.learning.score_capacity import assess_score_capacity, verify_score_capacity
    from core.learning.semantic_graph_counterexamples import counterfactual_inputs, find_graph_counterexample
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.refit_semantic_argument_proposals import load_source_examples
    from tools.run_semantic_program_replication import _load_json

    if args.output.exists():
        raise FileExistsError(args.output)
    model = compositional_semantic_program_transducer_from_dict(_load_json(args.candidate, max_bytes=64*1024*1024))
    source = _load_json(args.source_report, max_bytes=64*1024*1024)
    examples = load_source_examples(model, source, args.bundle)
    selected = [item for item in examples if item.split == "train"]
    if args.capacity_report:
        capacity = _load_json(args.capacity_report, max_bytes=64*1024*1024)
        if not verify_score_capacity(capacity) or capacity["status"] != "infeasible":
            raise ValueError("source selection needs a verified capacity contradiction")
        if capacity.get("source_binding", {}).get("parent_transducer_receipt_sha256") != model.receipt_sha256:
            raise ValueError("capacity comparison scores belong to a different parent")
        wanted = set(capacity["conflicting_comparison_ids"])
        selected = [item for item in selected if item.ir.source_text_sha256 in wanted]
        if {item.ir.source_text_sha256 for item in selected} != wanted:
            raise ValueError("capacity witness does not belong to admitted training sources")
    records, differences, offsets, ids = [], [], [], []
    scales = (model.argument_role_scale, model.argument_proposal_scale,
              model.definition_relation_scale, model.argument_pointer_scale)
    for item in selected:
        nodes = tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions)
        charts = []
        _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
            input_spans=item.ir.input_spans, operation_nodes=nodes,
            argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
            chart_observer=charts.append, retain_score_factors=True, build_only=True)
        row = {"source_text_sha256": item.ir.source_text_sha256}
        if not charts:
            row["status"] = "chart_unavailable"
        else:
            result = find_graph_counterexample(charts[0], nodes, tuple(i.args for i in item.ir.instructions),
                probes=counterfactual_inputs(item.public_inputs), max_graphs=args.max_graphs,
                solve_time_limit_s=args.solve_time_limit,
                progress=lambda event: print(json.dumps({**event, "source": item.ir.source_text_sha256}), flush=True))
            row.update(result.receipt)
            if result.negative is not None:
                factors = [p - n for p, n in zip(result.positive[1], result.negative[1], strict=True)]
                margin = result.positive[0][0] - result.negative[0][0]
                fixed = margin - sum(s * f for s, f in zip(scales, factors, strict=True))
                row.update(factor_difference=factors, fixed_difference=fixed, target_margin=margin)
                differences.append([factors[i] for i in (0, 2, 3)])
                offsets.append(fixed + scales[1] * factors[1])
                ids.append(item.ir.source_text_sha256)
        records.append(row)
        print(json.dumps({"completed": len(records), "total": len(selected), "row": row}, sort_keys=True), flush=True)
    capacity = assess_score_capacity(differences, offsets, lower_bounds=[1e-6]*3, comparison_ids=ids,
                                    margin=0., strict=True) if differences else None
    report = {"schema": "aura.semantic_graph_counterexample_diagnostic.v1", "rows": records,
              "candidate_receipt_sha256": model.receipt_sha256, "capacity": capacity,
              "training_examples": len(selected), "validation_examples": 0, "test_examples": 0,
              "operation_boundaries": "source_annotations", "serving_authority": False}
    with local_internal_governed_scope("semantic_graph_counterexample.report", domain="file_write"):
        if not atomic_write_bytes_if_absent(args.output, (json.dumps(report, sort_keys=True)+"\n").encode(), mode=0o400):
            raise FileExistsError(args.output)
    print(json.dumps({"complete": True, "capacity_status": capacity["status"] if capacity else None,
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
