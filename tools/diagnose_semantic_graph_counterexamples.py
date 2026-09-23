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


def select_diagnostic_examples(examples, *, split, source_ids=(), fit_steps=0, capacity_report=None):
    """Allow held-out attribution without admitting held-out labels to a fit."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown diagnostic split")
    if split != "train" and (fit_steps or capacity_report is not None):
        raise ValueError("held-out diagnostic rows cannot fit tissue or select source capacity")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate diagnostic source id")
    selected = tuple(item for item in examples if item.split == split
                     and (not source_ids or item.ir.source_text_sha256 in source_ids))
    if source_ids and {item.ir.source_text_sha256 for item in selected} != set(source_ids):
        raise ValueError("diagnostic source id is missing from the selected split")
    if not selected:
        raise ValueError("diagnostic split has no observations")
    return selected


def runtime_bank_diagnostic(model, item, *, max_charts, max_graphs, solve_time_limit_s):
    """Freeze answer-blind alternatives before comparing their operation spans."""
    bank = model.decode_candidates(source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states, public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
        max_charts=max_charts, max_graphs_per_chart=max_graphs,
        solve_time_limit_s=solve_time_limit_s)
    bank.validate()
    source_spans = tuple(instruction.operation_span.to_dict() for instruction in item.ir.instructions)
    selected_spans = (tuple(instruction.operation_span.to_dict()
                            for instruction in bank.selected.ir.instructions)
                      if bank.selected.ir is not None else None)
    return {"bank_receipt": bank.receipt, "source_operation_spans": source_spans,
            "selected_operation_spans": selected_spans,
            "source_operation_chart_reached": any(
                tuple(entry["span"] for entry in chart["operations"]) == source_spans
                for chart in bank.receipt["charts"]),
            "selected_operation_spans_match_source": (
                selected_spans == source_spans if selected_spans is not None else None),
            "source_labels_sent_to_candidate_search": False}


def runtime_matching_chart_diagnostics(model, item, *, bank_charts, max_charts, max_graphs, solve_time_limit_s):
    """Grade frozen operation hypotheses against source bindings after search."""
    from core.learning.semantic_graph_counterexamples import counterfactual_inputs, find_graph_counterexample
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments
    from core.learning.semantic_program_transducer import _hidden_array
    from core.learning.semantic_program_ir import normalize_semantic_value

    inputs = tuple(normalize_semantic_value(value) for value in item.public_inputs)
    hidden = _hidden_array(item.hidden_states, expected_width=model.hidden_size)
    tokens = tuple(item.ir.source_token_ids)
    spans, _, argument_scores, operation_charts = model._runtime_operation_charts(
        tokens, hidden, inputs, model.inference_step_limit(len(inputs)))
    source_operations = tuple(instruction.op for instruction in item.ir.instructions)
    target_arguments = tuple(instruction.args for instruction in item.ir.instructions)
    comparisons = []
    for index, nodes in enumerate(operation_charts):
        if index >= max_charts:
            break
        if index >= len(bank_charts) or [
            {"op": node.operation, "span": node.span.to_dict()} for node in nodes
        ] != bank_charts[index]["operations"]:
            raise ValueError("runtime chart replay differs from frozen candidate inventory")
        if tuple(node.operation for node in nodes) != source_operations:
            continue
        captured = []
        _assign_typed_arguments(model=model, hidden=hidden, inputs=inputs, input_spans=spans,
            source_token_ids=tokens, operation_nodes=nodes,
            argument_pointer_scores=argument_scores,
            chart_observer=captured.append, retain_score_factors=True, build_only=True)
        if not captured:
            comparisons.append({"chart_index": index, "status": "chart_unavailable"})
            continue
        result = find_graph_counterexample(captured[0], nodes, target_arguments,
            probes=counterfactual_inputs(item.public_inputs), max_graphs=max_graphs,
            solve_time_limit_s=solve_time_limit_s)
        comparisons.append({"chart_index": index, "status": result.receipt["status"],
            "target_margin": (result.positive[0][0] - result.negative[0][0]
                              if result.positive is not None and result.negative is not None else None),
            "positive_score": result.positive[0][0] if result.positive is not None else None,
            "negative_score": result.negative[0][0] if result.negative is not None else None,
            "positive_factors": result.positive[1] if result.positive is not None else None,
            "negative_factors": result.negative[1] if result.negative is not None else None,
            "target_arguments_used_after_operation_inventory": True})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--capacity-report", type=Path,
                        help="diagnose only the source comparisons in this capacity witness")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train",
                        help="held-out rows are diagnostic only and cannot update tissue")
    parser.add_argument("--source-id", action="append", default=[],
                        help="restrict the diagnostic to exact source identities in the selected split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-graphs", type=int, default=128)
    parser.add_argument("--solve-time-limit", type=float, default=20.)
    parser.add_argument("--fit-relation-steps", type=int, default=0,
                        help="diagnostic fit of witnessed pairs; never exports a serving candidate")
    parser.add_argument("--runtime-bank", action="store_true",
                        help="capture answer-blind operation charts alongside source-chart margins")
    parser.add_argument("--runtime-max-charts", type=int, default=8)
    parser.add_argument("--runtime-max-graphs", type=int, default=8)
    args = parser.parse_args()
    from core.governance_context import local_internal_governed_scope
    from core.learning.score_capacity import assess_score_capacity, verify_score_capacity
    from core.learning.semantic_graph_counterexamples import (
        argument_graph_program, compare_program_meanings, counterfactual_inputs, find_graph_counterexample,
    )
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
    from core.learning.semantic_relation_graph_learning import contrast_from_search, fit_relation_graph_contrasts
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.refit_semantic_argument_proposals import load_source_examples
    from tools.run_semantic_program_replication import _load_json

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.fit_relation_steps < 0:
        raise ValueError("diagnostic update steps must be nonnegative")
    if args.runtime_bank and (args.runtime_max_charts < 1 or args.runtime_max_graphs < 1):
        raise ValueError("runtime bank allowances must be positive")
    model = compositional_semantic_program_transducer_from_dict(_load_json(args.candidate, max_bytes=64*1024*1024))
    source = _load_json(args.source_report, max_bytes=64*1024*1024)
    examples = load_source_examples(model, source, args.bundle)
    selected = select_diagnostic_examples(examples, split=args.split, source_ids=args.source_id,
        fit_steps=args.fit_relation_steps, capacity_report=args.capacity_report)
    if args.capacity_report:
        capacity = _load_json(args.capacity_report, max_bytes=64*1024*1024)
        if not verify_score_capacity(capacity) or capacity["status"] != "infeasible":
            raise ValueError("source selection needs a verified capacity contradiction")
        if capacity.get("source_binding", {}).get("parent_transducer_receipt_sha256") != model.receipt_sha256:
            raise ValueError("capacity comparison scores belong to a different parent")
        wanted = set(capacity["conflicting_comparison_ids"])
        selected = tuple(item for item in selected if item.ir.source_text_sha256 in wanted)
        if {item.ir.source_text_sha256 for item in selected} != wanted:
            raise ValueError("capacity witness does not belong to admitted training sources")
    records, differences, offsets, ids = [], [], [], []
    relation_contrasts = []
    scales = (model.argument_role_scale, model.argument_proposal_scale,
              model.definition_relation_scale, model.argument_pointer_scale)
    for item in selected:
        nodes = tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions)
        charts = []
        _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
            input_spans=item.ir.input_spans, operation_nodes=nodes,
            argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
            chart_observer=charts.append, retain_score_factors=True, build_only=True,
            retain_relation_evidence=args.fit_relation_steps > 0)
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
                if args.fit_relation_steps:
                    relation_contrasts.append(contrast_from_search(result, model.definition_relation_head,
                                                                  scale=model.definition_relation_scale))
                factors = [p - n for p, n in zip(result.positive[1], result.negative[1], strict=True)]
                margin = result.positive[0][0] - result.negative[0][0]
                fixed = margin - sum(s * f for s, f in zip(scales, factors, strict=True))
                row.update(factor_difference=factors, fixed_difference=fixed, target_margin=margin)
                row.update(positive_score=result.positive[0][0], negative_score=result.negative[0][0],
                           positive_factors=result.positive[1], negative_factors=result.negative[1])
                differences.append([factors[i] for i in (0, 2, 3)])
                offsets.append(fixed + scales[1] * factors[1])
                ids.append(item.ir.source_text_sha256)
        if args.runtime_bank:
            row["runtime_bank"] = runtime_bank_diagnostic(model, item,
                max_charts=args.runtime_max_charts, max_graphs=args.runtime_max_graphs,
                solve_time_limit_s=args.solve_time_limit)
            row["runtime_bank"]["matched_operation_chart_comparisons"] = (
                runtime_matching_chart_diagnostics(model, item,
                    bank_charts=row["runtime_bank"]["bank_receipt"]["charts"],
                    max_charts=args.runtime_max_charts, max_graphs=args.runtime_max_graphs,
                    solve_time_limit_s=args.solve_time_limit))
        records.append(row)
        print(json.dumps({"completed": len(records), "total": len(selected), "row": row}, sort_keys=True), flush=True)
    capacity = assess_score_capacity(differences, offsets, lower_bounds=[1e-6]*3, comparison_ids=ids,
                                    margin=0., strict=True) if differences and args.split == "train" else None
    report = {"schema": "aura.semantic_graph_counterexample_diagnostic.v1", "rows": records,
              "candidate_receipt_sha256": model.receipt_sha256, "capacity": capacity,
              "split": args.split, "source_ids": [item.ir.source_text_sha256 for item in selected],
              "training_examples": len(selected) if args.split == "train" else 0,
              "validation_examples": len(selected) if args.split == "validation" else 0,
              "test_examples": len(selected) if args.split == "test" else 0,
              "held_out_labels_used_for_fit": False,
              "operation_boundaries": "source_annotations", "serving_authority": False}
    if args.fit_relation_steps and relation_contrasts:
        fitted, receipt = fit_relation_graph_contrasts(model.definition_relation_head, tuple(relation_contrasts),
                                                      scale=model.definition_relation_scale, steps=args.fit_relation_steps)
        def margins(head):
            projections = (head.query_projection.astype('float64'), head.definition_projection.astype('float64'))
            return [row.fixed_margin + model.definition_relation_scale * (
                sum(bank.score_gradient(index, *projections)[0] for bank, index in row.positive)
                - sum(bank.score_gradient(index, *projections)[0] for bank, index in row.negative))
                for row in relation_contrasts]
        report['diagnostic_relation_fit'] = {**receipt, 'initial_margins': margins(model.definition_relation_head),
            'fitted_margins': margins(fitted), 'candidate_exported': False,
            'post_update_graph_search_performed': False}
        replay_model = model._with_coefficients(definition_relation_head=fitted)
        replays = []
        for item in selected:
            nodes = tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions)
            charts = []
            _assign_typed_arguments(model=replay_model, hidden=item.hidden_states, inputs=item.public_inputs,
                input_spans=item.ir.input_spans, operation_nodes=nodes,
                argument_pointer_scores=replay_model.argument_pointer.score_sequence(item.hidden_states),
                chart_observer=charts.append, build_only=True)
            replay = {'source_text_sha256': item.ir.source_text_sha256, 'status': 'chart_unavailable'}
            if charts:
                try:
                    chosen = charts[0].solve(time_limit_s=args.solve_time_limit)
                except ArgumentOptimizationIncompleteError as exc:
                    replay.update(status='search_incomplete', reason=str(exc))
                else:
                    if chosen is None:
                        replay['status'] = 'no_feasible_graph'
                    else:
                        target = argument_graph_program(nodes, tuple(i.args for i in item.ir.instructions),
                                                        n_inputs=len(item.public_inputs))
                        actual = argument_graph_program(nodes, chosen[1], n_inputs=len(item.public_inputs))
                        replay.update(status='selected', arguments=chosen[1], comparison=compare_program_meanings(
                            target, actual, counterfactual_inputs(item.public_inputs)))
            replays.append(replay)
            print(json.dumps({'stage': 'fitted_graph_replay', 'row': replay}, sort_keys=True), flush=True)
        report['diagnostic_relation_fit'].update(post_update_graph_search_performed=True, replay=replays)
    with local_internal_governed_scope("semantic_graph_counterexample.report", domain="file_write"):
        if not atomic_write_bytes_if_absent(args.output, (json.dumps(report, sort_keys=True)+"\n").encode(), mode=0o400):
            raise FileExistsError(args.output)
    print(json.dumps({"complete": True, "capacity_status": capacity["status"] if capacity else None,
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
