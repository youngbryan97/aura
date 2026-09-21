"""Retain witnessed competitors from the decoder's operation and argument search."""

from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache, counterfactual_inputs, find_program_counterexample,
)
from core.learning.semantic_joint_graph_learning import (
    align_source_input_registers, graph_selection_key, preferred_semantic_graph, selection_graph_contrast, selection_score_margin,
    score_annotated_graph, scored_graph_evidence,
)
from core.learning.semantic_operation_search import OperationChartSearch, OperationSearchIncompleteError
from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments
from typing import Any


def mine_runtime_graph_constraints(
    model: Any,
    item: Any,
    *,
    weight: float=1.0,
    max_charts: int=32,
    max_graphs: int=32,
    solve_time_limit_s: float=20.0,
    learn_arguments: bool=False,
    learn_operation_pointer: bool=False,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Search source-training competitors without supplying operations to the decoder.

    Only the offline comparison sees the annotation. Search allowances are
    recorded; exhaustion of an allowance cannot certify semantic coverage.
    """
    if item.split != "train":
        raise ValueError("runtime constraint mining requires source training examples")
    if any(type(value) is not int or value < 1 for value in (max_charts, max_graphs)):
        raise ValueError("runtime constraint search allowances must be positive")
    record = {"schema": "aura.semantic_runtime_graph_retention.v1",
              "source_text_sha256": item.ir.source_text_sha256, "serving_authority": False,
              "candidate_origin": "runtime_operation_charts", "source_operations_supplied": False,
              "max_charts": max_charts, "max_graphs_per_chart": max_graphs,
              "charts": [], "candidate_inventory_exhausted": False,
              "operation_search_complete": False, "highest_incorrect_proven": False}
    step_limit = model.inference_step_limit(len(item.public_inputs))
    if step_limit is None:
        return (), {**record, "status": "public_input_count_unsupported"}
    argument_evidence_cache = {}
    try:
        spans, _, argument_scores, candidates = model._runtime_operation_charts(
            item.ir.source_token_ids, item.hidden_states, item.public_inputs, step_limit)
        instructions, mapping = align_source_input_registers(item, spans)
        positive = score_annotated_graph(model, item, instructions, spans,
            solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments,
            learn_operation_pointer=learn_operation_pointer,
            argument_evidence_cache=argument_evidence_cache)
    except (ValueError, ArgumentOptimizationIncompleteError) as exc:
        return (), {**record, "status": "source_graph_unavailable", "reason": str(exc)}
    if positive is None:
        return (), {**record, "status": "target_unreachable"}
    record["source_to_runtime_input_registers"] = list(mapping)
    target = positive["program"]
    probes = counterfactual_inputs(item.public_inputs)
    observation_cache = ProgramObservationCache()
    candidates = iter(candidates)
    relation_scores, relation_vectors = {}, {}
    definition_scores = model.definition_pointer.score_sequence(item.hidden_states)
    negatives = []
    try:
        for index in range(max_charts + 1):
            nodes = next(candidates, None)
            if nodes is None:
                record["candidate_inventory_exhausted"] = True
                record["operation_search_complete"] = (
                    isinstance(candidates, OperationChartSearch) and candidates.complete)
                break
            if index == max_charts:
                break
            charts = []
            _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
                source_token_ids=item.ir.source_token_ids,
                input_spans=spans, operation_nodes=nodes, argument_pointer_scores=argument_scores,
                relation_score_cache=relation_scores, relation_vector_cache=relation_vectors,
                definition_pointer_scores=definition_scores, chart_observer=charts.append,
                retain_score_factors=True, retain_relation_evidence=True, build_only=True)
            row = {"operations": [{"op": node.operation, "span": [node.span.start, node.span.end]}
                                  for node in nodes]}
            record["charts"].append(row)
            if not charts:
                row.update(status="typed_chart_empty", search_complete=True)
                continue
            result = find_program_counterexample(charts[0], nodes, target, probes=probes,
                max_graphs=max_graphs, solve_time_limit_s=solve_time_limit_s,
                observation_cache=observation_cache)
            row.update(result.receipt)
            normalizers = None
            if (model.training_receipt.get("argument_choice_normalization") == "local_categorical_v1"
                    and (result.positive is not None or result.negative is not None)):
                from core.learning.semantic_choice_evidence import argument_choice_normalizers

                normalizers = argument_choice_normalizers(model, item.hidden_states, nodes, charts[0],
                    learn_arguments=learn_arguments, argument_evidence_cache=argument_evidence_cache)
            if result.positive is not None:
                alternative = scored_graph_evidence(model, item, nodes, result.positive,
                    result.positive_evidence, learn_arguments=learn_arguments,
                    learn_operation_pointer=learn_operation_pointer, chart=charts[0],
                    choice_normalizers=normalizers)
                if preferred_semantic_graph(model, alternative, positive):
                    positive = alternative
            if result.negative is not None:
                negative = scored_graph_evidence(model, item, nodes, result.negative,
                    result.negative_evidence, learn_arguments=learn_arguments,
                    learn_operation_pointer=learn_operation_pointer, chart=charts[0],
                    choice_normalizers=normalizers)
                row["negative_index"] = len(negatives)
                negatives.append(negative)
    except (ArgumentOptimizationIncompleteError, OperationSearchIncompleteError) as exc:
        record["interruption"] = str(exc)
    record["highest_incorrect_proven"] = bool(negatives) and record["operation_search_complete"] and all(
        row.get("highest_incorrect_proven") or row["status"] in {"no_incorrect_graph", "typed_chart_empty"}
        for row in record["charts"])
    record.update(status="counterexamples" if negatives else "no_witnessed_competitor",
                  floor_observation_reuse=observation_cache.statistics(),
                  positive_program_sha256=positive["program"].sha(),
                  negative_program_sha256s=[row["program"].sha() for row in negatives],
                  initial_margins=[selection_score_margin(model, positive, row) for row in negatives],
                  selection_policy=model.training_receipt.get("operation_assignment_policy", "first_feasible_v1"),
                  positive_selection_key=graph_selection_key(model, positive),
                  negative_selection_keys=[graph_selection_key(model, row) for row in negatives])
    return tuple(selection_graph_contrast(model, positive, negative, weight=weight)
                 for negative in negatives), record
