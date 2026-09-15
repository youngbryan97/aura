"""Train the existing semantic heads against witnessed runtime interpretations."""

from collections import Counter
from dataclasses import replace
import math

import numpy as np

from core.learning.semantic_graph_counterexamples import (
    argument_graph_program, compare_program_meanings, counterfactual_inputs,
)
from core.learning.semantic_operation_graph_learning import operation_graph_evidence
from core.learning.semantic_relation_graph_learning import RelationGraphContrast, fit_joint_graph_contrasts


def score_annotated_graph(model, item, instructions, input_spans, *, solve_time_limit_s=20.):
    """Score a supplied graph with the runtime's latent mention and definition choices."""
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode

    # Runtime charts are source-ordered even when the annotation is execution-ordered.
    order = sorted(range(len(instructions)), key=lambda index: (
        instructions[index].operation_span.start, instructions[index].operation_span.end))
    count = len(item.public_inputs)
    remap = {count + old: count + new for new, old in enumerate(order)}
    arguments = tuple(tuple(register if register < count else remap[register]
                           for register in instructions[index].args) for index in order)
    pointer = model.operation_pointer.score_sequence(item.hidden_states)
    nodes = tuple(_OperationNode(instructions[index].operation_span, instructions[index].op, 0.,
                  pointer.score_span(instructions[index].operation_span), 1.) for index in order)
    operations = operation_graph_evidence(model, item.hidden_states, nodes)
    components = tuple(np.asarray(value, dtype=np.float64) for head in model.operation_head.heads
                       for value in (head.weight, head.bias))
    nodes = tuple(replace(node, score=node.pointer_score + bank.score_gradient(label, components)[0])
                  for node, (bank, label) in zip(nodes, operations, strict=True))
    charts = []
    _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
        input_spans=input_spans, operation_nodes=nodes,
        argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
        chart_observer=charts.append, retain_score_factors=True, retain_relation_evidence=True, build_only=True)
    if not charts:
        return None
    evidence = []
    result = charts[0].restrict_arguments(arguments).solve_with_factors(
        time_limit_s=solve_time_limit_s, relation_observer=evidence.append)
    if result is None:
        return None
    score = result[0][0] + sum(node.score for node in nodes) - model.operation_length_penalty * len(nodes)
    return {"score": score, "argument_score": result[0][0], "relations": evidence[0],
            "operations": operations, "program": argument_graph_program(nodes, arguments, n_inputs=count)}


def joint_graph_contrast(model, positive, negative, *, weight=1.):
    """Remove variable terms from the replayed margin before differentiating them."""
    head = model.definition_relation_head
    projections = (head.query_projection.astype(np.float64), head.definition_projection.astype(np.float64))
    operations = tuple(value.astype(np.float64) for head in model.operation_head.heads
                       for value in (head.weight, head.bias))
    variable = 0.
    for sign, graph in ((1., positive), (-1., negative)):
        variable += sign * (model.definition_relation_scale * sum(
            bank.score_gradient(index, *projections)[0] for bank, index in graph["relations"])
            + sum(bank.score_gradient(index, operations)[0] for bank, index in graph["operations"]))
    return RelationGraphContrast(positive["relations"], negative["relations"],
        positive["score"] - negative["score"] - variable, weight,
        positive["operations"], negative["operations"])


def mine_runtime_graph_contrast(model, item, *, weight=1., solve_time_limit_s=20.):
    """Interpret without annotations, then independently compare to the source target."""
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError

    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    record = {"source_text_sha256": item.ir.source_text_sha256,
              "serving_authority": False, "negative_origin": "runtime_decode"}
    if outcome.ir is None:
        return None, {**record, "status": "runtime_decode_unavailable", "reason": outcome.refusal}
    from types import SimpleNamespace

    def program(instructions):
        return argument_graph_program(tuple(SimpleNamespace(operation=ins.op, span=ins.operation_span)
            for ins in instructions), tuple(ins.args for ins in instructions), n_inputs=len(item.public_inputs))

    comparison = compare_program_meanings(program(item.ir.instructions), program(outcome.ir.instructions),
                                         counterfactual_inputs(item.public_inputs))
    record["comparison"] = comparison
    if comparison["status"] != "different":
        return None, {**record, "status": comparison["status"]}
    try:
        negative = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans,
                                        solve_time_limit_s=solve_time_limit_s)
        positive = score_annotated_graph(model, item, item.ir.instructions, outcome.ir.input_spans,
                                        solve_time_limit_s=solve_time_limit_s)
    except ArgumentOptimizationIncompleteError as exc:
        return None, {**record, "status": "graph_score_incomplete", "reason": str(exc)}
    if positive is None or negative is None:
        return None, {**record, "status": "target_unreachable" if positive is None else "runtime_replay_unavailable"}
    if not math.isclose(negative["argument_score"], outcome.pointer_scores["argument_graph_total"], abs_tol=1e-4, rel_tol=1e-6):
        return None, {**record, "status": "runtime_score_replay_differs",
                      "runtime_score": outcome.pointer_scores["argument_graph_total"],
                      "replayed_score": negative["argument_score"]}
    record.update(status="counterexample", initial_margin=positive["score"] - negative["score"],
                  positive_program_sha256=positive["program"].sha(), negative_program_sha256=negative["program"].sha())
    return joint_graph_contrast(model, positive, negative, weight=weight), record


def refit_compositional_joint_graphs(model, examples, *, rounds=3, steps=100,
                                    solve_time_limit_s=20., progress=None):
    """Remine source-training predictions after each joint operation/relation update."""
    from core.learning.semantic_graph_margin import graph_refit_source_splits
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_shared_transducer import _geometry

    if type(rounds) is not int or rounds < 1:
        raise ValueError("joint graph learning rounds must be positive")
    training, validation = graph_refit_source_splits(model, examples)
    if (model.training_receipt.get("definition_selection_policy") != "joint_graph_v1"
            or model.training_receipt.get("relation_score_strategy") != "categorical_log_margin_v1"
            or model.training_receipt.get("operation_assignment_policy") != "joint_factor_score_v2"):
        raise ValueError("joint training requires the joint categorical runtime decoder")
    weights = Counter(_geometry(item) for item in training)
    candidate, retained, history = model, [], []
    for round_index in range(rounds):
        records, new_pairs = [], 0
        for index, item in enumerate(training):
            contrast, record = mine_runtime_graph_contrast(candidate, item,
                weight=1. / weights[_geometry(item)], solve_time_limit_s=solve_time_limit_s)
            if contrast is not None:
                record["retained_pair"] = len(retained)
                retained.append(contrast)
                new_pairs += 1
            records.append(record)
            if progress:
                progress({"stage": "joint_graph_mining", "round": round_index + 1,
                          "completed": index + 1, "total": len(training), "row": record})
        if not new_pairs:
            history.append({"records": records, "fit": {"status": "no_new_witnessed_errors",
                            "pairs": len(retained), "coverage_complete": all(
                                row["status"] == "equivalent" for row in records)}})
            break
        relation, operation, fit = fit_joint_graph_contrasts(candidate.definition_relation_head,
            candidate.operation_head, tuple(retained), scale=candidate.definition_relation_scale, steps=steps)
        candidate = candidate._with_coefficients(definition_relation_head=relation, operation_head=operation)
        history.append({"records": records, "fit": fit})
        if progress:
            progress({"stage": "joint_graph_fit", "round": round_index + 1, "fit": fit})
    body = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
    body["joint_graph_refit"] = {
        "schema": "aura.semantic_joint_graph_refit.v1", "parent_transducer_receipt_sha256": model.receipt_sha256,
        "training_examples": len(training), "validation_examples": len(validation),
        "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in training)),
        "validation_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in validation)),
        "rounds": history, "negative_origin": "runtime_decode", "positive_origin": "source_annotations",
        "negative_admission": "universal_floor_distinguishing_execution", "test_examples_used": 0,
        "validation_used_for_fit": False, "serving_authority": False,
    }
    return replace(candidate, training_receipt={**body, "receipt_sha256": _sha(body)})
