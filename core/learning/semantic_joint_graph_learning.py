"""Train the existing semantic heads against witnessed runtime interpretations."""

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.learning.semantic_graph_counterexamples import (
    argument_graph_program,
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_operation_graph_learning import (
    OperationEvidenceBank,
    OperationSourceSupervision,
    operation_graph_evidence,
)
from core.learning.semantic_relation_graph_learning import (
    RelationGraphContrast,
    fit_joint_graph_contrasts,
    graph_margin,
)


def score_annotated_graph(model, item, instructions, input_spans, *, solve_time_limit_s=20., learn_arguments=False, learn_operation_pointer=False):
    """Score a supplied graph with the runtime's latent mention and definition choices."""
    from core.learning.semantic_program_transducer_fitting import (
        _assign_typed_arguments,
        _OperationNode,
    )

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
    nodes = tuple(replace(node, score=(node.pointer_score if bank.normalizer_label is None else 0.)
                          + bank.score_gradient(label, components)[0])
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
    return scored_graph_evidence(model, item, nodes, result, evidence[0], learn_arguments=learn_arguments,
                                 learn_operation_pointer=learn_operation_pointer)


def scored_graph_evidence(model, item, nodes, result, relations, *, learn_arguments=False, learn_operation_pointer=False):
    """Retain differentiable terms for the same latent graph the solver selected."""
    operation_score = sum(node.score for node in nodes) - model.operation_length_penalty * len(nodes)
    score = result[0][0] + operation_score
    from core.learning.semantic_argument_graph_learning import argument_graph_evidence
    terms = argument_graph_evidence(model, item.hidden_states, nodes, result[0][2]) if learn_arguments else ()
    binding_terms = terms
    pointer_terms = ()
    if learn_operation_pointer and model.training_receipt.get("operation_background_fit", {}).get("score") \
            != "joint_operation_background_log_odds_v2":
        from core.learning.semantic_operation_pointer_learning import (
            operation_pointer_graph_evidence,
        )
        pointer_terms = operation_pointer_graph_evidence(model, item.hidden_states, nodes)
        terms += pointer_terms
    return {"score": score, "operation_score": operation_score,
            "argument_score": result[0][0], "relations": relations, "argument_terms": terms,
            "operation_pointer_terms": pointer_terms,
            "binding_terms": binding_terms,
            "operation_signature": tuple((node.operation, node.span.start, node.span.end) for node in nodes),
            "operations": operation_graph_evidence(model, item.hidden_states, nodes),
            "program": argument_graph_program(nodes, result[0][1], n_inputs=len(item.public_inputs))}


def joint_graph_contrast(model, positive, negative, *, weight=1.):
    """Remove variable terms from the replayed margin before differentiating them."""
    head = model.definition_relation_head
    projections = (head.query_projection.astype(np.float64), head.definition_projection.astype(np.float64))
    operations = tuple(value.astype(np.float64) for head in model.operation_head.heads
                       for value in (head.weight, head.bias))
    from core.learning.semantic_argument_graph_learning import argument_parameters
    from core.learning.semantic_operation_pointer_learning import operation_pointer_parameters
    parameters = (*projections, *operations, *argument_parameters(model), *operation_pointer_parameters(model))
    terms = tuple((sign, term) for sign, graph in ((1., positive), (-1., negative))
                  for term in graph.get("argument_terms", ()))
    row = RelationGraphContrast(positive["relations"], negative["relations"], 0., weight,
        positive["operations"], negative["operations"], terms)
    variable = graph_margin(parameters, row, scale=model.definition_relation_scale)
    return replace(row, fixed_margin=math.fsum((positive["score"], -negative["score"], -variable)))


def graph_selection_key(model, graph):
    """Keep offline comparisons in the ordering the decoder actually uses."""
    policy = model.training_receipt.get("operation_assignment_policy", "first_feasible_v1")
    if policy == "joint_factor_score_v2":
        return (graph["score"],)
    if policy != "first_feasible_v1":
        raise ValueError("unknown graph selection policy")
    signature = graph["operation_signature"]
    return (graph["operation_score"], -len(signature),
            tuple((-start, -end) for _op, start, end in signature))


def preferred_semantic_graph(model, candidate, incumbent):
    """Compare bindings only within the same first-feasible operation chart."""
    if (model.training_receipt.get("operation_assignment_policy", "first_feasible_v1") == "first_feasible_v1"
            and candidate["operation_signature"] == incumbent["operation_signature"]):
        return candidate["argument_score"] > incumbent["argument_score"]
    return graph_selection_key(model, candidate) > graph_selection_key(model, incumbent)


def selection_score_margin(model, positive, negative):
    if len(graph_selection_key(model, positive)) == 1:
        return positive["score"] - negative["score"]
    if positive["operation_signature"] == negative["operation_signature"]:
        return positive["argument_score"] - negative["argument_score"]
    return graph_selection_key(model, positive)[0] - graph_selection_key(model, negative)[0]


def selection_graph_contrast(model, positive, negative, *, weight=1.):
    """Train the deciding component, not a sum the incumbent never compares.

    First-feasible decoding sorts feasible operation charts by operation
    score, then optimizes arguments within the chosen chart. Raising a
    secondary argument score cannot repair an earlier wrong operation chart.
    Joint decoding continues to use the existing summed-score objective.
    """
    if len(graph_selection_key(model, positive)) == 1:
        return joint_graph_contrast(model, positive, negative, weight=weight)
    same_chart = positive["operation_signature"] == negative["operation_signature"]

    def component(graph):
        if not same_chart:
            return {**graph, "score": graph["operation_score"],
                    "relations": (), "argument_terms": graph["operation_pointer_terms"]}
        return {**graph, "score": graph["argument_score"], "operations": (),
                "argument_terms": graph["binding_terms"]}

    return joint_graph_contrast(model, component(positive), component(negative), weight=weight)


def align_source_input_registers(item, input_spans):
    """Express source labels in the decoder's source-anchored input coordinates.

    Equal-valued literals can exchange register indices during grounding.
    Their source positions still distinguish them when counterfactual probes
    assign different values. A position change must not invert supervision.
    """
    source = tuple(item.ir.input_spans)
    destination = tuple(input_spans)
    if (len(source) != len(destination) or len(set(destination)) != len(destination)
            or set(source) != set(destination)):
        raise ValueError("source and runtime input anchors differ")
    positions = {span: index for index, span in enumerate(destination)}
    mapping = tuple(positions[span] for span in source)
    if any(item.public_inputs[index] != item.public_inputs[target]
           for index, target in enumerate(mapping)):
        raise ValueError("input anchor permutation changes public values")
    count = len(source)
    instructions = tuple(replace(instruction, args=tuple(
        mapping[argument] if argument < count else argument
        for argument in instruction.args)) for instruction in item.ir.instructions)
    return instructions, mapping


def mine_runtime_graph_contrast(model, item, *, weight=1., solve_time_limit_s=20., learn_arguments=False, learn_operation_pointer=False):
    """Interpret without annotations, then independently compare to the source target."""
    from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError

    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    record = {"source_text_sha256": item.ir.source_text_sha256,
              "serving_authority": False, "negative_origin": "runtime_decode"}
    if outcome.ir is None:
        return None, {**record, "status": "runtime_decode_unavailable", "reason": outcome.refusal}
    try:
        target_instructions, input_mapping = align_source_input_registers(item, outcome.ir.input_spans)
    except ValueError as exc:
        return None, {**record, "status": "input_grounding_unaligned", "reason": str(exc)}
    record["source_to_runtime_input_registers"] = list(input_mapping)
    from types import SimpleNamespace

    def program(instructions):
        return argument_graph_program(tuple(SimpleNamespace(operation=ins.op, span=ins.operation_span)
            for ins in instructions), tuple(ins.args for ins in instructions), n_inputs=len(item.public_inputs))

    comparison = compare_program_meanings(program(target_instructions), program(outcome.ir.instructions),
                                         counterfactual_inputs(item.public_inputs))
    record["comparison"] = comparison
    if comparison["status"] != "different":
        return None, {**record, "status": comparison["status"]}
    try:
        negative = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans,
                                        solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments,
                                        learn_operation_pointer=learn_operation_pointer)
        positive = score_annotated_graph(model, item, target_instructions, outcome.ir.input_spans,
                                        solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments,
                                        learn_operation_pointer=learn_operation_pointer)
    except ArgumentOptimizationIncompleteError as exc:
        return None, {**record, "status": "graph_score_incomplete", "reason": str(exc)}
    if positive is None or negative is None:
        return None, {**record, "status": "target_unreachable" if positive is None else "runtime_replay_unavailable"}
    if not math.isclose(negative["argument_score"], outcome.pointer_scores["argument_graph_total"], abs_tol=1e-4, rel_tol=1e-6):
        return None, {**record, "status": "runtime_score_replay_differs",
                      "runtime_score": outcome.pointer_scores["argument_graph_total"],
                      "replayed_score": negative["argument_score"]}
    record.update(status="counterexample", initial_margin=selection_score_margin(model, positive, negative),
                  selection_policy=model.training_receipt.get("operation_assignment_policy", "first_feasible_v1"),
                  positive_selection_key=graph_selection_key(model, positive),
                  negative_selection_key=graph_selection_key(model, negative),
                  positive_program_sha256=positive["program"].sha(), negative_program_sha256=negative["program"].sha())
    return selection_graph_contrast(model, positive, negative, weight=weight), record


def source_operation_supervision(model, training):
    """Preserve all source operation labels, not only the currently wrong graphs."""
    from core.learning.semantic_operation_background import operation_background_training_spans
    from core.learning.semantic_program_shared_transducer import _geometry
    from core.learning.semantic_program_transducer_fitting import _OperationNode

    if not training or any(item.split != 'train' for item in training):
        raise ValueError('operation retention requires source training examples only')
    counts = Counter(_geometry(item) for item in training)
    evidence, weights = [], []
    for item in training:
        spans = (operation_background_training_spans(item, model.operation_pointer, model.max_span_tokens)
                 if model.training_receipt.get("operation_background_fit") else
                 tuple((ins.operation_span, ins.op) for ins in item.ir.instructions))
        nodes = tuple(_OperationNode(span, label, 0., 0., 1.) for span, label in spans)
        rows = operation_graph_evidence(model, item.hidden_states, nodes)
        evidence.extend(rows)
        weights.extend([1. / (counts[_geometry(item)] * len(rows))] * len(rows))
    return OperationSourceSupervision(tuple(np.stack([bank.features[index] for bank, _ in evidence])
        for index in range(len(model.operation_head.modes))),
        np.array([label for _, label in evidence]), np.array(weights))


def mine_source_binding_constraint(model, item, *, weight=1., max_graphs=32, solve_time_limit_s=20., learn_arguments=False):
    """Keep a witnessed binding competitor even when the source already decodes correctly."""
    from core.learning.semantic_graph_counterexamples import find_graph_counterexample
    from core.learning.semantic_program_transducer_fitting import (
        _assign_typed_arguments,
        _OperationNode,
    )
    from core.learning.semantic_relation_graph_learning import contrast_from_search

    if item.split != "train":
        raise ValueError("semantic constraint mining requires source training examples")
    nodes = tuple(_OperationNode(ins.operation_span, ins.op, 0., 0., 1.) for ins in item.ir.instructions)
    charts = []
    _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
        input_spans=item.ir.input_spans, operation_nodes=nodes,
        argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
        chart_observer=charts.append, retain_score_factors=True, retain_relation_evidence=True, build_only=True)
    record = {"source_text_sha256": item.ir.source_text_sha256, "status": "chart_unavailable",
              "source_operations_supplied": True, "serving_authority": False}
    if not charts:
        return None, record
    result = find_graph_counterexample(charts[0], nodes, tuple(ins.args for ins in item.ir.instructions),
        probes=counterfactual_inputs(item.public_inputs), max_graphs=max_graphs,
        solve_time_limit_s=solve_time_limit_s)
    record.update(result.receipt)
    if result.negative is None:
        return None, record
    contrast = contrast_from_search(result, model.definition_relation_head,
                                   scale=model.definition_relation_scale, weight=weight)
    if learn_arguments:
        from core.learning.semantic_argument_graph_learning import (
            argument_graph_evidence,
            argument_parameters,
        )
        parameters = (model.definition_relation_head.query_projection,
                      model.definition_relation_head.definition_projection,
                      *(v for h in model.operation_head.heads for v in (h.weight, h.bias)),
                      *argument_parameters(model))
        terms = tuple((sign, term) for sign, graph in ((1., result.positive), (-1., result.negative))
                      for term in argument_graph_evidence(model, item.hidden_states, nodes, graph[0][2]))
        contrast = replace(contrast, argument_terms=terms, fixed_margin=contrast.fixed_margin
                           - sum(sign * term.score_gradient(parameters)[0] for sign, term in terms))
    record["initial_margin"] = result.positive[0][0] - result.negative[0][0]
    return contrast, record


def source_operation_constraints(model, supervision, *, weight=1.):
    """Retain every competing label, not just the currently second-ranked label."""
    constraints = []
    for row, label in enumerate(supervision.labels):
        bank = OperationEvidenceBank(tuple(view[row] for view in supervision.features))
        for alternative in range(len(model.operation_head.labels)):
            if alternative != label:
                constraints.append(RelationGraphContrast((), (), 0., weight * supervision.weights[row],
                    ((bank, int(label)),), ((bank, alternative),)))
    return constraints


def source_operation_pointer_constraints(model, training, *, weight=1., required_margin=.1,
                                         policy="supervised"):
    """Retain source operation boundaries against the pointer's hard negatives."""
    from types import SimpleNamespace

    from core.learning.semantic_operation_pointer_learning import (
        operation_pointer_graph_evidence,
        operation_pointer_parameters,
    )
    from core.learning.semantic_paired_pointer_refit import paired_boundary_training_spans

    if not training or any(item.split != "train" for item in training):
        raise ValueError("operation pointer retention requires source training examples")
    if (not math.isfinite(weight) or weight <= 0 or not math.isfinite(required_margin) or required_margin <= 0
            or policy not in {"supervised", "retain_existing"}):
        raise ValueError("operation pointer retention configuration is invalid")
    constraints = []
    pointer_weight, pointer_bias = operation_pointer_parameters(model)
    for item in training:
        positives = tuple(instruction.operation_span for instruction in item.ir.instructions)
        pairs = paired_boundary_training_spans(
            item, positives, model.operation_pointer, model.max_span_tokens,
        )
        negatives = tuple(span for span, label in pairs if label == 0)
        for positive in positives:
            positive_term = operation_pointer_graph_evidence(
                model, item.hidden_states, (SimpleNamespace(span=positive),),
            )[0]
            for negative in negatives:
                negative_term = operation_pointer_graph_evidence(
                    model, item.hidden_states, (SimpleNamespace(span=negative),),
                )[0]
                floor = 0.
                if policy == "retain_existing":
                    # Local annotations may choose a narrower span than a
                    # semantically equivalent runtime graph. Preserve its
                    # current evidence without creating an auxiliary deficit.
                    difference = (float(positive_term.feature @ pointer_weight + pointer_bias)
                                  - float(negative_term.feature @ pointer_weight + pointer_bias))
                    floor = min(difference, 0.)
                constraints.append(RelationGraphContrast(
                    (), (), required_margin - floor, weight,
                    argument_terms=((1., positive_term), (-1., negative_term)),
                ))
    if not constraints:
        raise ValueError("operation pointer retention found no boundary competitors")
    return tuple(constraints)


def refit_compositional_joint_graphs(model, examples, *, rounds=3, steps=100,
                                    solve_time_limit_s=20., progress=None, source_weight=1.,
                                    constraint_learning=False, learn_arguments=False,
                                    checkpoint_dir=None, retention_operation_charts=32,
                                    learn_operation_pointer=False, update_rule="working_face",
                                    boundary_policy="supervised", learn_operations=True,
                                    relation_metric="coefficient_euclidean",
                                    source_retention_examples=None):
    """Remine source-training predictions after each joint operation/relation update."""
    from core.learning.semantic_graph_margin import graph_refit_source_splits
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_shared_transducer import _geometry
    from core.learning.semantic_program_transducer import OPERATION_BACKGROUND_LABEL

    if type(rounds) is not int or rounds < 1 or type(constraint_learning) is not bool:
        raise ValueError("joint graph learning rounds must be positive")
    if type(learn_operations) is not bool or (not learn_operations and not constraint_learning):
        raise ValueError("frozen operation learning requires retained constraints")
    if relation_metric not in {"coefficient_euclidean", "factor_function"} or (
            relation_metric != "coefficient_euclidean" and not constraint_learning):
        raise ValueError("functional relation geometry requires retained constraints")
    if update_rule not in {"working_face", "minimum_change"} or (
            update_rule != "working_face" and not constraint_learning):
        raise ValueError("minimum-change graph learning requires retained constraints")
    if boundary_policy not in {"supervised", "retain_existing"}:
        raise ValueError("unknown graph boundary policy")
    if type(learn_arguments) is not bool or (learn_arguments and not constraint_learning):
        raise ValueError("argument graph learning requires retained constraints")
    if type(learn_operation_pointer) is not bool or (learn_operation_pointer and not constraint_learning):
        raise ValueError("operation pointer learning requires retained constraints")
    if checkpoint_dir is not None and not constraint_learning:
        raise ValueError("fit checkpoints require retained semantic constraints")
    if type(retention_operation_charts) is not int or retention_operation_charts < 1:
        raise ValueError("runtime retention chart allowance must be positive")
    training, validation = graph_refit_source_splits(model, examples)
    retention = training
    if source_retention_examples is not None:
        from core.learning.semantic_validation_checkpoint import validation_identity

        retention = tuple(source_retention_examples)
        if not retention or any(item.split != "train" for item in retention):
            raise ValueError("source retention requires training examples only")
        retention, _ = graph_refit_source_splits(model, (*retention, *validation))
        by_source = {item.ir.source_text_sha256: item for item in retention}
        if not all(item.ir.source_text_sha256 in by_source for item in training):
            raise ValueError("source retention must include the mining cohort")
        bound = tuple(by_source[item.ir.source_text_sha256] for item in training)
        def identity(rows):
            return validation_identity({"parent": model}, rows,
                scoring="source_anchors_v2", implementation="source-retention-input-v1")
        if identity(training) != identity(bound):
            raise ValueError("source retention changed a mining observation")
    if not np.isfinite(source_weight) or source_weight < 0:
        raise ValueError('invalid source operation retention weight')
    if (model.training_receipt.get("definition_selection_policy") != "joint_graph_v1"
            or model.training_receipt.get("relation_score_strategy") != "categorical_log_margin_v1"
            or model.training_receipt.get("operation_assignment_policy", "first_feasible_v1")
            not in {"first_feasible_v1", "joint_factor_score_v2"}):
        raise ValueError("graph training requires a supported categorical runtime decoder")
    weights = Counter(_geometry(item) for item in training)
    supervision = source_operation_supervision(model, retention) if source_weight else None
    candidate, retained, history = model, [], []
    if constraint_learning and supervision is not None:
        retained.extend(source_operation_constraints(model, supervision, weight=source_weight))
        if learn_operation_pointer:
            retained.extend(source_operation_pointer_constraints(
                model, retention, weight=source_weight, policy=boundary_policy,
            ))
    stop_reason = "round_budget_exhausted"
    for round_index in range(rounds):
        coefficients_before = _sha(candidate._coefficient_body())
        records, new_pairs = [], 0
        for index, item in enumerate(training):
            contrast, record = mine_runtime_graph_contrast(candidate, item,
                weight=1. / weights[_geometry(item)], solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments,
                learn_operation_pointer=learn_operation_pointer)
            if contrast is not None:
                record["retained_pair"] = len(retained)
                retained.append(contrast)
                new_pairs += 1
            if constraint_learning:
                from core.learning.semantic_runtime_graph_retention import (
                    mine_runtime_graph_constraints,
                )

                competitors, competitor_record = mine_runtime_graph_constraints(candidate, item,
                    weight=1. / weights[_geometry(item)], max_charts=retention_operation_charts,
                    solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments,
                    learn_operation_pointer=learn_operation_pointer)
                record["runtime_constraints"] = competitor_record
                competitor_record["retained_pairs"] = list(range(len(retained), len(retained) + len(competitors)))
                retained.extend(competitors)
                new_pairs += len(competitors)
                binding, binding_record = mine_source_binding_constraint(candidate, item,
                    weight=1. / weights[_geometry(item)], solve_time_limit_s=solve_time_limit_s, learn_arguments=learn_arguments)
                record["binding_constraint"] = binding_record
                if binding is not None:
                    binding_record["retained_pair"] = len(retained)
                    retained.append(binding)
                    new_pairs += 1
            records.append(record)
            if progress:
                progress({"stage": "joint_graph_mining", "round": round_index + 1,
                          "completed": index + 1, "total": len(training), "row": record})
        if not new_pairs and not (constraint_learning and retained):
            stop_reason = "no_new_witnessed_errors"
            history.append({"records": records, "fit": {"status": "no_new_witnessed_errors",
                            "pairs": len(retained), "coverage_complete": all(
                                row["status"] == "equivalent" for row in records)}})
            break
        fit_options = {}
        if constraint_learning:
            fit_options["update_rule"] = update_rule
            fit_options["learn_operations"] = learn_operations
            fit_options["relation_metric"] = relation_metric
            fit_options["progress"] = (lambda row, iteration=round_index + 1:
                                       progress({**row, "round": iteration})) if progress else None
            if checkpoint_dir is not None:
                fit_options.update(
                    checkpoint_path=Path(checkpoint_dir) / f"round-{round_index + 1}.npz",
                    checkpoint_identity={"parent": model.receipt_sha256, "round": round_index + 1,
                                         "training": _sha(sorted(item.ir.source_text_sha256 for item in training))},
                )
        if learn_arguments or learn_operation_pointer:
            from core.learning.semantic_graph_constraints import fit_complete_graph_constraints
            candidate, fit = fit_complete_graph_constraints(candidate, tuple(retained),
                scale=candidate.definition_relation_scale, steps=steps, adaptive_step=True,
                learn_operation_pointer=learn_operation_pointer, **fit_options)
        elif constraint_learning:
            from core.learning.semantic_graph_constraints import fit_graph_constraints
            relation, operation, fit = fit_graph_constraints(candidate.definition_relation_head,
                candidate.operation_head, tuple(retained), scale=candidate.definition_relation_scale,
                steps=steps, **fit_options)
        else:
            relation, operation, fit = fit_joint_graph_contrasts(candidate.definition_relation_head,
                candidate.operation_head, tuple(retained), scale=candidate.definition_relation_scale, steps=steps,
                source_supervision=supervision, source_weight=source_weight)
        if not (learn_arguments or learn_operation_pointer):
            candidate = candidate._with_coefficients(definition_relation_head=relation, operation_head=operation)
        history.append({"records": records, "fit": fit})
        if progress:
            progress({"stage": "joint_graph_fit", "round": round_index + 1, "fit": fit})
        if _sha(candidate._coefficient_body()) == coefficients_before:
            stop_reason = "coefficients_unchanged"
            break
    body = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
    body["joint_graph_refit"] = {
        "schema": "aura.semantic_joint_graph_refit.v1", "parent_transducer_receipt_sha256": model.receipt_sha256,
        "training_examples": len(training), "validation_examples": len(validation),
        "source_retention_examples": len(retention),
        "source_retention_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in retention)),
        "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in training)),
        "validation_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in validation)),
        "rounds": history, "requested_rounds": rounds, "completed_rounds": len(history),
        "stop_reason": stop_reason,
        "negative_origin": "runtime_decode", "positive_origin": "source_annotations",
        "negative_admission": "universal_floor_distinguishing_execution", "test_examples_used": 0,
        "selection_policy": model.training_receipt.get("operation_assignment_policy", "first_feasible_v1"),
        "selection_objective": "runtime_policy_aligned_v1",
        "validation_used_for_fit": False, "serving_authority": False,
        "source_operation_weight": source_weight,
        "source_operations": sum(model.operation_head.labels[index] != OPERATION_BACKGROUND_LABEL
                                 for index in supervision.labels) if supervision is not None else 0,
        "source_training_spans": len(supervision.labels) if supervision is not None else 0,
        "source_supervision_includes_background": bool(
            supervision is not None and model.training_receipt.get("operation_background_fit")),
        "constraint_learning": constraint_learning,
        "argument_heads_trainable": learn_arguments,
        "operation_pointer_trainable": learn_operation_pointer,
        "operation_head_trainable": learn_operations,
        "update_rule": update_rule,
        "boundary_policy": boundary_policy,
        "relation_metric": relation_metric,
        "already_correct_binding_competitors_retained": constraint_learning,
        "runtime_operation_competitors_retained": constraint_learning,
        "retention_operation_charts": retention_operation_charts if constraint_learning else None,
        "input_coordinate_policy": "source_anchor_value_preserving_permutation_v1",
    }
    return replace(candidate, training_receipt={**body, "receipt_sha256": _sha(body)})
