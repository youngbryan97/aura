"""Differentiate complete-graph contrasts through the existing relation tissue."""

from dataclasses import dataclass, replace

import numpy as np
from core.verify.invariants import invariant


@dataclass(frozen=True)
class RelationEvidenceBank:
    reference: np.ndarray
    definitions: np.ndarray
    base_logits: np.ndarray

    def __post_init__(self):
        reference = np.asarray(self.reference, dtype=np.float32)
        definitions = np.asarray(self.definitions, dtype=np.float32)
        base = np.asarray(self.base_logits, dtype=np.float64)
        if (reference.ndim != 1 or definitions.ndim != 2 or definitions.shape[1] != reference.size
                or base.shape != (len(definitions),) or not len(base)
                or not all(np.all(np.isfinite(v)) for v in (reference, definitions, base))):
            raise ValueError("relation evidence geometry differs")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "base_logits", base)

    def score(self, selected, query, definition):
        if type(selected) is not int or not 0 <= selected < len(self.base_logits):
            raise ValueError("selected relation hypothesis is invalid")
        logits = self.base_logits + (self.definitions @ definition) @ (self.reference @ query)
        return float(np.max(-np.logaddexp(0., -self.base_logits)) + logits[selected] - logits.max())

    def score_gradient(self, selected, query, definition):
        """Use the runtime categorical margin, including its moving maximum."""
        if type(selected) is not int or not 0 <= selected < len(self.base_logits):
            raise ValueError("selected relation hypothesis is invalid")
        q = self.reference @ query
        d = self.definitions @ definition
        logits = self.base_logits + d @ q
        peak = int(np.argmax(logits))
        value = float(np.max(-np.logaddexp(0., -self.base_logits)) + logits[selected] - logits[peak])
        delta = self.definitions[selected].astype(np.float64) - self.definitions[peak].astype(np.float64)
        return value, np.outer(self.reference, delta @ definition), np.outer(delta, q)


@dataclass(frozen=True)
class RelationGraphContrast:
    positive: tuple
    negative: tuple
    fixed_margin: float
    weight: float = 1.
    positive_operations: tuple = ()
    negative_operations: tuple = ()
    argument_terms: tuple = ()


def contrast_from_search(result, head, *, scale, weight=1.):
    """Keep the exact decoder margin while exposing its relation contribution."""
    if result.negative is None or not result.positive_evidence or not result.negative_evidence:
        raise ValueError("relation contrast needs witnessed selected graph evidence")
    projections = (head.query_projection.astype(np.float64), head.definition_projection.astype(np.float64))
    relation_margin = sum(bank.score_gradient(choice, *projections)[0]
                          for bank, choice in result.positive_evidence) - sum(
        bank.score_gradient(choice, *projections)[0] for bank, choice in result.negative_evidence)
    margin = result.positive[0][0] - result.negative[0][0]
    return RelationGraphContrast(result.positive_evidence, result.negative_evidence,
                                 margin - scale * relation_margin, weight)


def graph_margin_gradient(parameters, row, *, scale=1.):
    """Replay one complete-interpretation margin independently of an aggregate loss."""
    query, definition, *operations = parameters
    margin = row.fixed_margin
    gradients = [np.zeros_like(value) for value in parameters]
    for sign, choices in ((1., row.positive), (-1., row.negative)):
        for bank, index in choices:
            value, qgrad, dgrad = bank.score_gradient(index, query, definition)
            margin += sign * scale * value
            gradients[0] += sign * scale * qgrad
            gradients[1] += sign * scale * dgrad
    for sign, choices in ((1., row.positive_operations), (-1., row.negative_operations)):
        for bank, index in choices:
            value, derivatives = bank.score_gradient(index, operations[:2 * len(bank.features)])
            margin += sign * value
            for gradient, derivative in zip(gradients[2:2 + len(derivatives)], derivatives, strict=True):
                gradient += sign * derivative
    for sign, term in row.argument_terms:
        value, weight, bias = term.score_gradient(parameters)
        margin += sign * value
        gradients[term.parameter_index] += sign * weight
        gradients[term.parameter_index + 1] += sign * bias
    return float(margin), tuple(gradients)


def graph_margin(parameters, row, *, scale=1.):
    """Value-only replay avoids allocating one parameter gradient per witness."""
    query, definition, *operations = parameters
    return float(row.fixed_margin + sum(sign * (
        scale * sum(bank.score(index, query, definition) for bank, index in relations)
        + sum(bank.score(index, operations[:2 * len(bank.features)]) for bank, index in op_choices))
        for sign, relations, op_choices in (
            (1., row.positive, row.positive_operations), (-1., row.negative, row.negative_operations)))
        + sum(sign * term.score(parameters) for sign, term in row.argument_terms))


def relation_graph_loss(query, definition, contrasts, *, scale, initial, regularization,
                        operation_parameters=(), source_supervision=None, source_weight=0.):
    """Pairwise logistic loss on graph margins, with latent choices held fixed."""
    from scipy.special import expit

    total = sum(row.weight for row in contrasts)
    if (not contrasts or not np.isfinite(total) or total <= 0 or scale <= 0
            or regularization < 0 or not np.isfinite(scale + regularization)
            or any(not np.isfinite(row.fixed_margin) or not np.isfinite(row.weight)
                   or row.weight <= 0 for row in contrasts)):
        raise ValueError("invalid relation graph contrasts")
    parameters = (query, definition, *operation_parameters)
    if (not np.isfinite(source_weight) or source_weight < 0
            or (source_weight > 0 and source_supervision is None)):
        raise ValueError("invalid operation source retention")
    if len(initial) != len(parameters):
        raise ValueError("graph optimizer anchor geometry differs")
    loss, gradients = 0., [np.zeros_like(value) for value in parameters]
    for row in contrasts:
        margin, derivatives = graph_margin_gradient(parameters, row, scale=scale)
        weight = row.weight / total
        loss += weight * np.logaddexp(0., -margin)
        for gradient, derivative in zip(gradients, derivatives, strict=True):
            gradient -= weight * expit(-margin) * derivative
    if source_weight:
        source_loss, source_gradients = source_supervision.loss_gradient(operation_parameters)
        loss += source_weight * source_loss
        for gradient, derivative in zip(gradients[2:], source_gradients, strict=True):
            gradient += source_weight * derivative
    for value, start, gradient in zip(parameters, initial, gradients, strict=True):
        delta = value - start
        loss += .5 * regularization * np.sum(delta * delta)
        gradient += regularization * delta
    return float(loss), tuple(gradients)


def fit_relation_graph_contrasts(head, contrasts, *, scale=1., steps=100, learning_rate=.001,
                                regularization=.001):
    """Refit the shipped low-rank projections; base evidence stays unchanged."""
    candidate, _, receipt = fit_joint_graph_contrasts(head, None, contrasts, scale=scale,
        steps=steps, learning_rate=learning_rate, regularization=regularization)
    return candidate, receipt


def fit_joint_graph_contrasts(head, operation_head, contrasts, *, scale=1., steps=100,
                             learning_rate=.001, regularization=.001,
                             source_supervision=None, source_weight=0.):
    """Update existing relation and optional operation heads under one graph loss."""
    if type(steps) is not int or steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid relation graph optimizer settings")
    operation_parameters = tuple(v for component in operation_head.heads
                                 for v in (component.weight, component.bias)) if operation_head is not None else ()
    initial = tuple(np.asarray(v, dtype=np.float64).copy() for v in
                    (head.query_projection, head.definition_projection, *operation_parameters))
    values = [v.copy() for v in initial]
    moments, squares = ([np.zeros_like(v) for v in values] for _ in range(2))
    kwargs = dict(scale=scale, initial=initial, regularization=regularization,
                  source_supervision=source_supervision, source_weight=source_weight)
    def objective(parameters):
        return relation_graph_loss(*parameters[:2], contrasts, operation_parameters=parameters[2:], **kwargs)

    initial_loss = objective(values)[0]
    best_loss, best = initial_loss, [v.copy() for v in values]
    for step in range(1, steps + 1):
        _, gradients = objective(values)
        for index, gradient in enumerate(gradients):
            moments[index] = .9 * moments[index] + .1 * gradient
            squares[index] = .999 * squares[index] + .001 * gradient * gradient
            values[index] -= learning_rate * (moments[index] / (1 - .9 ** step)) / (
                np.sqrt(squares[index] / (1 - .999 ** step)) + 1e-8)
        loss = objective(values)[0]
        if not np.isfinite(loss):
            raise ValueError("nonfinite relation graph update")
        if loss < best_loss:
            best_loss, best = loss, [v.copy() for v in values]
    candidate = replace(head, query_projection=best[0], definition_projection=best[1])
    operations = (replace(operation_head, heads=tuple(replace(component, weight=best[2 + 2 * index],
        bias=best[3 + 2 * index]) for index, component in enumerate(operation_head.heads)))
        if operation_head is not None else None)
    stored = (candidate.query_projection, candidate.definition_projection,
              *(v for component in operations.heads for v in (component.weight, component.bias))) if operations else (
                  candidate.query_projection, candidate.definition_projection)
    stored_loss = objective(tuple(value.astype(np.float64) for value in stored))[0]
    if stored_loss > initial_loss:
        candidate, operations, stored_loss = head, operation_head, initial_loss
    return candidate, operations, {"objective": ("witnessed_joint_graph_with_source_operations_v2" if source_weight else
        "witnessed_complete_joint_graph_margin_v1" if operation_head is not None
        else "witnessed_complete_graph_relation_margin_v1"), "pairs": len(contrasts),
        "steps": steps, "initial_loss": initial_loss, "stored_loss": stored_loss,
        "learning_rate": learning_rate, "regularization": regularization, "relation_scale": scale,
        "operation_head_updated": operation_head is not None,
        "source_operation_weight": source_weight,
        "source_operations": len(source_supervision.labels) if source_supervision is not None else 0,
        "base_head_changed": False, "latent_choices_frozen_for_update": True,
        "serving_authority": False}


def refit_compositional_graph_relations(model, examples, *, rounds=3, steps=100,
                                       max_graphs=32, solve_time_limit_s=20., progress=None):
    """Mine witnessed graph errors and refit existing tissue against retained pairs."""
    from collections import Counter
    from core.learning.semantic_graph_margin import graph_refit_source_splits
    from core.learning.semantic_graph_counterexamples import counterfactual_inputs, find_graph_counterexample
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_shared_transducer import _geometry
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode

    if type(rounds) is not int or rounds < 1:
        raise ValueError("graph learning rounds must be positive")
    training, validation = graph_refit_source_splits(model, examples)
    if (model.training_receipt.get("definition_selection_policy") != "joint_graph_v1"
            or model.training_receipt.get("relation_score_strategy") != "categorical_log_margin_v1"):
        raise ValueError("graph relation refit requires joint categorical definitions")
    weights = Counter(_geometry(item) for item in training)
    candidate, retained, history = model, [], []
    for round_index in range(rounds):
        records = []
        for index, item in enumerate(training):
            charts = []
            nodes = tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions)
            _assign_typed_arguments(model=candidate, hidden=item.hidden_states, inputs=item.public_inputs,
                input_spans=item.ir.input_spans, operation_nodes=nodes,
                argument_pointer_scores=candidate.argument_pointer.score_sequence(item.hidden_states),
                chart_observer=charts.append, retain_score_factors=True,
                retain_relation_evidence=True, build_only=True)
            record = {"source_text_sha256": item.ir.source_text_sha256, "status": "chart_unavailable"}
            if charts:
                result = find_graph_counterexample(charts[0], nodes, tuple(i.args for i in item.ir.instructions),
                    probes=counterfactual_inputs(item.public_inputs), max_graphs=max_graphs,
                    solve_time_limit_s=solve_time_limit_s)
                record.update(result.receipt)
                if result.negative is not None:
                    retained.append(contrast_from_search(result, candidate.definition_relation_head,
                        scale=candidate.definition_relation_scale, weight=1. / weights[_geometry(item)]))
                    record["retained_pair"] = len(retained) - 1
            records.append(record)
            if progress:
                progress({"stage": "graph_relation_mining", "round": round_index + 1,
                          "completed": index + 1, "total": len(training), "row": record})
        if not retained:
            history.append({"records": records, "fit": {"status": "no_witnessed_training_errors", "pairs": 0}})
            break
        head, fit = fit_relation_graph_contrasts(candidate.definition_relation_head, tuple(retained),
                                               scale=candidate.definition_relation_scale, steps=steps)
        candidate = candidate._with_coefficients(definition_relation_head=head)
        history.append({"records": records, "fit": fit})
        if progress:
            progress({"stage": "graph_relation_fit", "round": round_index + 1, "fit": fit})
    body = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
    body["argument_graph_relation_refit"] = {
        "schema": "aura.semantic_graph_relation_refit.v1", "parent_transducer_receipt_sha256": model.receipt_sha256,
        "training_examples": len(training), "validation_examples": len(validation),
        "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in training)),
        "validation_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in validation)),
        "rounds": history, "operation_boundaries": "source_annotations_v1",
        "negative_admission": "universal_floor_distinguishing_execution", "test_examples_used": 0,
        "validation_used_for_fit": False, "serving_authority": False,
    }
    return replace(candidate, training_receipt={**body, "receipt_sha256": _sha(body)})


@invariant("learning.graph_relation_gradient_includes_competing_definition", scope="learning",
           owner="core/learning/semantic_relation_graph_learning.py", observational=False)
def _graph_relation_normalizer_gradient() -> tuple:
    bank = RelationEvidenceBank(np.array([1., 0.]), np.eye(2), np.zeros(2))
    _, query_gradient, definition_gradient = bank.score_gradient(
        0, np.array([[1.], [0.]]), np.array([[0.], [1.]]))
    assert np.array_equal(query_gradient, [[-1.], [0.]])
    assert np.array_equal(definition_gradient, [[1.], [-1.]])
    return ()
