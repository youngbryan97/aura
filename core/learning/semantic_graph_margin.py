"""Calibrate existing score factors against complete source-supervised graphs.

Operation identities are supplied by source annotations. Targets constrain
register bindings, while mention and definition choices remain latent. This
does not train operation recognition or grant runtime authority.
"""

from collections import Counter
from dataclasses import replace

import numpy as np


def _graph_margin_loss(scales, differences, offsets, weights, initial, regularization):
    from scipy.special import expit

    margins = differences @ scales + offsets
    normalized = weights / weights.sum()
    delta = scales - initial
    return (
        float(normalized @ np.logaddexp(0., -margins) + .5 * regularization * (delta @ delta)),
        -(normalized * expit(-margins)) @ differences + regularization * delta,
    )


def fit_graph_score_scales(differences, offsets, weights, initial, *, regularization=.01):
    """Fit positive role/relation/pointer scales; the proposal scale stays fixed."""
    from scipy.optimize import minimize

    differences, offsets, weights, initial = (
        np.asarray(value, dtype=np.float64) for value in (differences, offsets, weights, initial)
    )
    if (
        differences.ndim != 2 or differences.shape[1] != 3 or not len(differences)
        or offsets.shape != (len(differences),) or weights.shape != offsets.shape
        or initial.shape != (3,) or np.any(initial <= 0) or np.any(weights <= 0)
        or not all(np.all(np.isfinite(value)) for value in (differences, offsets, weights, initial))
        or not np.isfinite(regularization) or regularization <= 0
    ):
        raise ValueError("invalid graph margin supervision")
    objective = lambda value: _graph_margin_loss(  # noqa: E731
        value, differences, offsets, weights, initial, regularization,
    )
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(1e-6, None)] * 3, options={"maxiter": 400, "ftol": 1e-12})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"graph scale optimization incomplete: {result.message}")
    return result.x, {
        "objective": "source_complete_graph_pairwise_logistic_v1",
        "pairs": len(differences), "iterations": int(result.nit),
        "initial_loss": objective(initial)[0], "fitted_loss": objective(result.x)[0],
        "initial_scales": initial.tolist(), "fitted_scales": result.x.tolist(),
        "regularization": regularization, "converged": True,
        "proposal_scale_held_fixed": True,
    }


def refit_compositional_graph_scales(model, examples, *, progress=None):
    """Fit source graph contrasts, with explicit coverage and unchanged heads."""
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_shared_transducer import _geometry
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode
    from core.learning.semantic_graph_counterexamples import counterfactual_inputs, find_graph_counterexample

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    selected = (*training, *validation)
    if not training or not validation or (
        {item.ir.model_basis_receipt_sha256 for item in selected} != {model.model_basis_sha256}
        or {item.tokenizer_identity_sha256 for item in selected} != {model.input_grounding.tokenizer_identity_sha256}
        or {(item.hidden_channels, item.hidden_channel_widths) for item in selected}
        != {(model.hidden_channels, model.hidden_channel_widths)}
        or model.training_receipt.get("argument_search_strategy") != "global_constraint_v1"
    ):
        raise ValueError("graph refit needs compatible source splits and global argument search")
    ids = [item.ir.source_text_sha256 for item in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("graph refit source splits duplicate or overlap")
    scales = np.array([model.argument_role_scale, model.definition_relation_scale, model.argument_pointer_scale])
    all_scales = np.array([scales[0], model.argument_proposal_scale, scales[1], scales[2]])
    geometry_counts = Counter(_geometry(item) for item in training)
    differences, offsets, weights, records = [], [], [], []
    coverage = Counter()
    for item in training:
        captured = []
        nodes = tuple(_OperationNode(instruction.operation_span, instruction.op, 0., 0., 1.)
                      for instruction in item.ir.instructions)
        _assign_typed_arguments(
            model=model, hidden=item.hidden_states, inputs=item.public_inputs, input_spans=item.ir.input_spans,
            operation_nodes=nodes,
            argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
            chart_observer=captured.append, retain_score_factors=True, build_only=True,
        )
        target = tuple(instruction.args for instruction in item.ir.instructions)
        positive = negative = None
        semantic_search = None
        if not captured:
            status = "chart_unavailable"
        else:
            result = find_graph_counterexample(captured[0], nodes, target,
                probes=counterfactual_inputs(item.public_inputs), max_graphs=128)
            positive, negative, semantic_search = result.positive, result.negative, result.receipt
            status = "contrast" if negative is not None else semantic_search["status"]
        record = {"source_text_sha256": item.ir.source_text_sha256, "status": status}
        if semantic_search is not None:
            record["semantic_search"] = semantic_search
        if positive is not None and negative is not None:
            factor_difference = np.asarray(positive[1]) - np.asarray(negative[1])
            margin = positive[0][0] - negative[0][0]
            fixed = margin - factor_difference @ all_scales
            differences.append(factor_difference[[0, 2, 3]])
            offsets.append(float(fixed + factor_difference[1] * model.argument_proposal_scale))
            weights.append(1. / geometry_counts[_geometry(item)])
            record.update(target_score=positive[0][0], alternative_score=negative[0][0],
                          factor_difference=factor_difference.tolist(), fixed_difference=float(fixed))
        coverage[status] += 1
        records.append(record)
        if progress is not None:
            progress({"stage": "source_graph_contrasts", "completed": len(records),
                      "total": len(training), "coverage": dict(coverage), "row": record})
    if differences:
        fitted, fit = fit_graph_score_scales(differences, offsets, weights, scales)
    else:
        fitted, fit = scales, {"pairs": 0, "status": "no_witnessed_training_errors", "converged": False}
    candidate = model._with_coefficients(argument_role_scale=float(fitted[0]),
        definition_relation_scale=float(fitted[1]), argument_pointer_scale=float(fitted[2]))
    body = {key: value for key, value in candidate.training_receipt.items() if key != "receipt_sha256"}
    body["argument_graph_factor_refit"] = {
        "schema": "aura.semantic_graph_factor_refit.v2",
        "negative_admission": "universal_floor_distinguishing_execution",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in training)),
        "validation_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in validation)),
        "training_examples": len(training), "validation_examples": len(validation),
        "operation_boundaries": "source_annotations_v1", "neural_heads_changed": False,
        "candidate_pool": "parent_runtime_retained_options_v1", "coverage": dict(coverage),
        "contrast_rows_sha256": _sha(records), "fit": fit,
        "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
    }
    return replace(candidate, training_receipt={**body, "receipt_sha256": _sha(body)})
