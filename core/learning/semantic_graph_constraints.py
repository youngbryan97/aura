"""Fit witnessed graph inequalities without sacrificing satisfied witnesses.

This is a numerical constrained search, not an infeasibility prover. Every
accepted update is checked against the nonlinear margins after float32 storage.
The obligations concern retained latent interpretations; fresh decoding must
still search for new competitors and changed latent choices.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.learning.semantic_graph_batch import GraphConstraintBatch
from core.learning.semantic_relation_graph_learning import graph_margin, graph_margin_gradient
from core.verify.invariants import invariant


def _project_direction(direction, normals):
    """Project onto the intersection of linearized nondecreasing halfspaces."""
    if not normals:
        return direction
    a = np.stack(normals)
    norms = np.linalg.norm(a, axis=1)
    a = a[norms > 0] / norms[norms > 0, None]
    if not len(a):
        return direction
    gram, linear = a @ a.T, a @ direction
    result = minimize(lambda value: (.5 * value @ gram @ value + linear @ value,
                                     gram @ value + linear), np.zeros(len(a)),
                      jac=True, bounds=[(0., None)] * len(a), method="L-BFGS-B")
    if not result.success or not np.all(np.isfinite(result.x)):
        return np.zeros_like(direction)
    return direction + a.T @ result.x


def _fit_graph_parameters(initial, contrasts, *, scale=1., steps=100,
                          required_margin=.1, learning_rate=.001, max_active=32,
                          adaptive_step=False, checkpoint_path=None, progress=None,
                          checkpoint_identity=None, batched=True):
    """Search for all retained inequalities; retain every already-positive margin."""
    if (type(batched) is not bool or type(adaptive_step) is not bool or not contrasts or type(steps) is not int or steps < 1 or type(max_active) is not int
            or max_active < 1 or not np.isfinite(required_margin) or required_margin <= 0
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(scale) or scale <= 0
            or any(not np.isfinite(row.weight) or row.weight <= 0
                   or not np.isfinite(row.fixed_margin) for row in contrasts)):
        raise ValueError("invalid semantic graph constraint fit")
    initial = tuple(np.asarray(value, dtype=np.float64) for value in initial)
    shapes = tuple(value.shape for value in initial)
    ends = np.cumsum([value.size for value in initial])
    batch = GraphConstraintBatch(contrasts, scale) if batched else None

    def unpack(flat):
        return tuple(value.reshape(shape) for value, shape in
                     zip(np.split(flat, ends[:-1]), shapes, strict=True))

    def evaluate(flat):
        parameters = unpack(flat)
        values = (batch.margins(parameters) if batch is not None else
                  np.asarray([graph_margin(parameters, row, scale=scale) for row in contrasts]))
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite semantic constraint margin")
        return values

    flat = np.concatenate([value.ravel() for value in initial])
    margins = evaluate(flat)
    before = margins.copy()
    # A positive margin means this witnessed error already loses. Its floor
    # never decreases, including after new constraints become satisfied.
    floors = np.where(margins > 0., np.minimum(margins, required_margin), -np.inf)
    weights = np.array([row.weight for row in contrasts])
    weights /= weights.sum()
    trace, status = [], "search_budget_exhausted"
    checkpoint, start_step = None, 0
    if checkpoint_path is not None:
        from core.learning.semantic_fit_checkpoint import SemanticFitCheckpoint, fit_identity

        source_files = ("semantic_graph_constraints.py", "semantic_graph_batch.py", "semantic_relation_graph_learning.py",
                        "semantic_operation_graph_learning.py", "semantic_argument_graph_learning.py")
        identity = fit_identity({
            "algorithm": [Path(__file__).with_name(name).read_text() for name in source_files],
            "owner": checkpoint_identity, "initial": initial, "contrasts": tuple(contrasts),
            "options": (scale, steps, required_margin, learning_rate, max_active, adaptive_step, batched),
        })
        checkpoint = SemanticFitCheckpoint(checkpoint_path, identity)
        saved = checkpoint.load()
        if saved is not None:
            allowed = {"running", "search_budget_exhausted", "retained_constraints_satisfied",
                       "no_feasible_direction_found", "no_retention_preserving_step_found"}
            start_step, status, trace = saved["next_step"], saved["status"], saved["trace"]
            if (type(start_step) is not int or not 0 <= start_step <= steps or status not in allowed
                    or len(trace) != start_step
                    or [row.get("step") for row in trace] != list(range(1, start_step + 1))
                    or saved["flat"].shape != flat.shape or not np.all(np.isfinite(saved["flat"]))
                    or saved["margins"].shape != margins.shape or saved["floors"].shape != floors.shape):
                raise ValueError("fit checkpoint progress or geometry differs")
            verified = evaluate(saved["flat"])
            if (not np.allclose(verified, saved["margins"], atol=1e-12, rtol=1e-12)
                    or np.any(np.isnan(saved["floors"])) or not np.all(saved["floors"] >= floors)
                    or not np.all(verified >= saved["floors"])):
                raise ValueError("fit checkpoint does not retain the witnessed margins")
            flat, margins, floors = saved["flat"], verified, saved["floors"]
            if progress:
                progress({"stage": "constraint_fit_resumed", "completed": start_step,
                          "total": steps, "status": status})
            if status != "running":
                start_step = steps

    def persist(state):
        if checkpoint is not None:
            checkpoint.save(flat=flat, margins=margins, floors=floors, trace=trace,
                            next_step=len(trace), status=state)

    for step in range(start_step, steps):
        deficits = np.maximum(required_margin - margins, 0.)
        if not np.any(deficits):
            status = "retained_constraints_satisfied"
            break
        parameters = unpack(flat)
        direction = (np.concatenate([value.ravel() for value in
                     batch.weighted_gradient(parameters, weights * deficits)])
                     if batch is not None else np.zeros_like(flat))
        normals = []
        active = set(sorted(np.flatnonzero(np.isfinite(floors)),
                            key=lambda index: margins[index] - floors[index])[:max_active])
        for index, row in enumerate(contrasts):
            if index not in active and (batch is not None or not deficits[index]):
                continue
            _margin, gradient = graph_margin_gradient(parameters, row, scale=scale)
            vector = np.concatenate([value.ravel() for value in gradient])
            if batch is None:
                direction += weights[index] * deficits[index] * vector
            if index in active:
                normals.append(vector)
        direction = _project_direction(direction, normals)
        largest = np.max(np.abs(direction))
        if largest == 0 or not np.isfinite(largest):
            status = "no_feasible_direction_found"
            break
        direction /= largest
        step_size = learning_rate
        if adaptive_step:
            # Minimize the linearized squared deficit along the feasible direction.
            slopes = (batch.directional_derivative(parameters, unpack(direction))
                      if batch is not None else np.zeros_like(margins))
            if batch is None:
                for index in np.flatnonzero(deficits):
                    _margin, gradient = graph_margin_gradient(parameters, contrasts[index], scale=scale)
                    slopes[index] = sum(float(np.sum(value * part)) for value, part in
                                        zip(gradient, unpack(direction), strict=True))
            slopes[deficits == 0] = 0.
            denominator = float(weights @ (slopes ** 2))
            numerator = float(weights @ (deficits * slopes))
            if denominator > 0 and numerator > 0:
                step_size = numerator / denominator
        direction *= step_size
        loss = float(weights @ (deficits ** 2))
        accepted = False
        for backtrack in range(24):
            # The exported dtype participates in acceptance, not just a later
            # loss check that could silently erase a small positive margin.
            trial = (flat + direction * 2. ** -backtrack).astype(np.float32).astype(np.float64)
            trial_margins = evaluate(trial)
            trial_loss = float(weights @ np.maximum(required_margin - trial_margins, 0.) ** 2)
            if np.all(trial_margins >= floors) and trial_loss < loss:
                flat, margins = trial, trial_margins
                floors = np.maximum(floors, np.where(margins > 0.,
                    np.minimum(margins, required_margin), -np.inf))
                accepted = True
                trace.append({"step": step + 1, "loss": trial_loss,
                              "wrong_or_tied": int(np.count_nonzero(margins <= 0)),
                              "minimum_margin": float(margins.min()), "backtracks": backtrack,
                              "step_size": step_size * 2. ** -backtrack})
                persist("running")
                if progress:
                    progress({"stage": "constraint_fit_step", "completed": step + 1,
                              "total": steps, **trace[-1]})
                break
        if not accepted:
            status = "no_retention_preserving_step_found"
            break
    if np.all(margins >= required_margin):
        status = "retained_constraints_satisfied"
    elif status == "running":
        status = "search_budget_exhausted"
    persist(status)
    values = unpack(flat)
    return values, {
        "objective": "retained_semantic_inequalities_v1", "status": status,
        "pairs": len(contrasts), "required_margin": required_margin,
        "initial_margins": before.tolist(), "stored_margins": margins.tolist(),
        "initial_wrong_or_tied": int(np.count_nonzero(before <= 0)),
        "stored_wrong_or_tied": int(np.count_nonzero(margins <= 0)),
        "retained_positive_regressions": int(np.count_nonzero((before > 0) & (margins <= 0))),
        "accepted_steps": trace, "max_projected_constraints": max_active,
        "step_policy": "linearized_deficit_backtracking_v1" if adaptive_step else "fixed_max_parameter_step_v1",
        "all_constraints_checked_at_acceptance": True,
        "infeasibility_proven": False, "latent_choices_frozen_for_update": True,
        "serving_authority": False,
    }


def _model_parameters(head, operation_head):
    return (head.query_projection, head.definition_projection,
            *(value for part in operation_head.heads for value in (part.weight, part.bias)))


def _fitted_heads(head, operation_head, values):
    return (replace(head, query_projection=values[0], definition_projection=values[1]),
            replace(operation_head, heads=tuple(replace(part,
                weight=values[2 + 2 * index], bias=values[3 + 2 * index])
                for index, part in enumerate(operation_head.heads))))


def fit_graph_constraints(head, operation_head, contrasts, **options):
    values, receipt = _fit_graph_parameters(_model_parameters(head, operation_head), contrasts, **options)
    return (*_fitted_heads(head, operation_head, values), receipt)


def fit_complete_graph_constraints(model, contrasts, **options):
    from core.learning.semantic_argument_graph_learning import argument_parameters

    base = _model_parameters(model.definition_relation_head, model.operation_head)
    values, receipt = _fit_graph_parameters((*base, *argument_parameters(model)), contrasts, **options)
    relation, operation = _fitted_heads(model.definition_relation_head, model.operation_head, values)
    offset = len(base)
    roles = tuple(replace(head, weight=values[offset + 4 * index],
                          bias=float(values[offset + 4 * index + 1]))
                  for index, head in enumerate(model.argument_role_heads))
    proposals = tuple(replace(head, weight=values[offset + 4 * index + 2],
                              bias=float(values[offset + 4 * index + 3]))
                      for index, head in enumerate(model.argument_proposal_heads))
    return model._with_coefficients(definition_relation_head=relation, operation_head=operation,
        argument_role_heads=roles, argument_proposal_heads=proposals), receipt


@invariant("learning.graph_constraint_direction_respects_protected_halfspaces", scope="learning",
           owner="core/learning/semantic_graph_constraints.py", observational=False)
def _protected_direction() -> tuple:
    result = _project_direction(np.array([-1., 2.]), [np.array([1., 0.])])
    assert np.allclose(result, [0., 2.])
    return ()
