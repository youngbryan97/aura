"""Fit witnessed graph inequalities without sacrificing satisfied witnesses.

This is a numerical constrained search, not an infeasibility prover. Every
accepted update is checked against the nonlinear margins after float32 storage.
The obligations concern retained latent interpretations; fresh decoding must
still search for new competitors and changed latent choices.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

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
    magnitude = np.max(np.abs(direction))
    if magnitude == 0:
        return direction.copy()
    normalized = direction / magnitude
    # The dual is min ||A.T * multiplier + direction||^2, multiplier >= 0.
    # Its KKT conditions give A * projected >= 0. Normalizing the direction
    # keeps a small training gradient from satisfying an absolute stop rule.
    multiplier, _ = nnls(a.T, -normalized)
    return (normalized + a.T @ multiplier) * magnitude


def _fit_graph_parameters(initial, contrasts, *, scale=1., steps=100,
                          required_margin=.1, learning_rate=.001, max_active=32,
                          adaptive_step=False, checkpoint_path=None, progress=None,
                          checkpoint_identity=None, batched=True, objective="squared_deficit",
                          update_rule="working_face", trainable_parameters=None,
                          relation_metric="coefficient_euclidean"):
    """Search for all retained inequalities; retain every already-positive margin."""
    if (objective not in {"squared_deficit", "pairwise_logistic"}
            or update_rule not in {"working_face", "minimum_change"}
            or (update_rule == "minimum_change" and objective != "squared_deficit")
            or relation_metric not in {"coefficient_euclidean", "factor_function"}
            or type(batched) is not bool or type(adaptive_step) is not bool or not contrasts or type(steps) is not int or steps < 1 or type(max_active) is not int
            or max_active < 1 or not np.isfinite(required_margin) or required_margin <= 0
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(scale) or scale <= 0
            or any(not np.isfinite(row.weight) or row.weight <= 0
                   or not np.isfinite(row.fixed_margin) for row in contrasts)):
        raise ValueError("invalid semantic graph constraint fit")
    initial = tuple(np.asarray(value, dtype=np.float64) for value in initial)
    geometry = None
    if relation_metric == "factor_function":
        from core.learning.bilinear_geometry import BilinearFactorGeometry

        geometry = BilinearFactorGeometry.from_factors(*initial[:2], scale=scale)
    if trainable_parameters is None:
        trainable_parameters = (True,) * len(initial)
    trainable_parameters = tuple(trainable_parameters)
    if (len(trainable_parameters) != len(initial)
            or any(type(value) is not bool for value in trainable_parameters)
            or not any(trainable_parameters)):
        raise ValueError("trainable parameter blocks must match model geometry")
    shapes = tuple(value.shape for value in initial)
    ends = np.cumsum([value.size for value in initial])
    mutable = np.concatenate([np.full(value.size, trainable, dtype=bool)
                              for value, trainable in zip(initial, trainable_parameters, strict=True)])
    batch = GraphConstraintBatch(contrasts, scale) if batched else None

    def split(flat):
        return tuple(value.reshape(shape) for value, shape in
                     zip(np.split(flat, ends[:-1]), shapes, strict=True))

    def unpack(flat):
        parts = split(flat)
        return parts if geometry is None else geometry.decode(parts)

    def pack(parameters):
        parts = parameters if geometry is None else geometry.encode(parameters)
        return np.concatenate([value.ravel() for value in parts])

    def pack_gradient(gradients):
        parts = gradients if geometry is None else geometry.pullback(gradients)
        return np.concatenate([value.ravel() for value in parts]) * mutable

    def stored_parameters(flat):
        parts = unpack(flat)
        if geometry is None:
            return parts
        return tuple(part.astype(np.float32).astype(np.float64) if trainable else original
                     for part, original, trainable in zip(parts, initial, trainable_parameters, strict=True))

    def evaluate(flat):
        parameters = stored_parameters(flat)
        values = (batch.margins(parameters) if batch is not None else
                  np.asarray([graph_margin(parameters, row, scale=scale) for row in contrasts]))
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite semantic constraint margin")
        return values

    flat = pack(initial)
    anchor = flat.copy()

    def store_trial(value):
        stored = (value.astype(np.float32).astype(np.float64) if geometry is None else
                  pack(tuple(part.astype(np.float32).astype(np.float64) for part in unpack(value))))
        stored[~mutable] = anchor[~mutable]
        return stored

    margins = evaluate(flat)
    before = margins.copy()
    # A positive margin means this witnessed error already loses. Its floor
    # never decreases, including after new constraints become satisfied.
    floors = np.where(margins > 0., np.minimum(margins, required_margin), -np.inf)
    weights = np.array([row.weight for row in contrasts])
    weights /= weights.sum()

    def loss_at(values):
        terms = (np.logaddexp(0., -values) if objective == "pairwise_logistic"
                 else np.maximum(required_margin - values, 0.) ** 2)
        return float(weights @ terms)

    def restore_trial(trial, values):
        # Tangent motion can leave a curved feasible boundary at second order.
        # Correct violated faces together, then recheck every nonlinear margin
        # at exported precision. Iterations bound curvature refinement, not how
        # many independent retained witnesses can be restored.
        from core.learning.margin_repair import minimum_stored_margin_repair

        for attempt in range(8):
            violated = np.flatnonzero(values < floors)
            if not len(violated):
                return trial, values, attempt
            parameters = stored_parameters(trial)
            normals = []
            for index in violated:
                _, gradient = graph_margin_gradient(parameters, contrasts[index], scale=scale)
                normals.append(pack_gradient(gradient)[mutable])
            matrix = np.stack(normals)
            interior = 8 * np.finfo(np.float32).eps * np.maximum(1., np.abs(floors[violated]))
            required = floors[violated] - values[violated] + interior

            def stored_correction(point):
                full = trial.copy()
                full[mutable] = point
                return store_trial(full)[mutable]

            correction = minimum_stored_margin_repair(
                matrix, required, trial[mutable], tolerance=1e-7,
                store_point=stored_correction,
            )
            if not correction.receipt["stored_primal_feasible"]:
                break
            updated = trial.copy()
            updated[mutable] += correction.displacement
            trial = store_trial(updated)
            values = evaluate(trial)
        return trial, values, 8

    trace, status = [], "search_budget_exhausted"
    checkpoint, start_step = None, 0
    if checkpoint_path is not None:
        from core.learning.semantic_fit_checkpoint import SemanticFitCheckpoint, fit_identity

        source_files = ("semantic_graph_constraints.py", "semantic_graph_batch.py", "semantic_relation_graph_learning.py",
                        "semantic_operation_graph_learning.py", "semantic_argument_graph_learning.py",
                        "semantic_operation_pointer_learning.py", "margin_repair.py", "bilinear_geometry.py",
                        "affine_margin_polish.py", "semantic_fit_checkpoint.py")
        identity = fit_identity({
            "algorithm": [Path(__file__).with_name(name).read_text() for name in source_files],
            "owner": checkpoint_identity, "initial": initial, "contrasts": tuple(contrasts),
            "options": (scale, steps, required_margin, learning_rate, max_active, adaptive_step, batched, objective,
                        update_rule, trainable_parameters, relation_metric),
        })
        checkpoint = SemanticFitCheckpoint(checkpoint_path, identity)
        saved = checkpoint.load()
        from core.learning.semantic_fit_problem import save_fit_problem

        save_fit_problem(Path(checkpoint_path).with_suffix(".problem.npz"),
            identity=identity, initial=initial, contrasts=contrasts,
            options={"scale": scale, "steps": steps, "required_margin": required_margin,
                "learning_rate": learning_rate, "max_active": max_active,
                "adaptive_step": adaptive_step, "batched": batched, "objective": objective,
                "update_rule": update_rule, "trainable_parameters": trainable_parameters,
                "relation_metric": relation_metric})
        if saved is not None:
            allowed = {"running", "search_budget_exhausted", "retained_constraints_satisfied",
                       "no_feasible_direction_found", "no_retention_preserving_step_found",
                       "local_margin_projection_unverified"}
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
                    or not np.all(verified >= saved["floors"])
                    or not np.array_equal(saved["flat"][~mutable], anchor[~mutable])):
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
        deficits = (np.exp(-np.logaddexp(0., margins)) if objective == "pairwise_logistic"
                    else np.maximum(required_margin - margins, 0.))
        if not np.any(deficits):
            status = "retained_constraints_satisfied"
            break
        parameters = stored_parameters(flat)
        direction = (pack_gradient(batch.weighted_gradient(parameters, weights * deficits))
                     if batch is not None else np.zeros_like(flat))
        normals = {}
        # Only binding faces constrain infinitesimal motion. A witness with
        # slack may decrease while remaining above its retained floor.
        active = sorted(np.flatnonzero(np.isfinite(floors) & (margins == floors)),
                        key=lambda index: margins[index] - floors[index])[:max_active]
        if update_rule == "minimum_change":
            active = sorted(set(active) | set(sorted(np.flatnonzero(deficits),
                key=lambda index: (-deficits[index], index))[:max_active]))
        for index, row in enumerate(contrasts):
            if index not in active and (batch is not None or not deficits[index]):
                continue
            _margin, gradient = graph_margin_gradient(parameters, row, scale=scale)
            vector = pack_gradient(gradient)
            if batch is None:
                direction += weights[index] * deficits[index] * vector
            if index in active:
                normals[index] = vector
        descent = direction
        loss = loss_at(margins)
        accepted, cut_rounds, best_trial = False, 0, None
        while True:
            projection_receipt = None
            restoration_descent = False
            if update_rule == "minimum_change":
                from core.learning.margin_repair import minimum_stored_margin_repair

                indices = tuple(normals)
                matrix = np.stack(tuple(normals.values()))
                required = required_margin - margins[list(indices)] + matrix @ (flat - anchor)
                def stored_mutable(point):
                    full = anchor.copy()
                    full[mutable] = point
                    return store_trial(full)[mutable]

                proposal = minimum_stored_margin_repair(
                    matrix[:, mutable], required, anchor[mutable], tolerance=1e-7,
                    store_point=None if geometry is None else stored_mutable)
                projection_receipt = proposal.receipt["continuous_projection"]
                if not proposal.receipt["stored_primal_feasible"]:
                    status = "local_margin_projection_unverified"
                    if checkpoint is not None:
                        checkpoint.save_projection(normals=matrix[:, mutable], required=required,
                            anchor=anchor[mutable], receipt=proposal.receipt, step=step + 1)
                    if progress:
                        progress({"stage": "constraint_projection_unverified", "step": step + 1,
                                  "projection": proposal.receipt})
                    # An unreachable target margin does not rule out a useful
                    # loss decrease. Preserve existing floors while searching
                    # the same objective; never certify the failed projection.
                    direction = _project_direction(descent, [normal for index, normal in normals.items()
                        if np.isfinite(floors[index])])
                    restoration_descent = True
                    projection_receipt = None
                else:
                    direction = np.zeros_like(flat)
                    direction[mutable] = anchor[mutable] + proposal.displacement - flat[mutable]
            else:
                direction = _project_direction(descent, list(normals.values()))
            largest = np.max(np.abs(direction))
            if largest == 0 or not np.isfinite(largest):
                if not restoration_descent:
                    status = "no_feasible_direction_found"
                break
            step_size = 1. if update_rule == "minimum_change" else learning_rate
            if update_rule == "working_face" or restoration_descent:
                direction /= largest
            if (adaptive_step and update_rule == "working_face") or restoration_descent:
                # Use the local loss curvature along the protected direction.
                slopes = (batch.directional_derivative(parameters, unpack(direction))
                          if batch is not None else np.zeros_like(margins))
                if batch is None:
                    for index in np.flatnonzero(deficits):
                        _margin, gradient = graph_margin_gradient(parameters, contrasts[index], scale=scale)
                        slopes[index] = sum(float(np.sum(value * part)) for value, part in
                                            zip(gradient, unpack(direction), strict=True))
                slopes[deficits == 0] = 0.
                curvature = deficits * (1. - deficits) if objective == "pairwise_logistic" else 1.
                denominator = float(weights @ (curvature * slopes ** 2))
                numerator = float(weights @ (deficits * slopes))
                if denominator > 0 and numerator > 0:
                    step_size = numerator / denominator
            direction *= step_size
            blockers = set()
            for backtrack in range(24):
                # Accept only the margins obtained after export to float32.
                trial = store_trial(flat + direction * 2. ** -backtrack)
                trial_margins = evaluate(trial)
                trial_loss = loss_at(trial_margins)
                violated = np.flatnonzero(trial_margins < floors)
                restoration_steps = 0
                if len(violated) and ((update_rule == "minimum_change" and backtrack == 0)
                                     or all(index in normals for index in violated)):
                    trial, trial_margins, restoration_steps = restore_trial(trial, trial_margins)
                    trial_loss = loss_at(trial_margins)
                    violated = np.flatnonzero(trial_margins < floors)
                blockers.update(int(index) for index in violated if index not in normals)
                if update_rule == "minimum_change":
                    worsening = np.flatnonzero((trial_margins < required_margin) & (trial_margins < margins))
                    blockers.update(int(index) for index in worsening if index not in normals)
                if not len(violated) and trial_loss < loss:
                    accepted = True
                    entry = {"step": step + 1, "loss": trial_loss,
                                  "wrong_or_tied": int(np.count_nonzero(trial_margins <= 0)),
                                  "minimum_margin": float(trial_margins.min()), "backtracks": backtrack,
                                  "step_size": step_size * 2. ** -backtrack,
                                  "constraint_cut_rounds": cut_rounds,
                                  "restoration_steps": restoration_steps,
                                  "projected_constraints": len(normals)}
                    if restoration_descent:
                        entry["proposal_rule"] = "retained_loss_descent_after_unverified_projection"
                    if projection_receipt is not None:
                        entry.update(local_affine_projection=projection_receipt,
                                     stored_affine_projection=proposal.receipt,
                                     displacement_from_anchor=float(np.linalg.norm(trial - anchor)))
                    if best_trial is None or trial_loss < best_trial[2]["loss"]:
                        best_trial = trial, trial_margins, entry
                    break
                last_rejected_step = {
                    "step": step + 1, "backtracks": backtrack,
                    "constraint_cut_rounds": cut_rounds,
                    "projected_constraints": len(normals),
                    "retention_violations": len(violated),
                    "maximum_floor_deficit": float(np.max(floors[violated] - trial_margins[violated]))
                    if len(violated) else 0.,
                    "loss_before": loss, "loss_after": trial_loss,
                    "displacement_norm": float(np.linalg.norm(trial - flat)),
                    "restoration_steps": restoration_steps,
                }
                if progress and backtrack in {0, 23}:
                    progress({"stage": "constraint_step_rejected", **last_rejected_step})
            # Keep the feasible proposal while testing blocking-face
            # alternative. A tiny accepted step must not hide a much better
            # tangent step; adding a slack face must not discard useful motion.
            if accepted and not blockers:
                break
            if not blockers:
                status = "no_retention_preserving_step_found"
                break
            # A rejected proposal supplies additional constraints to the same
            # projection. The batch size limits discovery, not protection.
            for index in sorted(blockers, key=lambda i: (margins[i] - floors[i], i))[:max_active]:
                _, gradient = graph_margin_gradient(parameters, contrasts[index], scale=scale)
                normals[index] = pack_gradient(gradient)
            cut_rounds += 1
            if progress:
                progress({"stage": "constraint_direction_refined", "step": step + 1,
                          "constraint_cut_rounds": cut_rounds,
                          "projected_constraints": len(normals)})
        if not accepted:
            break
        flat, margins, entry = best_trial
        status = "search_budget_exhausted"
        floors = np.maximum(floors, np.where(margins > 0.,
            np.minimum(margins, required_margin), -np.inf))
        trace.append(entry)
        persist("running")
        if progress:
            progress({"stage": "constraint_fit_step", "completed": step + 1,
                      "total": steps, **entry})
    if objective == "squared_deficit" and np.all(margins >= required_margin):
        status = "retained_constraints_satisfied"
    elif status == "running":
        status = "search_budget_exhausted"
    persist(status)
    values = stored_parameters(flat)
    receipt = {
        "objective": ("retained_pairwise_likelihood_v1" if objective == "pairwise_logistic"
                      else "retained_semantic_inequalities_v1"), "status": status,
        "initial_loss": loss_at(before), "stored_loss": loss_at(margins),
        "pairs": len(contrasts), "required_margin": required_margin,
        "initial_margins": before.tolist(), "stored_margins": margins.tolist(),
        "initial_wrong_or_tied": int(np.count_nonzero(before <= 0)),
        "stored_wrong_or_tied": int(np.count_nonzero(margins <= 0)),
        "retained_positive_regressions": int(np.count_nonzero((before > 0) & (margins <= 0))),
        "accepted_steps": trace, "projection_batch_size": max_active,
        "accepted_constraint_cut_rounds": sum(row["constraint_cut_rounds"] for row in trace),
        "peak_accepted_projected_constraints": max((row["projected_constraints"] for row in trace), default=0),
        "step_policy": ("minimum_change_stored_working_set_v2" if update_rule == "minimum_change" else
                        "working_face_fixed_v6" if not adaptive_step else
                        "working_face_logistic_v6" if objective == "pairwise_logistic" else
                        "working_face_deficit_v6"),
        "update_rule": update_rule,
        "trainable_parameters": list(trainable_parameters),
        "trainable_coordinate_count": int(np.count_nonzero(mutable)),
        "frozen_coordinate_count": int(np.count_nonzero(~mutable)),
        "frozen_parameters_unchanged": bool(np.array_equal(flat[~mutable], anchor[~mutable])),
        "parameter_displacements": [float(np.linalg.norm(value - original))
                                    for value, original in zip(values, initial, strict=True)],
        "displacement_from_anchor": (float(np.linalg.norm(flat - anchor)) if geometry is None else
            float(np.sqrt(sum(np.sum((value - original) ** 2)
                for value, original in zip(values, initial, strict=True))))),
        "global_minimum_change_proven": False,
        "all_constraints_checked_at_acceptance": True,
        "infeasibility_proven": False, "latent_choices_frozen_for_update": True,
        "serving_authority": False,
    }
    if geometry is not None:
        receipt["relation_geometry"] = {**geometry.certificate(),
            "coordinate_displacement_norm": float(np.linalg.norm(flat - anchor)),
            "physical_storage_dtype": "float32"}
    return values, receipt


def _model_parameters(head, operation_head):
    return (head.query_projection, head.definition_projection,
            *(value for part in operation_head.heads for value in (part.weight, part.bias)))


def _fitted_heads(head, operation_head, values):
    return (replace(head, query_projection=values[0], definition_projection=values[1]),
            replace(operation_head, heads=tuple(replace(part,
                weight=values[2 + 2 * index], bias=values[3 + 2 * index])
                for index, part in enumerate(operation_head.heads))))


def fit_graph_constraints(head, operation_head, contrasts, *, learn_operations=True, **options):
    if type(learn_operations) is not bool:
        raise ValueError("operation learning option must be boolean")
    parameters = _model_parameters(head, operation_head)
    values, receipt = _fit_graph_parameters(parameters, contrasts,
        trainable_parameters=(True, True, *((learn_operations,) * (len(parameters) - 2))), **options)
    receipt["operation_head_trainable"] = learn_operations
    return (*_fitted_heads(head, operation_head, values), receipt)


def fit_complete_graph_constraints(model, contrasts, *, learn_operation_pointer=False,
                                   learn_operations=True, **options):
    from core.learning.semantic_argument_graph_learning import argument_parameters
    from core.learning.semantic_operation_pointer_learning import (
        operation_pointer_from_parameters,
        operation_pointer_parameters,
    )

    if type(learn_operation_pointer) is not bool or type(learn_operations) is not bool:
        raise ValueError("operation learning options must be boolean")
    base = _model_parameters(model.definition_relation_head, model.operation_head)
    pointers = operation_pointer_parameters(model) if learn_operation_pointer else ()
    parameters = (*base, *argument_parameters(model), *pointers)
    trainable = (True, True, *((learn_operations,) * (len(base) - 2)),
                 *((True,) * (len(parameters) - len(base))))
    values, receipt = _fit_graph_parameters(parameters, contrasts,
                                          trainable_parameters=trainable, **options)
    relation, operation = _fitted_heads(model.definition_relation_head, model.operation_head, values)
    offset = len(base)
    roles = tuple(replace(head, weight=values[offset + 4 * index],
                          bias=float(values[offset + 4 * index + 1]))
                  for index, head in enumerate(model.argument_role_heads))
    proposals = tuple(replace(head, weight=values[offset + 4 * index + 2],
                              bias=float(values[offset + 4 * index + 3]))
                      for index, head in enumerate(model.argument_proposal_heads))
    changes = {"operation_pointer": operation_pointer_from_parameters(
        values[-2:], pointer=model.operation_pointer)} if learn_operation_pointer else {}
    receipt["operation_pointer_trainable"] = learn_operation_pointer
    receipt["operation_head_trainable"] = learn_operations
    receipt["operation_pointer_capacity_preserved"] = True
    receipt["operation_pointer_interaction_trainable"] = bool(
        learn_operation_pointer and model.operation_pointer.pair_weight is not None)
    return model._with_coefficients(definition_relation_head=relation, operation_head=operation,
        argument_role_heads=roles, argument_proposal_heads=proposals, **changes), receipt


@invariant("learning.graph_constraint_direction_respects_protected_halfspaces", scope="learning",
           owner="core/learning/semantic_graph_constraints.py", observational=False)
def _protected_direction() -> tuple:
    result = _project_direction(np.array([-1., 2.]), [np.array([1., 0.])])
    assert np.allclose(result, [0., 2.])
    return ()
