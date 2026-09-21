"""Train shared boundaries against complete non-overlapping span sets."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_shared_transducer import _geometry
from core.learning.semantic_program_transducer import LinearPointerHead, _sha
from typing import Any


def span_set_partition(scores: np.ndarray, max_spans: int) -> tuple[float, np.ndarray]:
    """Return log partition and span marginals, including the empty set once.

    scores[start, length - 1] scores one interval. Uncovered tokens have zero
    score and a unique one-token skip edge, so background segmentations cannot
    multiply the probability of a span set. Intervals may touch, not overlap.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not all(scores.shape) or type(max_spans) is not int or max_spans < 1:
        raise ValueError("invalid span-set partition geometry")
    n, width = scores.shape
    valid = np.arange(n)[:, None] + np.arange(1, width + 1) <= n
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)) or not np.all(np.isneginf(scores[~valid])):
        raise ValueError("span-set scores need finite or excluded intervals and masked padding")
    count = min(max_spans, n)
    forward = np.full((n + 1, count + 1), -np.inf)
    forward[:, 0] = 0.0
    for end in range(1, n + 1):
        lengths = np.arange(1, min(width, end) + 1)
        starts = end - lengths
        for k in range(1, min(count, end) + 1):
            alternatives = forward[starts, k - 1] + scores[starts, lengths - 1]
            forward[end, k] = np.logaddexp(forward[end - 1, k], np.logaddexp.reduce(alternatives))
    partition = float(np.logaddexp.reduce(forward[n]))
    adjoint = np.zeros_like(forward)
    adjoint[n] = np.exp(forward[n] - partition)
    marginals = np.zeros_like(scores)
    # Reverse the same acyclic recurrence rather than approximate a top-k bank.
    for end in range(n, 0, -1):
        lengths = np.arange(1, min(width, end) + 1)
        starts = end - lengths
        for k in range(1, min(count, end) + 1):
            scale = adjoint[end, k]
            normalizer = forward[end, k]
            if scale == 0 or not np.isfinite(normalizer):
                continue
            adjoint[end - 1, k] += scale * np.exp(forward[end - 1, k] - normalizer)
            mass = scale * np.exp(forward[starts, k - 1] + scores[starts, lengths - 1] - normalizer)
            adjoint[starts, k - 1] += mass
            marginals[starts, lengths - 1] += mass
    return partition, marginals


def _span_set_loss(
    weight: Any,
    rows: Any,
    *,
    width: Any,
    max_spans: Any,
    regularization: Any,
    center: Any,
    learn_pair: bool=False,
) -> tuple[Any, Any]:
    delta = weight - center
    loss = 0.5 * regularization * float(delta @ delta)
    gradient = regularization * delta
    for hidden, starts, ends, fixed, target, sample_weight in rows:
        projected = hidden @ weight[:2 * width].reshape(2, width).T
        scores = np.full(fixed.shape, -np.inf)
        columns = ends - starts
        scores[starts, columns] = projected[starts, 0] + projected[ends, 1] + weight[-1] + fixed[starts, columns]
        if learn_pair:
            pair_weight = weight[2 * width:3 * width] * np.sqrt(width)
            # Stream one diagonal at a time; do not retain an intervals-by-width tensor.
            for column in range(scores.shape[1]):
                size = len(hidden) - column
                pair_score = np.einsum("ij,j,ij->i", hidden[:size], pair_weight, hidden[column:], optimize=False)
                scores[:size, column] += pair_score
        partition, marginals = span_set_partition(scores, max_spans)
        loss += sample_weight * (partition - sum(scores[s, length - 1] for s, length in target))
        residual = marginals
        for start, length in target:
            residual[start, length - 1] -= 1.0
        values = residual[starts, columns]
        start_mass = np.bincount(starts, weights=values, minlength=len(hidden))
        end_mass = np.bincount(ends, weights=values, minlength=len(hidden))
        gradient[:width] += sample_weight * (start_mass @ hidden)
        gradient[width:2 * width] += sample_weight * (end_mass @ hidden)
        if learn_pair:
            for column in range(scores.shape[1]):
                size = len(hidden) - column
                gradient[2 * width:3 * width] += sample_weight * np.sqrt(width) * np.einsum(
                    "i,ij,ij->j", residual[:size, column], hidden[:size], hidden[column:], optimize=False)
        gradient[-1] += sample_weight * float(values.sum())
    return loss, gradient


def fit_span_set_pointer(
    training: tuple[Any, ...],
    *,
    spans: Any,
    pointer: Any,
    max_span_tokens: Any,
    max_spans: Any,
    length_penalty: float=0.0,
    inverse_regularization: float=10.0,
    max_iter: int=200,
    progress: Any=None,
    excluded_spans: Any=None,
    learn_pair: bool=False,
) -> Any:
    """Fit complete source span sets, optionally learning existing pair interactions."""
    from scipy.optimize import minimize

    training = tuple(training)
    if (
        not training or any(item.split != "train" for item in training)
        or type(max_span_tokens) is not int or max_span_tokens < 1
        or type(max_spans) is not int or max_spans < 1
        or type(max_iter) is not int or max_iter < 1
        or not np.isfinite(length_penalty)
        or not np.isfinite(inverse_regularization) or inverse_regularization <= 0
        or (excluded_spans is not None and not callable(excluded_spans))
        or type(learn_pair) is not bool
    ):
        raise ValueError("invalid source span-set training contract")
    ids = [item.ir.source_text_sha256 for item in training]
    if len(set(ids)) != len(ids):
        raise ValueError("span-set training sources must be unique")
    counts = Counter(_geometry(item) for item in training)
    rows, targets, exclusions = [], [], []
    for item in training:
        hidden = item.hidden_states
        sequence = pointer.score_sequence(hidden)
        target = tuple(sorted(spans(item), key=lambda span: (span.start, span.end)))
        if len(target) > max_spans or len(set(target)) != len(target):
            raise ValueError("span-set target count is invalid")
        previous_end = 0
        for span in target:
            span.validate_bound(len(hidden))
            if span.start < previous_end or span.end - span.start > max_span_tokens:
                raise ValueError("span-set target intervals overlap or exceed the span bound")
            previous_end = span.end
        n = len(hidden)
        excluded = tuple(excluded_spans(item)) if excluded_spans is not None else ()
        for span in excluded:
            span.validate_bound(n)
        if any(a.start < b.end and b.start < a.end for a in target for b in excluded):
            raise ValueError("span-set target overlaps excluded evidence")
        exclusions.append([item.ir.source_text_sha256, [span.to_dict() for span in excluded]])
        length = min(max_span_tokens, n)
        starts, columns = np.nonzero(np.arange(n)[:, None] + np.arange(1, length + 1) <= n)
        ends = starts + columns
        allowed = np.ones(len(starts), dtype=bool)
        for span in excluded:
            allowed &= ~((starts < span.end) & (span.start <= ends))
        starts, ends = starts[allowed], ends[allowed]
        fixed = np.full((n, length), -np.inf)
        for start, end in zip(starts, ends, strict=True):
            fixed[start, end - start] = (
                -length_penalty if learn_pair else
                sequence.score_span(TokenSpan(int(start), int(end + 1)))
                - float(sequence.start[start] + sequence.end[end]) - length_penalty
            )
        target_pairs = tuple((span.start, span.end - span.start) for span in target)
        rows.append((hidden, starts, ends, fixed, target_pairs, 1.0 / counts[_geometry(item)] / len(counts)))
        targets.append([item.ir.source_text_sha256, target_pairs])
    width = pointer.width
    components = [pointer.start_weight, pointer.end_weight]
    if learn_pair:
        components.append(pointer.pair_weight if pointer.pair_weight is not None else np.zeros(width))
    initial = np.concatenate((*components, [pointer.start_bias + pointer.end_bias])).astype(np.float64)
    regularization = 1.0 / (inverse_regularization * len(training))

    def objective(weight: Any) -> Any:
        return _span_set_loss(weight, rows, width=width, max_spans=max_spans,
                              regularization=regularization, center=initial, learn_pair=learn_pair)

    initial_loss, _ = objective(initial)
    iterations = 0

    def callback(weight: Any) -> None:
        nonlocal iterations
        iterations += 1
        if progress is not None:
            progress({"stage": "span_set_fit", "iteration": iterations})

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"span-set fit incomplete: {result.status}: {result.message}")
    coefficients = np.asarray(result.x, dtype=np.float32)
    fitted = LinearPointerHead(coefficients[:width], float(coefficients[-1]) / 2,
                               coefficients[width:2 * width], float(coefficients[-1]) / 2,
                               coefficients[2 * width:3 * width] if learn_pair else pointer.pair_weight)
    exported_loss, _ = objective(coefficients.astype(np.float64))
    receipt = {
        "objective": "source_nonoverlapping_span_set_likelihood_v1",
        "training_examples": len(training), "targets_sha256": _sha(targets),
        "max_spans": max_spans, "max_span_tokens": max_span_tokens,
        "length_penalty": float(length_penalty), "regularization": regularization,
        "regularization_center": "parent_boundary_and_pair_coefficients" if learn_pair else "parent_boundary_coefficients",
        "pair_interaction": "learned_sqrt_width_diagonal_product" if learn_pair else "frozen_parent",
        "geometry_balanced": True,
        "excluded_spans_sha256": _sha(exclusions),
        "negative_space": "all_bounded_intervals_excluding_declared_spans",
        "initial_loss": initial_loss, "exported_loss": exported_loss,
        "iterations": int(result.nit), "converged": True,
        "validation_used_for_fit": False, "test_examples_used": 0,
    }
    return fitted, receipt


def refit_compositional_span_set_pointer(
    model: Any,
    examples: Any,
    *,
    max_iter: int=200,
    progress: Any=None,
    learn_pair: bool=False,
) -> Any:
    """Export the fitted boundaries through the existing runtime pointer contract."""
    if model.training_receipt.get("operation_background_fit", {}).get("score") == "joint_operation_background_log_odds_v2":
        raise ValueError("span-set objective requires runtime boundary scores")
    training = tuple(item for item in examples if item.split == "train")
    if any(
        item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
        or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
        or (item.hidden_channels, item.hidden_channel_widths) != (model.hidden_channels, model.hidden_channel_widths)
        for item in training
    ):
        raise ValueError("span-set source representation differs from the model")
    pointer, fit = fit_span_set_pointer(
        training, spans=lambda item: tuple(i.operation_span for i in item.ir.instructions),
        pointer=model.operation_pointer, max_span_tokens=model.max_span_tokens,
        max_spans=model.max_steps, length_penalty=model.operation_length_penalty,
        max_iter=max_iter, progress=progress, excluded_spans=lambda item: item.ir.input_spans,
        learn_pair=learn_pair,
    )
    coefficients = model._coefficient_body()
    coefficients["operation_pointer"] = pointer.to_dict()
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficients)
    body["span_set_pointer_refit"] = {
        "schema": "aura.semantic_span_set_pointer_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "fit": fit, "serving_authority": False,
        "operation_labels_and_arguments": "unchanged_not_trained_by_this_objective",
    }
    return replace(model, operation_pointer=pointer,
                   training_receipt={**body, "receipt_sha256": _sha(body)})
