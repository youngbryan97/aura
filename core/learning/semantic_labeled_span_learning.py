"""Learn operation meanings and boundaries in one exact span-set distribution."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from core.learning.semantic_program_shared_transducer import _geometry
from core.learning.semantic_program_transducer import (
    OPERATION_BACKGROUND_LABEL,
    LinearClassifierHead,
    MultiViewClassifierHead,
    _sha,
)
from core.learning.semantic_span_set_learning import span_set_partition
from core.verify.invariants import invariant


def labeled_span_partition(scores, max_spans):
    """Sum every non-overlapping labeled set; background has one skip path."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or not all(scores.shape) or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("invalid labeled span geometry")
    collapsed = logsumexp(scores, axis=2)
    partition, span_mass = span_set_partition(collapsed, max_spans)
    conditional = np.zeros_like(scores)
    valid = np.isfinite(collapsed)
    conditional[valid] = np.exp(scores[valid] - collapsed[valid, None])
    return partition, span_mass[:, :, None] * conditional


@invariant("semantic.labeled_span_partition", scope="semantic_program",
           owner="core/learning/semantic_labeled_span_learning.py", observational=False)
def _check_labeled_span_partition():
    scores = np.log(np.asarray([[[2., 3.]], [[5., 7.]]]))
    partition, marginal = labeled_span_partition(scores, 1)
    assert np.isclose(partition, np.log(18.))
    assert np.allclose(marginal, np.exp(scores) / 18.)
    return ()


def _runtime_odds(logits):
    """Match the existing scorer's clipped log probabilities with zero background."""
    all_logits = np.column_stack((logits, np.zeros(len(logits))))
    logp = all_logits - logsumexp(all_logits, axis=1, keepdims=True)
    floor = np.log(1e-12)
    return np.maximum(logp[:, :-1], floor) - np.maximum(logp[:, -1:], floor), logp


@dataclass(frozen=True)
class MeanSpanEvidence:
    """Pool by prefix sums and scatter gradients without storing all span vectors."""

    hidden: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    norms: np.ndarray
    span_width: int

    @classmethod
    def build(cls, hidden, span_width, excluded=()):
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim != 2 or not all(hidden.shape) or not np.all(np.isfinite(hidden)):
            raise ValueError("invalid mean-span hidden representation")
        n = len(hidden)
        if type(span_width) is not int or span_width < 1:
            raise ValueError("invalid mean-span width")
        width = min(span_width, n)
        start, column = np.nonzero(np.arange(n)[:, None] + np.arange(1, width + 1) <= n)
        end = start + column + 1
        keep = np.ones(len(start), dtype=bool)
        for span in excluded:
            span.validate_bound(n)
            keep &= ~((start < span.end) & (span.start < end))
        start, end = start[keep], end[keep]
        prefix = np.vstack((np.zeros(hidden.shape[1]), np.cumsum(hidden, axis=0, dtype=np.float64)))
        norms = np.empty(len(start))
        # One length at a time bounds temporary feature memory by tokens x width.
        for length in range(1, width + 1):
            selected = end - start == length
            total = prefix[end[selected]] - prefix[start[selected]]
            norm = np.linalg.norm(total, axis=1)
            norms[selected] = np.where(norm / length > 1e-8, norm, length)
        return cls(hidden, start, end, norms, width)

    def project(self, weight):
        projected = self.hidden @ weight.T
        prefix = np.vstack((np.zeros(weight.shape[0]), np.cumsum(projected, axis=0)))
        return (prefix[self.ends] - prefix[self.starts]) / self.norms[:, None]

    def adjoint(self, residual):
        mass = np.zeros((len(self.hidden) + 1, residual.shape[1]))
        scaled = residual / self.norms[:, None]
        np.add.at(mass, self.starts, scaled)
        np.add.at(mass, self.ends, -scaled)
        return np.cumsum(mass, axis=0)[:-1].T @ self.hidden


@dataclass(frozen=True)
class MeanTransitionSpanEvidence(MeanSpanEvidence):
    """Project two normalized views without retaining spans x hidden-width arrays."""

    transition_norms: np.ndarray
    joint_norms: np.ndarray

    @classmethod
    def build(cls, hidden, span_width, excluded=()):
        mean = MeanSpanEvidence.build(hidden, span_width, excluded)
        prefix = np.vstack((np.zeros(mean.hidden.shape[1]),
                            np.cumsum(mean.hidden, axis=0, dtype=np.float64)))
        transition_norms, joint_norms = np.empty(len(mean.starts)), np.empty(len(mean.starts))
        for length in range(1, mean.span_width + 1):
            selected = mean.ends - mean.starts == length
            starts, ends = mean.starts[selected], mean.ends[selected]
            before = mean.hidden[np.maximum(starts - 1, 0)].copy()
            before[starts == 0] = 0.
            transition = mean.hidden[ends - 1] - before
            raw_norm = np.linalg.norm(transition, axis=1)
            denominator = np.where(raw_norm > 1e-8, raw_norm, 1.)
            transition_norms[selected] = denominator
            local_norm = np.linalg.norm(prefix[ends] - prefix[starts], axis=1) / mean.norms[selected]
            joint = np.sqrt(local_norm ** 2 + (raw_norm / denominator) ** 2)
            joint_norms[selected] = np.where(joint > 1e-8, joint, 1.)
        return cls(mean.hidden, mean.starts, mean.ends, mean.norms, mean.span_width,
                   transition_norms, joint_norms)

    def project(self, weight):
        width = self.hidden.shape[1]
        if weight.shape[1] != 2 * width:
            raise ValueError("mean-transition projection width differs")
        projected = self.hidden @ weight[:, width:].T
        before = projected[np.maximum(self.starts - 1, 0)].copy()
        before[self.starts == 0] = 0.
        transition = (projected[self.ends - 1] - before) / self.transition_norms[:, None]
        return (super().project(weight[:, :width]) + transition) / self.joint_norms[:, None]

    def adjoint(self, residual):
        scaled = residual / self.joint_norms[:, None]
        mean = super().adjoint(scaled)
        transition = scaled / self.transition_norms[:, None]
        mass = np.zeros((len(self.hidden), residual.shape[1]))
        np.add.at(mass, self.ends - 1, transition)
        present = self.starts > 0
        np.add.at(mass, self.starts[present] - 1, -transition[present])
        return np.concatenate((mean, mass.T @ self.hidden), axis=1)


def _labeled_span_evidence(model, item):
    """Use the declared runtime view, including its actual channel coordinates."""
    channels = {
        "lexical_mean": "input_token_embedding",
        "middle_mean": "middle_causal_hidden",
        "contextual_mean": "final_causal_hidden",
        "contextual_mean_transition": "final_causal_hidden",
    }
    modes = model.operation_head.modes
    if len(modes) != 1 or modes[0] not in {*channels, "span_mean"}:
        raise ValueError("labeled span fitting requires a supported single pooled view")
    mode = modes[0]
    if mode == "span_mean":
        hidden = item.hidden_states
    else:
        try:
            channel = model.hidden_channels.index(channels[mode])
        except ValueError as exc:
            raise ValueError("labeled span fitting needs the declared evidence channel") from exc
        begin = sum(model.hidden_channel_widths[:channel])
        hidden = item.hidden_states[:, begin:begin + model.hidden_channel_widths[channel]]
    evidence_type = MeanTransitionSpanEvidence if mode == "contextual_mean_transition" else MeanSpanEvidence
    return evidence_type.build(hidden, model.max_span_tokens, item.ir.input_spans)


def _labeled_loss(parameters, rows, *, labels, width, max_spans, regularization, center):
    weight = parameters[:labels * width].reshape(labels, width)
    bias = parameters[labels * width:]
    delta = parameters - center
    loss, gradient = 0.5 * regularization * float(delta @ delta), regularization * delta
    for evidence, target, scale in rows:
        odds, logp = _runtime_odds(evidence.project(weight) + bias)
        scores = np.full((len(evidence.hidden), evidence.span_width, labels), -np.inf)
        columns = evidence.ends - evidence.starts - 1
        scores[evidence.starts, columns] = odds
        partition, marginal = labeled_span_partition(scores, max_spans)
        loss += scale * (partition - sum(scores[start, length - 1, label] for start, length, label in target))
        for start, length, label in target:
            marginal[start, length - 1, label] -= 1.
        residual = marginal[evidence.starts, columns]
        # Clipping is part of the shipped scorer, including its zero derivative.
        active = logp > np.log(1e-12)
        q = residual * active[:, :-1]
        q_background = -residual.sum(axis=1) * active[:, -1]
        dlogits = q - np.exp(logp[:, :-1]) * (q.sum(axis=1) + q_background)[:, None]
        gradient[:labels * width] += scale * evidence.adjoint(dlogits).ravel()
        gradient[labels * width:] += scale * dlogits.sum(axis=0)
    return loss, gradient


def _labeled_graph_loss(parameters, contrasts, *, labels, width, relation_parameters, scale):
    """Train operation evidence against the binding scores used in real selection."""
    from core.learning.semantic_relation_graph_learning import graph_margin_gradient

    weight = parameters[:labels * width].reshape(labels, width)
    bias = parameters[labels * width:]
    complete = (*relation_parameters, np.vstack((weight, np.zeros(width))), np.append(bias, 0.))
    loss, gradient, margins = 0., np.zeros_like(parameters), []
    total = sum(row.weight for row in contrasts)
    for row in contrasts:
        margin, derivatives = graph_margin_gradient(complete, row, scale=scale)
        margins.append(margin)
        deficit = max(.1 - margin, 0.)
        coefficient = row.weight / total
        loss += coefficient * deficit ** 2
        gradient[:labels * width] -= 2 * coefficient * deficit * derivatives[2][:-1].ravel()
        gradient[labels * width:] -= 2 * coefficient * deficit * derivatives[3][:-1]
    return loss, gradient, margins


def refit_compositional_labeled_spans(model, examples, *, max_iter=200, progress=None,
                                     runtime_constraint_sources=(), solve_time_limit_s=20.):
    """Fit source-only labeled sets and export through the existing odds scorer."""
    from scipy.optimize import minimize

    examples = tuple(examples)
    train = tuple(item for item in examples if item.split == "train")
    ids = [item.ir.source_text_sha256 for item in train]
    if (not ids or len(set(ids)) != len(ids)
            or set(ids) & {item.ir.source_text_sha256 for item in examples if item.split != "train"}
            or type(max_iter) is not int or max_iter < 1):
        raise ValueError("labeled span fitting needs unique disjoint source training")
    constraint_sources = tuple(runtime_constraint_sources)
    if (len(set(constraint_sources)) != len(constraint_sources)
            or not set(constraint_sources) <= set(ids)
            or type(solve_time_limit_s) not in (int, float)
            or not np.isfinite(solve_time_limit_s) or solve_time_limit_s <= 0):
        raise ValueError("runtime constraints require unique source-training identities and a finite allowance")
    if any(item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
           or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
           or (item.hidden_channels, item.hidden_channel_widths) != (model.hidden_channels, model.hidden_channel_widths)
           for item in train):
        raise ValueError("labeled span source representation differs")
    labels = tuple(label for label in model.operation_head.labels if label != OPERATION_BACKGROUND_LABEL)
    if {instruction.op for item in train for instruction in item.ir.instructions} != set(labels):
        raise ValueError("labeled span training must retain every source operation")
    width = model.operation_head.heads[0].width
    counts, rows, targets = Counter(_geometry(item) for item in train), [], []
    for item in train:
        evidence = _labeled_span_evidence(model, item)
        target = tuple(sorted((i.operation_span.start, i.operation_span.end - i.operation_span.start,
                               labels.index(i.op)) for i in item.ir.instructions))
        available = set(zip(evidence.starts, evidence.ends, strict=True))
        if (not target or len(target) > model.max_steps
                or any((start, start + length) not in available for start, length, _ in target)
                or any(a + length > b for (a, length, _), (b, _, _) in zip(target, target[1:], strict=False))):
            raise ValueError("labeled source targets are excluded or overlapping")
        rows.append((evidence, target, 1. / counts[_geometry(item)] / len(counts)))
        targets.append([item.ir.source_text_sha256, target])
    head = model.operation_head.heads[0]
    selected = [head.labels.index(label) for label in labels]
    weight, bias = head.weight[selected].astype(np.float64), head.bias[selected].astype(np.float64)
    if OPERATION_BACKGROUND_LABEL in head.labels:
        background = head.labels.index(OPERATION_BACKGROUND_LABEL)
        weight -= head.weight[background]
        bias -= head.bias[background]
    initial = np.concatenate((weight.ravel(), bias))
    contrasts, constraint_records = [], []
    if constraint_sources:
        from core.learning.semantic_joint_graph_learning import mine_runtime_graph_contrast

        if (head.labels != (*labels, OPERATION_BACKGROUND_LABEL)
                or model.training_receipt.get("operation_background_fit", {}).get("score")
                != "joint_operation_background_log_odds_v2"):
            raise ValueError("runtime graph fitting requires the exported labeled-span score contract")
        by_source = {item.ir.source_text_sha256: item for item in train}
        for source in constraint_sources:
            contrast, record = mine_runtime_graph_contrast(
                model, by_source[source], solve_time_limit_s=solve_time_limit_s,
                decode_time_limit_s=solve_time_limit_s,
            )
            if record["status"] not in {"equivalent", "counterexample"}:
                raise RuntimeError(f"runtime graph constraint is unresolved: {record['status']}")
            constraint_records.append(record)
            if contrast is not None:
                contrasts.append(contrast)
            if progress is not None:
                progress({"stage": "labeled_runtime_constraint", "source": source, "status": record["status"]})
    graph_options = dict(labels=len(labels), width=width,
                         relation_parameters=(model.definition_relation_head.query_projection.astype(np.float64),
                                              model.definition_relation_head.definition_projection.astype(np.float64)),
                         scale=model.definition_relation_scale)
    options = dict(labels=len(labels), width=width, max_spans=model.max_steps,
                   regularization=1. / (10. * len(train)), center=initial)
    def objective(value):
        loss, gradient = _labeled_loss(value, rows, **options)
        if contrasts:
            graph_loss, graph_gradient, _ = _labeled_graph_loss(value, contrasts, **graph_options)
            loss += graph_loss
            gradient += graph_gradient
        return loss, gradient
    initial_loss = objective(initial)[0]
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1
        if progress is not None:
            progress({"stage": "labeled_span_fit", "iteration": iterations})

    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5})
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"labeled span fit incomplete: {fitted.status}: {fitted.message}")
    value = fitted.x.astype(np.float32)
    operation_head = MultiViewClassifierHead(model.operation_head.modes, (LinearClassifierHead(
        (*labels, OPERATION_BACKGROUND_LABEL),
        np.vstack((value[:len(labels) * width].reshape(len(labels), width), np.zeros(width, dtype=np.float32))),
        np.append(value[len(labels) * width:], np.float32(0))),))
    coefficients = model._coefficient_body()
    coefficients.update(operation_head=operation_head.to_dict(), operation_length_penalty=0.)
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficients)
    positives = sum(len(target) for _, target, _ in rows)
    body["operation_background_fit"] = {
        "schema": "aura.semantic_operation_background_fit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "background_label": OPERATION_BACKGROUND_LABEL, "modes": list(operation_head.modes),
        "labels": list(operation_head.labels), "score": "joint_operation_background_log_odds_v2",
        "training_examples": len(train), "training_ids_sha256": _sha(sorted(ids)),
        "positive_spans": positives, "background_spans": sum(len(e.starts) for e, _, _ in rows) - positives,
        "targets_sha256": _sha(targets), "validation_used_for_fit": False, "test_examples_used": 0,
        "serving_authority": False,
    }
    body["labeled_span_fit"] = {
        "objective": "exact_source_labeled_nonoverlap_likelihood_v1",
        "negative_space": "all_bounded_intervals_excluding_public_input_spans",
        "source_target_sha256": _sha(targets), "geometry_balanced": True,
        "initial_loss": initial_loss, "exported_loss": objective(value.astype(np.float64))[0],
        "iterations": int(fitted.nit), "max_span_tokens": model.max_span_tokens,
        "max_spans": model.max_steps, "converged": True, "serving_authority": False,
    }
    if constraint_sources:
        from core.learning.semantic_fit_checkpoint import fit_identity

        initial_margins = _labeled_graph_loss(initial, contrasts, **graph_options)[2]
        exported_margins = _labeled_graph_loss(value.astype(np.float64), contrasts, **graph_options)[2]
        body["labeled_span_fit"]["runtime_graph_constraints"] = {
            "parent": model.receipt_sha256, "sources": list(constraint_sources),
            "records": constraint_records, "constraints_sha256": fit_identity(tuple(contrasts)),
            "objective": "source_span_likelihood_plus_runtime_squared_margin_deficit_v1",
            "required_margin": .1, "initial_margins": initial_margins,
            "exported_margins": exported_margins,
            "all_witnessed_margins_satisfied": (all(margin >= .1 for margin in exported_margins)
                                                if exported_margins else None),
            "full_source_decision_retention_measured": False,
            "binding_coefficients": "frozen_parent", "runtime_competitors": "frozen_at_mining",
            "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
        }
    if body.get("operation_search_policy") == "complete_bounded_v1":
        body["operation_label_limit"] = len(operation_head.labels)
    return replace(model, operation_head=operation_head, operation_length_penalty=0.,
                   training_receipt={**body, "receipt_sha256": _sha(body)})
