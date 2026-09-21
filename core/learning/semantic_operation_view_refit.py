"""Refit a shared operation classifier using the existing semantic views."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations

import numpy as np

from core.learning.semantic_program_shared_transducer import _geometry, _normalized_weights
from core.learning.semantic_program_transducer import (
    _MAX_OPERATION_VIEWS,
    _OPERATION_FEATURE_MODES,
    _OPERATION_FEATURE_MODES_V2,
    _OPERATION_FEATURE_MODES_V3,
    OPERATION_BACKGROUND_LABEL,
    MultiViewClassifierHead,
    _fit_classifier,
    _operation_feature,
    _operation_feature_width,
    _sha,
)


def _selection_key(row, *, predicted_charts):
    chart = (-row["graph_exact"], -row["operation_exact"], -row["span_exact"]) if predicted_charts else ()
    return (*chart, -row["correct"], row["cross_entropy"], len(row["modes"]), row["modes"])


def valid_operation_view_contract(head, receipt, channels, widths):
    """Check every selected view against both its evidence and tensor geometry."""
    from core.learning.semantic_operation_background import valid_background_contract

    if not valid_background_contract(head, receipt):
        return False
    try:
        if any(
            component.weight.shape[1] != _operation_feature_width(
                mode, hidden_channels=channels, hidden_channel_widths=widths,
            )
            for mode, component in zip(head.modes, head.heads, strict=True)
        ):
            return False
    except ValueError:
        return False
    selection = receipt.get("operation_view_selection")
    if selection is None:
        return head.modes == ("contextual_mean",)
    if not isinstance(selection, Mapping):
        return False
    predicted_charts = selection.get("schema") == "aura.semantic_operation_view_selection.v2"
    if selection.get("schema") not in {"aura.semantic_operation_view_selection.v1", "aura.semantic_operation_view_selection.v2"}:
        return False
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    for row in candidates:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("modes"), list)
            or not 1 <= len(row["modes"]) <= _MAX_OPERATION_VIEWS
            or any(mode not in _OPERATION_FEATURE_MODES for mode in row["modes"])
            or len(set(row["modes"])) != len(row["modes"])
            or type(row.get("selected")) is not bool
            or type(row.get("total")) is not int
            or type(row.get("correct")) is not int
            or not 0 <= row["correct"] <= row["total"]
            or row["total"] < 1
            or type(row.get("cross_entropy")) not in (int, float)
            or not np.isfinite(row["cross_entropy"])
            or row["cross_entropy"] < 0
        ):
            return False
        if predicted_charts and (
            type(row.get("validation_examples")) is not int
            or row["validation_examples"] < 1
            or any(type(row.get(key)) is not int or not 0 <= row[key] <= row["validation_examples"] for key in ("graph_exact", "operation_exact", "span_exact"))
            or type(row.get("length_penalty")) not in (int, float)
            or not np.isfinite(row["length_penalty"])
        ):
            return False
    if len({tuple(row["modes"]) for row in candidates}) != len(candidates) or len({row["total"] for row in candidates}) != 1:
        return False
    selected = [row for row in candidates if row["selected"]]
    winner = min(candidates, key=lambda row: _selection_key(row, predicted_charts=predicted_charts))
    return (
        selection.get("modes") == list(head.modes)
        and selection.get("objective") == ("source_validation_predicted_chart_v2" if predicted_charts else "source_validation_accuracy_then_cross_entropy_v1")
        and selection.get("validation_used_for_fit") is False
        and selection.get("test_examples_used") == 0
        and selection.get("serving_authority") is False
        and len(selected) == 1
        and selected[0] is winner
        and selected[0].get("modes") == list(head.modes)
        and type(selected[0].get("total")) is int
        and type(selected[0].get("correct")) is int
        and 0 <= selected[0]["correct"] <= selected[0]["total"]
        and selected[0]["total"] > 0
        and isinstance(selected[0].get("cross_entropy"), (int, float))
        and np.isfinite(selected[0]["cross_entropy"])
    )


def refit_compositional_operation_views(model, examples, *, candidate_modes=None, progress=None,
                                        conditional_labels=False):
    """Fit on source train, select views and chart length on source validation."""
    from core.learning.semantic_operation_background import (
        operation_background_receipt,
        operation_background_training_spans,
        valid_background_contract,
    )
    from core.learning.semantic_program_transducer_fitting import (
        _OPERATION_CANDIDATES,
        _best_nonoverlapping_nodes,
        _calibrate_operation_charts,
        _OperationNode,
        _overlap,
    )

    if type(conditional_labels) is not bool:
        raise ValueError("conditional operation labels must be boolean")
    train = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    train_ids = [item.ir.source_text_sha256 for item in train]
    validation_ids = [item.ir.source_text_sha256 for item in validation]
    if (
        not train or not validation
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids) & set(validation_ids)
    ):
        raise ValueError("operation view refit needs unique disjoint source splits")
    if any(
        item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
        or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
        or (item.hidden_channels, item.hidden_channel_widths)
        != (model.hidden_channels, model.hidden_channel_widths)
        for item in (*train, *validation)
    ):
        raise ValueError("operation view refit source representation differs")
    modes = _OPERATION_FEATURE_MODES_V3 if "middle_causal_hidden" in model.hidden_channels else _OPERATION_FEATURE_MODES_V2
    if candidate_modes is not None:
        if (not isinstance(candidate_modes, (tuple, list)) or not candidate_modes
                or any(mode not in _OPERATION_FEATURE_MODES for mode in candidate_modes)
                or len(set(candidate_modes)) != len(candidate_modes)):
            raise ValueError("operation view candidates must be unique supported modes")
        modes = tuple(candidate_modes)
    for mode in modes:
        _operation_feature_width(mode, hidden_channels=model.hidden_channels,
                                 hidden_channel_widths=model.hidden_channel_widths)
    if not valid_background_contract(model.operation_head, model.training_receipt):
        raise ValueError("operation view refit background evidence differs")
    background = None if conditional_labels else model.training_receipt.get("operation_background_fit")
    groups = tuple((item, operation_background_training_spans(
        item, model.operation_pointer, model.max_span_tokens,
    ) if background else tuple((instruction.operation_span, instruction.op)
                               for instruction in item.ir.instructions)) for item in train)
    train_rows = tuple((item, span, label) for item, rows in groups for span, label in rows)
    validation_rows = tuple((item, instruction.operation_span, instruction.op)
                            for item in validation for instruction in item.ir.instructions)
    labels = [label for _item, _span, label in train_rows]
    targets = [label for _item, _span, label in validation_rows]
    expected_labels = set(model.operation_head.labels)
    if conditional_labels:
        expected_labels.discard(OPERATION_BACKGROUND_LABEL)
    if set(labels) != expected_labels:
        raise ValueError("operation view refit must retain the source operation vocabulary")
    if not targets or not set(targets) <= set(labels):
        raise ValueError("operation view validation has no training label support")
    geometry_counts = Counter(_geometry(item) for item in train)
    weights = _normalized_weights([1.0 / geometry_counts[_geometry(item)]
                                   / (len(rows) if background else 1)
                                   for item, rows in groups for _ in rows])
    heads, probabilities, runtime_probabilities = {}, {}, {}
    def features(rows, mode):
        return np.stack([
            _operation_feature(item.hidden_states, span, mode=mode,
                               hidden_channels=model.hidden_channels, hidden_channel_widths=model.hidden_channel_widths)
            for item, span, _label in rows
        ])

    runtime_rows, runtime_groups = [], []
    for item in validation:
        candidates = tuple(
            (span, score) for span, score in model.operation_pointer.decode_candidates(
                item.hidden_states, limit=_OPERATION_CANDIDATES, max_span_tokens=model.max_span_tokens,
            ) if not any(_overlap(span, input_span) for input_span in item.ir.input_spans)
        )
        lower = len(runtime_rows)
        runtime_rows.extend((item, span) for span, _score in candidates)
        runtime_groups.append((item, candidates, slice(lower, len(runtime_rows))))

    for mode in modes:
        if progress is not None:
            progress({"stage": "operation_view_fit", "mode": mode, "rows": len(train_rows)})
        heads[mode] = _fit_classifier(features(train_rows, mode), labels, sample_weight=weights)
        probabilities[mode] = np.stack([heads[mode].predict_probabilities(row) for row in features(validation_rows, mode)])
        runtime_probabilities[mode] = np.stack([
            heads[mode].predict_probabilities(_operation_feature(
                item.hidden_states, span, mode=mode, hidden_channels=model.hidden_channels,
                hidden_channel_widths=model.hidden_channel_widths,
            ))
            for item, span in runtime_rows
        ])
    label_indices = {label: index for index, label in enumerate(heads[modes[0]].labels)}
    operation_indices = tuple(index for label, index in label_indices.items()
                              if label != OPERATION_BACKGROUND_LABEL)
    background_index = label_indices.get(OPERATION_BACKGROUND_LABEL)
    odds = background and background["score"] == "joint_operation_background_log_odds_v2"
    expected = np.asarray([label_indices[label] for label in targets])
    candidates = []
    calibrations = {}
    for count in range(1, min(_MAX_OPERATION_VIEWS, len(modes)) + 1):
        for selected_modes in combinations(modes, count):
            probability = np.mean([probabilities[mode] for mode in selected_modes], axis=0)
            runtime_probability = np.mean([runtime_probabilities[mode] for mode in selected_modes], axis=0)
            cached = []
            for item, proposals, indices in runtime_groups:
                scores = runtime_probability[indices]
                nodes = []
                for (span, score), p in zip(proposals, scores, strict=True):
                    index = max(operation_indices, key=lambda index: p[index])
                    confidence = float(p[index])
                    node_score = math.log(max(confidence, 1e-12))
                    node_score += (-math.log(max(float(p[background_index]), 1e-12))
                                   if odds else float(score))
                    nodes.append(_OperationNode(span, heads[modes[0]].labels[index],
                                                node_score, float(score), confidence))
                cached.append((item, tuple(_best_nonoverlapping_nodes(nodes, count) for count in range(1, model.max_steps + 1))))
            penalty, length_rows = _calibrate_operation_charts(cached)
            selected_length = next(row for row in length_rows if row["length_penalty"] == penalty)
            calibrations[selected_modes] = length_rows
            candidates.append({
                "modes": list(selected_modes), "correct": int(np.sum(probability.argmax(axis=1) == expected)),
                "total": len(targets), "cross_entropy": float(-np.log(np.maximum(probability[np.arange(len(targets)), expected], 1e-12)).mean()),
                **selected_length,
            })
    winner = min(candidates, key=lambda row: _selection_key(row, predicted_charts=True))
    for row in candidates:
        row["selected"] = row is winner
    head = MultiViewClassifierHead(tuple(winner["modes"]), tuple(heads[mode] for mode in winner["modes"]))
    penalty, length_rows = winner["length_penalty"], calibrations[head.modes]
    coefficient = model._coefficient_body()
    coefficient.update(operation_head=head.to_dict(), operation_length_penalty=penalty)
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    if background:
        body["operation_background_fit"] = operation_background_receipt(
            model, head, train, train_rows, score=background["score"],
        )
    elif conditional_labels:
        # Boundary evidence remains in the pointer. The label head is trained
        # conditional on an operation span, not on incompatible background spans.
        body.pop("operation_background_fit", None)
    if body.get("operation_search_policy") == "complete_bounded_v1":
        body["operation_label_limit"] = len(head.labels)
    elif "operation_label_limit" in body:
        body["operation_label_limit"] = min(body["operation_label_limit"], len(head.labels))
    body["operation_view_selection"] = {
        "schema": "aura.semantic_operation_view_selection.v2",
        "objective": "source_validation_predicted_chart_v2",
        "operation_spans": "predicted_source_pointer_candidates",
        "input_spans": "annotated_source_validation_inputs",
        "chart_order": "source_span_order",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "modes": list(head.modes), "candidates": candidates,
        "training_examples": len(train), "validation_examples": len(validation),
        "training_example_ids_sha256": _sha(sorted(train_ids)),
        "validation_example_ids_sha256": _sha(sorted(validation_ids)),
        "training_targets_sha256": _sha([
            [item.ir.source_text_sha256, label, span.start, span.end]
            for item, span, label in train_rows
        ]),
        "validation_targets_sha256": _sha([
            [item.ir.source_text_sha256, label, span.start, span.end]
            for item, span, label in validation_rows
        ]),
        "length_calibration": length_rows,
        "label_conditioning": "operation_span" if conditional_labels else "parent_supervision",
        "boundary_score": "pointer_plus_conditional_log_probability" if conditional_labels else "parent_score",
        "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
    }
    return replace(model, operation_head=head, operation_length_penalty=penalty,
                   training_receipt={**body, "receipt_sha256": _sha(body)})
