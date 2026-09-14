"""Fit semantic span boundaries as pairs rather than independent endpoints."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_shared_transducer import _geometry, _normalized_weights
from core.learning.semantic_program_transducer import LinearPointerHead, _fit_binary_head, _sha


def paired_boundary_feature(hidden, span):
    """Match the pointer's additive boundary and diagonal interaction scores."""
    span.validate_bound(len(hidden))
    start, end = hidden[span.start], hidden[span.end - 1]
    return np.concatenate((start, end, start * end * math.sqrt(hidden.shape[1])))


def paired_boundary_training_spans(item, positives, pointer, max_span_tokens):
    """Mine crossed endpoints and local boundary errors without false negatives."""
    positives = tuple(dict.fromkeys(positives))
    scores = pointer.score_sequence(item.hidden_states)
    candidates = {span for span, _ in scores.decode_candidates(limit=16, max_span_tokens=max_span_tokens)}
    for positive in positives:
        for start_offset in (-2, -1, 0, 1, 2):
            for end_offset in (-2, -1, 0, 1, 2):
                start, end = positive.start + start_offset, positive.end + end_offset
                if 0 <= start < end <= len(item.hidden_states) and end - start <= max_span_tokens:
                    candidates.add(TokenSpan(start, end))
        for other in positives:
            if positive.start < other.end and other.end - positive.start <= max_span_tokens:
                candidates.add(TokenSpan(positive.start, other.end))
    candidates.difference_update(positives)
    negatives = sorted(candidates, key=lambda span: (-scores.score_span(span), span.start, span.end))[:4 * len(positives)]
    return tuple((span, 1) for span in positives) + tuple((span, 0) for span in negatives)


def fit_paired_boundary_pointer(training, *, spans, pointer, max_span_tokens):
    """Fit one shared role from source training examples only."""
    counts = Counter(_geometry(item) for item in training)
    rows = [
        (item, paired_boundary_training_spans(item, spans(item), pointer, max_span_tokens))
        for item in training
    ]
    size = sum(len(pairs) for _, pairs in rows)
    if not size:
        raise ValueError("paired pointer has no training spans")
    features = np.empty((size, 3 * pointer.width), dtype=np.float32)
    labels, weights = np.empty(size, dtype=np.int8), np.empty(size, dtype=np.float64)
    index = 0
    for item, pairs in rows:
        for span, label in pairs:
            features[index] = paired_boundary_feature(item.hidden_states, span)
            labels[index] = label
            weights[index] = 1.0 / counts[_geometry(item)] / len(pairs)
            index += 1
    weight, bias = _fit_binary_head(features, labels, sample_weight=_normalized_weights(weights), max_iter=400, tolerance=1e-4)
    start, end, pair = np.split(weight, 3)
    return LinearPointerHead(start, bias / 2, end, bias / 2, pair), {
        "rows": size, "positive_spans": int(labels.sum()), "negative_spans": int(size - labels.sum()),
        "targets_sha256": _sha([
            [item.ir.source_text_sha256, [[span.start, span.end, label] for span, label in pairs]]
            for item, pairs in rows
        ]),
    }


def refit_compositional_paired_operation_pointer(model, examples):
    """Replace the operation pointer; calibrate length without fitting validation."""
    from core.learning.semantic_program_transducer_fitting import _select_operation_length_penalty

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    ids = [[item.ir.source_text_sha256 for item in split] for split in (training, validation)]
    if (
        not all(ids) or any(len(set(values)) != len(values) for values in ids)
        or set(ids[0]) & set(ids[1])
    ):
        raise ValueError("paired operation refit needs unique disjoint source splits")
    if any(
        item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
        or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
        or (item.hidden_channels, item.hidden_channel_widths) != (model.hidden_channels, model.hidden_channel_widths)
        for item in (*training, *validation)
    ):
        raise ValueError("paired operation refit source representation differs")
    pointer, supervision = fit_paired_boundary_pointer(
        training, spans=lambda item: tuple(i.operation_span for i in item.ir.instructions),
        pointer=model.operation_pointer, max_span_tokens=model.max_span_tokens,
    )
    penalty, calibration = _select_operation_length_penalty(
        validation, pointer=pointer, classifier=model.operation_head, max_steps=model.max_steps,
        max_span_tokens=model.max_span_tokens, hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    coefficient = model._coefficient_body()
    coefficient.update(operation_pointer=pointer.to_dict(), operation_length_penalty=penalty)
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["paired_operation_pointer_refit"] = {
        "schema": "aura.semantic_paired_operation_pointer_refit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "feature": "start_end_sqrt_width_diagonal_product_v1",
        "training_examples": len(training), "validation_examples": len(validation),
        "training_example_ids_sha256": _sha(sorted(ids[0])),
        "validation_example_ids_sha256": _sha(sorted(ids[1])),
        "validation_targets_sha256": _sha([
            [item.ir.source_text_sha256, [[i.op, i.operation_span.start, i.operation_span.end] for i in item.ir.instructions]]
            for item in validation
        ]),
        "supervision": supervision, "length_calibration": calibration,
        "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
    }
    return replace(model, operation_pointer=pointer, operation_length_penalty=penalty,
                   training_receipt={**body, "receipt_sha256": _sha(body)})
