"""Learn operation meaning against source spans that denote no operation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from core.learning.semantic_paired_pointer_refit import paired_boundary_training_spans
from core.learning.semantic_program_shared_transducer import _geometry, _normalized_weights
from core.learning.semantic_program_transducer import (
    OPERATION_BACKGROUND_LABEL,
    MultiViewClassifierHead,
    _fit_classifier,
    _operation_feature,
    _sha,
)


def valid_background_contract(head, receipt):
    """Bind background competition to its training record, not a class name alone."""
    record = receipt.get("operation_background_fit")
    if OPERATION_BACKGROUND_LABEL not in head.labels:
        return record is None
    if not isinstance(record, Mapping):
        return False
    return (
        record.get("schema") == "aura.semantic_operation_background_fit.v1"
        and record.get("background_label") == OPERATION_BACKGROUND_LABEL
        and record.get("modes") == list(head.modes)
        and record.get("labels") == list(head.labels)
        and record.get("score") in {"pointer_plus_log_joint_operation_probability_v1", "joint_operation_background_log_odds_v2"}
        and record.get("validation_used_for_fit") is False
        and record.get("test_examples_used") == 0
        and record.get("serving_authority") is False
        and all(type(record.get(key)) is int and record[key] > 0
                for key in ("training_examples", "positive_spans", "background_spans"))
        and all(isinstance(record.get(key), str) and len(record[key]) == 64
                and set(record[key]) <= set("0123456789abcdef")
                for key in ("parent_transducer_receipt_sha256", "training_ids_sha256", "targets_sha256"))
    )


def operation_background_training_spans(item, pointer, max_span_tokens):
    """Mine source-local boundary and pointer errors; keep every gold operation."""
    labels = {}
    for instruction in item.ir.instructions:
        span = instruction.operation_span
        if span in labels and labels[span] != instruction.op:
            raise ValueError("one source operation span has conflicting labels")
        labels[span] = instruction.op
    pairs = paired_boundary_training_spans(item, tuple(labels), pointer, max_span_tokens)
    rows = []
    for span, positive in pairs:
        if not positive and any(span.start < direct.end and direct.start < span.end
                                for direct in item.ir.input_spans):
            continue
        rows.append((span, labels[span] if positive else OPERATION_BACKGROUND_LABEL))
    return tuple(rows)


def operation_background_receipt(model, head, train, rows, *, score):
    """Bind a fitted background head to the actual source span supervision."""
    labels = [label for _, _, label in rows]
    return {
        "schema": "aura.semantic_operation_background_fit.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "background_label": OPERATION_BACKGROUND_LABEL,
        "modes": list(head.modes), "labels": list(head.labels), "score": score,
        "training_examples": len(train),
        "training_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in train)),
        "positive_spans": sum(label != OPERATION_BACKGROUND_LABEL for label in labels),
        "background_spans": labels.count(OPERATION_BACKGROUND_LABEL),
        "targets_sha256": _sha([[item.ir.source_text_sha256, span.start, span.end, label]
                                for item, span, label in rows]),
        "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
    }


def refit_compositional_operation_background(model, examples, *, progress=None, background_log_odds=False):
    """Fit one shared head on source training spans, including non-operation spans."""
    if type(background_log_odds) is not bool:
        raise ValueError("background odds option must be boolean")
    train = tuple(item for item in examples if item.split == "train")
    held = tuple(item for item in examples if item.split != "train")
    ids = [item.ir.source_text_sha256 for item in train]
    if not ids or len(set(ids)) != len(ids) or set(ids) & {item.ir.source_text_sha256 for item in held}:
        raise ValueError("background fitting needs unique disjoint source training")
    if any(item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
           or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
           or (item.hidden_channels, item.hidden_channel_widths)
           != (model.hidden_channels, model.hidden_channel_widths) for item in train):
        raise ValueError("background fitting source representation differs")
    groups = [(item, operation_background_training_spans(item, model.operation_pointer, model.max_span_tokens))
              for item in train]
    rows = [(item, span, label) for item, spans in groups for span, label in spans]
    labels = [label for _, _, label in rows]
    counts = Counter(_geometry(item) for item in train)
    weights = _normalized_weights([1.0 / counts[_geometry(item)] / len(spans)
                                   for item, spans in groups for _ in spans])
    if OPERATION_BACKGROUND_LABEL not in labels:
        raise ValueError("background fitting has no non-operation supervision")
    if set(labels) - {OPERATION_BACKGROUND_LABEL} != set(model.operation_head.labels) - {OPERATION_BACKGROUND_LABEL}:
        raise ValueError("background fitting must retain the source operation vocabulary")
    heads = []
    for mode in model.operation_head.modes:
        if progress is not None:
            progress({"stage": "operation_background_fit", "mode": mode, "rows": len(rows)})
        features = np.stack([_operation_feature(
            item.hidden_states, span, mode=mode, hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        ) for item, span, _ in rows])
        heads.append(_fit_classifier(features, labels, sample_weight=weights))
    head = MultiViewClassifierHead(model.operation_head.modes, tuple(heads))
    coefficient = model._coefficient_body()
    coefficient["operation_head"] = head.to_dict()
    penalty = 0.0 if background_log_odds else model.operation_length_penalty
    coefficient["operation_length_penalty"] = penalty
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["operation_background_fit"] = operation_background_receipt(
        model, head, train, rows,
        score="joint_operation_background_log_odds_v2" if background_log_odds
        else "pointer_plus_log_joint_operation_probability_v1",
    )
    if body.get("operation_search_policy") == "complete_bounded_v1":
        body["operation_label_limit"] = len(head.labels)
    return replace(model, operation_head=head, operation_length_penalty=penalty,
                   training_receipt={**body, "receipt_sha256": _sha(body)})
