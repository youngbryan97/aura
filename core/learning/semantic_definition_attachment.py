"""Learn which register owns a proposed definition in the shared semantic graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import logging
import math

import numpy as np

from core.learning.semantic_program_shared_transducer import (
    _geometry,
    _normalized_weights,
    _relation_span_vector,
)
from core.learning.semantic_program_transducer import _fit_binary_head, _sha
from core.learning.semantic_relation_tissue import _directional_relation_feature
from typing import Any

logger = logging.getLogger(__name__)


def valid_attachment_contract(receipt: Any) -> bool:
    fit = receipt.get("definition_attachment_fit")
    return (
        isinstance(fit, Mapping)
        and fit.get("schema") == "aura.semantic_definition_attachment.v1"
        and fit.get("objective")
        in {"binary_source_anchor_definition_v1", "conditional_anchor_owner_v2"}
        and fit.get("validation_used_for_fit") is False
        and fit.get("test_examples_used") == 0
        and fit.get("serving_authority") is False
        and fit.get("attachment_counting") == "once_per_used_definition"
        and type(fit.get("training_examples")) is int
        and fit["training_examples"] > 0
        and type(fit.get("training_rows")) is int
        and type(fit.get("positive_rows")) is int
        and 0 < fit["positive_rows"] < fit["training_rows"]
    )


def definition_proposal_pool(
    anchors: tuple[Any, ...],
    hidden: Any,
    pointer: Any,
    *,
    max_span_tokens: Any,
) -> tuple[Any, ...]:
    """One answer-blind proposal bank shared by all register anchors."""
    return tuple(
        dict.fromkeys(
            (
                *anchors,
                *(
                    span
                    for span, _score in pointer.decode_candidates(
                        hidden,
                        max_span_tokens=max_span_tokens,
                        limit=128,
                    )
                ),
            )
        )
    )


def _owner_scores(
    head: Any,
    vectors: dict[str, Any],
    anchors: tuple[Any, ...],
    *,
    conditional: bool,
) -> dict[str, Any]:
    scores = {}
    for span, vector in vectors.items():
        logits = [head.score(vector, vectors[anchor]) for anchor in anchors]
        peak = max(logits)
        normalizer = peak + math.log(sum(math.exp(value - peak) for value in logits))
        for register, logit in enumerate(logits):
            scores[register, span] = logit - normalizer if conditional else logit
    return scores


def attachment_hypotheses(model: Any, hidden: Any, anchors: Any, local_candidates: Any) -> Any:
    """Retain local hypotheses and add names selected by the learned attachment."""
    head = model.definition_attachment_head
    if head is None:
        raise ValueError("definition attachment needs a trained head")
    pool = definition_proposal_pool(
        anchors, hidden, model.definition_pointer, max_span_tokens=model.max_definition_span_tokens
    )

    def vector(span: Any) -> Any:
        return _relation_span_vector(
            hidden,
            span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )

    vectors = {
        span: vector(span)
        for span in set((*pool, *(span for row in local_candidates for span in row)))
    }
    owner_scores = _owner_scores(
        head,
        vectors,
        anchors,
        conditional=(
            model.training_receipt["definition_attachment_fit"]["objective"]
            == "conditional_anchor_owner_v2"
        ),
    )
    pointer = model.definition_pointer.score_sequence(hidden)
    hypotheses, scores = [], {}
    for register, (anchor, local) in enumerate(zip(anchors, local_candidates, strict=True)):
        ranked = sorted(
            pool,
            key=lambda span: (
                -owner_scores[register, span],
                span.end - span.start,
                span.start,
                span.end,
            ),
        )[:4]
        retained = sorted(
            local,
            key=lambda span: (
                -pointer.score_span(span),
                span.end - span.start,
                span.start,
                span.end,
            ),
        )[:4]
        for span in dict.fromkeys((anchor, *retained, *ranked)):
            hypotheses.append((register, span))
            scores[register, span] = owner_scores[register, span]
    return tuple(hypotheses), scores


def refit_definition_attachment(
    model: Any,
    examples: Any,
    *,
    objective: str='conditional_anchor_owner_v2',
) -> Any:
    """Fit attachment on training labels; report validation without fitting it."""
    from core.learning.semantic_program_transducer_fitting import (
        LinearArgumentRoleHead,
        _register_definition_spans,
    )

    if objective not in {"binary_source_anchor_definition_v1", "conditional_anchor_owner_v2"}:
        raise ValueError("unsupported definition attachment objective")

    train = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    train_ids = [item.ir.source_text_sha256 for item in train]
    validation_ids = [item.ir.source_text_sha256 for item in validation]
    if (
        not train
        or not validation
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids) & set(validation_ids)
    ):
        raise ValueError("attachment refit needs unique disjoint source splits")
    if any(
        item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
        or item.tokenizer_identity_sha256 != model.input_grounding.tokenizer_identity_sha256
        or (item.hidden_channels, item.hidden_channel_widths)
        != (model.hidden_channels, model.hidden_channel_widths)
        for item in (*train, *validation)
    ):
        raise ValueError("attachment refit source representation differs")

    geometries = Counter(_geometry(item) for item in train)
    features, labels, weights = [], [], []

    def vector(item: Any, span: Any) -> Any:
        return _relation_span_vector(
            item.hidden_states,
            span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )

    for item in train:
        anchors = (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
        )
        targets = _register_definition_spans(item)
        if objective == "conditional_anchor_owner_v2":
            if len(set(targets)) != len(targets):
                raise ValueError("one annotated definition has multiple register owners")
            for owner, target in enumerate(targets):
                for register, anchor in enumerate(anchors):
                    features.append(
                        _directional_relation_feature(vector(item, target), vector(item, anchor))
                    )
                    labels.append(int(register == owner))
                    weights.append(1.0 / geometries[_geometry(item)] / len(anchors) ** 2)
        else:
            proposals = definition_proposal_pool(
                anchors,
                item.hidden_states,
                model.definition_pointer,
                max_span_tokens=model.max_definition_span_tokens,
            )
            for anchor, target in zip(anchors, targets, strict=True):
                negatives = tuple(
                    span
                    for span in dict.fromkeys((*targets, *anchors, *proposals[:24]))
                    if span != target
                )
                reference = vector(item, anchor)
                for span in (target, *negatives):
                    features.append(_directional_relation_feature(vector(item, span), reference))
                    labels.append(int(span == target))
                    weights.append(
                        1.0 / geometries[_geometry(item)] / len(anchors) / (1 + len(negatives))
                    )
    weight, bias = _fit_binary_head(
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        sample_weight=_normalized_weights(weights),
        max_iter=400,
        tolerance=1e-5,
        solver="lbfgs",
    )
    logger.info("Definition attachment fit completed: %d rows", len(labels))
    head = LinearArgumentRoleHead(weight, bias)
    rows = []
    for item in validation:
        anchors = (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
        )
        pool = definition_proposal_pool(
            anchors,
            item.hidden_states,
            model.definition_pointer,
            max_span_tokens=model.max_definition_span_tokens,
        )
        vectors = {span: vector(item, span) for span in pool}
        scores = _owner_scores(
            head, vectors, anchors, conditional=objective == "conditional_anchor_owner_v2"
        )
        for register, (anchor, target) in enumerate(
            zip(anchors, _register_definition_spans(item), strict=True)
        ):
            ranked = sorted(
                pool,
                key=lambda span: (
                    -scores[register, span],
                    span.end - span.start,
                    span.start,
                    span.end,
                ),
            )
            rows.append(
                {
                    "source": item.ir.source_text_sha256,
                    "target": target.to_dict(),
                    "covered": target in pool,
                    "top1": ranked[0] == target,
                    "top4": target in ranked[:4],
                }
            )
    coefficient = model._coefficient_body()
    coefficient["definition_attachment_head"] = head.to_dict()
    body = {key: value for key, value in model.training_receipt.items() if key != "receipt_sha256"}
    body.update(
        {
            "coefficient_sha256": _sha(coefficient),
            "argument_search_strategy": "global_constraint_v1",
            "definition_selection_policy": "joint_graph_v1",
        }
    )
    body["definition_attachment_fit"] = {
        "schema": "aura.semantic_definition_attachment.v1",
        "objective": objective,
        "optimizer": "dense_lbfgs_regularized_bias_v1",
        "max_iterations": 400,
        "tolerance": 1e-5,
        "parent_receipt": model.receipt_sha256,
        "training_examples": len(train),
        "training_rows": len(labels),
        "positive_rows": sum(labels),
        "training_source_ids_sha256": _sha(sorted(train_ids)),
        "validation_source_ids_sha256": _sha(sorted(validation_ids)),
        "training_definition_targets_sha256": _sha(
            [
                {
                    "source": item.ir.source_text_sha256,
                    "origin": item.register_definition_origin,
                    "spans": [span.to_dict() for span in _register_definition_spans(item)],
                }
                for item in train
            ]
        ),
        "validation": rows,
        "validation_used_for_fit": False,
        "test_examples_used": 0,
        "serving_authority": False,
        "attachment_counting": "once_per_used_definition",
    }
    return replace(
        model,
        definition_attachment_head=head,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )
