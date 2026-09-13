"""The directed link from an argument mention to its definition.

Two pieces that belong together and to nothing else: the head that scores one
directed edge, and the low-rank tissue trained on top of it when the linear
score alone cannot separate the candidates.

They came out of the fitting module because that module had grown past the
2000-line ceiling, and a ceiling a new module is never grandfathered past. This
is the part of it that reads as one subject.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_shared_transducer import (
    _normalized_weights,
    _relation_span_vector,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    _fit_binary_head,
)

#: Product, absolute difference, signed difference. Three channels per pair, so
#: a weight vector is always a multiple of three and the width is recoverable
#: from it. Defined here because this is where the feature is built; the
#: fitting module reads it back for the argument-role head, which uses the same
#: feature.
_DIRECTIONAL_RELATION_PARTS: Final = 3


def _directional_relation_feature(
    reference: np.ndarray,
    definition: np.ndarray,
) -> np.ndarray:
    """Preserve pair similarity while making semantic edge direction observable."""

    if reference.shape != definition.shape or reference.ndim != 1:
        raise ValueError("compositional relation vectors differ")
    return np.concatenate(
        (
            reference * definition,
            np.abs(reference - definition),
            reference - definition,
        )
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class DirectionalRelationHead:
    """One directed linker from an argument mention to its definition."""

    weight: np.ndarray
    bias: float
    pointer_scale: float
    query_projection: np.ndarray
    definition_projection: np.ndarray

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        query_projection = np.asarray(self.query_projection, dtype=np.float32)
        definition_projection = np.asarray(
            self.definition_projection,
            dtype=np.float32,
        )
        channel_width = weight.size // _DIRECTIONAL_RELATION_PARTS
        if (
            weight.size < _DIRECTIONAL_RELATION_PARTS
            or weight.size % _DIRECTIONAL_RELATION_PARTS
            or not np.all(np.isfinite(weight))
            or not np.isfinite(self.bias)
            or not np.isfinite(self.pointer_scale)
            or self.pointer_scale < 0.0
            or query_projection.ndim != 2
            or definition_projection.shape != query_projection.shape
            or query_projection.shape[0] != channel_width
            or not 1 <= query_projection.shape[1] <= 64
            or not np.all(np.isfinite(query_projection))
            or not np.all(np.isfinite(definition_projection))
        ):
            raise ValueError("compositional directional relation head is invalid")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "query_projection", query_projection)
        object.__setattr__(self, "definition_projection", definition_projection)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def base_score(self, reference: np.ndarray, definition: np.ndarray) -> float:
        feature = _directional_relation_feature(reference, definition)
        if feature.shape != self.weight.shape:
            raise ValueError("compositional directional relation width differs")
        return float(feature @ self.weight + self.bias)

    def tissue_score(self, reference: np.ndarray, definition: np.ndarray) -> float:
        if reference.shape != definition.shape or reference.ndim != 1:
            raise ValueError("compositional relation vectors differ")
        return float(
            (reference @ self.query_projection) @ (definition @ self.definition_projection)
        )

    def score(self, reference: np.ndarray, definition: np.ndarray) -> float:
        return self.base_score(reference, definition) + self.tissue_score(
            reference,
            definition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.tolist(),
            "bias": float(self.bias),
            "pointer_scale": float(self.pointer_scale),
            "query_projection": self.query_projection.tolist(),
            "definition_projection": self.definition_projection.tolist(),
        }


def _fit_directional_relation_head(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    item_weights: Mapping[int, float],
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[np.ndarray, float]:
    from core.learning.semantic_program_transducer_fitting import (
        _register_definition_spans,
    )

    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for item in training:
        definitions = _register_definition_spans(item)
        for step, instruction in enumerate(item.ir.instructions):
            available = definitions[: item.ir.n_inputs + step]
            for reference_span, register in zip(
                instruction.argument_spans,
                instruction.args,
                strict=True,
            ):
                reference = _relation_span_vector(
                    item.hidden_states,
                    reference_span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                for candidate_register, definition_span in enumerate(available):
                    definition = _relation_span_vector(
                        item.hidden_states,
                        definition_span,
                        hidden_channels=hidden_channels,
                        hidden_channel_widths=hidden_channel_widths,
                    )
                    features.append(_directional_relation_feature(reference, definition))
                    labels.append(int(candidate_register == register))
                    weights.append(item_weights[id(item)] / len(available))
    return _fit_binary_head(
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        sample_weight=_normalized_weights(weights),
        max_iter=400,
        tolerance=1e-5,
    )


@dataclass(frozen=True, slots=True)
class _RelationDecisionBatch:
    references: np.ndarray
    definitions: np.ndarray
    mask: np.ndarray
    targets: np.ndarray
    base_logits: np.ndarray


def _relation_decision_batch(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    relation_weight: np.ndarray,
    relation_bias: float,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> _RelationDecisionBatch:
    from core.learning.semantic_program_transducer_fitting import _register_definition_spans

    decisions: list[tuple[np.ndarray, tuple[np.ndarray, ...], int]] = []
    max_candidates = 0
    for item in examples:
        definitions = _register_definition_spans(item)
        for step, instruction in enumerate(item.ir.instructions):
            available = definitions[: item.ir.n_inputs + step]
            definition_vectors = tuple(
                _relation_span_vector(
                    item.hidden_states,
                    span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                ).astype(np.float32)
                for span in available
            )
            max_candidates = max(max_candidates, len(definition_vectors))
            for reference_span, register in zip(
                instruction.argument_spans,
                instruction.args,
                strict=True,
            ):
                decisions.append(
                    (
                        _relation_span_vector(
                            item.hidden_states,
                            reference_span,
                            hidden_channels=hidden_channels,
                            hidden_channel_widths=hidden_channel_widths,
                        ).astype(np.float32),
                        definition_vectors,
                        register,
                    )
                )
    if not decisions or max_candidates < 1:
        raise ValueError("compositional relation tissue has no decisions")
    width = decisions[0][0].size
    references = np.zeros((len(decisions), width), dtype=np.float32)
    definitions = np.zeros(
        (len(decisions), max_candidates, width),
        dtype=np.float32,
    )
    mask = np.zeros((len(decisions), max_candidates), dtype=bool)
    targets = np.zeros(len(decisions), dtype=np.int64)
    base_logits = np.full(
        (len(decisions), max_candidates),
        -1e9,
        dtype=np.float32,
    )
    for row, (reference, candidates, target) in enumerate(decisions):
        if not 0 <= target < len(candidates):
            raise ValueError("compositional relation target is unavailable")
        references[row] = reference
        targets[row] = target
        for column, definition in enumerate(candidates):
            definitions[row, column] = definition
            mask[row, column] = True
            base_logits[row, column] = float(
                _directional_relation_feature(reference, definition) @ relation_weight
                + relation_bias
            )
    return _RelationDecisionBatch(
        references=references,
        definitions=definitions,
        mask=mask,
        targets=targets,
        base_logits=base_logits,
    )


def _relation_tissue_logits(
    batch: _RelationDecisionBatch,
    query_projection: np.ndarray,
    definition_projection: np.ndarray,
) -> np.ndarray:
    queries = batch.references @ query_projection
    definitions = np.einsum(
        "ncd,dr->ncr",
        batch.definitions,
        definition_projection,
        optimize=True,
    )
    interactions = np.einsum("nr,ncr->nc", queries, definitions, optimize=True)
    return np.where(batch.mask, batch.base_logits + interactions, -1e9)


def _relation_tissue_metrics(
    batch: _RelationDecisionBatch,
    query_projection: np.ndarray,
    definition_projection: np.ndarray,
) -> tuple[int, float]:
    logits = _relation_tissue_logits(batch, query_projection, definition_projection)
    centered = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
    rows = np.arange(batch.targets.size)
    return (
        int(np.count_nonzero(logits.argmax(axis=1) == batch.targets)),
        float(-log_probabilities[rows, batch.targets].mean()),
    )


def _fit_low_rank_relation_tissue(
    training: Sequence[SemanticTransducerTrainingExample],
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    relation_weight: np.ndarray,
    relation_bias: float,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Fit one cross-feature linker and select its epoch on source validation."""
    from core.learning.semantic_program_transducer_fitting import (
        _RELATION_TISSUE_BATCH_SIZE,
        _RELATION_TISSUE_EPOCHS,
        _RELATION_TISSUE_GRADIENT_CLIP,
        _RELATION_TISSUE_LEARNING_RATE,
        _RELATION_TISSUE_RANK,
        _RELATION_TISSUE_SEED,
        _RELATION_TISSUE_SELECTION_INTERVAL,
        _RELATION_TISSUE_WEIGHT_DECAY,
    )

    train = _relation_decision_batch(
        training,
        relation_weight=relation_weight,
        relation_bias=relation_bias,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    validate = _relation_decision_batch(
        validation,
        relation_weight=relation_weight,
        relation_bias=relation_bias,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    width = train.references.shape[1]
    rank = min(_RELATION_TISSUE_RANK, width)
    rng = np.random.default_rng(_RELATION_TISSUE_SEED)
    query = (rng.standard_normal((width, rank)) * 0.002).astype(np.float32)
    definition = (rng.standard_normal((width, rank)) * 0.002).astype(np.float32)
    query_moment = np.zeros_like(query)
    query_variance = np.zeros_like(query)
    definition_moment = np.zeros_like(definition)
    definition_variance = np.zeros_like(definition)
    zero = np.zeros((width, rank), dtype=np.float32)
    baseline_correct, baseline_loss = _relation_tissue_metrics(validate, zero, zero)
    rows = [
        {
            "epoch": 0,
            "validation_top1": baseline_correct,
            "validation_total": validate.targets.size,
            "validation_cross_entropy": baseline_loss,
        }
    ]
    best_key = (-baseline_loss, baseline_correct, 0)
    best_query = zero.copy()
    best_definition = zero.copy()
    best_epoch = 0
    update = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for epoch in range(1, _RELATION_TISSUE_EPOCHS + 1):
        permutation = rng.permutation(train.targets.size)
        for start in range(0, permutation.size, _RELATION_TISSUE_BATCH_SIZE):
            indices = permutation[start : start + _RELATION_TISSUE_BATCH_SIZE]
            batch = _RelationDecisionBatch(
                references=train.references[indices],
                definitions=train.definitions[indices],
                mask=train.mask[indices],
                targets=train.targets[indices],
                base_logits=train.base_logits[indices],
            )
            queries = batch.references @ query
            definition_keys = np.einsum(
                "ncd,dr->ncr",
                batch.definitions,
                definition,
                optimize=True,
            )
            logits = np.where(
                batch.mask,
                batch.base_logits
                + np.einsum("nr,ncr->nc", queries, definition_keys, optimize=True),
                -1e9,
            )
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            probabilities[np.arange(batch.targets.size), batch.targets] -= 1.0
            probabilities /= batch.targets.size
            query_gradient = batch.references.T @ np.einsum(
                "nc,ncr->nr",
                probabilities,
                definition_keys,
                optimize=True,
            )
            definition_gradient = np.einsum(
                "ncd,ncr->dr",
                batch.definitions,
                probabilities[:, :, None] * queries[:, None, :],
                optimize=True,
            )
            gradient_norm = float(
                np.sqrt(
                    np.sum(query_gradient * query_gradient)
                    + np.sum(definition_gradient * definition_gradient)
                )
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("compositional relation tissue gradient is non-finite")
            if gradient_norm > _RELATION_TISSUE_GRADIENT_CLIP:
                scale = _RELATION_TISSUE_GRADIENT_CLIP / gradient_norm
                query_gradient *= scale
                definition_gradient *= scale
            update += 1
            for parameter, gradient, moment, variance in (
                (query, query_gradient, query_moment, query_variance),
                (
                    definition,
                    definition_gradient,
                    definition_moment,
                    definition_variance,
                ),
            ):
                moment *= beta1
                moment += (1.0 - beta1) * gradient
                variance *= beta2
                variance += (1.0 - beta2) * gradient * gradient
                parameter *= 1.0 - (_RELATION_TISSUE_LEARNING_RATE * _RELATION_TISSUE_WEIGHT_DECAY)
                parameter -= (
                    _RELATION_TISSUE_LEARNING_RATE
                    * (moment / (1.0 - beta1**update))
                    / (np.sqrt(variance / (1.0 - beta2**update)) + epsilon)
                )
        if epoch % _RELATION_TISSUE_SELECTION_INTERVAL:
            continue
        correct, loss = _relation_tissue_metrics(validate, query, definition)
        rows.append(
            {
                "epoch": epoch,
                "validation_top1": correct,
                "validation_total": validate.targets.size,
                "validation_cross_entropy": loss,
            }
        )
        # Cross-entropy is the calibrated validation objective. Prioritising a
        # two-decision top-1 increase selected a later, less calibrated epoch
        # and measurably regressed source constructions.
        key = (-loss, correct, -epoch)
        if key > best_key:
            best_key = key
            best_query = query.copy()
            best_definition = definition.copy()
            best_epoch = epoch
    for row in rows:
        row["selected"] = row["epoch"] == best_epoch
    return best_query, best_definition, rows
