"""Source-trained operation/mention/definition evidence for proposal graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_shared_transducer import (
    _geometry,
    _normalized_weights,
    _relation_span_vector,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    _fit_binary_head,
)
from core.learning.semantic_relation_tissue import (
    _DIRECTIONAL_RELATION_PARTS,
    _directional_relation_feature,
    _fit_low_rank_relation_tissue,
)


def triadic_binding_feature(operation: np.ndarray, mention: np.ndarray,
                            definition: np.ndarray) -> np.ndarray:
    """Retain each occurrence while letting the operation condition their link."""
    if (operation.ndim != 1 or mention.shape != operation.shape
            or definition.shape != operation.shape):
        raise ValueError("triadic source spans differ in width")
    return _directional_relation_feature(mention * operation, definition)


def joint_source_binding_feature(
    operation: np.ndarray, mention: np.ndarray, definition: np.ndarray, *,
    operation_span: TokenSpan, mention_span: TokenSpan,
    definition_span: TokenSpan, token_count: int,
) -> np.ndarray:
    """Preserve three distinct occurrences and their relative source geometry."""
    if (operation.ndim != 1 or mention.shape != operation.shape
            or definition.shape != operation.shape):
        raise ValueError("joint source spans differ in width")
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("joint source token count is invalid")
    for span in (operation_span, mention_span, definition_span):
        span.validate_bound(token_count)
    width = float(token_count)
    geometry = np.asarray((
        (mention_span.start - operation_span.start) / width,
        (definition_span.start - operation_span.start) / width,
        (mention_span.start - definition_span.start) / width,
        (operation_span.end - operation_span.start) / width,
        (mention_span.end - mention_span.start) / width,
        (definition_span.end - definition_span.start) / width,
        float(mention_span == definition_span),
        float(mention_span == operation_span),
    ), dtype=np.float32)
    return np.concatenate((operation, mention, definition,
                           operation * mention, operation * definition,
                           mention * definition, operation * mention * definition,
                           mention - definition, geometry))


def joint_representation_binding_feature(operation: np.ndarray,
                                         mention: np.ndarray,
                                         definition: np.ndarray) -> np.ndarray:
    """Use the same joint vector basis without token-position shortcuts."""
    if (operation.ndim != 1 or mention.shape != operation.shape
            or definition.shape != operation.shape):
        raise ValueError("joint representation spans differ in width")
    return np.concatenate((operation, mention, definition,
                           operation * mention, operation * definition,
                           mention * definition, operation * mention * definition,
                           mention - definition))


def projected_joint_binding_feature(
    operation: np.ndarray, mention: np.ndarray, definition: np.ndarray,
    query_projection: np.ndarray, definition_projection: np.ndarray,
) -> np.ndarray:
    """Use a source-trained, low-rank relation basis for all three roles."""
    if (operation.ndim != 1 or mention.shape != operation.shape
            or definition.shape != operation.shape
            or query_projection.ndim != 2
            or definition_projection.shape != query_projection.shape
            or query_projection.shape[0] != operation.size):
        raise ValueError("projected triadic source or basis width differs")
    return joint_representation_binding_feature(
        operation @ query_projection,
        mention @ query_projection,
        definition @ definition_projection,
    )


def fit_source_triadic_projection(
    training: Sequence[SemanticTransducerTrainingExample],
    calibration: Sequence[SemanticTransducerTrainingExample],
    *,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Fit the relation basis without inherited coefficients or held sources."""
    training, calibration = tuple(training), tuple(calibration)
    train_ids = {item.ir.source_text_sha256 for item in training}
    calibration_ids = {item.ir.source_text_sha256 for item in calibration}
    if (not training or not calibration or train_ids & calibration_ids
            or any(item.split != "train" for item in training)
            or any(item.split not in {"train", "validation"} for item in calibration)
            or "middle_causal_hidden" not in hidden_channels):
        raise ValueError("triadic projection requires disjoint source fit and calibration")
    width = hidden_channel_widths[hidden_channels.index("middle_causal_hidden")]
    query, definition, epochs = _fit_low_rank_relation_tissue(
        training, calibration,
        relation_weight=np.zeros(width * _DIRECTIONAL_RELATION_PARTS, dtype=np.float32),
        relation_bias=0.0,
        hidden_channels=hidden_channels, hidden_channel_widths=hidden_channel_widths,
    )
    return (query, definition), {
        "algorithm": "source_fold_low_rank_relation_v1",
        "rank": int(query.shape[1]),
        "training_sources": len(train_ids),
        "calibration_sources": len(calibration_ids),
        "validation_epochs": epochs,
        "inherited_relation_score": False,
    }


@dataclass(frozen=True, slots=True)
class TriadicBindingHead:
    weight: np.ndarray
    bias: float
    feature_schema: str = "triple_product_v1"
    query_projection: np.ndarray | None = None
    definition_projection: np.ndarray | None = None

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        valid_width = (
            weight.size >= _DIRECTIONAL_RELATION_PARTS
            and weight.size % _DIRECTIONAL_RELATION_PARTS == 0
        ) if self.feature_schema == "triple_product_v1" else (
            weight.size > 8 and (weight.size - 8) % 8 == 0
        ) if self.feature_schema == "joint_source_v2" else (
            weight.size >= 8 and weight.size % 8 == 0
        ) if self.feature_schema == "joint_representation_v3" else False
        if self.feature_schema == "projected_joint_v4":
            query = np.asarray(self.query_projection, dtype=np.float32)
            definition = np.asarray(self.definition_projection, dtype=np.float32)
            valid_width = (weight.size >= 8 and weight.size % 8 == 0
                           and query.ndim == 2 and query.shape[0] >= 1
                           and 1 <= query.shape[1] <= 64
                           and query.shape[1] == weight.size // 8
                           and definition.shape == query.shape
                           and np.all(np.isfinite(query))
                           and np.all(np.isfinite(definition)))
            if valid_width:
                object.__setattr__(self, "query_projection", query)
                object.__setattr__(self, "definition_projection", definition)
        elif self.query_projection is not None or self.definition_projection is not None:
            valid_width = False
        if (not valid_width
                or not np.all(np.isfinite(weight)) or not np.isfinite(self.bias)):
            raise ValueError("triadic binding head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        if self.feature_schema == "projected_joint_v4":
            return int(self.query_projection.shape[0])
        if self.feature_schema == "joint_source_v2":
            return int((self.weight.size - 8) // 8)
        if self.feature_schema == "joint_representation_v3":
            return int(self.weight.size // 8)
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def score(self, operation: np.ndarray, mention: np.ndarray,
              definition: np.ndarray, *, operation_span: TokenSpan | None = None,
              mention_span: TokenSpan | None = None,
              definition_span: TokenSpan | None = None,
              token_count: int | None = None) -> float:
        if self.feature_schema == "joint_source_v2":
            if (operation_span is None or mention_span is None
                    or definition_span is None or token_count is None):
                raise ValueError("joint source binding needs all source spans")
            feature = joint_source_binding_feature(operation, mention, definition,
                operation_span=operation_span, mention_span=mention_span,
                definition_span=definition_span, token_count=token_count)
        elif self.feature_schema == "joint_representation_v3":
            feature = joint_representation_binding_feature(operation, mention, definition)
        elif self.feature_schema == "projected_joint_v4":
            feature = projected_joint_binding_feature(operation, mention, definition,
                self.query_projection, self.definition_projection)
        else:
            feature = triadic_binding_feature(operation, mention, definition)
        if feature.shape != self.weight.shape:
            raise ValueError("triadic binding feature width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict:
        body = {"weight": self.weight.tolist(), "bias": float(self.bias)}
        if self.feature_schema != "triple_product_v1":
            body["feature_schema"] = self.feature_schema
        if self.feature_schema == "projected_joint_v4":
            body["query_projection"] = self.query_projection.tolist()
            body["definition_projection"] = self.definition_projection.tolist()
        return body

    def score_lesion(self) -> TriadicBindingHead:
        return TriadicBindingHead(np.zeros_like(self.weight), 0.0,
            self.feature_schema,
            (np.zeros_like(self.query_projection) if self.query_projection is not None else None),
            (np.zeros_like(self.definition_projection) if self.definition_projection is not None else None))

    def scaled(self, factor: float) -> TriadicBindingHead:
        if not np.isfinite(factor) or factor < 0:
            raise ValueError("triadic factor scale must be finite and nonnegative")
        return TriadicBindingHead(self.weight * factor, self.bias * factor,
                                  self.feature_schema, self.query_projection,
                                  self.definition_projection)

    def role_lesion(self, removed: str) -> TriadicBindingHead:
        """Remove one role's evidence without refitting the other coefficients."""
        if self.feature_schema not in {"joint_representation_v3", "projected_joint_v4"}:
            raise ValueError("role lesion requires a joint representation")
        groups = {
            "operation": (0, 3, 4, 6),
            "mention": (1, 3, 5, 6, 7),
            "definition": (2, 4, 5, 6, 7),
        }
        if removed not in groups:
            raise ValueError("unknown triadic role")
        weights = self.weight.reshape(8, -1).copy()
        weights[list(groups[removed])] = 0
        return TriadicBindingHead(weights.reshape(-1), self.bias, self.feature_schema,
                                  self.query_projection, self.definition_projection)

    def component_lesion(self, retained: str) -> TriadicBindingHead:
        """Keep one fitted evidence component without retraining either arm."""
        if self.feature_schema != "joint_source_v2" or retained not in {
                "geometry", "representation"}:
            raise ValueError("component lesion needs a joint source head")
        weight = self.weight.copy()
        if retained == "geometry":
            weight[:-8] = 0.0
        else:
            weight[-8:] = 0.0
        return TriadicBindingHead(weight, self.bias, self.feature_schema)


def fit_triadic_binding_heads(
    examples: Sequence[SemanticTransducerTrainingExample], *, max_arity: int,
    hidden_channels: Sequence[str], hidden_channel_widths: Sequence[int],
    feature_schema: str = "triple_product_v1",
    projection_basis: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[tuple[TriadicBindingHead, ...], dict]:
    """Learn slot-local contrasts; every negative preserves the source request."""
    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("triadic fit needs source-only training examples")
    if feature_schema not in {"triple_product_v1", "joint_source_v2",
                              "joint_representation_v3", "projected_joint_v4"}:
        raise ValueError("triadic feature schema is unsupported")
    if (feature_schema == "projected_joint_v4") != (projection_basis is not None):
        raise ValueError("projected triadic fit needs only its source-trained basis")
    sources = {item.ir.source_text_sha256: item for item in examples}
    if any(_geometry(item) != _geometry(sources[item.ir.source_text_sha256])
           for item in examples):
        raise ValueError("one source has incompatible triadic view geometry")
    source_counts = Counter(item.ir.source_text_sha256 for item in examples)
    geometry_counts = Counter(_geometry(item) for item in sources.values())
    heads, support = [], []
    for position in range(max_arity):
        features, labels, weights = [], [], []
        for item in examples:
            ir = item.ir
            anchors = (*ir.input_spans, *(instruction.operation_span for instruction in ir.instructions))
            kinds = ["integer_sequence" if isinstance(value, (list, tuple)) else "integer"
                     for value in item.public_inputs]
            for instruction in ir.instructions:
                signature = semantic_primitive_type_signature(instruction.op)
                if signature is None:
                    raise ValueError("triadic fit needs typed source operations")
                kinds.append(signature[1])
            vectors = {span: _relation_span_vector(item.hidden_states, span,
                hidden_channels=hidden_channels, hidden_channel_widths=hidden_channel_widths)
                for span in set(anchors) | {
                    span for instruction in ir.instructions for span in instruction.argument_spans}}
            owners: dict[object, set[int]] = {}
            for index, span in enumerate(ir.input_spans):
                owners.setdefault(span, set()).add(index)
            for instruction in ir.instructions:
                for register, span in zip(instruction.args, instruction.argument_spans, strict=True):
                    owners.setdefault(span, set()).add(register)
            for index, instruction in enumerate(ir.instructions):
                signature = semantic_primitive_type_signature(instruction.op)
                if position >= len(instruction.args):
                    continue
                operation = vectors[instruction.operation_span]
                mention = instruction.argument_spans[position]
                register = instruction.args[position]
                required = signature[0][position]
                alternatives = [
                    (mention, anchors[other]) for other, kind in enumerate(kinds)
                    if other != register and other != len(ir.input_spans) + index
                    and kind == required
                ]
                alternatives.extend((other_span, anchors[register])
                    for other_span, identities in owners.items()
                    if other_span != mention and register not in identities)
                triples = [(mention, anchors[register]), *dict.fromkeys(alternatives)]
                group_weight = (1. / geometry_counts[_geometry(item)]
                                / source_counts[ir.source_text_sha256] / len(triples))
                for choice, definition in triples:
                    if feature_schema == "joint_source_v2":
                        feature = joint_source_binding_feature(
                            operation, vectors[choice], vectors[definition],
                            operation_span=instruction.operation_span,
                            mention_span=choice, definition_span=definition,
                            token_count=len(item.hidden_states))
                    elif feature_schema == "joint_representation_v3":
                        feature = joint_representation_binding_feature(
                            operation, vectors[choice], vectors[definition])
                    elif feature_schema == "projected_joint_v4":
                        feature = projected_joint_binding_feature(
                            operation, vectors[choice], vectors[definition],
                            *projection_basis)
                    else:
                        feature = triadic_binding_feature(
                            operation, vectors[choice], vectors[definition])
                    features.append(feature)
                    labels.append(int(choice == mention and definition == anchors[register]))
                    weights.append(group_weight)
        if not features or not any(labels) or all(labels):
            raise ValueError(f"triadic slot has no contrasting source support: {position}")
        weight, bias = _fit_binary_head(
            np.stack(features), np.asarray(labels, dtype=np.int8),
            sample_weight=_normalized_weights(weights), max_iter=400, tolerance=1e-5)
        heads.append(TriadicBindingHead(weight, bias, feature_schema,
            *(projection_basis if projection_basis is not None else (None, None))))
        support.append({"role": position, "positive": sum(labels),
                        "negative": len(labels) - sum(labels)})
    return tuple(heads), {"algorithm": "source_balanced_binary_contrasts_v1",
                          "feature_schema": feature_schema,
                          "support": support, "source_count": len(sources),
                          "training_views": len(examples)}


def evaluate_triadic_gold_binding(
    heads: tuple[TriadicBindingHead, ...],
    examples: Sequence[SemanticTransducerTrainingExample], *,
    hidden_channels: Sequence[str], hidden_channel_widths: Sequence[int],
    training_source_ids: frozenset[str],
) -> dict:
    """Test source binding on held constructions with gold spans, not answers."""
    if (not heads or not examples or not training_source_ids
            or any(item.ir.source_text_sha256 in training_source_ids for item in examples)):
        raise ValueError("gold binding probe requires unseen source identities")
    correct = wrong = tied = unopposed = 0
    margins = []
    families: dict[str, Counter[str]] = {}
    for item in examples:
        ir = item.ir
        family = item.construction_id.split(":", 1)[0]
        family_counts = families.setdefault(family, Counter())
        family_counts["sources"] += 1
        anchors = (*ir.input_spans, *(ins.operation_span for ins in ir.instructions))
        kinds = ["integer_sequence" if isinstance(value, (list, tuple)) else "integer"
                 for value in item.public_inputs]
        for instruction in ir.instructions:
            signature = semantic_primitive_type_signature(instruction.op)
            if signature is None:
                raise ValueError("gold binding probe needs typed operations")
            kinds.append(signature[1])
        vectors = {span: _relation_span_vector(item.hidden_states, span,
            hidden_channels=hidden_channels, hidden_channel_widths=hidden_channel_widths)
            for span in set(anchors) | {
                span for ins in ir.instructions for span in ins.argument_spans}}
        for step, instruction in enumerate(ir.instructions):
            operation = vectors[instruction.operation_span]
            signature = semantic_primitive_type_signature(instruction.op)
            for role, (register, mention) in enumerate(zip(
                    instruction.args, instruction.argument_spans, strict=True)):
                if role >= len(heads):
                    raise ValueError("gold binding probe lacks a trained role")
                eligible = [index for index, kind in enumerate(kinds[:len(ir.input_spans) + step])
                            if kind == signature[0][role]]
                if register not in eligible:
                    raise ValueError("gold binding reference is outside the typed source")
                if len(eligible) == 1:
                    unopposed += 1
                    family_counts["unopposed_roles"] += 1
                    continue
                scores = {index: heads[role].score(operation, vectors[mention],
                          vectors[anchors[index]],
                          operation_span=instruction.operation_span,
                          mention_span=mention, definition_span=anchors[index],
                          token_count=len(item.hidden_states)) for index in eligible}
                margin = scores[register] - max(score for index, score in scores.items()
                                                  if index != register)
                margins.append(margin)
                if margin > 1e-7:
                    correct += 1
                    family_counts["correct"] += 1
                elif margin < -1e-7:
                    wrong += 1
                    family_counts["wrong"] += 1
                else:
                    tied += 1
                    family_counts["tied"] += 1
    return {"source_count": len({item.ir.source_text_sha256 for item in examples}),
            "opposed_roles": correct + wrong + tied, "correct": correct,
            "wrong": wrong, "tied": tied, "unopposed_roles": unopposed,
            "mean_gold_margin": sum(margins) / len(margins) if margins else None,
            "by_family": {family: {key: counts[key] for key in
                ("sources", "correct", "wrong", "tied", "unopposed_roles")}
                for family, counts in sorted(families.items())},
            "gold_operation_and_mention_spans": True, "serving_authority": False}
