"""Source-trained operation/mention/definition evidence for proposal graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.learning.semantic_program_floor import semantic_primitive_type_signature
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
)


def triadic_binding_feature(operation: np.ndarray, mention: np.ndarray,
                            definition: np.ndarray) -> np.ndarray:
    """Retain each occurrence while letting the operation condition their link."""
    if (operation.ndim != 1 or mention.shape != operation.shape
            or definition.shape != operation.shape):
        raise ValueError("triadic source spans differ in width")
    return _directional_relation_feature(mention * operation, definition)


@dataclass(frozen=True, slots=True)
class TriadicBindingHead:
    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if (weight.size < _DIRECTIONAL_RELATION_PARTS
                or weight.size % _DIRECTIONAL_RELATION_PARTS
                or not np.all(np.isfinite(weight)) or not np.isfinite(self.bias)):
            raise ValueError("triadic binding head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def score(self, operation: np.ndarray, mention: np.ndarray,
              definition: np.ndarray) -> float:
        feature = triadic_binding_feature(operation, mention, definition)
        if feature.shape != self.weight.shape:
            raise ValueError("triadic binding feature width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict:
        return {"weight": self.weight.tolist(), "bias": float(self.bias)}


def fit_triadic_binding_heads(
    examples: Sequence[SemanticTransducerTrainingExample], *, max_arity: int,
    hidden_channels: Sequence[str], hidden_channel_widths: Sequence[int],
) -> tuple[tuple[TriadicBindingHead, ...], dict]:
    """Learn slot-local contrasts; every negative preserves the source request."""
    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("triadic fit needs source-only training examples")
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
                    features.append(triadic_binding_feature(
                        operation, vectors[choice], vectors[definition]))
                    labels.append(int(choice == mention and definition == anchors[register]))
                    weights.append(group_weight)
        if not features or not any(labels) or all(labels):
            raise ValueError(f"triadic slot has no contrasting source support: {position}")
        weight, bias = _fit_binary_head(
            np.stack(features), np.asarray(labels, dtype=np.int8),
            sample_weight=_normalized_weights(weights), max_iter=400, tolerance=1e-5)
        heads.append(TriadicBindingHead(weight, bias))
        support.append({"role": position, "positive": sum(labels),
                        "negative": len(labels) - sum(labels)})
    return tuple(heads), {"algorithm": "source_balanced_binary_contrasts_v1",
                          "support": support, "source_count": len(sources),
                          "training_views": len(examples)}
