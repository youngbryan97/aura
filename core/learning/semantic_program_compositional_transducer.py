"""Compose local neural meanings into a typed semantic program graph.

The v6 shared transducer learned several geometries at once, but decoded them
through a whole-request step-count classifier and heads named by absolute step.
That made an unlabelled family name recoverable from the hidden state and made
the geometry follow the family.  A leave-family-out probe exposed the result:
input grounding transferred perfectly while every structural decision failed.

This transducer has no geometry classifier and no step-indexed learned head.
It learns two reusable kinds of local evidence:

* which spans name operations and which primitive each span means;
* which spans are arguments of an operation and which earlier definition they
  refer to.

An exact operation chart and bounded typed graph search compose those atoms
into a connected, acyclic SSA program.  Step count is therefore the number of
operation nodes supported by the request, not a template selected from the
whole-request embedding.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from core.learning.semantic_input_grounding import (
    SemanticInputGroundingContract,
    semantic_input_grounding_contract_from_dict,
)
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)
from core.learning.semantic_program_shared_transducer import (
    _channel_span,
    _geometry,
    _geometry_name,
    _normalized_weights,
    _relation_span_vector,
)
from core.learning.semantic_program_transducer import (
    LinearClassifierHead,
    LinearPointerHead,
    LinearPointerSequenceScores,
    MultiViewClassifierHead,
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
    _fit_binary_head,
    _fit_classifier,
    _hidden_array,
    _joint_pointer_assignment,
    _operation_feature,
)

COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v11"
COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v11"
_OPERATION_MODE: Final = "contextual_mean"
_RELATION_CHANNEL: Final = "middle_causal_hidden"
_MAX_STEPS: Final = 16
_MAX_INPUTS: Final = 8
_OPERATION_CANDIDATES: Final = 256
_ARGUMENT_CANDIDATES: Final = 128
_ARGUMENT_BEAM: Final = 128
_ARGUMENT_MENTIONS_PER_DEFINITION: Final = 4
_POINTER_HARD_NEGATIVES: Final = 16
_OPERATION_PENALTY_POINTS: Final = 161
_OPERATION_CHART_BEAM: Final = 16
_DIRECTIONAL_RELATION_PARTS: Final = 3


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _log_sigmoid(value: float) -> float:
    """Stable log probability for one binary-head logit."""

    return -math.log1p(math.exp(-abs(value))) + min(value, 0.0)


def _overlap(left: TokenSpan, right: TokenSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _all_semantic_spans(
    item: SemanticTransducerTrainingExample,
) -> tuple[TokenSpan, ...]:
    return (
        *item.ir.input_spans,
        *(instruction.operation_span for instruction in item.ir.instructions),
        *(span for instruction in item.ir.instructions for span in instruction.argument_spans),
    )


def _register_definition_spans(
    item: SemanticTransducerTrainingExample,
) -> tuple[TokenSpan, ...]:
    definitions = item.register_definition_spans or (
        *item.ir.input_spans,
        *(instruction.operation_span for instruction in item.ir.instructions),
    )
    if len(definitions) != item.ir.n_inputs + len(item.ir.instructions):
        raise ValueError("compositional register-definition geometry differs")
    return definitions


def _definition_span_candidates(
    anchor: TokenSpan,
    *,
    token_count: int,
    max_span_tokens: int,
) -> tuple[TokenSpan, ...]:
    """Enumerate causal definition envelopes anchored to one exact register."""

    stop = max(anchor.end, min(token_count, anchor.start + max_span_tokens))
    return tuple(TokenSpan(anchor.start, end) for end in range(anchor.end, stop + 1))


def _best_penalized_operation_chart(
    by_count: Sequence[tuple[float, tuple[_OperationNode, ...]]],
    *,
    penalty: float,
) -> tuple[_OperationNode, ...]:
    """Choose a chart, or an empty refusal when this item has no finite chart."""

    candidates = tuple(
        (score - penalty * count, nodes)
        for count, (score, nodes) in enumerate(by_count, start=1)
        if math.isfinite(score)
    )
    if not candidates:
        return ()
    return max(candidates, key=lambda value: (value[0], -len(value[1])))[1]


def _shared_pointer_training_indices(
    item: SemanticTransducerTrainingExample,
    positive_span: TokenSpan,
    *,
    end: bool,
) -> tuple[int, ...]:
    positive = positive_span.end - 1 if end else positive_span.start
    candidates = [(span.end - 1 if end else span.start) for span in _all_semantic_spans(item)]
    candidates.extend(
        (
            positive - 2,
            positive - 1,
            positive + 1,
            positive + 2,
            0,
            item.hidden_states.shape[0] - 1,
        )
    )
    negatives = tuple(
        index
        for index in dict.fromkeys(candidates)
        if 0 <= index < item.hidden_states.shape[0] and index != positive
    )[:_POINTER_HARD_NEGATIVES]
    return (positive, *negatives)


def _fit_shared_pointer(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    spans: Callable[[SemanticTransducerTrainingExample], Sequence[TokenSpan]],
) -> LinearPointerHead:
    parameters: list[tuple[np.ndarray, float]] = []
    geometry_counts = Counter(_geometry(item) for item in training)
    for end in (False, True):
        features: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for item in training:
            positives = tuple(spans(item))
            if not positives:
                continue
            item_weight = 1.0 / geometry_counts[_geometry(item)]
            for positive in positives:
                indices = _shared_pointer_training_indices(
                    item,
                    positive,
                    end=end,
                )
                features.extend(item.hidden_states[index] for index in indices)
                labels.extend((1, *(0 for _ in indices[1:])))
                weights.extend([item_weight / len(positives) / len(indices)] * len(indices))
        if not features:
            raise ValueError("compositional pointer has no training support")
        parameters.append(
            _fit_binary_head(
                np.stack(features),
                np.asarray(labels, dtype=np.int8),
                sample_weight=_normalized_weights(weights),
                max_iter=250,
                tolerance=1e-3,
            )
        )
    return LinearPointerHead(
        parameters[0][0],
        parameters[0][1],
        parameters[1][0],
        parameters[1][1],
    )


@dataclass(frozen=True, slots=True)
class LinearArgumentRoleHead:
    """Shared relation from an argument mention to one operation and slot."""

    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if (
            weight.size < _DIRECTIONAL_RELATION_PARTS
            or weight.size % _DIRECTIONAL_RELATION_PARTS
            or not np.all(np.isfinite(weight))
            or not np.isfinite(self.bias)
        ):
            raise ValueError("compositional argument-role head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def score(self, reference: np.ndarray, operation: np.ndarray) -> float:
        feature = _directional_relation_feature(reference, operation)
        if feature.shape != self.weight.shape:
            raise ValueError("compositional argument-role feature width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {"weight": self.weight.tolist(), "bias": float(self.bias)}


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

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if (
            weight.size < _DIRECTIONAL_RELATION_PARTS
            or weight.size % _DIRECTIONAL_RELATION_PARTS
            or not np.all(np.isfinite(weight))
            or not np.isfinite(self.bias)
            or not np.isfinite(self.pointer_scale)
            or self.pointer_scale < 0.0
        ):
            raise ValueError("compositional directional relation head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // _DIRECTIONAL_RELATION_PARTS)

    def score(self, reference: np.ndarray, definition: np.ndarray) -> float:
        feature = _directional_relation_feature(reference, definition)
        if feature.shape != self.weight.shape:
            raise ValueError("compositional directional relation width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.tolist(),
            "bias": float(self.bias),
            "pointer_scale": float(self.pointer_scale),
        }


@dataclass(frozen=True, slots=True)
class RegisterUseContract:
    """Source-learned bounds for a well-formed register-use graph."""

    input_min_uses: int
    input_max_uses: int
    intermediate_min_uses: int
    intermediate_max_uses: int
    distinct_arguments: bool

    def __post_init__(self) -> None:
        if (
            type(self.input_min_uses) is not int
            or type(self.input_max_uses) is not int
            or type(self.intermediate_min_uses) is not int
            or type(self.intermediate_max_uses) is not int
            or not 0 <= self.input_min_uses <= self.input_max_uses
            or not 0 <= self.intermediate_min_uses <= self.intermediate_max_uses
            or type(self.distinct_arguments) is not bool
        ):
            raise ValueError("compositional register-use contract is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_min_uses": self.input_min_uses,
            "input_max_uses": self.input_max_uses,
            "intermediate_min_uses": self.intermediate_min_uses,
            "intermediate_max_uses": self.intermediate_max_uses,
            "distinct_arguments": self.distinct_arguments,
        }

    def allows_partial(self, counts: Counter[int], *, n_inputs: int) -> bool:
        return all(
            count
            <= (self.input_max_uses if register < n_inputs else self.intermediate_max_uses)
            for register, count in counts.items()
        )

    def accepts_complete(
        self,
        counts: Counter[int],
        *,
        n_inputs: int,
        operation_count: int,
        sink: int,
    ) -> bool:
        return all(
            self.input_min_uses <= counts[index] <= self.input_max_uses
            for index in range(n_inputs)
        ) and all(
            self.intermediate_min_uses
            <= counts[n_inputs + index]
            <= self.intermediate_max_uses
            for index in range(operation_count)
            if index != sink
        )


@dataclass(frozen=True, slots=True)
class _OperationNode:
    span: TokenSpan
    operation: str
    score: float
    pointer_score: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _TypedArgumentAssignment:
    """One complete typed graph together with its neural evidence score."""

    operation_nodes: tuple[_OperationNode, ...]
    arguments: tuple[tuple[int, ...], ...]
    argument_spans: tuple[tuple[TokenSpan, ...], ...]
    score: float


def _operation_nodes(
    *,
    pointer: LinearPointerHead,
    classifier: MultiViewClassifierHead,
    hidden: np.ndarray,
    input_spans: Sequence[TokenSpan],
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[_OperationNode, ...]:
    nodes: list[_OperationNode] = []
    for span, pointer_score in pointer.decode_candidates(
        hidden,
        limit=_OPERATION_CANDIDATES,
        max_span_tokens=max_span_tokens,
    ):
        if any(_overlap(span, input_span) for input_span in input_spans):
            continue
        operation, confidence = classifier.predict(
            (
                _operation_feature(
                    hidden,
                    span,
                    mode=_OPERATION_MODE,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                ),
            )
        )
        score = float(pointer_score + math.log(max(confidence, 1e-12)))
        nodes.append(
            _OperationNode(
                span=span,
                operation=operation,
                score=score,
                pointer_score=float(pointer_score),
                confidence=confidence,
            )
        )
    return tuple(nodes)


def _best_nonoverlapping_nodes(
    nodes: Sequence[_OperationNode],
    count: int,
) -> tuple[float, tuple[_OperationNode, ...]]:
    """Exact cardinality-constrained weighted interval scheduling."""

    ordered = tuple(
        sorted(
            nodes,
            key=lambda item: (item.span.end, item.span.start, -item.score),
        )
    )
    previous: list[int] = []
    for index, node in enumerate(ordered):
        prior = index - 1
        while prior >= 0 and ordered[prior].span.end > node.span.start:
            prior -= 1
        previous.append(prior)
    impossible = (-float("inf"), ())
    table: list[list[tuple[float, tuple[_OperationNode, ...]]]] = [
        [impossible for _ in range(count + 1)] for _ in range(len(ordered) + 1)
    ]
    table[0][0] = (0.0, ())
    for index, node in enumerate(ordered, start=1):
        for size in range(count + 1):
            table[index][size] = table[index - 1][size]
            if size < 1:
                continue
            prior_score, prior_nodes = table[previous[index - 1] + 1][size - 1]
            if not math.isfinite(prior_score):
                continue
            candidate = (prior_score + node.score, (*prior_nodes, node))
            incumbent = table[index][size]
            if (candidate[0], tuple((n.span.start, n.span.end) for n in candidate[1])) > (
                incumbent[0],
                tuple((n.span.start, n.span.end) for n in incumbent[1]),
            ):
                table[index][size] = candidate
    return table[len(ordered)][count]


def _best_nonoverlapping_node_charts(
    nodes: Sequence[_OperationNode],
    count: int,
    *,
    limit: int,
) -> tuple[tuple[float, tuple[_OperationNode, ...]], ...]:
    """Exact top-k cardinality-constrained weighted interval scheduling."""

    if limit < 1:
        raise ValueError("compositional operation-chart limit must be positive")
    ordered = tuple(
        sorted(nodes, key=lambda item: (item.span.end, item.span.start, -item.score))
    )
    previous: list[int] = []
    for index, node in enumerate(ordered):
        prior = index - 1
        while prior >= 0 and ordered[prior].span.end > node.span.start:
            prior -= 1
        previous.append(prior)
    table: list[list[tuple[tuple[float, tuple[_OperationNode, ...]], ...]]] = [
        [() for _ in range(count + 1)] for _ in range(len(ordered) + 1)
    ]
    table[0][0] = ((0.0, ()),)
    for index, node in enumerate(ordered, start=1):
        for size in range(count + 1):
            candidates = list(table[index - 1][size])
            if size >= 1:
                candidates.extend(
                    (score + node.score, (*selected, node))
                    for score, selected in table[previous[index - 1] + 1][size - 1]
                )
            unique: dict[tuple[tuple[int, int, str], ...], tuple[float, tuple[_OperationNode, ...]]] = {}
            for candidate in candidates:
                key = tuple(
                    (item.span.start, item.span.end, item.operation)
                    for item in candidate[1]
                )
                incumbent = unique.get(key)
                if incumbent is None or candidate[0] > incumbent[0]:
                    unique[key] = candidate
            table[index][size] = tuple(
                sorted(
                    unique.values(),
                    key=lambda item: (
                        -item[0],
                        tuple((node.span.start, node.span.end) for node in item[1]),
                    ),
                )[:limit]
            )
    return table[len(ordered)][count]


def _operation_chart(
    nodes: Sequence[_OperationNode],
    *,
    max_steps: int,
    length_penalty: float,
) -> tuple[_OperationNode, ...]:
    candidates = [
        (score - length_penalty * count, selected)
        for count in range(1, max_steps + 1)
        for score, selected in (_best_nonoverlapping_nodes(nodes, count),)
        if math.isfinite(score)
    ]
    if not candidates:
        return ()
    return max(
        candidates,
        key=lambda item: (
            item[0],
            -len(item[1]),
            tuple((-node.span.start, -node.span.end) for node in item[1]),
        ),
    )[1]


def _operation_chart_candidates(
    nodes: Sequence[_OperationNode],
    *,
    max_steps: int,
    length_penalty: float,
    limit: int = _OPERATION_CHART_BEAM,
) -> tuple[tuple[_OperationNode, ...], ...]:
    candidates = [
        (score - length_penalty * count, selected)
        for count in range(1, max_steps + 1)
        for score, selected in _best_nonoverlapping_node_charts(
            nodes,
            count,
            limit=limit,
        )
    ]
    return tuple(
        selected
        for _score, selected in sorted(
            candidates,
            key=lambda item: (
                -item[0],
                len(item[1]),
                tuple((node.span.start, node.span.end) for node in item[1]),
            ),
        )[:limit]
    )


def _fit_argument_role_heads(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    max_arity: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[LinearArgumentRoleHead, ...]:
    heads: list[LinearArgumentRoleHead] = []
    geometry_counts = Counter(_geometry(item) for item in training)
    for position in range(max_arity):
        features: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        for item in training:
            all_arguments = tuple(
                span for instruction in item.ir.instructions for span in instruction.argument_spans
            )
            for instruction in item.ir.instructions:
                if position >= len(instruction.argument_spans):
                    continue
                positive = instruction.argument_spans[position]
                negatives = tuple(
                    span
                    for span in dict.fromkeys(
                        (
                            *item.ir.input_spans,
                            *all_arguments,
                            *(step.operation_span for step in item.ir.instructions),
                        )
                    )
                    if span != positive
                )
                spans = (positive, *negatives)
                operation = _relation_span_vector(
                    item.hidden_states,
                    instruction.operation_span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                features.extend(
                    _directional_relation_feature(
                        _relation_span_vector(
                            item.hidden_states,
                            span,
                            hidden_channels=hidden_channels,
                            hidden_channel_widths=hidden_channel_widths,
                        ),
                        operation,
                    )
                    for span in spans
                )
                labels.extend((1, *(0 for _ in negatives)))
                decision_weight = 1.0 / geometry_counts[_geometry(item)] / len(spans)
                weights.extend([decision_weight] * len(spans))
        if not features:
            raise ValueError(f"compositional argument slot has no support: {position}")
        weight, bias = _fit_binary_head(
            np.stack(features),
            np.asarray(labels, dtype=np.int8),
            sample_weight=_normalized_weights(weights),
            max_iter=400,
            tolerance=1e-5,
        )
        heads.append(LinearArgumentRoleHead(weight, bias))
    return tuple(heads)


def _fit_directional_relation_head(
    training: Sequence[SemanticTransducerTrainingExample],
    *,
    item_weights: Mapping[int, float],
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[np.ndarray, float]:
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


def _fit_register_use_contract(
    training: Sequence[SemanticTransducerTrainingExample],
) -> RegisterUseContract:
    input_uses: list[int] = []
    intermediate_uses: list[int] = []
    distinct_arguments = True
    for item in training:
        counts: Counter[int] = Counter(
            register
            for instruction in item.ir.instructions
            for register in instruction.args
        )
        input_uses.extend(counts[index] for index in range(item.ir.n_inputs))
        intermediate_uses.extend(
            counts[item.ir.n_inputs + step]
            for step in range(len(item.ir.instructions))
            if item.ir.n_inputs + step != item.ir.report_value
        )
        distinct_arguments = distinct_arguments and all(
            len(set(instruction.args)) == len(instruction.args)
            for instruction in item.ir.instructions
        )
    if not input_uses or not intermediate_uses:
        raise ValueError("compositional register-use contract has insufficient support")
    return RegisterUseContract(
        input_min_uses=min(input_uses),
        input_max_uses=max(input_uses),
        intermediate_min_uses=min(intermediate_uses),
        intermediate_max_uses=max(intermediate_uses),
        distinct_arguments=distinct_arguments,
    )


def _select_definition_pointer_scale(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    relation_head: DirectionalRelationHead,
    definition_pointer: LinearPointerHead,
    max_definition_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    """Calibrate definition localization only on source validation references."""

    scales = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
    correct = {scale: 0 for scale in scales}
    total = 0
    for item in validation:
        anchors = (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
        )
        pointer_scores = definition_pointer.score_sequence(item.hidden_states)
        candidates = tuple(
            tuple(
                (
                    span,
                    _relation_span_vector(
                        item.hidden_states,
                        span,
                        hidden_channels=hidden_channels,
                        hidden_channel_widths=hidden_channel_widths,
                    ),
                )
                for span in _definition_span_candidates(
                    anchor,
                    token_count=item.hidden_states.shape[0],
                    max_span_tokens=max_definition_span_tokens,
                )
            )
            for anchor in anchors
        )
        for step, instruction in enumerate(item.ir.instructions):
            available = item.ir.n_inputs + step
            for reference_span, expected_register in zip(
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
                for scale in scales:
                    scores = tuple(
                        max(
                            relation_head.score(reference, definition)
                            + scale * pointer_scores.score_span(span)
                            for span, definition in register_candidates
                        )
                        for register_candidates in candidates[:available]
                    )
                    correct[scale] += int(
                        max(range(available), key=lambda index: scores[index])
                        == expected_register
                    )
                total += 1
    if total < 1:
        raise ValueError("compositional definition-pointer calibration has no support")
    rows = [
        {"pointer_scale": scale, "register_top1": correct[scale], "total": total}
        for scale in scales
    ]
    winner = max(rows, key=lambda row: (row["register_top1"], -row["pointer_scale"]))
    return float(winner["pointer_scale"]), rows


def _input_type(value: SemanticValue) -> str:
    return "integer" if isinstance(value, int) else "integer_sequence"


def _assign_typed_arguments(
    *,
    model: CompositionalSemanticProgramTransducer,
    hidden: np.ndarray,
    inputs: Sequence[SemanticValue],
    input_spans: Sequence[TokenSpan],
    operation_nodes: Sequence[_OperationNode],
    argument_pointer_scores: LinearPointerSequenceScores,
) -> _TypedArgumentAssignment | None:
    if (
        len(operation_nodes) > 1
        and not model.allow_computed_dependencies
        and model.register_use_contract.intermediate_min_uses > 0
    ):
        return None
    proposals = list(
        argument_pointer_scores.decode_candidates(
            limit=_ARGUMENT_CANDIDATES,
            max_span_tokens=model.max_span_tokens,
        )
    )
    observed = {span for span, _score in proposals}
    proposals.extend(
        (span, argument_pointer_scores.score_span(span))
        for span in input_spans
        if span not in observed
    )
    proposals = [
        (span, score)
        for span, score in proposals
        if not any(_overlap(span, node.span) for node in operation_nodes)
    ]
    operation_types: list[tuple[tuple[str, ...], str]] = []
    for node in operation_nodes:
        signature = semantic_primitive_type_signature(node.operation)
        if signature is None:
            return None
        operation_types.append(signature)
    definitions = (*input_spans, *(node.span for node in operation_nodes))
    register_types = (
        *(_input_type(value) for value in inputs),
        *(result_type for _argument_types, result_type in operation_types),
    )
    definition_pointer_scores = model.definition_pointer.score_sequence(hidden)
    definition_vectors = tuple(
        tuple(
            (
                candidate,
                _relation_span_vector(
                    hidden,
                    candidate,
                    hidden_channels=model.hidden_channels,
                    hidden_channel_widths=model.hidden_channel_widths,
                ),
            )
            for candidate in _definition_span_candidates(
                span,
                token_count=hidden.shape[0],
                max_span_tokens=model.max_definition_span_tokens,
            )
        )
        for span in definitions
    )
    reference_vectors = {
        span: _relation_span_vector(
            hidden,
            span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )
        for span, _score in proposals
    }
    relation_scores = {
        span: tuple(
            max(
                model.definition_relation_head.score(reference, definition)
                + model.definition_relation_head.pointer_scale
                * definition_pointer_scores.score_span(candidate)
                for candidate, definition in candidates
            )
            for candidates in definition_vectors
        )
        for span, reference in reference_vectors.items()
    }
    states: list[
        tuple[
            float,
            tuple[tuple[int, ...], ...],
            tuple[tuple[TokenSpan, ...], ...],
            tuple[tuple[int, ...], ...],
        ]
    ] = [(0.0, (), (), ())]
    for node_index, node in enumerate(operation_nodes):
        argument_types, _result_type = operation_types[node_index]
        if len(argument_types) > len(model.argument_role_heads):
            return None
        operation_vector = _relation_span_vector(
            hidden,
            node.span,
            hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths,
        )
        partial: list[tuple[float, tuple[int, ...], tuple[TokenSpan, ...]]] = [(0.0, (), ())]
        for position, required_type in enumerate(argument_types):
            role_head = model.argument_role_heads[position]
            by_register: dict[int, list[tuple[float, TokenSpan]]] = {}
            for span, pointer_score in proposals:
                if span.end - span.start > model.max_argument_span_tokens_by_type[required_type]:
                    continue
                reference = reference_vectors[span]
                role_score = role_head.score(reference, operation_vector)
                exact_inputs = tuple(
                    index for index, input_span in enumerate(input_spans) if span == input_span
                )
                candidate_registers = (
                    exact_inputs if exact_inputs else tuple(range(len(definitions)))
                )
                for register in candidate_registers:
                    register_type = register_types[register]
                    if register_type != required_type:
                        continue
                    if register == len(inputs) + node_index:
                        continue
                    if not model.allow_computed_dependencies and register >= len(inputs):
                        continue
                    relation_score = relation_scores[span][register]
                    if (
                        register >= len(inputs)
                        and register - len(inputs) > node_index
                        and relation_score <= 0.0
                    ):
                        continue
                    score = (
                        model.argument_role_scale * _log_sigmoid(role_score)
                        + model.definition_relation_scale * _log_sigmoid(relation_score)
                        + model.argument_pointer_scale * _log_sigmoid(pointer_score)
                    )
                    by_register.setdefault(register, []).append((score, span))
            if not by_register:
                return None
            options = sorted(
                (
                    (score, register, span)
                    for register, candidates in by_register.items()
                    for score, span in sorted(
                        candidates,
                        key=lambda item: (-item[0], item[1].start, item[1].end),
                    )[:_ARGUMENT_MENTIONS_PER_DEFINITION]
                ),
                key=lambda item: (-item[0], item[1], item[2].start, item[2].end),
            )
            partial = sorted(
                (
                    (
                        total + score,
                        (*registers, register),
                        (*spans, span),
                    )
                    for total, registers, spans in partial
                    for score, register, span in options
                    if not any(_overlap(span, previous) for previous in spans)
                    and (
                        not model.register_use_contract.distinct_arguments
                        or register not in registers
                    )
                ),
                key=lambda item: (
                    -item[0],
                    item[1],
                    tuple((span.start, span.end) for span in item[2]),
                ),
            )[:_ARGUMENT_BEAM]
            if not partial:
                return None
        candidates: list[
            tuple[
                float,
                tuple[tuple[int, ...], ...],
                tuple[tuple[TokenSpan, ...], ...],
                tuple[tuple[int, ...], ...],
            ]
        ] = []
        for total, arguments, spans, dependencies in states:
            for step_score, step_arguments, step_spans in partial:
                if any(
                    _overlap(current, previous)
                    for current in step_spans
                    for previous_step in spans
                    for previous in previous_step
                ):
                    continue
                step_dependencies = tuple(
                    sorted(
                        register - len(inputs)
                        for register in set(step_arguments)
                        if register >= len(inputs)
                    )
                )
                candidate_dependencies = (*dependencies, step_dependencies)
                if (
                    _operation_order(
                        candidate_dependencies,
                        operation_nodes,
                        require_connected=False,
                    )
                    is None
                ):
                    continue
                use_counts: Counter[int] = Counter(
                    register
                    for values in (*arguments, step_arguments)
                    for register in values
                )
                if not model.register_use_contract.allows_partial(
                    use_counts,
                    n_inputs=len(inputs),
                ):
                    continue
                candidates.append(
                    (
                        total + step_score,
                        (*arguments, step_arguments),
                        (*spans, step_spans),
                        candidate_dependencies,
                    )
                )
        states = sorted(candidates, key=lambda item: (-item[0], item[1]))[:_ARGUMENT_BEAM]
        if not states:
            return None
    for score, arguments, spans, dependencies in states:
        order = _operation_order(
            dependencies,
            operation_nodes,
            require_connected=True,
        )
        if order is None:
            continue
        referenced = {dependency for values in dependencies for dependency in values}
        sink = next(index for index in range(len(operation_nodes)) if index not in referenced)
        use_counts = Counter(register for values in arguments for register in values)
        if not model.register_use_contract.accepts_complete(
            use_counts,
            n_inputs=len(inputs),
            operation_count=len(operation_nodes),
            sink=sink,
        ):
            continue
        output_registers = {
            source_index: len(inputs) + target_index
            for target_index, source_index in enumerate(order)
        }
        ordered_arguments = tuple(
            tuple(
                register if register < len(inputs) else output_registers[register - len(inputs)]
                for register in arguments[source_index]
            )
            for source_index in order
        )
        return _TypedArgumentAssignment(
            operation_nodes=tuple(operation_nodes[index] for index in order),
            arguments=ordered_arguments,
            argument_spans=tuple(spans[index] for index in order),
            score=score,
        )
    return None


def _operation_order(
    dependencies: Sequence[Sequence[int]],
    operation_nodes: Sequence[_OperationNode],
    *,
    require_connected: bool,
) -> tuple[int, ...] | None:
    """Return a stable topological schedule for a partial or complete graph."""

    count = len(dependencies)
    if count > len(operation_nodes):
        return None
    normalized = tuple(tuple(sorted(set(values))) for values in dependencies)
    if any(
        dependency < 0 or dependency >= len(operation_nodes) or dependency == index
        for index, values in enumerate(normalized)
        for dependency in values
    ):
        return None
    completed: set[int] = set()
    order: list[int] = []
    while len(order) < count:
        ready = sorted(
            (
                index
                for index in range(count)
                if index not in completed
                and all(
                    dependency >= count or dependency in completed
                    for dependency in normalized[index]
                )
            ),
            key=lambda index: (
                operation_nodes[index].span.start,
                operation_nodes[index].span.end,
                index,
            ),
        )
        if not ready:
            return None
        completed.add(ready[0])
        order.append(ready[0])
    if not require_connected:
        return tuple(order)
    if count != len(operation_nodes):
        return None
    referenced = {dependency for values in normalized for dependency in values}
    sinks = tuple(index for index in range(count) if index not in referenced)
    if len(sinks) != 1:
        return None
    required = {sinks[0]}
    frontier = [sinks[0]]
    while frontier:
        current = frontier.pop()
        for dependency in normalized[current]:
            if dependency not in required:
                required.add(dependency)
                frontier.append(dependency)
    if required != set(range(count)):
        return None
    return tuple(order)


@dataclass(frozen=True, slots=True)
class CompositionalSemanticProgramTransducer:
    """One local-atom chart decoder with no family or geometry router."""

    hidden_size: int
    model_basis_sha256: str
    hidden_channels: tuple[str, ...]
    hidden_channel_widths: tuple[int, ...]
    input_grounding: SemanticInputGroundingContract
    operation_pointer: LinearPointerHead
    argument_pointer: LinearPointerHead
    definition_pointer: LinearPointerHead
    operation_head: MultiViewClassifierHead
    argument_role_heads: tuple[LinearArgumentRoleHead, ...]
    definition_relation_head: DirectionalRelationHead
    max_steps: int
    max_inputs: int
    max_span_tokens: int
    max_definition_span_tokens: int
    max_argument_span_tokens_by_type: dict[str, int]
    register_use_contract: RegisterUseContract
    operation_chart_beam: int
    operation_length_penalty: float
    argument_role_scale: float
    definition_relation_scale: float
    argument_pointer_scale: float
    allow_computed_dependencies: bool
    training_receipt: dict[str, Any]
    schema: str = COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA

    def __post_init__(self) -> None:
        receipt = json.loads(_canonical_bytes(self.training_receipt))
        relation_start, relation_end = _channel_span(
            _RELATION_CHANNEL,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
        )
        coefficient = self._coefficient_body()
        argument_bounds = dict(self.max_argument_span_tokens_by_type)
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            self.schema != COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or not _is_sha256(self.model_basis_sha256)
            or not self.hidden_channels
            or len(self.hidden_channels) != len(self.hidden_channel_widths)
            or sum(self.hidden_channel_widths) != self.hidden_size
            or self.operation_pointer.width != self.hidden_size
            or self.argument_pointer.width != self.hidden_size
            or self.definition_pointer.width != self.hidden_size
            or self.operation_head.modes != (_OPERATION_MODE,)
            or not self.argument_role_heads
            or any(
                head.channel_width != relation_end - relation_start
                for head in self.argument_role_heads
            )
            or self.definition_relation_head.channel_width != relation_end - relation_start
            or type(self.max_steps) is not int
            or not 1 <= self.max_steps <= _MAX_STEPS
            or type(self.max_inputs) is not int
            or not 1 <= self.max_inputs <= _MAX_INPUTS
            or type(self.max_span_tokens) is not int
            or self.max_span_tokens < 1
            or type(self.max_definition_span_tokens) is not int
            or self.max_definition_span_tokens < self.max_span_tokens
            or set(argument_bounds) != {"integer", "integer_sequence"}
            or any(type(value) is not int or value < 1 for value in argument_bounds.values())
            or any(
                not math.isfinite(value) or value <= 0
                for value in (
                    self.argument_role_scale,
                    self.definition_relation_scale,
                    self.argument_pointer_scale,
                )
            )
            or not math.isfinite(self.operation_length_penalty)
            or type(self.allow_computed_dependencies) is not bool
            or receipt.get("schema") != COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA
            or receipt.get("receipt_sha256") != _sha(body)
            or receipt.get("model_basis_sha256") != self.model_basis_sha256
            or receipt.get("input_grounding_sha256") != self.input_grounding.contract_sha256
            or receipt.get("coefficient_sha256") != _sha(coefficient)
            or receipt.get("global_geometry_classifier_present") is not False
            or receipt.get("step_indexed_heads_present") is not False
            or receipt.get("argument_span_bounds") != argument_bounds
            or receipt.get("definition_span_bound") != self.max_definition_span_tokens
            or type(self.operation_chart_beam) is not int
            or not 1 <= self.operation_chart_beam <= _OPERATION_CHART_BEAM
            or receipt.get("operation_chart_beam") != self.operation_chart_beam
            or receipt.get("register_use_contract") != self.register_use_contract.to_dict()
            or any(
                receipt.get(field) is not False
                for field in (
                    "expected_answers_available",
                    "verifier_traces_available",
                    "generated_compiler_text_available",
                    "correctness_authority",
                )
            )
        ):
            raise ValueError("compositional semantic transducer envelope is invalid")
        object.__setattr__(self, "training_receipt", receipt)
        object.__setattr__(self, "max_argument_span_tokens_by_type", argument_bounds)

    def _coefficient_body(self) -> dict[str, Any]:
        return {
            "operation_pointer": self.operation_pointer.to_dict(),
            "argument_pointer": self.argument_pointer.to_dict(),
            "definition_pointer": self.definition_pointer.to_dict(),
            "operation_head": self.operation_head.to_dict(),
            "argument_role_heads": [head.to_dict() for head in self.argument_role_heads],
            "definition_relation_head": self.definition_relation_head.to_dict(),
            "operation_length_penalty": self.operation_length_penalty,
            "argument_role_scale": self.argument_role_scale,
            "definition_relation_scale": self.definition_relation_scale,
            "argument_pointer_scale": self.argument_pointer_scale,
            "allow_computed_dependencies": self.allow_computed_dependencies,
            "register_use_contract": self.register_use_contract.to_dict(),
        }

    @property
    def receipt_sha256(self) -> str:
        return str(self.training_receipt["receipt_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hidden_size": self.hidden_size,
            "model_basis_sha256": self.model_basis_sha256,
            "hidden_channels": list(self.hidden_channels),
            "hidden_channel_widths": list(self.hidden_channel_widths),
            "input_grounding": self.input_grounding.to_dict(),
            "max_steps": self.max_steps,
            "max_inputs": self.max_inputs,
            "max_span_tokens": self.max_span_tokens,
            "max_definition_span_tokens": self.max_definition_span_tokens,
            "max_argument_span_tokens_by_type": self.max_argument_span_tokens_by_type,
            "operation_chart_beam": self.operation_chart_beam,
            **self._coefficient_body(),
            "training_receipt": self.training_receipt,
        }

    def _with_coefficients(self, **changes: Any) -> CompositionalSemanticProgramTransducer:
        values = {
            "operation_pointer": changes.get("operation_pointer", self.operation_pointer),
            "argument_pointer": changes.get("argument_pointer", self.argument_pointer),
            "definition_pointer": changes.get("definition_pointer", self.definition_pointer),
            "operation_head": changes.get("operation_head", self.operation_head),
            "argument_role_heads": changes.get("argument_role_heads", self.argument_role_heads),
            "definition_relation_head": changes.get(
                "definition_relation_head", self.definition_relation_head
            ),
            "operation_length_penalty": changes.get(
                "operation_length_penalty", self.operation_length_penalty
            ),
            "argument_role_scale": changes.get("argument_role_scale", self.argument_role_scale),
            "definition_relation_scale": changes.get(
                "definition_relation_scale", self.definition_relation_scale
            ),
            "argument_pointer_scale": changes.get(
                "argument_pointer_scale", self.argument_pointer_scale
            ),
            "allow_computed_dependencies": changes.get(
                "allow_computed_dependencies", self.allow_computed_dependencies
            ),
            "register_use_contract": changes.get(
                "register_use_contract", self.register_use_contract
            ),
        }
        coefficient = {
            "operation_pointer": values["operation_pointer"].to_dict(),
            "argument_pointer": values["argument_pointer"].to_dict(),
            "definition_pointer": values["definition_pointer"].to_dict(),
            "operation_head": values["operation_head"].to_dict(),
            "argument_role_heads": [head.to_dict() for head in values["argument_role_heads"]],
            "definition_relation_head": values["definition_relation_head"].to_dict(),
            "operation_length_penalty": values["operation_length_penalty"],
            "argument_role_scale": values["argument_role_scale"],
            "definition_relation_scale": values["definition_relation_scale"],
            "argument_pointer_scale": values["argument_pointer_scale"],
            "allow_computed_dependencies": values["allow_computed_dependencies"],
            "register_use_contract": values["register_use_contract"].to_dict(),
        }
        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["coefficient_sha256"] = _sha(coefficient)
        body["register_use_contract"] = values["register_use_contract"].to_dict()
        return replace(
            self,
            **values,
            training_receipt={**body, "receipt_sha256": _sha(body)},
        )

    def coefficient_lesion(self) -> CompositionalSemanticProgramTransducer:
        zero_pointer = lambda head: LinearPointerHead(  # noqa: E731
            np.zeros_like(head.start_weight),
            head.start_bias,
            np.zeros_like(head.end_weight),
            head.end_bias,
        )
        operation_component = self.operation_head.heads[0]
        return self._with_coefficients(
            operation_pointer=zero_pointer(self.operation_pointer),
            argument_pointer=zero_pointer(self.argument_pointer),
            definition_pointer=zero_pointer(self.definition_pointer),
            operation_head=MultiViewClassifierHead(
                self.operation_head.modes,
                (
                    LinearClassifierHead(
                        operation_component.labels,
                        np.zeros_like(operation_component.weight),
                        operation_component.bias,
                    ),
                ),
            ),
            argument_role_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_role_heads
            ),
            definition_relation_head=DirectionalRelationHead(
                np.zeros_like(self.definition_relation_head.weight),
                self.definition_relation_head.bias,
                self.definition_relation_head.pointer_scale,
            ),
            register_use_contract=RegisterUseContract(
                0,
                self.max_steps * len(self.argument_role_heads),
                0,
                self.max_steps * len(self.argument_role_heads),
                False,
            ),
        )

    def relation_lesion(self) -> CompositionalSemanticProgramTransducer:
        zero_definition_pointer = LinearPointerHead(
            np.zeros_like(self.definition_pointer.start_weight),
            self.definition_pointer.start_bias,
            np.zeros_like(self.definition_pointer.end_weight),
            self.definition_pointer.end_bias,
        )
        return self._with_coefficients(
            definition_pointer=zero_definition_pointer,
            argument_role_heads=tuple(
                LinearArgumentRoleHead(np.zeros_like(head.weight), head.bias)
                for head in self.argument_role_heads
            ),
            definition_relation_head=DirectionalRelationHead(
                np.zeros_like(self.definition_relation_head.weight),
                self.definition_relation_head.bias,
                self.definition_relation_head.pointer_scale,
            ),
        )

    def dependency_lesion(self) -> CompositionalSemanticProgramTransducer:
        return self._with_coefficients(allow_computed_dependencies=False)

    def chart_beam_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Keep every learned coefficient but consult only the first chart."""

        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["operation_chart_beam"] = 1
        return replace(
            self,
            operation_chart_beam=1,
            training_receipt={**body, "receipt_sha256": _sha(body)},
        )

    def register_use_lesion(self) -> CompositionalSemanticProgramTransducer:
        """Remove only the source-learned graph-use bounds."""

        return self._with_coefficients(
            register_use_contract=RegisterUseContract(
                0,
                self.max_steps * len(self.argument_role_heads),
                0,
                self.max_steps * len(self.argument_role_heads),
                False,
            )
        )

    def decode(
        self,
        *,
        source_token_ids: Sequence[int],
        hidden_states: Any,
        public_inputs: Sequence[SemanticValue],
        source_text_sha256: str,
        model_basis_sha256: str,
    ) -> SemanticTransductionOutcome:
        if model_basis_sha256 != self.model_basis_sha256:
            return SemanticTransductionOutcome(None, "model_basis_mismatch", {}, {})
        if not _is_sha256(source_text_sha256):
            return SemanticTransductionOutcome(None, "source_identity_invalid", {}, {})
        try:
            inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
            hidden = _hidden_array(hidden_states, expected_width=self.hidden_size)
        except ValueError as exc:
            return SemanticTransductionOutcome(None, str(exc), {}, {})
        tokens = tuple(source_token_ids)
        if hidden.shape[0] != len(tokens):
            return SemanticTransductionOutcome(None, "token_hidden_length_mismatch", {}, {})
        if not 1 <= len(inputs) <= self.max_inputs:
            return SemanticTransductionOutcome(None, "public_input_count_unsupported", {}, {})
        input_banks: list[tuple[tuple[TokenSpan, float], ...]] = []
        argument_pointer_scores = self.argument_pointer.score_sequence(hidden)
        for index, value in enumerate(inputs):
            spans = self.input_grounding.candidate_spans(tokens, value)
            if not spans:
                return SemanticTransductionOutcome(
                    None,
                    f"input_value_not_grounded:{index}",
                    {},
                    {},
                )
            input_banks.append(
                tuple((span, argument_pointer_scores.score_span(span)) for span in spans)
            )
        grounded = _joint_pointer_assignment(tuple(input_banks), ordered=False)
        if grounded is None:
            return SemanticTransductionOutcome(None, "input_pointer_assignment_failed", {}, {})
        input_spans, input_scores = grounded
        nodes = _operation_nodes(
            pointer=self.operation_pointer,
            classifier=self.operation_head,
            hidden=hidden,
            input_spans=input_spans,
            max_span_tokens=self.max_span_tokens,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
        )
        charts = _operation_chart_candidates(
            nodes,
            max_steps=self.max_steps,
            length_penalty=self.operation_length_penalty,
            limit=self.operation_chart_beam,
        )
        if not charts:
            return SemanticTransductionOutcome(None, "operation_chart_empty", {}, {})
        assigned = next(
            (
                candidate
                for selected in charts
                for candidate in (
                    _assign_typed_arguments(
                        model=self,
                        hidden=hidden,
                        inputs=inputs,
                        input_spans=input_spans,
                        operation_nodes=selected,
                        argument_pointer_scores=argument_pointer_scores,
                    ),
                )
                if candidate is not None
            ),
            None,
        )
        if assigned is None:
            return SemanticTransductionOutcome(None, "typed_argument_chart_empty", {}, {})
        selected = assigned.operation_nodes
        arguments = assigned.arguments
        argument_spans = assigned.argument_spans
        instructions = tuple(
            SemanticIRInstruction(
                op=node.operation,
                args=arguments[step],
                operation_span=node.span,
                argument_spans=argument_spans[step],
                depends_on=tuple(
                    sorted(
                        argument - len(inputs)
                        for argument in set(arguments[step])
                        if argument >= len(inputs)
                    )
                ),
            )
            for step, node in enumerate(selected)
        )
        try:
            ir = SemanticProgramIR(
                source_token_ids=tokens,
                source_text_sha256=source_text_sha256,
                input_spans=input_spans,
                instructions=instructions,
                report_value=len(inputs) + len(instructions) - 1,
                model_basis_receipt_sha256=model_basis_sha256,
                transducer_receipt_sha256=self.receipt_sha256,
            )
        except ValueError as exc:
            return SemanticTransductionOutcome(None, f"ir_rejected:{exc}", {}, {})
        pointer_scores = {
            **{f"input:{index}": score for index, score in enumerate(input_scores)},
            **{f"operation:{index}": node.pointer_score for index, node in enumerate(selected)},
        }
        confidences = {f"operation:{index}": node.confidence for index, node in enumerate(selected)}
        return SemanticTransductionOutcome(ir, "", pointer_scores, confidences)


def _select_operation_length_penalty(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    pointer: LinearPointerHead,
    classifier: MultiViewClassifierHead,
    max_steps: int,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    cached: list[
        tuple[
            SemanticTransducerTrainingExample,
            tuple[tuple[float, tuple[_OperationNode, ...]], ...],
        ]
    ] = []
    average_scores: list[float] = []
    for item in validation:
        nodes = _operation_nodes(
            pointer=pointer,
            classifier=classifier,
            hidden=item.hidden_states,
            input_spans=item.ir.input_spans,
            max_span_tokens=max_span_tokens,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        by_count = tuple(
            _best_nonoverlapping_nodes(nodes, count) for count in range(1, max_steps + 1)
        )
        average_scores.extend(
            score / count
            for count, (score, _selected) in enumerate(by_count, start=1)
            if math.isfinite(score)
        )
        cached.append((item, by_count))
    if not average_scores:
        raise ValueError("compositional operation chart has no validation candidates")
    penalties = np.linspace(
        min(average_scores) - 5.0,
        max(average_scores) + 5.0,
        _OPERATION_PENALTY_POINTS,
    )
    rows: list[dict[str, Any]] = []
    for raw_penalty in penalties:
        penalty = float(raw_penalty)
        span_exact = 0
        operation_exact = 0
        graph_exact = 0
        for item, by_count in cached:
            selected = _best_penalized_operation_chart(by_count, penalty=penalty)
            expected_spans = tuple(
                instruction.operation_span for instruction in item.ir.instructions
            )
            expected_operations = tuple(instruction.op for instruction in item.ir.instructions)
            observed_spans = tuple(node.span for node in selected)
            observed_operations = tuple(node.operation for node in selected)
            span_exact += int(observed_spans == expected_spans)
            operation_exact += int(observed_operations == expected_operations)
            graph_exact += int(
                (observed_spans, observed_operations) == (expected_spans, expected_operations)
            )
        rows.append(
            {
                "length_penalty": penalty,
                "graph_exact": graph_exact,
                "span_exact": span_exact,
                "operation_exact": operation_exact,
                "validation_examples": len(validation),
            }
        )
    winner = max(
        rows,
        key=lambda row: (
            row["graph_exact"],
            row["span_exact"],
            row["operation_exact"],
            -abs(row["length_penalty"]),
        ),
    )
    return float(winner["length_penalty"]), rows


def fit_compositional_semantic_program_transducer(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    input_grounding: SemanticInputGroundingContract,
) -> CompositionalSemanticProgramTransducer:
    """Fit local atom heads and calibrate only the source-side operation chart."""

    training = tuple(item for item in examples if item.split == "train")
    validation = tuple(item for item in examples if item.split == "validation")
    if not training or not validation:
        raise ValueError("compositional semantic fit needs train and validation examples")
    bases = {item.ir.model_basis_receipt_sha256 for item in training}
    tokenizers = {item.tokenizer_identity_sha256 for item in examples}
    channel_geometries = {(item.hidden_channels, item.hidden_channel_widths) for item in training}
    geometries = Counter(_geometry(item) for item in training)
    if (
        len(bases) != 1
        or len(channel_geometries) != 1
        or tokenizers != {input_grounding.tokenizer_identity_sha256}
        or len(geometries) < 2
    ):
        raise ValueError("compositional semantic neural bases or geometries differ")
    hidden_channels, hidden_channel_widths = next(iter(channel_geometries))
    hidden_size = sum(hidden_channel_widths)
    max_steps = max(len(item.ir.instructions) for item in training)
    max_inputs = max(item.ir.n_inputs for item in training)
    max_arity = max(
        len(instruction.args) for item in training for instruction in item.ir.instructions
    )
    max_span_tokens = max(
        span.end - span.start for item in training for span in _all_semantic_spans(item)
    )
    max_definition_span_tokens = max(
        span.end - span.start for item in training for span in _register_definition_spans(item)
    )
    max_argument_span_tokens_by_type = {
        "integer": 1,
        "integer_sequence": 1,
    }
    for item in training:
        for instruction in item.ir.instructions:
            signature = semantic_primitive_type_signature(instruction.op)
            if signature is None:
                raise ValueError(f"compositional primitive has no floor type: {instruction.op}")
            argument_types, _result_type = signature
            if len(argument_types) != len(instruction.argument_spans):
                raise ValueError("compositional primitive arity differs from its floor type")
            for argument_type, span in zip(
                argument_types,
                instruction.argument_spans,
                strict=True,
            ):
                max_argument_span_tokens_by_type[argument_type] = max(
                    max_argument_span_tokens_by_type[argument_type],
                    span.end - span.start,
                )
    operation_pointer = _fit_shared_pointer(
        training,
        spans=lambda item: tuple(
            instruction.operation_span for instruction in item.ir.instructions
        ),
    )
    argument_pointer = _fit_shared_pointer(
        training,
        spans=lambda item: tuple(
            span for instruction in item.ir.instructions for span in instruction.argument_spans
        ),
    )
    definition_pointer = _fit_shared_pointer(
        training,
        spans=_register_definition_spans,
    )
    register_use_contract = _fit_register_use_contract(training)
    operation_rows = tuple(
        (item, instruction) for item in training for instruction in item.ir.instructions
    )
    operation_head = MultiViewClassifierHead(
        (_OPERATION_MODE,),
        (
            _fit_classifier(
                np.stack(
                    [
                        _operation_feature(
                            item.hidden_states,
                            instruction.operation_span,
                            mode=_OPERATION_MODE,
                            hidden_channels=hidden_channels,
                            hidden_channel_widths=hidden_channel_widths,
                        )
                        for item, instruction in operation_rows
                    ]
                ),
                [instruction.op for item, instruction in operation_rows],
                sample_weight=_normalized_weights(
                    [1.0 / geometries[_geometry(item)] for item, _instruction in operation_rows]
                ),
            ),
        ),
    )
    argument_role_heads = _fit_argument_role_heads(
        training,
        max_arity=max_arity,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    item_weights = {id(item): 1.0 / geometries[_geometry(item)] for item in training}
    relation_weight, relation_bias = _fit_directional_relation_head(
        training,
        item_weights=item_weights,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    uncalibrated_relation_head = DirectionalRelationHead(
        relation_weight, relation_bias, 0.0
    )
    definition_pointer_scale, definition_pointer_rows = _select_definition_pointer_scale(
        validation,
        relation_head=uncalibrated_relation_head,
        definition_pointer=definition_pointer,
        max_definition_span_tokens=max_definition_span_tokens,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    definition_relation_head = DirectionalRelationHead(
        relation_weight,
        relation_bias,
        definition_pointer_scale,
    )
    operation_length_penalty, penalty_rows = _select_operation_length_penalty(
        validation,
        pointer=operation_pointer,
        classifier=operation_head,
        max_steps=max_steps,
        max_span_tokens=max_span_tokens,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    coefficient_body = {
        "operation_pointer": operation_pointer.to_dict(),
        "argument_pointer": argument_pointer.to_dict(),
        "definition_pointer": definition_pointer.to_dict(),
        "operation_head": operation_head.to_dict(),
        "argument_role_heads": [head.to_dict() for head in argument_role_heads],
        "definition_relation_head": definition_relation_head.to_dict(),
        "operation_length_penalty": operation_length_penalty,
        "argument_role_scale": 1.0,
        "definition_relation_scale": 1.0,
        "argument_pointer_scale": 0.5,
        "allow_computed_dependencies": True,
        "register_use_contract": register_use_contract.to_dict(),
    }
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA,
        "model_basis_sha256": next(iter(bases)),
        "input_grounding_sha256": input_grounding.contract_sha256,
        "training_example_count": len(training),
        "validation_example_count": len(validation),
        "training_example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in training)
        ),
        "validation_example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in validation)
        ),
        "observed_geometry_support": [
            {"geometry": _geometry_name(geometry), "example_count": count}
            for geometry, count in sorted(geometries.items())
        ],
        "primitive_support": sorted(
            {instruction.op for item in training for instruction in item.ir.instructions}
        ),
        "operation_length_penalty_selection": penalty_rows,
        "chart_decoder": "directional_joint_probability_kbest_interval_dag_v3",
        "operation_chart_beam": _OPERATION_CHART_BEAM,
        "argument_span_bounds": max_argument_span_tokens_by_type,
        "definition_span_bound": max_definition_span_tokens,
        "definition_pointer_scale_selection": definition_pointer_rows,
        "register_use_contract": register_use_contract.to_dict(),
        "global_geometry_classifier_present": False,
        "step_indexed_heads_present": False,
        "family_router_present": False,
        "expected_answers_available": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "correctness_authority": False,
        "coefficient_sha256": _sha(coefficient_body),
    }
    return CompositionalSemanticProgramTransducer(
        hidden_size=hidden_size,
        model_basis_sha256=next(iter(bases)),
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
        input_grounding=input_grounding,
        operation_pointer=operation_pointer,
        argument_pointer=argument_pointer,
        definition_pointer=definition_pointer,
        operation_head=operation_head,
        argument_role_heads=argument_role_heads,
        definition_relation_head=definition_relation_head,
        max_steps=max_steps,
        max_inputs=max_inputs,
        max_span_tokens=max_span_tokens,
        max_definition_span_tokens=max_definition_span_tokens,
        max_argument_span_tokens_by_type=max_argument_span_tokens_by_type,
        register_use_contract=register_use_contract,
        operation_chart_beam=_OPERATION_CHART_BEAM,
        operation_length_penalty=operation_length_penalty,
        argument_role_scale=1.0,
        definition_relation_scale=1.0,
        argument_pointer_scale=0.5,
        allow_computed_dependencies=True,
        training_receipt={**body, "receipt_sha256": _sha(body)},
    )


def _pointer_from_dict(value: Any) -> LinearPointerHead:
    if not isinstance(value, Mapping):
        raise ValueError("compositional pointer payload is invalid")
    return LinearPointerHead(
        np.asarray(value["start_weight"], dtype=np.float32),
        float(value["start_bias"]),
        np.asarray(value["end_weight"], dtype=np.float32),
        float(value["end_bias"]),
    )


def compositional_semantic_program_transducer_from_dict(
    payload: Any,
) -> CompositionalSemanticProgramTransducer:
    """Reload one immutable v11 transducer without fitting or calibration."""

    if not isinstance(payload, Mapping):
        raise ValueError("compositional semantic transducer payload is invalid")
    operation = payload.get("operation_head")
    if (
        not isinstance(operation, Mapping)
        or operation.get("schema") != "aura.semantic_program_multiview_classifier.v1"
        or operation.get("modes") != [_OPERATION_MODE]
        or not isinstance(operation.get("heads"), list)
        or len(operation["heads"]) != 1
    ):
        raise ValueError("compositional operation head payload is invalid")
    raw_head = operation["heads"][0]
    operation_head = MultiViewClassifierHead(
        (_OPERATION_MODE,),
        (
            LinearClassifierHead(
                tuple(raw_head["labels"]),
                np.asarray(raw_head["weight"], dtype=np.float32),
                np.asarray(raw_head["bias"], dtype=np.float32),
            ),
        ),
    )
    relation = payload["definition_relation_head"]
    return CompositionalSemanticProgramTransducer(
        hidden_size=int(payload["hidden_size"]),
        model_basis_sha256=str(payload["model_basis_sha256"]),
        hidden_channels=tuple(payload["hidden_channels"]),
        hidden_channel_widths=tuple(int(value) for value in payload["hidden_channel_widths"]),
        input_grounding=semantic_input_grounding_contract_from_dict(payload["input_grounding"]),
        operation_pointer=_pointer_from_dict(payload["operation_pointer"]),
        argument_pointer=_pointer_from_dict(payload["argument_pointer"]),
        definition_pointer=_pointer_from_dict(payload["definition_pointer"]),
        operation_head=operation_head,
        argument_role_heads=tuple(
            LinearArgumentRoleHead(
                np.asarray(value["weight"], dtype=np.float32),
                float(value["bias"]),
            )
            for value in payload["argument_role_heads"]
        ),
        definition_relation_head=DirectionalRelationHead(
            np.asarray(relation["weight"], dtype=np.float32),
            float(relation["bias"]),
            float(relation["pointer_scale"]),
        ),
        max_steps=int(payload["max_steps"]),
        max_inputs=int(payload["max_inputs"]),
        max_span_tokens=int(payload["max_span_tokens"]),
        max_definition_span_tokens=int(payload["max_definition_span_tokens"]),
        max_argument_span_tokens_by_type={
            str(key): int(value)
            for key, value in payload["max_argument_span_tokens_by_type"].items()
        },
        register_use_contract=RegisterUseContract(
            **{
                str(key): value
                for key, value in payload["register_use_contract"].items()
            }
        ),
        operation_chart_beam=int(
            payload.get(
                "operation_chart_beam",
                payload["training_receipt"]["operation_chart_beam"],
            )
        ),
        operation_length_penalty=float(payload["operation_length_penalty"]),
        argument_role_scale=float(payload["argument_role_scale"]),
        definition_relation_scale=float(payload["definition_relation_scale"]),
        argument_pointer_scale=float(payload["argument_pointer_scale"]),
        allow_computed_dependencies=bool(payload["allow_computed_dependencies"]),
        training_receipt=dict(payload["training_receipt"]),
    )


__all__ = [
    "COMPOSITIONAL_SEMANTIC_RECEIPT_SCHEMA",
    "COMPOSITIONAL_SEMANTIC_TRANSDUCER_SCHEMA",
    "CompositionalSemanticProgramTransducer",
    "DirectionalRelationHead",
    "LinearArgumentRoleHead",
    "RegisterUseContract",
    "compositional_semantic_program_transducer_from_dict",
    "fit_compositional_semantic_program_transducer",
]
