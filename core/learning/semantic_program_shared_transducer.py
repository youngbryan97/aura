"""Variable-geometry semantic transduction with one shared neural head bank."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.procedure_induction import PRIMITIVES_BY_NAME
from core.learning.semantic_input_grounding import (
    SemanticInputGroundingContract,
    semantic_input_grounding_contract_from_dict,
)
from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)
from core.learning.semantic_program_transducer import (
    LinearClassifierHead,
    LinearPointerHead,
    MultiViewClassifierHead,
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
    _fit_binary_head,
    _fit_classifier,
    _hidden_array,
    _joint_pointer_assignment,
    _operation_feature,
    _pool,
)

SHARED_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v6"
SHARED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v6"
SHARED_SEMANTIC_GEOMETRY_SCHEMA: Final = "aura.semantic_program_geometry_contract.v1"
_OPERATION_MODE: Final = "contextual_mean"
_ARGUMENT_CANDIDATES: Final = 64
_ARGUMENT_BEAM: Final = 1024
_MAX_INPUTS: Final = 8
_MAX_STEPS: Final = 16
_POINTER_HARD_NEGATIVES: Final = 12
_POINTER_MAX_ITER: Final = 250
_POINTER_TOLERANCE: Final = 1e-3
_RELATION_CHANNEL: Final = "middle_causal_hidden"
_RELATION_POINTER_SCALES: Final = (0.5, 1.0, 2.0, 4.0)


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


def _geometry(item: SemanticTransducerTrainingExample) -> tuple[int, int]:
    return item.ir.n_inputs, len(item.ir.instructions)


def _geometry_name(value: tuple[int, int]) -> str:
    return f"inputs:{value[0]}|steps:{value[1]}"


def _relative_register(register: int, *, input_count: int) -> str:
    return f"input:{register}" if register < input_count else f"result:{register - input_count}"


def _roles_for_bounds(
    *,
    max_inputs: int,
    max_steps: int,
    max_arity: int,
) -> tuple[str, ...]:
    return (
        *(f"input:{index}" for index in range(max_inputs)),
        *(f"operation:{step}" for step in range(max_steps)),
        *(
            f"argument:{step}:{position}"
            for step in range(max_steps)
            for position in range(max_arity)
        ),
    )


def _roles_for_example(item: SemanticTransducerTrainingExample) -> tuple[str, ...]:
    return (
        *(f"input:{index}" for index in range(item.ir.n_inputs)),
        *(f"operation:{step}" for step in range(len(item.ir.instructions))),
        *(
            f"argument:{step}:{position}"
            for step, instruction in enumerate(item.ir.instructions)
            for position in range(len(instruction.args))
        ),
    )


def _gold_span(item: SemanticTransducerTrainingExample, role: str) -> TokenSpan:
    category, raw_step, *raw_position = role.split(":")
    step = int(raw_step)
    if category == "input":
        return item.ir.input_spans[step]
    if category == "operation":
        return item.ir.instructions[step].operation_span
    if category == "argument" and len(raw_position) == 1:
        return item.ir.instructions[step].argument_spans[int(raw_position[0])]
    raise ValueError("shared semantic pointer role is invalid")


def _balanced_item_weights(
    items: Sequence[SemanticTransducerTrainingExample],
) -> dict[int, float]:
    counts = Counter(_geometry(item) for item in items)
    return {id(item): 1.0 / counts[_geometry(item)] for item in items}


def _normalized_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim != 1 or weights.size < 1 or np.any(weights <= 0):
        raise ValueError("shared semantic fit weights are invalid")
    return np.asarray(weights * (weights.size / float(np.sum(weights))), dtype=np.float64)


def _pointer_boundary(span: TokenSpan, *, end: bool) -> int:
    return span.end - 1 if end else span.start


def _pointer_training_indices(
    item: SemanticTransducerTrainingExample,
    role: str,
    *,
    end: bool,
) -> tuple[int, ...]:
    """Keep the positive boundary and deterministic semantic hard negatives."""

    positive = _pointer_boundary(_gold_span(item, role), end=end)
    candidates: list[int] = []
    for competing_role in _roles_for_example(item):
        if competing_role == role:
            continue
        candidates.append(_pointer_boundary(_gold_span(item, competing_role), end=end))
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
    )
    if len(negatives) > _POINTER_HARD_NEGATIVES:
        selected = np.linspace(
            0,
            len(negatives) - 1,
            _POINTER_HARD_NEGATIVES,
            dtype=np.int64,
        )
        negatives = tuple(negatives[int(index)] for index in selected)
    return (positive, *negatives)


def _fit_pointer_boundary(
    supported: Sequence[SemanticTransducerTrainingExample],
    *,
    role: str,
    end: bool,
    item_weights: Mapping[int, float],
) -> tuple[np.ndarray, float]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for item in supported:
        indices = _pointer_training_indices(item, role, end=end)
        features.extend(item.hidden_states[index] for index in indices)
        labels.extend((1, *(0 for _ in indices[1:])))
        weights.extend([item_weights[id(item)] / len(indices)] * len(indices))
    return _fit_binary_head(
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        sample_weight=_normalized_weights(weights),
        max_iter=_POINTER_MAX_ITER,
        tolerance=_POINTER_TOLERANCE,
    )


def _lesion_classifier(head: LinearClassifierHead) -> LinearClassifierHead:
    return LinearClassifierHead(
        head.labels,
        np.zeros_like(head.weight),
        np.array(head.bias, copy=True),
    )


def _channel_span(
    name: str,
    *,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[int, int]:
    try:
        index = tuple(hidden_channels).index(name)
    except ValueError as exc:
        raise ValueError(f"shared semantic relation needs missing channel: {name}") from exc
    start = sum(hidden_channel_widths[:index])
    return start, start + hidden_channel_widths[index]


def _relation_span_vector(
    hidden: np.ndarray,
    span: TokenSpan,
    *,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> np.ndarray:
    start, end = _channel_span(
        _RELATION_CHANNEL,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    return _pool(hidden[:, start:end], span)


def _relation_feature(reference: np.ndarray, definition: np.ndarray) -> np.ndarray:
    if reference.shape != definition.shape or reference.ndim != 1:
        raise ValueError("shared semantic relation vectors differ")
    return np.concatenate(
        (reference * definition, np.abs(reference - definition))
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class LinearRelationHead:
    """One shared binary linker from a reference span to a definition span."""

    weight: np.ndarray
    bias: float
    pointer_scale: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if (
            weight.size < 2
            or weight.size % 2
            or not np.all(np.isfinite(weight))
            or not np.isfinite(self.bias)
            or not np.isfinite(self.pointer_scale)
            or self.pointer_scale <= 0.0
        ):
            raise ValueError("shared semantic relation head is invalid")
        object.__setattr__(self, "weight", weight)

    @property
    def channel_width(self) -> int:
        return int(self.weight.size // 2)

    def score(self, reference: np.ndarray, definition: np.ndarray) -> float:
        feature = _relation_feature(reference, definition)
        if feature.shape != self.weight.shape:
            raise ValueError("shared semantic relation feature width differs")
        return float(feature @ self.weight + self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.tolist(),
            "bias": float(self.bias),
            "pointer_scale": float(self.pointer_scale),
        }


@dataclass(frozen=True, slots=True)
class _ArgumentCandidate:
    span: TokenSpan
    register: int
    pointer_score: float
    confidence: float
    score: float


def _argument_candidates(
    *,
    pointer_head: LinearPointerHead,
    relation_head: LinearRelationHead,
    hidden: np.ndarray,
    input_spans: Sequence[TokenSpan],
    operation_spans: Sequence[TokenSpan],
    current_step: int,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[_ArgumentCandidate, ...]:
    proposals = list(
        pointer_head.decode_candidates(
            hidden,
            limit=_ARGUMENT_CANDIDATES,
            max_span_tokens=max_span_tokens,
        )
    )
    observed = {span for span, _ in proposals}
    for span in input_spans:
        if span not in observed:
            proposals.append((span, pointer_head.score_span(hidden, span)))
    definitions = (*input_spans, *operation_spans[:current_step])
    definition_vectors = tuple(
        _relation_span_vector(
            hidden,
            span,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        for span in definitions
    )
    by_register: dict[int, _ArgumentCandidate] = {}
    for span, pointer_score in proposals:
        reference = _relation_span_vector(
            hidden,
            span,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        for register, definition in enumerate(definition_vectors):
            relation_score = relation_head.score(reference, definition)
            score = relation_score + relation_head.pointer_scale * pointer_score
            confidence = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, relation_score))))
            candidate = _ArgumentCandidate(
                span=span,
                register=register,
                pointer_score=pointer_score,
                confidence=confidence,
                score=score,
            )
            previous = by_register.get(register)
            if previous is None or (
                candidate.score,
                -candidate.span.start,
                -candidate.span.end,
            ) > (
                previous.score,
                -previous.span.start,
                -previous.span.end,
            ):
                by_register[register] = candidate
    return tuple(
        sorted(
            by_register.values(),
            key=lambda item: (-item.score, item.register, item.span.start, item.span.end),
        )
    )


def _assign_arguments(
    *,
    pointer_heads: Mapping[str, LinearPointerHead],
    relation_head: LinearRelationHead,
    arities: Sequence[int],
    hidden: np.ndarray,
    input_spans: Sequence[TokenSpan],
    operation_spans: Sequence[TokenSpan],
    input_count: int,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> (
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[TokenSpan, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]
    | None
):
    states: list[
        tuple[
            float,
            tuple[tuple[int, ...], ...],
            tuple[tuple[TokenSpan, ...], ...],
            tuple[tuple[float, ...], ...],
            tuple[tuple[float, ...], ...],
        ]
    ] = [(0.0, (), (), (), ())]
    for step, arity in enumerate(arities):
        partial: list[
            tuple[
                float,
                tuple[int, ...],
                tuple[TokenSpan, ...],
                tuple[float, ...],
                tuple[float, ...],
            ]
        ] = [(0.0, (), (), (), ())]
        for position in range(arity):
            role = f"argument:{step}:{position}"
            candidates = _argument_candidates(
                pointer_head=pointer_heads[role],
                relation_head=relation_head,
                hidden=hidden,
                input_spans=input_spans,
                operation_spans=operation_spans,
                current_step=step,
                max_span_tokens=max_span_tokens,
                hidden_channels=hidden_channels,
                hidden_channel_widths=hidden_channel_widths,
            )
            if not candidates:
                return None
            partial = [
                (
                    total + candidate.score,
                    (*registers, candidate.register),
                    (*spans, candidate.span),
                    (*pointer_scores, candidate.pointer_score),
                    (*confidences, candidate.confidence),
                )
                for total, registers, spans, pointer_scores, confidences in partial
                for candidate in candidates
            ]
            partial.sort(
                key=lambda value: (
                    -value[0],
                    value[1],
                    tuple((span.start, span.end) for span in value[2]),
                )
            )
            partial = partial[:_ARGUMENT_BEAM]
        expanded = [
            (
                total + step_score,
                (*arguments, step_arguments),
                (*spans, step_spans),
                (*pointer_scores, step_pointer_scores),
                (*confidences, step_confidences),
            )
            for total, arguments, spans, pointer_scores, confidences in states
            for (
                step_score,
                step_arguments,
                step_spans,
                step_pointer_scores,
                step_confidences,
            ) in partial
        ]
        expanded.sort(key=lambda value: (-value[0], value[1]))
        states = expanded[:_ARGUMENT_BEAM]
    terminal = input_count + len(arities) - 1
    expected_outputs = set(range(input_count, terminal + 1))
    for _, arguments, spans, pointer_scores, confidences in states:
        required = {terminal}
        for step in range(len(arities) - 1, -1, -1):
            if input_count + step in required:
                required.update(arguments[step])
        if expected_outputs.issubset(required):
            return arguments, spans, pointer_scores, confidences
    return None


def _fit_relation_head(
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
        definitions = (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
        )
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
                    features.append(_relation_feature(reference, definition))
                    labels.append(int(candidate_register == register))
                    weights.append(item_weights[id(item)] / len(available))
    return _fit_binary_head(
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        sample_weight=_normalized_weights(weights),
        max_iter=400,
        tolerance=1e-5,
    )


def _validation_relation_assignment(
    item: SemanticTransducerTrainingExample,
    *,
    pointer_heads: Mapping[str, LinearPointerHead],
    relation_weight: np.ndarray,
    relation_bias: float,
    pointer_scale: float,
    input_grounding: SemanticInputGroundingContract,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[int, ...] | None:
    input_banks: list[tuple[tuple[TokenSpan, float], ...]] = []
    tokens = tuple(item.ir.source_token_ids)
    for input_index, value in enumerate(item.public_inputs):
        role = f"input:{input_index}"
        spans = input_grounding.candidate_spans(tokens, value)
        if not spans:
            return None
        input_banks.append(
            tuple(
                (span, pointer_heads[role].score_span(item.hidden_states, span))
                for span in spans
            )
        )
    assignment = _joint_pointer_assignment(tuple(input_banks), ordered=False)
    if assignment is None:
        return None
    input_spans = assignment[0]
    definitions = (
        *input_spans,
        *(instruction.operation_span for instruction in item.ir.instructions),
    )
    definition_vectors = tuple(
        _relation_span_vector(
            item.hidden_states,
            span,
            hidden_channels=hidden_channels,
            hidden_channel_widths=hidden_channel_widths,
        )
        for span in definitions
    )
    predicted: list[int] = []
    for step, instruction in enumerate(item.ir.instructions):
        for position in range(len(instruction.args)):
            role = f"argument:{step}:{position}"
            proposals = list(
                pointer_heads[role].decode_candidates(
                    item.hidden_states,
                    limit=_ARGUMENT_CANDIDATES,
                    max_span_tokens=max_span_tokens,
                )
            )
            observed = {span for span, _score in proposals}
            proposals.extend(
                (span, pointer_heads[role].score_span(item.hidden_states, span))
                for span in input_spans
                if span not in observed
            )
            scored: list[tuple[float, int, TokenSpan]] = []
            for reference_span, pointer_score in proposals:
                reference = _relation_span_vector(
                    item.hidden_states,
                    reference_span,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                for register, definition in enumerate(
                    definition_vectors[: item.ir.n_inputs + step]
                ):
                    score = float(
                        _relation_feature(reference, definition) @ relation_weight
                        + relation_bias
                        + pointer_scale * pointer_score
                    )
                    scored.append((score, register, reference_span))
            if not scored:
                return None
            predicted.append(
                max(
                    scored,
                    key=lambda value: (
                        value[0],
                        -value[1],
                        -value[2].start,
                        -value[2].end,
                    ),
                )[1]
            )
    return tuple(predicted)


def _select_relation_pointer_scale(
    validation: Sequence[SemanticTransducerTrainingExample],
    *,
    pointer_heads: Mapping[str, LinearPointerHead],
    relation_weight: np.ndarray,
    relation_bias: float,
    input_grounding: SemanticInputGroundingContract,
    max_span_tokens: int,
    hidden_channels: Sequence[str],
    hidden_channel_widths: Sequence[int],
) -> tuple[float, list[dict[str, Any]]]:
    if not validation:
        raise ValueError("shared semantic relation scale needs validation examples")
    rows: list[dict[str, Any]] = []
    for scale in _RELATION_POINTER_SCALES:
        exact = 0
        for item in validation:
            predicted = _validation_relation_assignment(
                item,
                pointer_heads=pointer_heads,
                relation_weight=relation_weight,
                relation_bias=relation_bias,
                pointer_scale=scale,
                input_grounding=input_grounding,
                max_span_tokens=max_span_tokens,
                hidden_channels=hidden_channels,
                hidden_channel_widths=hidden_channel_widths,
            )
            expected = tuple(
                register
                for instruction in item.ir.instructions
                for register in instruction.args
            )
            exact += int(predicted == expected)
        rows.append(
            {
                "pointer_scale": scale,
                "complete_argument_graphs": exact,
                "validation_examples": len(validation),
            }
        )
    winner = max(
        rows,
        key=lambda row: (row["complete_argument_graphs"], -row["pointer_scale"]),
    )
    return float(winner["pointer_scale"]), rows


@dataclass(frozen=True, slots=True)
class SharedSemanticProgramTransducer:
    """One head bank that infers bounded program geometry per request."""

    hidden_size: int
    model_basis_sha256: str
    hidden_channels: tuple[str, ...]
    hidden_channel_widths: tuple[int, ...]
    geometry_contract: dict[str, Any]
    input_grounding: SemanticInputGroundingContract
    pointer_heads: dict[str, LinearPointerHead]
    step_presence_heads: tuple[LinearClassifierHead, ...]
    operation_head: MultiViewClassifierHead
    relation_head: LinearRelationHead
    training_receipt: dict[str, Any]
    schema: str = SHARED_SEMANTIC_TRANSDUCER_SCHEMA

    def __post_init__(self) -> None:
        contract = json.loads(_canonical_bytes(self.geometry_contract))
        receipt = json.loads(_canonical_bytes(self.training_receipt))
        pointers = dict(self.pointer_heads)
        if (
            self.schema != SHARED_SEMANTIC_TRANSDUCER_SCHEMA
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or not _is_sha256(self.model_basis_sha256)
            or not self.hidden_channels
            or len(self.hidden_channels) != len(self.hidden_channel_widths)
            or sum(self.hidden_channel_widths) != self.hidden_size
            or not isinstance(self.input_grounding, SemanticInputGroundingContract)
            or contract.get("schema") != SHARED_SEMANTIC_GEOMETRY_SCHEMA
            or contract.get("register_encoding") != "role_relative_v1"
            or contract.get("input_signature_source") != "public_inputs"
            or contract.get("step_count_source") != "learned_monotone_presence"
            or contract.get("arity_source") != "predicted_primitive_signature"
            or contract.get("geometry_labels_available_to_decode") is not False
            or contract.get("pointer_fit_policy")
            != "positive_boundary_plus_semantic_hard_negatives_v1"
            or contract.get("pointer_hard_negative_limit") != _POINTER_HARD_NEGATIVES
        ):
            raise ValueError("shared semantic transducer envelope is invalid")
        max_inputs = contract.get("max_inputs")
        min_steps = contract.get("min_steps")
        max_steps = contract.get("max_steps")
        max_arity = contract.get("max_arity")
        max_span_tokens = contract.get("max_span_tokens")
        if (
            type(max_inputs) is not int
            or not 1 <= max_inputs <= _MAX_INPUTS
            or type(min_steps) is not int
            or type(max_steps) is not int
            or not 1 <= min_steps < max_steps <= _MAX_STEPS
            or type(max_arity) is not int
            or max_arity < 1
            or type(max_span_tokens) is not int
            or max_span_tokens < 1
            or len(self.step_presence_heads) != max_steps - min_steps
            or set(pointers)
            != set(
                _roles_for_bounds(
                    max_inputs=max_inputs,
                    max_steps=max_steps,
                    max_arity=max_arity,
                )
            )
            or any(head.width != self.hidden_size for head in pointers.values())
            or any(head.width != self.hidden_size for head in self.step_presence_heads)
            or self.operation_head.modes != (_OPERATION_MODE,)
            or _RELATION_CHANNEL not in self.hidden_channels
            or self.relation_head.channel_width
            != self.hidden_channel_widths[self.hidden_channels.index(_RELATION_CHANNEL)]
        ):
            raise ValueError("shared semantic transducer head geometry is invalid")
        coefficient_body = self._coefficient_body(pointer_heads=pointers)
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("schema") != SHARED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
            or receipt.get("receipt_sha256") != _sha(receipt_body)
            or receipt.get("model_basis_sha256") != self.model_basis_sha256
            or receipt.get("geometry_contract_sha256") != _sha(contract)
            or receipt.get("input_grounding_sha256")
            != self.input_grounding.contract_sha256
            or receipt.get("coefficient_sha256") != _sha(coefficient_body)
            or receipt.get("fit_weighting_policy") != "equal_geometry_then_equal_decision"
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
            raise ValueError("shared semantic transducer receipt differs")
        object.__setattr__(self, "geometry_contract", contract)
        object.__setattr__(self, "pointer_heads", pointers)
        object.__setattr__(self, "training_receipt", receipt)

    def _coefficient_body(
        self,
        *,
        pointer_heads: Mapping[str, LinearPointerHead] | None = None,
    ) -> dict[str, Any]:
        pointers = self.pointer_heads if pointer_heads is None else pointer_heads
        return {
            "pointer_heads": {role: pointers[role].to_dict() for role in sorted(pointers)},
            "step_presence_heads": [head.to_dict() for head in self.step_presence_heads],
            "operation_head": self.operation_head.to_dict(),
            "relation_head": self.relation_head.to_dict(),
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
            "geometry_contract": self.geometry_contract,
            "input_grounding": self.input_grounding.to_dict(),
            **self._coefficient_body(),
            "training_receipt": self.training_receipt,
        }

    def coefficient_lesion(self) -> SharedSemanticProgramTransducer:
        pointers = {
            role: LinearPointerHead(
                np.zeros_like(head.start_weight),
                head.start_bias,
                np.zeros_like(head.end_weight),
                head.end_bias,
            )
            for role, head in self.pointer_heads.items()
        }
        operation = MultiViewClassifierHead(
            self.operation_head.modes,
            tuple(_lesion_classifier(head) for head in self.operation_head.heads),
        )
        presence = tuple(_lesion_classifier(head) for head in self.step_presence_heads)
        relation = LinearRelationHead(
            np.zeros_like(self.relation_head.weight),
            0.0,
            self.relation_head.pointer_scale,
        )
        coefficient_body = {
            "pointer_heads": {role: pointers[role].to_dict() for role in sorted(pointers)},
            "step_presence_heads": [head.to_dict() for head in presence],
            "operation_head": operation.to_dict(),
            "relation_head": relation.to_dict(),
        }
        body = {
            key: value for key, value in self.training_receipt.items() if key != "receipt_sha256"
        }
        body["coefficient_sha256"] = _sha(coefficient_body)
        return SharedSemanticProgramTransducer(
            hidden_size=self.hidden_size,
            model_basis_sha256=self.model_basis_sha256,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
            geometry_contract=self.geometry_contract,
            input_grounding=self.input_grounding,
            pointer_heads=pointers,
            step_presence_heads=presence,
            operation_head=operation,
            relation_head=relation,
            training_receipt={**body, "receipt_sha256": _sha(body)},
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
        input_count = len(inputs)
        if not 1 <= input_count <= self.geometry_contract["max_inputs"]:
            return SemanticTransductionOutcome(None, "public_input_count_unsupported", {}, {})
        global_feature = _pool(hidden, TokenSpan(0, len(tokens)))
        step_count = int(self.geometry_contract["min_steps"])
        for head in self.step_presence_heads:
            label, _ = head.predict(global_feature)
            if label != "present":
                break
            step_count += 1
        max_arity = int(self.geometry_contract["max_arity"])
        roles = (
            *(f"input:{index}" for index in range(input_count)),
            *(f"operation:{step}" for step in range(step_count)),
            *(
                f"argument:{step}:{position}"
                for step in range(step_count)
                for position in range(max_arity)
            ),
        )
        candidates: dict[str, tuple[tuple[TokenSpan, float], ...]] = {}
        max_span_tokens = int(self.geometry_contract["max_span_tokens"])
        for role in roles:
            if role.startswith("input:"):
                input_index = int(role.partition(":")[2])
                grounded = self.input_grounding.candidate_spans(
                    tokens,
                    inputs[input_index],
                )
                if not grounded:
                    return SemanticTransductionOutcome(
                        None,
                        f"input_value_not_grounded:{input_index}",
                        {},
                        {},
                    )
                candidates[role] = tuple(
                    (span, self.pointer_heads[role].score_span(hidden, span))
                    for span in grounded
                )
            else:
                candidates[role] = self.pointer_heads[role].decode_candidates(
                    hidden,
                    limit=_ARGUMENT_CANDIDATES if role.startswith("argument:") else 1,
                    max_span_tokens=max_span_tokens,
                )
        input_roles = tuple(f"input:{index}" for index in range(input_count))
        input_assignment = _joint_pointer_assignment(
            tuple(candidates[role] for role in input_roles),
            ordered=False,
        )
        if input_assignment is None:
            return SemanticTransductionOutcome(None, "input_pointer_assignment_failed", {}, {})
        input_spans, input_scores = input_assignment
        pointer_scores = dict(zip(input_roles, input_scores, strict=True))
        operation_spans = tuple(candidates[f"operation:{step}"][0][0] for step in range(step_count))
        operations: list[str] = []
        confidences: dict[str, float] = {}
        for step, span in enumerate(operation_spans):
            operation, confidence = self.operation_head.predict(
                (
                    _operation_feature(
                        hidden,
                        span,
                        mode=_OPERATION_MODE,
                        hidden_channels=self.hidden_channels,
                        hidden_channel_widths=self.hidden_channel_widths,
                    ),
                )
            )
            operations.append(operation)
            confidences[f"operation:{step}"] = confidence
            pointer_scores[f"operation:{step}"] = candidates[f"operation:{step}"][0][1]
        try:
            arities = tuple(PRIMITIVES_BY_NAME[operation].arity for operation in operations)
        except KeyError:
            return SemanticTransductionOutcome(None, "primitive_signature_missing", {}, {})
        if any(arity > max_arity for arity in arities):
            return SemanticTransductionOutcome(None, "primitive_arity_unsupported", {}, {})
        assigned = _assign_arguments(
            pointer_heads=self.pointer_heads,
            relation_head=self.relation_head,
            arities=arities,
            hidden=hidden,
            input_spans=input_spans,
            operation_spans=operation_spans,
            input_count=input_count,
            max_span_tokens=max_span_tokens,
            hidden_channels=self.hidden_channels,
            hidden_channel_widths=self.hidden_channel_widths,
        )
        if assigned is None:
            return SemanticTransductionOutcome(
                None, "argument_assignment_failed", pointer_scores, confidences
            )
        arguments, argument_spans, argument_scores, argument_confidences = assigned
        for step, arity in enumerate(arities):
            for position in range(arity):
                role = f"argument:{step}:{position}"
                pointer_scores[role] = argument_scores[step][position]
                confidences[role] = argument_confidences[step][position]
        instructions = tuple(
            SemanticIRInstruction(
                op=operations[step],
                args=arguments[step],
                operation_span=operation_spans[step],
                argument_spans=argument_spans[step],
                depends_on=tuple(
                    sorted(
                        argument - input_count
                        for argument in set(arguments[step])
                        if argument >= input_count
                    )
                ),
            )
            for step in range(step_count)
        )
        try:
            ir = SemanticProgramIR(
                source_token_ids=tokens,
                source_text_sha256=source_text_sha256,
                input_spans=input_spans,
                instructions=instructions,
                report_value=input_count + step_count - 1,
                model_basis_receipt_sha256=model_basis_sha256,
                transducer_receipt_sha256=self.receipt_sha256,
            )
        except ValueError as exc:
            return SemanticTransductionOutcome(
                None, f"ir_rejected:{exc}", pointer_scores, confidences
            )
        return SemanticTransductionOutcome(ir, "", pointer_scores, confidences)


def fit_shared_semantic_program_transducer(
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    input_grounding: SemanticInputGroundingContract,
) -> SharedSemanticProgramTransducer:
    """Fit one geometry-balanced head bank over multiple program geometries."""

    training = tuple(item for item in examples if item.split == "train")
    if not training:
        raise ValueError("shared semantic transducer received no training examples")
    bases = {item.ir.model_basis_receipt_sha256 for item in training}
    tokenizer_identities = {item.tokenizer_identity_sha256 for item in examples}
    channel_geometries = {(item.hidden_channels, item.hidden_channel_widths) for item in training}
    geometries = Counter(_geometry(item) for item in training)
    if (
        len(bases) != 1
        or len(channel_geometries) != 1
        or tokenizer_identities != {input_grounding.tokenizer_identity_sha256}
    ):
        raise ValueError("shared semantic transducer neural bases differ")
    if len(geometries) < 2:
        raise ValueError("shared semantic transducer needs multiple geometries")
    hidden_channels, hidden_channel_widths = next(iter(channel_geometries))
    hidden_size = sum(hidden_channel_widths)
    if any(item.hidden_states.shape[1] != hidden_size for item in training):
        raise ValueError("shared semantic transducer hidden widths differ")
    max_inputs = max(item.ir.n_inputs for item in training)
    min_steps = min(len(item.ir.instructions) for item in training)
    max_steps = max(len(item.ir.instructions) for item in training)
    max_arity = max(
        len(instruction.args) for item in training for instruction in item.ir.instructions
    )
    max_span_tokens = max(
        span.end - span.start
        for item in training
        for span in (
            *item.ir.input_spans,
            *(instruction.operation_span for instruction in item.ir.instructions),
            *(
                span
                for instruction in item.ir.instructions
                for span in instruction.argument_spans
            ),
        )
    )
    if min_steps == max_steps:
        raise ValueError("shared semantic transducer has no learned step boundary")
    contract = {
        "schema": SHARED_SEMANTIC_GEOMETRY_SCHEMA,
        "max_inputs": max_inputs,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "max_arity": max_arity,
        "max_span_tokens": max_span_tokens,
        "observed_geometries": [_geometry_name(geometry) for geometry in sorted(geometries)],
        "register_encoding": "role_relative_v1",
        "input_signature_source": "public_inputs",
        "step_count_source": "learned_monotone_presence",
        "arity_source": "predicted_primitive_signature",
        "geometry_labels_available_to_decode": False,
        "pointer_fit_policy": "positive_boundary_plus_semantic_hard_negatives_v1",
        "pointer_hard_negative_limit": _POINTER_HARD_NEGATIVES,
    }
    item_weights = _balanced_item_weights(training)
    pointer_heads: dict[str, LinearPointerHead] = {}
    for role in _roles_for_bounds(
        max_inputs=max_inputs,
        max_steps=max_steps,
        max_arity=max_arity,
    ):
        supported = tuple(item for item in training if role in _roles_for_example(item))
        if not supported:
            raise ValueError(f"shared semantic pointer has no support: {role}")
        start_weight, start_bias = _fit_pointer_boundary(
            supported,
            role=role,
            end=False,
            item_weights=item_weights,
        )
        end_weight, end_bias = _fit_pointer_boundary(
            supported,
            role=role,
            end=True,
            item_weights=item_weights,
        )
        pointer_heads[role] = LinearPointerHead(start_weight, start_bias, end_weight, end_bias)
    global_features = np.stack(
        [
            _pool(item.hidden_states, TokenSpan(0, len(item.ir.source_token_ids)))
            for item in training
        ]
    )
    step_presence_heads = tuple(
        _fit_classifier(
            global_features,
            ["present" if len(item.ir.instructions) > step else "absent" for item in training],
            sample_weight=_normalized_weights([item_weights[id(item)] for item in training]),
        )
        for step in range(min_steps, max_steps)
    )
    operation_rows = tuple(
        (item, step) for item in training for step in range(len(item.ir.instructions))
    )
    operation_geometry_counts = Counter(_geometry(item) for item, _ in operation_rows)
    operation_features = np.stack(
        [
            _operation_feature(
                item.hidden_states,
                item.ir.instructions[step].operation_span,
                mode=_OPERATION_MODE,
                hidden_channels=hidden_channels,
                hidden_channel_widths=hidden_channel_widths,
            )
            for item, step in operation_rows
        ]
    )
    operation_linear = _fit_classifier(
        operation_features,
        [item.ir.instructions[step].op for item, step in operation_rows],
        sample_weight=_normalized_weights(
            [1.0 / operation_geometry_counts[_geometry(item)] for item, _ in operation_rows]
        ),
    )
    operation_head = MultiViewClassifierHead((_OPERATION_MODE,), (operation_linear,))
    relation_weight, relation_bias = _fit_relation_head(
        training,
        item_weights=item_weights,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    relation_pointer_scale, relation_scale_selection = _select_relation_pointer_scale(
        tuple(item for item in examples if item.split == "validation"),
        pointer_heads=pointer_heads,
        relation_weight=relation_weight,
        relation_bias=relation_bias,
        input_grounding=input_grounding,
        max_span_tokens=max_span_tokens,
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
    )
    relation_head = LinearRelationHead(
        relation_weight,
        relation_bias,
        relation_pointer_scale,
    )
    coefficient_body = {
        "pointer_heads": {role: pointer_heads[role].to_dict() for role in sorted(pointer_heads)},
        "step_presence_heads": [head.to_dict() for head in step_presence_heads],
        "operation_head": operation_head.to_dict(),
        "relation_head": relation_head.to_dict(),
    }
    geometry_support = [
        {
            "geometry": _geometry_name(geometry),
            "example_count": count,
            "constructions": sorted(
                {item.construction_id for item in training if _geometry(item) == geometry}
            ),
            "topologies": sorted(
                {item.topology_id for item in training if _geometry(item) == geometry}
            ),
        }
        for geometry, count in sorted(geometries.items())
    ]
    receipt_body = {
        "schema": SHARED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA,
        "model_basis_sha256": next(iter(bases)),
        "hidden_size": hidden_size,
        "hidden_channels": list(hidden_channels),
        "hidden_channel_widths": list(hidden_channel_widths),
        "training_example_count": len(training),
        "training_example_ids_sha256": _sha(
            sorted(item.ir.source_text_sha256 for item in training)
        ),
        "geometry_contract_sha256": _sha(contract),
        "input_grounding_sha256": input_grounding.contract_sha256,
        "observed_geometry_support": geometry_support,
        "primitive_support": sorted(
            {instruction.op for item in training for instruction in item.ir.instructions}
        ),
        "register_support": sorted(
            {
                _relative_register(argument, input_count=item.ir.n_inputs)
                for item in training
                for instruction in item.ir.instructions
                for argument in instruction.args
            }
        ),
        "operation_feature_mode": _OPERATION_MODE,
        "relation_feature_mode": "middle_pair_product_abs_difference_v1",
        "relation_pointer_scale_selection": relation_scale_selection,
        "fit_weighting_policy": "equal_geometry_then_equal_decision",
        "coefficient_sha256": _sha(coefficient_body),
        "expected_answers_available": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "correctness_authority": False,
    }
    return SharedSemanticProgramTransducer(
        hidden_size=hidden_size,
        model_basis_sha256=next(iter(bases)),
        hidden_channels=hidden_channels,
        hidden_channel_widths=hidden_channel_widths,
        geometry_contract=contract,
        input_grounding=input_grounding,
        pointer_heads=pointer_heads,
        step_presence_heads=step_presence_heads,
        operation_head=operation_head,
        relation_head=relation_head,
        training_receipt={**receipt_body, "receipt_sha256": _sha(receipt_body)},
    )


def _pointer_from_dict(value: Any) -> LinearPointerHead:
    if not isinstance(value, dict) or set(value) != {
        "start_weight",
        "start_bias",
        "end_weight",
        "end_bias",
    }:
        raise ValueError("shared semantic pointer payload is invalid")
    return LinearPointerHead(
        value["start_weight"],
        value["start_bias"],
        value["end_weight"],
        value["end_bias"],
    )


def _classifier_from_dict(value: Any) -> LinearClassifierHead:
    if not isinstance(value, dict) or set(value) != {"labels", "weight", "bias"}:
        raise ValueError("shared semantic classifier payload is invalid")
    return LinearClassifierHead(tuple(value["labels"]), value["weight"], value["bias"])


def _relation_from_dict(value: Any) -> LinearRelationHead:
    if not isinstance(value, dict) or set(value) != {
        "weight",
        "bias",
        "pointer_scale",
    }:
        raise ValueError("shared semantic relation payload is invalid")
    return LinearRelationHead(value["weight"], value["bias"], value["pointer_scale"])


def shared_semantic_program_transducer_from_dict(
    payload: Any,
) -> SharedSemanticProgramTransducer:
    """Load v6 only when its geometry, coefficients, and receipt still agree."""

    expected = {
        "schema",
        "hidden_size",
        "model_basis_sha256",
        "hidden_channels",
        "hidden_channel_widths",
        "geometry_contract",
        "input_grounding",
        "pointer_heads",
        "step_presence_heads",
        "operation_head",
        "relation_head",
        "training_receipt",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("shared semantic transducer payload is invalid")
    operation = payload["operation_head"]
    if (
        payload.get("schema") != SHARED_SEMANTIC_TRANSDUCER_SCHEMA
        or not isinstance(payload.get("pointer_heads"), dict)
        or not isinstance(payload.get("step_presence_heads"), list)
        or not isinstance(operation, dict)
        or operation.get("schema") != "aura.semantic_program_multiview_classifier.v1"
        or operation.get("modes") != [_OPERATION_MODE]
        or not isinstance(operation.get("heads"), list)
        or len(operation["heads"]) != 1
        or not isinstance(payload.get("relation_head"), dict)
    ):
        raise ValueError("shared semantic transducer payload fields are invalid")
    return SharedSemanticProgramTransducer(
        hidden_size=payload["hidden_size"],
        model_basis_sha256=payload["model_basis_sha256"],
        hidden_channels=tuple(payload["hidden_channels"]),
        hidden_channel_widths=tuple(payload["hidden_channel_widths"]),
        geometry_contract=payload["geometry_contract"],
        input_grounding=semantic_input_grounding_contract_from_dict(
            payload["input_grounding"]
        ),
        pointer_heads={
            role: _pointer_from_dict(value) for role, value in payload["pointer_heads"].items()
        },
        step_presence_heads=tuple(
            _classifier_from_dict(value) for value in payload["step_presence_heads"]
        ),
        operation_head=MultiViewClassifierHead(
            (_OPERATION_MODE,), (_classifier_from_dict(operation["heads"][0]),)
        ),
        relation_head=_relation_from_dict(payload["relation_head"]),
        training_receipt=payload["training_receipt"],
    )


__all__ = [
    "SHARED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA",
    "SHARED_SEMANTIC_TRANSDUCER_SCHEMA",
    "SharedSemanticProgramTransducer",
    "fit_shared_semantic_program_transducer",
    "shared_semantic_program_transducer_from_dict",
]
