"""Learn token-grounded program structure from resident hidden states.

This is a bounded neural transducer, not a phrase parser.  Linear pointer heads
locate source spans; learned classifiers assign primitive and register meaning
to those spans.  Exact span identity binds direct inputs and unambiguous prior
result references before the classifier fallback.  A deterministic decoder then
proposes ``SemanticProgramIR``, whose validator remains the authority on type
and causal structure.

The legacy contract covers three public inputs and two binary instructions.
The current contract derives input, step, and per-step argument geometry from
the training examples while retaining exact legacy serialization for published
binary models. It remains a bounded compiler over a fixed primitive vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_ir import (
    SemanticIRInstruction,
    SemanticProgramIR,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
)

LEGACY_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v1"
SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v2"
TYPED_SEMANTIC_TRANSDUCER_SCHEMA: Final = "aura.semantic_program_transducer.v3"
LEGACY_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v1"
SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v2"
TYPED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA: Final = "aura.semantic_program_transducer_receipt.v3"
SEMANTIC_TRANSDUCER_INPUTS: Final = 3
SEMANTIC_TRANSDUCER_STEPS: Final = 2
SEMANTIC_TRANSDUCER_MAX_SPAN_TOKENS: Final = 24

_MAX_TRANSDUCER_INPUTS: Final = 8
_MAX_TRANSDUCER_STEPS: Final = 16
_STRUCTURED_INPUT_CANDIDATES: Final = 64
_STRUCTURED_ARGUMENT_CANDIDATES: Final = 16
_STRUCTURED_POINTER_BEAM: Final = 512
_STRUCTURED_ARGUMENT_BEAM: Final = 1024


def _pointer_roles(
    input_count: int,
    argument_arities: Sequence[int],
) -> tuple[str, ...]:
    return (
        *(f"input:{index}" for index in range(input_count)),
        *(f"operation:{step}" for step in range(len(argument_arities))),
        *(
            f"argument:{step}:{position}"
            for step, arity in enumerate(argument_arities)
            for position in range(arity)
        ),
    )


_POINTER_ROLES: Final = _pointer_roles(
    SEMANTIC_TRANSDUCER_INPUTS,
    (2,) * SEMANTIC_TRANSDUCER_STEPS,
)

_RECEIPT_FIELDS: Final = {
    "schema",
    "model_basis_sha256",
    "hidden_size",
    "training_example_count",
    "training_constructions",
    "training_topologies",
    "primitive_support",
    "register_support",
    "coefficient_sha256",
    "expected_answers_available",
    "verifier_traces_available",
    "generated_compiler_text_available",
    "correctness_authority",
    "receipt_sha256",
}
_RECEIPT_FIELDS_V2: Final = _RECEIPT_FIELDS | {
    "classifier_sharing",
    "input_count",
    "shared_argument_support",
    "step_count",
}
_RECEIPT_FIELDS_V3: Final = _RECEIPT_FIELDS_V2 | {"argument_arities"}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hidden_array(value: Any, *, expected_width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("semantic transducer hidden states must be a non-empty matrix")
    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError("semantic transducer hidden width differs from its model")
    if not np.all(np.isfinite(array)):
        raise ValueError("semantic transducer hidden states must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-4):
        raise ValueError("semantic transducer hidden states must be unit normalized")
    return np.ascontiguousarray(array)


def _resolve_prior_result_register(
    *,
    token_ids: Sequence[int],
    reference_span: TokenSpan,
    operation_spans: Sequence[TokenSpan],
    current_step: int,
    input_count: int,
) -> int | None:
    """Bind one later mention to a unique earlier result-definition window."""

    if current_step < 1 or current_step >= len(operation_spans):
        return None
    tokens = tuple(int(token) for token in token_ids)
    needle = tokens[reference_span.start : reference_span.end]
    if not needle:
        return None
    candidate_steps: set[int] = set()
    for prior_step in range(current_step):
        lower = operation_spans[prior_step].end
        upper = operation_spans[prior_step + 1].start
        if lower >= upper or len(needle) > upper - lower:
            continue
        for start in range(lower, upper - len(needle) + 1):
            if tokens[start : start + len(needle)] == needle:
                candidate_steps.add(prior_step)
                break
    if len(candidate_steps) != 1:
        return None
    return input_count + candidate_steps.pop()


@dataclass(frozen=True, slots=True)
class SemanticTransducerTrainingExample:
    """Gold IR paired only with answer-blind resident hidden evidence."""

    ir: SemanticProgramIR
    hidden_states: np.ndarray
    split: str
    construction_id: str
    topology_id: str
    public_inputs: tuple[SemanticValue, ...]

    def __post_init__(self) -> None:
        hidden = _hidden_array(self.hidden_states)
        if hidden.shape[0] != len(self.ir.source_token_ids):
            raise ValueError("semantic transducer tokens and hidden rows differ")
        if (
            not 1 <= self.ir.n_inputs <= _MAX_TRANSDUCER_INPUTS
            or not 1 <= len(self.ir.instructions) <= _MAX_TRANSDUCER_STEPS
        ):
            raise ValueError("semantic transducer training geometry is unsupported")
        if not isinstance(self.public_inputs, tuple) or len(self.public_inputs) != self.ir.n_inputs:
            raise ValueError("semantic transducer public inputs are invalid")
        if any(normalize_semantic_value(value) != value for value in self.public_inputs):
            raise ValueError("semantic transducer public inputs are invalid")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("semantic transducer split is invalid")
        if not self.construction_id or not self.topology_id:
            raise ValueError("semantic transducer provenance is incomplete")
        object.__setattr__(self, "hidden_states", hidden)


@dataclass(frozen=True, slots=True)
class LinearPointerHead:
    """Independent learned start/end scorers for one semantic role."""

    start_weight: np.ndarray
    start_bias: float
    end_weight: np.ndarray
    end_bias: float

    def __post_init__(self) -> None:
        start = np.asarray(self.start_weight, dtype=np.float32).reshape(-1)
        end = np.asarray(self.end_weight, dtype=np.float32).reshape(-1)
        if (
            start.shape != end.shape
            or start.size < 1
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or not np.isfinite(self.start_bias)
            or not np.isfinite(self.end_bias)
        ):
            raise ValueError("semantic pointer head parameters are invalid")
        object.__setattr__(self, "start_weight", start)
        object.__setattr__(self, "end_weight", end)

    @property
    def width(self) -> int:
        return int(self.start_weight.size)

    def decode_candidates(
        self,
        hidden: np.ndarray,
        *,
        limit: int,
    ) -> tuple[tuple[TokenSpan, float], ...]:
        """Return the strongest distinct source spans in stable score order."""

        if type(limit) is not int or limit < 1:
            raise ValueError("semantic pointer candidate limit is invalid")
        matrix = _hidden_array(hidden, expected_width=self.width)
        start_scores = matrix @ self.start_weight + self.start_bias
        end_scores = matrix @ self.end_weight + self.end_bias
        candidates: list[tuple[TokenSpan, float]] = []
        for start in range(matrix.shape[0]):
            stop = min(
                matrix.shape[0],
                start + SEMANTIC_TRANSDUCER_MAX_SPAN_TOKENS,
            )
            for end in range(start, stop):
                candidates.append(
                    (
                        TokenSpan(start, end + 1),
                        float(start_scores[start] + end_scores[end]),
                    )
                )
        candidates.sort(key=lambda item: (-item[1], item[0].start, item[0].end))
        return tuple(candidates[:limit])

    def score_span(self, hidden: np.ndarray, span: TokenSpan) -> float:
        matrix = _hidden_array(hidden, expected_width=self.width)
        span.validate_bound(matrix.shape[0])
        start_score = float(matrix[span.start] @ self.start_weight + self.start_bias)
        end_score = float(matrix[span.end - 1] @ self.end_weight + self.end_bias)
        return start_score + end_score

    def decode(self, hidden: np.ndarray) -> tuple[TokenSpan, float]:
        return self.decode_candidates(hidden, limit=1)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_weight": self.start_weight.tolist(),
            "start_bias": float(self.start_bias),
            "end_weight": self.end_weight.tolist(),
            "end_bias": float(self.end_bias),
        }


@dataclass(frozen=True, slots=True)
class LinearClassifierHead:
    """Multiclass linear readout over one pooled semantic span."""

    labels: tuple[str, ...]
    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32).reshape(-1)
        if (
            len(self.labels) < 2
            or len(set(self.labels)) != len(self.labels)
            or weight.ndim != 2
            or weight.shape[0] != len(self.labels)
            or weight.shape[1] < 1
            or bias.shape != (len(self.labels),)
            or not np.all(np.isfinite(weight))
            or not np.all(np.isfinite(bias))
        ):
            raise ValueError("semantic classifier head parameters are invalid")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    @property
    def width(self) -> int:
        return int(self.weight.shape[1])

    def predict(self, feature: np.ndarray) -> tuple[str, float]:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.shape != (self.width,) or not np.all(np.isfinite(vector)):
            raise ValueError("semantic classifier feature is invalid")
        logits = self.weight @ vector + self.bias
        winner = int(np.argmax(logits))
        shifted = logits - float(np.max(logits))
        probabilities = np.exp(shifted)
        probabilities /= float(np.sum(probabilities))
        return self.labels[winner], float(probabilities[winner])

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
        }


@dataclass(frozen=True, slots=True)
class SemanticTransductionOutcome:
    ir: SemanticProgramIR | None
    refusal: str
    pointer_scores: dict[str, float]
    classification_confidences: dict[str, float]

    @property
    def accepted(self) -> bool:
        return self.ir is not None


@dataclass(frozen=True, slots=True)
class _ArgumentCandidate:
    span: TokenSpan
    register: int
    pointer_score: float
    score: float
    confidence: float


def _spans_overlap(left: TokenSpan, right: TokenSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _joint_pointer_assignment(
    candidates: Sequence[Sequence[tuple[TokenSpan, float]]],
    *,
    ordered: bool,
) -> tuple[tuple[TokenSpan, ...], tuple[float, ...]] | None:
    """Choose one globally compatible span for every learned pointer role."""

    beam: list[tuple[float, tuple[TokenSpan, ...], tuple[float, ...]]] = [(0.0, (), ())]
    for role_candidates in candidates:
        expanded: list[tuple[float, tuple[TokenSpan, ...], tuple[float, ...]]] = []
        for total, spans, scores in beam:
            for span, score in role_candidates:
                if any(_spans_overlap(span, prior) for prior in spans):
                    continue
                if ordered and spans and spans[-1].end > span.start:
                    continue
                expanded.append((total + score, (*spans, span), (*scores, score)))
        if not expanded:
            return None
        expanded.sort(
            key=lambda item: (
                -item[0],
                tuple((span.start, span.end) for span in item[1]),
            )
        )
        beam = expanded[:_STRUCTURED_POINTER_BEAM]
    _, spans, scores = beam[0]
    return spans, scores


def _best_argument_candidates(
    *,
    pointer_candidates: Sequence[tuple[TokenSpan, float]],
    pointer_head: LinearPointerHead,
    classifier: LinearClassifierHead,
    hidden: np.ndarray,
    tokens: Sequence[int],
    input_spans: Sequence[TokenSpan],
    operation_spans: Sequence[TokenSpan],
    current_step: int,
    input_count: int,
) -> tuple[_ArgumentCandidate, ...]:
    """Collapse span proposals to the strongest causal proposal per register."""

    output_register = input_count + current_step
    by_register: dict[int, _ArgumentCandidate] = {}
    candidates = list(pointer_candidates)
    observed_spans = {span for span, _score in candidates}
    for input_span in input_spans:
        if input_span not in observed_spans:
            candidates.append((input_span, pointer_head.score_span(hidden, input_span)))
    for span, pointer_score in candidates:
        input_matches = [
            index for index, input_span in enumerate(input_spans) if span == input_span
        ]
        confidence: float
        if len(input_matches) == 1:
            register = input_matches[0]
            confidence = pointer_score
        else:
            prior_result = _resolve_prior_result_register(
                token_ids=tokens,
                reference_span=span,
                operation_spans=operation_spans,
                current_step=current_step,
                input_count=input_count,
            )
            if prior_result is not None:
                register = prior_result
                confidence = pointer_score
            else:
                label, classifier_confidence = classifier.predict(_pool(hidden, span))
                register = int(label)
                confidence = classifier_confidence
        if register < 0 or register >= output_register:
            continue
        score = pointer_score
        if 0.0 < confidence <= 1.0:
            score += math.log(confidence)
        candidate = _ArgumentCandidate(
            span,
            register,
            pointer_score,
            score,
            confidence,
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


def _structured_argument_assignment(
    *,
    pointer_candidates: dict[str, Sequence[tuple[TokenSpan, float]]],
    pointer_heads: dict[str, LinearPointerHead],
    argument_heads: Sequence[Sequence[LinearClassifierHead]],
    hidden: np.ndarray,
    tokens: Sequence[int],
    input_spans: Sequence[TokenSpan],
    operation_spans: Sequence[TokenSpan],
    input_count: int,
    argument_arities: Sequence[int] | None = None,
    step_count: int | None = None,
) -> (
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[TokenSpan, ...], ...],
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]
    | None
):
    """Find the strongest forward assignment that keeps every step causal."""

    if argument_arities is None:
        if type(step_count) is not int or step_count < 1:
            raise ValueError("structured argument geometry is missing")
        argument_arities = (2,) * step_count
    elif step_count is not None and step_count != len(argument_arities):
        raise ValueError("structured argument geometry disagrees")

    states: list[
        tuple[
            float,
            tuple[tuple[int, ...], ...],
            tuple[tuple[TokenSpan, ...], ...],
            tuple[tuple[float, ...], ...],
            tuple[tuple[float, ...], ...],
        ]
    ] = [(0.0, (), (), (), ())]
    for step, arity in enumerate(argument_arities):
        step_combinations: list[
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
            candidates = _best_argument_candidates(
                pointer_candidates=pointer_candidates[role],
                pointer_head=pointer_heads[role],
                classifier=argument_heads[step][position],
                hidden=hidden,
                tokens=tokens,
                input_spans=input_spans,
                operation_spans=operation_spans,
                current_step=step,
                input_count=input_count,
            )
            if not candidates:
                return None
            step_combinations = [
                (
                    total + candidate.score,
                    (*registers, candidate.register),
                    (*spans, candidate.span),
                    (*pointer_values, candidate.pointer_score),
                    (*confidence_values, candidate.confidence),
                )
                for total, registers, spans, pointer_values, confidence_values in step_combinations
                for candidate in candidates
            ]
            step_combinations.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    tuple((span.start, span.end) for span in item[2]),
                )
            )
            step_combinations = step_combinations[:_STRUCTURED_ARGUMENT_BEAM]
        expanded = []
        for total, arguments, spans, pointer_scores, confidences in states:
            for (
                step_score,
                step_arguments,
                step_spans,
                step_pointer_scores,
                step_confidences,
            ) in step_combinations:
                expanded.append(
                    (
                        total + step_score,
                        (*arguments, step_arguments),
                        (*spans, step_spans),
                        (*pointer_scores, step_pointer_scores),
                        (*confidences, step_confidences),
                    )
                )
        expanded.sort(
            key=lambda item: (
                -item[0],
                item[1],
                tuple(
                    tuple((span.start, span.end) for span in step_spans) for step_spans in item[2]
                ),
            )
        )
        states = expanded[:_STRUCTURED_ARGUMENT_BEAM]

    step_count = len(argument_arities)
    expected_outputs = set(range(input_count, input_count + step_count))
    terminal = input_count + step_count - 1
    for _, arguments, spans, pointer_scores, confidences in states:
        required = {terminal}
        for step in range(step_count - 1, -1, -1):
            output = input_count + step
            if output in required:
                required.update(arguments[step])
        if expected_outputs.issubset(required):
            return arguments, spans, pointer_scores, confidences
    return None


@dataclass(frozen=True, slots=True)
class SemanticProgramTransducer:
    """Learned, model-bound decoder for the bounded semantic IR contract."""

    hidden_size: int
    model_basis_sha256: str
    input_count: int
    step_count: int
    argument_arities: tuple[int, ...]
    pointer_heads: dict[str, LinearPointerHead]
    operation_heads: tuple[LinearClassifierHead, ...]
    argument_heads: tuple[tuple[LinearClassifierHead, ...], ...]
    training_receipt: dict[str, Any]
    schema: str = SEMANTIC_TRANSDUCER_SCHEMA

    def __post_init__(self) -> None:
        pointer_heads = dict(self.pointer_heads)
        receipt = json.loads(json.dumps(self.training_receipt, allow_nan=False))
        legacy = self.schema == LEGACY_SEMANTIC_TRANSDUCER_SCHEMA
        expected_schema = (
            LEGACY_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
            if legacy
            else TYPED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
            if self.schema == TYPED_SEMANTIC_TRANSDUCER_SCHEMA
            else SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
        )
        roles = _pointer_roles(self.input_count, self.argument_arities)
        if (
            self.schema
            not in {
                LEGACY_SEMANTIC_TRANSDUCER_SCHEMA,
                SEMANTIC_TRANSDUCER_SCHEMA,
                TYPED_SEMANTIC_TRANSDUCER_SCHEMA,
            }
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or not _is_sha256(self.model_basis_sha256)
            or type(self.input_count) is not int
            or not 1 <= self.input_count <= _MAX_TRANSDUCER_INPUTS
            or type(self.step_count) is not int
            or not 1 <= self.step_count <= _MAX_TRANSDUCER_STEPS
            or not isinstance(self.argument_arities, tuple)
            or len(self.argument_arities) != self.step_count
            or any(type(arity) is not int or arity not in {1, 2} for arity in self.argument_arities)
            or (legacy and (self.input_count, self.step_count) != (3, 2))
            or (
                self.schema in {LEGACY_SEMANTIC_TRANSDUCER_SCHEMA, SEMANTIC_TRANSDUCER_SCHEMA}
                and self.argument_arities != (2,) * self.step_count
            )
            or set(pointer_heads) != set(roles)
            or len(self.operation_heads) != self.step_count
            or len(self.argument_heads) != self.step_count
            or any(
                len(heads) != arity
                for heads, arity in zip(
                    self.argument_heads,
                    self.argument_arities,
                    strict=True,
                )
            )
            or receipt.get("schema") != expected_schema
        ):
            raise ValueError("semantic transducer envelope is invalid")
        all_heads = [*pointer_heads.values(), *self.operation_heads]
        all_heads.extend(head for heads in self.argument_heads for head in heads)
        if any(head.width != self.hidden_size for head in all_heads):
            raise ValueError("semantic transducer heads disagree on hidden width")
        expected_receipt_fields = (
            _RECEIPT_FIELDS
            if legacy
            else _RECEIPT_FIELDS_V3
            if self.schema == TYPED_SEMANTIC_TRANSDUCER_SCHEMA
            else _RECEIPT_FIELDS_V2
        )
        if set(receipt) != expected_receipt_fields:
            raise ValueError("semantic transducer receipt fields are invalid")
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != _sha(receipt_body)
            or receipt.get("model_basis_sha256") != self.model_basis_sha256
            or receipt.get("hidden_size") != self.hidden_size
            or receipt.get("coefficient_sha256")
            != _sha(self._coefficient_body(pointer_heads=pointer_heads))
            or type(receipt.get("training_example_count")) is not int
            or receipt["training_example_count"] < 1
            or not isinstance(receipt.get("training_constructions"), list)
            or not isinstance(receipt.get("training_topologies"), list)
            or not isinstance(receipt.get("primitive_support"), list)
            or not isinstance(receipt.get("register_support"), list)
            or len(receipt["register_support"]) != sum(self.argument_arities)
            or (
                not legacy
                and (
                    receipt.get("input_count") != self.input_count
                    or receipt.get("step_count") != self.step_count
                    or receipt.get("classifier_sharing") != "across_step_slots"
                    or receipt.get("shared_argument_support")
                    != [
                        sorted(
                            int(label)
                            for label in next(
                                heads[position]
                                for heads, arity in zip(
                                    self.argument_heads,
                                    self.argument_arities,
                                    strict=True,
                                )
                                if position < arity
                            ).labels
                        )
                        for position in range(max(self.argument_arities))
                    ]
                    or (
                        self.schema == TYPED_SEMANTIC_TRANSDUCER_SCHEMA
                        and receipt.get("argument_arities") != list(self.argument_arities)
                    )
                    or any(
                        head.to_dict() != self.operation_heads[0].to_dict()
                        for head in self.operation_heads[1:]
                    )
                    or any(
                        head.to_dict()
                        != next(
                            candidate_heads[position]
                            for candidate_heads, candidate_arity in zip(
                                self.argument_heads,
                                self.argument_arities,
                                strict=True,
                            )
                            if position < candidate_arity
                        ).to_dict()
                        for heads, arity in zip(
                            self.argument_heads,
                            self.argument_arities,
                            strict=True,
                        )
                        for position, head in enumerate(heads[:arity])
                    )
                )
            )
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
            raise ValueError("semantic transducer receipt does not match its model")
        object.__setattr__(self, "pointer_heads", pointer_heads)
        object.__setattr__(self, "training_receipt", receipt)

    def _coefficient_body(
        self,
        *,
        pointer_heads: dict[str, LinearPointerHead] | None = None,
    ) -> dict[str, Any]:
        pointers = self.pointer_heads if pointer_heads is None else pointer_heads
        return {
            "pointer_heads": {role: pointers[role].to_dict() for role in sorted(pointers)},
            "operation_heads": [head.to_dict() for head in self.operation_heads],
            "argument_heads": [[head.to_dict() for head in heads] for heads in self.argument_heads],
        }

    @property
    def receipt_sha256(self) -> str:
        value = self.training_receipt.get("receipt_sha256")
        if not _is_sha256(value):
            raise ValueError("semantic transducer receipt identity is invalid")
        return str(value)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "hidden_size": self.hidden_size,
            "model_basis_sha256": self.model_basis_sha256,
            **self._coefficient_body(),
            "training_receipt": self.training_receipt,
        }
        if self.schema in {
            SEMANTIC_TRANSDUCER_SCHEMA,
            TYPED_SEMANTIC_TRANSDUCER_SCHEMA,
        }:
            payload["input_count"] = self.input_count
            payload["step_count"] = self.step_count
        if self.schema == TYPED_SEMANTIC_TRANSDUCER_SCHEMA:
            payload["argument_arities"] = list(self.argument_arities)
        return payload

    def decode(
        self,
        *,
        source_token_ids: Sequence[int],
        hidden_states: Any,
        source_text_sha256: str,
        model_basis_sha256: str,
    ) -> SemanticTransductionOutcome:
        if model_basis_sha256 != self.model_basis_sha256:
            return SemanticTransductionOutcome(None, "model_basis_mismatch", {}, {})
        if not _is_sha256(source_text_sha256):
            return SemanticTransductionOutcome(None, "source_identity_invalid", {}, {})
        try:
            hidden = _hidden_array(hidden_states, expected_width=self.hidden_size)
        except ValueError as exc:
            return SemanticTransductionOutcome(None, str(exc), {}, {})
        tokens = tuple(source_token_ids)
        if hidden.shape[0] != len(tokens):
            return SemanticTransductionOutcome(
                None,
                "token_hidden_length_mismatch",
                {},
                {},
            )

        spans: dict[str, TokenSpan] = {}
        pointer_scores: dict[str, float] = {}
        pointer_candidates: dict[str, tuple[tuple[TokenSpan, float], ...]] = {}
        roles = _pointer_roles(self.input_count, self.argument_arities)
        for role in roles:
            if self.schema == LEGACY_SEMANTIC_TRANSDUCER_SCHEMA:
                candidate_limit = 1
            elif role.startswith("input:"):
                candidate_limit = _STRUCTURED_INPUT_CANDIDATES
            elif role.startswith("argument:"):
                candidate_limit = _STRUCTURED_ARGUMENT_CANDIDATES
            else:
                candidate_limit = 1
            candidates = self.pointer_heads[role].decode_candidates(
                hidden,
                limit=candidate_limit,
            )
            pointer_candidates[role] = candidates
            span, score = candidates[0]
            spans[role] = span
            pointer_scores[role] = score

        if self.schema in {
            SEMANTIC_TRANSDUCER_SCHEMA,
            TYPED_SEMANTIC_TRANSDUCER_SCHEMA,
        }:
            input_roles = tuple(f"input:{index}" for index in range(self.input_count))
            input_assignment = _joint_pointer_assignment(
                tuple(pointer_candidates[role] for role in input_roles),
                ordered=False,
            )
            if input_assignment is not None:
                selected_spans, selected_scores = input_assignment
                for role, span, score in zip(
                    input_roles,
                    selected_spans,
                    selected_scores,
                    strict=True,
                ):
                    spans[role] = span
                    pointer_scores[role] = score

        confidences: dict[str, float] = {}
        operations: list[str] = []
        operation_spans = tuple(spans[f"operation:{step}"] for step in range(self.step_count))
        for step in range(self.step_count):
            op_feature = _pool(hidden, spans[f"operation:{step}"])
            operation, confidence = self.operation_heads[step].predict(op_feature)
            operations.append(operation)
            confidences[f"operation:{step}"] = confidence

        structured_arguments = (
            _structured_argument_assignment(
                pointer_candidates=pointer_candidates,
                argument_heads=self.argument_heads,
                pointer_heads=self.pointer_heads,
                hidden=hidden,
                tokens=tokens,
                input_spans=tuple(spans[f"input:{index}"] for index in range(self.input_count)),
                operation_spans=operation_spans,
                input_count=self.input_count,
                argument_arities=self.argument_arities,
            )
            if self.schema in {SEMANTIC_TRANSDUCER_SCHEMA, TYPED_SEMANTIC_TRANSDUCER_SCHEMA}
            else None
        )
        arguments: list[tuple[int, ...]] = []
        if structured_arguments is not None:
            assigned_arguments, assigned_spans, assigned_scores, assigned_confidences = (
                structured_arguments
            )
            arguments.extend(assigned_arguments)
            for step in range(self.step_count):
                for position in range(self.argument_arities[step]):
                    role = f"argument:{step}:{position}"
                    spans[role] = assigned_spans[step][position]
                    pointer_scores[role] = assigned_scores[step][position]
                    confidences[role] = assigned_confidences[step][position]
        else:
            for step in range(self.step_count):
                step_args: list[int] = []
                for position in range(self.argument_arities[step]):
                    role = f"argument:{step}:{position}"
                    span = spans[role]
                    input_matches = [
                        index
                        for index in range(self.input_count)
                        if span == spans[f"input:{index}"]
                    ]
                    if len(input_matches) == 1:
                        argument = input_matches[0]
                        confidence = min(
                            pointer_scores[role],
                            pointer_scores[f"input:{argument}"],
                        )
                    else:
                        prior_result = _resolve_prior_result_register(
                            token_ids=tokens,
                            reference_span=span,
                            operation_spans=operation_spans,
                            current_step=step,
                            input_count=self.input_count,
                        )
                        if prior_result is not None:
                            argument = prior_result
                            confidence = min(
                                pointer_scores[role],
                                pointer_scores[f"operation:{prior_result - self.input_count}"],
                            )
                        else:
                            feature = _pool(hidden, span)
                            label, confidence = self.argument_heads[step][position].predict(feature)
                            argument = int(label)
                    step_args.append(argument)
                    confidences[role] = confidence
                arguments.append(tuple(step_args))

        instructions = tuple(
            SemanticIRInstruction(
                op=operations[step],
                args=arguments[step],
                operation_span=spans[f"operation:{step}"],
                argument_spans=tuple(
                    spans[f"argument:{step}:{position}"]
                    for position in range(self.argument_arities[step])
                ),
                depends_on=tuple(
                    sorted(
                        argument - self.input_count
                        for argument in set(arguments[step])
                        if argument >= self.input_count
                    )
                ),
            )
            for step in range(self.step_count)
        )
        try:
            ir = SemanticProgramIR(
                source_token_ids=tokens,
                source_text_sha256=source_text_sha256,
                input_spans=tuple(spans[f"input:{index}"] for index in range(self.input_count)),
                instructions=instructions,
                report_value=self.input_count + self.step_count - 1,
                model_basis_receipt_sha256=model_basis_sha256,
                transducer_receipt_sha256=self.receipt_sha256,
            )
        except ValueError as exc:
            return SemanticTransductionOutcome(
                None,
                f"ir_rejected:{exc}",
                pointer_scores,
                confidences,
            )
        return SemanticTransductionOutcome(ir, "", pointer_scores, confidences)


def _pool(hidden: np.ndarray, span: TokenSpan) -> np.ndarray:
    if span.end > hidden.shape[0]:
        raise ValueError("semantic transducer span exceeds hidden sequence")
    pooled = np.mean(hidden[span.start : span.end], axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(pooled))
    return pooled / norm if norm > 1e-8 else pooled


def _gold_spans(ir: SemanticProgramIR) -> dict[str, TokenSpan]:
    spans = {f"input:{index}": span for index, span in enumerate(ir.input_spans)}
    for step, instruction in enumerate(ir.instructions):
        spans[f"operation:{step}"] = instruction.operation_span
        for position, span in enumerate(instruction.argument_spans):
            spans[f"argument:{step}:{position}"] = span
    return spans


def _fit_binary_head(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("semantic pointer supervision lacks a positive or negative class")
    classifier = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    classifier.fit(features, labels)
    return (
        np.asarray(classifier.coef_[0], dtype=np.float32),
        float(classifier.intercept_[0]),
    )


def _fit_classifier(
    features: np.ndarray,
    labels: Sequence[str],
) -> LinearClassifierHead:
    from sklearn.linear_model import LogisticRegression

    unique = tuple(sorted(set(labels)))
    if len(unique) < 2:
        raise ValueError("semantic classifier supervision has fewer than two classes")
    classifier = LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=1500,
        random_state=0,
        solver="lbfgs",
    )
    classifier.fit(features, list(labels))
    classes = tuple(str(value) for value in classifier.classes_.tolist())
    weight = np.asarray(classifier.coef_, dtype=np.float32)
    bias = np.asarray(classifier.intercept_, dtype=np.float32)
    if len(classes) == 2 and weight.shape[0] == 1:
        weight = np.concatenate((np.zeros_like(weight), weight), axis=0)
        bias = np.concatenate((np.zeros_like(bias), bias), axis=0)
    return LinearClassifierHead(classes, weight, bias)


def fit_semantic_program_transducer(
    examples: Sequence[SemanticTransducerTrainingExample],
) -> SemanticProgramTransducer:
    """Fit all heads from training-split hidden states and gold token spans."""

    training = tuple(item for item in examples if item.split == "train")
    if not training:
        raise ValueError("semantic transducer received no training examples")
    hidden_size = int(training[0].hidden_states.shape[1])
    model_bases = {item.ir.model_basis_receipt_sha256 for item in training}
    if len(model_bases) != 1:
        raise ValueError("semantic transducer training spans multiple model bases")
    geometries = {
        (
            item.ir.n_inputs,
            tuple(len(instruction.args) for instruction in item.ir.instructions),
        )
        for item in training
    }
    if len(geometries) != 1:
        raise ValueError("semantic transducer training geometries differ")
    if any(item.hidden_states.shape[1] != hidden_size for item in training):
        raise ValueError("semantic transducer training hidden widths differ")
    input_count, argument_arities = next(iter(geometries))
    step_count = len(argument_arities)
    roles = _pointer_roles(input_count, argument_arities)
    legacy = (input_count, step_count, argument_arities) == (
        SEMANTIC_TRANSDUCER_INPUTS,
        SEMANTIC_TRANSDUCER_STEPS,
        (2,) * SEMANTIC_TRANSDUCER_STEPS,
    )

    pointer_heads: dict[str, LinearPointerHead] = {}
    token_features = np.concatenate([item.hidden_states for item in training], axis=0)
    for role in roles:
        start_labels: list[int] = []
        end_labels: list[int] = []
        for item in training:
            span = _gold_spans(item.ir)[role]
            start_labels.extend(
                int(index == span.start) for index in range(item.hidden_states.shape[0])
            )
            end_labels.extend(
                int(index == span.end - 1) for index in range(item.hidden_states.shape[0])
            )
        start_weight, start_bias = _fit_binary_head(
            token_features,
            np.asarray(start_labels, dtype=np.int8),
        )
        end_weight, end_bias = _fit_binary_head(
            token_features,
            np.asarray(end_labels, dtype=np.int8),
        )
        pointer_heads[role] = LinearPointerHead(
            start_weight,
            start_bias,
            end_weight,
            end_bias,
        )

    if legacy:
        operation_heads: list[LinearClassifierHead] = []
        argument_heads: list[tuple[LinearClassifierHead, LinearClassifierHead]] = []
        for step in range(step_count):
            operation_heads.append(
                _fit_classifier(
                    np.stack(
                        [
                            _pool(
                                item.hidden_states,
                                item.ir.instructions[step].operation_span,
                            )
                            for item in training
                        ]
                    ),
                    [item.ir.instructions[step].op for item in training],
                )
            )
            step_argument_heads: list[LinearClassifierHead] = []
            for position in range(2):
                step_argument_heads.append(
                    _fit_classifier(
                        np.stack(
                            [
                                _pool(
                                    item.hidden_states,
                                    item.ir.instructions[step].argument_spans[position],
                                )
                                for item in training
                            ]
                        ),
                        [str(item.ir.instructions[step].args[position]) for item in training],
                    )
                )
            argument_heads.append((step_argument_heads[0], step_argument_heads[1]))
    else:
        shared_operation = _fit_classifier(
            np.stack(
                [
                    _pool(item.hidden_states, instruction.operation_span)
                    for item in training
                    for instruction in item.ir.instructions
                ]
            ),
            [instruction.op for item in training for instruction in item.ir.instructions],
        )
        shared_arguments = tuple(
            _fit_classifier(
                np.stack(
                    [
                        _pool(
                            item.hidden_states,
                            instruction.argument_spans[position],
                        )
                        for item in training
                        for instruction in item.ir.instructions
                        if len(instruction.args) > position
                    ]
                ),
                [
                    str(instruction.args[position])
                    for item in training
                    for instruction in item.ir.instructions
                    if len(instruction.args) > position
                ],
            )
            for position in range(max(argument_arities))
        )
        operation_heads = [shared_operation for _ in range(step_count)]
        argument_heads = [shared_arguments[:arity] for arity in argument_arities]

    coefficient_body = {
        "pointer_heads": {role: pointer_heads[role].to_dict() for role in sorted(pointer_heads)},
        "operation_heads": [head.to_dict() for head in operation_heads],
        "argument_heads": [[head.to_dict() for head in heads] for heads in argument_heads],
    }
    receipt_body = {
        "schema": LEGACY_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA,
        "model_basis_sha256": next(iter(model_bases)),
        "hidden_size": hidden_size,
        "training_example_count": len(training),
        "training_constructions": sorted({item.construction_id for item in training}),
        "training_topologies": sorted({item.topology_id for item in training}),
        "primitive_support": sorted(
            {instruction.op for item in training for instruction in item.ir.instructions}
        ),
        "register_support": [
            sorted({item.ir.instructions[step].args[position] for item in training})
            for step in range(step_count)
            for position in range(argument_arities[step])
        ],
        "coefficient_sha256": _sha(coefficient_body),
        "expected_answers_available": False,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "correctness_authority": False,
    }
    typed = any(arity != 2 for arity in argument_arities)
    if not legacy:
        receipt_body["schema"] = (
            TYPED_SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
            if typed
            else SEMANTIC_TRANSDUCER_RECEIPT_SCHEMA
        )
        receipt_body["input_count"] = input_count
        receipt_body["step_count"] = step_count
        receipt_body["classifier_sharing"] = "across_step_slots"
        receipt_body["shared_argument_support"] = [
            sorted(
                {
                    instruction.args[position]
                    for item in training
                    for instruction in item.ir.instructions
                    if len(instruction.args) > position
                }
            )
            for position in range(max(argument_arities))
        ]
        if typed:
            receipt_body["argument_arities"] = list(argument_arities)
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    return SemanticProgramTransducer(
        schema=(
            LEGACY_SEMANTIC_TRANSDUCER_SCHEMA
            if legacy
            else TYPED_SEMANTIC_TRANSDUCER_SCHEMA
            if typed
            else SEMANTIC_TRANSDUCER_SCHEMA
        ),
        hidden_size=hidden_size,
        model_basis_sha256=next(iter(model_bases)),
        input_count=input_count,
        step_count=step_count,
        argument_arities=argument_arities,
        pointer_heads=pointer_heads,
        operation_heads=tuple(operation_heads),
        argument_heads=tuple(argument_heads),
        training_receipt=receipt,
    )


def _pointer_head_from_dict(value: Any) -> LinearPointerHead:
    if not isinstance(value, dict) or set(value) != {
        "start_weight",
        "start_bias",
        "end_weight",
        "end_bias",
    }:
        raise ValueError("serialized semantic pointer head is invalid")
    return LinearPointerHead(
        start_weight=value["start_weight"],
        start_bias=value["start_bias"],
        end_weight=value["end_weight"],
        end_bias=value["end_bias"],
    )


def _classifier_head_from_dict(value: Any) -> LinearClassifierHead:
    if not isinstance(value, dict) or set(value) != {"labels", "weight", "bias"}:
        raise ValueError("serialized semantic classifier head is invalid")
    if not isinstance(value["labels"], list):
        raise ValueError("serialized semantic classifier labels are invalid")
    return LinearClassifierHead(
        labels=tuple(value["labels"]),
        weight=value["weight"],
        bias=value["bias"],
    )


def semantic_program_transducer_from_dict(payload: Any) -> SemanticProgramTransducer:
    """Load a transducer only when coefficients and receipt still agree."""

    if not isinstance(payload, dict):
        raise ValueError("serialized semantic transducer fields are invalid")
    schema = payload.get("schema")
    common_fields = {
        "schema",
        "hidden_size",
        "model_basis_sha256",
        "pointer_heads",
        "operation_heads",
        "argument_heads",
        "training_receipt",
    }
    if schema == LEGACY_SEMANTIC_TRANSDUCER_SCHEMA:
        input_count = SEMANTIC_TRANSDUCER_INPUTS
        step_count = SEMANTIC_TRANSDUCER_STEPS
        argument_arities = (2,) * step_count
        expected_fields = common_fields
    elif schema == SEMANTIC_TRANSDUCER_SCHEMA:
        input_count = payload.get("input_count")
        step_count = payload.get("step_count")
        argument_arities = (2,) * step_count if type(step_count) is int else ()
        expected_fields = common_fields | {"input_count", "step_count"}
    elif schema == TYPED_SEMANTIC_TRANSDUCER_SCHEMA:
        input_count = payload.get("input_count")
        step_count = payload.get("step_count")
        raw_arities = payload.get("argument_arities")
        argument_arities = tuple(raw_arities) if isinstance(raw_arities, list) else ()
        expected_fields = common_fields | {
            "input_count",
            "step_count",
            "argument_arities",
        }
    else:
        raise ValueError("serialized semantic transducer schema is invalid")
    if set(payload) != expected_fields:
        raise ValueError("serialized semantic transducer fields are invalid")
    if (
        type(input_count) is not int
        or type(step_count) is not int
        or len(argument_arities) != step_count
        or any(type(arity) is not int or arity not in {1, 2} for arity in argument_arities)
    ):
        raise ValueError("serialized semantic transducer geometry is invalid")
    roles = _pointer_roles(input_count, argument_arities)
    raw_pointers = payload["pointer_heads"]
    raw_operations = payload["operation_heads"]
    raw_arguments = payload["argument_heads"]
    if (
        not isinstance(raw_pointers, dict)
        or set(raw_pointers) != set(roles)
        or not isinstance(raw_operations, list)
        or len(raw_operations) != step_count
        or not isinstance(raw_arguments, list)
        or len(raw_arguments) != step_count
        or any(
            not isinstance(heads, list) or len(heads) != argument_arities[index]
            for index, heads in enumerate(raw_arguments)
        )
    ):
        raise ValueError("serialized semantic transducer topology is invalid")
    return SemanticProgramTransducer(
        schema=payload["schema"],
        hidden_size=payload["hidden_size"],
        model_basis_sha256=payload["model_basis_sha256"],
        input_count=input_count,
        step_count=step_count,
        argument_arities=argument_arities,
        pointer_heads={role: _pointer_head_from_dict(raw_pointers[role]) for role in roles},
        operation_heads=tuple(_classifier_head_from_dict(value) for value in raw_operations),
        argument_heads=tuple(
            tuple(_classifier_head_from_dict(value) for value in heads) for heads in raw_arguments
        ),
        training_receipt=payload["training_receipt"],
    )


__all__ = [
    "LinearClassifierHead",
    "LinearPointerHead",
    "SEMANTIC_TRANSDUCER_INPUTS",
    "SEMANTIC_TRANSDUCER_SCHEMA",
    "SEMANTIC_TRANSDUCER_STEPS",
    "SemanticProgramTransducer",
    "SemanticTransducerTrainingExample",
    "SemanticTransductionOutcome",
    "TYPED_SEMANTIC_TRANSDUCER_SCHEMA",
    "fit_semantic_program_transducer",
    "semantic_program_transducer_from_dict",
]
