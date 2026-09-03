"""Recover the exact public value algebra from tokenizer-aligned source text.

The semantic transducer learns what operations and references mean. Integer
and integer-sequence literals are already a closed, exact language at the
execution boundary, so recovering their values is parsing rather than model
inference. This module is the inverse of ``SemanticInputGroundingContract``:
it binds each public value to measured source tokens without a family label,
answer key, verifier trace, or generated text.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.semantic_program_ir import (
    MAX_SEMANTIC_SEQUENCE_ITEMS,
    SemanticValue,
    TokenSpan,
    normalize_semantic_value,
    semantic_value_to_json,
)

SEMANTIC_PUBLIC_INPUTS_SCHEMA: Final = "aura.semantic_public_inputs.v1"
_INTEGER = r"-?(?:0|[1-9][0-9]*)"
_SEQUENCE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:"
    rf"\[\s*(?:{_INTEGER}(?:\s*,\s*{_INTEGER})*)?\s*\]"
    rf"|\(\s*(?:{_INTEGER}(?:\s*,\s*{_INTEGER})*\s*,?)?\s*\)"
    rf")(?![A-Za-z0-9_])"
)
_SCALAR = re.compile(rf"(?<![A-Za-z0-9_]){_INTEGER}(?![A-Za-z0-9_])")


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


@dataclass(frozen=True, slots=True)
class SemanticPublicLiteral:
    """One exact value and its source/token identity."""

    value: SemanticValue
    character_start: int
    character_end: int
    token_span: TokenSpan | None = None

    def __post_init__(self) -> None:
        if (
            normalize_semantic_value(self.value) != self.value
            or type(self.character_start) is not int
            or type(self.character_end) is not int
            or self.character_start < 0
            or self.character_end <= self.character_start
            or (self.token_span is not None and not isinstance(self.token_span, TokenSpan))
        ):
            raise ValueError("semantic public literal is invalid")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "value": semantic_value_to_json(self.value),
            "character_span": [self.character_start, self.character_end],
        }
        if self.token_span is not None:
            value["token_span"] = self.token_span.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class SemanticPublicInputs:
    """A complete source-order projection of top-level exact literals."""

    source_text_sha256: str
    literals: tuple[SemanticPublicLiteral, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_text_sha256, str)
            or len(self.source_text_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_text_sha256)
            or not isinstance(self.literals, tuple)
            or any(
                left.character_start < right.character_end
                and right.character_start < left.character_end
                for index, left in enumerate(self.literals)
                for right in self.literals[index + 1 :]
            )
            or tuple(
                sorted(
                    self.literals,
                    key=lambda item: (item.character_start, item.character_end),
                )
            )
            != self.literals
        ):
            raise ValueError("semantic public inputs are invalid")

    @property
    def values(self) -> tuple[SemanticValue, ...]:
        return tuple(item.value for item in self.literals)

    @property
    def token_spans(self) -> tuple[TokenSpan, ...]:
        spans = tuple(item.token_span for item in self.literals)
        if any(span is None for span in spans):
            raise ValueError("semantic public inputs are not token aligned")
        return tuple(span for span in spans if span is not None)

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": SEMANTIC_PUBLIC_INPUTS_SCHEMA,
            "source_text_sha256": self.source_text_sha256,
            "literals": [item.to_dict() for item in self.literals],
            "input_order": "source_character_order",
            "value_algebra": "exact_integer_or_integer_sequence",
            "family_router_present": False,
            "expected_answer_available": False,
            "verifier_trace_available": False,
            "generated_text_available": False,
            "correctness_authority": False,
        }
        return {**body, "receipt_sha256": _sha(body)}


def _sequence_value(text: str) -> SemanticValue | None:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        normalized = normalize_semantic_value(value)
    except ValueError:
        return None
    if not isinstance(normalized, tuple) or len(normalized) > MAX_SEMANTIC_SEQUENCE_ITEMS:
        return None
    return normalized


def _top_level_character_literals(text: str) -> tuple[SemanticPublicLiteral, ...]:
    if type(text) is not str or not text or "\x00" in text:
        raise ValueError("semantic public source text is invalid")
    sequences: list[SemanticPublicLiteral] = []
    occupied: list[tuple[int, int]] = []
    for match in _SEQUENCE.finditer(text):
        value = _sequence_value(match.group(0))
        if value is None:
            continue
        start, end = match.span()
        sequences.append(SemanticPublicLiteral(value, start, end))
        occupied.append((start, end))

    scalars: list[SemanticPublicLiteral] = []
    for match in _SCALAR.finditer(text):
        start, end = match.span()
        if any(left <= start and end <= right for left, right in occupied):
            continue
        # Decimal fragments are not integers in the closed public algebra.
        if (
            start >= 2
            and text[start - 1] == "."
            and text[start - 2].isdigit()
        ) or (
            end + 1 < len(text)
            and text[end] == "."
            and text[end + 1].isdigit()
        ):
            continue
        scalars.append(SemanticPublicLiteral(int(match.group(0)), start, end))
    return tuple(
        sorted((*sequences, *scalars), key=lambda item: (item.character_start, item.character_end))
    )


def semantic_public_character_inputs(text: str) -> SemanticPublicInputs:
    """Extract top-level exact literals without requiring a tokenizer."""

    return SemanticPublicInputs(
        source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        literals=_top_level_character_literals(text),
    )


def _token_span(
    literal: SemanticPublicLiteral,
    offsets: Sequence[tuple[int, int]],
) -> TokenSpan:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start
        and start < literal.character_end
        and end > literal.character_start
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("tokenizer offsets do not preserve a public literal")
    covered_start = min(offsets[index][0] for index in indices)
    covered_end = max(offsets[index][1] for index in indices)
    if covered_start > literal.character_start or covered_end < literal.character_end:
        raise ValueError("tokenizer offsets do not cover a public literal")
    return TokenSpan(indices[0], indices[-1] + 1)


def semantic_public_token_inputs(
    text: str,
    offset_mapping: Sequence[Sequence[int]],
) -> SemanticPublicInputs:
    """Bind every top-level exact literal to the measured token sequence."""

    offsets: list[tuple[int, int]] = []
    for value in offset_mapping:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError("semantic public tokenizer offset is malformed")
        start, end = value
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > len(text)
        ):
            raise ValueError("semantic public tokenizer offset is out of bounds")
        offsets.append((start, end))
    if not offsets:
        raise ValueError("semantic public tokenizer offsets are empty")
    character_inputs = semantic_public_character_inputs(text)
    literals = tuple(
        SemanticPublicLiteral(
            item.value,
            item.character_start,
            item.character_end,
            _token_span(item, offsets),
        )
        for item in character_inputs.literals
    )
    spans = tuple(item.token_span for item in literals)
    if len(set(spans)) != len(spans):
        raise ValueError("semantic public literals collapse onto one token span")
    return SemanticPublicInputs(character_inputs.source_text_sha256, literals)


__all__ = [
    "SEMANTIC_PUBLIC_INPUTS_SCHEMA",
    "SemanticPublicInputs",
    "SemanticPublicLiteral",
    "semantic_public_character_inputs",
    "semantic_public_token_inputs",
]
