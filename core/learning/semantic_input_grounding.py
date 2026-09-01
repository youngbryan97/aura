"""Tokenizer-bound exact grounding for the semantic value algebra."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrent_literal_grounding import tokenizer_digit_token_ids
from core.learning.semantic_program_ir import SemanticValue, TokenSpan, normalize_semantic_value

SEMANTIC_INPUT_GROUNDING_SCHEMA: Final = "aura.semantic_input_grounding.v1"
_BOUNDARY_SUFFIXES: Final = ("", ",", ".", ":", ";", "?", "!")


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


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if not isinstance(encoded, (list, tuple)) or any(
        type(token_id) is not int or token_id < 0 for token_id in encoded
    ):
        raise ValueError("semantic input tokenizer returned invalid ids")
    return tuple(encoded)


@dataclass(frozen=True, slots=True)
class SequenceTokenFormat:
    positive_prefix: tuple[int, ...]
    negative_prefix: tuple[int, ...]
    positive_separator: tuple[int, ...]
    negative_separator: tuple[int, ...]
    suffixes: tuple[tuple[int, ...], ...]
    singleton_suffixes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.positive_prefix
            or not self.negative_prefix
            or not self.positive_separator
            or not self.negative_separator
            or not self.suffixes
            or not self.singleton_suffixes
            or any(
                type(token_id) is not int or token_id < 0
                for group in (
                    self.positive_prefix,
                    self.negative_prefix,
                    self.positive_separator,
                    self.negative_separator,
                    *self.suffixes,
                    *self.singleton_suffixes,
                )
                for token_id in group
            )
        ):
            raise ValueError("semantic sequence token format is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_prefix": list(self.positive_prefix),
            "negative_prefix": list(self.negative_prefix),
            "positive_separator": list(self.positive_separator),
            "negative_separator": list(self.negative_separator),
            "suffixes": [list(value) for value in self.suffixes],
            "singleton_suffixes": [list(value) for value in self.singleton_suffixes],
        }


@dataclass(frozen=True, slots=True)
class SemanticInputGroundingContract:
    """Reconstruct exact source-token forms from typed public inputs."""

    tokenizer_identity_sha256: str
    digit_token_ids: tuple[int, ...]
    positive_integer_prefixes: tuple[tuple[int, ...], ...]
    negative_integer_prefixes: tuple[tuple[int, ...], ...]
    integer_suffixes: tuple[tuple[int, ...], ...]
    empty_sequence_variants: tuple[tuple[int, ...], ...]
    sequence_formats: tuple[SequenceTokenFormat, ...]
    schema: str = SEMANTIC_INPUT_GROUNDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SEMANTIC_INPUT_GROUNDING_SCHEMA
            or not _is_sha256(self.tokenizer_identity_sha256)
            or len(self.digit_token_ids) != 10
            or len(set(self.digit_token_ids)) != 10
            or any(type(value) is not int or value < 0 for value in self.digit_token_ids)
            or not self.positive_integer_prefixes
            or not self.negative_integer_prefixes
            or not self.integer_suffixes
            or not self.empty_sequence_variants
            or not self.sequence_formats
            or any(
                type(token_id) is not int or token_id < 0
                for group in (
                    *self.positive_integer_prefixes,
                    *self.negative_integer_prefixes,
                    *self.integer_suffixes,
                    *self.empty_sequence_variants,
                )
                for token_id in group
            )
        ):
            raise ValueError("semantic input grounding contract is invalid")

    @property
    def contract_sha256(self) -> str:
        return _sha(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "digit_token_ids": list(self.digit_token_ids),
            "positive_integer_prefixes": [
                list(value) for value in self.positive_integer_prefixes
            ],
            "negative_integer_prefixes": [
                list(value) for value in self.negative_integer_prefixes
            ],
            "integer_suffixes": [list(value) for value in self.integer_suffixes],
            "empty_sequence_variants": [
                list(value) for value in self.empty_sequence_variants
            ],
            "sequence_formats": [value.to_dict() for value in self.sequence_formats],
        }

    def _integer_digits(self, value: int) -> tuple[int, ...]:
        return tuple(self.digit_token_ids[int(character)] for character in str(abs(value)))

    def token_variants(self, value: SemanticValue) -> tuple[tuple[int, ...], ...]:
        normalized = normalize_semantic_value(value)
        if isinstance(normalized, int):
            body = self._integer_digits(normalized)
            prefixes = (
                self.negative_integer_prefixes
                if normalized < 0
                else self.positive_integer_prefixes
            )
            return tuple(
                dict.fromkeys(
                    (*prefix, *body, *suffix)
                    for prefix in prefixes
                    for suffix in self.integer_suffixes
                )
            )
        if not normalized:
            return self.empty_sequence_variants
        encoded_items = tuple(self._integer_digits(item) for item in normalized)
        variants: list[tuple[int, ...]] = []
        for token_format in self.sequence_formats:
            suffixes = (
                token_format.singleton_suffixes
                if len(encoded_items) == 1
                else token_format.suffixes
            )
            body: tuple[int, ...] = (
                token_format.negative_prefix
                if normalized[0] < 0
                else token_format.positive_prefix
            )
            for index, item in enumerate(encoded_items):
                if index:
                    separator = (
                        token_format.negative_separator
                        if normalized[index] < 0
                        else token_format.positive_separator
                    )
                    body = (*body, *separator)
                body = (*body, *item)
            variants.extend(
                (*body, *suffix) for suffix in suffixes
            )
        return tuple(dict.fromkeys(variants))

    def candidate_spans(
        self,
        source_token_ids: tuple[int, ...],
        value: SemanticValue,
    ) -> tuple[TokenSpan, ...]:
        spans: set[TokenSpan] = set()
        for needle in self.token_variants(value):
            if not needle or len(needle) > len(source_token_ids):
                continue
            for start in range(len(source_token_ids) - len(needle) + 1):
                if source_token_ids[start : start + len(needle)] == needle:
                    spans.add(TokenSpan(start, start + len(needle)))
        return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


def semantic_input_grounding_contract_from_tokenizer(
    tokenizer: Any,
    *,
    tokenizer_identity_sha256: str,
) -> SemanticInputGroundingContract:
    """Compile a closed token grammar and prove composition on representative values."""

    digit_ids = tokenizer_digit_token_ids(tokenizer)
    positive_integer_prefixes = tuple(
        dict.fromkeys((_encode(tokenizer, ""), _encode(tokenizer, " ")))
    )
    negative_integer_prefixes = tuple(
        dict.fromkeys((_encode(tokenizer, "-"), _encode(tokenizer, " -")))
    )
    integer_suffixes = tuple(
        dict.fromkeys(_encode(tokenizer, suffix) for suffix in _BOUNDARY_SUFFIXES)
    )
    sequence_formats: list[SequenceTokenFormat] = []
    empty_sequence_variants: list[tuple[int, ...]] = []
    for opener, closer in (("[", "]"), ("(", ")")):
        for leading in ("", " "):
            empty_sequence_variants.extend(
                _encode(tokenizer, leading + opener + closer + suffix)
                for suffix in _BOUNDARY_SUFFIXES
            )
            sequence_formats.append(
                SequenceTokenFormat(
                    positive_prefix=_encode(tokenizer, leading + opener),
                    negative_prefix=_encode(tokenizer, leading + opener + "-"),
                    positive_separator=_encode(tokenizer, ", "),
                    negative_separator=_encode(tokenizer, ", -"),
                    suffixes=tuple(
                        dict.fromkeys(
                            _encode(tokenizer, closer + suffix)
                            for suffix in _BOUNDARY_SUFFIXES
                        )
                    ),
                    singleton_suffixes=tuple(
                        dict.fromkeys(
                            _encode(
                                tokenizer,
                                (("," if opener == "(" else "") + closer + suffix),
                            )
                            for suffix in _BOUNDARY_SUFFIXES
                        )
                    ),
                )
            )
    contract = SemanticInputGroundingContract(
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        digit_token_ids=digit_ids,
        positive_integer_prefixes=positive_integer_prefixes,
        negative_integer_prefixes=negative_integer_prefixes,
        integer_suffixes=integer_suffixes,
        empty_sequence_variants=tuple(dict.fromkeys(empty_sequence_variants)),
        sequence_formats=tuple(sequence_formats),
    )
    checks: tuple[SemanticValue, ...] = (
        0,
        -12,
        57,
        (),
        (3,),
        (-3, 14),
        (3, -14, 5),
    )
    expected = {
        tuple(_encode(tokenizer, prefix + text + suffix))
        for value in checks
        for text in (
            str(value) if isinstance(value, int) else str(list(value)),
            *(() if isinstance(value, int) else (str(value),)),
        )
        for prefix in ("", " ")
        for suffix in _BOUNDARY_SUFFIXES
    }
    observed = {tokens for value in checks for tokens in contract.token_variants(value)}
    if not expected.issubset(observed):
        raise ValueError("semantic input token grammar does not compose exactly")
    return contract


def semantic_input_grounding_contract_from_dict(
    value: Any,
) -> SemanticInputGroundingContract:
    expected = {
        "schema",
        "tokenizer_identity_sha256",
        "digit_token_ids",
        "positive_integer_prefixes",
        "negative_integer_prefixes",
        "integer_suffixes",
        "empty_sequence_variants",
        "sequence_formats",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("semantic input grounding payload is invalid")
    formats = value["sequence_formats"]
    if not isinstance(formats, list):
        raise ValueError("semantic input grounding formats are invalid")
    parsed: list[SequenceTokenFormat] = []
    for item in formats:
        if not isinstance(item, dict) or set(item) != {
            "positive_prefix",
            "negative_prefix",
            "positive_separator",
            "negative_separator",
            "suffixes",
            "singleton_suffixes",
        }:
            raise ValueError("semantic sequence token format payload is invalid")
        parsed.append(
            SequenceTokenFormat(
                positive_prefix=tuple(item["positive_prefix"]),
                negative_prefix=tuple(item["negative_prefix"]),
                positive_separator=tuple(item["positive_separator"]),
                negative_separator=tuple(item["negative_separator"]),
                suffixes=tuple(tuple(part) for part in item["suffixes"]),
                singleton_suffixes=tuple(tuple(part) for part in item["singleton_suffixes"]),
            )
        )
    return SemanticInputGroundingContract(
        schema=value["schema"],
        tokenizer_identity_sha256=value["tokenizer_identity_sha256"],
        digit_token_ids=tuple(value["digit_token_ids"]),
        positive_integer_prefixes=tuple(
            tuple(part) for part in value["positive_integer_prefixes"]
        ),
        negative_integer_prefixes=tuple(
            tuple(part) for part in value["negative_integer_prefixes"]
        ),
        integer_suffixes=tuple(tuple(part) for part in value["integer_suffixes"]),
        empty_sequence_variants=tuple(
            tuple(part) for part in value["empty_sequence_variants"]
        ),
        sequence_formats=tuple(parsed),
    )


__all__ = [
    "SEMANTIC_INPUT_GROUNDING_SCHEMA",
    "SemanticInputGroundingContract",
    "SequenceTokenFormat",
    "semantic_input_grounding_contract_from_dict",
    "semantic_input_grounding_contract_from_tokenizer",
]
