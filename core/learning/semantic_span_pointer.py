"""Shared learned semantic span pointers and validated hidden features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_ir import TokenSpan

SEMANTIC_TRANSDUCER_MAX_SPAN_TOKENS: Final = 24


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


@dataclass(frozen=True, slots=True)
class LinearPointerSequenceScores:
    """Validated pointer logits for one hidden sequence."""

    start: np.ndarray
    end: np.ndarray
    hidden: np.ndarray | None = None
    pair_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.float32).reshape(-1)
        end = np.asarray(self.end, dtype=np.float32).reshape(-1)
        if (
            start.shape != end.shape
            or start.size < 1
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
        ):
            raise ValueError("semantic pointer sequence scores are invalid")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if (self.hidden is None) != (self.pair_weight is None):
            raise ValueError("semantic pointer paired scores are incomplete")
        if self.hidden is not None:
            hidden = _hidden_array(self.hidden)
            pair_weight = np.asarray(self.pair_weight, dtype=np.float32).reshape(-1)
            if hidden.shape != (start.size, pair_weight.size) or not np.all(np.isfinite(pair_weight)):
                raise ValueError("semantic pointer paired score geometry differs")
            object.__setattr__(self, "hidden", hidden)
            object.__setattr__(self, "pair_weight", pair_weight)

    def score_span(self, span: TokenSpan) -> float:
        span.validate_bound(self.start.size)
        score = float(self.start[span.start] + self.end[span.end - 1])
        if self.hidden is not None:
            score += float(
                (self.hidden[span.start] * self.hidden[span.end - 1]) @ self.pair_weight
                * math.sqrt(self.hidden.shape[1])
            )
        return score

    def decode_candidates(
        self,
        *,
        limit: int,
        max_span_tokens: int = SEMANTIC_TRANSDUCER_MAX_SPAN_TOKENS,
    ) -> tuple[tuple[TokenSpan, float], ...]:
        if type(limit) is not int or limit < 1:
            raise ValueError("semantic pointer candidate limit is invalid")
        if type(max_span_tokens) is not int or max_span_tokens < 1:
            raise ValueError("semantic pointer span limit is invalid")
        candidates: list[tuple[TokenSpan, float]] = []
        for start in range(self.start.size):
            stop = min(self.start.size, start + max_span_tokens)
            pair_scores = None
            if self.hidden is not None:
                pair_scores = (
                    self.hidden[start:stop] @ (self.hidden[start] * self.pair_weight)
                    * math.sqrt(self.hidden.shape[1])
                )
            for end in range(start, stop):
                span = TokenSpan(start, end + 1)
                score = float(self.start[start] + self.end[end])
                if pair_scores is not None:
                    score += float(pair_scores[end - start])
                candidates.append((span, score))
        candidates.sort(key=lambda item: (-item[1], item[0].start, item[0].end))
        return tuple(candidates[:limit])


@dataclass(frozen=True, slots=True)
class LinearPointerHead:
    """Learned boundary scores with optional start/end compatibility."""

    start_weight: np.ndarray
    start_bias: float
    end_weight: np.ndarray
    end_bias: float
    pair_weight: np.ndarray | None = None

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
        if self.pair_weight is not None:
            pair = np.asarray(self.pair_weight, dtype=np.float32).reshape(-1)
            if pair.shape != start.shape or not np.all(np.isfinite(pair)):
                raise ValueError("semantic pointer boundary pair geometry differs")
            object.__setattr__(self, "pair_weight", pair)

    @property
    def width(self) -> int:
        return int(self.start_weight.size)

    def decode_candidates(
        self,
        hidden: np.ndarray,
        *,
        limit: int,
        max_span_tokens: int = SEMANTIC_TRANSDUCER_MAX_SPAN_TOKENS,
    ) -> tuple[tuple[TokenSpan, float], ...]:
        """Return the strongest distinct source spans in stable score order."""

        return self.score_sequence(hidden).decode_candidates(
            limit=limit,
            max_span_tokens=max_span_tokens,
        )

    def score_sequence(self, hidden: np.ndarray) -> LinearPointerSequenceScores:
        """Validate once and retain every endpoint score for repeated span queries."""

        matrix = _hidden_array(hidden, expected_width=self.width)
        return LinearPointerSequenceScores(
            matrix @ self.start_weight + self.start_bias,
            matrix @ self.end_weight + self.end_bias,
            matrix if self.pair_weight is not None else None,
            self.pair_weight,
        )

    def score_span(self, hidden: np.ndarray, span: TokenSpan) -> float:
        return self.score_sequence(hidden).score_span(span)

    def decode(self, hidden: np.ndarray) -> tuple[TokenSpan, float]:
        return self.decode_candidates(hidden, limit=1)[0]

    def to_dict(self) -> dict[str, Any]:
        body = {
            "start_weight": self.start_weight.tolist(),
            "start_bias": float(self.start_bias),
            "end_weight": self.end_weight.tolist(),
            "end_bias": float(self.end_bias),
        }
        if self.pair_weight is not None:
            body["pair_weight"] = self.pair_weight.tolist()
        return body
