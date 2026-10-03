"""Operations found where the request reads as one, in the resident model's own states.

The compositional transducer proposes operation spans with a pointer trained on
three wordings per operation, then scores whole charts jointly with their
argument bindings. Development validation is written with three other wordings
per operation, none of them in training ("reading one indexed entry", "with a
factor of", "after removing"). On 2 October all 23 incumbent failures were in
the 144 sequence-chain tasks: 19 had the wrong operations and 4 the wrong
binding. With the annotated span supplied, a linear readout of the middle layer
at the span's last token labels 99.3% of validation operations correctly. The
label was in the representation; the span was not found.

This recognizer finds the span by where the request reads as an operation and
names it at its last token:

* a token tagger over the normalised middle and final layers marks how much
  each token reads as part of an operation phrase. A wording it never saw scores
  low in absolute terms, but it still scores above the words around it, so
  operations are the request's local maxima of that score, not tokens over a
  threshold;
* each maximum grows into a span while its neighbours keep at least half of its
  score, and the span is named by a linear readout of the middle layer at its
  last token;
* the first token is read from the input embedding instead. A causal model's
  first position has no context and carries the attention sink, so the
  contextual layers do not describe the word there. A sentence that opens with
  "Add" or "Return" is decided at that position.

Charts are then ordered by this evidence and the first one whose arguments
assign is selected: operations decide and the existing argument machinery
follows. The joint score let argument evidence overrule a better operation
reading, which cost 22 tasks when this recognizer was first tried under it.

Every parameter is fitted on training rows only. Validation and test rows are
refused by the fitter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import (
    OPERATION_BACKGROUND_LABEL,
    LinearClassifierHead,
)


class OperationCandidate(NamedTuple):
    """One reading of a span as an operation; the transducer's operation node takes these fields."""

    span: TokenSpan
    operation: str
    score: float
    pointer_score: float
    confidence: float


PEAK_RECOGNIZER_SCHEMA: Final = "aura.semantic_operation_peaks.v1"
PEAK_TRANSDUCER_SCHEMA: Final = "aura.semantic_peak_recognition_transducer.v1"

#: The two classes of the token tagger.
OPERATION_TAG: Final = "operation"
BACKGROUND_TAG: Final = "background"

#: Channels as the feature bundles name them.
TAGGER_CHANNELS: Final = ("middle_causal_hidden", "final_causal_hidden")
LABELER_CHANNEL: Final = "middle_causal_hidden"
FIRST_POSITION_CHANNEL: Final = "input_token_embedding"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _channel(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], name: str
) -> np.ndarray:
    if name not in channels:
        raise ValueError(f"peak recognition needs the {name} channel")
    offsets = np.cumsum((0, *widths))
    index = list(channels).index(name)
    return np.asarray(hidden[:, offsets[index] : offsets[index + 1]], dtype=np.float64)


def _unit_rows(block: np.ndarray) -> np.ndarray:
    return block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-9)


def _tagger_features(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]
) -> np.ndarray:
    return np.concatenate(
        [_unit_rows(_channel(hidden, channels, widths, name)) for name in TAGGER_CHANNELS], axis=1
    )


def _labeler_feature(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], end: int
) -> np.ndarray:
    return _unit_rows(_channel(hidden[end - 1 : end], channels, widths, LABELER_CHANNEL))[0]


def _first_feature(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]
) -> np.ndarray:
    return _unit_rows(_channel(hidden[:1], channels, widths, FIRST_POSITION_CHANNEL))[0]


def _probabilities(head: LinearClassifierHead, rows: np.ndarray) -> np.ndarray:
    logits = np.asarray(rows, dtype=np.float64) @ np.asarray(
        head.weight, dtype=np.float64
    ).T + np.asarray(head.bias, dtype=np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _head_dict(head: LinearClassifierHead) -> dict[str, Any]:
    return {
        "labels": list(head.labels),
        "weight": np.asarray(head.weight, dtype=np.float32).tolist(),
        "bias": np.asarray(head.bias, dtype=np.float32).tolist(),
    }


def _head_from_dict(value: Mapping[str, Any]) -> LinearClassifierHead:
    return LinearClassifierHead(
        tuple(str(label) for label in value["labels"]),
        np.asarray(value["weight"], dtype=np.float32),
        np.asarray(value["bias"], dtype=np.float32),
    )


@dataclass(frozen=True)
class PeakOperationRecognizer:
    """Operation spans located by peaks of a token tagger and named at their last token."""

    tagger: LinearClassifierHead
    labeler: LinearClassifierHead
    first_tagger: LinearClassifierHead
    first_labeler: LinearClassifierHead
    #: How many of the request's local maxima become operation candidates.
    peak_limit: int
    #: A span keeps growing while a neighbour scores at least this share of its peak.
    span_floor_ratio: float
    #: Operation names offered per span, best first.
    label_limit: int
    fit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.tagger.labels != (BACKGROUND_TAG, OPERATION_TAG)
            or self.first_tagger.labels != (BACKGROUND_TAG, OPERATION_TAG)
            or self.labeler.labels != self.first_labeler.labels
            or OPERATION_BACKGROUND_LABEL in self.labeler.labels
            or type(self.peak_limit) is not int
            or self.peak_limit < 1
            or not 0.0 < float(self.span_floor_ratio) <= 1.0
            or type(self.label_limit) is not int
            or not 1 <= self.label_limit <= len(self.labeler.labels)
        ):
            raise ValueError("peak operation recognizer parameters are invalid")

    def operation_probabilities(
        self, hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]
    ) -> np.ndarray:
        """How much each token reads as part of an operation phrase."""
        scores = _probabilities(self.tagger, _tagger_features(hidden, channels, widths))[:, 1]
        scores[0] = _probabilities(
            self.first_tagger, _first_feature(hidden, channels, widths)[None]
        )[0, 1]
        return scores

    def _labels(
        self, hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], span: TokenSpan
    ) -> np.ndarray:
        if span.end == 1:
            return _probabilities(
                self.first_labeler, _first_feature(hidden, channels, widths)[None]
            )[0]
        return _probabilities(
            self.labeler, _labeler_feature(hidden, channels, widths, span.end)[None]
        )[0]

    def operation_candidates(
        self,
        *,
        hidden: np.ndarray,
        input_spans: Sequence[TokenSpan],
        max_span_tokens: int,
        hidden_channels: Sequence[str],
        hidden_channel_widths: Sequence[int],
    ) -> tuple[OperationCandidate, ...]:
        """Every span and name worth a chart, strongest peak first."""
        hidden = np.asarray(hidden)
        scores = self.operation_probabilities(hidden, hidden_channels, hidden_channel_widths)
        blocked = np.zeros(len(scores), dtype=bool)
        for span in input_spans:
            blocked[span.start : span.end] = True
        scores = np.where(blocked, 0.0, scores)
        last = len(scores) - 1
        peaks = [
            t
            for t in range(len(scores))
            if scores[t] > 0
            and (t == 0 or scores[t] >= scores[t - 1])
            and (t == last or scores[t] >= scores[t + 1])
        ]
        peaks = sorted(peaks, key=lambda t: (-scores[t], t))[: self.peak_limit]
        nodes: list[OperationCandidate] = []
        seen: set[TokenSpan] = set()
        for peak in peaks:
            floor = self.span_floor_ratio * scores[peak]
            start = end = peak
            while (
                start - 1 >= 0
                and not blocked[start - 1]
                and scores[start - 1] >= floor
                and end - start + 1 < max_span_tokens
            ):
                start -= 1
            while (
                end + 1 <= last
                and not blocked[end + 1]
                and scores[end + 1] >= floor
                and end - start + 1 < max_span_tokens
            ):
                end += 1
            for span in (
                TokenSpan(start, end + 1),
                TokenSpan(peak, peak + 1),
                TokenSpan(start, peak + 1),
            ):
                if span in seen or any(
                    span.start < other.end and other.start < span.end for other in input_spans
                ):
                    continue
                seen.add(span)
                probabilities = self._labels(hidden, hidden_channels, hidden_channel_widths, span)
                evidence = math.log(max(float(scores[peak]), 1e-12))
                for index in np.argsort(-probabilities, kind="stable")[: self.label_limit]:
                    confidence = float(probabilities[index])
                    nodes.append(
                        OperationCandidate(
                            span=span,
                            operation=self.labeler.labels[int(index)],
                            score=evidence + math.log(max(confidence, 1e-12)),
                            pointer_score=evidence,
                            confidence=confidence,
                        )
                    )
        return tuple(nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PEAK_RECOGNIZER_SCHEMA,
            "tagger": _head_dict(self.tagger),
            "labeler": _head_dict(self.labeler),
            "first_tagger": _head_dict(self.first_tagger),
            "first_labeler": _head_dict(self.first_labeler),
            "tagger_channels": list(TAGGER_CHANNELS),
            "labeler_channel": LABELER_CHANNEL,
            "first_position_channel": FIRST_POSITION_CHANNEL,
            "peak_limit": self.peak_limit,
            "span_floor_ratio": float(self.span_floor_ratio),
            "label_limit": self.label_limit,
            "fit_receipt": dict(self.fit_receipt),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_dict())


def peak_operation_recognizer_from_dict(value: Mapping[str, Any]) -> PeakOperationRecognizer:
    if (
        value.get("schema") != PEAK_RECOGNIZER_SCHEMA
        or list(value.get("tagger_channels", ())) != list(TAGGER_CHANNELS)
        or value.get("labeler_channel") != LABELER_CHANNEL
        or value.get("first_position_channel") != FIRST_POSITION_CHANNEL
    ):
        raise ValueError("peak operation recognizer payload is not this schema")
    return PeakOperationRecognizer(
        tagger=_head_from_dict(value["tagger"]),
        labeler=_head_from_dict(value["labeler"]),
        first_tagger=_head_from_dict(value["first_tagger"]),
        first_labeler=_head_from_dict(value["first_labeler"]),
        peak_limit=int(value["peak_limit"]),
        span_floor_ratio=float(value["span_floor_ratio"]),
        label_limit=int(value["label_limit"]),
        fit_receipt=dict(value["fit_receipt"]),
    )


def _centred_head(
    labels: tuple[str, ...],
    features: np.ndarray,
    targets: Sequence[str],
    *,
    balanced: bool,
    mean: np.ndarray | None = None,
) -> LinearClassifierHead:
    """A logistic readout fitted on centred rows, with the centring folded into the bias."""
    from sklearn.linear_model import LogisticRegression

    mean = features.mean(axis=0) if mean is None else mean
    fitted = LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced" if balanced else None
    ).fit(features - mean, list(targets))
    classes = tuple(str(value) for value in fitted.classes_.tolist())
    weight = np.asarray(fitted.coef_, dtype=np.float64)
    bias = np.asarray(fitted.intercept_, dtype=np.float64)
    if len(classes) == 2 and weight.shape[0] == 1:
        weight = np.concatenate((np.zeros_like(weight), weight), axis=0)
        bias = np.concatenate((np.zeros_like(bias), bias), axis=0)
    bias = bias - weight @ mean
    order = [classes.index(label) for label in labels]
    return LinearClassifierHead(
        labels, weight[order].astype(np.float32), bias[order].astype(np.float32)
    )


def fit_peak_operation_recognizer(
    examples: Sequence[Any],
    *,
    peak_limit: int = 10,
    span_floor_ratio: float = 0.5,
    label_limit: int = 2,
) -> PeakOperationRecognizer:
    """Fit every readout on training rows; any other split is refused."""
    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in examples):
        raise ValueError("peak operation recognition is fitted on training rows only")
    token_rows, token_tags, first_rows = [], [], []
    label_rows, label_firsts, label_names = [], [], []
    for item in examples:
        hidden = np.asarray(item.hidden_states)
        channels, widths = item.hidden_channels, item.hidden_channel_widths
        inside = np.zeros(hidden.shape[0], dtype=bool)
        for instruction in item.ir.instructions:
            inside[instruction.operation_span.start : instruction.operation_span.end] = True
        for span in item.ir.input_spans:
            inside[span.start : span.end] = False
        token_rows.append(_tagger_features(hidden, channels, widths))
        token_tags.extend(OPERATION_TAG if flag else BACKGROUND_TAG for flag in inside)
        first_rows.append(_unit_rows(_channel(hidden, channels, widths, FIRST_POSITION_CHANNEL)))
        for instruction in item.ir.instructions:
            end = instruction.operation_span.end
            label_rows.append(_labeler_feature(hidden, channels, widths, end))
            label_firsts.append(
                _unit_rows(
                    _channel(hidden[end - 1 : end], channels, widths, FIRST_POSITION_CHANNEL)
                )[0]
            )
            label_names.append(instruction.op)
    operations = tuple(sorted(set(label_names)))
    if len(operations) < 2:
        raise ValueError("peak operation recognition needs at least two operations")
    tags = (BACKGROUND_TAG, OPERATION_TAG)
    first_matrix = np.concatenate(first_rows)
    # The first-position readouts see every training token's input embedding,
    # because what the first position says is lexical and any word can open a request.
    first_tagger = _centred_head(tags, first_matrix, token_tags, balanced=True)
    first_labeler = _centred_head(
        operations,
        np.stack(label_firsts),
        label_names,
        balanced=False,
        mean=first_matrix.mean(axis=0),
    )
    receipt = {
        "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
        "training_rows": len(examples),
        "operation_tokens": int(sum(tag == OPERATION_TAG for tag in token_tags)),
        "operations": list(operations),
        "splits_used": ["train"],
    }
    return PeakOperationRecognizer(
        tagger=_centred_head(tags, np.concatenate(token_rows), token_tags, balanced=True),
        labeler=_centred_head(operations, np.stack(label_rows), label_names, balanced=False),
        first_tagger=first_tagger,
        first_labeler=first_labeler,
        peak_limit=peak_limit,
        span_floor_ratio=span_floor_ratio,
        label_limit=label_limit,
        fit_receipt={**receipt, "receipt_sha256": _sha(receipt)},
    )


class PeakRecognitionTransducer:
    """A compositional transducer whose operations come from peak recognition.

    Everything but operation proposal and chart selection is the base
    transducer's, so graders and scorers that read its heads read the same heads.
    ``ownership``, when given, adds where each mention stands to its argument
    scores (core/learning/semantic_argument_ownership.py).
    """

    def __init__(
        self, base: Any, recognizer: PeakOperationRecognizer, ownership: Any = None
    ) -> None:
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "recognizer", recognizer)
        object.__setattr__(self, "ownership", ownership)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a peak recognition transducer is immutable")

    def decode(self, **kwargs: Any) -> Any:
        if self.ownership is not None:
            kwargs["argument_ownership"] = self.ownership
        return self.base.decode(**kwargs, operation_recognizer=self.recognizer)

    def _identity(self) -> dict[str, Any]:
        identity = {
            "schema": PEAK_TRANSDUCER_SCHEMA,
            "base": self.base.receipt_sha256,
            "recognizer": self.recognizer.identity_sha256,
        }
        if self.ownership is not None:
            identity["ownership"] = self.ownership.identity_sha256
        return identity

    @property
    def receipt_sha256(self) -> str:
        return _sha(self._identity())

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": PEAK_TRANSDUCER_SCHEMA,
            "base": self.base.to_dict(),
            "recognizer": self.recognizer.to_dict(),
        }
        if self.ownership is not None:
            value["ownership"] = self.ownership.to_dict()
        return value


def peak_recognition_transducer_from_dict(
    value: Mapping[str, Any], *, restore_base: Callable[[Mapping[str, Any]], Any]
) -> PeakRecognitionTransducer:
    """Restore a saved candidate; ``restore_base`` restores the transducer it wraps."""
    from core.learning.semantic_argument_ownership import argument_ownership_from_dict

    if value.get("schema") != PEAK_TRANSDUCER_SCHEMA:
        raise ValueError("peak recognition transducer payload is not this schema")
    return PeakRecognitionTransducer(
        restore_base(value["base"]),
        peak_operation_recognizer_from_dict(value["recognizer"]),
        argument_ownership_from_dict(value["ownership"]) if "ownership" in value else None,
    )
