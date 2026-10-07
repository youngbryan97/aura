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

A causal model's state at a verb cannot hold what the rest of its phrase says.
"take 7 away from 46" reads as addition at "take", "share 31 evenly among 5"
as addition at "share"; the operation is only settled at "away from" or
"evenly among". With phrase reading, two more readouts name an operation where
its phrase has closed: the middle layer at the phrase's last word, and the
mean over the phrase. Where a phrase closes depends on where the next
operation begins, which is the chart's decision, so phrase reading renames a
chart's operations after the chart is chosen (relabel_chart): each phrase runs
to the next operation of that chart in its sentence, or to the sentence's
end, less literals and punctuation; training reads at the close its gold
operations give. Candidates are still named from the verb and its word. Held
out a wording at a time over 24 training wordings (6 October), the reading at
the verb named 65.8% of unseen wordings and the three pooled, closed this
way, 86.5%; held-out constructions stayed at 0.995 and validation at 0.993.
A close guessed per candidate (the next tagger peak at least as strong) ran
an outer operation's phrase through a weaker inner one: on held-out nested
nominals ("the quotient of 84 divided by the quotient of 47 divided by 3")
it read 44 of 64 where the verb alone read 56.

Every parameter is fitted on training rows only. Validation and test rows are
refused by the fitter.
"""

from __future__ import annotations

import base64
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
#: Where a stacked labeler reads the words: the span's mean, the tagger's peak
#: token, or the whole word holding the peak.
LEXICAL_AT: Final = ("span", "peak", "word")
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


def _lexical_feature(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], span: TokenSpan
) -> np.ndarray:
    """The span's words, out of context: the mean of their input embeddings."""
    rows = _unit_rows(_channel(hidden[span.start : span.end], channels, widths, FIRST_POSITION_CHANNEL))
    mean = rows.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def _word_at(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], token: int
) -> np.ndarray:
    """One token's word out of context: its input embedding.

    Read at the tagger's peak, the word that names the operation. A span's mean
    is mostly the words around it ("after", "the", "only"), and the decoder
    names spans of every length grown from one peak.
    """
    return _unit_rows(_channel(hidden[token : token + 1], channels, widths, FIRST_POSITION_CHANNEL))[0]


def _words_at(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], start: int, end: int
) -> np.ndarray:
    rows = _unit_rows(_channel(hidden[start:end], channels, widths, FIRST_POSITION_CHANNEL))
    mean = rows.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def _word_around(token_ids: Sequence[int], token: int, continuations: frozenset[int]) -> tuple[int, int]:
    """The tokens of the word holding ``token``: "multip" and "licity" are one word."""
    start = token
    while start > 0 and int(token_ids[start]) in continuations:
        start -= 1
    end = token + 1
    while end < len(token_ids) and int(token_ids[end]) in continuations:
        end += 1
    return start, end


def word_continuation_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """The tokenizer's tokens that carry on the word before them: text that starts with a letter."""
    return tuple(sorted(
        int(token_id) for token_id in tokenizer.get_vocab().values()
        if (tokenizer.decode([token_id]) or "")[:1].isalpha()
    ))


def _bitmap(ids: frozenset[int]) -> dict[str, Any]:
    size = max(ids, default=-1) + 1
    flags = np.zeros(size, dtype=np.uint8)
    flags[list(ids)] = 1
    return {"size": size, "bits": base64.b64encode(np.packbits(flags).tobytes()).decode("ascii")}


def _from_bitmap(value: Mapping[str, Any]) -> frozenset[int]:
    flags = np.unpackbits(np.frombuffer(base64.b64decode(value["bits"]), dtype=np.uint8))[: int(value["size"])]
    return frozenset(int(index) for index in np.flatnonzero(flags))


def _first_feature(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]
) -> np.ndarray:
    return _word_at(hidden, channels, widths, 0)


def punctuation_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """The tokenizer's tokens with no letter or digit: a phrase does not close on one."""
    return tuple(sorted(
        int(token_id) for token_id in tokenizer.get_vocab().values()
        if not any(char.isalnum() for char in (tokenizer.decode([token_id]) or ""))
    ))


def _sentence_starts(token_ids: Sequence[int], ends: frozenset[int], blocked: np.ndarray) -> list[int]:
    """Where sentences begin: after a sentence-ending token outside a literal."""
    return [0] + [
        index + 1 for index, token in enumerate(token_ids)
        if int(token) in ends and not blocked[index] and index + 1 < len(token_ids)
    ]


def _phrase_close(
    span: TokenSpan,
    others: Sequence[TokenSpan],
    starts: Sequence[int],
    blocked: np.ndarray,
    token_ids: Sequence[int],
    punctuation: frozenset[int],
) -> int:
    """The last token of the phrase ``span`` opens, given the other operations ``others``.

    The phrase runs to the next of ``others`` in its sentence or to the
    sentence's end, whichever comes first. Literals and punctuation do not
    close it.
    """
    stop = min([start for start in starts if start > span.start] + [len(token_ids)])
    stop = min([other.start for other in others if span.end <= other.start < stop] + [stop])
    close = stop - 1
    while close > span.end - 1 and (blocked[close] or int(token_ids[close]) in punctuation):
        close -= 1
    return close


def _phrase_feature(
    hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int], start: int, close: int
) -> np.ndarray:
    """The phrase's mean in the middle layer, from its first token to its close."""
    mean = _unit_rows(_channel(hidden[start : close + 1], channels, widths, LABELER_CHANNEL)).mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def _probabilities(head: LinearClassifierHead, rows: np.ndarray) -> np.ndarray:
    logits = np.asarray(rows, dtype=np.float64) @ np.asarray(
        head.weight, dtype=np.float64
    ).T + np.asarray(head.bias, dtype=np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _operation_probabilities(
    tagger: LinearClassifierHead,
    first_tagger: LinearClassifierHead,
    hidden: np.ndarray,
    channels: Sequence[str],
    widths: Sequence[int],
) -> np.ndarray:
    scores = _probabilities(tagger, _tagger_features(hidden, channels, widths))[:, 1]
    scores[0] = _probabilities(first_tagger, _first_feature(hidden, channels, widths)[None])[0, 1]
    return scores


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
    #: A readout of the span's words out of context, and how much each readout
    #: is trusted, fitted on held-out constructions (fit_peak_operation_recognizer).
    lexical_labeler: LinearClassifierHead | None = None
    label_weights: tuple[float, float] = (1.0, 0.0)
    #: Where the words are read: the span's mean ("span"), its peak ("peak"),
    #: or the word holding its peak ("word").
    lexical_at: str = "span"
    #: Tokens that carry on the word before them, bound from the tokenizer.
    word_continuations: frozenset[int] = frozenset()
    #: The words choose which operation a span names and leave how sure the
    #: span is of naming one to the context. A reader takes "integer" in "Use
    #: integer arithmetic" for an adjective from the sentence, whatever the
    #: word alone suggests. Held out on 6 October, the words read it as
    #: integer division and lost "the product of 25 and the sum of 72 and 2".
    words_name_only: bool = False
    #: Readouts of a span where its phrase has closed (_phrase_close): the middle
    #: layer at the phrase's last word, and its mean over the phrase.
    close_labeler: LinearClassifierHead | None = None
    phrase_labeler: LinearClassifierHead | None = None
    #: The four readouts' weights when a chart is renamed (context, words, close,
    #: phrase), fitted together at the contextual readout's scale. Candidates
    #: keep label_weights, fitted for the two readouts they are named with:
    #: named with the four-way weights less two terms, the words outweighed the
    #: context and 35 of 37 sequence validation requests v14 had right were lost
    #: (candidate v15, 7 October).
    chart_weights: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: Whether renaming may only replace a candidate's name with a surer one:
    #: "always" takes the renamed reading; "when_surer" takes it only if its
    #: best name is more probable than the candidate reading's best. A phrase
    #: closed at the chart's next operation can carry that operation's lead-in:
    #: "after removing the subsequently computed number only after computing
    #: the subsequently computed number with reading one indexed entry" renamed
    #: a confident subtraction to an addition (candidate v17, two validation
    #: requests G03 had right, 7 October).
    chart_revision: str = "always"
    #: Sentence-ending and punctuation tokens, bound from the tokenizer.
    sentence_ends: frozenset[int] = frozenset()
    punctuation: frozenset[int] = frozenset()

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
            or self.lexical_at not in LEXICAL_AT
            or (self.lexical_at == "word" and not self.word_continuations)
            or self.chart_revision not in ("always", "when_surer")
            or (self.close_labeler is None) != (self.phrase_labeler is None)
            or (self.close_labeler is not None and (
                self.close_labeler.labels != self.labeler.labels
                or self.phrase_labeler.labels != self.labeler.labels
                or not self.sentence_ends or not self.punctuation
                or len(self.chart_weights) != 4
                or not all(math.isfinite(w) and w >= 0.0 for w in self.chart_weights)
                or self.lexical_labeler is None
            ))
        ):
            raise ValueError("peak operation recognizer parameters are invalid")

    def operation_probabilities(
        self, hidden: np.ndarray, channels: Sequence[str], widths: Sequence[int]
    ) -> np.ndarray:
        """How much each token reads as part of an operation phrase."""
        return _operation_probabilities(self.tagger, self.first_tagger, hidden, channels, widths)

    def _labels(
        self,
        hidden: np.ndarray,
        channels: Sequence[str],
        widths: Sequence[int],
        span: TokenSpan,
        peak: int,
        token_ids: Sequence[int] | None,
        close: int | None = None,
    ) -> np.ndarray:
        if span.end == 1:
            return _probabilities(
                self.first_labeler, _first_feature(hidden, channels, widths)[None]
            )[0]
        contextual = _probabilities(
            self.labeler, _labeler_feature(hidden, channels, widths, span.end)[None]
        )[0]
        if self.lexical_labeler is None and self.close_labeler is None:
            return contextual
        renaming = self.close_labeler is not None and close is not None
        weights = self.chart_weights if renaming else self.label_weights
        logits = weights[0] * np.log(np.clip(contextual, 1e-12, 1.0))
        if self.lexical_labeler is not None:
            if self.lexical_at == "word":
                if token_ids is None:
                    raise ValueError("the word readout needs the request's token ids")
                words = _words_at(hidden, channels, widths, *_word_around(token_ids, peak, self.word_continuations))
            elif self.lexical_at == "peak":
                words = _word_at(hidden, channels, widths, peak)
            else:
                words = _lexical_feature(hidden, channels, widths, span)
            lexical = _probabilities(self.lexical_labeler, words[None])[0]
            logits = logits + weights[1] * np.log(np.clip(lexical, 1e-12, 1.0))
        if renaming:
            closing = _probabilities(self.close_labeler, _labeler_feature(hidden, channels, widths, close + 1)[None])[0]
            phrase = _probabilities(self.phrase_labeler, _phrase_feature(hidden, channels, widths, span.start, close)[None])[0]
            logits = (logits + weights[2] * np.log(np.clip(closing, 1e-12, 1.0))
                      + weights[3] * np.log(np.clip(phrase, 1e-12, 1.0)))
        logits -= logits.max()
        pooled = np.exp(logits) / np.exp(logits).sum()
        if not self.words_name_only:
            return pooled
        # The words say which operation; the context alone says how sure it
        # is that the span names one. The best name keeps the contextual
        # readout's confidence and the rest are ranked by the pooled reading.
        return float(contextual.max()) * pooled / float(pooled.max())

    def operation_candidates(
        self,
        *,
        hidden: np.ndarray,
        input_spans: Sequence[TokenSpan],
        max_span_tokens: int,
        hidden_channels: Sequence[str],
        hidden_channel_widths: Sequence[int],
        token_ids: Sequence[int] | None = None,
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
                probabilities = self._labels(
                    hidden, hidden_channels, hidden_channel_widths, span, peak, token_ids
                )
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

    def relabel_chart(
        self,
        nodes: Sequence[Any],
        *,
        hidden: np.ndarray,
        input_spans: Sequence[TokenSpan],
        hidden_channels: Sequence[str],
        hidden_channel_widths: Sequence[int],
        token_ids: Sequence[int],
    ) -> tuple[Any, ...]:
        """A chosen chart's operations named again, each where its phrase closes in that chart.

        Without phrase readouts the chart is returned as it is. A node keeps its
        span and pointer evidence; its name is the pooled reading's best and its
        score the evidence plus that name's log-confidence, as candidates are scored.
        """
        if self.close_labeler is None:
            return tuple(nodes)
        hidden = np.asarray(hidden)
        blocked = np.zeros(hidden.shape[0], dtype=bool)
        for literal in input_spans:
            blocked[literal.start : literal.end] = True
        scores = np.where(blocked, 0.0, self.operation_probabilities(hidden, hidden_channels, hidden_channel_widths))
        starts = _sentence_starts(token_ids, self.sentence_ends, blocked)
        spans = [node.span for node in nodes]
        renamed = []
        for node in nodes:
            span = node.span
            peak = span.start + int(np.argmax(scores[span.start : span.end]))
            close = _phrase_close(span, [other for other in spans if other != span], starts, blocked,
                                  token_ids, self.punctuation)
            probabilities = self._labels(
                hidden, hidden_channels, hidden_channel_widths, span, peak, token_ids, close
            )
            best = int(np.argmax(probabilities))
            confidence = float(probabilities[best])
            if self.chart_revision == "when_surer":
                first = self._labels(hidden, hidden_channels, hidden_channel_widths, span, peak, token_ids)
                if float(np.max(first)) >= confidence:
                    renamed.append(node)
                    continue
            renamed.append(type(node)(
                span=span, operation=self.labeler.labels[best],
                score=node.pointer_score + math.log(max(confidence, 1e-12)),
                pointer_score=node.pointer_score, confidence=confidence,
            ))
        return tuple(renamed)

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
            **(
                {
                    "lexical_labeler": _head_dict(self.lexical_labeler),
                    "label_weights": [float(weight) for weight in self.label_weights],
                    **({"lexical_at": self.lexical_at} if self.lexical_at != "span" else {}),
                    **(
                        {"word_continuations": _bitmap(self.word_continuations)}
                        if self.lexical_at == "word"
                        else {}
                    ),
                    **({"words_name_only": True} if self.words_name_only else {}),
                }
                if self.lexical_labeler is not None
                else {}
            ),
            **(
                {
                    "close_labeler": _head_dict(self.close_labeler),
                    "phrase_labeler": _head_dict(self.phrase_labeler),
                    "chart_weights": [float(weight) for weight in self.chart_weights],
                    **({"chart_revision": self.chart_revision} if self.chart_revision != "always" else {}),
                    "sentence_ends": _bitmap(self.sentence_ends),
                    "punctuation": _bitmap(self.punctuation),
                }
                if self.close_labeler is not None
                else {}
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_dict())


def peak_operation_recognizer_from_dict(value: Mapping[str, Any]) -> PeakOperationRecognizer:
    if "phrase_weights" in value:
        raise ValueError(
            "this recognizer's phrase readouts name candidates at a guessed close (commits f87331dc0 to "
            "bb315ef74, candidates v13 and v14); read it with that code"
        )
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
        lexical_labeler=_head_from_dict(value["lexical_labeler"]) if "lexical_labeler" in value else None,
        label_weights=tuple(float(weight) for weight in value.get("label_weights", (1.0, 0.0))),
        lexical_at=str(value.get("lexical_at", "span")),
        word_continuations=(
            _from_bitmap(value["word_continuations"]) if "word_continuations" in value else frozenset()
        ),
        words_name_only=bool(value.get("words_name_only", False)),
        close_labeler=_head_from_dict(value["close_labeler"]) if "close_labeler" in value else None,
        phrase_labeler=_head_from_dict(value["phrase_labeler"]) if "phrase_labeler" in value else None,
        chart_weights=tuple(float(weight) for weight in value.get("chart_weights", (0.0, 0.0, 0.0, 0.0))),
        chart_revision=str(value.get("chart_revision", "always")),
        sentence_ends=_from_bitmap(value["sentence_ends"]) if "sentence_ends" in value else frozenset(),
        punctuation=_from_bitmap(value["punctuation"]) if "punctuation" in value else frozenset(),
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


def _stacked_weights(
    readouts: Sequence[np.ndarray],
    names: Sequence[str],
    groups: Sequence[Any],
    operations: tuple[str, ...],
) -> tuple[float, ...]:
    """How much to trust each readout, from how each does on constructions it was not fitted on.

    Every readout is refitted without each held-out group and scores that
    group; the weights maximise the likelihood of those out-of-fold labels
    under the readouts' pooled log-probabilities. Fitted on the same rows the
    readouts saw, the contextual readout is near perfect and would take all
    the weight; the question is how each does on a wording it has not seen,
    which is what a held-out construction is.
    """
    from scipy.optimize import minimize

    groups = np.asarray(list(groups))
    target = np.asarray([operations.index(name) for name in names])
    out = [np.zeros((len(target), len(operations))) for _ in readouts]
    for group in sorted(set(groups.tolist())):
        held = groups == group
        kept_names = [name for name, keep in zip(names, ~held, strict=True) if keep]
        if len(set(kept_names)) < len(operations):
            continue
        for index, rows in enumerate(readouts):
            out[index][held] = _probabilities(
                _centred_head(operations, rows[~held], kept_names, balanced=False), rows[held]
            )
    scored = out[0].sum(axis=1) > 0
    if not scored.any():
        return (1.0, *(0.0 for _ in readouts[1:]))  # no group could be held out: the contextual readout alone
    logs = [np.log(np.clip(matrix[scored], 1e-12, 1.0)) for matrix in out]
    rows = np.arange(int(scored.sum()))

    def loss(weights: np.ndarray) -> float:
        logits = sum(weight * log for weight, log in zip(weights, logs, strict=True))
        logits = logits - logits.max(axis=1, keepdims=True)
        return -float((logits[rows, target[scored]] - np.log(np.exp(logits).sum(axis=1))).mean())

    start = np.zeros(len(readouts))
    start[0] = 1.0
    fitted = minimize(loss, x0=start, method="L-BFGS-B", bounds=[(0.0, None)] * len(readouts))
    weights = [float(value) for value in fitted.x]
    if weights[0] <= 0.0:
        top = max(weights)
        return tuple(weight / top for weight in weights) if top > 0 else (1.0, *(0.0 for _ in weights[1:]))
    # The trust between readouts, at the contextual readout's own scale. The
    # fit's overall size is a temperature for the held-out likelihood, and the
    # chart adds a span's log-confidence to scores whose scale was set by the
    # contextual readout alone. Fitted (16.29, 7.75) as given, LIVE validation
    # 2026-10-05 lost five cataphoric rows; at (1, 0.48) it lost none.
    return tuple(weight / weights[0] for weight in weights)


def _stacked_label_weights(
    contextual: np.ndarray,
    lexical: np.ndarray,
    names: Sequence[str],
    groups: Sequence[Any],
    operations: tuple[str, ...],
) -> tuple[float, float]:
    """The contextual and lexical readouts' weights (_stacked_weights)."""
    first, second = _stacked_weights((contextual, lexical), names, groups, operations)
    return first, second


def fit_peak_operation_recognizer(
    examples: Sequence[Any],
    *,
    peak_limit: int = 10,
    span_floor_ratio: float = 0.5,
    label_limit: int = 2,
    construction_groups: Mapping[str, Any] | None = None,
    lexical_at: str = "span",
    word_continuations: Sequence[int] = (),
    words_name_only: bool = False,
    phrase_reading: bool = False,
    sentence_end_token_ids: Sequence[int] = (),
    punctuation_token_ids: Sequence[int] = (),
    chart_weighting: str = "fitted",
    phrase_examples: Sequence[Any] = (),
    chart_revision: str = "always",
) -> PeakOperationRecognizer:
    """Fit every readout on training rows; any other split is refused.

    With ``construction_groups`` (source sha256 to held-out group), a readout of
    the span's words out of context joins the contextual one, each weighted by
    how it does on constructions it was not fitted on. LIVE validation
    2026-10-05: "after removing", a wording training never had, read at its
    last token as add 0.347 against sub 0.346; the words alone read it as sub.

    ``lexical_at="peak"`` reads the words at the token the fitted tagger scores
    highest inside each operation span, which is where the decoder reads them;
    ``"word"`` reads the whole word holding that token, with
    ``word_continuations`` from ``word_continuation_token_ids``.

    ``phrase_reading`` adds the readouts at each operation's phrase close,
    read at the close its gold operations give, and needs
    ``construction_groups``, ``sentence_end_token_ids`` and
    ``punctuation_token_ids`` (punctuation_token_ids). ``chart_weighting``
    "fitted" weighs the four readouts by held-out likelihood; "equal" gives
    each one vote. Fitted, the context's weight went to zero: nothing held out
    in training shows the phrase wrong and the verb right, and on validation's
    "counting copies of one value with selector 1" the phrase read a lookup.
    """
    if chart_weighting not in ("fitted", "equal"):
        raise ValueError("chart weighting is fitted or equal")
    # ``phrase_examples`` teach where operations are and where their phrases
    # settle (tagger, phrase readouts), not what a verb or its word names:
    # wordings built with misleading verbs ("take ... lots of", "share ...
    # among") taught the candidate labelers that verbs mislead, and "apply
    # multiplicity calculation", a count, was read as a multiplication
    # (candidate v16, two validation requests, 7 October).
    phrase_examples = tuple(phrase_examples)
    if phrase_examples and not phrase_reading:
        raise ValueError("phrase examples only join a fit with phrase reading")
    continuations = frozenset(int(token_id) for token_id in word_continuations)
    ends = frozenset(int(token_id) for token_id in sentence_end_token_ids)
    punctuation = frozenset(int(token_id) for token_id in punctuation_token_ids)
    if phrase_reading and (construction_groups is None or not ends or not punctuation):
        raise ValueError("phrase reading needs held-out groups, sentence ends and punctuation")
    examples = tuple(examples)
    if not examples or any(item.split != "train" for item in (*examples, *phrase_examples)):
        raise ValueError("peak operation recognition is fitted on training rows only")
    token_rows, token_tags, first_rows = [], [], []
    label_rows, label_firsts, label_names, label_spans, label_groups = [], [], [], [], []
    naming: list[bool] = []
    naming_set = {id(item) for item in examples}
    for item in (*examples, *phrase_examples):
        names_operations = id(item) in naming_set
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
            naming.append(names_operations)
            if construction_groups is not None:
                label_spans.append((item, instruction.operation_span))
                label_groups.append(construction_groups[item.ir.source_text_sha256])
    naming_rows = np.asarray(naming)
    named = [name for name, keep in zip(label_names, naming, strict=True) if keep]
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
        np.stack(label_firsts)[naming_rows],
        named,
        balanced=False,
        mean=first_matrix.mean(axis=0),
    )
    tagger = _centred_head(tags, np.concatenate(token_rows), token_tags, balanced=True)
    lexical_labeler, label_weights = None, (1.0, 0.0)
    close_labeler = phrase_labeler = None
    chart_weights = (0.0, 0.0, 0.0, 0.0)
    if construction_groups is not None:
        lexical_matrix = np.stack([
            _training_words(item, span, tagger, first_tagger, lexical_at, continuations)
            for item, span in label_spans
        ])
        lexical_labeler = _centred_head(operations, lexical_matrix[naming_rows], named, balanced=False)
        label_weights = _stacked_label_weights(
            np.stack(label_rows)[naming_rows], lexical_matrix[naming_rows], named,
            [group for group, keep in zip(label_groups, naming, strict=True) if keep], operations,
        )
        if phrase_reading:
            close_matrix, phrase_matrix = _training_phrases(
                label_spans, tagger, first_tagger, ends, punctuation
            )
            close_labeler = _centred_head(operations, close_matrix, label_names, balanced=False)
            phrase_labeler = _centred_head(operations, phrase_matrix, label_names, balanced=False)
            weights = _stacked_weights(
                (np.stack(label_rows), lexical_matrix, close_matrix, phrase_matrix),
                label_names, label_groups, operations,
            )
            chart_weights = tuple(weights) if chart_weighting == "fitted" else (1.0, 1.0, 1.0, 1.0)
    receipt = {
        "training_sources": sorted(item.ir.source_text_sha256 for item in examples),
        "training_rows": len(examples),
        **({"phrase_sources": sorted(item.ir.source_text_sha256 for item in phrase_examples)}
           if phrase_examples else {}),
        "operation_tokens": int(sum(tag == OPERATION_TAG for tag in token_tags)),
        "operations": list(operations),
        "splits_used": ["train"],
        **({"label_weights": list(label_weights)} if lexical_labeler is not None else {}),
        **({"lexical_at": lexical_at} if lexical_labeler is not None and lexical_at != "span" else {}),
        **({"words_name_only": True} if lexical_labeler is not None and words_name_only else {}),
        **({"chart_weights": list(chart_weights), "chart_weighting": chart_weighting,
            "chart_revision": chart_revision} if close_labeler is not None else {}),
    }
    return PeakOperationRecognizer(
        tagger=tagger,
        labeler=_centred_head(operations, np.stack(label_rows)[naming_rows], named, balanced=False),
        first_tagger=first_tagger,
        first_labeler=first_labeler,
        peak_limit=peak_limit,
        span_floor_ratio=span_floor_ratio,
        label_limit=label_limit,
        fit_receipt={**receipt, "receipt_sha256": _sha(receipt)},
        lexical_labeler=lexical_labeler,
        label_weights=label_weights,
        lexical_at=lexical_at,
        word_continuations=continuations if lexical_at == "word" else frozenset(),
        words_name_only=words_name_only and lexical_labeler is not None,
        close_labeler=close_labeler,
        phrase_labeler=phrase_labeler,
        chart_weights=chart_weights,
        sentence_ends=ends if close_labeler is not None else frozenset(),
        punctuation=punctuation if close_labeler is not None else frozenset(),
        chart_revision=chart_revision,
    )


def _training_phrases(
    label_spans: Sequence[tuple[Any, TokenSpan]],
    tagger: LinearClassifierHead,
    first_tagger: LinearClassifierHead,
    ends: frozenset[int],
    punctuation: frozenset[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Each training operation's close and phrase rows, its close given by its request's gold operations.

    The decoder closes a phrase at the next operation of the chart it chose;
    a training request's chart is its gold one.
    """
    del tagger, first_tagger
    close_rows, phrase_rows = [], []
    for item, span in label_spans:
        hidden = np.asarray(item.hidden_states)
        channels, widths = item.hidden_channels, item.hidden_channel_widths
        blocked = np.zeros(hidden.shape[0], dtype=bool)
        for literal in item.ir.input_spans:
            blocked[literal.start : literal.end] = True
        tokens = item.ir.source_token_ids
        others = [ins.operation_span for ins in item.ir.instructions if ins.operation_span != span]
        close = _phrase_close(span, others, _sentence_starts(tokens, ends, blocked), blocked, tokens, punctuation)
        close_rows.append(_labeler_feature(hidden, channels, widths, close + 1))
        phrase_rows.append(_phrase_feature(hidden, channels, widths, span.start, close))
    return np.stack(close_rows), np.stack(phrase_rows)


def _training_words(
    item: Any,
    span: TokenSpan,
    tagger: LinearClassifierHead,
    first_tagger: LinearClassifierHead,
    lexical_at: str,
    continuations: frozenset[int],
) -> np.ndarray:
    """A training operation's words, read where the decoder will read them."""
    hidden = np.asarray(item.hidden_states)
    channels, widths = item.hidden_channels, item.hidden_channel_widths
    if lexical_at == "span":
        return _lexical_feature(hidden, channels, widths, span)
    scores = _operation_probabilities(tagger, first_tagger, hidden, channels, widths)
    for literal in item.ir.input_spans:
        scores[literal.start : literal.end] = 0.0
    peak = span.start + int(np.argmax(scores[span.start : span.end]))
    if lexical_at == "word":
        return _words_at(
            hidden, channels, widths, *_word_around(item.ir.source_token_ids, peak, continuations)
        )
    return _word_at(hidden, channels, widths, peak)


class PeakRecognitionTransducer:
    """A compositional transducer whose operations come from peak recognition.

    Everything but operation proposal and chart selection is the base
    transducer's, so graders and scorers that read its heads read the same heads.
    ``ownership``, when given, adds where each mention stands to its argument
    scores (core/learning/semantic_argument_ownership.py), and ``antecedent``
    where each name was given (core/learning/semantic_argument_antecedent.py).
    """

    def __init__(
        self,
        base: Any,
        recognizer: PeakOperationRecognizer,
        ownership: Any = None,
        antecedent: Any = None,
    ) -> None:
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "recognizer", recognizer)
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(self, "antecedent", antecedent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a peak recognition transducer is immutable")

    def decode(self, **kwargs: Any) -> Any:
        if self.ownership is not None:
            kwargs["argument_ownership"] = self.ownership
        if self.antecedent is not None:
            kwargs["argument_antecedent"] = self.antecedent
        return self.base.decode(**kwargs, operation_recognizer=self.recognizer)

    def _identity(self) -> dict[str, Any]:
        identity = {
            "schema": PEAK_TRANSDUCER_SCHEMA,
            "base": self.base.receipt_sha256,
            "recognizer": self.recognizer.identity_sha256,
        }
        if self.ownership is not None:
            identity["ownership"] = self.ownership.identity_sha256
        if self.antecedent is not None:
            identity["antecedent"] = self.antecedent.identity_sha256
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
        if self.antecedent is not None:
            value["antecedent"] = self.antecedent.to_dict()
        return value


def peak_recognition_transducer_from_dict(
    value: Mapping[str, Any], *, restore_base: Callable[[Mapping[str, Any]], Any]
) -> PeakRecognitionTransducer:
    """Restore a saved candidate; ``restore_base`` restores the transducer it wraps."""
    from core.learning.semantic_argument_antecedent import argument_antecedent_from_dict
    from core.learning.semantic_argument_ownership import argument_ownership_from_dict

    if value.get("schema") != PEAK_TRANSDUCER_SCHEMA:
        raise ValueError("peak recognition transducer payload is not this schema")
    return PeakRecognitionTransducer(
        restore_base(value["base"]),
        peak_operation_recognizer_from_dict(value["recognizer"]),
        argument_ownership_from_dict(value["ownership"]) if "ownership" in value else None,
        argument_antecedent_from_dict(value["antecedent"]) if "antecedent" in value else None,
    )
