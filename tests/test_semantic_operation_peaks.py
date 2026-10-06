"""Operations are found where a request reads as one and named at the span's last token."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_operation_peaks import (
    BACKGROUND_TAG,
    OPERATION_TAG,
    PeakOperationRecognizer,
    PeakRecognitionTransducer,
    fit_peak_operation_recognizer,
    peak_operation_recognizer_from_dict,
    peak_recognition_transducer_from_dict,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearClassifierHead

CHANNELS = ("input_token_embedding", "middle_causal_hidden", "final_causal_hidden")
WIDTHS = (3, 3, 3)
# Each channel's three columns: reads as background, reads as an operation,
# and which operation (positive is add, negative is sub).
BACKGROUND = (1.0, 0.0, 0.0)


def _head(labels: tuple[str, ...], weight: list[list[float]]) -> LinearClassifierHead:
    return LinearClassifierHead(
        labels, np.asarray(weight, dtype=np.float32), np.zeros(len(labels), dtype=np.float32)
    )


def _recognizer(**overrides: object) -> PeakOperationRecognizer:
    tags = (BACKGROUND_TAG, OPERATION_TAG)
    names = ("add", "sub")
    fields = dict(
        tagger=_head(tags, [[0] * 6, [-4, 4, 0, -4, 4, 0]]),
        labeler=_head(names, [[0, 0, 4], [0, 0, -4]]),
        first_tagger=_head(tags, [[0] * 3, [-8, 8, 0]]),
        first_labeler=_head(names, [[0, 0, 4], [0, 0, -4]]),
        peak_limit=10,
        span_floor_ratio=0.5,
        label_limit=2,
        fit_receipt={"splits_used": ["train"]},
    )
    fields.update(overrides)
    return PeakOperationRecognizer(**fields)


def _hidden(
    contextual: list[tuple[float, float, float]], embedding: list[tuple[float, float, float]]
) -> np.ndarray:
    return np.concatenate(
        [np.asarray(embedding), np.asarray(contextual), np.asarray(contextual)], axis=1
    )


def _nodes(recognizer: PeakOperationRecognizer, hidden: np.ndarray, input_spans=()) -> tuple:
    return recognizer.operation_candidates(
        hidden=hidden,
        input_spans=input_spans,
        max_span_tokens=4,
        hidden_channels=CHANNELS,
        hidden_channel_widths=WIDTHS,
    )


def test_a_weak_reading_is_still_an_operation_where_it_stands_above_its_neighbours() -> None:
    recognizer = _recognizer()
    contextual = [BACKGROUND] * 6
    contextual[3] = (1.0, 0.8, 0.4)
    hidden = _hidden(contextual, [BACKGROUND] * 6)
    probability = recognizer.operation_probabilities(hidden, CHANNELS, WIDTHS)
    assert probability[3] < 0.5
    nodes = _nodes(recognizer, hidden)
    assert [(node.span, node.operation) for node in nodes[:2]] == [
        (TokenSpan(3, 4), "add"),
        (TokenSpan(3, 4), "sub"),
    ]
    assert nodes[0].pointer_score == pytest.approx(np.log(probability[3]), rel=1e-5)
    assert nodes[0].score > nodes[1].score
    # Ties in the background are maxima too; the limit keeps the strongest.
    assert {node.span for node in _nodes(_recognizer(peak_limit=1), hidden)} == {TokenSpan(3, 4)}


def test_a_span_grows_over_neighbours_that_keep_half_its_reading_and_is_named_at_its_end() -> None:
    contextual = [BACKGROUND] * 7
    contextual[2] = (0.0, 1.0, 0.0)
    contextual[3] = (0.0, 1.0, -0.6)
    hidden = _hidden(contextual, [BACKGROUND] * 7)
    nodes = _nodes(_recognizer(), hidden)
    by_span = {}
    for node in nodes:
        by_span.setdefault(node.span, []).append(node.operation)
    assert TokenSpan(2, 4) in by_span
    # The span ends on the token that says sub, so sub is offered first.
    assert by_span[TokenSpan(2, 4)][0] == "sub"


def test_an_input_span_is_never_read_as_an_operation() -> None:
    contextual = [BACKGROUND] * 6
    contextual[2] = contextual[3] = (0.0, 1.0, 0.5)
    hidden = _hidden(contextual, [BACKGROUND] * 6)
    nodes = _nodes(_recognizer(), hidden, input_spans=(TokenSpan(3, 4),))
    assert nodes[0].span == TokenSpan(2, 3)
    assert not any(node.span.start <= 3 < node.span.end for node in nodes)


def test_the_first_position_is_read_from_the_input_embedding() -> None:
    # A causal model's first position carries the attention sink, so its
    # contextual layers do not describe the word there. Here they read as add.
    contextual = [(0.0, 1.0, 0.9)] + [BACKGROUND] * 4
    embedding = [(0.0, 1.0, -0.5)] + [BACKGROUND] * 4
    nodes = _nodes(_recognizer(), _hidden(contextual, embedding))
    assert nodes[0].span == TokenSpan(0, 1) and nodes[0].operation == "sub"
    silent = _nodes(_recognizer(), _hidden(contextual, [BACKGROUND] * 5))
    assert all(node.pointer_score < np.log(0.01) for node in silent if node.span == TokenSpan(0, 1))


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="invalid"):
        _recognizer(span_floor_ratio=0.0)
    with pytest.raises(ValueError, match="invalid"):
        _recognizer(label_limit=3)
    with pytest.raises(ValueError, match="invalid"):
        _recognizer(first_labeler=_head(("add", "mul"), [[0, 0, 1], [0, 0, -1]]))


def _fixture_examples() -> tuple:
    from tests.test_semantic_program_shared_transducer import _examples

    return _examples()


def test_the_fitter_sees_training_rows_only() -> None:
    examples = _fixture_examples()
    with pytest.raises(ValueError, match="training rows only"):
        fit_peak_operation_recognizer(examples)
    training = tuple(item for item in examples if item.split == "train")
    recognizer = fit_peak_operation_recognizer(training)
    assert recognizer.fit_receipt["splits_used"] == ["train"]
    assert recognizer.fit_receipt["training_sources"] == sorted(
        item.ir.source_text_sha256 for item in training
    )
    replay = peak_operation_recognizer_from_dict(recognizer.to_dict())
    assert replay.identity_sha256 == recognizer.identity_sha256


def test_held_out_operations_are_found_and_named_from_the_fitted_readouts() -> None:
    examples = _fixture_examples()
    recognizer = fit_peak_operation_recognizer(
        tuple(item for item in examples if item.split == "train")
    )
    for item in (item for item in examples if item.split != "train"):
        nodes = recognizer.operation_candidates(
            hidden=item.hidden_states,
            input_spans=item.ir.input_spans,
            max_span_tokens=2,
            hidden_channels=item.hidden_channels,
            hidden_channel_widths=item.hidden_channel_widths,
        )
        first_reading = {}
        for node in nodes:
            first_reading.setdefault(node.span, node.operation)
        for instruction in item.ir.instructions:
            assert first_reading.get(instruction.operation_span) == instruction.op


def test_decode_takes_its_operations_from_the_recognizer() -> None:
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _grounding

    examples = _fixture_examples()
    base = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    recognizer = fit_peak_operation_recognizer(
        tuple(item for item in examples if item.split == "train")
    )
    candidate = PeakRecognitionTransducer(base, recognizer)
    assert candidate.receipt_sha256 != base.receipt_sha256
    assert (
        peak_recognition_transducer_from_dict(
            candidate.to_dict(), restore_base=compositional_semantic_program_transducer_from_dict
        ).receipt_sha256
        == candidate.receipt_sha256
    )
    item = next(item for item in examples if item.split == "test")
    arguments = dict(
        source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states,
        public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=base.model_basis_sha256,
    )
    decoded = candidate.decode(**arguments)
    assert decoded.ir is not None, decoded.refusal
    assert decoded.ir.to_program() == item.ir.to_program()

    # Rename the operations the readouts return and the decoded program follows them.
    renamed = tuple(reversed(recognizer.labeler.labels))
    misnamed = replace(
        recognizer,
        labeler=replace(recognizer.labeler, labels=renamed),
        first_labeler=replace(recognizer.first_labeler, labels=renamed),
    )
    wrong = PeakRecognitionTransducer(base, misnamed).decode(**arguments)
    assert wrong.ir is None or wrong.ir.to_program() != item.ir.to_program()
    with pytest.raises(ValueError, match="replaces the other proposal hooks"):
        base.decode(
            **arguments, operation_recognizer=recognizer, binding_chart_solver=lambda *a, **k: None
        )
    with pytest.raises(AttributeError, match="immutable"):
        candidate.recognizer = misnamed


def test_the_run_pairs_both_arms_by_source_split_and_family() -> None:
    from tools.run_semantic_peak_recognition import family_of, paired_development_comparison

    def row(source: str, split: str, construction: str, status: str) -> dict:
        return {
            "source_text_sha256": source,
            "split": split,
            "construction_id": construction,
            "observation": {"semantic_status": status},
        }

    incumbent = [
        row("a", "validation", "cataphoric:c1", "different"),
        row("b", "validation", "cataphoric:c1", "equivalent"),
        row("c", "train", "arithmetic:c2", "equivalent"),
    ]
    candidate = [
        row("a", "validation", "cataphoric:c1", "equivalent"),
        row("b", "validation", "cataphoric:c1", "different"),
        row("c", "train", "arithmetic:c2", "equivalent"),
    ]
    comparison = paired_development_comparison(incumbent, candidate)
    cataphoric = comparison["validation"]["families"]["cataphoric"]
    assert (cataphoric["incumbent"], cataphoric["candidate"]) == (1, 1)
    assert (cataphoric["gains"], cataphoric["losses"]) == (["a"], ["b"])
    assert comparison["train"]["candidate"] == 1 and comparison["train"]["gains"] == 0
    assert family_of("natural_source:x:y") == "natural_source"
    with pytest.raises(ValueError, match="same sources"):
        paired_development_comparison(incumbent, candidate[:2])
    with pytest.raises(ValueError, match="split or construction"):
        paired_development_comparison(
            incumbent, [*candidate[:2], row("c", "validation", "arithmetic:c2", "x")]
        )


def test_representation_differences_name_every_field_that_moved() -> None:
    from tools.run_semantic_peak_recognition import representation_differences

    first = {"model": {"path": "m", "count": 27}, "layers": [10, 20], "source": "a"}
    second = {"model": {"path": "m", "count": 4}, "layers": [10, 21], "source": "a", "extra": 1}
    assert representation_differences(first, second) == ["extra", "layers[1]", "model.count"]
    assert representation_differences(first, first) == []


def test_a_tie_in_context_is_broken_by_the_words_when_the_words_have_earned_it() -> None:
    """LIVE validation 2026-10-05: "after removing" read as add 0.347 against sub 0.346 in context."""
    tied = _recognizer(
        labeler=_head(("add", "sub"), [[0, 0, 0], [0, 0, 0]]),
        lexical_labeler=_head(("add", "sub"), [[0, 0, 4], [0, 0, -4]]),
        label_weights=(1.0, 1.0),
    )
    # Context says nothing; the span's own word embedding reads as sub.
    hidden = _hidden([BACKGROUND, (0.0, 1.0, 0.0), BACKGROUND], [BACKGROUND, (0.0, 1.0, -1.0), BACKGROUND])
    first = {}
    for node in _nodes(tied, hidden):
        first.setdefault(node.span, node.operation)
    assert first[TokenSpan(1, 2)] == "sub"
    restored = peak_operation_recognizer_from_dict(tied.to_dict())
    assert restored.label_weights == (1.0, 1.0) and restored.lexical_labeler is not None


def test_the_stacked_readout_is_fitted_with_held_out_groups() -> None:
    examples = _fixture_examples()
    train = tuple(item for item in examples if item.split == "train")
    groups = {item.ir.source_text_sha256: index % 2 for index, item in enumerate(train)}

    recognizer = fit_peak_operation_recognizer(train, construction_groups=groups)

    assert recognizer.lexical_labeler is not None
    assert all(weight >= 0.0 for weight in recognizer.label_weights)
    assert recognizer.fit_receipt["label_weights"] == list(recognizer.label_weights)
    # The words are weighed against the context at the context's own scale,
    # the scale the chart's scores were built on.
    assert recognizer.label_weights[0] in (0.0, 1.0)


def test_the_words_read_at_the_peak_name_every_span_grown_from_it() -> None:
    """v10 read a span's words as their mean, and "after" outvoted "removing".

    Every span the decoder grows from one peak shares the peak's word, so it
    shares the reading of what that word does.
    """
    contextual = [BACKGROUND, (0.5, 1.0, 0.0), (0.0, 1.0, 0.0)]
    embedding = [BACKGROUND, (0.0, 0.1, 1.0), (0.0, 1.0, -1.0)]
    hidden = _hidden(contextual, embedding)
    readings = {}
    for lexical_at in ("span", "peak"):
        tied = _recognizer(
            labeler=_head(("add", "sub"), [[0, 0, 0], [0, 0, 0]]),
            lexical_labeler=_head(("add", "sub"), [[0, 0, 4], [0, 0, -4]]),
            label_weights=(1.0, 1.0),
            lexical_at=lexical_at,
        )
        first = {}
        for node in _nodes(tied, hidden):
            first.setdefault(node.span, node.operation)
        readings[lexical_at] = first
    assert set(readings["peak"]) == {TokenSpan(1, 3), TokenSpan(2, 3)}
    assert readings["span"][TokenSpan(1, 3)] == "add"
    assert set(readings["peak"].values()) == {"sub"}


def test_the_peak_readout_is_fitted_where_the_decoder_reads_it() -> None:
    examples = _fixture_examples()
    train = tuple(item for item in examples if item.split == "train")
    groups = {item.ir.source_text_sha256: index % 2 for index, item in enumerate(train)}

    recognizer = fit_peak_operation_recognizer(train, construction_groups=groups, lexical_at="peak")

    assert recognizer.lexical_at == "peak"
    assert recognizer.fit_receipt["lexical_at"] == "peak"
    replay = peak_operation_recognizer_from_dict(recognizer.to_dict())
    assert replay.lexical_at == "peak"
    assert replay.identity_sha256 == recognizer.identity_sha256
    with pytest.raises(ValueError):
        _recognizer(lexical_at="middle")


def test_the_word_readout_takes_in_every_piece_of_the_word() -> None:
    """"multiplicity" is two tokens, "multip" and "licity"; the peak may be either."""
    from core.learning.semantic_operation_peaks import _word_around

    # 7 = " the", 8 = " multip", 9 = "licity", 10 = " of"; 9 carries on the word.
    ids = (7, 8, 9, 10)
    assert _word_around(ids, 1, frozenset({9})) == (1, 3)
    assert _word_around(ids, 2, frozenset({9})) == (1, 3)
    assert _word_around(ids, 3, frozenset({9})) == (3, 4)


def test_the_word_readout_needs_the_tokenizer_and_the_request() -> None:
    with pytest.raises(ValueError):
        _recognizer(lexical_at="word")
    contextual = [BACKGROUND, (0.0, 1.0, 0.0), BACKGROUND]
    hidden = _hidden(contextual, [BACKGROUND, (0.0, 1.0, -1.0), BACKGROUND])
    reader = _recognizer(
        lexical_labeler=_head(("add", "sub"), [[0, 0, 4], [0, 0, -4]]),
        label_weights=(1.0, 1.0),
        lexical_at="word",
        word_continuations=frozenset({99}),
    )
    with pytest.raises(ValueError, match="token ids"):
        _nodes(reader, hidden)
    nodes = reader.operation_candidates(
        hidden=hidden, input_spans=(), max_span_tokens=4,
        hidden_channels=CHANNELS, hidden_channel_widths=WIDTHS, token_ids=(1, 2, 3),
    )
    assert nodes[0].operation == "sub"
    replay = peak_operation_recognizer_from_dict(reader.to_dict())
    assert replay.word_continuations == frozenset({99})
    assert replay.identity_sha256 == reader.identity_sha256


def test_the_words_name_the_operation_and_leave_whether_to_the_context() -> None:
    """In "Use integer arithmetic" the word alone reads integer division and the
    sentence reads an adjective. The words may change which name a span gets,
    never how sure the span is of naming an operation at all."""
    contextual = [BACKGROUND, (0.0, 1.0, 0.1), BACKGROUND]
    hidden = _hidden(contextual, [BACKGROUND, (0.0, 1.0, -1.0), BACKGROUND])
    plain = _recognizer(lexical_labeler=_head(("add", "sub"), [[0, 0, 4], [0, 0, -4]]), label_weights=(1.0, 2.0))
    named = replace(plain, words_name_only=True)
    context_only = _recognizer()

    def first(recognizer):
        return next(node for node in _nodes(recognizer, hidden) if node.span == TokenSpan(1, 2))

    assert first(context_only).operation == "add"
    assert first(plain).operation == first(named).operation == "sub"
    assert first(plain).confidence > first(context_only).confidence
    assert first(named).confidence == pytest.approx(first(context_only).confidence)
    replay = peak_operation_recognizer_from_dict(named.to_dict())
    assert replay.words_name_only and replay.identity_sha256 != plain.identity_sha256
