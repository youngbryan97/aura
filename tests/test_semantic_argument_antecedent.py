"""A named result is read back to the step that named it, not to an input."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_argument_antecedent import (
    ArgumentAntecedent,
    argument_antecedent_from_dict,
    fit_argument_antecedent,
    register_stretches,
)
from core.learning.semantic_program_ir import TokenSpan

CHANNELS = ("input_token_embedding", "middle_causal_hidden")
WIDTH = 24


def _vectors(words: list[str], seed: int) -> np.ndarray:
    """Each word one fixed direction; the middle layer adds a little context."""
    vocabulary = sorted(set(words))
    directions = np.random.default_rng(7).normal(size=(len(vocabulary), WIDTH))
    embedding = np.stack([directions[vocabulary.index(word)] for word in words])
    noise = np.random.default_rng(seed).normal(scale=0.2, size=embedding.shape)
    return np.concatenate([embedding, embedding + noise], axis=1)


def _request(split: str, names: tuple[str, str], seed: int):
    """'a = 1 ; b = 2 ; c = 3 ; add a and b . save as N0 . sub c from b . save as N1 . mul N0 by N1'."""
    first, second = names
    words = ("a = 1 ; b = 2 ; c = 3 ; add a and b . save as " + first + " . sub c from b . save as "
             + second + " . mul " + first + " by " + second).split()
    at = {index: word for index, word in enumerate(words)}

    def span(start: int) -> TokenSpan:
        return TokenSpan(start, start + 1)

    inputs = (span(2), span(6), span(10))
    add, sub, mul = (next(i for i, w in at.items() if w == op) for op in ("add", "sub", "mul"))
    instructions = (
        SimpleNamespace(operation_span=span(add), args=(0, 1), argument_spans=(span(add + 1), span(add + 3))),
        SimpleNamespace(operation_span=span(sub), args=(1, 2), argument_spans=(span(sub + 3), span(sub + 1))),
        SimpleNamespace(operation_span=span(mul), args=(3, 4), argument_spans=(span(mul + 1), span(mul + 3))),
    )
    ir = SimpleNamespace(input_spans=inputs, instructions=instructions, n_inputs=3,
                         source_text_sha256=f"{split}-{first}-{second}-{seed}")
    return SimpleNamespace(split=split, ir=ir, hidden_states=_vectors(words, seed),
                           hidden_channels=CHANNELS, hidden_channel_widths=(WIDTH, WIDTH))


TRAINING = [_request("train", names, seed) for seed, names in enumerate(
    [("lead", "side"), ("primary", "auxiliary"), ("first", "second"), ("upper", "lower")]
)]


def test_each_register_owns_its_declaration_or_its_clause() -> None:
    stretches = register_stretches(
        [TokenSpan(2, 3), TokenSpan(6, 7)], [TokenSpan(10, 11), TokenSpan(20, 21)], 30
    )
    assert stretches == ((0, 3), (3, 7), (10, 20), (20, 30))


def test_the_fitter_sees_training_rows_only() -> None:
    with pytest.raises(ValueError, match="training rows only"):
        fit_argument_antecedent([*TRAINING, _request("validation", ("x", "y"), 9)])
    fitted = fit_argument_antecedent(TRAINING)
    assert fitted.fit_receipt["splits_used"] == ["train"]
    assert argument_antecedent_from_dict(fitted.to_dict()).identity_sha256 == fitted.identity_sha256


def test_a_name_never_seen_in_training_is_read_back_to_the_step_that_gave_it() -> None:
    fitted = fit_argument_antecedent(TRAINING)
    item = _request("test", ("refined", "spare"), 11)
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    multiply = item.ir.instructions[2]
    for mention, named in zip(multiply.argument_spans, multiply.args, strict=True):
        scores = scorer.log_probabilities(mention)
        assert int(np.argmax(scores)) == named
        assert math.isclose(sum(math.exp(value) for value in scores), 1.0, rel_tol=1e-9)
    # An input's name read back to its declaration.
    subtract = item.ir.instructions[1]
    assert int(np.argmax(scorer.log_probabilities(subtract.argument_spans[0]))) == 1


def test_a_mention_with_no_earlier_window_spreads_its_probability() -> None:
    fitted = fit_argument_antecedent(TRAINING)
    item = TRAINING[0]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    first_token = scorer.log_probabilities(TokenSpan(0, 1))
    assert max(first_token) - min(first_token) < 1.0


def test_decode_scores_arguments_with_antecedents() -> None:
    from core.learning.semantic_argument_ownership import fit_argument_ownership
    from core.learning.semantic_operation_peaks import (
        PeakRecognitionTransducer,
        fit_peak_operation_recognizer,
        peak_recognition_transducer_from_dict,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    training = tuple(item for item in examples if item.split == "train")
    base = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    recognizer = fit_peak_operation_recognizer(training)
    # In this fixture every mention follows every operation, so which operation
    # owns a mention is the ownership readout's to decide; antecedents ground it.
    ownership = fit_argument_ownership(training)
    antecedent = fit_argument_antecedent(training)
    candidate = PeakRecognitionTransducer(base, recognizer, ownership, antecedent)
    assert candidate.receipt_sha256 != PeakRecognitionTransducer(base, recognizer, ownership).receipt_sha256
    restored = peak_recognition_transducer_from_dict(
        candidate.to_dict(), restore_base=compositional_semantic_program_transducer_from_dict
    )
    assert restored.receipt_sha256 == candidate.receipt_sha256
    item = next(item for item in examples if item.split == "test" and len(item.ir.instructions) == 3)
    arguments = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                     public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                     model_basis_sha256=base.model_basis_sha256)
    decoded = candidate.decode(**arguments)
    assert decoded.ir is not None and decoded.ir.to_program() == item.ir.to_program()
    # A readout that sends every mention away from where its name was given moves the program.
    perverse = ArgumentAntecedent(tuple(-50.0 * value for value in antecedent.weight), 0.0, antecedent.fit_receipt)
    moved = PeakRecognitionTransducer(base, recognizer, ownership, perverse).decode(**arguments)
    assert moved.ir is None or moved.ir.to_program() != item.ir.to_program()


def test_an_inputs_literal_value_gets_no_antecedent_evidence() -> None:
    """The literal grammar binds it exactly; the readout is neither fitted on nor applied to it."""
    fitted = fit_argument_antecedent(TRAINING)
    item = TRAINING[1]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    assert set(scorer.log_probabilities(item.ir.input_spans[1])) == {0.0}
