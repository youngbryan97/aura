"""An argument belongs to the operation it stands beside, and the score knows it."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.learning.semantic_argument_ownership import (
    ArgumentOwnership,
    argument_ownership_from_dict,
    fit_argument_ownership,
    ownership_features,
)
from core.learning.semantic_program_ir import TokenSpan


def _request(split: str, operations: list[tuple[int, int]], arguments: list[list[tuple[int, int]]]):
    instructions = tuple(
        SimpleNamespace(operation_span=TokenSpan(*operation), argument_spans=tuple(TokenSpan(*m) for m in mentions))
        for operation, mentions in zip(operations, arguments, strict=True)
    )
    return SimpleNamespace(split=split, ir=SimpleNamespace(instructions=instructions, source_text_sha256=f"{split}{operations}"))


#: "add A and B. Call that C. multiply C by D." Each operation's arguments follow it.
TRAINING = [
    _request("train", [(2, 3), (12, 13)], [[(3, 5), (6, 8)], [(13, 15), (16, 18)]]),
    _request("train", [(0, 1), (9, 10), (20, 21)], [[(1, 3), (4, 5)], [(10, 12), (13, 14)], [(21, 23), (24, 26)]]),
    _request("train", [(4, 5), (14, 15)], [[(5, 6), (7, 9)], [(15, 17), (18, 19)]]),
]


def test_the_fitter_sees_training_rows_only() -> None:
    with pytest.raises(ValueError, match="training rows only"):
        fit_argument_ownership([*TRAINING, _request("validation", [(0, 1), (5, 6)], [[(1, 2)], [(6, 7)]])])
    fitted = fit_argument_ownership(TRAINING)
    assert fitted.fit_receipt["splits_used"] == ["train"]
    assert argument_ownership_from_dict(fitted.to_dict()).identity_sha256 == fitted.identity_sha256


def test_a_mention_after_an_operation_belongs_to_it_and_not_to_the_one_before() -> None:
    fitted = fit_argument_ownership(TRAINING)
    operations = [TokenSpan(2, 3), TokenSpan(12, 13), TokenSpan(25, 26)]
    # "primary result" in the add clause, read from the multiply that follows.
    earlier_clause = fitted.log_probabilities(TokenSpan(5, 7), operations)
    own_clause = fitted.log_probabilities(TokenSpan(13, 15), operations)
    assert earlier_clause[0] > earlier_clause[1]
    assert own_clause[1] > own_clause[0] and own_clause[1] > own_clause[2]
    assert math.isclose(sum(math.exp(value) for value in own_clause), 1.0, rel_tol=1e-9)


def test_an_operation_owns_nothing_it_overlaps() -> None:
    fitted = fit_argument_ownership(TRAINING)
    scores = fitted.log_probabilities(TokenSpan(2, 4), [TokenSpan(2, 3), TokenSpan(12, 13)])
    assert scores[0] == -math.inf and scores[1] == 0.0


def test_features_count_the_operations_between() -> None:
    operations = [TokenSpan(0, 1), TokenSpan(5, 6), TokenSpan(10, 11)]
    far = ownership_features(TokenSpan(12, 13), operations[0], operations)
    near = ownership_features(TokenSpan(12, 13), operations[2], operations)
    assert far[5] == 1.0 and far[7] == 0.0
    assert near[3] == 1.0 and near[6] == 1.0


def test_decode_scores_arguments_with_ownership() -> None:
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
    ownership = fit_argument_ownership(training)
    candidate = PeakRecognitionTransducer(base, recognizer, ownership)
    assert candidate.receipt_sha256 != PeakRecognitionTransducer(base, recognizer).receipt_sha256
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
    # Ownership that sends every mention to the farthest operation moves the program.
    perverse = ArgumentOwnership(tuple(-50.0 * value for value in ownership.weight), 0.0, ownership.fit_receipt)
    moved = PeakRecognitionTransducer(base, recognizer, perverse).decode(**arguments)
    assert moved.ir is None or moved.ir.to_program() != item.ir.to_program()
