"""Parsed literals remain whole atoms during learned reference assignment."""

import json
from pathlib import Path

import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    _argument_literal_boundaries,
    _argument_span_respects_literals,
)


@pytest.mark.parametrize("bounds,valid", [
    ((2, 6), True), ((1, 7), True), ((0, 2), True), ((6, 8), True),
    ((2, 5), False), ((3, 6), False), ((3, 4), False),
    ((0, 4), False), ((4, 8), False),
])
def test_partial_literal_cannot_be_an_unrelated_register_mention(bounds, valid):
    assert _argument_span_respects_literals(TokenSpan(*bounds), (TokenSpan(2, 6),)) is valid


def test_wrapped_expression_may_contain_whole_literals_but_not_fragments():
    literals = (TokenSpan(2, 5), TokenSpan(8, 11))
    assert _argument_span_respects_literals(TokenSpan(1, 12), literals)
    assert not _argument_span_respects_literals(TokenSpan(1, 10), literals)
    assert _argument_span_respects_literals(TokenSpan(5, 8), literals)
    assert _argument_span_respects_literals(TokenSpan(0, 1), ())


def test_boundary_policy_is_bound_and_does_not_change_learned_coefficients():
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_atomic_literal_arguments()
    assert parent._coefficient_body() == candidate._coefficient_body()
    assert "argument_literal_boundaries" not in parent.training_receipt
    assert candidate.training_receipt["argument_literal_boundaries"] == "atomic_v1"
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


def test_registered_boundary_invariant():
    assert _argument_literal_boundaries()
