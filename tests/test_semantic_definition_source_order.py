"""Source clause boundaries are independent of input and instruction numbering."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerSequenceScores
from core.learning.semantic_program_transducer_fitting import (
    _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
    _STABLE_REGISTER_TABLE_STRATEGY,
    _register_definition_candidates,
)


@pytest.mark.parametrize("strategy", [
    _LOCAL_DEFINITION_CANDIDATE_STRATEGY, _STABLE_REGISTER_TABLE_STRATEGY,
])
@pytest.mark.parametrize("seed", range(3))
def test_definition_banks_are_equivariant_to_register_renumbering(strategy, seed):
    rng = np.random.default_rng(seed)
    scores = LinearPointerSequenceScores(rng.normal(size=80), rng.normal(size=80))
    inputs = (TokenSpan(50, 53), TokenSpan(6, 7), TokenSpan(70, 72))
    operations = (TokenSpan(62, 64), TokenSpan(30, 32))

    def banks(anchors):
        return dict(zip(anchors, _register_definition_candidates(
            anchors, input_count=3, token_count=80, max_span_tokens=15,
            pointer_scores=scores, strategy=strategy, source_ordered_boundaries=True,
        ), strict=True))

    expected = banks((*inputs, *operations))
    for input_order in itertools.permutations(inputs):
        for execution_order in itertools.permutations(operations):
            assert banks((*input_order, *execution_order)) == expected


def test_public_definition_search_does_not_select_a_preceding_operation():
    start = np.zeros(80)
    end = np.zeros(80)
    start[30], end[31] = 100., 100.
    literal, operation = TokenSpan(35, 55), TokenSpan(30, 32)
    banks = _register_definition_candidates(
        (literal, operation), input_count=1, token_count=80, max_span_tokens=24,
        pointer_scores=LinearPointerSequenceScores(start, end),
        strategy=_LOCAL_DEFINITION_CANDIDATE_STRATEGY, source_ordered_boundaries=True,
    )
    assert all(span.start >= operation.end for span in banks[0])
    assert literal in banks[0]


def test_boundary_change_keeps_the_frozen_parent_and_coefficients_intact():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_source_ordered_definitions()
    assert "definition_boundary_policy" not in parent.training_receipt
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()
