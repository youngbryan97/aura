"""Input definition attachment is independent of register enumeration."""

import itertools
import json
from pathlib import Path

import numpy as np

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerSequenceScores
from core.learning.semantic_program_transducer_fitting import (
    _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
    _overlap,
    _register_definition_candidates,
)


def banks(anchors, scores, *, enabled=True):
    return _register_definition_candidates(
        anchors,
        input_count=3,
        token_count=32,
        max_span_tokens=12,
        pointer_scores=scores,
        strategy=_LOCAL_DEFINITION_CANDIDATE_STRATEGY,
        source_ordered_boundaries=enabled,
        bidirectional_inputs=enabled,
    )


def test_names_on_either_side_belong_to_their_source_neighbor():
    anchors = (TokenSpan(11, 13), TokenSpan(15, 17), TokenSpan(2, 3), TokenSpan(9, 10))
    start, end = np.zeros(32), np.zeros(32)
    start[5], end[6] = 20.0, 20.0
    scores = LinearPointerSequenceScores(start, end)
    observed = banks(anchors, scores)
    trailing_name = TokenSpan(5, 7)
    assert trailing_name in observed[2]
    assert all(trailing_name not in observed[i] for i in (0, 1))
    assert trailing_name not in banks(anchors, scores, enabled=False)[2]
    start[0], end[0] = 30.0, 30.0
    leading = banks(anchors, LinearPointerSequenceScores(start, end))
    assert TokenSpan(0, 1) in leading[2]


def test_permuting_registers_does_not_change_source_candidate_ownership():
    inputs = (TokenSpan(11, 13), TokenSpan(15, 17), TokenSpan(2, 3))
    operation = TokenSpan(9, 10)
    rng = np.random.default_rng(431)
    scores = LinearPointerSequenceScores(rng.normal(size=32), rng.normal(size=32))
    expected = dict(zip(inputs, banks((*inputs, operation), scores)[:3], strict=True))
    for permutation in itertools.permutations(inputs):
        observed = banks((*permutation, operation), scores)
        assert dict(zip(permutation, observed[:3], strict=True)) == expected
        for anchor, candidates in zip(permutation, observed[:3], strict=True):
            assert anchor in candidates
            for span in candidates:
                assert 0 <= span.start < span.end <= 32
                assert span.end - span.start <= 12
                assert not any(
                    _overlap(span, other) for other in (*inputs, operation) if other != anchor
                )


def test_touching_anchors_retain_the_literal_without_an_alias_gap():
    anchors = (TokenSpan(0, 1), TokenSpan(1, 2), TokenSpan(2, 3), TokenSpan(3, 4))
    scores = LinearPointerSequenceScores(np.zeros(32), np.zeros(32))
    assert banks(anchors, scores)[:3] == tuple((anchor,) for anchor in anchors[:3])


def test_candidate_roundtrip_preserves_the_separate_boundary_policy():
    path = (
        Path(__file__).parents[1]
        / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    )
    parent = compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))
    candidate = parent.with_bidirectional_input_definitions()
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert "definition_boundary_policy" not in parent.training_receipt
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.receipt_sha256 == candidate.receipt_sha256
    assert restored.training_receipt["definition_boundary_policy"] == "source_neighborhood_v2"
