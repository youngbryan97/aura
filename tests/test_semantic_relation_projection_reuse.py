"""Per-chart projection reuse must preserve the frozen scalar scorer exactly."""

import numpy as np
import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import LinearPointerSequenceScores
from core.learning.semantic_program_transducer_fitting import (
    DirectionalRelationHead,
    _definition_relation_score_banks,
)


@pytest.mark.parametrize("seed,width,rank", [(0, 8, 2), (1, 128, 16), (2, 64, 1)])
@pytest.mark.parametrize("pointer_scale", [0.0, 0.25, 2.0])
def test_cached_scores_are_bitwise_equal_to_original_scalar_arithmetic(seed, width, rank, pointer_scale):
    rng = np.random.default_rng(seed)

    def array(shape):
        return rng.normal(size=shape).astype(np.float32)

    head = DirectionalRelationHead(
        array(3 * width), 0.17, pointer_scale, array((width, rank)), array((width, rank))
    )
    references = {TokenSpan(i, i + 1): array(width) for i in range(6)}
    definitions = tuple(
        tuple((TokenSpan(j, j + 1), array(width)) for j in range(i + 1))
        for i in range(4)
    )
    pointer = LinearPointerSequenceScores(array(12), array(12))
    expected_combined = {
        span: tuple(max(
            head.score(reference, definition) + head.pointer_scale * pointer.score_span(candidate)
            for candidate, definition in candidates
        ) for candidates in definitions)
        for span, reference in references.items()
    }
    expected_base = {
        span: tuple(max(
            head.base_score(reference, definition) + head.pointer_scale * pointer.score_span(candidate)
            for candidate, definition in candidates
        ) for candidates in definitions)
        for span, reference in references.items()
    }
    actual = _definition_relation_score_banks(head, references, definitions, pointer)
    assert actual == (expected_combined, expected_base)


def test_projection_cache_is_not_reused_across_changed_inputs():
    head = DirectionalRelationHead(
        np.ones(6, dtype=np.float32), 0.0, 0.0,
        np.ones((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32),
    )
    span = TokenSpan(0, 1)
    references = {span: np.array([1, 2], dtype=np.float32)}
    definitions = (((span, np.array([2, 3], dtype=np.float32)),),)
    pointer = LinearPointerSequenceScores(np.zeros(2), np.zeros(2))
    first = _definition_relation_score_banks(head, references, definitions, pointer)
    references[span][0] = 7
    second = _definition_relation_score_banks(head, references, definitions, pointer)
    assert first != second
