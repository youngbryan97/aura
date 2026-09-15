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


def test_decode_local_pair_cache_reuses_overlapping_chart_work(monkeypatch):
    head = DirectionalRelationHead(
        np.ones(6, dtype=np.float32), 0.0, .2,
        np.ones((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32),
    )
    a, b = TokenSpan(0, 1), TokenSpan(1, 2)
    references = {a: np.array([1, 2], dtype=np.float32)}
    definitions = (((b, np.array([2, 3], dtype=np.float32)),),)
    pointer = LinearPointerSequenceScores(np.zeros(3), np.zeros(3))
    expected = _definition_relation_score_banks(head, references, definitions, pointer)
    original = DirectionalRelationHead.base_score
    calls = []
    def counted(self, reference, definition):
        calls.append(1)
        return original(self, reference, definition)
    monkeypatch.setattr(DirectionalRelationHead, 'base_score', counted)
    cache = {}
    assert _definition_relation_score_banks(head, references, definitions, pointer, cache) == expected
    assert _definition_relation_score_banks(head, references, definitions * 2, pointer, cache) == (
        {a: expected[0][a] * 2}, {a: expected[1][a] * 2},
    )
    assert len(calls) == 1


def test_chart_feature_reuse_preserves_assignment_and_avoids_repeated_span_work(monkeypatch):
    from core.learning import semantic_program_transducer_fitting as fitting
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    item = examples[0]
    kwargs = dict(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
        input_spans=item.ir.input_spans,
        operation_nodes=tuple(fitting._OperationNode(ins.operation_span, ins.op, 0., 0., 1.)
                              for ins in item.ir.instructions),
        argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states))
    expected = fitting._assign_typed_arguments(**kwargs)
    assert expected is not None
    original, calls = fitting._relation_span_vector, []
    def counted(*args, **options):
        calls.append(args[1])
        return original(*args, **options)
    monkeypatch.setattr(fitting, '_relation_span_vector', counted)
    cache = {}
    options = dict(relation_vector_cache=cache,
                   definition_pointer_scores=model.definition_pointer.score_sequence(item.hidden_states))
    assert fitting._assign_typed_arguments(**kwargs, **options) == expected
    count = len(calls)
    assert count == len(cache) and count > 0
    assert fitting._assign_typed_arguments(**kwargs, **options) == expected
    assert len(calls) == count


def test_runtime_decode_owns_fresh_feature_caches_and_matches_uncached_selection(monkeypatch):
    from core.learning import semantic_program_compositional_transducer as transducer
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    item = examples[0]
    kwargs = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                  public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                  model_basis_sha256=model.model_basis_sha256)
    original, caches = transducer._assign_typed_arguments, []
    def observed(**options):
        caches.append(options['relation_vector_cache'])
        return original(**options)
    monkeypatch.setattr(transducer, '_assign_typed_arguments', observed)
    cached = model.decode(**kwargs)
    first_caches = tuple(caches)
    assert first_caches and all(value is first_caches[0] for value in first_caches)
    caches.clear()
    assert model.decode(**kwargs) == cached
    assert caches and caches[0] is not first_caches[0]
    def uncached(**options):
        options.pop('relation_vector_cache')
        options.pop('definition_pointer_scores')
        return original(**options)
    monkeypatch.setattr(transducer, '_assign_typed_arguments', uncached)
    assert model.decode(**kwargs) == cached
