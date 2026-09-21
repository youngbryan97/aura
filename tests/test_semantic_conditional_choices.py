"""Whole-graph choices use the same categorical evidence during decode and fit."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_argument_graph_learning import ArgumentScoreTerm
from core.learning.semantic_graph_batch import GraphConstraintBatch
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract
from core.learning.semantic_relation_graph_learning import (
    GraphChoiceNormalizer, RelationGraphContrast, graph_margin, graph_margin_gradient,
)


def chart():
    return ScoredArgumentChart(
        ((((8., 0, TokenSpan(0, 1)), (7., 1, TokenSpan(2, 3))),
          ((9., 1, TokenSpan(2, 3)), (6., 0, TokenSpan(0, 1)))),),
        2, RegisterUseContract(1, 1, 0, 1, True))


def test_conditional_choice_cancels_arbitrary_slot_offsets():
    original = chart()
    shifted = replace(original, options=tuple(tuple(tuple((score + offset, reg, span) for score, reg, span in slot)
        for offset, slot in zip((1000., -200.), node, strict=True)) for node in original.options))
    assert shifted.solve()[0] != original.solve()[0]
    a, b = original.with_conditional_choices(), shifted.with_conditional_choices()
    assert a.solve()[0] == pytest.approx(b.solve()[0], abs=1e-12)
    assert a.solve()[1:] == b.solve()[1:] == original.solve()[1:]
    assert a.solve()[0] <= 0.
    assert a.score_upper_bound() >= a.solve()[0]


def test_target_restriction_and_exclusion_keep_original_denominator():
    normalized = chart().with_conditional_choices()
    target = normalized.restrict_arguments(((0, 1),))
    assert target.choice_log_normalizer == normalized.choice_log_normalizer
    assert target.solve() == normalized.solve()
    assert normalized.diagnose_target(((0, 1),))['target_score'] == pytest.approx(normalized.solve()[0])
    rival = normalized.solve(excluded_graphs=(((0, 1),),))
    assert rival[0] == pytest.approx(13. - normalized.choice_log_normalizer)


def test_choice_product_sums_to_one_before_global_constraints():
    import itertools

    normalized = chart().with_conditional_choices()
    mass = sum(math.exp(sum(choice[0] for choice in choices) - normalized.choice_log_normalizer)
               for choices in itertools.product(*normalized.options[0]))
    assert mass == pytest.approx(1.)


def evidence():
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.array([.3, -.2]), np.array(.4))
    terms = [ArgumentScoreTerm(2, value, 1., 'conditional_log_odds_v1')
             for value in (np.array([1., 2.]), np.array([-1., .4]), np.array([.1, -.3]))]
    choices = tuple(RelationGraphContrast((), (), float(index) / 10,
        argument_terms=((1., term),)) for index, term in enumerate(terms))
    normalizer = GraphChoiceNormalizer(choices)
    selected = replace(choices[0], normalizer_terms=((-1., normalizer),))
    return parameters, choices, normalizer, selected


def test_denominator_gradient_includes_competing_choices():
    parameters, choices, normalizer, selected = evidence()
    scores = np.array([graph_margin(parameters, row) for row in choices])
    assert normalizer.score(parameters) == pytest.approx(logsumexp(scores))
    value, gradient = graph_margin_gradient(parameters, selected)
    assert value == pytest.approx(scores[0] - logsumexp(scores))
    # A shared additive bias must cancel, including its derivative.
    assert abs(gradient[3]) < 1e-12
    for which, part in enumerate(parameters):
        for index in np.ndindex(part.shape):
            plus, minus = [x.copy() for x in parameters], [x.copy() for x in parameters]
            plus[which][index] += 1e-5
            minus[which][index] -= 1e-5
            numeric = (graph_margin(plus, selected) - graph_margin(minus, selected)) / 2e-5
            assert gradient[which][index] == pytest.approx(numeric, abs=1e-8)


def test_batched_conditional_margin_and_derivative_match_scalar():
    parameters, _, normalizer, selected = evidence()
    rows = (selected, replace(selected, fixed_margin=.7, normalizer_terms=((1., normalizer),)))
    batch = GraphConstraintBatch(rows)
    weights = np.array([.3, -.6])
    direction = tuple(np.full_like(x, .25) for x in parameters)
    for index, row in enumerate(rows):
        value, derivatives = graph_margin_gradient(parameters, row)
        assert batch.margins(parameters)[index] == pytest.approx(value)
        assert batch.directional_derivative(parameters, direction)[index] == pytest.approx(
            sum(np.sum(a * b) for a, b in zip(derivatives, direction, strict=True)))
    expected = [np.zeros_like(x) for x in parameters]
    for weight, row in zip(weights, rows, strict=True):
        for target, gradient in zip(expected, graph_margin_gradient(parameters, row)[1], strict=True):
            target += weight * gradient
    for a, b in zip(batch.weighted_gradient(parameters, weights), expected, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_denominator_batches_keep_independent_scales_and_pool_boundaries():
    from core.learning.semantic_relation_graph_learning import RelationEvidenceBank

    parameters, choices, normalizer, selected = evidence()
    relation = RelationEvidenceBank(np.ones(1), np.array([[1.], [-1.]]), np.array([.1, -.3]))
    alternatives = tuple(replace(row, positive=((relation, index % 2),)) for index, row in enumerate(choices))
    other = GraphChoiceNormalizer(alternatives, scale=2.)
    shared_scale = GraphChoiceNormalizer(choices[:2])
    rows = (selected, replace(selected, normalizer_terms=((1., other), (-1., shared_scale))),
            replace(selected, normalizer_terms=((1., normalizer), (-1., other))))
    batch = GraphConstraintBatch(rows)
    assert len(batch.choice_batches) == 2
    assert sorted(len(groups) for _, groups in batch.choice_batches) == [1, 2]
    coefficients = np.array([.4, -.3, .8])
    direction = tuple(np.full_like(part, .2) for part in parameters)
    expected = [graph_margin_gradient(parameters, row) for row in rows]
    np.testing.assert_allclose(batch.margins(parameters), [row[0] for row in expected], atol=1e-12)
    for index, actual in enumerate(batch.weighted_gradient(parameters, coefficients)):
        np.testing.assert_allclose(actual, sum(c * row[1][index]
            for c, row in zip(coefficients, expected, strict=True)), atol=1e-12)
    np.testing.assert_allclose(batch.directional_derivative(parameters, direction),
        [sum(np.sum(a * b) for a, b in zip(row[1], direction, strict=True)) for row in expected], atol=1e-12)


def test_choice_evidence_archive_roundtrips_shared_denominators(tmp_path):
    from core.learning.semantic_fit_problem import load_fit_problem, save_fit_problem
    from core.learning.semantic_fit_checkpoint import fit_identity

    parameters, _, normalizer, selected = evidence()
    normalizer.score(parameters)  # Cached batch is not serialized or hashed.
    rows = (selected, replace(selected, fixed_margin=.8))
    path = tmp_path / 'problem.npz'
    identity = fit_identity((parameters, rows))
    save_fit_problem(path, identity=identity, initial=parameters, contrasts=rows, options={})
    replay = load_fit_problem(path, expected_identity=identity)
    a, b = replay['contrasts']
    assert a.normalizer_terms[0][1] is b.normalizer_terms[0][1]
    assert fit_identity((replay['initial'], replay['contrasts'])) == identity
    np.testing.assert_allclose(GraphConstraintBatch(rows).margins(parameters),
        GraphConstraintBatch(replay['contrasts']).margins(replay['initial']))


@pytest.mark.parametrize('choices', [(), (RelationGraphContrast((), (), 0., normalizer_terms=((1., None),)),)])
def test_normalizer_rejects_empty_and_recursive_evidence(choices):
    with pytest.raises(ValueError, match='nonrecursive'):
        GraphChoiceNormalizer(choices)


def test_shared_offset_invariant():
    from core.learning.semantic_relation_graph_learning import _graph_choice_shared_offset_cancels

    assert _graph_choice_shared_offset_cancels() == ()


@pytest.fixture(scope='module')
def source():
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    return model.with_conditional_argument_choices(), examples


def test_runtime_policy_is_identity_bound_and_serializable(source):
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, _ = source
    restored = compositional_semantic_program_transducer_from_dict(model.to_dict())
    assert restored.receipt_sha256 == model.receipt_sha256
    assert restored.training_receipt['argument_choice_normalization'] == 'local_categorical_v1'


def test_register_alternatives_share_their_identical_argument_features(source):
    from core.learning.semantic_joint_graph_learning import score_annotated_graph

    model, examples = source
    item = next(row for row in examples if row.split == 'train')
    cache = {}
    a = score_annotated_graph(model, item, item.ir.instructions, item.ir.input_spans,
        learn_arguments=True, argument_evidence_cache=cache)
    b = score_annotated_graph(model, item, item.ir.instructions, item.ir.input_spans,
        learn_arguments=True, argument_evidence_cache=cache)
    assert a is not None and b is not None
    terms_a = [term for normalizer in a['normalizers'] for row in normalizer.choices
               for _sign, term in row.argument_terms]
    terms_b = [term for normalizer in b['normalizers'] for row in normalizer.choices
               for _sign, term in row.argument_terms]
    assert terms_a and len({id(term) for term in terms_a}) < len(terms_a)
    assert all(left is right for left, right in zip(terms_a, terms_b, strict=True))
    assert a['score'] == b['score']


def test_complete_source_chart_replays_the_normalized_runtime_score(source):
    from core.learning.semantic_joint_graph_learning import (
        align_source_input_registers, score_annotated_graph, joint_graph_contrast,
    )
    from core.learning.semantic_argument_graph_learning import argument_parameters
    from core.learning.semantic_operation_pointer_learning import operation_pointer_parameters

    model, examples = source
    item = next(x for x in examples if x.split == 'train')
    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    assert outcome.ir is not None
    graph = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans,
                                 learn_arguments=True, learn_operation_pointer=True)
    assert graph['normalizers']
    assert graph['argument_score'] == pytest.approx(outcome.pointer_scores['argument_graph_total'], abs=1e-4)
    target, _ = align_source_input_registers(item, outcome.ir.input_spans)
    positive = score_annotated_graph(model, item, target, outcome.ir.input_spans,
                                    learn_arguments=True, learn_operation_pointer=True)
    row = joint_graph_contrast(model, positive, graph)
    parameters = (model.definition_relation_head.query_projection.astype(float), model.definition_relation_head.definition_projection.astype(float),
        *(v.astype(float) for h in model.operation_head.heads for v in (h.weight, h.bias)),
        *argument_parameters(model), *operation_pointer_parameters(model))
    assert graph_margin(parameters, row, scale=model.definition_relation_scale) == pytest.approx(
        positive['score'] - graph['score'], abs=1e-7)
    assert row.normalizer_terms


def test_source_training_preserves_conditional_selection_and_does_not_fit_evaluation(source):
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, examples = source
    operation = replace(model.operation_head, heads=tuple(replace(head,
        weight=np.zeros_like(head.weight), bias=np.arange(len(head.labels), dtype=float) * 5.)
        for head in model.operation_head.heads))
    model = model._with_coefficients(operation_head=operation)
    progress = []
    candidate = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
        constraint_learning=True, learn_arguments=True, learn_operation_pointer=True,
        retention_operation_charts=4, progress=progress.append)
    receipt = candidate.training_receipt['joint_graph_refit']
    assert receipt['test_examples_used'] == 0 and not receipt['validation_used_for_fit']
    assert receipt['selection_policy'] == 'joint_factor_score_v2'
    mined = [row['row'] for row in progress if row['stage'] == 'joint_graph_mining']
    assert {row['source_text_sha256'] for row in mined} == {
        row.ir.source_text_sha256 for row in examples if row.split == 'train'}
    assert any(row['status'] == 'counterexample' for row in mined)
    fit = receipt['rounds'][0]['fit']
    assert fit['stored_loss'] < fit['initial_loss']
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.receipt_sha256 == candidate.receipt_sha256
    assert restored.training_receipt['argument_choice_normalization'] == 'local_categorical_v1'
