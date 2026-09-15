"""Complete graph gradients follow the actual multiview and relation scores."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_operation_graph_learning import OperationEvidenceBank
from core.learning.semantic_relation_graph_learning import (
    fit_joint_graph_contrasts, RelationGraphContrast, relation_graph_loss,
)
from core.learning.semantic_program_transducer import LinearClassifierHead, MultiViewClassifierHead
from tests.test_semantic_relation_graph_learning import fixture, model_examples


def operation_fixture():
    rng = np.random.default_rng(19)
    features = (rng.normal(size=3), rng.normal(size=2))
    head = MultiViewClassifierHead(('span_mean', 'lexical_mean'), tuple(
        LinearClassifierHead(('add', 'sub', 'mul'), rng.normal(size=(3, len(value))), rng.normal(size=3))
        for value in features))
    parameters = tuple(value.astype(np.float64) for part in head.heads for value in (part.weight, part.bias))
    return head, OperationEvidenceBank(features), parameters


def test_operation_score_is_log_of_the_runtime_probability_mixture():
    head, bank, parameters = operation_fixture()
    for selected in range(len(head.labels)):
        score, _ = bank.score_gradient(selected, parameters)
        assert score == pytest.approx(np.log(head.predict_probabilities(bank.features)[selected]), abs=2e-7)


def test_joint_gradient_matches_central_differences_for_all_heads():
    query, definition, relation = fixture()
    _, bank, operations = operation_fixture()
    row = replace(relation, positive_operations=((bank, 1),), negative_operations=((bank, 0),))
    initial = (query + .1, definition - .2, *(value + .05 for value in operations))
    parameters = (query, definition, *operations)
    def objective(values):
        return relation_graph_loss(*values[:2], (row,), scale=1.3, initial=initial,
                                   regularization=.02, operation_parameters=values[2:])
    _, derivatives = objective(parameters)
    for which, value in enumerate(parameters):
        for index in np.ndindex(value.shape):
            plus, minus = [v.copy() for v in parameters], [v.copy() for v in parameters]
            plus[which][index] += 1e-5
            minus[which][index] -= 1e-5
            numeric = (objective(plus)[0] - objective(minus)[0]) / 2e-5
            assert derivatives[which][index] == pytest.approx(numeric, abs=1e-7)


def test_probability_floor_has_zero_operation_gradient():
    bank = OperationEvidenceBank((np.array([1.]),))
    score, gradients = bank.score_gradient(0, (np.array([[-1000.], [1000.]]), np.zeros(2)))
    assert score == pytest.approx(np.log(1e-12))
    assert all(np.array_equal(value, np.zeros_like(value)) for value in gradients)


def test_joint_update_changes_operation_tissue_without_changing_base_relation():
    from core.learning.semantic_relation_tissue import DirectionalRelationHead
    query, definition, relation = fixture()
    head = DirectionalRelationHead(np.zeros(9), .2, .3, query, definition)
    operation, bank, _ = operation_fixture()
    row = replace(relation, positive_operations=((bank, 1),), negative_operations=((bank, 0),))
    fitted_relation, fitted_operation, receipt = fit_joint_graph_contrasts(head, operation, (row,),
                                                                         steps=40, learning_rate=.01)
    assert receipt['stored_loss'] < receipt['initial_loss']
    assert receipt['operation_head_updated']
    assert np.array_equal(fitted_relation.weight, head.weight)
    assert fitted_operation.modes == operation.modes and fitted_operation.labels == operation.labels
    assert not np.array_equal(fitted_operation.heads[0].weight, operation.heads[0].weight)


def test_rebuilt_source_ordered_graph_replays_the_actual_decoder_score():
    from core.learning.semantic_joint_graph_learning import score_annotated_graph
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    measured = 0
    for item in examples:
        outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
            public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=model.model_basis_sha256)
        if outcome.ir is None:
            continue
        graph = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans)
        assert graph is not None
        assert graph['argument_score'] == pytest.approx(outcome.pointer_scores['argument_graph_total'], abs=1e-4)
        measured += 1
    assert measured > 0


def test_runtime_negative_mining_uses_no_test_examples_and_roundtrips():
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    candidate = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2)
    replay = refit_compositional_joint_graphs(model, tuple(x for x in examples if x.split != 'test'), rounds=1, steps=2)
    receipt = candidate.training_receipt['joint_graph_refit']
    assert receipt['training_examples'] == sum(x.split == 'train' for x in examples)
    assert receipt['test_examples_used'] == 0 and not receipt['validation_used_for_fit']
    assert not receipt['serving_authority']
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


def test_missing_operation_parameters_cannot_silently_ignore_operation_supervision():
    query, definition, _ = fixture()
    _, bank, _ = operation_fixture()
    row = RelationGraphContrast((), (), 0., positive_operations=((bank, 0),))
    with pytest.raises(ValueError, match='views differ'):
        relation_graph_loss(query, definition, (row,), scale=1., initial=(query, definition), regularization=0.)


def test_actual_runtime_operation_errors_drive_joint_training():
    from core.learning.semantic_joint_graph_learning import mine_runtime_graph_contrast, refit_compositional_joint_graphs
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    operation = replace(model.operation_head, heads=tuple(replace(head,
        weight=np.zeros_like(head.weight), bias=np.arange(len(head.labels), dtype=float) * 5)
        for head in model.operation_head.heads))
    model = model._with_coefficients(operation_head=operation)
    item = next(row for row in examples if row.split == 'train')
    contrast, record = mine_runtime_graph_contrast(model, item)
    assert contrast is not None and record['status'] == 'counterexample'
    assert record['comparison']['status'] == 'different'
    assert contrast.positive_operations and contrast.negative_operations
    candidate = refit_compositional_joint_graphs(model, examples, rounds=2, steps=3)
    rounds = candidate.training_receipt['joint_graph_refit']['rounds']
    assert rounds[0]['fit']['operation_head_updated']
    assert rounds[0]['fit']['pairs'] > 0
    assert rounds[1]['fit']['pairs'] >= rounds[0]['fit']['pairs']
    assert rounds[0]['fit']['stored_loss'] < rounds[0]['fit']['initial_loss']
    assert not np.array_equal(candidate.operation_head.heads[0].weight, operation.heads[0].weight)


@pytest.mark.parametrize('feature', [(), (np.array([]),), (np.array([float('nan')]),), (np.eye(2),)])
def test_invalid_operation_evidence_is_rejected(feature):
    with pytest.raises(ValueError):
        OperationEvidenceBank(feature)
