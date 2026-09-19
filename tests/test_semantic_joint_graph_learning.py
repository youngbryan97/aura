"""Complete graph gradients follow the actual multiview and relation scores."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_operation_graph_learning import OperationEvidenceBank, OperationSourceSupervision
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
    source = OperationSourceSupervision(tuple(value[None, :] for value in bank.features),
                                       np.array([2]), np.array([1.]))
    def objective(values):
        return relation_graph_loss(*values[:2], (row,), scale=1.3, initial=initial,
                                   regularization=.02, operation_parameters=values[2:],
                                   source_supervision=source, source_weight=.7)
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
    assert receipt['source_operation_weight'] == 1.
    assert receipt['source_operations'] == sum(len(x.ir.instructions) for x in examples if x.split == 'train')
    assert not receipt['serving_authority']
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256


def test_missing_operation_parameters_cannot_silently_ignore_operation_supervision():
    query, definition, _ = fixture()
    _, bank, _ = operation_fixture()
    row = RelationGraphContrast((), (), 0., positive_operations=((bank, 0),))
    with pytest.raises(ValueError, match='views differ'):
        relation_graph_loss(query, definition, (row,), scale=1., initial=(query, definition), regularization=0.)


def test_unchanged_coefficients_stop_repeated_mining_without_claiming_convergence(monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning

    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    mined = []

    def mine(candidate, item, **kwargs):
        mined.append(item.ir.source_text_sha256)
        return object(), {"status": "counterexample"}

    def unchanged(relation, operation, contrasts, **kwargs):
        return relation, operation, {"status": "no_retention_preserving_step_found"}

    monkeypatch.setattr(learning, "mine_runtime_graph_contrast", mine)
    monkeypatch.setattr(learning, "fit_joint_graph_contrasts", unchanged)
    candidate = learning.refit_compositional_joint_graphs(model, examples, rounds=3)
    receipt = candidate.training_receipt["joint_graph_refit"]
    assert len(mined) == receipt["training_examples"]
    assert receipt["requested_rounds"] == 3
    assert receipt["completed_rounds"] == 1
    assert receipt["stop_reason"] == "coefficients_unchanged"
    assert receipt["rounds"][0]["records"][0]["status"] == "counterexample"
    assert not receipt["serving_authority"]


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


def test_batched_source_loss_matches_individual_runtime_scores_and_gradients():
    _, bank, parameters = operation_fixture()
    second = OperationEvidenceBank(tuple(value * .6 for value in bank.features))
    source = OperationSourceSupervision(tuple(np.stack((a, b)) for a, b in
        zip(bank.features, second.features, strict=True)), np.array([0, 2]), np.array([1., 3.]))
    loss, gradients = source.loss_gradient(parameters)
    first_score, first_grad = bank.score_gradient(0, parameters)
    second_score, second_grad = second.score_gradient(2, parameters)
    assert loss == pytest.approx(-.25 * first_score - .75 * second_score)
    for actual, a, b in zip(gradients, first_grad, second_grad, strict=True):
        np.testing.assert_allclose(actual, -.25 * a - .75 * b, atol=1e-12)


def test_source_retention_refuses_validation_and_test_rows():
    from core.learning.semantic_joint_graph_learning import source_operation_supervision
    model, examples = model_examples()
    training = tuple(row for row in examples if row.split == 'train')
    source = source_operation_supervision(model, training)
    assert len(source.labels) == sum(len(row.ir.instructions) for row in training)
    for split in ('validation', 'test'):
        with pytest.raises(ValueError, match='source training'):
            source_operation_supervision(model, (replace(training[0], split=split),))


def test_source_retention_changes_update_even_for_an_operation_absent_from_errors():
    from core.learning.semantic_relation_tissue import DirectionalRelationHead
    query, definition, relation = fixture()
    head = DirectionalRelationHead(np.zeros(9), .2, .3, query, definition)
    operation, bank, parameters = operation_fixture()
    source = OperationSourceSupervision(tuple(value[None, :] for value in bank.features),
                                       np.array([2]), np.array([1.]))
    # This graph error contains only relation evidence; source operations still learn.
    _, fitted, receipt = fit_joint_graph_contrasts(head, operation, (relation,), steps=30,
        learning_rate=.01, source_supervision=source, source_weight=1.)
    after = tuple(value.astype(np.float64) for part in fitted.heads for value in (part.weight, part.bias))
    assert source.loss_gradient(after)[0] < source.loss_gradient(parameters)[0]
    assert receipt['source_operations'] == 1


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf'), 1.])
def test_invalid_or_missing_source_retention_fails(weight):
    query, definition, row = fixture()
    with pytest.raises(ValueError, match='source retention'):
        relation_graph_loss(query, definition, (row,), scale=1., initial=(query, definition),
                            regularization=0., source_weight=weight)
