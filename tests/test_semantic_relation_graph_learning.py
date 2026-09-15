"""Graph supervision differentiates the same normalized evidence used in decode."""

import numpy as np
import pytest

from core.learning.semantic_relation_graph_learning import (
    RelationEvidenceBank, RelationGraphContrast, fit_relation_graph_contrasts, relation_graph_loss,
)
from core.learning.semantic_relation_tissue import DirectionalRelationHead
from core.learning.semantic_program_transducer_fitting import _mention_invariant_relation_evidence


def fixture():
    rng = np.random.default_rng(27)
    query, definition = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
    banks = [RelationEvidenceBank(rng.normal(size=3), rng.normal(size=(4, 3)), rng.normal(size=4))
             for _ in range(3)]
    row = RelationGraphContrast(((banks[0], 1), (banks[1], 0)), ((banks[2], 3),), .4)
    return query, definition, row


def test_relation_bank_replays_runtime_normalized_evidence():
    query, definition, row = fixture()
    for bank, index in (*row.positive, *row.negative):
        logits = bank.base_logits + (bank.definitions @ definition) @ (bank.reference @ query)
        expected = _mention_invariant_relation_evidence(tuple(bank.base_logits), tuple(logits),
                                                        strategy="categorical_log_margin_v1")
        assert bank.score_gradient(index, query, definition)[0] == pytest.approx(expected[index], abs=1e-12)


def test_complete_graph_gradient_includes_moving_normalizers():
    query, definition, row = fixture()
    initial = (query + .1, definition - .2)
    kwargs = dict(scale=1.7, initial=initial, regularization=.02)
    _, gradients = relation_graph_loss(query, definition, (row,), **kwargs)
    for matrix_index, matrix in enumerate((query, definition)):
        for index in np.ndindex(matrix.shape):
            plus, minus = [query.copy(), definition.copy()], [query.copy(), definition.copy()]
            plus[matrix_index][index] += 1e-5
            minus[matrix_index][index] -= 1e-5
            numeric = (relation_graph_loss(*plus, (row,), **kwargs)[0]
                       - relation_graph_loss(*minus, (row,), **kwargs)[0]) / 2e-5
            assert gradients[matrix_index][index] == pytest.approx(numeric, abs=1e-7)


def test_graph_update_changes_existing_tissue_and_preserves_base_evidence():
    query, definition, row = fixture()
    head = DirectionalRelationHead(np.zeros(9), .2, .3, query, definition)
    candidate, receipt = fit_relation_graph_contrasts(head, (row,), steps=80, learning_rate=.01)
    assert receipt['stored_loss'] < receipt['initial_loss']
    assert np.array_equal(candidate.weight, head.weight)
    assert candidate.bias == head.bias and candidate.pointer_scale == head.pointer_scale
    assert not np.array_equal(candidate.query_projection, head.query_projection)
    assert not receipt['serving_authority']


@pytest.mark.parametrize('value', [0, -1, float('nan')])
def test_invalid_contrast_weights_do_not_produce_candidates(value):
    query, definition, row = fixture()
    with pytest.raises(ValueError):
        relation_graph_loss(query, definition, (RelationGraphContrast(row.positive, row.negative, 0., value),),
                            scale=1., initial=(query, definition), regularization=.1)


def model_examples():
    from core.learning.semantic_program_compositional_transducer import fit_compositional_semantic_program_transducer
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    return model.with_global_constraint_arguments().with_joint_definition_graph().with_categorical_relation_scores(), examples


def test_real_decoder_retains_the_selected_relation_normalizers():
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments, _OperationNode

    model, examples = model_examples()
    item = examples[0]
    charts = []
    _assign_typed_arguments(model=model, hidden=item.hidden_states, inputs=item.public_inputs,
        input_spans=item.ir.input_spans,
        operation_nodes=tuple(_OperationNode(i.operation_span, i.op, 0., 0., 1.) for i in item.ir.instructions),
        argument_pointer_scores=model.argument_pointer.score_sequence(item.hidden_states),
        chart_observer=charts.append, retain_score_factors=True, retain_relation_evidence=True, build_only=True)
    selected = []
    result = charts[0].solve_with_factors(relation_observer=selected.append)
    head = model.definition_relation_head
    score = sum(bank.score_gradient(index, head.query_projection.astype(np.float64),
                                    head.definition_projection.astype(np.float64))[0]
                for bank, index in selected[0])
    assert score == pytest.approx(result[1][2], abs=1e-4)
    target = charts[0].restrict_arguments(result[0][1])
    assert target.option_relation_evidence is not None
    captured = []
    assert target.solve_with_factors(relation_observer=captured.append)[0] == result[0]
    assert len(captured[0]) == len(selected[0])


def test_source_graph_relation_refit_replays_and_keeps_test_data_out():
    from core.learning.semantic_relation_graph_learning import refit_compositional_graph_relations
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, examples = model_examples()
    candidate = refit_compositional_graph_relations(model, examples, rounds=1, steps=2, max_graphs=4)
    report = candidate.training_receipt['argument_graph_relation_refit']
    assert report['test_examples_used'] == 0 and report['validation_used_for_fit'] is False
    assert report['training_examples'] == sum(x.split == 'train' for x in examples)
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
    replay = refit_compositional_graph_relations(model, tuple(x for x in examples if x.split != 'test'),
                                                rounds=1, steps=2, max_graphs=4)
    assert replay.receipt_sha256 == candidate.receipt_sha256
    for name, value in model._coefficient_body().items():
        if name != 'definition_relation_head':
            assert candidate._coefficient_body()[name] == value
