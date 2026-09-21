"""Recognition learning must include the binding evidence that can overrule it."""

import numpy as np
import pytest

from core.learning.semantic_labeled_span_learning import (
    _labeled_graph_loss,
    refit_compositional_labeled_spans,
)
from core.learning.semantic_operation_graph_learning import OperationEvidenceBank
from core.learning.semantic_relation_graph_learning import RelationGraphContrast, graph_margin


def _problem():
    positive = OperationEvidenceBank((np.array([1., 0.]),), 2)
    negative = OperationEvidenceBank((np.array([0., 1.]),), 2)
    row = RelationGraphContrast((), (), -5., positive_operations=((positive, 0),),
                               negative_operations=((negative, 0),))
    options = dict(labels=2, width=2, relation_parameters=(np.zeros((1, 1)), np.zeros((1, 1))), scale=1.)
    return row, options


def test_correct_recognition_is_not_enough_when_binding_outvotes_it():
    row, options = _problem()
    value = np.array([2., 0., 0., 0., 0., 0.])
    loss, gradient, margins = _labeled_graph_loss(value, (row,), **options)
    assert margins == pytest.approx([-3.])
    assert loss == pytest.approx(3.1 ** 2)
    assert gradient[0] < 0 < gradient[1]
    updated = value - .5 * gradient
    assert _labeled_graph_loss(updated, (row,), **options)[0] < loss


@pytest.mark.parametrize("magnitude", [1., 80.])
def test_runtime_margin_loss_gradient_includes_background_clipping(magnitude):
    row, options = _problem()
    value = np.random.default_rng(13).normal(size=6) * magnitude
    _, gradient, margins = _labeled_graph_loss(value, (row,), **options)
    parameters = (*options['relation_parameters'], np.vstack((value[:4].reshape(2, 2), np.zeros(2))),
                  np.append(value[4:], 0.))
    assert margins == pytest.approx([graph_margin(parameters, row)])
    for index in range(len(value)):
        plus, minus = value.copy(), value.copy()
        plus[index] += 1e-4
        minus[index] -= 1e-4
        difference = (_labeled_graph_loss(plus, (row,), **options)[0]
                      - _labeled_graph_loss(minus, (row,), **options)[0]) / 2e-4
        assert difference == pytest.approx(gradient[index], abs=2e-5)


def test_a_binding_only_error_has_no_fake_recognition_gradient():
    row, options = _problem()
    same = RelationGraphContrast((), (), -5., positive_operations=row.positive_operations,
                                 negative_operations=row.positive_operations)
    loss, gradient, margins = _labeled_graph_loss(np.zeros(6), (same,), **options)
    assert loss > 0 and margins == [-5.]
    np.testing.assert_array_equal(gradient, np.zeros(6))


def test_no_runtime_witness_is_not_a_successful_repair():
    _, options = _problem()
    loss, gradient, margins = _labeled_graph_loss(np.zeros(6), (), **options)
    assert loss == 0 and margins == []
    np.testing.assert_array_equal(gradient, np.zeros(6))


@pytest.fixture
def source_fit():
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    return refit_compositional_labeled_spans(parent, examples), examples


def test_runtime_constraints_cannot_train_on_validation_or_unknown_sources(source_fit):
    model, examples = source_fit
    held = next(item for item in examples if item.split == 'validation')
    source = next(item.ir.source_text_sha256 for item in examples if item.split == 'train')
    for identities in ((held.ir.source_text_sha256,), ('f' * 64,), (source, source)):
        with pytest.raises(ValueError, match='source-training'):
            refit_compositional_labeled_spans(model, examples, runtime_constraint_sources=identities)


def test_incomplete_runtime_mining_cannot_silently_export(source_fit, monkeypatch):
    model, examples = source_fit
    source = next(item.ir.source_text_sha256 for item in examples if item.split == 'train')
    monkeypatch.setattr('core.learning.semantic_joint_graph_learning.mine_runtime_graph_contrast',
                        lambda *args, **kwargs: (None, {'status': 'runtime_decode_unavailable'}))
    with pytest.raises(RuntimeError, match='unresolved'):
        refit_compositional_labeled_spans(model, examples, runtime_constraint_sources=(source,))


def test_equivalent_observation_keeps_empty_margin_evidence_unmeasured(source_fit, monkeypatch):
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )

    model, examples = source_fit
    source = next(item.ir.source_text_sha256 for item in examples if item.split == 'train')
    seen = []
    def mine(candidate, item, **kwargs):
        seen.append((candidate.receipt_sha256, item.split, item.ir.source_text_sha256))
        return None, {'status': 'equivalent', 'source_text_sha256': item.ir.source_text_sha256}
    monkeypatch.setattr('core.learning.semantic_joint_graph_learning.mine_runtime_graph_contrast', mine)
    fitted = refit_compositional_labeled_spans(model, examples, runtime_constraint_sources=(source,))
    record = fitted.training_receipt['labeled_span_fit']['runtime_graph_constraints']
    assert seen == [(model.receipt_sha256, 'train', source)]
    assert record['all_witnessed_margins_satisfied'] is None
    assert record['full_source_decision_retention_measured'] is False
    assert record['binding_coefficients'] == 'frozen_parent'
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).to_dict() == fitted.to_dict()
