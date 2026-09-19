"""Complete graph learning must reach the operation span scores used at runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_argument_graph_learning import argument_parameters
from core.learning.semantic_graph_batch import GraphConstraintBatch
from core.learning.semantic_graph_constraints import fit_complete_graph_constraints
from core.learning.semantic_joint_graph_learning import (
    joint_graph_contrast, refit_compositional_joint_graphs, score_annotated_graph,
    source_operation_pointer_constraints,
)
from core.learning.semantic_operation_pointer_learning import (
    operation_pointer_from_parameters, operation_pointer_graph_evidence, operation_pointer_parameters,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_relation_graph_learning import RelationGraphContrast, graph_margin, graph_margin_gradient
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def trained():
    model, examples = model_examples()
    return model.with_joint_operation_argument_scores().with_source_ordered_definitions(), examples


def parameters(model):
    return (model.definition_relation_head.query_projection.astype(np.float64),
            model.definition_relation_head.definition_projection.astype(np.float64),
            *(v.astype(np.float64) for h in model.operation_head.heads for v in (h.weight, h.bias)),
            *argument_parameters(model), *operation_pointer_parameters(model))


def test_boundary_terms_replay_the_shipped_pointer(trained):
    model, examples = trained
    for item in examples[:3]:
        nodes = [SimpleNamespace(span=ins.operation_span) for ins in item.ir.instructions]
        terms = operation_pointer_graph_evidence(model, item.hidden_states, nodes)
        scores = model.operation_pointer.score_sequence(item.hidden_states)
        for node, term in zip(nodes, terms, strict=True):
            assert term.score(parameters(model)) == pytest.approx(scores.score_span(node.span), abs=2e-5)


def test_pointer_gradient_matches_finite_differences_and_batched_engine(trained):
    model, examples = trained
    item = examples[0]
    terms = operation_pointer_graph_evidence(model, item.hidden_states,
        [SimpleNamespace(span=TokenSpan(0, 1)), SimpleNamespace(span=TokenSpan(1, 2))])
    row = RelationGraphContrast((), (), -0.2, argument_terms=((1., terms[0]), (-1., terms[1])))
    values = parameters(model)
    value, gradient = graph_margin_gradient(values, row)
    batch = GraphConstraintBatch((row,))
    assert batch.margins(values)[0] == pytest.approx(value)
    for a, b in zip(batch.weighted_gradient(values, [1.]), gradient, strict=True):
        np.testing.assert_allclose(a, b)
    for index, position in ((len(values) - 2, 0), (len(values) - 2, len(values[-2]) - 1), (len(values) - 1, ())):
        plus, minus = [x.copy() for x in values], [x.copy() for x in values]
        plus[index][position] += 1e-5
        minus[index][position] -= 1e-5
        assert gradient[index][position] == pytest.approx((graph_margin(plus, row) - graph_margin(minus, row)) / 2e-5, abs=1e-7)
    direction = tuple(np.ones_like(v) * 0.01 for v in values)
    assert batch.directional_derivative(values, direction)[0] == pytest.approx(
        sum(np.sum(a * b) for a, b in zip(gradient, direction, strict=True)))


def test_complete_fit_can_repair_a_boundary_constraint_the_old_fit_cannot_reach(trained):
    model, examples = trained
    item = examples[0]
    term = operation_pointer_graph_evidence(model, item.hidden_states,
        [SimpleNamespace(span=TokenSpan(0, 1))])[0]
    row = RelationGraphContrast((), (), -5. - term.score(parameters(model)), argument_terms=((1., term),))
    with pytest.raises(ValueError, match="parameter block"):
        fit_complete_graph_constraints(model, (row,), steps=3)
    fitted, receipt = fit_complete_graph_constraints(model, (row,), steps=3,
        adaptive_step=True, learn_operation_pointer=True)
    assert receipt["stored_wrong_or_tied"] == 0
    assert receipt["operation_pointer_trainable"] is True
    assert not np.array_equal(fitted.operation_pointer.start_weight, model.operation_pointer.start_weight)
    assert fitted.input_grounding == model.input_grounding
    assert graph_margin(parameters(fitted), row) == pytest.approx(receipt["stored_margins"][0], abs=1e-5)


def test_pointer_parameter_roundtrip_preserves_scores(trained):
    model, examples = trained
    fitted = operation_pointer_from_parameters(operation_pointer_parameters(model))
    item = examples[0]
    for ins in item.ir.instructions:
        assert fitted.score_sequence(item.hidden_states).score_span(ins.operation_span) == pytest.approx(
            model.operation_pointer.score_sequence(item.hidden_states).score_span(ins.operation_span), abs=1e-6)


def test_graph_contrast_separates_pointer_terms_from_fixed_margin(trained):
    model, examples = trained
    item = next(x for x in examples if x.split == "train")
    positive = score_annotated_graph(model, item, item.ir.instructions, item.ir.input_spans,
                                    learn_arguments=True, learn_operation_pointer=True)
    assert positive is not None
    negative = dict(positive, score=positive["score"] - 0.25)
    row = joint_graph_contrast(model, positive, negative)
    assert any(term.parameter_index == len(parameters(model)) - 2 for _, term in row.argument_terms)
    assert graph_margin(parameters(model), row, scale=model.definition_relation_scale) == pytest.approx(0.25, abs=1e-6)


def test_source_trainer_wires_pointer_learning_and_roundtrips(trained):
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, examples = trained
    fitted = refit_compositional_joint_graphs(model, examples, rounds=1, steps=2,
        retention_operation_charts=2, constraint_learning=True, learn_operation_pointer=True)
    receipt = fitted.training_receipt["joint_graph_refit"]
    assert receipt["operation_pointer_trainable"] is True
    assert receipt["argument_heads_trainable"] is False
    assert receipt["validation_used_for_fit"] is False and receipt["test_examples_used"] == 0
    assert all(row["fit"]["retained_positive_regressions"] == 0 for row in receipt["rounds"])
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).receipt_sha256 == fitted.receipt_sha256


def test_pointer_learning_requires_retention(trained):
    with pytest.raises(ValueError, match="retained constraints"):
        refit_compositional_joint_graphs(*trained, learn_operation_pointer=True)


def test_source_boundary_constraints_replay_real_pointer_and_are_not_empty(trained):
    model, examples = trained
    rows = source_operation_pointer_constraints(
        model, tuple(item for item in examples if item.split == "train"),
    )
    assert rows
    values = parameters(model)
    for row in rows[:10]:
        margin, gradient = graph_margin_gradient(values, row)
        assert np.isfinite(margin)
        assert len(gradient) == len(values)
        assert row.fixed_margin == pytest.approx(.1)
        assert any(np.any(part) for part in gradient[-2:])
