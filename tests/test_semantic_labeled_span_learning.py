"""Labeled interval likelihood, gradients, and shipped score agreement."""

from dataclasses import replace
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from core.learning.semantic_labeled_span_learning import (
    MeanSpanEvidence,
    MeanTransitionSpanEvidence,
    _labeled_loss,
    _runtime_odds,
    labeled_span_partition,
    refit_compositional_labeled_spans,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import _operation_feature
from core.learning.semantic_span_set_learning import span_set_partition


def test_labeled_partition_equals_every_disjoint_labeled_set():
    scores = np.random.default_rng(4).normal(size=(4, 2, 3))
    scores[-1, 1] = -np.inf
    scores[1, 0] = -np.inf
    intervals = [(a, b) for a in range(4) for b in (1, 2) if a + b <= 4]
    sets = [()]
    for count in (1, 2):
        for spans in combinations(intervals, count):
            if all(a + b <= c for (a, b), (c, _) in zip(spans, spans[1:], strict=False)):
                sets.extend(tuple((s, length, op) for (s, length), op in zip(spans, labels, strict=True))
                            for labels in product(range(3), repeat=count))
    total = np.asarray([sum(scores[a, b - 1, c] for a, b, c in row) for row in sets])
    expected, actual = logsumexp(total), labeled_span_partition(scores, 2)
    marginal = np.zeros_like(scores)
    for probability, row in zip(np.exp(total - expected), sets, strict=True):
        for a, b, c in row:
            marginal[a, b - 1, c] += probability
    assert actual[0] == pytest.approx(expected)
    np.testing.assert_allclose(actual[1], marginal, atol=1e-12)


def test_an_excluded_grammar_has_only_the_empty_set_and_finite_zero_gradient():
    scores = np.full((5, 3), -np.inf)
    partition, gradient = span_set_partition(scores, 4)
    assert partition == 0 and np.count_nonzero(gradient) == 0


@pytest.mark.parametrize("mode,evidence_type", [
    ("contextual_mean", MeanSpanEvidence),
    ("contextual_mean_transition", MeanTransitionSpanEvidence),
])
def test_prefix_projection_and_adjoint_equal_explicit_runtime_features(mode, evidence_type):
    hidden = np.random.default_rng(5).normal(size=(7, 8)).astype(np.float32)
    evidence = evidence_type.build(hidden[:, 4:], 3, (TokenSpan(3, 4),))
    features = np.stack([_operation_feature(hidden, TokenSpan(int(a), int(b)), mode=mode,
        hidden_channels=("input_token_embedding", "final_causal_hidden"), hidden_channel_widths=(4, 4))
        for a, b in zip(evidence.starts, evidence.ends, strict=True)])
    weight = np.random.default_rng(6).normal(size=(3, features.shape[1]))
    residual = np.random.default_rng(7).normal(size=(len(features), 3))
    np.testing.assert_allclose(evidence.project(weight), features @ weight.T, atol=2e-7)
    np.testing.assert_allclose(evidence.adjoint(residual), residual.T @ features, atol=3e-7)


@pytest.mark.parametrize("magnitude", [1., 80.])
@pytest.mark.parametrize("evidence_type,multiplier", [(MeanSpanEvidence, 1), (MeanTransitionSpanEvidence, 2)])
def test_joint_loss_gradient_includes_competing_spans_labels_count_and_runtime_clipping(magnitude, evidence_type, multiplier):
    hidden = np.random.default_rng(9).normal(size=(5, 3))
    evidence = evidence_type.build(hidden, 3, (TokenSpan(4, 5),))
    rows = [(evidence, ((0, 2, 1), (2, 1, 0)), 1.)]
    size = 6 * multiplier + 2
    parameters = np.random.default_rng(12).normal(size=size) * magnitude
    def objective(value):
        return _labeled_loss(value, rows, labels=2, width=3 * multiplier, max_spans=2,
                             regularization=.01, center=np.zeros(size))
    _, gradient = objective(parameters)
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += 1e-4
        minus[index] -= 1e-4
        assert (objective(plus)[0] - objective(minus)[0]) / 2e-4 == pytest.approx(gradient[index], abs=2e-6)


def test_odds_match_shipped_background_probability_clipping():
    from core.learning.semantic_program_transducer import LinearClassifierHead

    for logits in ([1., 4.], [1000., -1000.], [-50., -80.]):
        p = LinearClassifierHead(("add", "sub", "background"), np.zeros((3, 1)), np.array([*logits, 0.]))
        probabilities = p.predict_probabilities(np.zeros(1))
        expected = np.log(np.maximum(probabilities[:-1], 1e-12)) - np.log(max(probabilities[-1], 1e-12))
        np.testing.assert_allclose(_runtime_odds(np.asarray([logits]))[0][0], expected, atol=2e-6)


@pytest.mark.parametrize("mode", ["contextual_mean", "contextual_mean_transition"])
def test_fit_serializes_existing_heads_and_never_uses_validation_for_fit(mode):
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    assert parent.operation_head.modes == ("contextual_mean",)
    if mode != "contextual_mean":
        from core.learning.semantic_operation_view_refit import refit_compositional_operation_views
        parent = refit_compositional_operation_views(parent, examples, candidate_modes=(mode,))
    fitted = refit_compositional_labeled_spans(parent, examples)
    assert fitted.operation_head.modes == (mode,)
    record = fitted.training_receipt["labeled_span_fit"]
    assert record["exported_loss"] < record["initial_loss"]
    assert not fitted.training_receipt["operation_background_fit"]["validation_used_for_fit"]
    assert fitted.argument_pointer.to_dict() == parent.argument_pointer.to_dict()
    assert fitted.definition_relation_head.to_dict() == parent.definition_relation_head.to_dict()
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).to_dict() == fitted.to_dict()


@pytest.fixture
def source_fit():
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    return fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding()), examples


def test_fit_rejects_duplicate_and_cross_split_training_sources(source_fit):
    model, examples = source_fit
    one = next(item for item in examples if item.split == "train")
    for added in (one, replace(one, split="validation")):
        with pytest.raises(ValueError, match="unique disjoint"):
            refit_compositional_labeled_spans(model, (*examples, added))


def test_fit_refuses_changed_basis_and_missing_source_operations(source_fit):
    model, examples = source_fit
    training = tuple(item for item in examples if item.split == "train")
    foreign = replace(training[0], tokenizer_identity_sha256="f" * 64)
    with pytest.raises(ValueError, match="representation differs"):
        refit_compositional_labeled_spans(model, (foreign, *training[1:]))
    label = model.operation_head.labels[0]
    one_label = (replace(training[0], ir=replace(training[0].ir,
        instructions=tuple(replace(i, op=label) for i in training[0].ir.instructions))),)
    with pytest.raises(ValueError, match="every source operation"):
        refit_compositional_labeled_spans(model, one_label)


def test_an_unconverged_optimizer_cannot_export_a_candidate(source_fit, monkeypatch):
    model, examples = source_fit
    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=False, status=1, message="iteration budget exhausted"))
    with pytest.raises(RuntimeError, match="fit incomplete"):
        refit_compositional_labeled_spans(model, examples)


def test_declared_partition_invariant_runs():
    from core.learning.semantic_labeled_span_learning import _check_labeled_span_partition

    assert _check_labeled_span_partition() == ()


def test_transition_evidence_zero_states_and_start_boundary_remain_finite():
    evidence = MeanTransitionSpanEvidence.build(np.zeros((4, 3)), 3)
    assert 0 in evidence.starts
    np.testing.assert_array_equal(evidence.project(np.ones((2, 6))), np.zeros((len(evidence.starts), 2)))
    np.testing.assert_array_equal(evidence.adjoint(np.ones((len(evidence.starts), 2))), np.zeros((2, 6)))
    with pytest.raises(ValueError, match="width differs"):
        evidence.project(np.ones((2, 3)))
