"""Check the source-only ranking objective and its runtime parameterization."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_argument_ranking import _pairwise_loss, fit_pairwise_argument_weight


def test_pairwise_gradient_matches_finite_differences_and_batch_sizes():
    rng = np.random.default_rng(81)
    features = rng.normal(size=(8, 4)).astype(np.float32)
    positive, negative = np.array([0, 0, 3, 3, 3]), np.array([1, 2, 4, 5, 6])
    sample_weight = np.array([1., 2., 1., .5, 3.])
    weight = rng.normal(size=4)
    kwargs = dict(regularization=.1)
    loss, gradient = _pairwise_loss(weight, features, positive, negative, sample_weight, **kwargs)
    for index in range(4):
        delta = np.zeros(4)
        delta[index] = 1e-5
        above = _pairwise_loss(weight + delta, features, positive, negative, sample_weight, **kwargs)[0]
        below = _pairwise_loss(weight - delta, features, positive, negative, sample_weight, **kwargs)[0]
        assert gradient[index] == pytest.approx((above - below) / 2e-5, abs=1e-8)
    other_loss, other_gradient = _pairwise_loss(
        weight, features, positive, negative, sample_weight, batch_size=1, **kwargs
    )
    assert other_loss == pytest.approx(loss, abs=1e-12)
    assert np.allclose(other_gradient, gradient, atol=1e-12)


def test_fitter_learns_group_preferences_without_a_slot_bias():
    features = np.array([[2., 0.], [0., 2.], [1., 0.], [-1., 1.]])
    labels = np.array([1, 0, 1, 0])
    weight, report = fit_pairwise_argument_weight(
        features, labels, np.ones(4), initial_weight=np.zeros(2)
    )
    assert features[0] @ weight > features[1] @ weight
    assert features[2] @ weight > features[3] @ weight
    assert report["converged"] is True
    assert report["exported_loss"] < report["initial_loss"]
    assert report["groups"] == report["pairs"] == 2
    assert report["bias_fitted"] is False


def test_fixed_pointer_margins_have_the_actual_runtime_gradient():
    features = np.array([[2., 1.], [-1., 2.], [1., 0.]])
    positive, negative = np.array([0, 0]), np.array([1, 2])
    weights, offsets = np.array([1., 2.]), np.array([-4., -.1, -2.])
    point = np.array([.2, -.4])
    options = dict(regularization=.1, fixed_scores=offsets)
    loss, gradient = _pairwise_loss(point, features, positive, negative, weights, **options)
    for index in range(2):
        delta = np.eye(2)[index] * 1e-5
        above = _pairwise_loss(point + delta, features, positive, negative, weights, **options)[0]
        below = _pairwise_loss(point - delta, features, positive, negative, weights, **options)[0]
        assert gradient[index] == pytest.approx((above - below) / 2e-5, abs=1e-8)
    shifted = _pairwise_loss(point, features, positive, negative, weights,
                            regularization=.1, fixed_scores=offsets + 17.)
    assert shifted[0] == pytest.approx(loss)
    np.testing.assert_allclose(shifted[1], gradient)


def test_fixed_margin_fit_compensates_for_a_frozen_pointer_disadvantage():
    features = np.array([[1.], [0.]])
    options = dict(labels=np.array([1, 0]), sample_weight=np.ones(2),
                   initial_weight=np.zeros(1), inverse_regularization=1000.)
    ordinary, _ = fit_pairwise_argument_weight(features, **options)
    fitted, receipt = fit_pairwise_argument_weight(features, **options,
                                                  fixed_scores=np.array([-12., 0.]))
    assert ordinary[0] - 12. < 0
    assert fitted[0] - 12. > 0
    assert receipt["objective"] == "source_grouped_pairwise_fixed_margin_v2"
    zeros, _ = fit_pairwise_argument_weight(features, **options, fixed_scores=np.zeros(2))
    np.testing.assert_array_equal(zeros, ordinary)


def test_factorized_rows_preserve_expanded_features_loss_and_fitted_weights():
    from core.learning.semantic_relation_tissue import DirectionalFeatureRows, _directional_relation_feature

    rng = np.random.default_rng(42)
    references = rng.normal(size=(3, 4)).astype(np.float32)
    operations = rng.normal(size=(2, 4)).astype(np.float32)
    rows = DirectionalFeatureRows()
    dense = []
    for operation in operations:
        for reference in references:
            rows.append(reference, operation)
            dense.append(_directional_relation_feature(reference, operation))
    dense = np.stack(dense)
    np.testing.assert_array_equal(rows[:], dense)
    np.testing.assert_array_equal(rows[np.array([4, 0, 4])], dense[[4, 0, 4]])
    assert rows[0:0].shape == (0, 12)
    assert rows.vector_storage_bytes < dense.nbytes
    options = dict(labels=np.array([1, 0, 0, 1, 0, 0]), sample_weight=np.ones(6),
                   initial_weight=np.zeros(12), fixed_scores=np.arange(6) * -.2)
    expected, receipt = fit_pairwise_argument_weight(dense, **options)
    actual, compact = fit_pairwise_argument_weight(rows, **options)
    np.testing.assert_array_equal(actual, expected)
    assert compact["exported_loss"] == receipt["exported_loss"]
    assert compact["vector_storage_bytes"] == rows.vector_storage_bytes
    with pytest.raises(ValueError, match="geometry"):
        rows.append(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="width"):
        rows.append(np.ones(2), np.ones(2))


@pytest.mark.parametrize("offsets", [[0.], [0., float("nan")], [[0.], [1.]]])
def test_invalid_fixed_scores_are_rejected(offsets):
    with pytest.raises(ValueError, match="supervision"):
        fit_pairwise_argument_weight(np.array([[1.], [0.]]), np.array([1, 0]),
                                     np.ones(2), initial_weight=np.zeros(1), fixed_scores=offsets)


@pytest.mark.parametrize("labels", [[0, 1], [1, 1], [1, 0, 1], [1, 2]])
def test_malformed_or_unsupported_groups_are_rejected(labels):
    with pytest.raises(ValueError):
        fit_pairwise_argument_weight(np.ones((len(labels), 2)), np.array(labels),
            np.ones(len(labels)), initial_weight=np.zeros(2))


def test_source_refit_preserves_other_heads_and_excludes_test_labels():
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
        refit_compositional_argument_rankings,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    candidate = refit_compositional_argument_rankings(parent, examples)
    for key, value in parent._coefficient_body().items():
        if key != "argument_role_heads":
            assert candidate._coefficient_body()[key] == value
    receipt = candidate.training_receipt["argument_ranking_refit"]
    assert receipt["test_examples_used"] == 0
    assert receipt["validation_used_for_fit"] is False
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
    without_test = tuple(x for x in examples if x.split != "test")
    assert refit_compositional_argument_rankings(parent, without_test).receipt_sha256 == candidate.receipt_sha256
    identity_candidate = refit_compositional_argument_rankings(
        parent, examples, preserve_coreferent_mentions=True,
    )
    assert identity_candidate.training_receipt["argument_ranking_refit"]["negative_source"].endswith("v2")
    assert sum(fit["pairs"] for fit in identity_candidate.training_receipt["argument_ranking_refit"]["fits"]) < sum(fit["pairs"] for fit in receipt["fits"])
    assert compositional_semantic_program_transducer_from_dict(identity_candidate.to_dict()).receipt_sha256 == identity_candidate.receipt_sha256
    runtime = refit_compositional_argument_rankings(parent, without_test, runtime_mention_margin=True)
    runtime_receipt = runtime.training_receipt["argument_ranking_refit"]
    assert runtime_receipt["mention_objective"] == "runtime_pointer_margin_v1"
    assert runtime_receipt["graph_relation_terms_fitted"] is False
    assert runtime_receipt["validation_used_for_fit"] is False
    assert runtime_receipt["test_examples_used"] == 0
    assert compositional_semantic_program_transducer_from_dict(runtime.to_dict()).receipt_sha256 == runtime.receipt_sha256
    for key, value in parent._coefficient_body().items():
        if key != "argument_role_heads":
            assert runtime._coefficient_body()[key] == value
    with pytest.raises(ValueError, match="overlap"):
        refit_compositional_argument_rankings(parent,
            (replace(examples[0], split="train"), replace(examples[0], split="validation")))


def test_source_aliases_are_identity_based_not_value_based():
    from core.learning.semantic_program_transducer_fitting import _argument_identity_spans
    from tests.test_semantic_program_shared_transducer import _examples

    item = _examples()[0]
    aliases = _argument_identity_spans(item, 0)
    assert item.ir.input_spans[0] in aliases
    assert item.ir.instructions[0].argument_spans[0] in aliases
    assert item.ir.input_spans[1] not in aliases
    assert item.ir.instructions[0].argument_spans[1] not in aliases
    # Reusing a surface for conflicting labels must not erase a negative.
    instruction = replace(item.ir.instructions[0], argument_spans=(
        item.ir.input_spans[1], item.ir.instructions[0].argument_spans[1],
    ))
    ambiguous = replace(item, ir=replace(item.ir, instructions=(instruction, *item.ir.instructions[1:])))
    assert item.ir.input_spans[1] not in _argument_identity_spans(ambiguous, 0)
    assert item.ir.input_spans[1] not in _argument_identity_spans(ambiguous, 1)


def test_full_mention_rows_keep_every_pointer_candidate_and_aligned_offset(monkeypatch):
    from core.learning import semantic_program_transducer_fitting as fitting
    from core.learning.semantic_program_transducer import LinearPointerHead
    from core.learning.semantic_program_ir import TokenSpan
    from tests.test_semantic_program_shared_transducer import _examples

    item = _examples()[0]
    candidates = tuple(TokenSpan(i, i + 1) for i in range(item.hidden_states.shape[0]))
    weight = np.ones(item.hidden_states.shape[1], dtype=np.float32)
    pointer = LinearPointerHead(weight, 0., -weight, 0.)
    monkeypatch.setattr(fitting, "_argument_proposals_by_operation", lambda *args, **kwargs:
                        tuple(tuple((span, 0.) for span in candidates) for _ in item.ir.instructions))
    options = dict(argument_pointer=pointer, position=0, max_span_tokens=1,
                   max_argument_span_tokens_by_type={"integer": 1, "integer_sequence": 1},
                   hidden_channels=item.hidden_channels, hidden_channel_widths=item.hidden_channel_widths)
    legacy = fitting._argument_proposal_rows((item,), **options)
    offsets = []
    features, labels, _, _, negatives = fitting._argument_proposal_rows(
        (item,), **options, full_runtime_mentions=True, fixed_pointer_scores=offsets, pointer_scale=.5)
    assert negatives > legacy[-1]
    assert len(features) == len(labels) == len(offsets) == len(candidates) * len(item.ir.instructions)
    scores = pointer.score_sequence(item.hidden_states)
    expected = [
        .5 * fitting._log_sigmoid(scores.score_span(span))
        for instruction in item.ir.instructions
        for span in (instruction.argument_spans[0],
                     *(span for span in candidates if span != instruction.argument_spans[0]))
    ]
    np.testing.assert_allclose(offsets, expected)
    compact = fitting._argument_proposal_rows((item,), **options, full_runtime_mentions=True,
                                              factorized_features=True)
    np.testing.assert_array_equal(compact[0][:], features)
    np.testing.assert_array_equal(compact[1], labels)
