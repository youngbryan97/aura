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
