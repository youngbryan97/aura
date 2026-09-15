"""Complete graph contrasts exclude meanings, not alternate mentions."""

import itertools

import pytest

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def chart():
    a, b, c, d = (TokenSpan(i, i + 1) for i in range(4))
    return ScoredArgumentChart(
        ((((9., 0, a), (8., 0, b), (7., 1, c)),
          ((6., 1, d), (5., 0, b))),),
        2, RegisterUseContract(1, 1, 1, 1, True),
    )


@pytest.mark.parametrize("prune", [False, True])
def test_exclusion_preserves_optimum_over_every_other_graph(prune):
    from dataclasses import replace

    value = replace(chart(), prune_dominated=prune)
    target = ((0, 1),)
    positive = value.restrict_arguments(target).solve()
    negative = value.solve(excluded_arguments=target)
    assert positive[0] == 15. and positive[1] == target
    assert negative[0] == 12. and negative[1] == ((1, 0),)
    # The second-best mention of the target scores 14; it is not a negative.
    assert negative[0] < 14.


def test_exhaustive_permutations_match_excluded_solver():
    value = chart()
    for target in (((0, 1),), ((1, 0),)):
        candidates = []
        for first, second in itertools.product(*value.options[0]):
            if first[1] == second[1] or first[2] == second[2]:
                continue
            if ((first[1], second[1]),) != target:
                candidates.append(first[0] + second[0])
        assert value.solve(excluded_arguments=target)[0] == max(candidates)


def test_excluding_only_feasible_meaning_returns_no_negative():
    value = chart().restrict_arguments(((0, 1),))
    assert value.solve(excluded_arguments=((0, 1),)) is None
    assert value.solve()[1] == ((0, 1),)


@pytest.mark.parametrize("target", [(), ((0,),), ((0, True),), ((0, 3),)])
def test_malformed_targets_are_rejected(target):
    with pytest.raises(ValueError):
        chart().restrict_arguments(target)
    with pytest.raises(ValueError):
        chart().solve(excluded_arguments=target)


def test_selected_factors_follow_the_actual_latent_definition_choice():
    from dataclasses import replace

    value = chart()
    factors = tuple(tuple(tuple((option[0], 0., 0., 0.) for option in slot)
                          for slot in node) for node in value.options)
    value = replace(value, option_factors=factors)
    solution, total = value.solve_with_factors()
    assert solution[0] == total[0] == 15.
    target = value.restrict_arguments(((1, 0),))
    assert target.solve_with_factors()[1] == (12., 0., 0., 0.)
    assert value.solve_with_factors(excluded_arguments=((0, 1),))[1] == (12., 0., 0., 0.)
    with pytest.raises(ValueError, match="score factors"):
        chart().solve_with_factors()
    with pytest.raises(ValueError, match="factors differ"):
        replace(value, option_factors=((((),),),))


def test_graph_scale_gradient_and_fit_use_complete_margin():
    import numpy as np
    from core.learning.semantic_graph_margin import _graph_margin_loss, fit_graph_score_scales

    differences = np.array([[1., -2., 0.], [2., -1., .2], [1., -1., -.1]])
    offsets = np.array([-.5, -.2, -.1])
    weights, initial = np.ones(3), np.ones(3)
    loss, gradient = _graph_margin_loss(initial, differences, offsets, weights, initial, .01)
    for index in range(3):
        delta = np.eye(3)[index] * 1e-5
        upper = _graph_margin_loss(initial + delta, differences, offsets, weights, initial, .01)[0]
        lower = _graph_margin_loss(initial - delta, differences, offsets, weights, initial, .01)[0]
        assert gradient[index] == pytest.approx((upper - lower) / 2e-5, abs=1e-8)
    fitted, receipt = fit_graph_score_scales(differences, offsets, weights, initial)
    assert np.all(fitted > 0)
    assert np.all(differences @ fitted + offsets > 0)
    assert receipt["fitted_loss"] < loss
    assert receipt["proposal_scale_held_fixed"]
    with pytest.raises(ValueError, match="supervision"):
        fit_graph_score_scales(differences, np.array([float("nan")] * 3), weights, initial)


def test_source_graph_fit_preserves_heads_and_excludes_test_examples():
    from core.learning.semantic_graph_margin import refit_compositional_graph_scales
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer, compositional_semantic_program_transducer_from_dict,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    model = model.with_global_constraint_arguments().with_conditional_argument_scores()
    rows = []
    candidate = refit_compositional_graph_scales(model, examples, progress=rows.append)
    report = candidate.training_receipt["argument_graph_factor_refit"]
    assert report["test_examples_used"] == 0 and report["validation_used_for_fit"] is False
    assert len(rows) == report["training_examples"]
    assert sum(report["coverage"].values()) == report["training_examples"]
    assert report["fit"]["pairs"] == report["coverage"]["contrast"]
    assert candidate.argument_proposal_scale == model.argument_proposal_scale
    for key, value in model._coefficient_body().items():
        if key not in ("argument_role_scale", "definition_relation_scale", "argument_pointer_scale"):
            assert candidate._coefficient_body()[key] == value
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
    assert refit_compositional_graph_scales(model, tuple(item for item in examples if item.split != "test")).receipt_sha256 == candidate.receipt_sha256
