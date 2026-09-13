"""Distinguish proposal, constraint and ranking failures on the actual chart."""

import pytest

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def _chart(options, **kwargs):
    return ScoredArgumentChart(
        options, 2, RegisterUseContract(1, 1, 0, 1, True), **kwargs,
    )


def test_feasible_gold_graph_is_distinct_from_a_higher_scoring_wrong_graph():
    chart = _chart(((((1., 0, TokenSpan(0, 1)), (3., 1, TokenSpan(0, 1))),
                     ((1., 1, TokenSpan(2, 3)), (3., 0, TokenSpan(2, 3)))),))
    before = chart.solve()
    diagnosis = chart.diagnose_target(((0, 1),))
    assert diagnosis["cause"] == "target_feasible_but_not_selected"
    assert diagnosis["target_score_gap"] == pytest.approx(4)
    assert chart.solve() == before
    assert diagnosis["oracle_target_supplied"] and not diagnosis["serving_authority"]


def test_missing_register_proposal_is_reported_separately():
    chart = _chart(((((3., 1, TokenSpan(0, 1)),), ((3., 0, TokenSpan(2, 3)),)),))
    diagnosis = chart.diagnose_target(((0, 1),))
    assert diagnosis["cause"] == "target_register_not_proposed"
    assert diagnosis["missing_slots"] == [(0, 0, 0), (0, 1, 1)]
    assert diagnosis["target_score_gap"] is None


def test_overlapping_mentions_exclude_an_otherwise_covered_target():
    chart = _chart(((((1., 0, TokenSpan(0, 2)), (3., 1, TokenSpan(4, 5))),
                     ((1., 1, TokenSpan(1, 3)), (3., 0, TokenSpan(6, 7)))),))
    diagnosis = chart.diagnose_target(((0, 1),))
    assert diagnosis["cause"] == "target_excluded_by_joint_constraints"
    assert not diagnosis["missing_slots"]
    assert diagnosis["target_without_mention_exclusivity_feasible"] is True


def test_selected_target_retains_its_definition_attachment_objective():
    name = TokenSpan(8, 9)
    scores = {(0, name): 2., (1, name): -1.}
    chart = _chart(((((3., 0, TokenSpan(0, 1)),), ((4., 1, TokenSpan(2, 3)),)),),
                   definition_options=(((name,), (name,)),), definition_scores=scores)
    scores[0, name] = 100
    result = chart.diagnose_target(((0, 1),))
    assert result["cause"] == "target_selected"
    assert result["target_score"] == pytest.approx(8)
    with pytest.raises(TypeError):
        chart.definition_scores[0, name] = 5


@pytest.mark.parametrize("targets", [(), ((0,),), ((True, 1),), ((0, 8),)])
def test_invalid_target_is_rejected(targets):
    chart = _chart(((((1., 0, TokenSpan(0, 1)),), ((1., 1, TokenSpan(2, 3)),)),))
    with pytest.raises(ValueError, match="target"):
        chart.diagnose_target(targets)


def test_definition_consistency_exclusion_is_distinguished_from_other_constraints():
    first, second = TokenSpan(10, 11), TokenSpan(12, 13)
    chart = ScoredArgumentChart(
        ((((1., 0, TokenSpan(0, 1)),),),
         (((1., 0, TokenSpan(2, 3)),), ((1., 1, TokenSpan(4, 5)),))),
        1, RegisterUseContract(0, 3, 0, 3, False),
        definition_options=(((first,),), ((second,), (first,))),
    )
    result = chart.diagnose_target(((0,), (0, 1)))
    assert result["cause"] == "target_excluded_by_joint_constraints"
    assert result["target_without_definition_consistency_feasible"] is True


def test_literal_retention_preserves_identity_without_forcing_selection():
    from core.learning.semantic_program_transducer_fitting import _retained_argument_mentions

    candidates = [(float(index), TokenSpan(index, index + 1)) for index in range(6)]
    literal = candidates[0][1]
    legacy = _retained_argument_mentions(candidates)
    retained = _retained_argument_mentions(candidates, literal_anchor=literal)
    assert retained[:4] == legacy
    assert retained[4:] == [candidates[0]]
    assert len(_retained_argument_mentions(candidates, literal_anchor=candidates[-1][1])) == 4
    assert _retained_argument_mentions(candidates, literal_anchor=TokenSpan(8, 9)) == legacy


def test_literal_policy_has_its_own_identity_and_round_trips():
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    parent = fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())
    candidate = parent.with_literal_anchor_retention()
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
