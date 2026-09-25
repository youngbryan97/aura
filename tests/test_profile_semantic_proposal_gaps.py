"""Bounded proposal misses distinguish missing charts from graph selection."""

from types import SimpleNamespace

import pytest

from tools.profile_semantic_proposal_gaps import classify_proposal_gap


def _item():
    return SimpleNamespace(construction_id="new-composition", ir=SimpleNamespace(
        instructions=[SimpleNamespace(op="mul"), SimpleNamespace(op="idiv")]))


def _row(operations, *, reachable=None, complete=False):
    candidates = ([{"program_sha256": "correct", "program": {"instructions": [["mul"]]}}]
                  if reachable is True else [])
    comparisons = ([{"program_sha256": "correct", "status": "equivalent"}]
                   if reachable is True else [])
    return {"source": "source", "diagnosis": {"correct_reachable": reachable,
            "comparisons": comparisons},
            "bank": {"search_complete": complete, "candidates": candidates,
                     "selected_program_sha256": None,
                     "selected_search_interrupted": False, "charts": [
                {"operations": [{"op": name} for name in chart]}
                for chart in operations]}}


def test_missing_target_chart_is_not_called_impossible_when_search_incomplete():
    result = classify_proposal_gap(_item(), _row([("idiv", "idiv")]))
    assert result["stage"] == "target_operation_chart_not_observed"
    assert result["correct_reachable"] is None
    assert result["search_complete"] is False


def test_target_chart_without_equivalent_graph_is_separate():
    result = classify_proposal_gap(_item(), _row([("mul", "idiv")]))
    assert result["stage"] == "target_operation_chart_observed_no_equivalent_graph"


def test_witnessed_equivalent_candidate_dominates_target_form():
    result = classify_proposal_gap(_item(), _row([("idiv", "idiv")], reachable=True))
    assert result["stage"] == "correct_candidate_observed"


def test_incomplete_search_cannot_certify_absence():
    with pytest.raises(ValueError, match="search completeness"):
        classify_proposal_gap(_item(), _row([], reachable=False))
