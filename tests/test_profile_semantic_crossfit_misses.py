"""Source-fold miss profiles preserve reach uncertainty and failure mechanism."""

import pytest

from tools.profile_semantic_crossfit_misses import classify_row


def _candidate(identity, operations):
    return {"program_sha256": identity,
            "program": {"instructions": [[operation, []] for operation in operations]}}


def _row(selected, correct, *, complete=False, reachable=True, interrupted=False):
    candidates = [_candidate("good", ("add", "mul")),
                  _candidate("same", ("mul", "add")),
                  _candidate("different", ("sub", "mul"))]
    if not correct:
        candidates = candidates[1:]
    return {"source": "case", "bank": {"candidates": candidates,
            "selected_program_sha256": selected, "search_complete": complete,
            "selected_search_interrupted": interrupted},
            "diagnosis": {"correct_reachable": reachable,
                          "comparisons": [{"program_sha256": item["program_sha256"],
                                           "status": "equivalent" if item["program_sha256"]
                                           == "good" else "different"}
                                          for item in candidates]}}


def test_selection_misses_are_separated_by_operation_inventory():
    assert classify_row(_row("same", True))["mechanism"] == (
        "same_operations_wrong_structure_or_arguments")
    assert classify_row(_row("different", True))["mechanism"] == (
        "operation_inventory_mismatch")
    assert classify_row(_row("good", True))["stage"] == "correct_selection"


def test_incomplete_search_is_not_an_unreachable_domain():
    result = classify_row(_row("same", False, reachable=None))
    assert result["stage"] == "reach_unobserved"
    assert result["observed_correct_operation_signatures"] == []


def test_interrupted_bounded_decode_is_not_ranker_error():
    result = classify_row(_row(None, True, interrupted=True))
    assert result["stage"] == "bounded_decode_interrupted"
    assert result["mechanism"] == "diagnostic_budget_expired_after_correct_candidate"


def test_inconsistent_reach_and_comparison_refused():
    with pytest.raises(ValueError, match="equivalent"):
        classify_row(_row("same", False, reachable=True))
