"""Agreement is measured as a policy, never presumed correct."""

import pytest

from core.learning.semantic_method_overlap import (
    _mcnemar_exact, analyze_method_overlap, classify_program_structure,
)


def test_structural_diagnostic_separates_operation_and_binding_errors():
    target = [["count_of", [0, 1]], ["sub", [2, 3]]]
    assert classify_program_structure(target, target) == "same_program"
    assert classify_program_structure(target, [["count_of", [0, 1]], ["sub", [3, 2]]]) == "binding_only"
    assert classify_program_structure(target, [["count_of", [0, 1]], ["add", [2, 3]]]) == "operation_only"
    assert classify_program_structure(target, [["count_of", [0, 1]], ["add", [3, 2]]]) == "operation_and_binding"
    assert classify_program_structure(target, [["count_of", [0, 1]]]) == "depth_mismatch"


def _row(source, methods):
    return {"source": source, "construction": "variant", "methods": {
        name: {"program_sha256": key, "correct": correct}
        for name, (key, correct) in methods.items()}}


def test_overlap_counts_unique_wins_and_wrong_consensus():
    rows = [
        _row("a", {"incumbent": ("wrong", False), "ranker": ("right", True),
                   "direct": ("right", True), "prototype": ("other", False)}),
        _row("b", {"incumbent": ("wrong", False), "ranker": ("wrong", False),
                   "direct": ("other", False), "prototype": ("right", True)}),
        _row("c", {"incumbent": ("wrong", False), "ranker": ("wrong", False),
                   "direct": ("other", False), "prototype": ("other", False)}),
    ]
    result = analyze_method_overlap(rows)
    assert result["population"] == 3
    assert result["correct"] == {"incumbent": 0, "ranker": 1, "direct": 1,
                                 "prototype": 1}
    assert result["unique_successes"]["prototype"] == 1
    assert result["oracle_union"] == 2
    assert result["all_wrong"] == 1
    assert result["consensus_covered"] == 2
    assert result["consensus_correct"] == 1
    assert result["consensus_abstained"] == 1
    assert result["pairwise"]["ranker:direct"]["correct_when_agree"] == 1
    assert result["pairwise"]["incumbent:ranker"]["correct_when_agree"] == 0


def test_exact_pair_and_invalid_population():
    assert _mcnemar_exact(0, 0) == 1.
    assert _mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert _mcnemar_exact(2, 2) == 1.
    with pytest.raises(ValueError, match="distinct"):
        analyze_method_overlap([_row("a", {"incumbent": ("x", True),
                                           "ranker": ("x", True),
                                           "direct": ("x", True),
                                           "prototype": ("x", True)})] * 2)
