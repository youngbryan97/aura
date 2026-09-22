"""Gap attribution must keep observed labels out of serving authority."""

from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.diagnose_semantic_portfolio_gaps import (
    assess_composition, classify_program_difference, diagnose,
)


def _example():
    instruction = SimpleNamespace(op="add", args=(0, 1))
    program = SimpleNamespace(to_dict=lambda: {"instructions": [["add", [0, 1]]]})
    return SimpleNamespace(instructions=[SimpleNamespace(instruction=instruction)],
                           construction_id="source_construction", topology_id="linear",
                           source_text="Use both inputs.", program=program)


@pytest.mark.parametrize("predicted, expected", [
    (None, "no_program"),
    ({"instructions": []}, "depth"),
    ({"instructions": [["sub", [0, 1]]]}, "operation_or_order"),
    ({"instructions": [["add", [1, 0]]]}, "argument_binding"),
    ({"instructions": [["add", [0, 1]]]}, "syntactic_match_but_semantic_failure"),
])
def test_gap_classification_is_structural_not_phrase_specific(predicted, expected):
    assert classify_program_difference(_example(), predicted) == expected


def test_diagnosis_reproduces_missing_and_selection_counts():
    examples = {"a": _example(), "b": _example()}
    portfolio = {
        "plan": {"source_ids": ["a", "b"]},
        "rows": [
            {"source": "a", "selected": "baseline", "correct": False, "oracle_available": False,
             "method_correctness": {"baseline": False, "alternative": False}},
            {"source": "b", "selected": "baseline", "correct": False, "oracle_available": True,
             "method_correctness": {"baseline": False, "alternative": True}},
        ],
        "summary": {"candidate_unavailable": 1, "available_but_not_selected": 1},
    }
    graphs = {
        ("a", "baseline"): {"comparison": {"status": "different"}, "program": None},
        ("a", "alternative"): {"comparison": {"status": "different"},
                               "program": {"instructions": [["sub", [0, 1]]]}},
        ("b", "baseline"): {"comparison": {"status": "different"},
                            "program": {"instructions": [["add", [1, 0]]]}},
        ("b", "alternative"): {"comparison": {"status": "equivalent"},
                               "program": {"instructions": [["add", [0, 1]]]}},
    }
    report = diagnose(portfolio, examples, graphs, portfolio_sha="a" * 64,
                      manifest_shas={}, graph_shas={})
    assert report["summary"]["candidate_unavailable"] == 1
    assert report["summary"]["selection_miss"] == 1
    assert report["failures"][0]["method_failures"] == {
        "baseline": "no_program", "alternative": "operation_or_order"}
    assert report["failures"][1]["method_programs"]["alternative"] == {
        "instructions": [["add", [0, 1]]]}
    assert report["promotion_authorized"] is False
    portfolio["rows"][0]["method_correctness"]["baseline"] = True
    with pytest.raises(ValueError, match="selected correctness differs"):
        diagnose(portfolio, examples, graphs, portfolio_sha="a" * 64,
                 manifest_shas={}, graph_shas={})


def test_composition_assessment_proposes_before_reading_exposed_labels():
    gold = Program(3, (Instruction("add", (0, 1)), Instruction("mul", (3, 2))))
    left = Program(3, (Instruction("sub", (0, 1)), Instruction("mul", (3, 2))))
    right = Program(3, (Instruction("add", (0, 1)), Instruction("add", (3, 2))))
    gap = {"failures": [{"source": "s", "kind": "candidate_unavailable",
                         "method_programs": {"left": left.to_dict(), "right": right.to_dict()}}]}
    examples = {"s": SimpleNamespace(program=gold, inputs=(7, 3, 2))}
    report = assess_composition(gap, examples, budget=8, examined_limit=16)
    assert report["summary"]["candidate_unavailable_proved_recovered"] == 1
    assert report["summary"]["incomplete_search_rows"] == 0
    assert report["selection_changed"] is False
    assert any(candidate["program_sha256"] == gold.sha()
               for candidate in report["rows"][0]["candidates"])
    gap["failures"][0]["method_programs"]["left"]["sha"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="identity differs"):
        assess_composition(gap, examples, budget=8, examined_limit=16)
