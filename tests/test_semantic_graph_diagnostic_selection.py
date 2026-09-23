"""Held-out graph diagnosis must stay separate from source-only fitting."""

from types import SimpleNamespace

import pytest

from tools.diagnose_semantic_graph_counterexamples import (
    runtime_bank_diagnostic,
    select_diagnostic_examples,
)


def row(split, source):
    return SimpleNamespace(split=split, ir=SimpleNamespace(source_text_sha256=source))


def test_frozen_heldout_identity_selection():
    examples = (row("train", "a"), row("validation", "b"), row("test", "c"))
    assert select_diagnostic_examples(examples, split="test", source_ids=("c",)) == examples[2:]
    assert select_diagnostic_examples(examples, split="train", source_ids=("a",)) == examples[:1]


@pytest.mark.parametrize("split", ("validation", "test"))
def test_heldout_diagnosis_cannot_fit_or_reuse_source_capacity(split):
    examples = (row(split, "a"),)
    with pytest.raises(ValueError, match="cannot fit"):
        select_diagnostic_examples(examples, split=split, fit_steps=1)
    with pytest.raises(ValueError, match="cannot fit"):
        select_diagnostic_examples(examples, split=split, capacity_report="source.json")


def test_diagnostic_source_ids_must_match_selected_split():
    examples = (row("train", "a"), row("test", "b"))
    with pytest.raises(ValueError, match="missing"):
        select_diagnostic_examples(examples, split="test", source_ids=("a",))
    with pytest.raises(ValueError, match="duplicate"):
        select_diagnostic_examples(examples, split="test", source_ids=("b", "b"))
    with pytest.raises(ValueError, match="no observations"):
        select_diagnostic_examples(examples, split="validation")


def test_runtime_bank_search_receives_observation_only():
    source_span = SimpleNamespace(to_dict=lambda: {"start": 2, "end": 3})
    instruction = SimpleNamespace(operation_span=source_span, op="mul")
    ir = SimpleNamespace(source_token_ids=(4, 5), source_text_sha256="source",
                         model_basis_receipt_sha256="basis", instructions=(instruction,))
    item = SimpleNamespace(ir=ir, hidden_states="hidden", public_inputs=(2, 3))
    selected = SimpleNamespace(ir=SimpleNamespace(instructions=(instruction,)))
    bank = SimpleNamespace(selected=selected, receipt={"charts": [
        {"operations": [{"op": "mul", "span": {"start": 2, "end": 3}}]}]},
        validate=lambda: None)
    calls = []
    model = SimpleNamespace(decode_candidates=lambda **kwargs: (calls.append(kwargs) or bank))
    report = runtime_bank_diagnostic(model, item, max_charts=4, max_graphs=2,
                                     solve_time_limit_s=1.)
    assert report["source_operation_chart_reached"]
    assert report["selected_operation_spans_match_source"]
    assert report["source_labels_sent_to_candidate_search"] is False
    assert calls == [{"source_token_ids": (4, 5), "hidden_states": "hidden",
                      "public_inputs": (2, 3), "source_text_sha256": "source",
                      "model_basis_sha256": "basis", "max_charts": 4,
                      "max_graphs_per_chart": 2, "solve_time_limit_s": 1.}]
