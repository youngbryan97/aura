"""Retaining source bindings must include alternate runtime operation interpretations."""

from dataclasses import replace

import pytest

from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
from core.learning.semantic_joint_graph_learning import align_source_input_registers, score_annotated_graph
from core.learning.semantic_runtime_graph_retention import mine_runtime_graph_constraints
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def trained():
    model, examples = model_examples()
    return model.with_joint_operation_argument_scores().with_source_ordered_definitions(), examples


def test_retention_uses_the_exact_runtime_grounding_and_operation_chart_inventory(trained):
    model, examples = trained
    item = next(row for row in examples if row.split == "train")
    outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256)
    assert outcome.ir is not None
    spans, _, _, candidates = model._runtime_operation_charts(item.ir.source_token_ids,
        item.hidden_states, item.public_inputs, model.inference_step_limit(len(item.public_inputs)))
    assert spans == outcome.ir.input_spans
    assert any(tuple((node.operation, node.span) for node in chart) == tuple(
        (ins.op, ins.operation_span) for ins in sorted(outcome.ir.instructions,
            key=lambda ins: (ins.operation_span.start, ins.operation_span.end))) for chart in candidates)


def test_correct_runtime_answer_still_retains_witnessed_operation_competitors(trained):
    model, examples = trained
    model = model.with_operation_label_alternatives(2)
    for item in examples:
        if item.split != "train":
            continue
        outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
            public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=model.model_basis_sha256)
        if outcome.ir is None:
            continue
        target, _ = align_source_input_registers(item, outcome.ir.input_spans)
        positive = score_annotated_graph(model, item, target, outcome.ir.input_spans)
        selected = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans)
        if compare_program_meanings(positive["program"], selected["program"],
                counterfactual_inputs(item.public_inputs))["status"] != "equivalent":
            continue
        constraints, record = mine_runtime_graph_constraints(model, item, max_charts=8)
        if constraints:
            assert all(row.positive_operations and row.negative_operations for row in constraints)
            assert not record["source_operations_supplied"] and not record["serving_authority"]
            assert any(row.get("negative_index") is not None and tuple(
                (node["op"], tuple(node["span"])) for node in row["operations"])
                != tuple((ins.op, (ins.operation_span.start, ins.operation_span.end)) for ins in target)
                for row in record["charts"])
            return
    pytest.fail("no correct source retained an alternative operation interpretation")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_rows_never_enter_runtime_retention(trained, split):
    model, examples = trained
    with pytest.raises(ValueError, match="source training"):
        mine_runtime_graph_constraints(model, replace(examples[0], split=split))


def test_partial_operation_inventory_is_not_reported_as_complete(trained):
    model, examples = trained
    item = next(row for row in examples if row.split == "train")
    _, record = mine_runtime_graph_constraints(model, item, max_charts=1, max_graphs=1)
    assert len(record["charts"]) == 1
    assert not record["operation_search_complete"] and not record["highest_incorrect_proven"]


@pytest.mark.parametrize("allowance", [0, -1, True])
def test_invalid_retention_allowances_fail(trained, allowance):
    model, examples = trained
    with pytest.raises(ValueError, match="allowances"):
        mine_runtime_graph_constraints(model, examples[0], max_charts=allowance)
