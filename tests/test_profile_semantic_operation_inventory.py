"""Source-only diagnostics do not confuse span and label pruning."""

from types import SimpleNamespace

from core.learning.semantic_program_ir import TokenSpan
from tools.profile_semantic_operation_inventory import classify_inventory, signature_rank


def _item():
    return SimpleNamespace(ir=SimpleNamespace(instructions=[
        SimpleNamespace(operation_span=(0, 1), op="mul"),
        SimpleNamespace(operation_span=(3, 4), op="idiv")]))


def _node(span, operation):
    return SimpleNamespace(span=span, operation=operation)


def test_source_span_pruning_is_not_label_pruning():
    ranked = [_node((0, 1), "add"), _node((3, 4), "idiv")]
    complete = ranked + [_node((0, 1), "mul")]
    assert classify_inventory(_item(), ranked, complete)["operation_nodes"] == [
        "operation_label_pruned", "operation_node_retained"]
    assert classify_inventory(_item(), ranked[1:], complete)["operation_nodes"] == [
        "source_span_pruned", "operation_node_retained"]


def test_target_span_outside_declared_inventory_is_separate():
    result = classify_inventory(_item(), [_node((3, 4), "idiv")],
                                [_node((3, 4), "idiv")])
    assert result["operation_nodes"] == ["source_span_outside_declared_inventory",
                                         "operation_node_retained"]
    assert result["all_target_nodes_retained"] is False


def test_signature_rank_is_explicitly_a_bounded_inventory():
    item = SimpleNamespace(public_inputs=(2, 3), ir=SimpleNamespace(instructions=[
        SimpleNamespace(op="mul"), SimpleNamespace(op="idiv")]))
    model = SimpleNamespace(inference_step_limit=lambda _n: 2,
                            operation_length_penalty=0.,
                            register_use_contract=SimpleNamespace(
                                input_min_uses=0, input_max_uses=3,
                                intermediate_min_uses=0, intermediate_max_uses=3,
                                distinct_arguments=False))
    nodes = [SimpleNamespace(span=TokenSpan(0, 1), operation="mul", score=1.),
             SimpleNamespace(span=TokenSpan(2, 3), operation="idiv", score=1.)]
    result = signature_rank(model, item, nodes, max_table_entries=100)
    assert result["signature_inventory_complete"] is True
    assert result["signature_rank"] is not None
    truncated = signature_rank(model, item, nodes, max_table_entries=1)
    assert truncated["signature_inventory_complete"] is False
    assert truncated["signature_rank"] is None
