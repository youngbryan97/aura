"""Source-only diagnostics do not confuse span and label pruning."""

from types import SimpleNamespace

from tools.profile_semantic_operation_inventory import classify_inventory


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
