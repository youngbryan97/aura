"""Source spans and execution dependencies use different orders."""

from types import SimpleNamespace

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import (
    _calibrate_operation_charts,
    _OperationNode,
)


def test_cataphoric_execution_order_does_not_count_as_span_failure():
    first = _OperationNode(TokenSpan(1, 2), "sub", 6.0, 6.0, 1.0)
    second = _OperationNode(TokenSpan(7, 8), "at", 6.0, 6.0, 1.0)
    instructions = (
        SimpleNamespace(op="at", operation_span=second.span),
        SimpleNamespace(op="sub", operation_span=first.span),
    )
    item = SimpleNamespace(ir=SimpleNamespace(instructions=instructions))
    by_count = ((6.0, (first,)), (12.0, (first, second)))
    penalty, rows = _calibrate_operation_charts([(item, by_count)])
    winner = next(row for row in rows if row["length_penalty"] == penalty)
    assert winner["graph_exact"] == winner["span_exact"] == winner["operation_exact"] == 1
    assert item.ir.instructions is instructions


def test_wrong_opcode_remains_a_failure_after_order_normalization():
    node = _OperationNode(TokenSpan(1, 2), "add", 6.0, 6.0, 1.0)
    item = SimpleNamespace(ir=SimpleNamespace(instructions=(SimpleNamespace(op="sub", operation_span=node.span),)))
    _, rows = _calibrate_operation_charts([(item, ((6.0, (node,)),))])
    assert all(row["span_exact"] == 1 for row in rows)
    assert all(row["operation_exact"] == row["graph_exact"] == 0 for row in rows)
