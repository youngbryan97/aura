"""Duration estimates cannot authorize a weaker model or a shorter answer."""

import ast
import inspect
from pathlib import Path

from core.brain import inference_gate


def test_answer_clock_cannot_change_the_selected_cortex():
    """The clock prices the turn. It does not get to pick a weaker model.

    Taken as the clock's own `if` statement rather than as the text between
    two markers. Those two were `_clock_blocked_by =` and the serving-lane
    assignment, and the method-size work moved the serving lane ABOVE the
    clock — so `index` searched forward for something now behind it and
    raised ValueError rather than failing an assertion. The statement is the
    unit; where it sits in the method is not the point.
    """
    source = Path(inspect.getfile(inference_gate)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "_clock_blocked_by" in ast.dump(node.test)
    ]
    assert blocks, "the answer clock's own branch is gone, not merely moved"

    assignments = [
        target.id
        for block in blocks
        for node in ast.walk(block)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert "requested_tier" not in assignments
    assert "answer_clock_demoted_from_primary" not in source


def test_foreground_budget_assignment_requires_a_larger_allowance():
    tree = ast.parse(Path(inspect.getfile(inference_gate)).read_text(encoding="utf-8"))
    writes = [
        node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name) and node.value.id == "_affordable"
    ]
    assert len(writes) == 1
    owner = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and writes[0] in node.body)
    assert ast.unparse(owner.test) == "_affordable > max_tokens"
