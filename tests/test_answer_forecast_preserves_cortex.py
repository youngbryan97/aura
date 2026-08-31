"""Duration estimates cannot authorize a weaker model or a shorter answer."""

import ast
import inspect
from pathlib import Path

from core.brain import inference_gate


def test_answer_clock_cannot_change_the_selected_cortex():
    source = Path(inspect.getfile(inference_gate)).read_text(encoding="utf-8")
    start = source.index("_clock_blocked_by =")
    end = source.index("serving_lane = self._cortex_serving_lane", start)
    tree = ast.parse(inspect.cleandoc(source[start:end]))
    assignments = [
        target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
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
