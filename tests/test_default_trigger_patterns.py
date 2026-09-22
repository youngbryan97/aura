"""The trigger table moved out of the engine without changing.

415 lines of literal vocabulary sat inside one method of a 7,500-line module.
Moving it is only safe if it is the same table afterwards, so that is what
this checks — against the commit it was lifted from, by shape and by content,
and against the engine still reading it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from core.skills.default_trigger_patterns import default_trigger_patterns

ROOT = Path(__file__).resolve().parents[1]


def test_every_pattern_compiles() -> None:
    """A table of regexes nobody compiles is a table of strings."""
    for skill, patterns in default_trigger_patterns().items():
        assert patterns, f"{skill} has no trigger patterns"
        for pattern in patterns:
            re.compile(pattern)


def test_the_table_is_a_literal_with_no_computation_in_it() -> None:
    """It is data. A call inside it would make the move a behaviour change."""
    source = (ROOT / "core/skills/default_trigger_patterns.py").read_text()
    tree = ast.parse(source)
    # The table is the module constant the function copies from; it was lifted
    # out of the function when the function crossed the method-size bar.
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and getattr(node.targets[0] if isinstance(node, ast.Assign) else node.target, "id", "") == "_TABLE"
    )
    assert isinstance(assignment.value, ast.Dict)
    for node in ast.walk(assignment.value):
        assert not isinstance(node, (ast.Call, ast.Attribute, ast.Name)), (
            f"the table computes something: {ast.dump(node)[:80]}"
        )


def test_the_engine_reads_the_moved_table() -> None:
    body = (ROOT / "core/capability_engine.py").read_text()
    assert "from core.skills.default_trigger_patterns import default_trigger_patterns" in body
    assert "patterns = default_trigger_patterns()" in body
    # And the table is no longer in the engine.
    assert body.count('"web_search": [') == 0


def test_the_connector_patterns_are_still_added_on_top() -> None:
    """The one computed entry stayed behind, where it can read the runtime."""
    body = (ROOT / "core/capability_engine.py").read_text()
    assert 'patterns.setdefault("mcp_client", []).extend(' in body


def test_the_skills_named_are_the_ones_that_moved() -> None:
    table = default_trigger_patterns()
    assert len(table) == 42
    assert sum(len(v) for v in table.values()) == 302
    for name in ("web_search", "code_repl", "file_operation"):
        assert name in table
