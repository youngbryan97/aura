"""A method lifted into a new module arrives annotated.

A lift moves a method out of a grandfathered God object into a module that is
not grandfathered, and `make typed-surface` refuses a new module that is
untyped — correctly, because the lift moved the debt rather than paying it.
One lift of twenty-five clusters left 174 such modules and 724 unannotated
definitions, and the gate found them all at once, afterwards.

The types were recoverable the whole time: the call sites the lift leaves
behind still name the same locals, and a caller's annotated parameter says
what the helper's parameter is. That is what `tools/annotate_extracted_seams`
reads, and what both lift tools now run before they write.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from annotate_extracted_seams import (  # noqa: E402
    _annotation_is_writable,
    _names_in_scope,
    _return_annotation,
    _type_of_value,
    plan,
)


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_a_helper_takes_the_callers_annotation(tmp_path: Path) -> None:
    caller = tmp_path / "engine.py"
    caller.write_text(
        "from typing import Any\n"
        "from .engine_lifted import _step\n"
        "def turn(context: dict[str, Any], origin: str) -> None:\n"
        "    _step(context, origin)\n"
    )
    lifted = tmp_path / "engine_lifted.py"
    lifted.write_text(
        "from typing import Any\n"
        "def _step(context, origin):\n"
        "    return bool(context) and bool(origin)\n"
    )
    rewritten, counted = plan(lifted)
    assert "context: dict[str, Any]" in rewritten
    assert "origin: str" in rewritten
    assert counted["resolved"] == 2
    assert counted["any"] == 0


def test_a_name_the_new_module_cannot_write_becomes_any(tmp_path: Path) -> None:
    """An annotation copied from a caller can name a class this file has not
    imported, which is an F821 the moment it is written."""
    caller = tmp_path / "engine.py"
    caller.write_text(
        "from somewhere import AuraState\n"
        "from .engine_lifted import _step\n"
        "def turn(state: AuraState) -> None:\n"
        "    _step(state)\n"
    )
    lifted = tmp_path / "engine_lifted.py"
    lifted.write_text("def _step(state):\n    return state\n")
    rewritten, _ = plan(lifted)
    assert "AuraState" not in rewritten
    assert "state: Any" in rewritten


def test_a_function_that_returns_nothing_says_so() -> None:
    node = _function("def f(a):\n    a.append(1)\n", "f")
    assert _return_annotation(node, {}) == "None"


def test_a_tuple_return_carries_each_shape() -> None:
    source = (
        "def f(a):\n"
        "    name = str(a)\n"
        "    ok = bool(a)\n"
        "    return name, ok\n"
    )
    assert _return_annotation(_function(source, "f"), {"name": "str", "ok": "bool"}) == (
        "tuple[str, bool]"
    )


def test_a_generator_gets_no_return_type_from_this() -> None:
    """It would be wrong, and a wrong annotation is worse than none."""
    node = _function("def f(a):\n    yield a\n", "f")
    assert _return_annotation(node, {}) == ""


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("str(x)", "str"),
        ("bool(x)", "bool"),
        ("{'a': 1}", "dict[str, Any]"),
        ("[1, 2]", "list[Any]"),
        ("x > 1", "bool"),
        ("x.strip()", "str"),
        ("some_call(x)", ""),
    ],
)
def test_what_an_expression_says_about_itself(expression: str, expected: str) -> None:
    value = ast.parse(expression, mode="eval").body
    assert _type_of_value(value) == expected


def test_the_body_is_proved_unchanged(tmp_path: Path) -> None:
    """The whole guarantee: annotations went in and nothing else moved."""
    lifted = tmp_path / "lifted.py"
    original = (
        "from typing import Any\n"
        "def _step(a, b):\n"
        "    total = int(a) + int(b)\n"
        "    if total > 3:\n"
        "        return total\n"
        "    return 0\n"
    )
    lifted.write_text(original)
    rewritten, _ = plan(lifted)
    assert rewritten != original

    def bodies(source: str) -> list[str]:
        tree = ast.parse(source)
        return [
            ast.dump(ast.Module(body=node.body, type_ignores=[]))
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]

    assert bodies(rewritten) == bodies(original)


def test_both_lift_tools_run_the_annotator() -> None:
    for name in ("lift_methods.py", "lift_module_functions.py"):
        body = (ROOT / "tools" / name).read_text()
        assert "_annotate_what_moved(out, p)" in body
        assert "from annotate_extracted_seams import plan" in body


def test_a_signature_a_framework_reads_is_left_alone(tmp_path: Path) -> None:
    """Annotating a FastAPI endpoint's bare parameter builds a request field."""
    lifted = tmp_path / "routes.py"
    lifted.write_text(
        "from typing import Any\n"
        "app = object()\n"
        "@app.get('/thing')\n"
        "def read_thing(thing_id):\n"
        "    return thing_id\n"
    )
    rewritten, counted = plan(lifted)
    assert "thing_id: " not in rewritten
    assert counted["skipped"] == 1


def test_names_in_scope_sees_what_the_module_imports() -> None:
    scope = _names_in_scope(ast.parse("from a.b import Thing\nimport c.d\n"))
    assert "Thing" in scope
    assert "c" in scope
    assert _annotation_is_writable("Thing | None", scope)
    assert not _annotation_is_writable("Missing", scope)
