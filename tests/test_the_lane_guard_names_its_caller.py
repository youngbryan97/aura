"""A lock taken inside a context manager reports the manager, not the caller.

LIVE 2026-08-29: 117 blocking holds on the event loop, the longest 88ms against
a 50ms limit, every one reported as "taken at contextlib.py:137". Eight blocks
in mlx_client use that guard and the splat named none of them, so the
measurement said the loop had stalled and could not say where.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SOURCE = Path("core/brain/llm/mlx_client.py")


def _guard_calls() -> list[ast.Call]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_request_lane_state_guard"
            ):
                found.append(call)
    return found


def test_every_use_of_the_guard_names_where_it_is() -> None:
    calls = _guard_calls()
    assert calls, "the guard should still be in use"
    unnamed = [c.lineno for c in calls if not c.args]
    assert not unnamed, f"lane-state guard used without a site at lines {unnamed}"


def test_the_names_are_distinct_so_a_splat_points_somewhere() -> None:
    sites = [
        c.args[0].value
        for c in _guard_calls()
        if c.args and isinstance(c.args[0], ast.Constant)
    ]
    assert len(sites) == len(set(sites)), f"duplicate guard sites: {sites}"


def test_nothing_logs_while_holding_the_lane_state_lock() -> None:
    """The file sink JSON-wraps and redacts every record — that is real work."""

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "_request_lane_state_guard"
            for item in node.items
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "logger"
                and inner.func.attr in ("info", "warning", "error")
            ):
                offenders.append(inner.lineno)
    assert not offenders, f"logging under the lane-state lock at lines {offenders}"
