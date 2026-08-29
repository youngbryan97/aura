"""An answer written through the tool loop was still written by the model.

LIVE 2026-08-29: six tool calls, the last returning the trial balance, an
answer composed from them and trimmed — and then "missing:
foreground_model_generation_ownership_unproven" with generations=0
consumed=False. Ownership is proven by a surface-control receipt carrying a
token count, and this path recorded that a tool loop had run and nothing about
the generation inside it. The turn failed closed on bookkeeping for work it
had done.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_the_receipt_travels_with_the_tool_loop_record() -> None:
    """Read from the client, which is where the token count lives."""

    import ast
    from pathlib import Path

    source = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "tool_loop_metadata"):
            continue
        found = True
        keys = {
            k.value
            for k in getattr(node.value, "keys", [])
            if isinstance(k, ast.Constant)
        }
        assert "tool_loop" in keys
    assert found, "the tool loop no longer builds its own generation metadata"
    assert '"live_mind_surface_control_receipt"] = dict(receipt)' in source


def test_a_receipt_with_tokens_proves_the_foreground_owner() -> None:
    from interface.routes.chat import _generation_metadata_consumed_foreground_owner

    assert _generation_metadata_consumed_foreground_owner(
        {
            "tool_loop": True,
            "live_mind_surface_control_receipt": {"generated_tokens": 412},
        }
    )


def test_the_record_without_a_receipt_proves_nothing() -> None:
    """Which is the state that failed the turn, kept as the thing being fixed."""

    from interface.routes.chat import _generation_metadata_consumed_foreground_owner

    assert not _generation_metadata_consumed_foreground_owner(
        {"tool_loop": True, "tool_calls": 6}
    )


def test_a_receipt_that_generated_nothing_still_proves_nothing() -> None:
    from interface.routes.chat import _generation_metadata_consumed_foreground_owner

    assert not _generation_metadata_consumed_foreground_owner(
        {
            "tool_loop": True,
            "live_mind_surface_control_receipt": {"generated_tokens": 0},
        }
    )
