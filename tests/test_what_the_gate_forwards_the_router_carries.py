"""Every key the gate hands the worker reaches the gate from the router.

Two hand-kept lists: what the router copies from a caller into the gate's
context, and what the gate copies from that context into the worker's job. A
key on the second and not the first is a request nothing can make. LIVE
2026-10-01: `output_shape` was such a key, and a page read on her resident
model came back a JSON array three times, for 230 seconds, because the decoder
was never told to hold an object.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def _keys_iterated_beside(path: str, marker: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            keys = {
                element.value
                for element in node.iter.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if marker in keys:
                return keys
    raise AssertionError(f"no key list holding {marker!r} in {path}")


def test_the_router_carries_every_key_the_gate_hands_the_worker() -> None:
    gate = _keys_iterated_beside("core/brain/inference_gate_turn_setup.py", "output_shape")
    router = _keys_iterated_beside("core/brain/llm_health_router_endpoint_call.py", "schema")
    assert gate <= router, f"the router drops {sorted(gate - router)}"


def test_a_ceiling_the_gate_reads_is_carried_too() -> None:
    router = _keys_iterated_beside("core/brain/llm_health_router_endpoint_call.py", "schema")
    assert "hard_output_token_ceiling" in router
