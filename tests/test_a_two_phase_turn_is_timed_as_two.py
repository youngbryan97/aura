"""Needing two generations is a different fact from needing a long answer.

The clock that extends a turn's deadline to cover its own decoding sat entirely
behind a completion floor — an entitlement that says "this ANSWER must be
long". Needing two generations says "this TURN has two phases": one to make the
call and one to say what came back. A turn with the second and not the first
was timed as though it had one phase.

LIVE, 2026-08-28: the same request ran twice. With a floor it was given 516
seconds and read three files. Without one it was given 148, its tool loop was
squeezed to the floor below which no call completes, and the answer — over a
prompt its own worker measured at 120 seconds to read — was cancelled with
nothing said.
"""

from __future__ import annotations

import ast
from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def _the_clock_gate() -> ast.If:
    """The `if` that opens the clock, found by what it tests.

    Held on the condition rather than on its text. The condition has since
    been widened to cover any user-facing turn, which is a superset of both
    entitlements, and a test pinning the old spelling failed over the change
    that made the clock run more often.
    """
    tree = ast.parse(_GATE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        named = {
            inner.id for inner in ast.walk(node.test) if isinstance(inner, ast.Name)
        }
        if {"_answer_floor_final", "_generations"} <= named:
            return node
    raise AssertionError("nothing gates the clock on the floor and the phase count")


def test_the_clock_runs_for_either_entitlement() -> None:
    gate = _the_clock_gate()
    assert isinstance(gate.test, ast.BoolOp) and isinstance(gate.test.op, ast.Or)
    said = {ast.unparse(one) for one in gate.test.values}
    assert "0 < _answer_floor_final" in said
    assert "_generations > 1" in said


def test_the_phase_count_is_decided_before_the_clock_is_gated() -> None:
    """It was computed inside the block it now helps open."""

    body = _GATE.read_text()
    decided = body.index("if points_at_something_real(initial_visible_user_prompt)")
    gated = _the_clock_gate().lineno
    assert body[:decided].count("\n") + 1 < gated, (
        "the phase count must be known before it can open the gate"
    )


def test_the_extension_is_still_measured_rather_than_invented() -> None:
    """An unmeasured decode rate must extend nothing, as before."""

    inside = "\n".join(ast.unparse(one) for one in _the_clock_gate().body)
    # Not `_seconds_to_decode(max_tokens)`: the worker adds the reasoning
    # reserve on the far side of this, so the clock was sized for the tokens
    # asked for while up to twice that many were decoded against it.
    assert "_decode_s = _seconds_to_decode(_tokens_to_pay_for)" in inside
    assert "_tokens_to_pay_for = max_tokens + _reserve_the_worker_adds" in inside
    assert "if _decode_s > 0.0:" in inside
