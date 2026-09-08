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
    """The `if` that decides whether the deadline is extended for decoding.

    Found in the tree rather than by matching source text. The first version of
    this looked for the literal
    ``if 0 < _answer_floor_final or _generations > 1:`` and went red when a
    third entitlement was added in front of it — a change that widened the gate,
    which is the direction this file exists to protect.
    """
    tree = ast.parse(_GATE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            inner.id for inner in ast.walk(node.test) if isinstance(inner, ast.Name)
        }
        if {"_answer_floor_final", "_generations"} <= names:
            return node
    raise AssertionError("no gate tests both the completion floor and the phase count")


def test_the_clock_runs_for_either_entitlement() -> None:
    gate = _the_clock_gate()
    # Either, not both: an `or` at the top, so one entitlement is enough.
    assert isinstance(gate.test, ast.BoolOp) and isinstance(gate.test.op, ast.Or)


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

    gate = _the_clock_gate()
    inside = ast.unparse(ast.Module(body=gate.body, type_ignores=[]))
    # What is decoded is measured, not assumed. The argument is no longer
    # max_tokens — a thinking model adds a reasoning reserve on the far side of
    # this calculation, so the clock has to pay for more than the gate asked
    # for — and what this test is about is that a rate is measured at all.
    assert "_seconds_to_decode(" in inside
    assert "_decode_s > 0.0" in inside, "an unmeasured rate must extend nothing"
