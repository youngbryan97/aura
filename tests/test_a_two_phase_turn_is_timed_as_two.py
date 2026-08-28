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

from pathlib import Path

_GATE = Path("core/brain/inference_gate.py")


def test_the_clock_runs_for_either_entitlement() -> None:
    body = _GATE.read_text()
    assert "if 0 < _answer_floor_final or _generations > 1:" in body


def test_the_phase_count_is_decided_before_the_clock_is_gated() -> None:
    """It was computed inside the block it now helps open."""

    body = _GATE.read_text()
    decided = body.index("if points_at_something_real(initial_visible_user_prompt)")
    gated = body.index("if 0 < _answer_floor_final or _generations > 1:")
    assert decided < gated, "the phase count must be known before it can open the gate"


def test_the_extension_is_still_measured_rather_than_invented() -> None:
    """An unmeasured decode rate must extend nothing, as before."""

    body = _GATE.read_text()
    start = body.index("if 0 < _answer_floor_final or _generations > 1:")
    window = body[start : start + 400]
    assert "_decode_s = _seconds_to_decode(max_tokens)" in window
    assert "if _decode_s > 0.0:" in window
