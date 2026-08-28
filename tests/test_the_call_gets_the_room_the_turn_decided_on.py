"""A tool call cut off mid-argument is not a refusal to call.

LIVE, 2026-08-28: "read the docs at <path>, then use it" read three files, said
"Running it now:", and emitted a code_repl call whose argument stopped inside
`from ledgerkit imp`. The turn's own clock had allocated 1536 tokens for
decoding and every generation inside the tool loop received 399, so the
narration and the opening of the program together reached the ceiling.

What arrived downstream was prose containing tool scaffolding, which is exactly
what it looked like and exactly what the reply gate refuses. She had decided to
run the code. The room to say so was missing, and the room had already been
worked out one function up.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from core.brain.inference_gate import InferenceGate


def test_the_loop_is_offered_the_turns_budget() -> None:
    signature = inspect.signature(InferenceGate._tool_grounded_answer)
    assert "decode_budget" in signature.parameters


def test_the_budget_travels_from_the_clock_to_the_call() -> None:
    """Both halves, because either alone leaves the default in place."""

    body = Path("core/brain/inference_gate.py").read_text()
    # Handed down from the caller, which is where the clock's figure lives.
    assert "decode_budget=int(max_tokens or 0)" in body
    # And forwarded into the loop rather than dropped.
    assert '{"max_tokens": int(decode_budget)}' in body


def test_nothing_is_passed_when_there_is_nothing_to_pass() -> None:
    """A zero budget must leave the client's own default alone.

    Passing max_tokens=0 would be worse than passing nothing: it reads as a
    request for no output at all.
    """

    body = Path("core/brain/inference_gate.py").read_text()
    start = body.index('{"max_tokens": int(decode_budget)}')
    window = body[start : start + 160]
    assert "if int(decode_budget or 0) > 0" in window
    assert "else {}" in window


def test_a_call_carrying_code_is_allowed_a_document() -> None:
    """The other half of the same budget, already there and worth holding.

    A tool whose argument is a program needs room for a program. This asserts
    the rule rather than the number, because the number is the runtime's own.
    """

    from core.brain.llm.mlx_client import _tool_call_budget, _tools_can_carry_a_document

    carries = {
        "code_repl": {"parameters": {"properties": {"code": {"type": "string"}}}}
    }
    plain = {
        "file_operation": {"parameters": {"properties": {"path": {"type": "string"}}}}
    }
    assert _tools_can_carry_a_document(carries)
    assert not _tools_can_carry_a_document(plain)
    assert _tool_call_budget(399, 4096, carries) == 4096
    assert _tool_call_budget(399, 4096, plain) == 399
