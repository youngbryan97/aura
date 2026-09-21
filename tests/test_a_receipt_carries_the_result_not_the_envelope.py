"""A tool result reaches the screen as what it found, at any budget.

LIVE 2026-09-20. "Read the file CLAUDE.md and tell me what the first rule
under 'The live instance is sacred' says." The read succeeded, the cortex
could not finish the turn, and what the fallback served was:

    file_operation returned:
    {"note":"Result exceeded the context budget; preview only.",
     "original_chars":11582,"preview":"{\\"authority_closure\\":{\\"closed\\":true…

The answer was inside that, four levels down. Three bounds in a row produced
it: the tool result is re-emitted as a valid envelope at 4000 characters for
the model, the receipt recorder cut that envelope at 2000, and the strip at
display time could only work on JSON that parses.
"""

from __future__ import annotations

import json

from core.brain.inference_gate import (
    _a_readable_field_of_broken_json,
    _what_a_tool_returned,
)
from core.brain.llm.mlx_client import _truncate_tool_result
from interface.routes.chat_reply_shaping import _readable_result

_RULE = "**Never kill, restart, or port-collide with it.**"

_A_FILE_READ = {
    "authority_closure": {"closed": True, "token_revoked": True},
    "content": f"0001: # CLAUDE.md\n0010: {_RULE}\n" + "0011: filler\n" * 800,
    "ok": True,
    "duration_ms": 6,
}


def _envelope() -> str:
    """What the model is handed for a result too big for its budget."""
    return _truncate_tool_result(json.dumps(_A_FILE_READ), limit=4000)


def test_the_envelope_is_valid_json_before_anything_cuts_it() -> None:
    parsed = json.loads(_envelope())
    assert parsed["truncated"] is True
    assert parsed["note"].startswith("Result exceeded")


def test_the_rule_survives_every_cut_the_recorders_make() -> None:
    envelope = _envelope()
    for limit in (2000, 1800, 1500, 1000, len(envelope)):
        shown = _readable_result(envelope[:limit])
        assert "authority_closure" not in shown, f"envelope on screen at {limit}"
        assert "token_revoked" not in shown
        if _RULE in envelope[:limit]:
            assert _RULE in shown, f"the rule was in the bytes and not on screen at {limit}"


def test_a_cut_that_reached_nothing_readable_says_nothing() -> None:
    """An empty receipt is dropped by the caller; an envelope is not."""
    envelope = json.dumps(
        {"note": "preview only", "preview": '{"authority_closure":{"clos'}
    )
    assert _readable_result(envelope[: len(envelope) - 4]) == ""


def test_the_reader_gets_inside_a_whole_envelope() -> None:
    assert _RULE in _what_a_tool_returned(json.loads(_envelope()))


def test_the_reader_gets_inside_a_cut_envelope() -> None:
    envelope = _envelope()
    assert _RULE in _a_readable_field_of_broken_json(envelope[:2000])


def test_a_plain_result_is_untouched() -> None:
    assert _readable_result("the file has 412 lines") == "the file has 412 lines"
    assert _what_a_tool_returned({"content": "hello"}) == "hello"


def test_the_receipt_carries_the_result_itself() -> None:
    """The recorder reads the result, not the string built for the model."""
    from core.brain.llm import mlx_client

    recorded: dict[str, object] = {}

    def _capture(name, **fields):
        recorded["name"] = name
        recorded.update(fields)

    import core.conversation.surface_disposition as surface

    original = surface.record_tool_receipt
    surface.record_tool_receipt = _capture
    try:
        mlx_client._record_tool_receipt_for_this_turn(
            "file_operation", {"path": "CLAUDE.md"}, _A_FILE_READ, _envelope()
        )
    finally:
        surface.record_tool_receipt = original

    observed = str(recorded.get("observed_content") or "")
    assert _RULE in observed
    assert "authority_closure" not in observed
