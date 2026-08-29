"""A generation offered tools may answer with a call, and that is not a bad answer.

LIVE 2026-08-29. Asked to read a library's docs and then use it, turn four of
the tool loop emitted a complete, correct code_repl call — the right library,
the right arguments, the invoice posted the right way round. The surface
quality gate read it as prose, called it a prompt artifact because it is angle
brackets rather than sentences, and the generation returned nothing. The turn
ended on "I couldn't get to an answer I'd stand behind" after four successful
tool calls.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import _text_is_a_complete_tool_call

pytestmark = pytest.mark.unit

A_REAL_CALL = (
    "<tool_call>\n"
    "<function=code_repl>\n"
    "<parameter=code>\n"
    "from ledgerkit import Ledger\n"
    'ledger = Ledger("aura-hosting")\n'
    'ledger.post(date="2026-03-01", debit="Accounts Receivable", '
    'credit="Revenue", amount_cents=25000)\n'
    "print(ledger.trial_balance())\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def test_the_call_the_gate_threw_away_is_recognised() -> None:
    assert _text_is_a_complete_tool_call(A_REAL_CALL)


def test_a_typed_parameter_the_schema_does_not_name_does_not_disqualify_it() -> None:
    """The schema validator owns unknown properties; this only reads the shape."""

    with_language = A_REAL_CALL.replace(
        "<parameter=code>",
        "<parameter=language>\npython\n</parameter>\n<parameter=code>",
    )
    assert _text_is_a_complete_tool_call(with_language)


def test_prose_with_a_call_inside_it_is_still_prose() -> None:
    """A draft that talks first is a draft, and judging it as one is correct."""

    assert not _text_is_a_complete_tool_call(
        "Let me try that again with the module's own directory:\n\n" + A_REAL_CALL
    )


@pytest.mark.parametrize(
    "not_a_call",
    [
        "",
        "   ",
        "Here is the trial balance: Accounts Receivable 25000, Revenue -25000.",
        "<tool_call>\n{\"name\": ",
        "<tool_call>\n<function=code_repl>\n<parameter=code>\nunclosed",
        "<function=code_repl></function>",
    ],
)
def test_anything_that_is_not_one_complete_call_is_not_one(not_a_call: str) -> None:
    assert not _text_is_a_complete_tool_call(not_a_call)


def test_the_loop_and_the_gate_read_a_call_the_same_way() -> None:
    """Same parser, so the two cannot disagree about what a call is."""

    from core.brain.llm.mlx_client import _native_xml_tool_payload

    start = A_REAL_CALL.index("<tool_call>") + len("<tool_call>")
    payload, _why = _native_xml_tool_payload(
        A_REAL_CALL, start=start, tool_definitions=None
    )
    assert payload is not None
    assert payload["name"] == "code_repl"
    assert "Accounts Receivable" in payload["arguments"]["code"]
    assert _text_is_a_complete_tool_call(A_REAL_CALL)
