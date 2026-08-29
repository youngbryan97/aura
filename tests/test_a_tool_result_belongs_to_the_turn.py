"""What a tool returned belongs to the turn however the loop that ran it ends.

LIVE 2026-08-29: six tool turns, a code_repl that ran and produced the answer
on turn six, and then the loop ran out of time. Receipts were written for the
whole batch after the loop returned, so a loop that did not return wrote none —
the salvage that reports what the tools found logged "custody=present
admits=True" with nothing to report, and the person got the canned apology on
top of six successful tool calls.
"""

from __future__ import annotations

import pytest

from core.conversation.surface_disposition import (
    begin_turn_tool_receipts,
    record_tool_receipt,
    turn_tool_receipts,
)
from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

pytestmark = pytest.mark.unit


def test_the_same_call_recorded_twice_is_one_call() -> None:
    """Written when the tool returns, and again when the loop finishes."""

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        for _ in range(2):
            assert record_tool_receipt(
                "code_repl",
                ok=True,
                action="execute",
                object_ref="{'code': 'print(1)'}",
                effect_observed=True,
                observed_content="Accounts Receivable 25000",
            )
        assert len(turn_tool_receipts()) == 1


def test_two_different_calls_stay_two() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        record_tool_receipt(
            "code_repl", ok=True, object_ref="{'code': 'a'}", observed_content="1"
        )
        record_tool_receipt(
            "code_repl", ok=True, object_ref="{'code': 'b'}", observed_content="2"
        )
        assert len(turn_tool_receipts()) == 2


def test_the_same_call_with_a_different_result_is_a_different_receipt() -> None:
    """A tool retried after a failure is two facts about the turn, not one."""

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        record_tool_receipt(
            "code_repl", ok=False, object_ref="{'code': 'x'}", observed_content="ImportError"
        )
        record_tool_receipt(
            "code_repl", ok=True, object_ref="{'code': 'x'}", observed_content="Revenue -25000"
        )
        assert len(turn_tool_receipts()) == 2


def test_the_loop_writes_a_receipt_when_the_tool_returns() -> None:
    """Not when the loop ends, which is the case that lost the work."""

    from core.brain.llm.mlx_client import _record_tool_receipt_for_this_turn

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        _record_tool_receipt_for_this_turn(
            "code_repl",
            {"code": "print(ledger.trial_balance())"},
            {"ok": True, "status": "ok"},
            "{'Accounts Receivable': 25000, 'Revenue': -25000}",
        )
        (receipt,) = turn_tool_receipts()
        assert receipt["tool"] == "code_repl"
        assert receipt["ok"] is True
        assert "25000" in receipt["observed_content"]


def test_a_refused_tool_is_recorded_as_refused_not_as_done() -> None:
    from core.brain.llm.mlx_client import _record_tool_receipt_for_this_turn

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        _record_tool_receipt_for_this_turn(
            "code_repl",
            {"code": "x"},
            {"ok": False, "status": "denied", "error": "governance"},
            "denied",
        )
        (receipt,) = turn_tool_receipts()
        assert receipt["ok"] is False


def test_recording_never_ends_a_turn() -> None:
    """A receipt is bookkeeping; it must not be able to fail the work."""

    from core.brain.llm.mlx_client import _record_tool_receipt_for_this_turn

    class _Explodes:
        def __str__(self) -> str:
            raise RuntimeError("no")

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()
        _record_tool_receipt_for_this_turn("code_repl", _Explodes(), {"ok": True}, "x")
