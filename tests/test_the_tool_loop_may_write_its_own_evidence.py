"""The turn's own tool loop is not an ambient child.

Turn evidence custody identifies an execution by (thread, task). The loop that
runs a turn's tools does not always run under the identity that opened the
turn — a task or a thread somewhere in the dispatch is enough — and when it
does not, every receipt it writes is refused:

    🔧 Tool Dispatch: file_operation (origin=unknown)
    🧾 tool receipt for file_operation refused: this execution is not an
       admitted participant of the turn holding the evidence
    🧾 nothing to report from this turn's tools: custody=present admits=True

Live on 2026-08-28 that turn read three files and reported nothing.

The rule is right — an ambient child must not write into somebody else's turn
— and there is a lease for a child that is not ambient.
"""

from __future__ import annotations

import asyncio

from core.conversation.turn_evidence_custody import (
    current_turn_evidence_custody,
    bind_turn_evidence_custody,
    run_as_turn_evidence_participant,
)
from core.conversation.surface_disposition import record_tool_receipt


def test_a_child_task_cannot_write_evidence_without_a_lease() -> None:
    """The rule this fix must not weaken."""

    async def scenario() -> bool:
        with bind_turn_evidence_custody(session_id="s", turn_id="t"):
            async def child() -> bool:
                return record_tool_receipt("file_operation", ok=True, action="read")

            # A task of its own is a different execution, whoever made it.
            return await asyncio.ensure_future(child())

    assert asyncio.run(scenario()) is False


def test_the_turns_own_loop_may_write_with_a_lease() -> None:
    """What the gate now does around think_and_act."""

    async def scenario() -> bool:
        with bind_turn_evidence_custody(session_id="s", turn_id="t"):
            async def child() -> bool:
                return record_tool_receipt("file_operation", ok=True, action="read")

            return await asyncio.ensure_future(
                run_as_turn_evidence_participant(child(), purpose="tool loop")
            )

    assert asyncio.run(scenario()) is True


def test_the_receipt_actually_lands_where_the_turn_can_read_it() -> None:
    """Admitting the child is only useful if the evidence is then there."""

    async def scenario() -> int:
        with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
            async def child() -> bool:
                return record_tool_receipt(
                    "file_operation",
                    ok=True,
                    action="read",
                    object_ref="/tmp/ledgerkit/API.md",
                )

            await asyncio.ensure_future(
                run_as_turn_evidence_participant(child(), purpose="tool loop")
            )
            return len(custody.receipts())

    assert asyncio.run(scenario()) == 1
