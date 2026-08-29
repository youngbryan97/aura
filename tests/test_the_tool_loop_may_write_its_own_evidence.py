"""Belonging to a turn is inherited, not enumerated.

Turn evidence custody used to require that an execution's (thread, task) pair
had been registered as a participant. That refused the turn's own children —
a tool loop, a repair pass, anything the turn started in a task of its own —
and a lease had to be threaded through by hand to undo it. Live on
2026-08-28 a turn read three files and told the person "nothing to report from
this turn's tools", because five receipts were written by executions nobody
had enumerated.

The set was also a second copy of a proof the runtime already had. Custody is
reachable only through a contextvar, and a contextvar is inherited by exactly
the children a turn starts. So the rule is now what the contextvar says, plus
the turn still being open — and these are the cases that matters for.
"""

from __future__ import annotations

import asyncio

from core.conversation.surface_disposition import record_tool_receipt
from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    current_turn_evidence_custody,
)


async def _write_one() -> bool:
    return record_tool_receipt(
        "file_operation", ok=True, action="read", object_ref="/tmp/ledgerkit/API.md"
    )


def test_a_child_the_turn_started_may_write() -> None:
    """The case that was broken, and the reason the whole turn said nothing."""

    async def scenario() -> bool:
        with bind_turn_evidence_custody(session_id="s", turn_id="t"):
            return await asyncio.ensure_future(_write_one())

    assert asyncio.run(scenario()) is True


def test_the_receipt_lands_where_the_turn_can_read_it() -> None:
    """Admitting the child is only worth anything if the evidence is there."""

    async def scenario() -> int:
        with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
            await asyncio.ensure_future(_write_one())
            return len(custody.receipts())

    assert asyncio.run(scenario()) == 1


def test_a_task_started_outside_the_turn_sees_nothing() -> None:
    """The boundary that does the real work.

    A contextvar is inherited by children of the context that set it. An
    execution started before the turn, or beside it, does not see the custody
    at all — so it cannot write into it, and no participant set is needed to
    say so.
    """

    async def scenario() -> tuple[bool, bool]:
        # Started before any custody exists, and kept running across one.
        started_outside = asyncio.get_running_loop().create_future()

        async def outsider() -> bool:
            await started_outside
            return record_tool_receipt("file_operation", ok=True, action="read")

        outside_task = asyncio.ensure_future(outsider())
        with bind_turn_evidence_custody(session_id="s", turn_id="t"):
            inside = await asyncio.ensure_future(_write_one())
            started_outside.set_result(None)
            outside = await outside_task
        return inside, outside

    inside, outside = asyncio.run(scenario())
    assert inside is True
    assert outside is False


def test_a_closed_turn_accepts_nothing() -> None:
    """A child that outlives its turn writes into a turn that has ended."""

    async def scenario() -> bool:
        with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
            keep = custody
        # The block has exited; the custody is closed.
        assert keep.closed is True
        return keep.admits_current_execution()

    assert asyncio.run(scenario()) is False


def test_another_turn_cannot_write_into_this_one() -> None:
    """The session and turn the execution runs under have to be this one's."""

    async def scenario() -> bool:
        with bind_turn_evidence_custody(session_id="s", turn_id="one") as first:
            with bind_turn_evidence_custody(session_id="s", turn_id="two"):
                # Inside turn two, turn one must refuse.
                return first.admits_current_execution()

    assert asyncio.run(scenario()) is False
