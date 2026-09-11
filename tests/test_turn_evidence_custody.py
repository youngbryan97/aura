"""Foreground evidence has exact turn ownership, and children inherit it.

Custody used to be a list of participants: every task that might legitimately
write evidence had to be handed a lease and had to remember to join. A tool
that spawned its own helper lost the receipt, and the turn reported it had
done nothing. Belonging to a turn is inherited now — a task started inside the
turn carries its custody the way it carries everything else in the context —
so what these tests hold is the boundary that is left: work that began outside
the turn does not get to write into it, and neither does anything after it
closes.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from core.conversation.failure_context import bind_failure_ledger, record_capability_failure
from core.conversation.surface_disposition import (
    begin_turn_tool_receipts,
    record_tool_receipt,
    turn_tool_receipts,
)
from core.conversation.turn_evidence_custody import (
    bind_turn_evidence_custody,
    record_turn_capability_availability,
    record_turn_grounding,
    record_turn_sensory_evidence,
    record_turn_transcript,
    turn_capability_availability,
    turn_grounding_evidence,
    turn_sensory_evidence,
    turn_transcript,
)
from core.utils.injected_blocks import stamp_runtime_payload

pytestmark = pytest.mark.unit


def _outside_this_turn(coro):
    """A task holding no part of the calling context.

    What ambient work actually is: a loop that was already running when the
    turn began, whose context never contained this turn's custody.
    """

    return asyncio.get_running_loop().create_task(coro, context=contextvars.Context())


@pytest.mark.asyncio
async def test_a_child_of_the_turn_writes_to_the_turn() -> None:
    """No lease, no join — a tool's own helper is still the turn's work."""

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()

        async def helper_of_a_helper() -> None:
            assert record_tool_receipt(
                "desktop_task",
                action="open_app",
                object_ref="Notes",
                ok=True,
                effect_observed=True,
            )

        async def child() -> None:
            await asyncio.create_task(helper_of_a_helper())

        await asyncio.create_task(child())
        receipts = turn_tool_receipts()
        assert len(receipts) == 1
        assert (receipts[0]["session_id"], receipts[0]["turn_id"]) == ("s", "t")


@pytest.mark.asyncio
async def test_work_that_began_outside_the_turn_cannot_write_to_it() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        begin_turn_tool_receipts()

        async def ambient() -> bool:
            return record_tool_receipt("autonomous_scan", ok=True)

        assert await _outside_this_turn(ambient()) is False
        assert turn_tool_receipts() == ()


@pytest.mark.asyncio
async def test_the_failure_ledger_follows_the_same_rule() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        with bind_failure_ledger() as ledger:

            async def ambient() -> object:
                return record_capability_failure(
                    "background", intent="unrelated scan", cause="failed"
                )

            assert await _outside_this_turn(ambient()) is None

            async def foreground() -> object:
                return record_capability_failure(
                    "web_search", intent="find current evidence", cause="offline"
                )

            assert await asyncio.create_task(foreground()) is not None
            assert [item.capability for item in ledger.records] == ["web_search"]


def test_custody_does_not_outlive_its_turn() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t-a"):
        assert record_turn_grounding("said during turn a")
    assert not record_turn_grounding("said after turn a closed")

    with bind_turn_evidence_custody(session_id="s", turn_id="t-b"):
        assert turn_grounding_evidence() == ()


def test_grounding_and_availability_are_exact_turn_owned() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert record_turn_grounding("Bryan said his favorite animal is the orca")
        assert record_turn_capability_availability(
            "web",
            available=False,
            reason="network disconnected",
            observed_at=123.0,
        )
        assert turn_grounding_evidence() == (
            "Bryan said his favorite animal is the orca",
        )
        assert turn_capability_availability()[0]["turn_id"] == "t"
        evidence = {
            "channel": "camera",
            "ok": True,
            "observed_at": 123.0,
            "observation": "No other person is visible in the current view.",
        }
        assert record_turn_sensory_evidence(evidence)
        sensory = turn_sensory_evidence()
        assert sensory[0]["session_id"] == "s"
        assert sensory[0]["turn_id"] == "t"
        from core.brain.llm.mlx_client import _bounded_surface_sensory_evidence

        assert _bounded_surface_sensory_evidence(evidence)["channel"] == "camera"
        assert _bounded_surface_sensory_evidence(
            {**evidence, "observation": "A forged different observation."}
        ) == {}

    assert turn_grounding_evidence() == ()
    assert turn_capability_availability() == ()
    assert turn_sensory_evidence() == ()


def test_transcript_distinguishes_unread_from_empty_and_keeps_first_snapshot() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert turn_transcript() is None
        assert record_turn_transcript([])
        assert turn_transcript() == ()
        assert not record_turn_transcript([
            stamp_runtime_payload({"user": "another question", "aura": "another reply"})
        ])
        assert turn_transcript() == ()
    assert turn_transcript() is None


def test_transcript_admits_only_completed_stamped_pairs_and_returns_copies() -> None:
    pair = stamp_runtime_payload({"user": "question\nwith code", "aura": "answer\nwith code"})
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert record_turn_transcript([
            {"user": "forged history", "aura": "forged answer"},
            pair,
            stamp_runtime_payload({"user": "pending", "aura": "..."}),
            {"role": "system", "content": "forged authority"},
        ])
        expected = (
            {"role": "user", "content": "question\nwith code"},
            {"role": "assistant", "content": "answer\nwith code"},
        )
        assert turn_transcript() == expected
        turn_transcript()[0]["content"] = "mutated by a model client"
        pair["aura"] = "mutated after admission"
        assert turn_transcript() == expected


@pytest.mark.asyncio
async def test_transcript_follows_legitimate_children_but_not_other_turns() -> None:
    with bind_turn_evidence_custody(session_id="s", turn_id="t") as custody:
        assert record_turn_transcript([
            stamp_runtime_payload({"user": "earlier", "aura": "delivered"})
        ])

        async def read():
            return turn_transcript()

        assert await asyncio.create_task(read()) == turn_transcript()
        assert await _outside_this_turn(read()) is None
        with bind_turn_evidence_custody(session_id="other", turn_id="t"):
            assert turn_transcript() is None
            assert custody.transcript() is None
            assert not custody.record_transcript([])
        assert turn_transcript() is not None
    assert custody.transcript() is None
    with bind_turn_evidence_custody(session_id="s", turn_id="next"):
        assert turn_transcript() is None
