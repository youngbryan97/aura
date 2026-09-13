"""A turn's latency, split into the five components R11 named, on its receipt.

Prefill and decode were measured in the worker, retrieval was a phase
duration in a "budget exceeded" line, and tool time and queue time were not
written down anywhere a turn could be read back from. One place now. A
component nobody measured is absent, never zero — zero would read as instant.
"""

from __future__ import annotations

import pytest

from core.verify.turn_receipt import (
    LATENCY_COMPONENTS,
    current_receipt,
    record_latency,
    recording_turn,
    reset_turn_receipts_for_test,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_turn_receipts_for_test()
    yield
    reset_turn_receipts_for_test()


def test_the_five_components_and_delivery_are_the_whole_vocabulary() -> None:
    assert LATENCY_COMPONENTS == {"queue", "retrieval", "prefill", "decode", "tool", "delivery"}


def test_each_component_is_recorded_by_name_and_accumulates() -> None:
    with recording_turn("t1", phases_available=("a",)) as receipt:
        record_latency("prefill", 15.66)
        record_latency("decode", 20.22)
        record_latency("tool", 1.5)
        record_latency("tool", 2.5)
        record_latency("queue", 0.8)
    as_dict = receipt.as_dict()
    assert as_dict["latency_s"] == {"prefill": 15.66, "decode": 20.22, "tool": 4.0, "queue": 0.8}
    assert as_dict["latency_samples"]["tool"] == 2
    assert "retrieval" not in as_dict["latency_s"], "unmeasured is absent, not zero"


def test_an_unknown_component_is_refused_rather_than_invented() -> None:
    with recording_turn("t2", phases_available=()):
        with pytest.raises(ValueError):
            record_latency("thinking", 3.0)


def test_a_nonsense_span_is_ignored_and_a_missing_turn_is_fine() -> None:
    assert current_receipt() is None
    record_latency("prefill", 1.0)  # no turn: nothing to record on, no error
    with recording_turn("t3", phases_available=()) as receipt:
        record_latency("decode", float("nan"))
        record_latency("decode", -1.0)
        record_latency("decode", "soon")
    assert receipt.latency_s == {}


def test_what_nothing_accounted_for_is_derived_from_the_wall_clock() -> None:
    with recording_turn("t4", phases_available=()) as receipt:
        record_latency("prefill", 0.0)
    receipt.finished_at = receipt.started_at + 10.0
    receipt.latency_s = {"prefill": 3.0, "decode": 4.0}
    assert receipt.unattributed_s == 3.0
    assert receipt.as_dict()["unattributed_s"] == 3.0


@pytest.mark.asyncio
async def test_a_skill_writes_its_tool_time_on_the_turn() -> None:
    import asyncio

    from core.skills.base_skill import BaseSkill

    class _Sleeps(BaseSkill):
        name = "sleeps_a_little"
        description = "sleeps"
        requires_approval = False

        async def execute(self, params, context):
            await asyncio.sleep(0.05)
            return {"ok": True}

    with recording_turn("t5", phases_available=()) as receipt:
        await _Sleeps().safe_execute({}, {})
    assert receipt.latency_s.get("tool", 0.0) >= 0.05
    assert receipt.latency_samples.get("tool") == 1
