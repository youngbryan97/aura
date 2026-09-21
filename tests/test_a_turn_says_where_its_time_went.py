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


def test_the_split_can_be_read_from_outside_the_process() -> None:
    """R11 asks for the five measured SEPARATELY — which is only true if the
    split can be read. Each had a recorder and none of it left the process:
    the health route publishes a curated payload and this was not in it
    (2026-09-20), so the measurement existed and nobody outside could see it.
    """
    from core.verify.turn_receipt import (
        LATENCY_COMPONENTS,
        latency_by_component,
        record_latency,
        recording_turn,
        reset_turn_receipts_for_test,
    )

    reset_turn_receipts_for_test()
    with recording_turn("turn-a", phases_available=["P"]):
        record_latency("queue", 0.5)
        record_latency("retrieval", 1.0)
        record_latency("prefill", 2.0)
    with recording_turn("turn-b", phases_available=["P"]):
        record_latency("queue", 1.5)
        record_latency("decode", 4.0)

    said = latency_by_component()
    assert said["turns_read"] == 2
    # Each component reports on its own turns, not averaged over every turn.
    assert said["components"]["queue"] == {
        "turns": 2, "median_s": 1.0, "slowest_s": 1.5, "total_s": 2.0
    }
    assert said["components"]["prefill"]["turns"] == 1
    assert said["components"]["decode"]["slowest_s"] == 4.0
    # A component nobody measured is named as never measured, not shown as zero.
    assert set(said["never_measured"]) == LATENCY_COMPONENTS - {
        "queue", "retrieval", "prefill", "decode"
    }
    assert "tool" not in said["components"]
    reset_turn_receipts_for_test()


def test_the_health_payload_carries_the_split() -> None:
    """The route's payload is a whitelist; a measurement missing from it is
    unreadable however carefully it was taken."""
    import inspect

    from interface.routes import system

    source = inspect.getsource(system._runtime_integrity_public_payload)
    assert '"turn_latency"' in source

    from core.runtime import health_contract

    block = inspect.getsource(health_contract._integrity_of_taint_locks_and_custody)
    assert "latency_by_component" in block
