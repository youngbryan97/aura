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
    import time

    now = time.time()
    with recording_turn("t1", phases_available=("a",)) as receipt:
        record_latency("queue", 0.8, ended_at=now - 40.0)  # the wait came before the work
        record_latency("prefill", 15.66)
        record_latency("decode", 20.22)
        record_latency("tool", 1.5)
        record_latency("tool", 2.5)
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
    import time

    with recording_turn("turn-a", phases_available=["P"]):
        record_latency("queue", 0.5, ended_at=time.time() - 10.0)  # waits before the work they wait for
        record_latency("retrieval", 1.0)
        record_latency("prefill", 2.0)
    with recording_turn("turn-b", phases_available=["P"]):
        record_latency("queue", 1.5, ended_at=time.time() - 10.0)
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


def test_a_finished_turn_says_its_split_in_her_log(caplog) -> None:
    """R11: the five read off a live turn. The health route served them as an empty block on 10 November."""
    import logging

    from core.verify.turn_receipt import record_latency, recording_turn

    with caplog.at_level(logging.INFO, logger="Aura.TurnReceipt"):
        with recording_turn("turn-r11-0001", phases_available=()):
            record_latency("prefill", 1.25)
            record_latency("decode", 3.5)
    said = [r.getMessage() for r in caplog.records if "took" in r.getMessage()]
    assert len(said) == 1
    assert "decode=3.50s prefill=1.25s" in said[0]
    assert "unmeasured: delivery,queue,retrieval,tool" in said[0]


def test_her_desktop_turn_records_its_retrieval_and_not_a_tool_twice(monkeypatch) -> None:
    """LIVE 2026-11-10: every turn on her desktop lane, which runs no retrieval phase, said "unmeasured: retrieval" with
    nine to thirteen seconds unattributed, two of them her web check. A search run as a tool keeps its own seconds."""
    import asyncio

    import core.conversation.what_the_world_says as world
    import interface.routes.chat_desktop_evidence as evidence_route

    async def gathered(*_a, **_k):
        await asyncio.sleep(0.05)
        record_latency("tool", 0.03)  # a search the gathering ran as a tool
        return world.WorldEvidence()

    async def checked(*_a, **_k):
        await asyncio.sleep(0.05)
        return []

    async def nothing_earlier(**_k):
        return []

    monkeypatch.setattr(world, "gather_world_evidence", gathered)
    monkeypatch.setattr(world, "her_check_reads", checked)
    monkeypatch.setattr(evidence_route._chat_memory_state, "_recent_completed_conversation_exchanges", nothing_earlier)
    monkeypatch.setattr(evidence_route, "_should_collect_desktop_required_search_evidence", lambda _m: (False, "", None))
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    with bind_turn_evidence_custody(session_id="s", turn_id="t-desktop"):  # as her chat route binds a turn
        receipt = current_receipt()
        asyncio.run(evidence_route._what_the_world_says_for_the_turn("Is that true?", "Is that true?", session_id="s"))
    assert receipt.latency_s["tool"] == pytest.approx(0.03)
    assert 0.07 <= receipt.latency_s["retrieval"] < 0.5


def test_her_thinking_joins_the_turn_from_message_to_delivery() -> None:
    """LIVE 2026-11-10: the receipt began when her thinking did; her web check before it and the gates and shaping after
    it were outside the turn it timed. One receipt now, and what follows her answer is delivery."""
    import time

    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    with bind_turn_evidence_custody(session_id="s", turn_id="t-whole"):
        outer = current_receipt()
        record_latency("retrieval", 0.02)
        with recording_turn("think-1", phases_available=("Retrieval", "Response")) as joined:
            record_latency("prefill", 0.01)
            record_latency("decode", 0.03)
        time.sleep(0.05)
    assert joined is outer and outer.turn_id == "t-whole"
    assert outer.phases_available == ("Retrieval", "Response")
    assert set(outer.latency_s) == {"retrieval", "prefill", "decode", "delivery"}
    assert 0.05 <= outer.latency_s["delivery"] < 0.5
    record_latency("decode", 9.0)  # after the turn: not its time
    assert outer.latency_s["decode"] == pytest.approx(0.03)


def test_a_turn_that_ran_nothing_is_not_said_as_a_split(caplog) -> None:
    """LIVE 2026-11-10: four turns closed at once, each "took 0.00s: nothing measured"."""
    import logging

    from core.verify.turn_receipt import recording_turn

    with caplog.at_level(logging.DEBUG, logger="Aura.TurnReceipt"):
        with recording_turn("turn-r11-0002", phases_available=()):
            pass
    said = [(r.levelno, r.getMessage()) for r in caplog.records if "turn-r11-000" in r.getMessage()]
    assert said == [(logging.DEBUG, said[0][1])]
    assert "without running a phase" in said[0][1]


@pytest.mark.asyncio
async def test_the_wait_for_her_model_is_the_turns_queue(monkeypatch) -> None:
    """LIVE 11 November 2026: voice turns during her boot took 168 s each with 0.3 s measured; the rest was the wait
    for her foreground lane, recorded nowhere."""
    import asyncio

    from interface.routes import chat_foreground_lane

    class _Gate:
        async def ensure_foreground_ready(self, timeout):  # noqa: ASYNC109 - the gate's own signature
            await asyncio.sleep(0.05)
            return {"conversation_ready": True}

    with recording_turn("t-admission", phases_available=()) as receipt:
        await chat_foreground_lane._admit_to_foreground_lane(
            _remaining_foreground_budget=lambda reserve: 30.0, gate=_Gate(), lane={})
    assert receipt.latency_s.get("queue", 0.0) >= 0.05


def test_a_generation_that_never_finished_still_says_what_it_cost() -> None:
    """LIVE 11 November 2026: voice turns during her boot took 170 s with under 2 s measured; the rest was a generation
    that never finished, and only a finished one recorded its prefill and decode."""
    import time
    from types import SimpleNamespace

    from core.brain.llm.mlx_client_recording import _RecordsWhatTheWorkerDid

    now = time.time()
    unfinished = SimpleNamespace(_current_request_started_at=now - 30.0, _current_first_token_at=now - 10.0)
    with recording_turn("t-unfinished", phases_available=()) as receipt:
        _RecordsWhatTheWorkerDid._record_an_unfinished_generations_time(unfinished, "r1")
    assert receipt.latency_s["prefill"] == pytest.approx(20.0, abs=0.5)
    assert receipt.latency_s["decode"] == pytest.approx(10.0, abs=0.5)

    no_token = SimpleNamespace(_current_request_started_at=now - 5.0, _current_first_token_at=0.0)
    with recording_turn("t-no-token", phases_available=()) as receipt:
        _RecordsWhatTheWorkerDid._record_an_unfinished_generations_time(no_token, "r2")
    assert receipt.latency_s["prefill"] == pytest.approx(5.0, abs=0.5) and "decode" not in receipt.latency_s

    measured = SimpleNamespace(_current_request_started_at=now - 5.0, _current_first_token_at=now - 1.0,
                               _latency_recorded_request="r3")
    with recording_turn("t-measured", phases_available=()) as receipt:
        _RecordsWhatTheWorkerDid._record_an_unfinished_generations_time(measured, "r3")
    assert receipt.latency_s == {}, "a finished generation's worker already said it, once"



def test_work_inside_a_wait_is_counted_once_under_its_own_name() -> None:
    """LIVE 11 November 2026, two typed turns: her gate's queue held a read of her local corpus and a one-token
    generation, and delivery held the generation that repaired her reply; each was also recorded as prefill and
    decode, so the second turn's parts added up past its 52 s."""
    import time

    now = time.time()
    with recording_turn("t-overlap", phases_available=()) as receipt:
        record_latency("queue", 3.31, ended_at=now - 20.0)  # entry to the cortex's call
        record_latency("retrieval", 1.2, ended_at=now - 21.0)  # the corpus read inside it
        record_latency("prefill", 0.4, ended_at=now - 20.4)  # the one-token pass inside it
        record_latency("decode", 0.1, ended_at=now - 20.3)
        record_latency("prefill", 7.0, ended_at=now - 6.0)
        record_latency("decode", 6.0, ended_at=now)
    assert receipt.latency_s["queue"] == pytest.approx(3.31 - 1.2 - 0.4 - 0.1, abs=0.01)
    assert receipt.latency_s["prefill"] == pytest.approx(7.4) and receipt.latency_s["retrieval"] == 1.2
