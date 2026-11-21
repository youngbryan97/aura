"""How far each turn's answer got is on its receipt, kept across restarts, and counted apart (U07-11).

An outside review of 19 November 2026: of the first outside benchmark's 20 missed questions, 17 ended with no usable
answer, so the score measured whether an answer reached the end as much as whether it was right. Reasoning and
delivery are counted apart here: drafted, checked, committed, delivered, and why a turn that served no answer stopped.
"""
from __future__ import annotations

import json

import pytest

from core.verify import turn_receipt
from core.verify.turn_receipt import STAGES, record_reached, record_stopped, recording_turn

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _kept_lines(monkeypatch):
    kept: list[str] = []
    import core.runtime.file_write_gateway as gateway

    monkeypatch.setattr(gateway, "append_text_behind", lambda path, text, **_k: kept.append(text) or True)
    turn_receipt.reset_turn_receipts_for_test()
    return kept


def test_each_stage_is_kept_once_with_its_seconds_and_unknown_ones_are_not(_kept_lines):
    with recording_turn("t-stages", phases_available=()) as receipt:
        record_reached("drafted")
        record_reached("drafted")
        record_reached("checked")
        record_reached("polished")
    assert list(receipt.reached) == ["drafted", "checked"]
    assert receipt.reached["drafted"] <= receipt.reached["checked"]
    record_reached("delivered")  # outside any turn
    assert "delivered" not in receipt.reached


def test_a_finished_turn_is_kept_as_one_line_with_how_far_it_got(_kept_lines):
    with recording_turn("t-kept", phases_available=()):
        record_reached("drafted")
        record_stopped("cognitive_engine_failed")
    [line] = _kept_lines
    row = json.loads(line)
    assert row["turn_id"] == "t-kept" and set(row["reached"]) == {"drafted"}
    assert row["stopped"] == "cognitive_engine_failed" and row["started_at"] > 0


def test_a_turn_that_did_nothing_is_not_kept(_kept_lines):
    with recording_turn("t-empty", phases_available=()):
        pass
    assert _kept_lines == []


def test_what_was_served_says_delivered_or_why_it_stopped(_kept_lines):
    from core.conversation.reply_provenance import ReplyProvenance, declare_provenance
    from interface.routes.chat_http_shapes import _how_far_it_got

    with recording_turn("t-served", phases_available=()) as served:
        _how_far_it_got("It prints 7.", "ok")
    assert "delivered" in served.reached and served.stopped == ""
    with recording_turn("t-nothing", phases_available=()) as nothing:
        _how_far_it_got("", "foreground_busy")
    assert "delivered" not in nothing.reached and nothing.stopped == "foreground_busy"
    with recording_turn("t-failure", phases_available=()) as failure:
        text = "I couldn't get to an answer I'd stand behind."
        declare_provenance(text, ReplyProvenance.HONEST_FAILURE)
        _how_far_it_got(text, "cognitive_engine_failed")
    assert "delivered" not in failure.reached and failure.stopped == "cognitive_engine_failed"


def test_the_chat_route_marks_each_stage_where_it_is_reached():
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    for stage in ("drafted", "checked", "committed"):
        assert f'record_reached("{stage}")' in source
    assert set(STAGES) == {"drafted", "checked", "committed", "delivered"}


def test_the_metrics_count_each_stage_apart(tmp_path):
    from tools.delivery_metrics import as_text, read_kept, summary

    rows = [
        {"started_at": 10.0, "reached": {"drafted": 40.0, "checked": 44.0, "committed": 45.0, "delivered": 45.5}},
        {"started_at": 20.0, "reached": {"drafted": 60.0}, "stopped": "deadline_exceeded"},
        {"started_at": 30.0, "reached": {"delivered": 3.0}},
        {"started_at": 40.0, "reached": {}, "stopped": "foreground_busy"},
        {"started_at": 50.0, "reached": {}},  # a background tick of her own
    ]
    kept = tmp_path / "turn_receipts.jsonl"
    kept.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not json\n", encoding="utf-8")
    measured = summary(read_kept(kept))
    assert measured["turns"] == 4 and measured["her_own_turns"] == 1
    assert measured["reached"] == {"drafted": 2, "checked": 1, "committed": 1, "delivered": 2}
    assert measured["median_s"] == {"to_first_draft": 50.0, "checking_the_draft": 4.0, "to_delivered": 24.25}
    assert measured["delivered_with_no_draft_recorded"] == 1
    assert measured["stopped"] == {"deadline_exceeded": 1, "foreground_busy": 1}
    assert summary(read_kept(kept, since=25.0))["turns"] == 2
    assert as_text(measured).startswith("Turns a person asked: 4 (and 1 of her own, not counted)")
    assert "Stopped, deadline_exceeded: 1" in as_text(measured)


def test_a_call_inside_a_call_is_counted_once(_kept_lines):
    """LIVE 20 November 2026: a desktop turn of 4,197 s said "tool=8321.30s", her desktop task's call holding the calls
    it made inside it, each added to the one around it."""
    import time

    from core.verify.turn_receipt import record_latency

    with recording_turn("t-nested", phases_available=()) as receipt:
        now = time.time()
        record_latency("tool", 2.0, ended_at=now - 5.0)  # a call inside the one below
        record_latency("tool", 1.0, ended_at=now - 4.5)  # and another inside it
        record_latency("tool", 6.0, ended_at=now - 4.0)  # the call around both
        record_latency("tool", 1.0, ended_at=now)  # a later call of its own
    assert receipt.latency_s["tool"] == pytest.approx(7.0)
    assert receipt.latency_samples["tool"] == 4
