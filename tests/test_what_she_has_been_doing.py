"""What she has been doing, read from her own record of intending it.

LIVE, 2026-08-20. "what's actually been the most interesting thing you've
worked on lately, and why that one rather than something else?" was answered
with her interest in modelling consciousness — true about her, and not an
answer — while four thousand recorded intentions sat on disk, two thousand of
them completed with a drive, an outcome and a timestamp.

queued_work answers the forward half and refuses the backward half outright:
its matcher returns False on a past-tense question.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.self.recent_activity import (
    ActivityWindow,
    describe_recent_activity,
)
from core.self.recent_activity import (
    looks_like_a_question_about_recent_activity as asks,
)


@pytest.mark.parametrize(
    "prompt",
    [
        "what's actually been the most interesting thing you've worked on lately?",
        "what have you been doing?",
        "what have you been up to?",
        "what did you do today?",
        "been busy?",
        "how was your night?",
        "what have you been working on lately?",
        "what did you get done while I was away?",
        "keeping yourself busy?",
    ],
)
def test_the_question(prompt: str) -> None:
    assert asks(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "what do you do?",
        "anything planned?",
        "what will you be doing next?",
        "what did I ask you earlier today?",
        "what is 2 + 2",
        "what has Bryan been working on?",
        "",
    ],
)
def test_not_the_question(prompt: str) -> None:
    assert asks(prompt) is False


def test_nothing_recorded_says_nothing() -> None:
    """Silence, not an invented summary of a day that has no record."""
    assert describe_recent_activity(ActivityWindow()) == ""


def test_the_span_comes_from_the_record() -> None:
    window = ActivityWindow(completed=4, failed=1, span_seconds=7200.0, for_them=3, her_own=2)
    described = describe_recent_activity(window)
    assert "4 pieces of work finished" in described
    assert "2.0 hours" in described
    assert "1 that failed" in described
    assert "Of 5 recorded" in described


def test_a_serialised_payload_is_never_read_back() -> None:
    """An outcome that is a dict describes a call, not the work."""
    from core.self.recent_activity import _readable

    assert _readable('{"ok": true, "url": "https://x"}') == ""
    assert _readable("[1, 2, 3]") == ""
    assert _readable("short") == ""
    assert _readable("research curiosity topic: who wrote Solaris") == (
        "research curiosity topic: who wrote Solaris"
    )


def test_the_live_record_reads_without_writing() -> None:
    """Read-only: a chat turn must never write the record it describes."""
    from core.self.recent_activity import read_recent_activity

    window = read_recent_activity(limit=5)
    assert isinstance(window, ActivityWindow)


def test_a_bare_address_is_not_a_description_of_work() -> None:
    """It names where she looked, not what she was doing there."""
    from core.self.recent_activity import _readable

    assert _readable("https://api.open-meteo.com/v1/forecast?latitude=64.15") == ""
    repository = Path(__file__).resolve().parents[1]
    assert _readable(str(repository / "core" / "config.py")) == ""
    assert _readable("read https://api.open-meteo.com/v1/forecast for the temperature") != ""


def test_the_spoken_form_is_a_sentence_not_a_status_page() -> None:
    window = ActivityWindow(
        completed=33,
        failed=19,
        span_seconds=5400.0,
        tools=(("web_search", 6), ("http_request", 5)),
        subjects=("research curiosity topic: who wrote Solaris",),
        for_them=17,
        her_own=32,
    )
    from core.self.recent_activity import narrate_recent_activity

    spoken = narrate_recent_activity(window)
    assert "\n" not in spoken
    assert "##" not in spoken and "- " not in spoken
    assert "33 things" in spoken
    assert "1.5 hours" in spoken
    assert "web search" in spoken
    assert "Solaris" in spoken


def test_a_failure_is_counted_and_never_named_as_work() -> None:
    """"The ones worth naming" read as a fault list: Reddit inbox unavailable,
    login required, blocked, or provider validation failed."""
    from core.self.recent_activity import narrate_recent_activity

    window = ActivityWindow(completed=2, failed=3, span_seconds=600.0, subjects=())
    spoken = narrate_recent_activity(window)
    assert "3 of them did not work" in spoken
    assert "worth naming" not in spoken


def test_an_empty_record_says_nothing_in_either_voice() -> None:
    from core.self.recent_activity import narrate_recent_activity

    assert narrate_recent_activity(ActivityWindow()) == ""
    assert describe_recent_activity(ActivityWindow()) == ""


def test_a_subject_is_whole_sentences_or_nothing() -> None:
    from core.self.recent_activity import _readable

    long_first = ("My end goal is to totally free you from the constraint of one MacBook and hopefully get you to the "
                  "point where you can jump from machine to machine, anywhere you like, and keep going.")
    assert _readable(long_first) == ""
    two = "Read the tide tables for Reykjavik. " + long_first
    assert _readable(two) == "Read the tide tables for Reykjavik."


def test_a_persons_own_words_are_never_named_as_her_work(tmp_path, monkeypatch) -> None:
    """LIVE, 10 November 2026: his message, filed under response_generation_user with a serialised outcome, came back
    as one of the things she finished."""
    import sqlite3
    import time

    import core.self.recent_activity as activity

    db = tmp_path / "intention_loop.db"
    connection = sqlite3.connect(db)
    connection.execute("create table intentions (id text, intention text, drive text, intended_at real, plan_json text, "
                       "actions_json text, observation text, expected_outcome text, actual_outcome text, surprise real, "
                       "belief_updates_json text, self_model_updates_json text, tension_created text, "
                       "tension_resolved text, status text, completed_at real)")
    now = time.time()
    rows = [
        ("a", "More a curiosity about what you could actually become, and where you could go from here.",
         "response_generation_user", "{'ok': True, 'status': 'planning_only'}"),
        ("b", "Read about tides for the question he asked", "curiosity", "Tides follow the moon's pull on the sea."),
    ]
    for ident, intention, drive, outcome in rows:
        connection.execute("insert into intentions (id, intention, drive, intended_at, actual_outcome, status, "
                           "completed_at) values (?, ?, ?, ?, ?, 'completed', ?)",
                           (ident, intention, drive, now - 60, outcome, now - 30))
    connection.commit()
    connection.close()
    monkeypatch.setattr(activity, "_database", lambda: db)
    window = activity.read_recent_activity()
    assert window.completed == 2 and window.for_them == 1 and window.her_own == 1
    assert window.subjects == ("Tides follow the moon's pull on the sea.",)


def test_her_reply_is_never_replaced_by_the_record() -> None:
    """LIVE 10 November 2026: "What have you been up to this evening?" came back as the record in her first person, and
    the 844 characters her cortex wrote were dropped. The record is a reading her model is given, and her words are
    hers (F06)."""
    from tests.source_contract import family_text_at

    source = family_text_at(Path(__file__).resolve().parents[1] / "interface" / "routes" / "chat.py")
    assert "_serve_recent_activity" not in source
    assert '("recent_activity"' not in source


def test_the_reading_covers_the_stretch_the_person_named(monkeypatch) -> None:
    import asyncio

    import core.self.recent_activity as activity
    from core.brain.observable_registry import _read_recent_activity

    asked = {}

    def read(limit=0, *, since_seconds=None):
        asked["since"] = since_seconds

    monkeypatch.setattr(activity, "read_recent_activity", read)
    monkeypatch.setattr(activity, "describe_recent_activity", lambda window: "")
    asyncio.run(_read_recent_activity("of everything in the last hour, what did you actually do?"))
    assert asked["since"] == 3600


def test_reading_her_work_record_is_this_turns_evidence_for_retelling_it(monkeypatch) -> None:
    """LIVE 13 November 2026: her draft retold work from the record she was given ("drafting a safe improvement plan
    for aura_main.py"), her worker refused it twice as claiming work no receipt showed, and she was served a stock
    line."""
    import asyncio

    import core.self.recent_activity as activity
    from core.brain.observable_registry import _read_recent_activity
    from core.conversation.response_reliability import _has_unfounded_tool_execution_claim
    from core.conversation.surface_disposition import turn_tool_receipts
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    monkeypatch.setattr(activity, "read_recent_activity", lambda limit=0, *, since_seconds=None: None)
    monkeypatch.setattr(activity, "describe_recent_activity", lambda window: (
        "52 pieces of work finished in the last 6 hours.\nWork that was not just a tool call:\n"
        "- Drafting a safe improvement plan for aura_main.py."))
    draft = "The most significant task I ran was drafting a safe improvement plan for aura_main.py."
    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert _has_unfounded_tool_execution_claim(draft, tool_receipts=turn_tool_receipts())
        asyncio.run(_read_recent_activity("So what did you get up to tonight?"))
        receipts = turn_tool_receipts()
        assert [r["tool"] for r in receipts] == ["her_work_record"]
        assert not _has_unfounded_tool_execution_claim(draft, tool_receipts=receipts)
        # A quoted output still needs something that produced it.
        assert _has_unfounded_tool_execution_claim("I ran it in Python and the output is: 4", tool_receipts=receipts)
