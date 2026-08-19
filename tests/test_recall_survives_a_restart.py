"""What she was asked before the restart is on disk, so she can answer from it.

LIVE, 2026-08-19. "what did i ask you about earlier today, before you
restarted? be specific." was answered:

    You asked about my cognitive architecture and whether I could articulate
    it clearly.

Nothing of the sort had been asked. That day's turns were about running
Python, an arithmetic product, and naming a position she had dropped — and
every one of them was sitting in the episodic store, timestamped, while the
answer was invented.

``_user_turns`` had three sources: the caller's history buffer, live working
memory, and the transcript singleton. All three are process-local, so after a
restart every one is empty and the question has no source at all. Durable
memory written on every turn and never read by the reading that needs it —
which costs her the one thing a person notices immediately about whether
something remembers them.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from core.conversation.durable_turns import (
    describe_durable_turns,
    durable_user_turns,
)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "episodic.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE episodes (episode_id TEXT, timestamp REAL, context TEXT, "
        "action TEXT, outcome TEXT)"
    )
    now = time.time()
    rows = [
        ("1", now - 7200, "User asked: run a tiny bit of python", "a", "b"),
        ("2", now - 3600, "User asked: what is 7919 * 6367?", "a", "b"),
        ("3", now - 1800, "Since my last awakening, 2 commits by Codex", "a", "b"),
        ("4", now - 600, "User asked: name one position you dropped", "a", "b"),
        ("5", now - 86400 * 9, "User asked: something from last week", "a", "b"),
    ]
    connection.executemany("INSERT INTO episodes VALUES (?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def test_the_turns_come_back_in_the_order_they_happened(store: Path):
    turns = durable_user_turns(path=store)
    assert [turn.text for turn in turns] == [
        "run a tiny bit of python",
        "what is 7919 * 6367?",
        "name one position you dropped",
    ]


def test_only_what_the_person_said_counts(store: Path):
    """The store holds her own awakening notes too; those are not his turns."""
    assert all("awakening" not in turn.text for turn in durable_user_turns(path=store))


def test_a_time_window_keeps_earlier_today_from_becoming_everything(store: Path):
    within_a_day = durable_user_turns(path=store, within_s=86400.0)
    assert all("last week" not in turn.text for turn in within_a_day)
    week = durable_user_turns(path=store, within_s=86400.0 * 30)
    assert any("last week" in turn.text for turn in week)


def test_each_turn_carries_when_it_was_said(store: Path):
    """"Be specific" was the request; a turn with no time answers half of it."""
    for turn in durable_user_turns(path=store):
        assert turn.when() != "at an unrecorded time"


def test_a_missing_store_reads_as_nothing(tmp_path: Path):
    assert durable_user_turns(path=tmp_path / "absent.db") == ()


def test_a_corrupt_store_does_not_break_the_turn(tmp_path: Path):
    broken = tmp_path / "episodic.db"
    broken.write_bytes(b"this is not a database")
    assert durable_user_turns(path=broken) == ()


def test_the_reading_is_empty_rather_than_wrong_when_there_is_nothing(tmp_path: Path):
    from core.conversation import durable_turns as module

    original = module._episodic_path
    module._episodic_path = lambda: tmp_path / "absent.db"
    try:
        assert describe_durable_turns() == ""
    finally:
        module._episodic_path = original


def test_the_cascade_reaches_the_durable_store_last():
    """The three sources above it are process-local; this one is not."""
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1] / "core/conversation/grounded_recall.py").read_text()
    order = source.index("turns = _history_user_turns")
    working = source.index("_working_memory_user_turns(exclude_norm)", order)
    transcript = source.index("_transcript_user_turns(exclude_norm)", order)
    durable = source.index("_durable_user_turns(exclude_norm)", order)
    assert order < working < transcript < durable


def test_a_question_that_says_it_reaches_past_the_session_is_recognised():
    """The cascade is first-non-empty, so one greeting shadows the whole day.

    LIVE, 2026-08-19: "what did i ask you about earlier today, before you
    restarted? be specific." was answered from a transcript holding one "hi
    again", by inventing a topic. Session scope is right for positional recall
    — "what was my first question?" means this conversation — and wrong for a
    question that says out loud it is asking about before.
    """
    from core.brain.observable_registry import _reaches_past_this_session

    for beyond in (
        "what did i ask you about earlier today, before you restarted?",
        "what did we talk about yesterday?",
        "what was I asking in the previous session?",
        "you restarted — do you remember what we were doing?",
        "what did I say last time?",
    ):
        assert _reaches_past_this_session(beyond), beyond

    for within in (
        "what was my first question?",
        "what did I just ask you?",
        "what did I ask you two messages ago?",
        "how are you doing",
    ):
        assert not _reaches_past_this_session(within), within


def test_the_earlier_conversation_is_composed_from_the_record(monkeypatch, store: Path):
    """Evidence informs; it does not enforce.

    With the durable turns in its prompt, the model answered "what did i ask
    you about earlier today" with the immediately previous message. Before the
    turns reached it at all, it invented topics. The record holds the answer
    exactly, with times, so the answer is composed rather than requested.
    """
    from core.conversation import durable_turns as module

    monkeypatch.setattr(module, "_episodic_path", lambda: store)
    composed = module.earlier_conversation_answer()
    assert "Earlier today you asked me:" in composed
    assert "run a tiny bit of python" in composed
    assert "what is 7919 * 6367?" in composed
    # Her own notes are not his turns.
    assert "awakening" not in composed


def test_the_turn_being_answered_is_not_quoted_back(monkeypatch, store: Path):
    from core.conversation import durable_turns as module

    monkeypatch.setattr(module, "_episodic_path", lambda: store)
    composed = module.earlier_conversation_answer(exclude="what is 7919 * 6367?")
    assert "7919" not in composed


def test_the_conversation_happening_now_is_not_earlier(monkeypatch, tmp_path: Path):
    """A turn from a minute ago is this conversation, not earlier today."""
    import sqlite3
    import time as _time

    from core.conversation import durable_turns as module

    path = tmp_path / "episodic.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE episodes (episode_id TEXT, timestamp REAL, context TEXT, "
        "action TEXT, outcome TEXT)"
    )
    connection.execute(
        "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
        ("1", _time.time() - 60, "User asked: something just now", "a", "b"),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(module, "_episodic_path", lambda: path)
    assert module.earlier_conversation_answer() == ""


def test_only_a_question_that_asks_for_it_is_served():
    from interface.routes.chat import _serve_earlier_conversation

    for unrelated in ("what is 2 + 2", "run some python", "how are you"):
        assert _serve_earlier_conversation(unrelated, "the model's reply") == "the model's reply"
