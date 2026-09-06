"""The log outlives the projections taken from it.

OpenHands keeps a canonical persisted causal history for its runtime. Letta
stores recall automatically and never mutates it: summaries and memories
reference ranges of experience and may change, the experience itself does not.

Aura's spine was append-only in memory, which is half of the property. A log
that dies with the process cannot be what a summary references, because the
range the summary names is gone before anybody reads it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.runtime.event_spine import EventLog, Lane


@pytest.fixture()
def kept_at():
    with tempfile.TemporaryDirectory() as where:
        yield Path(where) / "experience.jsonl"


def test_an_in_memory_log_still_works_and_says_it_is_not_durable():
    log = EventLog()
    log.append("something", {"a": 1})
    said = log.report()
    assert said["durable"] is False and said["kept_through"] == 0
    assert log.head == 1


def test_what_was_appended_comes_back_after_a_restart(kept_at):
    log = EventLog(kept_at=kept_at)
    for at in range(5):
        log.append("a_thing", {"n": at}, lane=Lane.SYSTEM, actor="test")
    assert log.kept_through == 5
    assert len(kept_at.read_text().splitlines()) == 5

    again = EventLog(kept_at=kept_at)
    assert again.head == 5
    assert [one.payload["n"] for one in again.events()] == [0, 1, 2, 3, 4]
    assert again.append("one more", {}).seq == 6, "the sequence did not continue"


def test_compaction_drops_from_memory_and_leaves_the_experience_alone(kept_at):
    """The property both findings are about, in one assertion."""

    log = EventLog(kept_at=kept_at)
    for at in range(5):
        log.append("a_thing", {"n": at})
    said = log.compact({"folded": "up"}, through=3)
    assert said["dropped"] == 3 and said["remaining"] == 2
    assert said["still_kept_through"] == 5
    assert len(kept_at.read_text().splitlines()) == 5, "compaction rewrote the log"
    assert len(EventLog(kept_at=kept_at).events()) == 5


def test_the_log_is_appended_never_rewritten(kept_at):
    log = EventLog(kept_at=kept_at)
    log.append("first", {})
    first_line = kept_at.read_text().splitlines()[0]
    for _ in range(20):
        log.append("more", {})
    assert kept_at.read_text().splitlines()[0] == first_line


def test_a_truncated_log_comes_back_short_rather_than_holed(kept_at):
    """A hole in the sequence is worse than a short log, because it is invisible."""

    log = EventLog(kept_at=kept_at)
    for at in range(4):
        log.append("a_thing", {"n": at})
    lines = kept_at.read_text().splitlines()
    kept_at.write_text("\n".join(lines[:2] + ["{not json"] + lines[2:]) + "\n")

    again = EventLog(kept_at=kept_at)
    assert [one.payload["n"] for one in again.events()] == [0, 1]
    assert again.head == 2


def test_every_kept_line_is_one_event_as_json(kept_at):
    log = EventLog(kept_at=kept_at)
    log.append("a_thing", {"n": 1}, lane=Lane.SYSTEM, actor="someone", causal_parent=0)
    row = json.loads(kept_at.read_text().splitlines()[0])
    assert set(row) >= {"seq", "kind", "lane", "payload", "at"}
    assert row["payload"] == {"n": 1} and row["actor"] == "someone"


def test_a_test_process_does_not_write_into_her_experience():
    """Reading is safe anywhere; writing is for a process that owns the state."""

    from core.runtime.event_spine import _where_the_experience_is_kept  # noqa: PLC2701

    assert _where_the_experience_is_kept() is None, (
        "a test run would have appended to the live log"
    )
