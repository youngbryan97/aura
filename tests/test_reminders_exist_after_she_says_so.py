"""A reminder she says she set has to be there afterwards.

LIVE 2026-08-19: "remind me in 20 minutes to check the oven" was answered "I've
set a reminder for 20 minutes to check the oven." No tool ran and no reminder
existed. The person stops thinking about the oven, which is the whole cost of
the sentence.

Most of the parts were already here — an intention loop, a commitment engine, a
Scheduler holding one-shot timers in a list — and what was missing is what
makes a reminder a reminder: it survives a restart, and something reports it
when it comes due.

The failure branch is the one that matters. When the store cannot be written,
add_reminder returns None and the skill reports ok=False saying it is NOT set.
A receipt that lies is worse than no capability.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.agency import reminders as module
from core.agency.reminders import (
    add_reminder,
    complete_reminder,
    due_reminders,
    pending_reminders,
    requested_reminder,
)
from core.skills.reminder import ReminderSkill


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """A reminder file of this test's own, never the live one."""
    path = tmp_path / "reminders.json"
    monkeypatch.setattr(module, "reminders_path", lambda: path)
    return path


@pytest.mark.parametrize(
    ("message", "seconds", "text"),
    [
        ("remind me in 20 minutes to check the oven", 1200, "check the oven"),
        ("set a timer for 5 mins", 300, "set a timer"),
        ("remind me in an hour about the call", 3600, "the call"),
        ("nudge me in 30 seconds that the pasta is done", 30, "the pasta is done"),
        ("remind me after 45 minutes to stretch", 2700, "stretch"),
    ],
)
def test_the_request_is_parsed_as_asked(message: str, seconds: int, text: str) -> None:
    asked = requested_reminder(message)

    assert asked is not None
    assert asked.delay_s == seconds
    assert asked.text == text


@pytest.mark.parametrize(
    "message",
    ["what is 2 + 2", "remind me to call mum", "how are you"],
)
def test_a_request_without_a_time_is_not_invented(message: str) -> None:
    """Guessing a delay would be a different promise from the one made."""
    assert requested_reminder(message) is None


def test_a_stored_reminder_survives_a_fresh_read(store) -> None:
    stored = add_reminder("check the oven", 1200)

    assert stored is not None
    assert json.loads(store.read_text())[0]["text"] == "check the oven"
    assert [item.text for item in pending_reminders()] == ["check the oven"]


def test_a_reminder_becomes_due(store) -> None:
    add_reminder("already due", 0)

    assert [item.text for item in due_reminders()] == ["already due"]


def test_completing_one_removes_it_from_pending(store) -> None:
    stored = add_reminder("check the oven", 1200)

    assert complete_reminder(stored.id)
    assert pending_reminders() == []
    assert not complete_reminder(stored.id), "completing twice must not succeed"


def test_the_skill_reports_the_stored_reminder(store) -> None:
    result = asyncio.run(
        ReminderSkill().execute(
            {"objective": "remind me in 20 minutes to check the oven"}, {}
        )
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert result["reminder_id"] in result["effect_evidence"]
    assert pending_reminders()[0].id == result["reminder_id"]


def test_a_store_that_cannot_be_written_is_reported_as_not_set(monkeypatch) -> None:
    monkeypatch.setattr(module, "_save", lambda _reminders: False)

    result = asyncio.run(
        ReminderSkill().execute(
            {"objective": "remind me in 20 minutes to check the oven"}, {}
        )
    )

    assert result["ok"] is False
    assert result["effect_verified"] is False
    assert "NOT set" in result["summary"]


def test_the_skill_is_discovered_without_being_wired() -> None:
    """A new capability has to arrive by existing, not by being listed."""
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine()
    skills = getattr(engine, "skills", {}) or {}
    if not skills:
        pytest.skip("no capability registry in this process")

    assert "reminder" in skills
    assert engine._retrieved_tool_candidates(
        "remind me in 20 minutes to check the oven", 3
    )[0] == "reminder"
