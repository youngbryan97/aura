"""A reminder reaches the person when it comes due, as a notification on their desktop.

LIVE 19 November 2026: "Can you remind me in 20 minutes to stretch my back?" A reminder was a record in her store, and
nothing told the person when it came due (core/agency/reminders.py said so). No notification is sent from here: the
notification is a stand-in that keeps what it was handed.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import reminders as store_module
from core.agency import reminders_delivered as delivered
from core.agency.reminders import add_reminder, pending_reminders
from core.executive.her_own_passes import (
    REMINDER_DUE_GRANT,
    REMINDER_DUE_ORIGIN,
    carries_bryans_standing_yes,
    her_passes_allow,
)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """A reminder file of this test's own, never the live one."""
    path = tmp_path / "reminders.json"
    monkeypatch.setattr(store_module, "reminders_path", lambda: path)
    delivered._RUNNING.clear()
    yield path
    delivered._RUNNING.clear()


def _run(notify, until, *, added_while_waiting=None):
    async def go():
        loop = asyncio.create_task(delivered.deliver_reminders_as_they_come_due(notify, until=until))
        if added_while_waiting is not None:
            await asyncio.sleep(0)
            await asyncio.to_thread(added_while_waiting)
        await asyncio.wait_for(loop, timeout=30)

    asyncio.run(go())


def test_a_due_reminder_goes_out_and_is_done():
    add_reminder("stretch my back", 0)
    sent: list[str] = []

    async def notify(reminder):
        sent.append(reminder.text)
        return True

    _run(notify, until=lambda: bool(sent))
    assert sent == ["stretch my back"] and pending_reminders() == []


def test_a_reminder_that_did_not_go_out_stays_and_is_tried_again_only_when_the_store_changes():
    add_reminder("water the plants", 0)
    tries: list[str] = []

    async def refused(reminder):
        tries.append(reminder.text)
        return False

    async def go():
        loop = asyncio.create_task(delivered.deliver_reminders_as_they_come_due(refused))
        await asyncio.sleep(0.3)
        quiet = len(tries)  # nothing changed: the same refusal is not asked for again
        delivered.a_reminder_was_added()
        await asyncio.sleep(0.3)
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        return quiet

    assert asyncio.run(go()) == 1
    assert tries == ["water the plants", "water the plants"]
    assert [reminder.text for reminder in pending_reminders()] == ["water the plants"]


def test_a_reminder_added_while_the_loop_waits_wakes_it():
    add_reminder("far off", 3600)
    sent: list[str] = []

    async def notify(reminder):
        sent.append(reminder.text)
        return True

    _run(notify, until=lambda: bool(sent), added_while_waiting=lambda: add_reminder("now", 0))
    assert sent == ["now"]
    assert [reminder.text for reminder in pending_reminders()] == ["far off"]


def test_the_grant_lets_out_only_the_text_of_a_reminder_now_due():
    add_reminder("stretch my back", 0)
    add_reminder("not yet", 3600)
    allow = her_passes_allow
    assert allow("reminder_due", "notify_user", {"title": "Reminder", "message": "stretch my back"}) == (
        True, "reminder_due")
    assert allow("reminder_due", "notify_user", {"title": "Reminder", "message": "not yet"})[0] is False
    assert allow("reminder_due", "notify_user", {"message": "anything you like"})[0] is False
    assert allow("reminder_due", "email_adapter", {"message": "stretch my back"})[0] is False
    assert allow("reminder_due", "notify_user", {"message": "stretch my back", "url": "https://x.test"})[0] is False
    assert carries_bryans_standing_yes(REMINDER_DUE_ORIGIN, "notify_user")
    assert not carries_bryans_standing_yes(REMINDER_DUE_ORIGIN, "email_adapter")


def test_the_grant_is_built_in_for_that_origin_and_that_tool_alone():
    from core.executive.standing_authority import _builtin_grants

    grant = next(grant for grant in _builtin_grants() if grant.grant_id == REMINDER_DUE_GRANT)
    assert grant.allowed_origins == (REMINDER_DUE_ORIGIN,) and grant.allowed_tools == ("notify_user",)
    assert grant.argument_policy == "reminder_due"


def test_setting_a_reminder_starts_what_delivers_it(monkeypatch):
    from core.skills.reminder import ReminderSkill

    started: list[bool] = []
    monkeypatch.setattr(delivered, "ensure_reminders_are_delivered", lambda: started.append(True) or True)
    result = asyncio.run(ReminderSkill().execute({"objective": "remind me in 20 minutes to stretch my back"}, {}))
    assert result["ok"] is True and started == [True]


def test_a_reminder_that_went_out_and_could_not_be_marked_done_is_not_sent_again(monkeypatch):
    """LIVE 21 November 2026, 19:41: the first reminder went to the desktop, marking it done was refused as a write
    outside a governed context, and the store kept it pending for the next change to send again."""
    add_reminder("stretch my back", 0)
    monkeypatch.setattr(store_module, "complete_reminder", lambda reminder_id: False)
    sent: list[str] = []

    async def notify(reminder):
        sent.append(reminder.text)
        return True

    async def go():
        loop = asyncio.create_task(delivered.deliver_reminders_as_they_come_due(notify))
        await asyncio.sleep(0.3)
        delivered.a_reminder_was_added()
        await asyncio.sleep(0.3)
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)

    asyncio.run(go())
    assert sent == ["stretch my back"]


def test_marking_a_sent_reminder_done_is_her_own_governed_upkeep():
    from core.governance_context import is_governed

    seen: list[bool] = []

    def complete(reminder_id):
        seen.append(is_governed())
        return True

    assert not is_governed()
    assert delivered._marked_done(complete, "r1") is True and seen == [True]
