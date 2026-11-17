"""A notification the person asks for goes out inside quiet hours, and its words never become AppleScript.

LIVE 16 November 2026, 22:16 local: "Could you put up a desktop notification that says time to stretch?" Her model
called notify_user (the call had been cut at "notify" by a text rule until 22:12), and the notifier suppressed it: quiet
hours, 21:00 to 08:30. Quiet hours keep her from interrupting on her own; they are not a refusal of what the person has
just asked for.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.senses import notifications
from core.senses.notifications import DesktopNotifier

ASKED = "Could you put up a desktop notification that says time to stretch?"


@pytest.fixture
def quiet_hours_now(monkeypatch):
    ran: list[list[str]] = []

    from core.runtime.subprocess_gateway import _coerce_argv

    class _Gateway:
        def run(self, args, **_kwargs):
            # The real gateway's own argument check: it refuses an empty one, and a stand-in that did not let a call
            # with no subtitle pass here and fail live (16 November 2026, 23:31).
            ran.append(_coerce_argv(args))
            return SimpleNamespace(returncode=0, args=args, stdout="", stderr="")

    monkeypatch.setattr(notifications, "_within_quiet_hours", lambda *_args: True)
    monkeypatch.setattr(notifications, "get_subprocess_gateway", lambda: _Gateway())
    return ran


def test_quiet_hours_hold_what_she_sends_on_her_own(quiet_hours_now):
    outcome = DesktopNotifier.send("Aura", "Your build finished.")
    assert outcome.status == "suppressed_quiet_hours" and quiet_hours_now == []


def test_quiet_hours_do_not_refuse_what_the_person_asked_for(quiet_hours_now):
    outcome = DesktopNotifier.send("Aura", "Time to stretch", sound="Glass", asked_for=True)
    assert outcome.delivered and outcome.status == "delivered" and len(quiet_hours_now) == 1


def test_notifications_turned_off_still_refuse_it(quiet_hours_now, monkeypatch):
    monkeypatch.setattr(notifications, "get_runtime_setting", lambda key, default=None: False
                        if key == "notify.enabled" else default)
    assert DesktopNotifier.send("Aura", "Time to stretch", asked_for=True).status == "disabled"
    assert quiet_hours_now == []


def test_her_words_reach_applescript_as_arguments_not_source(quiet_hours_now):
    words = 'time "to\\" stretch" & (do shell script "echo pwned") & "'
    DesktopNotifier.send("Aura", words, sound="Glass", asked_for=True)

    args = quiet_hours_now[0]
    source = [args[i + 1] for i, arg in enumerate(args) if arg == "-e"]
    assert words in args and not any(words in line or "pwned" in line for line in source)
    assert source == ["on run argv", "display notification (item 1 of argv) with title (item 2 of argv) "
                      "sound name (item 3 of argv)", "end run"]
    assert args[-3:] == [words, "Aura", "Glass"]


def test_a_notification_with_a_subtitle_reads_each_argument_it_was_given(quiet_hours_now):
    DesktopNotifier.send("Aura", "Time to stretch", subtitle="From Aura", sound="Glass", asked_for=True)
    DesktopNotifier.send("Aura", "Time to stretch", sound="", asked_for=True)

    with_subtitle, plain = quiet_hours_now
    assert with_subtitle[-4:] == ["Time to stretch", "Aura", "From Aura", "Glass"]
    assert "subtitle (item 3 of argv) sound name (item 4 of argv)" in with_subtitle[4]
    assert plain[-2:] == ["Time to stretch", "Aura"] and plain[4].endswith("with title (item 2 of argv)")


def test_a_notification_with_no_words_is_not_sent(quiet_hours_now):
    assert DesktopNotifier.send("Aura", "   ", asked_for=True).status == "failed" and quiet_hours_now == []


@pytest.mark.parametrize(("context", "asked"), [
    ({"required_skills": ["notify_user"], "a_person_is_waiting": True, "message": ASKED, "origin": "desktop_quick_user"},
     True),
    ({"required_skills": ["notify_user"], "a_person_is_waiting": False, "message": ASKED,
      "origin": "autonomous_initiative_loop"}, False),
    ({"required_skills": ["web_search"], "a_person_is_waiting": True, "message": ASKED, "origin": "desktop_quick_user"},
     False),
    ({"origin": "autonomous_initiative_loop", "message": "Your build finished."}, False),
])
def test_the_skill_says_whether_the_person_asked_for_this_one(monkeypatch, context, asked):
    from core.skills import notify_user

    seen: dict[str, object] = {}

    def send(**kwargs):
        seen.update(kwargs)
        return notifications.DeliveryResult(delivered=True, status="delivered")

    monkeypatch.setattr(notify_user.DesktopNotifier, "send", staticmethod(send))
    result = asyncio.run(notify_user.NotifyUserSkill().execute({"message": "Time to stretch"}, dict(context)))

    assert result["delivered"] is True and seen["asked_for"] is asked
