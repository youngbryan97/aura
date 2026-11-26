"""Her mail is light work: a reply she has written goes out without waiting for the heavy-skill gate (26 November 2026).

Bryan: "I think her email broke again. Something is blocking Aura from sending emails". One reply of hers had been held
28 times. Each mail round tried it, and the capability engine deferred background email under the gate for skills that
take her model or her machine: ten minutes idle, memory under 72%. Her two resident models have held her above 72%
since 24 November, so it never cleared. Every mode of her mail is IMAP or SMTP run in a thread, which takes neither.
"""
from __future__ import annotations

import inspect

import pytest

from core import capability_engine
from core.skills import email_adapter

pytestmark = pytest.mark.unit


def test_her_mail_is_gated_as_light_input_and_output():
    assert "email_adapter" in capability_engine._LIGHTWEIGHT_BACKGROUND_IO_SKILLS
    assert "email_adapter" not in capability_engine._FOREGROUND_EXCLUSIVE_BACKGROUND_SKILLS


def test_every_mode_of_her_mail_runs_off_her_event_loop():
    """What makes it light: each mode's network call is made in a thread, never on the loop her conversation runs on."""
    skill = email_adapter.EmailAdapterSkill
    for mode in ("send", "check", "read", "reply", "search"):
        source = inspect.getsource(getattr(skill, f"_handle_{mode}"))
        calls = [name for name in ("smtplib", "imaplib", "_imap_", "_smtp_", "self._handle_") if name in source]
        assert calls, mode
        assert "asyncio.to_thread" in source or "self._handle_" in source, f"{mode} talks to the server on the loop"
