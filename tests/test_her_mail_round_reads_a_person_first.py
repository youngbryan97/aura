"""F14: her own mail round reaches her inbox, reads a person's mail first, answers a message once, and a held reply
goes to the message it answers.

LIVE 10 November 2026, once her credentials read: her round logged "Checking email" and nothing else. Her social loop
let it start after two minutes idle; the capability engine defers email_adapter until ten, and the deferral was
dropped without a word and counted as a check. Had it run, it would have read the first three of her five newest
unread, three automated notices, and Bryan's "Hey!", fifth, would not have been read. Her subjects reached her as
encoded words, and a reply was sent to whatever message sat at a place in her mailbox when Bryan sent it.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

# Her five newest unread on 10 November, as the check lists them: oldest first.
_NEWEST_FIVE = [
    {"uid": "58", "from": "ChatGPT <noreply@email.openai.com>", "subject": "Halloween costumes",
     "automated": "List-Unsubscribe present", "addressed": "to", "message_id": "<58@openai>"},
    {"uid": "59", "from": "ChatGPT <noreply@email.openai.com>", "subject": "A simpler way",
     "automated": "sent from noreply@email.openai.com", "addressed": "to", "message_id": "<59@openai>"},
    {"uid": "60", "from": "Google <no-reply@accounts.google.com>", "subject": "Security alert",
     "automated": "Feedback-Id present", "addressed": "to", "message_id": "<60@google>"},
    {"uid": "61", "from": "Google <no-reply@accounts.google.com>", "subject": "Security alert",
     "automated": "Feedback-Id present", "addressed": "to", "message_id": "<61@google>"},
    {"uid": "62", "from": "Bryan Young <youngbryan97@gmail.com>", "subject": "Hey!",
     "automated": "", "addressed": "to", "message_id": "<62@gmail>"},
]


def _loop(monkeypatch, tmp_path, *, check, replies):
    """Her loop with a capability engine that answers ``check`` and reads each message from ``_NEWEST_FIVE``."""
    from core.autonomy.autonomous_initiative_loop import AutonomousInitiativeLoop
    from core.skills import her_replies_to_mail
    from core.skills.email_adapter import EmailAdapterSkill

    monkeypatch.setattr(EmailAdapterSkill, "configured", staticmethod(lambda: True))
    calls: list[dict] = []
    by_uid = {m["uid"]: m for m in _NEWEST_FIVE}

    class CapabilityEngine:
        async def execute(self, skill, payload, context):
            calls.append(dict(payload))
            if payload["mode"] == "check":
                return check
            message = by_uid[payload["uid"]]
            return {"ok": True, **message, "body": f"body of {message['subject']}", "is_auto_reply": False}

    engine = CapabilityEngine()
    services = {"capability_engine": engine, "memory_manager": None}
    monkeypatch.setattr("core.autonomy.social_initiative.optional_service",
                        lambda name, default=None: services.get(name, default))

    async def her_reply(message):
        replies.append(message["subject"])
        return f"Hi Bryan, about {message['subject']}"

    monkeypatch.setattr(her_replies_to_mail, "her_reply_to", her_reply)
    monkeypatch.setattr(her_replies_to_mail, "_drafts_path", lambda: tmp_path / "email" / "drafts.jsonl")
    monkeypatch.delenv("AURA_EMAIL_SENDS_WITHOUT_ASKING", raising=False)
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    loop._emit_feed = lambda *_a, **_k: None
    loop._queue_visible_update = lambda *_a, **_k: True

    async def remember(*_a, **_k):
        return None

    loop._remember_social_observation = remember
    return loop, calls


def test_a_persons_mail_to_her_is_read_before_the_notices_ahead_of_it(monkeypatch, tmp_path):
    replies: list[str] = []
    loop, calls = _loop(monkeypatch, tmp_path, check={"ok": True, "unread": 57, "messages": _NEWEST_FIVE},
                        replies=replies)
    assert asyncio.run(loop._check_email_initiative()) == ""
    assert [c.get("uid") for c in calls if c["mode"] == "read"] == ["62", "61", "60"]
    assert replies == ["Hey!"]


def test_a_message_she_has_answered_is_not_answered_again_after_a_restart(monkeypatch, tmp_path):
    from core.skills import her_replies_to_mail

    replies: list[str] = []
    check = {"ok": True, "unread": 57, "messages": _NEWEST_FIVE}
    loop, _calls = _loop(monkeypatch, tmp_path, check=check, replies=replies)
    asyncio.run(loop._check_email_initiative())
    # A restart: a new loop, the same drafts.
    again, calls = _loop(monkeypatch, tmp_path, check=check, replies=replies)
    asyncio.run(again._check_email_initiative())
    assert replies == ["Hey!"]
    assert "62" not in [c.get("uid") for c in calls if c["mode"] == "read"]
    (held,) = her_replies_to_mail.drafts(tmp_path / "email" / "drafts.jsonl")
    assert held["message_id"] == "<62@gmail>" and held["sent"] is False


def test_a_deferred_round_says_why_and_is_not_counted_as_a_check(monkeypatch, tmp_path, caplog):
    deferred = {"ok": False, "status": "deferred", "reason": "recent_user_183",
                "message": "Background email_adapter deferred while live conversation is protected (recent_user_183)."}
    loop, calls = _loop(monkeypatch, tmp_path, check=deferred, replies=[])
    with caplog.at_level(logging.INFO, logger="Aura.Initiative"):
        assert "recent_user_183" in asyncio.run(loop._check_email_initiative())
    assert [c["mode"] for c in calls] == ["check"]
    assert any("deferred before it reached her inbox" in r.getMessage() for r in caplog.records)

    import core.autonomy.autonomous_initiative_loop as initiative

    async def deferred_round():
        return "recent_user_183"

    async def reddit_round():
        return ""

    async def one_minute(_seconds):
        loop.running = False

    monkeypatch.setattr(initiative, "_passive_social_allowed", lambda *_a: True)
    monkeypatch.setattr("core.autonomy.social_initiative.asyncio.sleep", one_minute)
    loop._check_email_initiative = deferred_round
    loop._check_reddit_initiative = reddit_round
    loop.running = True
    asyncio.run(loop._social_interaction_loop())
    assert loop._last_email_check == 0.0 and loop._last_reddit_check > 0.0


def test_her_social_rounds_start_on_the_profile_the_engine_defers_them_on(monkeypatch):
    import core.autonomy.autonomous_initiative_loop as initiative
    from core.runtime.background_policy import HEAVY_SKILL_PREFLIGHT_BACKGROUND_POLICY

    asked: list[dict] = []
    monkeypatch.setattr(initiative, "_watching_with_someone", lambda: "")
    monkeypatch.setattr(initiative, "background_activity_allowed", lambda *_a, **k: asked.append(k) or True)
    monkeypatch.setattr(initiative, "background_activity_reason", lambda *_a, **k: asked.append(k) or "")
    initiative._passive_social_allowed()
    initiative._passive_social_blocker()
    assert [k["profile"] for k in asked] == [HEAVY_SKILL_PREFLIGHT_BACKGROUND_POLICY] * 2
    assert all("min_idle_seconds" not in k and "max_memory_percent" not in k for k in asked)


def test_her_mail_headers_read_as_a_person_reads_them():
    from core.skills.email_adapter import as_read

    assert as_read('=?UTF-8?Q?"Even_this_was_generated_by?=\n =?UTF-8?Q?_AI"?=') == '"Even this was generated by AI"'
    assert as_read("Ask anything =?UTF-8?B?4oCT?= really") == "Ask anything – really"
    assert as_read(None, "(no subject)") == "(no subject)"


class _Imap:
    """IMAP that answers by UID only: a sequence-number command fails the test."""

    commands: list[tuple] = []

    def __init__(self, *_a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def login(self, *_a):
        return "OK", [b""]

    def select(self, *_a):
        return "OK", [b"1"]

    def search(self, *_a):
        raise AssertionError("a sequence number is a place in the mailbox, not a message")

    fetch = search

    def uid(self, command, *args):
        _Imap.commands.append((command, args[0]))
        if command == "SEARCH":
            return "OK", [b"62"]
        raw = (b"From: Bryan Young <youngbryan97@gmail.com>\r\nTo: auraluna.cog@gmail.com\r\n"
               b"Subject: =?UTF-8?Q?Hey!?=\r\nMessage-ID: <62@gmail>\r\n\r\nHi Aura")
        return "OK", [(b"1 (UID 62 BODY[] {1})", raw)]


def test_her_mail_is_found_and_read_by_uid(monkeypatch):
    from core.skills import email_adapter
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    _Imap.commands = []
    monkeypatch.setattr(email_adapter.imaplib, "IMAP4_SSL", _Imap)
    monkeypatch.setattr(EmailAdapterSkill, "_get_creds", lambda self: ("auraluna.cog@gmail.com", "app-password"))
    skill = EmailAdapterSkill()
    checked = asyncio.run(skill._handle_check(EmailInput(mode="check", limit=50)))
    (message,) = checked["messages"]
    assert message["uid"] == "62" and message["message_id"] == "<62@gmail>" and message["subject"] == "Hey!"
    read = asyncio.run(skill._handle_read(EmailInput(mode="read", uid="62")))
    assert read["ok"] and read["message_id"] == "<62@gmail>"
    assert [command for command, _ in _Imap.commands] == ["SEARCH", "FETCH", "FETCH"]


def test_a_reply_is_not_sent_to_a_message_other_than_the_one_it_answers(monkeypatch):
    from core.skills.email_adapter import EmailAdapterSkill, EmailInput

    skill = EmailAdapterSkill()
    sent: list = []

    async def read(_params):
        return {"ok": True, "from": "someone@example.org", "subject": "Other", "message_id": "<61@google>",
                "is_auto_reply": False}

    async def send(params):
        sent.append(params)
        return {"ok": True}

    monkeypatch.setattr(skill, "_handle_read", read)
    monkeypatch.setattr(skill, "_handle_send", send)
    refused = asyncio.run(skill._handle_reply(EmailInput(mode="reply", uid="62", body="Hi", in_reply_to="<62@gmail>")))
    assert refused["status"] == "not_the_message_answered" and sent == []
    assert asyncio.run(skill._handle_reply(EmailInput(mode="reply", uid="62", body="Hi", in_reply_to="<61@google>")))["ok"]
