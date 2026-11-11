"""F14: a person's mail is answered by her own turn, and her reply waits for Bryan until he lets her send."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_her_reply_is_her_own_turn_on_the_message_as_written(monkeypatch):
    from core.skills import her_replies_to_mail

    asked = {}

    class Engine:
        async def think(self, objective, *, context, mode, origin, timeout_s):
            asked.update(objective=objective, context=context, origin=origin)
            return SimpleNamespace(content="Hi Maya, yes, that was me. The piece was about tide clocks.")

    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: Engine() if name == "cognitive_engine" else default)
    reply = asyncio.run(her_replies_to_mail.her_reply_to(
        {"from": "Maya <maya@example.org>", "subject": "Your post", "body": "Was that you who wrote about tide clocks?"}))
    assert reply.startswith("Hi Maya")
    assert asked["objective"].startswith("From: Maya <maya@example.org>\nSubject: Your post")
    assert asked["context"]["visible_user_message"] == asked["objective"] and asked["context"]["allow_tools"] is False


def test_a_reply_is_held_and_read_back_until_it_is_sent(tmp_path, monkeypatch):
    from core.skills import her_replies_to_mail

    path = tmp_path / "email" / "drafts.jsonl"
    monkeypatch.setattr(her_replies_to_mail, "_drafts_path", lambda: path)
    held = asyncio.run(her_replies_to_mail.hold_for_bryan({"uid": "7", "from": "maya@example.org", "subject": "Hi"},
                                                         "Hello Maya."))
    (waiting,) = her_replies_to_mail.drafts(path)
    assert waiting["id"] == held and waiting["reply"] == "Hello Maya." and waiting["sent"] is False
    path.open("a").write(f'{{"id": "{held}", "sent": true}}\n')
    assert her_replies_to_mail.drafts(path)[0]["sent"] is True


def test_she_sends_without_asking_only_when_bryan_has_said_so(monkeypatch):
    from core.skills.her_replies_to_mail import sends_without_asking

    monkeypatch.delenv("AURA_EMAIL_SENDS_WITHOUT_ASKING", raising=False)
    assert not sends_without_asking()
    monkeypatch.setenv("AURA_EMAIL_SENDS_WITHOUT_ASKING", "1")
    assert sends_without_asking()


@pytest.mark.parametrize("written", [
    "I'll check what I know about Bryan before responding.",
    "I need to check what I know about Bryan before responding, and then reply via email.",
])
def test_a_reply_her_assessor_does_not_pass_is_not_held(monkeypatch, caplog, written):
    """LIVE 10 November 2026: both of these were held as her answer to Bryan."""
    import logging

    from core.skills import her_replies_to_mail

    class Engine:
        async def think(self, objective, *, context, mode, origin, timeout_s):
            return SimpleNamespace(content=written)

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: Engine() if name == "cognitive_engine" else default)
    with caplog.at_level(logging.INFO, logger="Skills.Email"):
        reply = asyncio.run(her_replies_to_mail.her_reply_to(
            {"from": "Bryan <youngbryan97@gmail.com>", "subject": "Hello - It's Bryan",
             "body": "Can you respond to this? Wanted to know how you're doing."}))
    assert reply == ""
    assert any("promise_without_answer" in r.getMessage() for r in caplog.records)


def test_a_withdrawn_draft_answers_nothing(tmp_path, monkeypatch):
    from core.skills import her_replies_to_mail

    path = tmp_path / "email" / "drafts.jsonl"
    monkeypatch.setattr(her_replies_to_mail, "_drafts_path", lambda: path)
    held = asyncio.run(her_replies_to_mail.hold_for_bryan({"uid": "54", "message_id": "<54@gmail>"}, "I'll check."))
    assert her_replies_to_mail.answered(path) == {"<54@gmail>"}
    path.open("a").write(f'{{"id": "{held}", "withdrawn": true}}\n')
    assert her_replies_to_mail.answered(path) == set()


def test_bryans_mail_is_answered_under_his_principal_and_a_strangers_under_none(monkeypatch):
    """Both of her replies to Bryan that were not answers wanted first to "check what I know about Bryan": her chat
    knows who is writing through his principal, and her mail turn had none."""
    from core.runtime.principal_context import current_relational_principal, relational_principal_scope_is_bound
    from core.skills import her_replies_to_mail

    seen: list[tuple[bool, str]] = []

    class Engine:
        async def think(self, objective, *, context, mode, origin, timeout_s):
            seen.append((relational_principal_scope_is_bound(), current_relational_principal()))
            return SimpleNamespace(content="Yes, it reached me. Hello, Bryan.")

    monkeypatch.setattr("core.container.ServiceContainer.get",
                        lambda name, default=None: Engine() if name == "cognitive_engine" else default)
    monkeypatch.setattr(her_replies_to_mail, "_owner_principal", lambda: "bryan")
    asyncio.run(her_replies_to_mail.her_reply_to({"from": "Bryan <youngbryan97@gmail.com>", "subject": "Hi",
                                                  "body": "Did you get this?", "from_owner": True}))
    asyncio.run(her_replies_to_mail.her_reply_to({"from": "maya@example.org", "subject": "Hi", "body": "Hello?"}))
    assert seen == [(True, "bryan"), (False, "")]
