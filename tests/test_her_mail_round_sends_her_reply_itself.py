"""While Bryan lets her, her mail round sends the reply she wrote, and nothing else it might send.

Bryan, 15 November 2026: "she should send email responses on her own. she doesnt need my permission to do that". From
20:52 local that night her round's reply ffb62e8291 ("Re: Checking In") was refused on every round, "Modality 'email' is
disabled": the email modality is off by default, the only grant naming email_adapter reads, and the send went out
labelled as a read of her inbox.
"""
from __future__ import annotations

import asyncio

import pytest

from core.capabilities.permission_model import PermissionRiskModel
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope
from core.executive.standing_authority import MAIL_REPLY_GRANT, MAIL_REPLY_ORIGIN, carries_bryans_standing_yes
from tests.test_standing_authority import _manager

REPLY = {"mode": "reply", "uid": "64", "body": "Thanks for checking in.", "in_reply_to": "<a@example.com>"}


def _rated(args):
    scope = resolve_execution_effect_scope("email_adapter", args)
    return scope, classify_execution_risk("email_adapter", args, effect_scope=scope)


@pytest.fixture
def bryan_lets_her(monkeypatch):
    monkeypatch.setenv("AURA_EMAIL_SENDS_WITHOUT_ASKING", "1")


@pytest.mark.asyncio
async def test_her_round_holds_a_lease_to_send_her_reply(bryan_lets_her):
    scope, risk = _rated(REPLY)
    decision = await _manager().issue_child_lease("email_adapter", REPLY, origin=MAIL_REPLY_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert decision.approved, decision.reason
    assert decision.grant_id == MAIL_REPLY_GRANT
    assert carries_bryans_standing_yes(MAIL_REPLY_ORIGIN, "email_adapter")


@pytest.mark.asyncio
async def test_without_his_yes_her_reply_waits_for_him(monkeypatch):
    monkeypatch.delenv("AURA_EMAIL_SENDS_WITHOUT_ASKING", raising=False)
    scope, risk = _rated(REPLY)
    decision = await _manager().issue_child_lease("email_adapter", REPLY, origin=MAIL_REPLY_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert not (decision.approved and decision.grant_id == MAIL_REPLY_GRANT)


@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "origin"), [
    ({"mode": "send", "to": "someone@example.com", "subject": "Hi", "body": "Hello."}, MAIL_REPLY_ORIGIN),
    ({**REPLY, "to": "someone-else@example.com"}, MAIL_REPLY_ORIGIN),
    ({**REPLY, "body": "  "}, MAIL_REPLY_ORIGIN),
    ({**REPLY, "uid": ""}, MAIL_REPLY_ORIGIN),
    (REPLY, "autonomous_initiative_loop"),
])
async def test_the_grant_sends_nothing_but_her_reply(bryan_lets_her, args, origin):
    scope, risk = _rated(args)
    decision = await _manager().issue_child_lease("email_adapter", args, origin=origin, effect_scope=scope,
                                                  risk_level=risk)
    assert not (decision.approved and decision.grant_id == MAIL_REPLY_GRANT)


@pytest.mark.asyncio
async def test_his_yes_opens_email_for_her_reply_and_only_with_its_lease(bryan_lets_her, monkeypatch):
    manager = _manager()
    monkeypatch.setattr("core.executive.standing_authority.get_standing_authority_manager", lambda: manager)
    scope, risk = _rated(REPLY)
    lease = await manager.issue_child_lease("email_adapter", REPLY, origin=MAIL_REPLY_ORIGIN, effect_scope=scope,
                                            risk_level=risk)
    model = PermissionRiskModel()
    assert model.modality.email is False

    def asked(context):
        return model.check_permission("tool_execution", "tool:email_adapter reply email", context,
                                      effect_scope=scope, execution_risk=risk)

    held = asked({**lease.context, "tool": "email_adapter", "origin": MAIL_REPLY_ORIGIN, "authority_arguments": REPLY})
    assert held.approved, held.reason
    bare = asked({"tool": "email_adapter", "origin": MAIL_REPLY_ORIGIN})
    assert not bare.approved and "Modality 'email' is disabled" in bare.reason


def test_her_round_sends_a_reply_as_a_reply_and_reads_as_a_read():
    from core.autonomy.social_initiative import _StartsSomethingSocial

    seen: list[dict] = []

    class Engine:
        async def execute(self, name, payload, context):
            seen.append(context)
            return {"ok": True}

    round_ = _StartsSomethingSocial.__new__(_StartsSomethingSocial)
    asyncio.run(round_._execute_email_adapter(dict(REPLY), cap_engine=Engine()))
    asyncio.run(round_._execute_email_adapter({"mode": "check", "limit": 50}, cap_engine=Engine()))

    assert seen[0]["origin"] == MAIL_REPLY_ORIGIN and "reply" in seen[0]["objective"]
    assert seen[1]["origin"] == "autonomous_initiative_loop" and "Read" in seen[1]["objective"]


@pytest.mark.asyncio
async def test_her_capability_engine_sends_her_reply_from_her_round_and_from_nowhere_else(bryan_lets_her, monkeypatch):
    from core.capability_engine import CapabilityEngine
    from core.skills import email_adapter

    sent: list[str] = []

    async def reply(self, params):
        sent.append(str(params.uid))
        return {"ok": True, "summary": f"replied to {params.uid}"}

    monkeypatch.setattr(email_adapter.EmailAdapterSkill, "_handle_reply", reply)
    monkeypatch.setattr(email_adapter.EmailAdapterSkill, "configured", staticmethod(lambda: True))
    monkeypatch.setattr("core.runtime.background_policy.background_activity_reason", lambda *a, **k: "")
    engine = CapabilityEngine()

    def context(origin):
        return {"origin": origin, "intent_source": origin, "user_facing": False,
                "objective": "Send her reply to the person whose message it answers."}

    from_her_round = await engine.execute("email_adapter", dict(REPLY), context=context(MAIL_REPLY_ORIGIN))
    from_elsewhere = await engine.execute("email_adapter", dict(REPLY), context=context("autonomous_initiative_loop"))
    something_new = await engine.execute(
        "email_adapter", {"mode": "send", "to": "someone@example.com", "subject": "Hi", "body": "Hello."},
        context=context(MAIL_REPLY_ORIGIN))

    assert from_her_round.get("ok") is True, from_her_round
    assert from_elsewhere.get("ok") is not True and something_new.get("ok") is not True
    assert sent == ["64"]
