from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class _Decision:
    approved = True
    reason = "ok"
    capability_token_id = "tok"
    executive_intent_id = "intent"
    substrate_receipt_id = "sub"
    will_receipt_id = "will"
    outcome = "approved"


class _Gateway:
    def __init__(self):
        self.calls = []
        self.finalized = []

    async def authorize_tool_execution(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args, kwargs))
        return _Decision()

    async def authorize_environment_action(self, intent_name, payload, **kwargs):
        self.calls.append((intent_name, payload, kwargs))
        return _Decision()

    def verify_tool_access(self, tool_name, token):
        return token == "tok"

    def finalize_tool_execution(self, **kwargs):
        self.finalized.append(kwargs)


@pytest.mark.asyncio
async def test_shell_and_webclient_require_authority(monkeypatch):
    import core.capability_engine as cap
    import core.executive.authority_gateway as authority

    gateway = _Gateway()
    monkeypatch.setattr(authority, "get_authority_gateway", lambda: gateway)
    monkeypatch.setattr(
        cap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    monkeypatch.setattr(cap.requests, "get", lambda *args, **kwargs: SimpleNamespace(text="body"))

    shell = cap.Shell(cwd="/tmp", allowed_commands=["echo"])
    ok, output = await shell.run(["echo", "hi"])
    assert ok and output == "ok"
    web = cap.WebClient(allowed_domains=["example.com"])
    ok, body = await web.get("https://example.com")
    assert ok and body == "body"
    assert [call[0] for call in gateway.calls] == ["shell_command", "network_get"]
    assert len(gateway.finalized) == 2


@pytest.mark.asyncio
async def test_environment_governance_authorizes_even_safe_intents():
    from core.environment.command import ActionIntent
    from core.environment.governance_bridge import EnvironmentGovernanceBridge

    gateway = _Gateway()
    bridge = EnvironmentGovernanceBridge(authority_gateway=gateway)
    decision = await bridge.decide_action(ActionIntent(name="observe", risk="safe"))
    assert decision.approved
    assert gateway.calls and gateway.calls[0][0] == "observe"
    assert decision.will_receipt_id == "will"


@pytest.mark.asyncio
async def test_belief_write_fails_closed_when_authority_refuses(monkeypatch):
    import core.executive.authority_gateway as authority
    from core.skills.belief_ops import AddBeliefSkill

    class RefusingGateway:
        def authorize_belief_update_sync(self, *args, **kwargs):
            return SimpleNamespace(approved=False, reason="blocked", outcome="rejected", will_receipt_id="will")

    monkeypatch.setattr(authority, "get_authority_gateway", lambda: RefusingGateway())
    result = await AddBeliefSkill().execute(
        {"source": "Bryan", "relation": "prefers", "target": "closure", "confidence": 0.9},
        {},
    )
    assert not result["ok"]
    assert "AuthorityGateway" in result["error"]
