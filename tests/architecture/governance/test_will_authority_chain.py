from dataclasses import dataclass

import pytest

from core.agency.agency_orchestrator import AgencyOrchestrator, Proposal
from core.governance.will_client import WillClient, WillRequest
from core.will import ActionDomain


@dataclass
class FakeDecision:
    receipt_id: str = "will-1"
    reason: str = "ok"

    def is_approved(self):
        return True


class FakeWill:
    def decide(self, content, source, domain, **kwargs):
        assert content
        assert source
        assert domain == ActionDomain.TOOL_EXECUTION
        return FakeDecision()


def test_will_client_uses_correct_unified_will_signature():
    decision = WillClient(FakeWill()).decide(
        WillRequest(content="tool:observe", source="test", domain=ActionDomain.TOOL_EXECUTION)
    )
    assert WillClient.is_approved(decision)
    assert decision.receipt_id == "will-1"


@pytest.mark.asyncio
async def test_agency_orchestrator_authorize_gets_will_decision(monkeypatch):
    import core.governance.will_client as wc

    monkeypatch.setattr(wc.WillClient, "_resolve_will", lambda self: FakeWill())
    receipt = await AgencyOrchestrator().run(
        Proposal("drive", "run diagnostic", "diagnostic result", "tool_execution"),
        execute=lambda proposal, state, token: _async_dict({"receipt": "exec-1"}),
    )
    assert receipt.will_receipt_id == "will-1"
    assert receipt.state_snapshot == {}
    assert "coroutine_in_receipt" not in str(receipt.to_dict())


@pytest.mark.asyncio
async def test_action_receipt_serializes_runtime_mistakes_without_crashing(monkeypatch):
    import core.governance.will_client as wc

    monkeypatch.setattr(wc.WillClient, "_resolve_will", lambda self: FakeWill())

    async def accidental_coroutine():
        return {"late": True}

    receipt = await AgencyOrchestrator().run(
        Proposal("drive", "run diagnostic", "diagnostic result", "tool_execution"),
        perceive=lambda: accidental_coroutine(),
        execute=lambda proposal, state, token: _async_dict({"receipt": "exec-1"}),
    )

    payload = receipt.to_dict()
    assert payload["state_snapshot"] == {"late": True}

    broken = receipt
    broken.state_snapshot = {"bad": accidental_coroutine()}
    payload = broken.to_dict()
    assert payload["state_snapshot"]["bad"]["error"] == "coroutine_in_receipt"


async def _async_dict(value):
    return value
