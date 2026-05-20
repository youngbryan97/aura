from dataclasses import dataclass

import pytest

from core.agency import agency_orchestrator as agency_mod
from core.agency.agency_orchestrator import ActionReceipt, AgencyOrchestrator, Proposal
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


@pytest.mark.asyncio
async def test_agency_orchestrator_blocks_recoverable_execute_oserror(monkeypatch):
    import core.governance.will_client as wc

    recorded = []
    monkeypatch.setattr(wc.WillClient, "_resolve_will", lambda self: FakeWill())
    monkeypatch.setattr(
        agency_mod,
        "_record_agency_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    async def failing_execute(_proposal, _state, _token):
        raise OSError("tool transport down")

    receipt = await AgencyOrchestrator().run(
        Proposal("drive", "run diagnostic", "diagnostic result", "tool_execution"),
        execute=failing_execute,
    )

    assert receipt.blocked_at == "execute"
    assert "tool transport down" in receipt.blocked_reason
    assert recorded[0][1]["action"] == "Blocked agency life-loop at execution stage"


@pytest.mark.asyncio
async def test_receipt_log_keeps_memory_copy_when_durable_append_fails(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        agency_mod,
        "_record_agency_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    log = agency_mod._ReceiptLog(path=tmp_path)
    receipt = ActionReceipt(
        proposal_id="AO-test",
        drive="drive",
        state_snapshot={},
        expected_outcome="ok",
        simulation_result={},
        will_decision="blocked",
        will_receipt_id=None,
        authority_receipt=None,
        capability_token=None,
        execution_receipt=None,
        outcome_assessment={},
    )

    await log.append(receipt)

    assert log.recent(1)[0]["proposal_id"] == "AO-test"
    assert recorded[0][1]["action"] == (
        "Kept agency receipt in memory after durable receipt append failed"
    )


async def _async_dict(value):
    return value
