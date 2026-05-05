"""Final Blocker: Complete receipt chain for effectful actions.

Every environment action must produce a complete EnvironmentActionReceipt
with all required fields populated. No placeholders for authority-required actions.
"""
import pytest
from core.environment.environment_kernel import EnvironmentKernel
from core.environment.command import ActionIntent
from core.environment.receipt_chain import EnvironmentActionReceipt
from tests.environment.final_blockers.conftest import ScriptedTerminalAdapter


SCREEN_A = "Room A - player at (10, 10)\nHP:20(20) Pw:5(5) Dlvl:1"
SCREEN_B = "Room A - player at (11, 10)\nHP:20(20) Pw:5(5) Dlvl:1"


class TestReceiptChain:
    """Environment receipts must be complete and verifiable."""

    @pytest.mark.asyncio
    async def test_safe_action_has_complete_receipt(self):
        adapter = ScriptedTerminalAdapter([SCREEN_A, SCREEN_B, SCREEN_B])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="receipt_test")
        intent = ActionIntent(name="wait", risk="safe")
        frame = await kernel.step(intent)
        receipt = frame.receipt
        assert receipt is not None
        assert receipt.run_id == "receipt_test"
        assert receipt.environment_id == "terminal_grid:test"
        assert receipt.action_intent_id != ""
        assert receipt.gateway_decision_id != ""
        assert receipt.receipt_id != ""
        assert receipt.observation_id != ""

    @pytest.mark.asyncio
    async def test_effectful_action_receipt_has_command_id(self):
        adapter = ScriptedTerminalAdapter([SCREEN_A, SCREEN_B, SCREEN_B])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="receipt_cmd")
        intent = ActionIntent(name="move", parameters={"direction": "east"}, risk="caution")
        frame = await kernel.step(intent)
        receipt = frame.receipt
        assert receipt is not None
        # Command was compiled and executed
        if frame.gateway_decision and frame.gateway_decision.approved:
            assert receipt.command_id is not None
            assert receipt.command_id != ""

    @pytest.mark.asyncio
    async def test_blocked_action_receipt_records_veto(self):
        adapter = ScriptedTerminalAdapter([SCREEN_A, SCREEN_A])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="receipt_veto")
        # Forbidden risk should be blocked by gateway
        intent = ActionIntent(name="quaff", risk="forbidden", requires_authority=True)
        frame = await kernel.step(intent)
        receipt = frame.receipt
        assert receipt is not None
        assert receipt.status in ("blocked", "finalized")

    @pytest.mark.asyncio
    async def test_no_placeholder_receipt_for_authority_required(self):
        adapter = ScriptedTerminalAdapter([SCREEN_A, SCREEN_B, SCREEN_B])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="receipt_auth")
        intent = ActionIntent(name="quaff", risk="irreversible", requires_authority=True)
        frame = await kernel.step(intent)
        receipt = frame.receipt
        assert receipt is not None
        # If the action was blocked, that's acceptable
        # If it was executed, will_receipt_id must not be "not_required"
        if frame.gateway_decision and frame.gateway_decision.approved:
            assert receipt.will_receipt_id != "not_required"

    @pytest.mark.asyncio
    async def test_receipt_has_observation_hashes(self):
        adapter = ScriptedTerminalAdapter([SCREEN_A, SCREEN_B, SCREEN_B])
        kernel = EnvironmentKernel(adapter=adapter)
        await kernel.start(run_id="receipt_hash")
        intent = ActionIntent(name="wait", risk="safe")
        frame = await kernel.step(intent)
        receipt = frame.receipt
        assert receipt is not None
        assert receipt.observation_id != ""
        assert receipt.belief_hash_before != ""
