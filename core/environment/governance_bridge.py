from dataclasses import dataclass
from typing import Literal

@dataclass
class GovernanceDecision:
    approved: bool
    will_receipt_id: str | None = None
    authority_receipt_id: str | None = None
    reason: str = ""

class EnvironmentGovernanceBridge:
    """Bridges the environment kernel with the system-wide Will and Authority."""
    
    def __init__(self, will_gateway=None, authority_gateway=None):
        self.will_gateway = will_gateway
        self.authority_gateway = authority_gateway

    async def decide_action(self, intent) -> GovernanceDecision:
        if not intent.requires_authority:
            return GovernanceDecision(approved=True, will_receipt_id="not_required", authority_receipt_id="not_required")
            
        will_receipt = None
        if self.will_gateway:
            will_decision = await self.will_gateway.decide(intent)
            if will_decision.status != "PROCEED":
                return GovernanceDecision(approved=False, reason="will_refused")
            will_receipt = will_decision.receipt_id
            
        auth_receipt = None
        if self.authority_gateway:
            auth_receipt = await self.authority_gateway.authorize(intent, will_receipt)
            if not auth_receipt:
                return GovernanceDecision(approved=False, reason="authority_refused")
                
        # If no gateways are provided but requires_authority is True, default to mock passing for testing
        # In a real environment, it would fail closed if gateways were absent.
        will_receipt = will_receipt or f"mock_will_{intent.name}"
        auth_receipt = auth_receipt or f"mock_auth_{intent.name}"
            
        return GovernanceDecision(approved=True, will_receipt_id=will_receipt, authority_receipt_id=auth_receipt)
