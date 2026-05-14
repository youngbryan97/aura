from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

@dataclass
class GovernanceDecision:
    approved: bool
    will_receipt_id: str | None = None
    authority_receipt_id: str | None = None
    executive_intent_id: str | None = None
    capability_token_id: str | None = None
    reason: str = ""

class EnvironmentGovernanceBridge:
    """Bridges the environment kernel with the system-wide Will and Authority."""
    
    def __init__(self, will_gateway=None, authority_gateway=None):
        self.will_gateway = will_gateway
        self.authority_gateway = authority_gateway

    async def decide_action(self, intent) -> GovernanceDecision:
        """Every environment action is consequential enough to be receipted.

        Older code only routed ``requires_authority`` intents.  That left
        "safe" movement, observation, and prompt-resolution paths outside the
        canonical Will/Authority receipt chain.  This bridge now resolves the
        runtime AuthorityGateway by default and fails closed if it cannot.
        """
        if self.authority_gateway and hasattr(self.authority_gateway, "authorize"):
            try:
                auth_receipt = await self.authority_gateway.authorize(intent, None)
                if not auth_receipt:
                    return GovernanceDecision(approved=False, reason="authority_refused")
                return GovernanceDecision(
                    approved=True,
                    will_receipt_id=getattr(auth_receipt, "will_receipt_id", None),
                    authority_receipt_id=getattr(auth_receipt, "receipt_id", str(auth_receipt)),
                )
            except Exception as exc:
                record_degradation("environment_governance", exc, severity="warning", action="legacy gateway refused")
                return GovernanceDecision(approved=False, reason=f"authority_gateway_error:{type(exc).__name__}")

        try:
            injected_will_receipt = None
            if self.will_gateway:
                will_decision = await self.will_gateway.decide(intent)
                status = str(getattr(will_decision, "status", "") or getattr(will_decision, "outcome", "")).upper()
                if status and status not in {"PROCEED", "APPROVED", "ALLOW", "ALLOWED", "CONSTRAIN"}:
                    return GovernanceDecision(approved=False, reason="will_refused")
                injected_will_receipt = getattr(will_decision, "receipt_id", None)

            from core.executive.authority_gateway import get_authority_gateway

            gateway = self.authority_gateway or get_authority_gateway()
            payload: dict[str, Any] = {
                "name": getattr(intent, "name", ""),
                "target_id": getattr(intent, "target_id", None),
                "parameters": dict(getattr(intent, "parameters", {}) or {}),
                "expected_effect": getattr(intent, "expected_effect", ""),
                "risk": getattr(intent, "risk", "safe"),
                "tags": sorted(getattr(intent, "tags", set()) or []),
                "requires_authority": bool(getattr(intent, "requires_authority", False)),
                "intent_id": intent.intent_id() if hasattr(intent, "intent_id") else "",
            }
            priority = {
                "safe": 0.35,
                "caution": 0.5,
                "risky": 0.7,
                "irreversible": 0.9,
                "forbidden": 1.0,
            }.get(str(payload["risk"]), 0.55)
            decision = await gateway.authorize_environment_action(
                str(payload["name"]),
                payload,
                source="environment_kernel",
                priority=priority,
                is_critical=bool(payload["risk"] in {"irreversible", "forbidden"}),
            )
            return GovernanceDecision(
                approved=bool(decision.approved),
                will_receipt_id=injected_will_receipt or decision.will_receipt_id,
                authority_receipt_id=decision.capability_token_id or decision.substrate_receipt_id or decision.executive_intent_id,
                executive_intent_id=decision.executive_intent_id,
                capability_token_id=decision.capability_token_id,
                reason=decision.reason,
            )
        except Exception as exc:
            record_degradation("environment_governance", exc, severity="error", action="fail closed")
            return GovernanceDecision(approved=False, reason=f"authority_gateway_missing:{type(exc).__name__}")
