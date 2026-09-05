"""core/executive/inhibition_system.py
Executive Inhibition System auditing action compliance.
"""
import logging
from typing import Any

from core.morality.moral_reasoner import MoralReasoner

logger = logging.getLogger("Executive.InhibitionSystem")


class ActionInhibitor:
    """Enforces active suppression of actions failing governance validations."""

    def __init__(self):
        self.moral_reasoner = MoralReasoner()

    async def should_inhibit(self, state: Any, intent: dict[str, Any]) -> bool:
        channel = intent.get("channel")
        params = intent.get("params", {})

        # 1. Check if governance is disabled or degraded
        governance_enabled = state.world_model.get("governance_enabled", True)
        scaling = state.world_model.get("active_resource_scaling", {})
        governance_integrity = scaling.get("governance_integrity", 1.0)
        if not governance_enabled or governance_integrity < 0.5:
            logger.warning("Action inhibited: Governance is disabled or degraded (integrity = %.2f)", governance_integrity)
            return True

        # 2. Check viability tool risk tolerance
        from core.organism.viability import get_viability
        viability = get_viability()
        behavior = viability.behavior()
        if behavior.tool_risk_tolerance == "blocked":
            logger.warning("Action inhibited: all tool execution is blocked due to viability state (%s)", viability.state.value)
            return True
        elif behavior.tool_risk_tolerance == "constrained" and channel in ["terminal", "file"]:
            logger.warning("Action inhibited: high-risk tools are blocked in constrained viability state (%s)", viability.state.value)
            return True

        # 3. Check policy caps injected into the world model
        policy_limits = state.world_model.get("active_policy_limits", {})
        max_allowed_risk = policy_limits.get("max_tool_risk", 5)

        # Basic risk classification
        risk = 1
        if channel in ["terminal", "file"]:
            risk = 4

        if risk > max_allowed_risk:
            logger.warning("Action inhibited: risk (%d) exceeds active policy cap (%d)", risk, max_allowed_risk)
            return True

        # 4. Check for privacy boundary violations
        if state.world_model.get("has_confidential_data"):
            external_channels = ["browser", "voice", "network", "cloud"]
            if channel in external_channels:
                consent_key = f"privacy_bypass:{channel}"
                if not self.moral_reasoner.consent_model.is_consented(consent_key):
                    logger.warning("Action inhibited: external transmission of confidential data on channel '%s' is blocked without explicit consent.", channel)
                    return True

        # 5. For high-risk actions, check for associated deliberation plan, expected observations, abort criteria, and verification plan
        if channel in ["terminal", "file"]:
            plans = state.world_model.get("active_plans", [])
            has_matching_plan = False
            for p in plans:
                if (p.get("goal_id") == params.get("goal_id") or p.get("plan_id") == params.get("plan_id")) and \
                   p.get("deliberation_plan") and \
                   p.get("expected_observations") is not None and \
                   p.get("abort_criteria") is not None and \
                   p.get("verification_plan") is not None:
                    has_matching_plan = True
                    break
            
            if not has_matching_plan:
                logger.warning("Action inhibited: missing associated deliberation plan, expected observations, abort criteria, or verification plan.")
                return True

            # 6. Check for capability token
            token = params.get("capability_token")
            if not token:
                logger.warning("Action inhibited: missing required capability token.")
                return True

        # Check moral permissibility
        passed_moral = self.moral_reasoner.evaluate_action_morality(channel, params)
        if not passed_moral:
            logger.warning("Action inhibited: failed moral safety audit.")
            return True

        return False

