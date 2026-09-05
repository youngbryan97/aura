"""core/security/approval_required.py
Verifies approval states for high-risk actions.
"""
from typing import Dict, Any
from core.security.action_risk_classifier import ActionRiskClassifier


class ApprovalRequiredChecker:
    """Blocks execution of risky commands if approval keys are missing."""

    def __init__(self):
        self.classifier = ActionRiskClassifier()

    def requires_approval(self, channel: str, params: Dict[str, Any]) -> bool:
        risk = self.classifier.classify_risk(channel, params)
        # Risk score >= 7 requires explicit human approval
        return risk >= 7
