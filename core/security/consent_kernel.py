"""core/security/consent_kernel.py
Central Security and Consent Kernel managing sandboxes, secrets, and networks.
"""
import logging
from typing import Any

from core.security.action_risk_classifier import ActionRiskClassifier
from core.security.approval_required import ApprovalRequiredChecker
from core.security.audit_log import SecurityAuditLogger
from core.security.egress_monitor import EgressConnectionMonitor
from core.security.network_policy import NetworkPolicyEngine
from core.security.sandbox import LocalCommandSandbox
from core.security.secret_guard import SecretGuard

logger = logging.getLogger("Security.ConsentKernel")


class ConsentKernel:
    """Canonical security manager auditing actions before execution."""

    def __init__(self):
        self.secret_guard = SecretGuard()
        self.network_policy = NetworkPolicyEngine()
        self.sandbox = LocalCommandSandbox()
        self.risk_classifier = ActionRiskClassifier()
        self.approval_checker = ApprovalRequiredChecker()
        self.audit_logger = SecurityAuditLogger()
        self.egress_monitor = EgressConnectionMonitor()

    def audit_and_verify_action(self, channel: str, params: dict[str, Any]) -> bool:
        """Runs the complete safety verification chain. Returns True if approved."""
        # 1. Secret leakage check
        if channel == "file" and params.get("action") == "write":
            content = params.get("content", "")
            if self.secret_guard.contains_secrets(content):
                logger.warning("ConsentKernel Block: command contains raw secret API/ssh keys.")
                self.audit_logger.log_event("block_secret_leakage", {"channel": channel, "path": params.get("path")})
                return False

        # 2. Network policy check
        if channel == "network":
            host = params.get("host", "localhost")
            port = params.get("port", 80)
            if not self.network_policy.is_egress_allowed(host):
                logger.warning("ConsentKernel Block: connection to '%s' violates network rules.", host)
                self.audit_logger.log_event("block_unauthorized_egress", {"host": host})
                return False
            self.egress_monitor.record_connection(host, port)

        # 3. Approval verification
        if self.approval_checker.requires_approval(channel, params):
            # Block action: requires operator input
            logger.warning("ConsentKernel Block: high-risk action requires human approval key.")
            self.audit_logger.log_event("block_approval_missing", {"channel": channel, "params": params})
            return False

        # Log pass event
        self.audit_logger.log_event("verify_action_success", {"channel": channel, "params": params})
        return True
