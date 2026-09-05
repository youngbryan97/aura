"""core/security/network_policy.py
Network Policy engine enforcing egress domain boundaries.
"""
from typing import List
from core.config import get_config


class NetworkPolicyEngine:
    """Verifies that network connections only reach approved domains."""

    def __init__(self):
        self.config = get_config()

    def is_egress_allowed(self, host: str) -> bool:
        """Determines if the target host is permitted under active network rules."""
        if not self.config.security.allow_network_access:
            return False
            
        allowed = self.config.security.allowed_domains
        if "*" in allowed:
            return True
            
        return host in allowed
