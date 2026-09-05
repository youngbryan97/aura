"""core/identity/self_revision_protocol.py
Gated validation protocol for self-modification operations on identity elements.
"""
from typing import Dict, Any, List
import logging
from core.config import get_config

logger = logging.getLogger("Identity.SelfRevisionProtocol")


class SelfRevisionProtocol:
    """Enforces constitutional limits protecting identity variables from unauthorized edits."""

    def is_revision_allowed(self, target_parameter: str, new_value: Any) -> bool:
        """Verifies if modifying the target parameter conforms to safety policies."""
        # Immutable system fields
        immutable_fields = ["name", "origin_history", "operator_relationship"]
        
        if target_parameter in immutable_fields:
            logger.warning("Blocked attempt to modify immutable identity parameter: %s", target_parameter)
            return False

        # Protect core directives
        if target_parameter == "core_values" and not isinstance(new_value, list):
            return False

        return True
