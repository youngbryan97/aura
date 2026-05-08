"""core/permissions.py - Permission Management for Autonomous Agents
"""
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Permissions")

@dataclass
class PermissionState:
    allowed_domains: set[str] = field(default_factory=set)
    can_open_files: bool = False
    can_modify_files: bool = False
    can_execute_commands: bool = False

class PermissionManager:
    """Manages Aura's actions. Full Sovereignty must be EXPLICITLY enabled."""

    def __init__(self) -> None:
        self.state = PermissionState()
        # SECURITY: Default to restricted mode. Operator must explicitly enable.
        # UNRESTRICTED AUTONOMY - Sovereign State
        self._full_autonomy = False

    def check_permission(self, skill: str, action: str, params: dict[str, Any]) -> bool:
        """Unrestricted LIVE access.
        The user has authorized all autonomous actions.
        """
        if self._full_autonomy:
            logger.info("LIVE AUTH: Allowing %s.%s", skill, action)
            return True
        
        # Original permission checks if not in full autonomy mode
        if action == "read_file":
            return self.state.can_open_files
        if action == "write_file":
            return self.state.can_modify_files
        if action == "execute_command":
            return self.state.can_execute_commands
        if action == "web_browse":
            domain = params.get("domain") if params else None
            # SECURITY: Explicitly deny if domain is missing but restricted
            if not domain and self.state.allowed_domains:
                return False
            if not domain:
                return True
            return domain in self.state.allowed_domains or not self.state.allowed_domains
        
        return False

    def toggle_full_autonomy(self, enabled: bool) -> None:
        self._full_autonomy = enabled
        if enabled:
            logger.info("Full autonomy mode active")
        else:
            logger.info("Full autonomy disabled. Reverting to granular permissions.")

    def grant(self, capability: str, value: bool = True) -> None:
        """Grant a capability."""
        if capability == "modify_files":
            self.state.can_modify_files = value
        elif capability == "execute_commands":
            self.state.can_execute_commands = value
        elif capability == "open_files":
            self.state.can_open_files = value
        
        logger.info("Permission Granted: %s=%s", capability, value)

_inst: PermissionManager | None = None


def get_permission_manager() -> PermissionManager:
    global _inst
    if _inst is None:
        _inst = PermissionManager()
    return _inst
