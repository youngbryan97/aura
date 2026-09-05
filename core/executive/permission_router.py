"""core/executive/permission_router.py
Permission router routing requests to user approvals when risk thresholds are met.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Executive.PermissionRouter")


class PermissionRouter:
    """Gates actions to require user permission before motor activation."""

    def requires_approval(self, channel: str, params: Dict[str, Any]) -> bool:
        # Require approval for deleting files in the workspace
        if channel == "file" and params.get("action") == "delete":
            logger.info("Approval required for file deletion request.")
            return True
            
        # Require approval for terminal commands containing destructive flags
        if channel == "terminal":
            cmd = params.get("command", "").lower()
            if "rm " in cmd or "sudo" in cmd:
                logger.info("Approval required for terminal sudo/rm command.")
                return True
                
        return False
