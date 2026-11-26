"""core/tools/tool_permission.py — Tool Permissions Guard."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tools.tool_manifest import ToolManifest

logger = logging.getLogger("Aura.ToolPermission")


class ToolPermissionGuard:
    """Validates if a tool is permitted to perform network or file operations under its manifest."""

    @staticmethod
    def is_directory_allowed(manifest: ToolManifest, path: str) -> bool:
        """Checks if a target file path is within the tool's allowed directories."""
        if "all" in manifest.permissions or not manifest.sandbox_required:
            return True
        for allowed in manifest.allowed_directories:
            if path.startswith(allowed):
                return True
        logger.warning("Tool %s attempted unauthorized file access: %s", manifest.name, path)
        return False

    @staticmethod
    def is_domain_allowed(manifest: ToolManifest, domain: str) -> bool:
        """Checks if a network request is targetting an allowed domain."""
        if "network" not in manifest.permissions:
            logger.warning("Tool %s does not have network permission", manifest.name)
            return False
        if "*" in manifest.allowed_domains:
            return True
        for allowed in manifest.allowed_domains:
            if domain == allowed or domain.endswith("." + allowed):
                return True
        logger.warning("Tool %s attempted unauthorized network access: %s", manifest.name, domain)
        return False
