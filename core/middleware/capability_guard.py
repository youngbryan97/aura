"""core/middleware/capability_guard.py

Enforces the capabilities manifest at runtime.
Acts as a gatekeeper for tool calls and file system operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config

logger = logging.getLogger("Aura.CapabilityGuard")

class CapabilityGuard:
    """Runtime enforcement of system capabilities."""

    def __init__(self, manifest_path: Optional[str] = None):
        if manifest_path is None:
            manifest_path = os.path.join(os.path.dirname(__file__), "../capabilities_manifest.json")
        
        self.manifest_path = Path(manifest_path)
        self.capabilities = {}
        self.load_manifest()

    def load_manifest(self):
        """Loads the capabilities manifest from JSON."""
        try:
            if self.manifest_path.exists():
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                    self.capabilities = data.get("capabilities", {})
                    logger.info("Capability Manifest loaded (v%s)", data.get("version", "unknown"))
            else:
                logger.warning("No capability manifest found at %s. Using default restricted mode.", self.manifest_path)
                self.capabilities = self._get_default_capabilities()
        except Exception as e:
            logger.error("Failed to load capability manifest: %s", e)
            self.capabilities = self._get_default_capabilities()

    def _get_default_capabilities(self) -> Dict[str, Any]:
        """Highly restrictive default capabilities."""
        return {
            "file_system": {
                "allowed_read": ["data/"],
                "allowed_write": ["data/logs/"],
                "restricted_paths": ["*.key", "core/security/**", "core/guardians/**", "core/prime_directives.py"],
            },
            "network": {"allowed_domains": ["localhost"]},
            "self_modification": {"allowed": False}
        }

    def _is_restricted_path(self, path: Path) -> bool:
        restricted_patterns = self.capabilities.get("file_system", {}).get("restricted_paths", [])
        if not restricted_patterns:
            return False

        candidates = [path.as_posix()]
        try:
            rel_path = path.relative_to(config.paths.project_root).as_posix()
            candidates.append(rel_path)
        except ValueError:
            rel_path = None

        for pattern in restricted_patterns:
            normalized_pattern = str(pattern).replace("\\", "/")
            for candidate in candidates:
                if Path(candidate).match(normalized_pattern):
                    return True
                if normalized_pattern.endswith("/**"):
                    prefix = normalized_pattern[:-3].rstrip("/")
                    if candidate == prefix or candidate.startswith(prefix + "/"):
                        return True
                if candidate == normalized_pattern or candidate.startswith(normalized_pattern.rstrip("/") + "/"):
                    return True
        return False

    def can_read_path(self, path: str) -> bool:
        """Checks if the system is allowed to read from a path."""
        p = Path(path).resolve()
        if self._is_restricted_path(p):
            logger.warning(f"SecurityViolation: Access Denied (Restricted Pattern): {path}")
            return False

        allowed_dirs = self.capabilities.get("file_system", {}).get("allowed_read", [])
        if "/" in allowed_dirs or "*" in allowed_dirs:
            return True

        # If it's in the project root and matches allowed dirs
        try:
            rel_p = p.relative_to(config.paths.project_root)
            for allowed in allowed_dirs:
                if str(rel_p).startswith(allowed.rstrip("/")):
                    return True
        except ValueError as _e:
            # Path is not under project root
            logger.debug('Ignored ValueError in capability_guard.py: %s', _e)

        # Check for explicit data dir access
        try:
            if p.relative_to(config.paths.data_dir):
                return True
        except ValueError as e:
            # Explicitly log the mismatch without suppressing the full context
            logger.debug("Path %s is not under Data Dir: %s", p, e)

        logger.warning(f"SecurityViolation: Access Denied (Path not in manifest): {path}")
        return False

    def can_write_path(self, path: str) -> bool:
        """Checks if the system is allowed to write to a path."""
        p = Path(path).resolve()
        if self._is_restricted_path(p):
            logger.warning(f"SecurityViolation: Write Denied (Restricted Pattern): {path}")
            return False

        allowed_dirs = self.capabilities.get("file_system", {}).get("allowed_write", [])
        if "/" in allowed_dirs or "*" in allowed_dirs:
            return True

        try:
            rel_p = p.relative_to(config.paths.project_root)
            for allowed in allowed_dirs:
                if str(rel_p).startswith(allowed.rstrip("/")):
                    return True
        except ValueError as _e:
            logger.debug('Ignored ValueError in capability_guard.py: %s', _e)
            
        try:
            if p.relative_to(config.paths.data_dir):
                return True
        except ValueError as e:
            logger.debug("Write path %s check: not under data dir (%s)", p, e)

        logger.warning(f"SecurityViolation: Write Denied (Path not in manifest): {path}")
        return False

    def can_call_tool(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Checks if a tool call is permitted with given arguments."""
        # 1. Network check if the tool involves external requests
        if tool_name in ["read_url_content", "search_web"]:
            allowed_domains = self.capabilities.get("network", {}).get("allowed_domains", [])
            if "*" in allowed_domains:
                return True
            # further domain validation logic...
            
        # 2. Command check for potentially destructive operations
        if tool_name in ["run_command", "shell_execute"]:
            # Check for restricted commands
            cmd = args.get("CommandLine", args.get("command", ""))
            if any(x in cmd for x in ["rm -rf /", "mkfs", "dd "]):
                raise PermissionError(f"SecurityError: Potentially destructive command blocked: {cmd}")
        
        return True

    def validate_self_mod(self) -> bool:
        """Checks if autonomous self-modification is currently permitted."""
        return self.capabilities.get("self_modification", {}).get("allowed", False)
