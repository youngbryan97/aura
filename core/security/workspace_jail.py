"""core/security/workspace_jail.py
==================================
Workspace root jail for file system skills.

Prevents path traversal attacks by normalizing all paths against a
configured workspace root. Any path that escapes the jail is rejected.

This is the LAST LINE OF DEFENSE before file I/O.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import FrozenSet, Optional, Set, Tuple

logger = logging.getLogger("Aura.Security.WorkspaceJail")

# Default allowed roots. The jail permits I/O within these trees.
_DEFAULT_ALLOWED_ROOTS: Tuple[str, ...] = (
    str(Path.home() / ".aura"),
    str(Path.home() / "Desktop"),
    str(Path.home() / "Documents"),
    str(Path.home() / "Downloads"),
    "/tmp/aura",
)

# Absolutely denied paths, even if under an allowed root.
_DENIED_PATHS: FrozenSet[str] = frozenset({
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    str(Path.home() / ".ssh"),
    str(Path.home() / ".gnupg"),
    str(Path.home() / ".aws" / "credentials"),
    str(Path.home() / ".config" / "gcloud"),
    str(Path.home() / ".kube"),
})

# Denied filename patterns
_DENIED_FILENAMES: FrozenSet[str] = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
})


class WorkspaceJail:
    """Path sanitization and access control for file skills."""

    def __init__(
        self,
        allowed_roots: Optional[Tuple[str, ...]] = None,
        extra_denied: Optional[Set[str]] = None,
    ) -> None:
        self._allowed_roots = tuple(
            str(Path(r).resolve()) for r in (allowed_roots or _DEFAULT_ALLOWED_ROOTS)
        )
        self._extra_denied: Set[str] = extra_denied or set()
        logger.info(
            "WorkspaceJail initialized with %d allowed roots",
            len(self._allowed_roots),
        )

    def validate_path(self, raw_path: str) -> Tuple[bool, str, str]:
        """Validate a path against the jail.

        Returns:
            (allowed: bool, resolved_path: str, reason: str)
        """
        if not raw_path or not raw_path.strip():
            return False, "", "empty_path"

        try:
            # Resolve symlinks and normalize
            resolved = str(Path(raw_path).expanduser().resolve())
        except (OSError, ValueError) as e:
            return False, "", f"path_resolution_failed: {e}"

        # Check absolute denied list
        for denied in _DENIED_PATHS:
            if resolved.startswith(denied) or resolved == denied:
                logger.warning(
                    "🚫 WorkspaceJail DENIED (hardcoded): %s", raw_path
                )
                return False, resolved, "denied_path"

        if resolved in self._extra_denied:
            logger.warning(
                "🚫 WorkspaceJail DENIED (custom): %s", raw_path
            )
            return False, resolved, "denied_custom"

        # Check denied filenames
        filename = Path(resolved).name
        if filename in _DENIED_FILENAMES:
            logger.warning(
                "🚫 WorkspaceJail DENIED (filename): %s", raw_path
            )
            return False, resolved, "denied_filename"

        # Check if under an allowed root
        for root in self._allowed_roots:
            if resolved.startswith(root + os.sep) or resolved == root:
                return True, resolved, "allowed"

        logger.warning(
            "🚫 WorkspaceJail DENIED (not in allowed roots): %s -> %s",
            raw_path,
            resolved,
        )
        return False, resolved, "outside_jail"

    def sanitize_path(self, raw_path: str) -> Optional[str]:
        """Sanitize and return the resolved path, or None if denied."""
        allowed, resolved, reason = self.validate_path(raw_path)
        if allowed:
            return resolved
        return None

    def add_allowed_root(self, path: str) -> None:
        """Dynamically add an allowed root."""
        resolved = str(Path(path).resolve())
        self._allowed_roots = self._allowed_roots + (resolved,)
        logger.info("WorkspaceJail: added allowed root %s", resolved)

    def get_status(self) -> dict:
        return {
            "allowed_roots": list(self._allowed_roots),
            "denied_paths_count": len(_DENIED_PATHS) + len(self._extra_denied),
            "denied_filenames_count": len(_DENIED_FILENAMES),
        }


# Singleton
_jail_instance: Optional[WorkspaceJail] = None


def get_workspace_jail() -> WorkspaceJail:
    global _jail_instance
    if _jail_instance is None:
        _jail_instance = WorkspaceJail()
    return _jail_instance
