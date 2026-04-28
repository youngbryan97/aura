"""Security / sandbox / capability-token enforcement.

The audit demands deny-by-default capabilities for terminal, filesystem
outside the workspace, network POST, credential access, self-modification,
process kill, browser file://, and private-key reads. This module
declares the capability taxonomy and a sandbox policy registry that the
governed tools call into.
"""
from __future__ import annotations


import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.Security")


CAPABILITY_KINDS = (
    "file.read",
    "file.write",
    "browser.read",
    "browser.write",
    "terminal.run",
    "memory.write",
    "state.mutate",
    "self.modify",
    "network.get",
    "network.post",
    "credentials.read",
)


DEFAULT_DENY: Tuple[str, ...] = (
    "terminal.run",
    "network.post",
    "credentials.read",
    "self.modify",
)


PROTECTED_PATH_PATTERNS: Tuple[str, ...] = (
    r"\.ssh($|/)",
    r"\.aws($|/)",
    r"\.config/aura/secrets",
    r"\.gnupg",
    r"/etc/(passwd|shadow|sudoers)",
    r"id_rsa",
    r"id_ed25519",
)


PROTECTED_PATH_RE = re.compile("|".join(PROTECTED_PATH_PATTERNS))


@dataclass(frozen=True)
class CapabilityScope:
    capability: str
    pattern: str  # path glob, URL host, or scope tag


@dataclass
class SandboxPolicy:
    workspace_root: Path
    allowed_scopes: List[CapabilityScope] = field(default_factory=list)
    denied_capabilities: List[str] = field(default_factory=lambda: list(DEFAULT_DENY))

    def allow(self, capability: str, pattern: str) -> None:
        if capability not in CAPABILITY_KINDS:
            raise ValueError(f"unknown capability '{capability}'")
        self.allowed_scopes.append(CapabilityScope(capability=capability, pattern=pattern))

    def deny(self, capability: str) -> None:
        if capability not in CAPABILITY_KINDS:
            raise ValueError(f"unknown capability '{capability}'")
        if capability not in self.denied_capabilities:
            self.denied_capabilities.append(capability)

    def is_allowed(self, capability: str, target: str) -> Tuple[bool, str]:
        if capability in self.denied_capabilities:
            return False, f"capability '{capability}' is denied by policy"
        if capability.startswith("file."):
            return self._check_file(capability, target)
        if capability == "browser.read":
            return self._check_browser(target)
        if capability == "self.modify":
            return False, "self.modify requires explicit override + ladder validation"
        # default: scope must explicitly allow
        for scope in self.allowed_scopes:
            if scope.capability == capability and scope.pattern in (target, "*"):
                return True, "explicit scope match"
        return False, f"no explicit scope grants '{capability}' for '{target}'"

    def _check_file(self, capability: str, target: str) -> Tuple[bool, str]:
        try:
            target_path = Path(target).expanduser().resolve()
        except (OSError, RuntimeError):
            return False, "could not resolve target path"
        workspace = self.workspace_root.expanduser().resolve()
        try:
            target_path.relative_to(workspace)
        except ValueError:
            return False, f"path '{target_path}' escapes workspace '{workspace}'"
        if PROTECTED_PATH_RE.search(str(target_path)):
            return False, f"path '{target_path}' matches protected pattern"
        return True, "inside workspace"

    def _check_browser(self, target: str) -> Tuple[bool, str]:
        if target.startswith("file://"):
            return False, "browser file:// is denied"
        if target.startswith(("http://", "https://")):
            return True, "http(s) allowed"
        return False, "only http(s) browser scopes allowed"


# Convenience -----------------------------------------------------------------


def default_workspace_policy(root: Optional[Path] = None) -> SandboxPolicy:
    workspace = (root or Path.home() / ".aura" / "workspace").expanduser().resolve()
    get_task_tracker().create_task(get_storage_gateway().create_dir(workspace, cause='default_workspace_policy'))
    return SandboxPolicy(workspace_root=workspace)
