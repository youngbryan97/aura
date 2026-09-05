"""
macOS Sandbox Manager — Ported from gemini-cli/sandbox

Generates and enforces native Apple sandbox (sandbox-exec) profiles.
Provides fine-grained file, network, and execution constraints when running
unknown or potentially dangerous shell commands.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.Sandbox")

@dataclass
class SandboxConfig:
    """Configuration for a macOS sandbox profile."""
    allow_network: bool = False
    allow_exec: bool = False
    read_paths: List[str] = field(default_factory=list)
    write_paths: List[str] = field(default_factory=list)
    execute_paths: List[str] = field(default_factory=list)

class MacOSSandbox:
    """Manages creation and execution of native macOS sandboxes via sandbox-exec."""

    def __init__(self, config: SandboxConfig):
        self.config = config

    def _generate_profile(self) -> str:
        """Generate the Scheme-based macOS sandbox profile."""
        lines = [
            "(version 1)",
            "(deny default)",  # Deny everything by default
            
            # Basic system operations needed for most commands
            "(allow process-exec (literal \"/bin/sh\") (literal \"/bin/bash\") (literal \"/bin/zsh\"))",
            "(allow process-fork)",
            "(allow sysv-ipc*)",
            "(allow signal)",
            "(allow mach-lookup)",
            
            # Read standard system read-only directories
            "(allow file-read*",
            "    (subpath \"/System\")",
            "    (subpath \"/Library\")",
            "    (subpath \"/usr\")",
            "    (subpath \"/bin\")",
            "    (subpath \"/sbin\")",
            ")"
        ]

        # Network
        if self.config.allow_network:
            lines.append("(allow network*)")
        else:
            lines.append("(deny network*)")

        # Command execution
        if self.config.allow_exec:
            lines.append("(allow process-exec*)")
        else:
            if self.config.execute_paths:
                lines.append("(allow process-exec")
                for path in self.config.execute_paths:
                    lines.append(f"    (literal \"{path}\")")
                    lines.append(f"    (subpath \"{path}\")")
                lines.append(")")

        # File reading
        if self.config.read_paths:
            lines.append("(allow file-read*")
            for path in self.config.read_paths:
                lines.append(f"    (subpath \"{path}\")")
            lines.append(")")

        # File writing
        if self.config.write_paths:
            lines.append("(allow file-write*")
            for path in self.config.write_paths:
                lines.append(f"    (subpath \"{path}\")")
            # Always allow writing to /dev/null and /dev/tty
            lines.append("    (literal \"/dev/null\")")
            lines.append("    (literal \"/dev/tty\")")
            lines.append(")")

        return "\n".join(lines)

    def execute_command(self, command: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """Execute a command within the generated sandbox."""
        profile_content = self._generate_profile()
        
        # Write profile to temporary file
        fd, profile_path = tempfile.mkstemp(prefix="aura_sandbox_", suffix=".sb")
        os.close(fd)
        try:
            get_file_write_gateway().write_text(
                profile_path,
                profile_content,
                source="sandbox.macos_sandbox.profile",
            )
                
            sandbox_cmd = ["sandbox-exec", "-f", profile_path] + command
            logger.info("Executing under sandbox: %s", " ".join(command))
            
            result = get_subprocess_gateway().run(
                sandbox_cmd,
                cwd=cwd,
                capture_output=True,
                source="sandbox.macos_sandbox.execute_command",
                accelerator_capability="auto",
            )
            return result
            
        finally:
            try:
                os.remove(profile_path)
            except OSError as exc:
                logger.debug("Failed to remove temporary sandbox profile %s: %s", profile_path, exc)
