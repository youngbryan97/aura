"""
Sandbox for autonomous code execution.

Enforcement: command allowlisting per security level, rlimits on child
processes, workdir containment, env-var stripping.

Limitations: no filesystem or network namespace isolation. True sandboxing
on macOS requires sandbox-exec or a container runtime. This is defense-in-depth,
not a hard security boundary.
"""
import os
import sys
import resource
import tempfile
import shutil
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple
import subprocess
import signal
import time

# Try to import Unix-specific modules
try:
    import grp
    import pwd
    HAS_UNIX = True
except ImportError:
    HAS_UNIX = False

logger = logging.getLogger("security.sandbox")


class SecurityLevel(Enum):
    """Security isolation levels"""
    UNTRUSTED = auto()     # Maximum restrictions
    RESTRICTED = auto()    # Restricted FS access, no network
    TRUSTED = auto()       # Controlled access with logging
    PRIVILEGED = auto()    # Full access (internal only)


# Command allowlists per security level
_ALLOWED_COMMANDS = {
    SecurityLevel.UNTRUSTED: frozenset(),  # Nothing allowed
    SecurityLevel.RESTRICTED: frozenset({
        "python", "python3",
    }),
    SecurityLevel.TRUSTED: frozenset({
        "python", "python3", "git", "pip",
    }),
    SecurityLevel.PRIVILEGED: None,  # All commands (internal use only)
}


@dataclass
class ResourceLimits:
    """Resource limits for sandbox"""
    cpu_time_seconds: float = 30.0
    memory_mb: int = 512
    max_processes: int = 50
    max_open_files: int = 100
    max_file_size_mb: int = 10
    wall_clock_seconds: float = 60.0

    def to_rlimit_args(self) -> Dict[int, Tuple[int, int]]:
        """Convert to resource limit arguments"""
        limits = {}

        if not HAS_UNIX:
            return limits

        # CPU time (seconds)
        limits[resource.RLIMIT_CPU] = (
            int(self.cpu_time_seconds),
            int(self.cpu_time_seconds) + 1
        )

        # Memory (bytes)
        memory_bytes = self.memory_mb * 1024 * 1024
        limits[resource.RLIMIT_AS] = (memory_bytes, memory_bytes)

        # Processes/Threads
        try:
            limits[resource.RLIMIT_NPROC] = (self.max_processes, self.max_processes)
        except ValueError:
            pass

        # File descriptors
        try:
            limits[resource.RLIMIT_NOFILE] = (self.max_open_files, self.max_open_files)
        except ValueError:
            pass

        # File size (bytes)
        file_size_bytes = self.max_file_size_mb * 1024 * 1024
        limits[resource.RLIMIT_FSIZE] = (file_size_bytes, file_size_bytes)

        return limits


@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_used_mb: float
    security_violations: List[str]
    metrics: Dict[str, Any]


class SecurityViolation(Exception):
    """Security policy violation"""
    pass


class SecureSandbox:
    """Execution environment with resource limits and command allowlisting.

    Enforces:
    - Command allowlisting based on security level
    - Workdir containment (child process cwd)
    - rlimits on CPU time, memory, file descriptors, file size
    - Stdout/stderr size caps (1MB)
    - Sensitive env-var stripping

    Does NOT provide: filesystem isolation, network isolation, or
    mount namespaces. On macOS, true isolation requires sandbox-exec
    or a container runtime.
    """

    MAX_OUTPUT_BYTES = 1024 * 1024  # 1MB output cap

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.RESTRICTED,
        workdir: Optional[Path] = None,
        allowed_paths: Optional[List[Path]] = None,
        allowed_commands: Optional[List[str]] = None
    ):
        self.security_level = security_level
        self.allowed_paths = [p.resolve() for p in (allowed_paths or [])]
        self.allowed_commands = set(allowed_commands or [])

        # Merge with level-based allowlist
        level_commands = _ALLOWED_COMMANDS.get(security_level)
        if level_commands is not None:
            self.allowed_commands = self.allowed_commands | set(level_commands)
        else:
            self.allowed_commands = None  # None = all allowed (PRIVILEGED)

        # Create isolated workspace
        if workdir:
            self.workdir = Path(workdir).resolve()
            get_task_tracker().create_task(get_storage_gateway().create_dir(self.workdir, cause='SecureSandbox.__init__'))
            self._cleanup_workdir = False
        else:
            self.workdir = Path(tempfile.mkdtemp(prefix="sandbox_")).resolve()
            self._cleanup_workdir = True

        self.resource_limits = ResourceLimits()
        self.violations: List[str] = []
        self.execution_history: List[ExecutionResult] = []

        logger.info(
            "Sandbox initialized at %s (level: %s)", self.workdir, security_level.name
        )

    def _validate_command(self, cmd: List[str]) -> List[str]:
        """Validate command against allowlist."""
        if not cmd:
            raise SecurityViolation("Empty command")

        binary = Path(cmd[0]).name  # Basename only
        if self.allowed_commands is not None and binary not in self.allowed_commands:
            raise SecurityViolation(
                f"Command '{binary}' not in allowlist: {self.allowed_commands}"
            )

        # No metacharacter filtering — we use subprocess.Popen with a list
        # (no shell=True), so shell metacharacters have no special meaning.
        # The allowlist above is the actual security boundary.

        return cmd

    def execute_command(
        self,
        cmd: List[str],
        timeout: float = 30.0,
        input_data: Optional[str] = None
    ) -> ExecutionResult:
        """Execute command with resource limits, allowlisting, and monitoring."""
        start_time = time.time()
        violations = []

        # Validate command before execution
        try:
            cmd = self._validate_command(cmd)
        except SecurityViolation as sv:
            violations.append(str(sv))
            logger.warning("Sandbox blocked command: %s", sv)
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(sv),
                execution_time=0.0,
                memory_used_mb=0.0,
                security_violations=violations,
                metrics={}
            )

        try:
            # Build environment with restricted vars
            env = os.environ.copy()
            # Strip sensitive env vars from sandbox processes
            for key in list(env.keys()):
                if any(s in key.upper() for s in (
                    "TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH"
                )):
                    del env[key]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if input_data else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workdir),
                env=env,
                preexec_fn=self._set_resource_limits if HAS_UNIX else None
            )

            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=timeout
                )
                # Cap output size
                if len(stdout) > self.MAX_OUTPUT_BYTES:
                    stdout = stdout[:self.MAX_OUTPUT_BYTES] + "\n[OUTPUT TRUNCATED]"
                if len(stderr) > self.MAX_OUTPUT_BYTES:
                    stderr = stderr[:self.MAX_OUTPUT_BYTES] + "\n[OUTPUT TRUNCATED]"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                violations.append("Execution timeout")

            exit_code = process.returncode
            if exit_code != 0:
                violations.append(f"Non-zero exit code: {exit_code}")

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=exit_code == 0 and not violations,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                memory_used_mb=0.0,
                security_violations=violations,
                metrics={
                    "start_time": start_time,
                    "end_time": time.time(),
                    "security_level": self.security_level.name,
                }
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                memory_used_mb=0.0,
                security_violations=[str(e)],
                metrics={}
            )

    def _set_resource_limits(self) -> None:
        """Set resource limits for child process"""
        if not HAS_UNIX:
            return

        for resource_id, limits in self.resource_limits.to_rlimit_args().items():
            try:
                resource.setrlimit(resource_id, limits)
            except (ValueError, OSError):
                pass  # Non-critical fallback

    def cleanup(self):
        """Clean up the sandbox workdir if we created it."""
        if self._cleanup_workdir and self.workdir.exists():
            try:
                get_task_tracker().create_task(get_storage_gateway().delete_tree(self.workdir, cause='SecureSandbox.cleanup'))
                logger.debug("Sandbox workdir cleaned: %s", self.workdir)
            except Exception as e:
                logger.warning("Failed to clean sandbox workdir: %s", e)
