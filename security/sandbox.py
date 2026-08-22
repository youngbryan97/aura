"""Sandbox for autonomous code execution.

What the kernel enforces, on macOS, for every level except ``PRIVILEGED``: a
``sandbox-exec`` seatbelt profile that denies network outright, confines
writes to the workdir and the temp roots, and — when the caller names
``read_paths`` — confines reads to the system roots a program needs to load
plus the paths it was given. Those denials are refused by the kernel, not by
this module, so a command that ignores them fails rather than succeeds.

What this module enforces on top: rlimits on the child (CPU, address space,
processes, descriptors, file size), a 1MB cap on captured output, workdir
containment for the child's cwd, and an environment stripped of secrets
through the same classifier the subprocess gateway refuses spawns over.

What it does not do: process, mount or PID namespace separation, which macOS
does not offer outside a VM or container runtime; and off macOS the seatbelt
step is absent, leaving the allowlist and the rlimits. ``ExecutionResult``
does not hide that difference —
:func:`core.security.sandbox._kernel_enforced` reports which boundary the
caller actually got, because ``exit_code: 0`` from a degraded sandbox reads
identically to one from an enforced one.

Authorization is a separate question from isolation and is answered
elsewhere: :mod:`core.security.execution_authority` asks the Will and fails
closed. ``SecurityLevel.CONFINED`` exists so a caller that has already asked
gets the isolation without a second, weaker allowlist deciding for it.
"""
import asyncio
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import AcceleratorCapability, get_subprocess_gateway

HAS_UNIX = os.name == "posix"
_SANDBOX_EXECUTION_ERRORS = (
    OSError,
    subprocess.SubprocessError,
    UnicodeError,
    ValueError,
)
_RESOURCE_LIMIT_ERRORS = (OSError, ValueError)

logger = logging.getLogger("security.sandbox")


class SecurityLevel(Enum):
    """Security isolation levels"""
    UNTRUSTED = auto()     # Maximum restrictions
    RESTRICTED = auto()    # Restricted FS access, no network
    TRUSTED = auto()       # Controlled access with logging
    CONFINED = auto()      # Any program, but inside the OS boundary
    PRIVILEGED = auto()    # Full access (internal only)


# Command allowlists per security level
#
# ``CONFINED`` carries no allowlist on purpose. An allowlist is a lexical gate
# deciding a semantic question: it answers "is this binary in my set", never
# "should this run now". The repository already has the surface that answers
# the second question — ``core.security.execution_authority``, which asks the
# Will and refuses when the Will cannot be reached. Levels below CONFINED keep
# their allowlists because their callers have no Will to ask.
#
# What CONFINED does NOT drop is the isolation: the seatbelt profile, the
# rlimits, the secret-stripped environment and the explicit read/write scope
# all still apply, because they are the part that holds when the decision is
# wrong. PRIVILEGED is the only level with neither, and it is the escape
# hatch, not a tier.
_ALLOWED_COMMANDS = {
    SecurityLevel.UNTRUSTED: frozenset(),  # Nothing allowed
    SecurityLevel.RESTRICTED: frozenset({
        "python", "python3",
    }),
    SecurityLevel.TRUSTED: frozenset({
        "python", "python3", "git", "pip",
    }),
    SecurityLevel.CONFINED: None,  # Authorization is the Will's job, not a list
    SecurityLevel.PRIVILEGED: None,  # All commands (internal use only)
}

# The roots that hold a person's files. Reads here are denied whenever a
# caller names ``read_paths``, and the workdir is re-allowed afterwards
# because it usually sits inside one of them. Denying by root rather than by
# home directory covers other accounts and mounted volumes as well, which an
# ``expanduser("~")`` would not.
_USER_DATA_READ_ROOTS: tuple[str, ...] = (
    "/Users",
    "/Volumes",
    "/private/var/root",
    # Every per-user temp container on the machine, and the shared one. A
    # confined command has its own scratch space inside the workdir; it has
    # no business reading what another process left in /tmp.
    "/private/var/folders",
    "/private/tmp",
)


def _dedupe_paths(paths: "list[Path]") -> "list[Path]":
    """Resolved, ordered, no duplicates and no path already covered by a parent.

    A seatbelt profile with ``/usr`` and ``/usr/bin`` in it is not wrong, but
    the redundant entry hides which root actually granted an access when the
    profile is read back during an incident.
    """
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        try:
            candidate = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate)
    kept: list[Path] = []
    for candidate in resolved:
        if any(other == candidate or other in candidate.parents for other in resolved if other != candidate):
            continue
        kept.append(candidate)
    return kept


@dataclass
class ResourceLimits:
    """Resource limits for sandbox"""
    cpu_time_seconds: float = 30.0
    memory_mb: int = 512
    max_processes: int = 50
    max_open_files: int = 100
    max_file_size_mb: int = 10
    wall_clock_seconds: float = 60.0

    def to_rlimit_args(self) -> dict[int, tuple[int, int]]:
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


@dataclass(frozen=True)
class ConfinedLaunch:
    """A spawn that has not happened yet, with its confinement already applied.

    ``kernel_enforced`` is the honest half: on a platform with no seatbelt the
    argv carries no prefix and the boundary is the allowlist plus the rlimits.
    A caller that reports "sandboxed" without reading this field is repeating
    the defect the module docstring names.
    """

    argv: list[str]
    env: dict[str, str]
    preexec_fn: "Callable[[], None] | None"
    cwd: str
    kernel_enforced: bool
    profile_path: str | None


@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_used_mb: float
    security_violations: list[str]
    metrics: dict[str, Any]


class SecurityViolationError(Exception):
    """Security policy violation"""


SecurityViolation = SecurityViolationError


class SecureSandbox:
    """Execution environment with resource limits and command allowlisting.

    Enforces:
    - Command allowlisting based on security level
    - Workdir containment (child process cwd)
    - rlimits on CPU time, memory, file descriptors, file size
    - Stdout/stderr size caps (1MB)
    - Sensitive env-var stripping
    
    macOS Hardening:
    - Automatically injects `sandbox-exec` with a strict version 1 profile
    - Denies network access entirely
    - Restricts file-write strictly to the workdir/tmp
    """

    MAX_OUTPUT_BYTES = 1024 * 1024  # 1MB output cap

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.RESTRICTED,
        workdir: Path | None = None,
        allowed_paths: list[Path] | None = None,
        allowed_commands: list[str] | None = None,
        read_paths: list[Path] | None = None,
        resource_limits: "ResourceLimits | None" = None,
    ):
        self.security_level = security_level
        self.allowed_paths = [p.resolve() for p in (allowed_paths or [])]
        self.allowed_commands = set(allowed_commands or [])
        # ``None`` means "read anything the OS lets this user read", which is
        # what every caller got before read scope existed. A list means the
        # profile names its read roots and denies the rest.
        self.read_paths: list[Path] | None = (
            [Path(p).resolve() for p in read_paths] if read_paths is not None else None
        )

        # Merge with level-based allowlist
        level_commands = _ALLOWED_COMMANDS.get(security_level)
        if level_commands is not None:
            self.allowed_commands = self.allowed_commands | set(level_commands)
        else:
            self.allowed_commands = None  # None = all allowed (PRIVILEGED)

        # Create isolated workspace
        if workdir:
            self.workdir = Path(workdir).resolve()
            self.workdir.mkdir(parents=True, exist_ok=True)
            self._cleanup_workdir = False
        else:
            self.workdir = Path(tempfile.mkdtemp(prefix="sandbox_")).resolve()
            self._cleanup_workdir = True

        self.resource_limits = resource_limits or ResourceLimits()
        self.violations: list[str] = []
        self.execution_history: list[ExecutionResult] = []

        logger.info(
            "Sandbox initialized at %s (level: %s)", self.workdir, security_level.name
        )

    def _validate_command(self, cmd: list[str]) -> list[str]:
        """Validate command against allowlist."""
        if not cmd:
            raise SecurityViolationError("Empty command")

        binary_path = Path(cmd[0])
        binary = binary_path.name  # Basename only
        is_python_executable = (
            binary == os.path.basename(sys.executable)
            or (binary.startswith("python") and all(c.isdigit() or c == "." for c in binary[6:]))
        )
        allowed_list = self.allowed_commands
        if allowed_list is not None:
            is_allowed = binary in allowed_list
            if not is_allowed and ("python" in allowed_list or "python3" in allowed_list):
                if is_python_executable:
                    is_allowed = True
            if not is_allowed:
                raise SecurityViolationError(
                    f"Command '{binary}' not in allowlist: {self.allowed_commands}"
                )

        if self.security_level != SecurityLevel.PRIVILEGED and binary_path.parent != Path("."):
            self._validate_canonical_binary(binary_path, binary)

        # No metacharacter filtering — we use subprocess.Popen with a list
        # (no shell=True), so shell metacharacters have no special meaning.
        # The allowlist above is the actual security boundary.

        return cmd

    def _validate_canonical_binary(self, binary_path: Path, binary: str) -> None:
        """Reject path-based allowlist bypasses for restricted commands."""
        allowed_targets = {os.path.realpath(sys.executable)}
        discovered = shutil.which(binary)
        if discovered:
            allowed_targets.add(os.path.realpath(discovered))

        try:
            candidate = os.path.realpath(binary_path)
        except OSError as exc:
            raise SecurityViolationError(
                f"Command path could not be resolved: {binary_path}"
            ) from exc

        if candidate not in allowed_targets:
            raise SecurityViolationError(
                "Command path is not an approved runtime binary: "
                f"{binary_path}"
            )

    @staticmethod
    def _sandbox_profile_literal(path: Path) -> str:
        """Escape paths embedded in a sandbox-exec string literal."""
        return str(path.absolute()).replace("\\", "\\\\").replace('"', '\\"')

    def build_seatbelt_profile(self) -> str:
        """The macOS seatbelt profile this sandbox will run under.

        Public because a profile nobody can read is a claim nobody can check:
        ``tests/test_shell_execution_isolation.py`` asserts the denials, and
        then runs a real command under them, because a profile that never
        reaches the kernel proves nothing.

        Read scope is expressed as a denial of user data rather than an
        allowlist of system paths, and that is a measurement, not a
        preference. An allowlist was tried first: ``/System /Library /usr
        /bin /sbin /opt /private/etc /private/var/db/dyld`` plus the workdir.
        Under it ``/bin/cat`` aborts with SIGABRT before ``main`` — dyld on
        this OS reaches for more than any hand-written list contains, and the
        failure is indistinguishable from a broken sandbox. Seatbelt takes
        the LAST matching rule, so the working shape is: allow reads, deny
        the roots that hold a person's data, then re-allow the workdir, which
        normally sits inside one of those roots. ``cat ~/.ssh/id_rsa`` is
        refused by the kernel; ``/bin/cat`` still starts.

        Write scope is the workdir and the caller's ``allowed_paths``. The
        whole of ``/private/tmp`` and ``/private/var/folders`` used to be
        writable — the shared temp directory and every per-user temp
        container on the machine. A confined command gets its own scratch
        space instead: :meth:`prepare_launch` points its ``TMPDIR`` at a
        directory inside the workdir, so the programs that need a temp file
        still get one and it lands inside the boundary.
        """
        write_roots = [self.workdir]
        write_roots.extend(self.allowed_paths)

        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process-exec*)",
            "(allow process-fork)",
            "(allow sysctl-read)",
            "(allow ipc-posix-shm)",
            "(allow signal)",
            "(allow mach-lookup)",
            "(allow file-read*)",
        ]

        if self.read_paths is not None:
            lines.append("(deny file-read*")
            for root in _USER_DATA_READ_ROOTS:
                lines.append(f'    (subpath "{root}")')
            lines.append(")")
            lines.append("(allow file-read*")
            for root in _dedupe_paths([self.workdir, *self.read_paths]):
                lines.append(f'    (subpath "{self._sandbox_profile_literal(root)}")')
            lines.append(")")

        lines.append("(allow file-write*")
        for root in _dedupe_paths(write_roots):
            lines.append(f'    (subpath "{self._sandbox_profile_literal(root)}")')
        lines.append('    (literal "/dev/null")')
        lines.append('    (literal "/dev/tty")')
        lines.append(")")

        lines.append("(deny network*)")
        return "\n".join(lines) + "\n"

    def secret_free_environment(self) -> dict[str, str]:
        """This process's environment with every sensitive key removed.

        Stripped with the SAME classifier the subprocess gateway refuses
        spawns over. This list used to be its own — TOKEN/SECRET/PASSWORD/
        KEY/CREDENTIAL/AUTH — and the gateway's is broader (session_id,
        cookie, cert, bearer, passphrase, signature, ssn). So the sandbox
        stripped what it knew about, the gateway then refused the spawn over
        what it did not, and the sandbox could not launch at all. Two
        definitions of "sensitive", disagreeing.

        Sharing the classifier means stripping is exactly what passing is,
        and anything added to one is honoured by both.
        """
        try:
            from core.security.structural_redaction import is_sensitive_key
        except ImportError:  # pragma: no cover - keep the sandbox launchable
            def is_sensitive_key(key: str) -> bool:
                return any(
                    marker in key.upper()
                    for marker in ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "AUTH")
                )
        env = os.environ.copy()
        for key in list(env.keys()):
            if is_sensitive_key(key):
                del env[key]
        return env

    def kernel_enforced(self) -> bool:
        """Will the OS refuse this child's forbidden syscalls, or only the list?"""
        return sys.platform == "darwin" and self.security_level != SecurityLevel.PRIVILEGED

    def prepare_launch(self, cmd: list[str], *, validate: bool = True) -> ConfinedLaunch:
        """Everything a spawn needs to be confined, without doing the spawn.

        The sync path in :meth:`execute_command` and every async caller that
        needs streaming, a background job or a persistent session all want the
        same four things: a validated argv, the seatbelt prefix, a secret-free
        environment and the child rlimits. Before this existed the sync path
        had them and async callers wrote a second, weaker version — which is
        how a module whose own docstring warns against "a second, weaker idea
        of what a sandbox is" grows one.

        Raises :class:`SecurityViolationError` when ``validate`` and the
        command does not pass this level's checks.
        """
        argv = list(cmd)
        if validate:
            argv = self._validate_command(argv)

        profile_path: str | None = None
        if self.kernel_enforced():
            written = self.workdir / ".sandbox_profile.sb"
            atomic_write_text(written, self.build_seatbelt_profile(), encoding="utf-8")
            written.chmod(0o600)
            profile_path = str(written)
            argv = ["sandbox-exec", "-f", profile_path] + argv

        env = self.secret_free_environment()
        # The child's scratch space, inside the boundary. Without this the
        # child inherits a TMPDIR the profile does not grant, and every
        # program that writes a temp file fails in a way that looks like a
        # bug in the program.
        scratch = self.workdir / ".tmp"
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(scratch)
        except OSError as exc:
            record_degradation(
                "security.sandbox",
                exc,
                action="left the child's TMPDIR unset; temp writes will be refused",
                extra={"workdir": str(self.workdir)},
            )

        return ConfinedLaunch(
            argv=argv,
            env=env,
            preexec_fn=self._set_resource_limits if HAS_UNIX else None,
            cwd=str(self.workdir),
            kernel_enforced=self.kernel_enforced(),
            profile_path=profile_path,
        )

    def execute_command(
        self,
        cmd: list[str],
        timeout: float = 30.0,
        input_data: str | None = None
    ) -> ExecutionResult:
        """Execute command with resource limits, allowlisting, and monitoring."""
        start_time = time.time()
        violations = []

        # Validate command before execution
        try:
            cmd = self._validate_command(cmd)
        except SecurityViolationError as sv:
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
            launch = self.prepare_launch(cmd, validate=False)
            env = launch.env
            cmd = list(launch.argv)

            process = get_subprocess_gateway().spawn(
                cmd,
                stdin=subprocess.PIPE if input_data else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workdir),
                env=env,
                preexec_fn=self._set_resource_limits if HAS_UNIX else None,
                source="security.sandbox.execute_command",
                accelerator_capability=AcceleratorCapability.NONE,
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
        except _SANDBOX_EXECUTION_ERRORS as e:
            record_degradation("sandbox", e)
            logger.exception("Sandbox execution failed before completion")
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

    async def execute_command_async(
        self,
        cmd: list[str],
        timeout: float = 30.0,
        input_data: str | None = None,
    ) -> ExecutionResult:
        """``execute_command`` from a coroutine, off the event loop.

        ``execute_command`` writes the profile, spawns, and blocks in
        ``communicate`` until the child exits or the timeout fires. Called
        directly from a coroutine that is the event loop stalled for the
        whole run — the same failure mode as the on-loop fsync that froze the
        live loop for twenty minutes. The work is unchanged; only the thread
        it happens on is.
        """
        return await asyncio.to_thread(
            self.execute_command, cmd, timeout=timeout, input_data=input_data
        )

    def _set_resource_limits(self) -> None:
        """Set resource limits for child process"""
        if not HAS_UNIX:
            return

        for resource_id, limits in self.resource_limits.to_rlimit_args().items():
            try:
                resource.setrlimit(resource_id, limits)
            except _RESOURCE_LIMIT_ERRORS:
                continue  # Non-critical fallback inside the child process.

    def cleanup(self):
        """Clean up the sandbox workdir if we created it."""
        if self._cleanup_workdir and self.workdir.exists():
            try:
                shutil.rmtree(self.workdir)
                logger.debug("Sandbox workdir cleaned: %s", self.workdir)
            except OSError as e:
                record_degradation("sandbox.cleanup", e)
                logger.warning("Failed to clean sandbox workdir: %s", e)
