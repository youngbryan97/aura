"""Shell execution: isolation first, then authorization, then the command.

Two properties this skill did not have, and the reason each one mattered.

**It was not isolated.** ``sandbox`` defaulted to ``False``, so the ordinary
path spawned the caller's program straight onto the host with the full
environment, the whole filesystem readable, the network open and no ceiling on
what it could consume. The controls that did exist were a destructive-pattern
blocklist and per-command flag rules — real, useful, and not an OS boundary. A
blocklist answers "does this string look dangerous"; it cannot answer "can this
program read the user's SSH key", and it loses to any spelling nobody thought
of. The isolation already existed in :mod:`security.sandbox`; this skill simply
did not use it. It does now, by default, and the host path is an explicit,
separately authorized escape hatch rather than the default.

**It was not governed.** Every other general-execution surface —
``sovereign_terminal``, ``mcp_client``, ``host_automation`` — routes through
:func:`core.security.execution_authority.authorize_execution`, which asks the
Will and fails closed. This one did not, and the guard that was written to stop
exactly that (``test_general_execution_surfaces_are_governed``) scanned only
``core/`` and only for ``spawn_shell_async``. So the ungoverned surface was in
``skills/`` calling ``spawn_async``, in the blind spot of its own guard. Both
the surface and the guard are fixed here.

**Path containment was string-prefixed.** ``rm`` compared
``resolved.startswith(allowed_root)``, which admits ``/allowed/project-evil``
for a root of ``/allowed/project``: a sibling directory whose name begins with
the root's name is not inside it. ``cd`` in this same file already did it
correctly with ``Path`` ancestry. There is now one helper and both use it.
"""
import asyncio
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.execution_authority import (
    KIND_SHELL,
    authorize_execution,
    release_execution,
)
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.Shell")

# Defence in depth behind the Will's decision and the kernel's boundary, never
# either one's substitute. See core.security.execution_authority for why a
# denylist is not permitted to be the decision.
_BLOCKED_PATTERNS = {"rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:",
                     "chmod -R 777 /", "curl | sh", "wget | sh", "eval ", "nc -l"}

#: Parameter a caller sets to ask for the host instead of the sandbox. Named,
#: not implied: ``sandbox=False`` alone is refused, so a caller that has merely
#: not been updated cannot fall out of isolation by omission.
HOST_EXECUTION_PARAM = "host_execution"


def _workspace_root() -> Path:
    return Path(getattr(config.paths, "base_dir", Path.cwd())).resolve()


def _contains(root: Path, candidate: Path) -> bool:
    """Is ``candidate`` the root itself or inside it?

    ``str.startswith`` is not this question. ``/allowed/project-evil``
    starts with ``/allowed/project`` and is not inside it. Ancestry on
    resolved paths is the containment test; symlinks are already followed by
    ``resolve()``, so a link out of the workspace is caught here rather than
    at open() time.
    """
    return candidate == root or root in candidate.parents


def _resolve_cd_target(cwd: str, target: str) -> tuple[str, bool]:
    new_path = (Path(cwd) / target).expanduser().resolve()
    return str(new_path), _contains(_workspace_root(), new_path)


class ShellSkill(BaseSkill):
    name = "shell"
    description = "Execute terminal commands inside an OS-enforced sandbox."
    inputs = {
        "command": "The shell command to run",
        "timeout": "Timeout in seconds (default 10)",
        "background": "Run the command in the background (default False)",
        "sandbox": "Run inside the OS sandbox (default True)",
        HOST_EXECUTION_PARAM: (
            "Run on the host with no OS boundary. Requires sandbox=False and a "
            "separate authorization; denied by default."
        ),
        "persistent_session_id": "Use a persistent bash session across commands",
    }

    _background_jobs: dict[str, asyncio.subprocess.Process] = {}

    def __init__(self):
        self.cwd = str(_workspace_root())
        os.makedirs(self.cwd, exist_ok=True)

    # Additional restrictions for commands that can be dangerous
    _RESTRICTED_COMMANDS = {
        "rm":      {"max_args": 5, "blocked_flags": {"-rf", "-fr", "--no-preserve-root"}},
        "python":  {"blocked_flags": {"-c"}},
        "python3": {"blocked_flags": {"-c"}},
        "curl":    {"blocked_flags": {"-o", "--output"}},
    }

    def _is_safe_command(self, cmd_str: str) -> tuple:
        """Validate command against blocklist and per-command restrictions.

        Returns (safe, reason). Execution uses the subprocess gateway with a
        pre-split argument list (no shell=True), so shell metacharacters are
        treated as literal strings by the OS.
        """
        cmd_lower = cmd_str.lower()
        for pattern in _BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return False, f"Blocked dangerous pattern: {pattern}"

        try:
            tokens = shlex.split(cmd_str)
        except ValueError:
            return False, "Malformed command string."

        if not tokens:
            return False, "Empty command."

        base_cmd = os.path.basename(tokens[0])

        restrictions = self._RESTRICTED_COMMANDS.get(base_cmd)
        if restrictions:
            blocked_flags = restrictions.get("blocked_flags", set())
            for tok in tokens[1:]:
                if tok in blocked_flags:
                    return False, f"Flag '{tok}' is blocked for '{base_cmd}'."
            max_args = restrictions.get("max_args")
            if max_args and len(tokens) - 1 > max_args:
                return False, f"Too many arguments for '{base_cmd}' (max {max_args})."

        if base_cmd == "rm":
            root = _workspace_root()
            for arg in tokens[1:]:
                if arg.startswith("-"):
                    continue
                resolved = (Path(self.cwd) / arg).expanduser().resolve()
                if not _contains(root, resolved):
                    return False, f"rm blocked: '{arg}' resolves outside workspace."

        full_path = shutil.which(tokens[0])
        if full_path:
            suspicious_dirs = ["/tmp", "/var/tmp", "/dev/shm"]
            for s_dir in suspicious_dirs:
                if full_path.startswith(s_dir):
                    logger.warning("Suspicious binary location: '%s' -> '%s'", cmd_str, full_path)
                    break

        return True, "ok"

    def _build_sandbox(self):
        """The confinement this skill runs commands under.

        ``CONFINED`` rather than ``TRUSTED``: the authorization decision was
        already made by the Will a few lines above, and stacking a four-entry
        binary allowlist on top of it would refuse everything the Will
        approved that is not python, git or pip — which is how a governed
        surface ends up narrower than the ungoverned one it replaced. What
        CONFINED keeps is the part that holds when a decision is wrong: the
        seatbelt profile, the rlimits, the stripped environment, and read and
        write scope that name the workspace instead of the disk.
        """
        from security.sandbox import SecureSandbox, SecurityLevel

        root = _workspace_root()
        workdir = Path(self.cwd)
        return SecureSandbox(
            security_level=SecurityLevel.CONFINED,
            workdir=workdir,
            allowed_paths=[root],
            read_paths=[root],
        )

    async def execute(self, goal: dict, context: dict) -> dict:
        params = goal.get("params", {}) or {}
        cmd_str = params.get("command", "")
        timeout = params.get("timeout", 10)
        background = params.get("background", False)
        use_sandbox = params.get("sandbox", True)
        host_requested = bool(params.get(HOST_EXECUTION_PARAM, False))

        if not cmd_str:
            return {"ok": False, "error": "No command provided."}

        safe, reason = self._is_safe_command(cmd_str)
        if not safe:
            logger.warning("Shell BLOCKED: %s — %s", cmd_str, reason)
            return {"ok": False, "error": f"Command blocked: {reason}"}

        try:
            argv = shlex.split(cmd_str)
        except ValueError:
            return {"ok": False, "error": "Malformed command string."}
        base_cmd = argv[0]

        # `cd` never spawns anything; the boundary it needs is containment,
        # which _resolve_cd_target applies, and there is no process for the
        # Will to authorize.
        if os.path.basename(base_cmd) == "cd":
            if len(argv) > 1:
                new_path, allowed = await asyncio.to_thread(
                    _resolve_cd_target, self.cwd, argv[1]
                )
                if not allowed:
                    return {"ok": False, "error": "Access Denied: Cannot leave workspace."}
                self.cwd = new_path
                return {"ok": True, "summary": f"Changed directory to {self.cwd}"}
            return {"ok": True, "summary": f"Working directory is {self.cwd}"}

        if not use_sandbox and not host_requested:
            return {
                "ok": False,
                "error": (
                    "Unsandboxed execution must be asked for by name: pass "
                    f"{HOST_EXECUTION_PARAM}=true alongside sandbox=false."
                ),
            }
        isolation = "host" if (host_requested and not use_sandbox) else "sandbox"

        verdict = await authorize_execution(
            KIND_SHELL,
            argv,
            source="skills.shell",
            cwd=self.cwd,
            extra={
                # A standing directive is written against these values. "Never
                # run anything outside the sandbox" is one rule over this field,
                # not a rule per command.
                "isolation": isolation,
                "background": bool(background),
                "persistent_session": bool(params.get("persistent_session_id")),
            },
        )
        if not verdict.approved:
            return verdict.as_error()

        logger.info(
            "Shell Execution: [%s] (timeout=%s, bg=%s, isolation=%s)",
            cmd_str, timeout, background, isolation,
        )

        succeeded = False
        failure = ""
        try:
            if isolation == "host":
                result = await self._run_on_host(argv, timeout, background)
            else:
                result = await self._run_confined(
                    argv, timeout, background, params.get("persistent_session_id")
                )
            succeeded = bool(result.get("ok"))
            failure = "" if succeeded else str(result.get("error", "") or "")
            result["governance"] = verdict.receipt()
            return result
        except subprocess.TimeoutExpired:
            failure = "timeout"
            return {"ok": False, "error": "Command timed out."}
        except (ImportError, AttributeError, RuntimeError, OSError) as e:
            failure = str(e)
            record_degradation("shell", e)
            logger.debug("Shell command execution failed: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            # The grant closes whether the command worked or not. A capability
            # token that outlives its command is a live grant nobody tracks.
            release_execution(
                verdict, source="skills.shell", success=succeeded, error=failure
            )

    async def _run_confined(
        self,
        argv: list[str],
        timeout: float,
        background: bool,
        persistent_session_id: str | None,
    ) -> dict:
        """Run inside the OS boundary.

        The persistent-session path is deliberately absent here rather than
        silently falling through to an unconfined daemon: ``bash_daemon`` keeps
        one long-lived host shell per session id, so a command routed to it
        would report ``sandbox: True`` while running with none of it. A caller
        that needs the session must ask for the host explicitly and be
        authorized for it.
        """
        if persistent_session_id:
            return {
                "ok": False,
                "error": (
                    "A persistent session is a long-lived host shell and cannot "
                    f"be confined; ask for {HOST_EXECUTION_PARAM}=true with "
                    "sandbox=false if that is what you need."
                ),
            }

        sandbox = await asyncio.to_thread(self._build_sandbox)
        launch = await asyncio.to_thread(sandbox.prepare_launch, argv)
        process = await get_subprocess_gateway().spawn_async(
            launch.argv,
            cwd=launch.cwd,
            env=launch.env,
            preexec_fn=launch.preexec_fn,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="tool_execution:shell.skill",
            accelerator_capability="auto",
        )
        return await self._collect(
            process,
            argv,
            timeout,
            background,
            extra={"sandbox": True, "kernel_enforced": launch.kernel_enforced},
        )

    async def _run_on_host(self, argv: list[str], timeout: float, background: bool) -> dict:
        process = await get_subprocess_gateway().spawn_async(
            argv,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="tool_execution:shell.skill.host",
            accelerator_capability="auto",
        )
        return await self._collect(
            process,
            argv,
            timeout,
            background,
            extra={"sandbox": False, "kernel_enforced": False},
        )

    async def _collect(
        self,
        process: asyncio.subprocess.Process,
        argv: list[str],
        timeout: float,
        background: bool,
        *,
        extra: dict,
    ) -> dict:
        if background:
            job_id = f"job_{process.pid}"
            self._background_jobs[job_id] = process
            return {
                "ok": True,
                "job_id": job_id,
                "message": (
                    f"Process {process.pid} started in background. "
                    "Use command_status to check."
                ),
                **extra,
            }

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=float(timeout)
            )
        except TimeoutError:
            try:
                process.kill()
                await process.wait()
            except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as exc:
                record_degradation(
                    "shell",
                    exc,
                    severity="warning",
                    action="returned shell timeout result after process cleanup failed",
                    extra={"command": shlex.join(argv)[:240], "timeout_s": timeout},
                )
                logger.debug("Shell timed-out process cleanup failed: %s", exc)
            return {"ok": False, "error": f"Command timed out after {timeout}s.", **extra}

        return {
            "ok": process.returncode == 0,
            "stdout": stdout.decode(errors="replace").strip(),
            "stderr": stderr.decode(errors="replace").strip(),
            "cwd": self.cwd,
            **extra,
        }
