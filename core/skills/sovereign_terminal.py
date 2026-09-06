import asyncio
import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.execution_authority import (
    KIND_OPEN,
    KIND_SHELL,
    authorize_execution,
    release_execution,
)
from core.skills.base_skill import BaseSkill
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Skills.SovereignTerminal")


def _resolve_terminal_path(base: str, target: str) -> Path:
    return (Path(base) / target).expanduser().resolve()


def _resolve_allowed_root() -> Path:
    return Path(getattr(config.paths, "base_dir", "/")).expanduser().resolve()


class TerminalInput(BaseModel):
    action: str = Field("execute", description="Action: 'execute', 'open_app', 'open_file', 'cd'")
    command: str | None = Field(None, description="Shell command to run (for 'execute').")
    target: str | None = Field(None, description="App name or file path (for 'open' actions).")
    cwd: str | None = Field(None, description="Current working directory for execution or 'cd'.")
    timeout: int = Field(15, description="Timeout in seconds for execution.")

class SovereignTerminalSkill(BaseSkill):
    """The unified terminal and system operation capability for Aura.
    Handles shell command execution, application launching, and file opening.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    
    name = "sovereign_terminal"
    description = "Execute shell commands, launch system apps, and open files via CLI."
    input_model = TerminalInput
    
    def __init__(self):
        super().__init__()
        # Use workspace root as default CWD if available
        self.default_cwd = str(getattr(config.paths, "base_dir", os.getcwd()))

    async def execute(self, params: TerminalInput, context: dict[str, Any]) -> dict[str, Any]:
        """Unified entry point for all system operations."""
        if isinstance(params, dict):
            try:
                params = TerminalInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('sovereign_terminal', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        action = params.action
        cwd = params.cwd or self.default_cwd
        
        try:
            if action == "execute":
                return await self._run_command(params.command, cwd, params.timeout)
            elif action in ["open_app", "open_file"]:
                return await self._open_target(params.target, action)
            elif action == "cd":
                new_path = str(await asyncio.to_thread(_resolve_terminal_path, cwd, params.target or "."))
                # Security shortcut: Ensure we stay in workspace if it's strictly enforced (optional for sovereign)
                return {"ok": True, "new_cwd": new_path, "message": f"Directory changed to {new_path}"}
            else:
                return {"ok": False, "error": f"Unsupported terminal action: {action}"}
        except OSError as e:
            record_degradation('sovereign_terminal', e)
            logger.error("Terminal skill failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _run_command(self, cmd: str, cwd: str, timeout_s: int) -> dict[str, Any]:
        if not cmd:
            return {"ok": False, "error": "Execute action requires a 'command'."}

        # The decision is the Will's, and it is made BEFORE the denylists
        # below — not after, and not instead of them.
        #
        # This skill runs arbitrary shell, which is the widest capability in
        # the system, and until now it asked nobody: the substring checks
        # below were the entire gate. A substring check is a lexical test
        # standing in for a semantic question, so it refuses `rm -rf /` and
        # waves through `rm -rf "$HOME"`, `find . -delete`, or any of the
        # spellings nobody enumerated. Keeping them is worthwhile as a cheap
        # second layer; treating them as the gate was the defect.
        verdict = await authorize_execution(
            KIND_SHELL,
            cmd,
            source="tool_execution:sovereign_terminal.shell",
            cwd=cwd,
            extra={"timeout_s": int(timeout_s)},
        )
        if not verdict.approved:
            return verdict.as_error()

        try:
            return await self._run_authorized_command(cmd, cwd, timeout_s, verdict)
        finally:
            release_execution(verdict, source="sovereign_terminal.shell")

    async def _run_authorized_command(
        self,
        cmd: str,
        cwd: str,
        timeout_s: int,
        verdict: Any,
    ) -> dict[str, Any]:
        # --- Defence in depth, behind the Will's decision ---
        normalized_cmd = cmd.lower()
        obfuscation_patterns = ["base64 -d", "base64 --decode", "\\x", "\\u", "${", "eval $(", "echo -e"]
        if any(p in normalized_cmd for p in obfuscation_patterns):
            logger.warning("🛡️ Potential obfuscation bypass attempt: %s", cmd)
            return {"ok": False, "error": "Command blocked: Obfuscation patterns detected."}

        destructive_patterns = [
            "rm -rf /", "rm -rf *", ":(){ :|:& };:", "dd if=/dev/", 
            "mkfs.", "chmod -r 777", "chown -r", "> /dev/sda",
            "shutdown", "reboot", "halt", "poweroff"
        ]
        
        for pattern in destructive_patterns:
            if pattern in normalized_cmd:
                logger.warning("🛡️ Destructive command blocked: %s", pattern)
                return {"ok": False, "error": f"Command blocked: Destructive operation '{pattern}' detected."}

        # RM Specific Guard: rm must only operate on relative paths within workspace
        # We parse the command for 'rm' but 'execute' can be anything, so we look for 'rm ' anywhere
        if "rm " in normalized_cmd:
            tokens = shlex.split(cmd)
            for i, tok in enumerate(tokens):
                if tok == "rm":
                    for arg in tokens[i + 1:]:
                        if arg.startswith("-"):
                            continue
                        resolved = await asyncio.to_thread(_resolve_terminal_path, cwd, arg)
                        allowed_root = await asyncio.to_thread(_resolve_allowed_root)
                        if allowed_root != Path("/") and resolved != allowed_root and allowed_root not in resolved.parents:
                            logger.warning("🛡️ RM blocked: path %s is outside %s", resolved, allowed_root)
                            return {"ok": False, "error": f"rm blocked: '{arg}' resolves outside sanctioned path."}

        logger.info("🐚 Shell Execute: %s (CWD: %s)", cmd, cwd)
        
        try:
            process = await get_subprocess_gateway().spawn_shell_async(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="tool_execution:sovereign_terminal.shell",
                accelerator_capability="auto",
            )
            
            stdout_chunks = []
            stderr_chunks = []
            
            async def read_stream(stream, chunks_list):
                interactive_prompts = [b"password:", b"y/n", b"yes/no", b"enter ", b"continue?"]
                try:
                    while stream is not None and not stream.at_eof():
                        line = await stream.read(4096)
                        if not line:
                            break
                        chunks_list.append(line)
                        
                        # Anti-hang heuristic: look for interactive stall markers
                        lower_line = line.lower()
                        if any(p in lower_line for p in interactive_prompts):
                            # If the terminal hasn't flushed a newline and is stalled waiting
                            pass  # no-op: intentional
                except ValueError:
                    pass  # no-op: intentional
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(process.stdout, stdout_chunks),
                        read_stream(process.stderr, stderr_chunks),
                        process.wait()
                    ),
                    timeout=float(timeout_s)
                )
            except TimeoutError:
                try:
                    process.kill()
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation(
                        "sovereign_terminal",
                        e,
                        severity="warning",
                        action="returned terminal timeout result after process kill failed",
                        extra={"pid": getattr(process, "pid", None), "command": cmd[:240]},
                    )
                    logger.debug("Failed to kill process %s: %s", process.pid, e)
                
                stdout_str = b"".join(stdout_chunks).decode(errors="replace")
                stderr_str = b"".join(stderr_chunks).decode(errors="replace")
                return {
                    "ok": False,
                    "error": "Execution timed out or hung on interactive prompt.",
                    "stdout": self._smart_truncate(stdout_str),
                    "stderr": self._smart_truncate(stderr_str),
                    "summary": self._build_command_summary(
                        cmd,
                        stdout_str,
                        stderr_str or "Execution timed out or hung on interactive prompt.",
                        return_code=None,
                    ),
                    "governance": verdict.receipt(),
                }

            stdout_str = b"".join(stdout_chunks).decode(errors="replace")
            stderr_str = b"".join(stderr_chunks).decode(errors="replace")
            
            # If command exited with error, it's ok=False conceptually, 
            # but to Sovereign Terminal, the SYSTEM successfully executed the command.
            # However, providing ok=False natively tells the orchestrator a failure occurred. 
            return {
                "ok": process.returncode == 0,
                "stdout": self._smart_truncate(stdout_str),
                "stderr": self._smart_truncate(stderr_str),
                "return_code": process.returncode,
                "cwd": cwd,
                "summary": self._build_command_summary(
                    cmd,
                    stdout_str,
                    stderr_str,
                    return_code=process.returncode,
                ),
                "governance": verdict.receipt(),
            }
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('sovereign_terminal', e)
            return {"ok": False, "error": f"Shell error: {e}"}

    def _smart_truncate(self, text: str, max_len: int = 5000) -> str:
        """Keep head and tail of logs, preserving the most useful error contexts."""
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        
        head_len = max_len // 2
        tail_len = max_len // 2
        truncated_msg = f"\n... [TRUNCATED {len(text) - max_len} CHARS] ...\n"
        return text[:head_len] + truncated_msg + text[-tail_len:]

    def _build_command_summary(
        self,
        command: str,
        stdout: str,
        stderr: str,
        *,
        return_code: int | None,
    ) -> str:
        signal = ""
        for candidate in (stderr, stdout):
            for raw_line in str(candidate or "").splitlines():
                line = " ".join(raw_line.split())
                if line:
                    signal = line
                    break
            if signal:
                break
        status = "ok" if return_code == 0 else "failed" if return_code is not None else "timed out"
        summary = f"{command} -> {status}"
        if signal:
            summary = f"{summary} ({signal[:140]})"
        return summary[:220]

    async def _open_target(self, target: str, action: str) -> dict[str, Any]:
        if not target:
            return {"ok": False, "error": "Open action requires a 'target'."}
        
        system = platform.system()
        cmd = []
        if system == "Darwin":
            if action == "open_app":
                cmd = ["open", "-a", target]
            else:
                cmd = ["open", target]
        elif system == "Linux":
            cmd = ["xdg-open", target]
        else:
            return {"ok": False, "error": f"Unsupported OS for 'open': {system}"}
            
        # `open -a <anything>` launches an arbitrary program with the owner's
        # session and privileges. That is execution, not navigation, and it
        # was the way around the shell path: a command refused as a command
        # ran fine as a `.app` or a document with a handler.
        verdict = await authorize_execution(
            KIND_OPEN,
            cmd,
            source="tool_execution:sovereign_terminal.open",
            extra={"target": target, "open_action": action},
        )
        if not verdict.approved:
            return verdict.as_error()

        logger.info("🚀 Launching %s: %s", action, target)
        try:
            # Tracking open actions too
            with task_tracker.track("system_open", details={"target": target}):
                process = await get_subprocess_gateway().spawn_async(
                    cmd,
                    source="tool_execution:sovereign_terminal.open",
                    accelerator_capability="auto",
                )
                await process.wait()
                if process.returncode != 0:
                    return {
                        "ok": False,
                        "error": (
                            f"'{' '.join(cmd)}' exited {process.returncode} — "
                            f"{target} was NOT opened (missing app or bad target?)."
                        ),
                        "governance": verdict.receipt(),
                    }
                return {
                    "ok": True,
                    "summary": f"Target {target} opened successfully.",
                    "governance": verdict.receipt(),
                }
        except (RuntimeError, TimeoutError, AttributeError) as e:
            record_degradation('sovereign_terminal', e)
            return {"ok": False, "error": f"Failed to open target: {e}"}
        finally:
            release_execution(verdict, source="sovereign_terminal.open")
