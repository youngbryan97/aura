from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import shlex
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from core.config import config
from core.utils.task_tracker import task_tracker
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SovereignTerminal")

class TerminalInput(BaseModel):
    action: str = Field("execute", description="Action: 'execute', 'open_app', 'open_file', 'cd'")
    command: Optional[str] = Field(None, description="Shell command to run (for 'execute').")
    target: Optional[str] = Field(None, description="App name or file path (for 'open' actions).")
    cwd: Optional[str] = Field(None, description="Current working directory for execution or 'cd'.")
    timeout: int = Field(15, description="Timeout in seconds for execution.")

class SovereignTerminalSkill(BaseSkill):
    """The unified terminal and system operation capability for Aura.
    Handles shell command execution, application launching, and file opening.
    """
    
    name = "sovereign_terminal"
    description = "Execute shell commands, launch system apps, and open files via CLI."
    input_model = TerminalInput
    
    def __init__(self):
        super().__init__()
        # Use workspace root as default CWD if available
        self.default_cwd = str(getattr(config.paths, "base_dir", os.getcwd()))

    async def execute(self, params: TerminalInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified entry point for all system operations."""
        if isinstance(params, dict):
            try:
                params = TerminalInput(**params)
            except Exception as e:
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
                new_path = os.path.abspath(os.path.join(cwd, params.target or "."))
                # Security shortcut: Ensure we stay in workspace if it's strictly enforced (optional for sovereign)
                return {"ok": True, "new_cwd": new_path, "message": f"Directory changed to {new_path}"}
            else:
                return {"ok": False, "error": f"Unsupported terminal action: {action}"}
        except Exception as e:
            record_degradation('sovereign_terminal', e)
            logger.error("Terminal skill failed: %s", e)
            return {"ok": False, "error": str(e)}

    async def _run_command(self, cmd: str, cwd: str, timeout: int) -> Dict[str, Any]:
        if not cmd:
            return {"ok": False, "error": "Execute action requires a 'command'."}
        
        # --- Advanced Security Hardening ---
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
                     for arg in tokens[i+1:]:
                         if arg.startswith("-"): continue
                         resolved = os.path.realpath(os.path.join(cwd, arg))
                         allowed_root = str(getattr(config.paths, "base_dir", "/")) # Default to root if not set
                         if not resolved.startswith(allowed_root) and allowed_root != "/":
                             logger.warning("🛡️ RM blocked: path %s is outside %s", resolved, allowed_root)
                             return {"ok": False, "error": f"rm blocked: '{arg}' resolves outside sanctioned path."}

        logger.info("🐚 Shell Execute: %s (CWD: %s)", cmd, cwd)
        
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout_chunks = []
            stderr_chunks = []
            
            async def read_stream(stream, chunks_list):
                interactive_prompts = [b"password:", b"y/n", b"yes/no", b"enter ", b"continue?"]
                try:
                    while True:
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
                    timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except Exception as e:
                    record_degradation('sovereign_terminal', e)
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
            }
        except Exception as e:
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
        return_code: Optional[int],
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

    async def _open_target(self, target: str, action: str) -> Dict[str, Any]:
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
            
        logger.info("🚀 Launching %s: %s", action, target)
        try:
            # Tracking open actions too
            with task_tracker.track("system_open", details={"target": target}):
                process = await asyncio.create_subprocess_exec(*cmd)
                await process.wait()
                return {"ok": True, "summary": f"Target {target} opened successfully."}
        except Exception as e:
            record_degradation('sovereign_terminal', e)
            return {"ok": False, "error": f"Failed to open target: {e}"}
