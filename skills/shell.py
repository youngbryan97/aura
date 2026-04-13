# skills/shell.py — Shell execution skill (subprocess with list args, no shell=True)
import asyncio
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from core.config import config
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.Shell")

# Dangerous patterns that should NEVER be executed
_BLOCKED_PATTERNS = {"rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:",
                     "chmod -R 777 /", "curl | sh", "wget | sh", "eval ", "nc -l"}

class ShellSkill(BaseSkill):
    name = "shell"
    description = "Execute safe terminal commands."
    inputs = {
        "command": "The shell command to run",
        "timeout": "Timeout in seconds (default 10)",
        "background": "Run the command in the background (default False)",
        "sandbox": "Enable native macOS sandbox isolation (default False)",
        "persistent_session_id": "Use a persistent bash session across commands"
    }
    
    _background_jobs: Dict[str, asyncio.subprocess.Process] = {}

    def __init__(self):
        # No command whitelist — full autonomy. Destructive-pattern blocklist
        # and metacharacter checks remain as infrastructure safety guards.
        # Defensive path resolution to avoid AttributeError
        self.cwd = str(getattr(config.paths, "base_dir", Path.cwd()))
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

        Returns (safe, reason). Note: we use create_subprocess_exec with a
        pre-split argument list (no shell=True), so shell metacharacters
        are not a concern — they're treated as literal strings by the OS.
        """
        # Check destructive pattern blocklist
        cmd_lower = cmd_str.lower()
        for pattern in _BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                return False, f"Blocked dangerous pattern: {pattern}"

        # Parse into tokens
        try:
            tokens = shlex.split(cmd_str)
        except ValueError:
            return False, "Malformed command string."

        if not tokens:
            return False, "Empty command."

        base_cmd = os.path.basename(tokens[0])

        # Per-command flag restrictions
        restrictions = self._RESTRICTED_COMMANDS.get(base_cmd)
        if restrictions:
            blocked_flags = restrictions.get("blocked_flags", set())
            for tok in tokens[1:]:
                if tok in blocked_flags:
                    return False, f"Flag '{tok}' is blocked for '{base_cmd}'."
            max_args = restrictions.get("max_args")
            if max_args and len(tokens) - 1 > max_args:
                return False, f"Too many arguments for '{base_cmd}' (max {max_args})."

        # rm must only operate on paths within workspace
        if base_cmd == "rm":
            for arg in tokens[1:]:
                if arg.startswith("-"):
                    continue
                resolved = os.path.realpath(os.path.join(self.cwd, arg))
                allowed_root = str(getattr(config.paths, "base_dir", Path.cwd()))
                if not resolved.startswith(allowed_root):
                    return False, f"rm blocked: '{arg}' resolves outside workspace."

        # Log if binary resolves to a suspicious location
        full_path = shutil.which(tokens[0])
        if full_path:
            suspicious_dirs = ["/tmp", "/var/tmp", "/dev/shm"]
            for s_dir in suspicious_dirs:
                if full_path.startswith(s_dir):
                    logger.warning("Suspicious binary location: '%s' -> '%s'", cmd_str, full_path)
                    break

        return True, "ok"


    async def execute(self, goal: Dict, context: Dict) -> Dict:
        cmd_str = goal.get("params", {}).get("command", "")
        timeout = goal.get("params", {}).get("timeout", 10)

        background = goal.get("params", {}).get("background", False)
        use_sandbox = goal.get("params", {}).get("sandbox", False)

        if not cmd_str:
            return {"ok": False, "error": "No command provided."}

        # 1. Safety Check — ENFORCED (no bypass)
        safe, reason = self._is_safe_command(cmd_str)
        if not safe:
            logger.warning("Shell BLOCKED: %s — %s", cmd_str, reason)
            return {"ok": False, "error": f"Command blocked: {reason}"}

        base_cmd = shlex.split(cmd_str)[0]
        logger.info("Shell Execution: [%s] (timeout=%s, bg=%s, sandbox=%s)", 
                    cmd_str, timeout, background, use_sandbox)

        # 2. Execution
        try:
            # Handle 'cd' manually because subprocess starts fresh every time
            if os.path.basename(base_cmd) == "cd":
                parts = shlex.split(cmd_str)
                if len(parts) > 1:
                    target = parts[1]
                    new_path = os.path.abspath(os.path.join(self.cwd, target))
                    allowed_root = str(getattr(config.paths, "base_dir", Path.cwd()))
                    if new_path.startswith(allowed_root):
                        self.cwd = new_path
                        return {"ok": True, "summary": f"Changed directory to {self.cwd}"}
                    else:
                        return {"ok": False, "error": "Access Denied: Cannot leave workspace."}

            if use_sandbox:
                from core.sandbox.macos_sandbox import MacOSSandbox, SandboxConfig
                sb_cfg = SandboxConfig(
                    allow_network=True,
                    allow_exec=True,
                    read_paths=[self.cwd],
                    write_paths=[self.cwd]
                )
                sandbox = MacOSSandbox(sb_cfg)
                # Note: sandbox-exec is synchronous in this implementation
                res = sandbox.execute_command(shlex.split(cmd_str), cwd=self.cwd)
                return {
                    "ok": res.returncode == 0,
                    "stdout": res.stdout.strip()[:4000],
                    "stderr": res.stderr.strip()[:4000],
                    "cwd": self.cwd,
                    "sandbox": True
                }

            # Wholesale addition: Persistent Bash Session
            persistent_session_id = goal.get("params", {}).get("persistent_session_id")
            if persistent_session_id:
                from core.sandbox.bash_daemon import bash_manager
                session = bash_manager.get_session(persistent_session_id, self.cwd)
                success, output = await session.execute(cmd_str, timeout=float(timeout))
                return {
                    "ok": success,
                    "stdout": output,
                    "stderr": "", # Daemon merges stdout/stderr
                    "cwd": "PERSISTENT" # Handled internally
                }

            # Normal Async Execution
            process = await asyncio.create_subprocess_exec(
                *shlex.split(cmd_str),
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            if background:
                job_id = f"job_{process.pid}"
                self._background_jobs[job_id] = process
                return {
                    "ok": True, 
                    "job_id": job_id, 
                    "message": f"Process {process.pid} started in background. Use command_status to check."
                }
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=float(timeout))
            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()  # Ensure cleanup
                except Exception:
                    pass
                return {"ok": False, "error": f"Command timed out after {timeout}s."}

            # v2.0: Let the Tool Distillation Service handle long outputs (Phase 1B integration)
            return {
                "ok": process.returncode == 0,
                "stdout": stdout.decode().strip(),
                "stderr": stderr.decode().strip(),
                "cwd": self.cwd
            }

        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command timed out."}
        except Exception as e:
            return {"ok": False, "error": str(e)}
