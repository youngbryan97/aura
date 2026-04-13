"""
Persistent Bash Daemon — Wholesale Addition

Provides a persistent Bash session for the Shell Skill. Enables stateful
interactions (e.g., maintaining `cd`, `export` variables, and activating 
virtual environments) across consecutive shell commands.
"""

import asyncio
import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger("Aura.BashDaemon")

class PersistentBashSession:
    def __init__(self, cwd: str):
        self.cwd = cwd
        self._process = None
        self._delimiter = f"---AURA_CMD_DELIM_{os.urandom(4).hex()}---"
        self._lock = asyncio.Lock()

    async def _start(self):
        env = os.environ.copy()
        # Start bash and immediately set it to print our delimiter after every command
        self._process = await asyncio.create_subprocess_exec(
            "bash", "--noprofile", "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.cwd,
            env=env
        )
        
        # Setup bash to echo the delimiter and the exit code
        setup_cmd = f"export PS1=''\nPROMPT_COMMAND='echo \"\n{self._delimiter}:$?\"'\n"
        self._process.stdin.write(setup_cmd.encode('utf-8'))
        await self._process.stdin.drain()
        
        # Read until first delimiter
        await self._read_until_delimiter()

    async def _read_until_delimiter(self) -> Tuple[str, int]:
        """Reads stdout until the delimiter is found. Returns (output, exit_code)."""
        output = []
        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace')
                if line_str.startswith(self._delimiter):
                    # Extract exit code
                    parts = line_str.strip().split(":")
                    exit_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    return "".join(output).strip(), exit_code
                output.append(line_str)
            except Exception as e:
                logger.error("Error reading from bash daemon: %s", e)
                break
        return "".join(output).strip(), -1

    async def execute(self, cmd: str, timeout: float = 10.0) -> Tuple[bool, str]:
        async with self._lock:
            if self._process is None or self._process.returncode is not None:
                await self._start()

            # Write command
            try:
                self._process.stdin.write(f"{cmd}\n".encode('utf-8'))
                await self._process.stdin.drain()
            except Exception as e:
                return False, f"Failed to write to daemon: {e}"

            # Wait for output up to timeout
            try:
                output, exit_code = await asyncio.wait_for(self._read_until_delimiter(), timeout=timeout)
                return exit_code == 0, output
            except asyncio.TimeoutError:
                return False, f"Command timed out after {timeout}s."
            except Exception as e:
                return False, f"Execution failed: {e}"

    async def kill(self):
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()

class BashDaemonManager:
    """Manages persistent sessions per conversation or agent."""
    def __init__(self):
        self.sessions: Dict[str, PersistentBashSession] = {}

    def get_session(self, session_id: str, cwd: str) -> PersistentBashSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = PersistentBashSession(cwd)
        return self.sessions[session_id]

    async def shutdown(self):
        for s in self.sessions.values():
            await s.kill()
        self.sessions.clear()

# Global singleton
bash_manager = BashDaemonManager()
