"""Local Sandbox — Concrete implementation of the Sandbox interface.
Executes Python code and shell commands in an isolated temporary directory
using subprocess with timeouts and resource limits.
"""
import logging
import os
import subprocess
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional

from .sandbox import ExecutionResult, Sandbox

logger = logging.getLogger("Aura.LocalSandbox")


class LocalSandbox(Sandbox):
    """Sandbox that runs code in a temporary directory using subprocess.
    Provides isolation via a clean temp folder (not a true OS sandbox).
    """

    def __init__(self, work_dir: Optional[str] = None):
        self._work_dir = work_dir
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._active = False
        self._code_history = []  # Emulates Jupyter statefulness

    @property
    def work_path(self) -> Path:
        if self._work_dir:
            return Path(self._work_dir)
        if self._temp_dir:
            return Path(self._temp_dir.name)
        raise RuntimeError("Sandbox not started. Call start() first.")

    def start(self):
        """Create the sandbox working directory."""
        if self._active:
            return
        if not self._work_dir:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="aura_sandbox_")
            logger.info("LocalSandbox started at %s", self._temp_dir.name)
        else:
            os.makedirs(self._work_dir, exist_ok=True)
            logger.info("LocalSandbox started at %s", self._work_dir)
        self._active = True

    def stop(self):
        """Clean up the sandbox."""
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                logger.warning("Sandbox cleanup failed: %s", e)
            self._temp_dir = None
        self._active = False
        self._code_history.clear()
        logger.info("LocalSandbox stopped.")

    async def run_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute a Python script in the sandbox (Async Offload)."""
        script_path = self.work_path / "_aura_run.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.monotonic()
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python3", script_path.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_path),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                exit_code=-1,
                duration=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="", stderr=str(e), exit_code=-1, duration=duration
            )

    async def run_stateful_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Executes code persistently, retaining variables and imports if successful.
        If the execution fails, the state is rolled back.
        """
        self._code_history.append(code)
        full_code = "\n".join(self._code_history)
        
        result = await self.run_code(full_code, timeout)
        
        if result.exit_code != 0:
            # Revert state if the code crashed (emulating a failed notebook cell)
            self._code_history.pop()
        
        return result

    async def run_command(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a shell command in the sandbox (Async Offload)."""
        start = time.monotonic()
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_path),
            )
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                duration=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="", stderr=str(e), exit_code=-1, duration=duration
            )

    def _safe_path(self, path: str) -> Path:
        """Enforce strict path resolution to prevent directory traversal."""
        # v51: Sandbox Hardening
        p = (self.work_path / path).resolve()
        if not str(p).startswith(str(self.work_path.resolve())):
            raise PermissionError(f"Attempted directory traversal: {path}")
        return p

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        full_path = self._safe_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        return full_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str):
        """Write a file to the sandbox."""
        full_path = self._safe_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")