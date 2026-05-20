import asyncio
import asyncio.subprocess
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill

# Prevent basic unsafe operations inside the sandbox
SECURITY_PREAMBLE = """
import sys
import builtins
import os
import subprocess

_original_import = builtins.__import__

_BLOCKED_MODULES = frozenset({
    'shutil', 'socket', 'http', 'urllib', 'requests', 'ctypes',
    'multiprocessing', 'signal', 'pty', 'fcntl', 'termios',
    'webbrowser', 'ftplib', 'smtplib', 'telnetlib', 'xmlrpc',
    'importlib', 'code', 'codeop', 'compileall', 'py_compile',
})

def _safe_import(name, *args, **kwargs):
    top_level = name.split('.')[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(f"Module '{name}' is blocked in sandbox environment")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _safe_import

try:
    import resource

    def _cap_limit(resource_kind, ceiling):
        soft, hard = resource.getrlimit(resource_kind)
        if hard == resource.RLIM_INFINITY:
            new_hard = ceiling
        else:
            new_hard = min(hard, ceiling)
        if soft == resource.RLIM_INFINITY:
            new_soft = new_hard
        else:
            new_soft = min(soft, new_hard)
        if new_soft > 0 and new_hard > 0:
            resource.setrlimit(resource_kind, (new_soft, new_hard))

    _cap_limit(resource.RLIMIT_AS, 512 * 1024 * 1024)
    _cap_limit(resource.RLIMIT_CPU, 10)
except (ImportError, ValueError, OSError):
    pass  # Windows/macOS fallback

# Block dangerous builtins
_forbidden_builtins = ['eval', 'exec', 'open', 'compile', 'input']
for b in _forbidden_builtins:
    if hasattr(builtins, b):
        setattr(builtins, b, None)

# Disable os.system, os.popen, os.exec*, os.spawn*, os.fork
for _attr in ['system', 'popen', 'execl', 'execle', 'execlp', 'execlpe',
              'execv', 'execve', 'execvp', 'execvpe', 'spawnl', 'spawnle',
              'spawnlp', 'spawnlpe', 'spawnv', 'spawnve', 'spawnvp',
              'spawnvpe', 'fork', 'forkpty', 'kill', 'killpg', 'remove',
              'unlink', 'rmdir', 'removedirs']:
    if hasattr(os, _attr):
        setattr(os, _attr, None)

subprocess.Popen = None
subprocess.run = None
subprocess.call = None
subprocess.check_call = None
subprocess.check_output = None
"""

logger = logging.getLogger("Skills.InternalSandbox")


_SANDBOX_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    subprocess.SubprocessError,
)


def _record_sandbox_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "internal_sandbox",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "internal_sandbox",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _safe_decode(data: bytes, *, max_chars: int) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        return text[: max_chars // 2] + "\n... [TRUNCATED] ...\n" + text[-max_chars // 2 :]
    return text


def _resolve_cwd(cwd: str | None) -> str:
    path = Path(cwd or tempfile.gettempdir()).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Sandbox cwd is not a directory: {path}")
    return str(path)


class SandboxInput(BaseModel):
    code: str | None = Field(None, description="Python code to execute immediately.")
    notes: str | None = Field(None, description="Text to store in temporary scratchpad.")


class SandboxSkill(BaseSkill):
    name = "internal_sandbox"
    description = "An invisible scratchpad/terminal to test Python code or write notes purely for internal thought processing. Data here is ephemeral."
    input_model = SandboxInput

    # Safety limits
    MAX_EXECUTION_TIME = 30  # seconds
    MAX_OUTPUT_SIZE = 10000  # characters

    def __init__(self):
        self.scratchpad = ""

    async def execute(self, params: SandboxInput, context: dict[str, Any]) -> dict[str, Any]:
        """Execute sandboxed code or notes."""
        if isinstance(params, dict):
            try:
                params = SandboxInput(**params)
            except _SANDBOX_RECOVERABLE_ERRORS as e:
                _record_sandbox_degradation(
                    e,
                    action="rejected invalid sandbox input before code execution",
                    stage="input_validation",
                    severity="warning",
                )
                return {"ok": False, "error": f"Invalid input: {e}"}

        code = params.code
        notes = params.notes

        if notes:
            self.scratchpad += f"\n--- note ---\n{notes}\n"
            return {"ok": True, "summary": "Notes added to internal scratchpad."}

        if code:
            return await self.execute_code_safely(code)

        return {"ok": True, "result": self.scratchpad, "summary": "Viewed scratchpad."}

    async def execute_code_safely(self, code: str, cwd: str | None = None) -> dict[str, Any]:
        """Execute code in a subprocess with timeout (Async).
        A security preamble blocks dangerous operations before exec.

        Can also be used directly by the SelfModification engine.
        """
        temp_path: str | None = None
        resolved_cwd = tempfile.gettempdir()

        try:
            resolved_cwd = _resolve_cwd(cwd)
            # Prepend security preamble to user code
            sandboxed_code = SECURITY_PREAMBLE + "\n" + code

            def _write_temp_script() -> str:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(sandboxed_code)
                    return f.name

            temp_path = await asyncio.to_thread(_write_temp_script)

            try:
                # Run in subprocess with timeout
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=resolved_cwd,
                )

                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        process.communicate(), timeout=self.MAX_EXECUTION_TIME
                    )
                    stdout = _safe_decode(stdout_b, max_chars=self.MAX_OUTPUT_SIZE)
                    stderr = _safe_decode(stderr_b, max_chars=self.MAX_OUTPUT_SIZE)
                except TimeoutError:
                    try:
                        process.kill()
                        await asyncio.wait_for(process.communicate(), timeout=2.0)
                    except _SANDBOX_RECOVERABLE_ERRORS as e:
                        _record_sandbox_degradation(
                            e,
                            action="returned timeout while sandbox process cleanup remained uncertain",
                            stage="timeout.kill",
                            severity="degraded",
                        )
                        logger.debug("Failed to kill sandboxed process: %s", e)
                    return {
                        "ok": False,
                        "error": f"Code execution timed out after {self.MAX_EXECUTION_TIME}s",
                    }

                output = f"Stdout:\n{stdout}"
                if stderr:
                    output += f"\nStderr:\n{stderr}"

                if process.returncode != 0:
                    return {
                        "ok": False,
                        "error": f"Code exited with code {process.returncode}",
                        "result": output,
                        "summary": "Code execution failed.",
                    }

                return {"ok": True, "result": output, "summary": "Code executed in sandbox."}
            finally:
                # Clean up temp file
                if temp_path:
                    try:
                        await asyncio.to_thread(os.unlink, temp_path)
                    except _SANDBOX_RECOVERABLE_ERRORS as e:
                        _record_sandbox_degradation(
                            e,
                            action="left sandbox temp file for operator cleanup after unlink failed",
                            stage="cleanup.temp_file",
                            severity="warning",
                            extra={"temp_path": temp_path},
                        )
                        logger.debug("Failed to delete temp sandbox file %s: %s", temp_path, e)

        except _SANDBOX_RECOVERABLE_ERRORS as e:
            _record_sandbox_degradation(
                e,
                action="returned explicit sandbox failure payload after execution boundary failure",
                stage="execute_code_safely",
                severity="degraded",
                extra={"cwd": resolved_cwd},
            )
            return {"ok": False, "error": f"Sandbox Exception: {e}"}
