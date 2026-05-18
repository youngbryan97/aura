"""Bounded project command runner for Celery task entrypoints."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Tasks.Command")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTEST_SELECTOR_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:.[]-/")
_RECOVERABLE_COMMAND_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class ManagedCommandResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def status_payload(self) -> dict[str, object]:
        if self.ok:
            return {"status": "success", "stdout": self.stdout, "stderr": self.stderr, "returncode": self.returncode}
        message = self.stderr or self.stdout or "managed command failed"
        if self.timed_out:
            message = "managed command timed out"
        return {
            "status": "error",
            "message": message,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
        }

    def mutation_payload(self) -> dict[str, object]:
        payload = {
            "success": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
        }
        if not self.ok:
            payload["error"] = self.stderr or self.stdout or "mutation test command failed"
        return payload


def run_project_python(relative_script: str, *, timeout_s: float = 1800.0) -> ManagedCommandResult:
    try:
        script = _resolve_project_file(relative_script, suffix=".py")
    except _RECOVERABLE_COMMAND_ERRORS as exc:
        return _validation_failure((sys.executable, relative_script), exc)
    return run_project_command((sys.executable, str(script.relative_to(PROJECT_ROOT))), timeout_s=timeout_s)


def run_project_pytest(target: str, *, timeout_s: float = 600.0) -> ManagedCommandResult:
    try:
        normalized_target = _normalize_pytest_target(target)
    except _RECOVERABLE_COMMAND_ERRORS as exc:
        return _validation_failure((sys.executable, "-m", "pytest", "-q", target), exc)
    return run_project_command((sys.executable, "-m", "pytest", "-q", normalized_target), timeout_s=timeout_s)


def run_project_command(command: tuple[str, ...], *, timeout_s: float) -> ManagedCommandResult:
    try:
        return _run_async_blocking(lambda: _run_project_command_async(command, timeout_s=timeout_s))
    except _RECOVERABLE_COMMAND_ERRORS as exc:
        record_degradation("tasks_command", exc)
        logger.warning("Managed command failed before launch: %s", exc)
        return ManagedCommandResult(command, 127, "", str(exc), 0.0)


def _validation_failure(command: tuple[str, ...], exc: BaseException) -> ManagedCommandResult:
    record_degradation("tasks_command", exc)
    logger.warning("Managed command rejected invalid target: %s", exc)
    return ManagedCommandResult(command, 127, "", str(exc), 0.0)


def _resolve_project_file(relative_path: str, *, suffix: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("project command target must stay inside the Aura workspace") from exc

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"project command target not found: {relative_path}")
    if candidate.suffix != suffix:
        raise ValueError(f"project command target must be a {suffix} file")
    return candidate


def _normalize_pytest_target(target: str) -> str:
    path_text, separator, selector = target.partition("::")
    path = _resolve_project_file(path_text, suffix=".py")
    if selector and any(char not in _PYTEST_SELECTOR_CHARS for char in selector):
        raise ValueError("pytest selector contains unsupported characters")
    normalized = str(path.relative_to(PROJECT_ROOT))
    return f"{normalized}{separator}{selector}" if separator else normalized


def _run_async_blocking(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result: dict[str, ManagedCommandResult] = {}
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except _RECOVERABLE_COMMAND_ERRORS as exc:
            failure.append(exc)

    thread = threading.Thread(target=runner, name="aura-managed-command", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise RuntimeError(str(failure[0])) from failure[0]
    return result["value"]


async def _run_project_command_async(command: tuple[str, ...], *, timeout_s: float) -> ManagedCommandResult:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=PROJECT_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    elapsed = time.perf_counter() - started
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return ManagedCommandResult(command, process.returncode, stdout, stderr, elapsed, timed_out=timed_out)
