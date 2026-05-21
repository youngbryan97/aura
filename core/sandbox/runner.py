from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_MEM_BYTES = 200 * 1024 * 1024  # 200MB limit
DEFAULT_OUTPUT_LIMIT = 200 * 1024  # 200KB limit for std output
DEFAULT_CODE_BYTES = 512 * 1024
_PIPE = -1
_SANDBOX_RUNNER_ERRORS = (
    FileNotFoundError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

RUNNER_PY = r"""
import sys
import json
import contextlib
import io
import traceback

try:
    import resource
except ImportError:
    resource = None

params = json.loads(sys.stdin.read())
code = params.get("code", "")
mem_bytes = params.get("mem_bytes", None)
cpu_seconds = params.get("cpu_seconds", None)
resource_warning = None

try:
    if resource:
        def apply_limit(kind, desired):
            if not desired:
                return
            soft, hard = resource.getrlimit(kind)
            if hard == resource.RLIM_INFINITY:
                new_hard = desired
            else:
                new_hard = min(hard, desired)
            new_soft = min(desired, new_hard)
            if soft != new_soft:
                resource.setrlimit(kind, (new_soft, hard))
            if hard != new_hard:
                resource.setrlimit(kind, (new_soft, new_hard))

        if mem_bytes and sys.platform != "darwin":
            apply_limit(resource.RLIMIT_AS, int(mem_bytes))
        if cpu_seconds:
            apply_limit(resource.RLIMIT_CPU, int(cpu_seconds))
except (OSError, ValueError) as e:
    resource_warning = repr(e)

# Strip dangerous builtins to prevent arbitrary execution or network egress
import builtins
safe_builtins = {
    '__build_class__': builtins.__build_class__,
    'abs': builtins.abs, 'all': builtins.all, 'any': builtins.any, 'ascii': builtins.ascii,
    'bin': builtins.bin, 'bool': builtins.bool, 'bytearray': builtins.bytearray, 
    'bytes': builtins.bytes, 'callable': builtins.callable, 'chr': builtins.chr,
    'complex': builtins.complex, 'dict': builtins.dict, 'dir': builtins.dir,
    'Exception': builtins.Exception,
    'divmod': builtins.divmod, 'enumerate': builtins.enumerate, 'filter': builtins.filter,
    'float': builtins.float, 'format': builtins.format, 'frozenset': builtins.frozenset,
    'getattr': builtins.getattr, 'hash': builtins.hash,
    'hex': builtins.hex, 'id': builtins.id, 'int': builtins.int, 'isinstance': builtins.isinstance,
    'issubclass': builtins.issubclass, 'iter': builtins.iter, 'len': builtins.len,
    'list': builtins.list, 'map': builtins.map, 'max': builtins.max, 'min': builtins.min,
    'next': builtins.next, 'object': builtins.object, 'oct': builtins.oct, 'ord': builtins.ord,
    'pow': builtins.pow, 'print': builtins.print, 'property': builtins.property, 'range': builtins.range, 'repr': builtins.repr,
    'reversed': builtins.reversed, 'round': builtins.round, 'set': builtins.set,
    'slice': builtins.slice, 'sorted': builtins.sorted, 'str': builtins.str, 'super': builtins.super,
    'sum': builtins.sum, 'tuple': builtins.tuple, 'type': builtins.type, 'zip': builtins.zip,
    'None': None, 'True': True, 'False': False,
    # specifically exclude __import__, open, eval, exec, compile, globals, locals
}

try:
    globals_dict = {"__name__": "__main__", "__builtins__": safe_builtins}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        exec(code, globals_dict, globals_dict)
    print(json.dumps({
        "status": "ok",
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "resource_warning": resource_warning,
    }))
except SystemExit as e:
    print(json.dumps({
        "status": "exit",
        "code": int(e.code if isinstance(e.code, int) else 0),
        "stdout": "",
        "stderr": "",
        "resource_warning": resource_warning,
    }))
except BaseException as e:
    tb = traceback.format_exc()
    print(json.dumps({
        "status": "error",
        "repr": repr(e),
        "traceback": tb,
        "stdout": "",
        "stderr": "",
        "resource_warning": resource_warning,
    }))
"""


@dataclass(frozen=True)
class _RunnerProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    memory_exceeded: bool = False


def _truncate(s: str, limit: int) -> str:
    if not s:
        return s
    if len(s) <= limit:
        return s
    return s[:limit] + "...<truncated>"


def _termination_detail(returncode: int | None) -> str:
    if returncode is None or returncode >= 0:
        return ""
    signum = -returncode
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = f"signal {signum}"
    return f"child terminated by {signal_name}"


async def _communicate_process(
    command: tuple[str, ...],
    payload: bytes,
    timeout_s: float,
    mem_bytes: int,
) -> _RunnerProcessResult:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=_PIPE,
        stdout=_PIPE,
        stderr=_PIPE,
    )
    communicate_task = asyncio.create_task(process.communicate(input=payload))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    timed_out = False
    memory_exceeded = False

    while not communicate_task.done():
        if loop.time() >= deadline:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            break

        rss = _process_rss_bytes(process.pid) if mem_bytes else None
        if rss is not None and rss > mem_bytes:
            memory_exceeded = True
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            break

        await asyncio.sleep(0.05)

    if timed_out:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    stdout_bytes, stderr_bytes = await communicate_task

    return _RunnerProcessResult(
        process.returncode,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        memory_exceeded=memory_exceeded,
    )


def _process_rss_bytes(pid: int) -> int | None:
    try:
        import psutil
    except ImportError as exc:
        record_degradation("sandbox_runner", exc)
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("sandbox_runner", exc)
        return None


def _run_process_blocking(
    command: tuple[str, ...],
    payload: bytes,
    timeout: float,
    mem_bytes: int,
) -> _RunnerProcessResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_communicate_process(command, payload, timeout, mem_bytes))

    result: dict[str, _RunnerProcessResult] = {}
    failures: list[BaseException] = []

    def runner() -> None:
        try:
            result["value"] = asyncio.run(
                _communicate_process(command, payload, timeout, mem_bytes)
            )
        except _SANDBOX_RUNNER_ERRORS as exc:
            failures.append(exc)

    thread = threading.Thread(target=runner, name="aura-untrusted-runner", daemon=True)
    thread.start()
    thread.join()
    if failures:
        exc = failures[0]
        record_degradation("sandbox_runner", exc)
        return _RunnerProcessResult(127, "", str(exc))
    return result["value"]


def run_untrusted(
    code: str, timeout: int = DEFAULT_TIMEOUT, mem_bytes: int = DEFAULT_MEM_BYTES
) -> dict:
    """
    Executes an untrusted block of Python code in an isolated child process with strict safety limits.

    Args:
        code: The Python script string to execute.
        timeout: Maximum execution CPU time allowed.
        mem_bytes: Maximum RAM consumption allowed.

    Returns:
        Dict: Structured result payload with stdout and stderr output.
    """
    if len(code.encode("utf-8", errors="replace")) > DEFAULT_CODE_BYTES:
        return {
            "status": "rejected",
            "stdout": "",
            "stderr": f"code payload exceeds {DEFAULT_CODE_BYTES} bytes",
            "returncode": None,
        }

    with tempfile.TemporaryDirectory() as d:
        runner_path = Path(d) / "runner.py"
        atomic_write_text(runner_path, RUNNER_PY)

        payload = json.dumps(
            {
                "code": code,
                "mem_bytes": mem_bytes,
                "cpu_seconds": timeout,
            }
        ).encode("utf-8")

        command = (sys.executable, "-I", str(runner_path))
        process_result = _run_process_blocking(command, payload, timeout + 2, mem_bytes)

        if process_result.timed_out:
            return {
                "status": "timeout",
                "stdout": _truncate(process_result.stdout, DEFAULT_OUTPUT_LIMIT),
                "stderr": _truncate(
                    process_result.stderr or "timeout expired", DEFAULT_OUTPUT_LIMIT
                ),
                "returncode": process_result.returncode,
            }
        if process_result.memory_exceeded:
            return {
                "status": "memory_limit",
                "stdout": _truncate(process_result.stdout, DEFAULT_OUTPUT_LIMIT),
                "stderr": _truncate(
                    process_result.stderr or "memory limit exceeded", DEFAULT_OUTPUT_LIMIT
                ),
                "returncode": process_result.returncode,
            }
        if process_result.returncode == -getattr(signal, "SIGXCPU", 0):
            return {
                "status": "timeout",
                "stdout": _truncate(process_result.stdout, DEFAULT_OUTPUT_LIMIT),
                "stderr": _truncate(
                    process_result.stderr or "cpu time limit exceeded", DEFAULT_OUTPUT_LIMIT
                ),
                "returncode": process_result.returncode,
            }

        stderr = _truncate(process_result.stderr, DEFAULT_OUTPUT_LIMIT)
        try:
            child_payload = json.loads(process_result.stdout or "{}")
        except _SANDBOX_RUNNER_ERRORS as exc:
            record_degradation("sandbox_runner", exc)
            termination_detail = _termination_detail(process_result.returncode)
            return {
                "status": "terminated" if termination_detail else "runner_error",
                "stdout": _truncate(process_result.stdout, DEFAULT_OUTPUT_LIMIT),
                "stderr": stderr or termination_detail or str(exc),
                "returncode": process_result.returncode,
            }

        child_stderr = str(child_payload.get("stderr") or "")
        combined_stderr = "\n".join(part for part in (child_stderr, stderr) if part)
        result: dict[str, object] = {
            "status": str(child_payload.get("status", "runner_error")),
            "stdout": _truncate(str(child_payload.get("stdout") or ""), DEFAULT_OUTPUT_LIMIT),
            "stderr": _truncate(combined_stderr, DEFAULT_OUTPUT_LIMIT),
            "returncode": process_result.returncode,
        }
        for key in ("code", "repr", "traceback", "resource_warning"):
            value = child_payload.get(key)
            if value:
                result[key] = _truncate(str(value), DEFAULT_OUTPUT_LIMIT)
        return result
