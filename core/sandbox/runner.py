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
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger(__name__)

#: The tree the sandboxed runner adds to its path, so the engineering
#: primitives can be handed to code that has no import of its own.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TIMEOUT = 10
DEFAULT_MEM_BYTES = 500 * 1024 * 1024  # 500MB limit
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
repo_root = params.get("repo_root", "")
# One directory the person named, made importable while real imports still
# exist, and handed over as ready-made names — the same treatment the
# engineering primitives get and for the same reason.
#
# The sandbox bans `sys` by design, and sys.path is the only way to import a
# module from a directory, so "read the docs at this path, then use it" was
# impossible by construction. The path is checked before it gets here, and the
# one line that uses the interpreter is this one rather than a line inside a
# program nobody has read.
library_root = params.get("library_root", "")
resource_warning = None

# Engineering primitives, imported HERE while real builtins still exist and
# handed to the sandboxed code as ready-made names. The sandbox has no
# __import__ by design, and it stays that way: these are pure computation
# with no file, network or process reach, so making them available adds
# arithmetic that carries its units rather than any new capability.
#
# Without this the only way to do dimensional arithmetic in the REPL was to
# open the import door for everything.
engineering_namespace = {}
engineering_error = None
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)
library_namespace = {}
library_error = None
if library_root and library_root not in sys.path:
    sys.path.insert(0, library_root)
if library_root:
    import importlib as _importlib
    import os as _os

    try:
        for _entry in sorted(_os.listdir(library_root)):
            if not _entry.endswith(".py") or _entry.startswith("_"):
                continue
            _name = _entry[:-3]
            _module = _importlib.import_module(_name)
            library_namespace[_name] = _module
            for _attr in dir(_module):
                if not _attr.startswith("_"):
                    library_namespace.setdefault(_attr, getattr(_module, _attr))
    except Exception as _exc:  # noqa: BLE001 - reported, never raised at the child
        library_error = f"{type(_exc).__name__}: {_exc}"
try:
    from core.engineering.geometry import (
        Box, Capsule, Cone, Cylinder, Dome, Ellipsoid, Frustum, Plate, Prism,
        Sphere, Torus, Tube, solid_from_spec,
    )
    from core.engineering.materials import fluid, material
    from core.engineering.uncertainty import Uncertain, propagate, rss_stack
    from core.engineering.units import Q, DimensionError, parse_quantity
    engineering_namespace = {
        "Q": Q, "parse_quantity": parse_quantity, "DimensionError": DimensionError,
        "material": material, "fluid": fluid,
        "Box": Box, "Plate": Plate, "Cylinder": Cylinder, "Tube": Tube,
        "Sphere": Sphere, "Dome": Dome, "Cone": Cone, "Frustum": Frustum,
        "Torus": Torus, "Capsule": Capsule, "Prism": Prism,
        "Ellipsoid": Ellipsoid, "solid_from_spec": solid_from_spec,
        "Uncertain": Uncertain, "propagate": propagate, "rss_stack": rss_stack,
    }
except (ImportError, AttributeError) as exc:
    # A REPL without the engineering package is still a REPL.
    engineering_error = repr(exc)
RUNNER_RUNTIME_ERRORS = (Exception, KeyboardInterrupt, SystemExit)

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
    'NameError': builtins.NameError,
    'RuntimeError': builtins.RuntimeError,
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
    globals_dict.update(engineering_namespace)
    globals_dict.update(library_namespace)
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
except RUNNER_RUNTIME_ERRORS as e:
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


#: The only variables untrusted code's process is given. Everything else in
#: the parent environment — API keys, tokens, session paths — was being
#: handed to it, which the subprocess gateway had been recording as a
#: privilege-inheritance degradation on every single run. The child needs a
#: PATH to find nothing in particular and a temp directory it can write to,
#: and it is started with -I so it reads no user site or PYTHONPATH anyway.
_UNTRUSTED_ENV_KEYS = ("PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")


def _untrusted_environment() -> dict[str, str]:
    """A minimal environment for the untrusted child."""
    import os

    env = {
        key: os.environ[key]
        for key in _UNTRUSTED_ENV_KEYS
        if os.environ.get(key)
    }
    env.setdefault("PATH", "/usr/bin:/bin")
    # A home it cannot learn anything from, and no site packages of its own.
    env["HOME"] = env.get("TMPDIR", "/tmp")
    env["PYTHONNOUSERSITE"] = "1"
    return env


async def _communicate_process(
    command: tuple[str, ...],
    payload: bytes,
    timeout_s: float,
    mem_bytes: int,
) -> _RunnerProcessResult:
    process = await get_subprocess_gateway().spawn_async(
        command,
        stdin=_PIPE,
        stdout=_PIPE,
        stderr=_PIPE,
        source="sandbox.runner.untrusted_child",
        accelerator_capability="auto",
        env=_untrusted_environment(),
    )
    communicate_task = create_tracked_task(
        process.communicate(input=payload),
        name="sandbox.runner.communicate",
    )
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
        from core.runtime.resource_observation import get_resource_observer

        process = get_resource_observer().process(pid)
        return int(process.rss_bytes) if process is not None else None
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
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
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    library_root: str = "",
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
                # The runner starts with -I, so it inherits no path. It needs
                # this to find the engineering primitives it hands the
                # sandboxed code; nothing else is read from the tree.
                "repo_root": str(_REPO_ROOT),
                "library_root": str(library_root or ""),
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
