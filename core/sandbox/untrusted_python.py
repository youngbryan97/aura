"""One way to run model-written Python, with an actual OS boundary.

CP126 raised the same critical finding against two benchmark harnesses
independently:

* ``aura_bench/aletheia_runner_live.py`` — "Rulescript and device handlers
  import and execute Aura-generated modules with importlib in the
  privileged runner process. AST parsing and temporary paths do not provide
  an OS sandbox, capability boundary, or resource limit."
* ``aura_bench/hard_suite.py`` — "The denylist is bypassable through allowed
  modules, object traversal, or indirect builtins, and ``-I`` only isolates
  import configuration rather than filesystem, process, network, or resource
  access."

They are two call sites of one defect, and the two defences they reached
for are both known-broken in the same way:

*AST screening is a denylist.* ``__import__`` reachable through
``().__class__.__mro__[1].__subclasses__()`` never appears in the import
table, and a screen that reads source text cannot see what
``getattr(mod, name)`` will resolve to at runtime.

*``python -I`` is not isolation.* It ignores ``PYTHON*`` environment
variables and drops the script directory from ``sys.path``. That is import
hygiene. The child keeps the parent's filesystem, network, process and
signal access in full — including the user's home directory, the live
runtime's sockets, and this machine's keychain.

What is actually enforceable
----------------------------
A kernel boundary. On macOS that is ``sandbox-exec`` (Seatbelt): deny by
default, then re-allow the interpreter's own read paths and one scratch
directory. Network is denied outright, so exfiltration and egress are not
policy questions. On Linux ``bwrap`` gives the same shape.

And the property that makes it worth having: **when no boundary is
available, this refuses to run the code.** The previous behaviour — run it
anyway, unsandboxed, and report a normal result — is what turns a benchmark
into an execution service. A refusal is a visible, fixable failure; silent
unsandboxed execution is neither.

``AURA_SANDBOX_ALLOW_UNCONFINED=1`` exists for platforms with no boundary
at all, and it is deliberately awkward: the returned outcome carries
``boundary="none"`` and ``sandboxed=False`` forever after, so a caller that
records results cannot later claim they were confined.

Two entry points, because untrusted code arrives in exactly two shapes:

* :func:`run_untrusted_script` — run it, keep stdout.
* :func:`call_untrusted_function` — load it, call one named function with
  JSON arguments, keep JSON results. This is what the benchmark harnesses
  need, and it is the shape that previously forced ``exec_module`` into the
  privileged process because nothing else offered it.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEM_BYTES = 512 * 1024 * 1024
DEFAULT_OUTPUT_LIMIT = 200 * 1024
DEFAULT_CODE_BYTES = 512 * 1024
DEFAULT_FILE_SIZE_BYTES = 32 * 1024 * 1024

#: Escape hatch for platforms with no kernel boundary. Off by default.
UNCONFINED_ENV = "AURA_SANDBOX_ALLOW_UNCONFINED"

_SANDBOX_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    subprocess.SubprocessError,
)


@dataclass(frozen=True)
class SandboxOutcome:
    """What happened, and — separately — whether it was actually confined.

    ``sandboxed`` is not a detail. A caller that reports "we ran the model's
    code safely" is making a claim about this field, so it is recorded
    beside the result rather than inferred from the absence of an error.
    """

    status: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    sandboxed: bool = False
    boundary: str = "none"
    results: list[Any] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "sandboxed": self.sandboxed,
            "boundary": self.boundary,
            "results": self.results,
            "error": self.error,
        }


class UntrustedExecutionError(RuntimeError):
    """Untrusted code could not be run under a boundary, or failed under one."""


# ── Boundary discovery ────────────────────────────────────────────────────


def available_boundary() -> str:
    """The strongest kernel boundary this host can actually apply.

    Returns ``"seatbelt"``, ``"bubblewrap"``, or ``""``. Presence of the
    binary is checked rather than assumed from ``sys.platform``: a stripped
    container image reports darwin and has no ``sandbox-exec``.
    """
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return ""


def _unconfined_permitted() -> bool:
    return os.environ.get(UNCONFINED_ENV, "").strip().lower() in {"1", "true", "yes"}


# ── Profile generation ────────────────────────────────────────────────────


#: Trees holding user and machine data. Denied for reading; the point of
#: the whole exercise. Everything else on the volume is OS and toolchain.
_USER_DATA_TREES: tuple[str, ...] = (
    "/Users",
    "/Volumes",
    "/private/var/folders",
    "/private/var/root",
)


def _interpreter_paths() -> list[str]:
    """Paths the interpreter itself lives in, fully resolved.

    Resolved rather than as-configured, because Seatbelt matches the real
    path: ``/Users/x/.venv/bin/python`` is a symlink into a Homebrew
    Cellar, and a profile naming the symlink denies the exec it meant to
    allow. That failure mode cost an afternoon — ``execvp() ... Operation
    not permitted`` with no indication that a symlink was involved.

    ``sysconfig`` supplies the rest, because a venv, a framework build and
    a Homebrew Python put the standard library in three different places.
    """
    paths: set[str] = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib", "data"):
        value = sysconfig.get_paths().get(key)
        if isinstance(value, str) and value:
            paths.add(value)
    paths.add(str(Path(sys.executable).resolve().parent))
    paths.add(str(Path(sys.prefix).resolve()))
    if hasattr(sys, "base_prefix"):
        paths.add(str(Path(sys.base_prefix).resolve()))
    resolved = set()
    for path in paths:
        try:
            resolved.add(str(Path(path).resolve()))
        except OSError:
            continue
    return sorted(p for p in resolved if p and Path(p).exists())


def _seatbelt_profile(*, scratch: Path, read_paths: Sequence[str]) -> str:
    """Deny-by-default Seatbelt profile.

    Reads are expressed as allow-all-then-deny-user-data rather than as a
    read allowlist. That is not laziness — a from-scratch read allowlist
    was tried first and CPython aborted with SIGABRT and no output at all,
    because dyld's shared-cache and cryptex paths move between macOS
    releases and a profile that misses one kills the interpreter before it
    can say which. An allowlist that must be rediscovered every OS update
    is a sandbox that will be disabled the first time it breaks CI.

    The property being bought is not "cannot read /usr/lib". It is: no user
    data, no network, no exec, no writes outside one scratch directory.
    Those four are stated directly here, and each is verified by
    ``tests/test_untrusted_python_sandbox.py`` against live escape attempts.
    """
    interpreter = str(Path(sys.executable).resolve())
    lines = [
        "(version 1)",
        "(deny default)",
        "(deny network*)",
        "(allow file-read-metadata)",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm*)",
        "(allow file-read*)",
    ]
    lines.append("(deny file-read*")
    for tree in _USER_DATA_TREES:
        lines.append(f'    (subpath "{_escape(tree)}")')
    lines.append(")")
    # Re-allowed after the deny, because the interpreter and the scratch
    # directory both commonly live inside a denied tree (a venv under the
    # user's home, a temp dir under /private/var/folders). Seatbelt is
    # last-match-wins, so order is load-bearing here.
    lines.append("(allow file-read*")
    for path in read_paths:
        lines.append(f'    (subpath "{_escape(path)}")')
    lines.append(f'    (subpath "{_escape(str(scratch))}")')
    lines.append('    (literal "/dev/null")')
    lines.append('    (literal "/dev/urandom")')
    lines.append('    (literal "/dev/random")')
    lines.append(")")
    # Writes are confined to the scratch directory, so everything the code
    # produces is inspectable and disposable.
    lines.append("(allow file-write*")
    lines.append(f'    (subpath "{_escape(str(scratch))}")')
    lines.append('    (literal "/dev/null")')
    lines.append(")")
    # Only the interpreter may be executed — no shell, no helper binaries.
    lines.append("(deny process-exec*)")
    lines.append("(allow process-exec")
    for path in _interpreter_paths():
        lines.append(f'    (subpath "{_escape(path)}")')
    lines.append(f'    (literal "{_escape(interpreter)}")')
    lines.append(f'    (literal "{_escape(str(sys.executable))}")')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _bubblewrap_argv(*, scratch: Path, read_paths: Sequence[str]) -> list[str]:
    argv = [
        _which("bwrap"),
        "--unshare-all",          # no network, no pids, no ipc
        "--die-with-parent",
        "--new-session",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    for path in read_paths:
        argv += ["--ro-bind-try", path, path]
    argv += ["--bind", str(scratch), str(scratch)]
    argv += ["--chdir", str(scratch)]
    return argv


# ── The child harness ─────────────────────────────────────────────────────

#: Runs inside the sandbox. Applies rlimits the kernel boundary does not
#: cover (CPU, address space, file size, subprocess count), then either
#: executes the script or imports it and calls one function.
_HARNESS = r'''
import json, sys, io, contextlib, traceback, runpy

try:
    import resource
except ImportError:
    resource = None

request = json.loads(sys.stdin.read())
limits = request.get("limits") or {}
warnings = []

def _limit(kind_name, value):
    if resource is None or not value:
        return
    kind = getattr(resource, kind_name, None)
    if kind is None:
        return
    try:
        soft, hard = resource.getrlimit(kind)
        target = int(value)
        new_hard = target if hard == resource.RLIM_INFINITY else min(hard, target)
        resource.setrlimit(kind, (min(target, new_hard), new_hard))
    except (OSError, ValueError) as exc:
        warnings.append("%s: %r" % (kind_name, exc))

_limit("RLIMIT_CPU", limits.get("cpu_seconds"))
_limit("RLIMIT_FSIZE", limits.get("file_size_bytes"))
_limit("RLIMIT_NPROC", limits.get("processes"))
if sys.platform != "darwin":
    # RLIMIT_AS on macOS breaks CPython's own allocator before user code runs.
    _limit("RLIMIT_AS", limits.get("mem_bytes"))

module_path = request["module_path"]
mode = request.get("mode", "script")
out = io.StringIO()
err = io.StringIO()
payload = {"status": "ok", "results": [], "warnings": warnings}

try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        if mode == "script":
            runpy.run_path(module_path, run_name="__main__")
        else:
            namespace = runpy.run_path(module_path, run_name="aura_untrusted_candidate")
            target = namespace.get(request["function"])
            if not callable(target):
                raise NameError("%s is not defined or not callable" % request["function"])
            for call in request.get("calls", []):
                value = target(*call.get("args", []), **call.get("kwargs", {}))
                payload["results"].append(value)
except BaseException as exc:  # noqa: BLE001 - the whole point is to report anything
    payload["status"] = "error"
    payload["error"] = repr(exc)
    payload["traceback"] = traceback.format_exc()

payload["stdout"] = out.getvalue()
payload["stderr"] = err.getvalue()

try:
    encoded = json.dumps(payload)
except (TypeError, ValueError):
    # A function may legitimately return something unserialisable. Say so
    # rather than losing the whole run to an encoder error.
    payload["results"] = [repr(r) for r in payload["results"]]
    payload["unserialisable_results"] = True
    encoded = json.dumps(payload)

sys.__stdout__.write("\x00AURA_SANDBOX\x00" + encoded)
'''

_SENTINEL = "\x00AURA_SANDBOX\x00"


# ── Public API ────────────────────────────────────────────────────────────


def run_untrusted_script(
    code: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    extra_read_paths: Sequence[str | os.PathLike[str]] = (),
    require_boundary: bool = True,
    source: str = "unknown",
) -> SandboxOutcome:
    """Execute model-written Python as a script under a kernel boundary."""
    return _execute(
        code,
        mode="script",
        function=None,
        calls=(),
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
        extra_read_paths=extra_read_paths,
        require_boundary=require_boundary,
        source=source,
    )


def call_untrusted_function(
    code: str,
    function: str,
    calls: Sequence[Sequence[Any]] | Sequence[dict[str, Any]] = (),
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    extra_read_paths: Sequence[str | os.PathLike[str]] = (),
    require_boundary: bool = True,
    source: str = "unknown",
) -> SandboxOutcome:
    """Load model-written Python and call one function in it, out-of-process.

    ``calls`` is a sequence of positional-argument tuples, or of
    ``{"args": [...], "kwargs": {...}}`` mappings. Every call runs in one
    child, in order, so a candidate that is stateful behaves the way it
    would in-process — minus the ability to touch this machine.

    Arguments and return values cross a JSON boundary. That is a real
    constraint and a deliberate one: anything richer would hand the
    untrusted side a live object graph, which is the thing being prevented.
    """
    normalised: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            normalised.append(
                {"args": list(call.get("args", [])), "kwargs": dict(call.get("kwargs", {}))}
            )
        else:
            normalised.append({"args": list(call), "kwargs": {}})
    return _execute(
        code,
        mode="call",
        function=function,
        calls=normalised,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
        extra_read_paths=extra_read_paths,
        require_boundary=require_boundary,
        source=source,
    )


def _execute(
    code: str,
    *,
    mode: str,
    function: str | None,
    calls: Sequence[dict[str, Any]],
    timeout_s: float,
    mem_bytes: int,
    extra_read_paths: Sequence[str | os.PathLike[str]],
    require_boundary: bool,
    source: str,
) -> SandboxOutcome:
    if not isinstance(code, str) or not code.strip():
        return SandboxOutcome(status="rejected", error="no code supplied")
    if len(code.encode("utf-8", errors="replace")) > DEFAULT_CODE_BYTES:
        return SandboxOutcome(
            status="rejected",
            error=f"code payload exceeds {DEFAULT_CODE_BYTES} bytes",
        )

    boundary = available_boundary()
    if not boundary:
        if require_boundary and not _unconfined_permitted():
            # The refusal. Running it anyway is how a benchmark becomes an
            # execution service for whatever the model happened to write.
            return SandboxOutcome(
                status="no_boundary",
                error=(
                    "no OS sandbox available on this host "
                    f"(platform={sys.platform}); refusing to execute untrusted "
                    f"code. Set {UNCONFINED_ENV}=1 to run unconfined, which "
                    "will be recorded on every result."
                ),
            )
        logger.warning(
            "executing untrusted code UNCONFINED (source=%s): no OS boundary available",
            source,
        )

    with tempfile.TemporaryDirectory(prefix="aura_untrusted_") as tmp:
        # Resolved, not as-returned. TemporaryDirectory hands back
        # /var/folders/... while the kernel sees /private/var/folders/... —
        # and Seatbelt matches the real path. An unresolved scratch path
        # means the profile's own allow rule never matches its own scratch
        # directory, the child cannot read the harness it was given, and
        # CPython aborts with SIGABRT and no diagnostic whatsoever.
        scratch = Path(tmp).resolve()
        module_path = scratch / "candidate.py"
        harness_path = scratch / "_aura_harness.py"
        atomic_write_text(module_path, code)
        atomic_write_text(harness_path, _HARNESS)

        read_paths = _interpreter_paths()
        read_paths += [str(Path(p).resolve()) for p in extra_read_paths]

        argv = _wrap_with_boundary(
            boundary,
            scratch=scratch,
            read_paths=read_paths,
            command=[sys.executable, "-I", "-B", "-S", str(harness_path)],
        )
        request = json.dumps(
            {
                "module_path": str(module_path),
                "mode": mode,
                "function": function,
                "calls": list(calls),
                "limits": {
                    "cpu_seconds": max(1, int(timeout_s)),
                    "mem_bytes": int(mem_bytes),
                    "file_size_bytes": DEFAULT_FILE_SIZE_BYTES,
                    "processes": 64,
                },
            }
        )
        return _spawn(
            argv,
            request,
            timeout_s=timeout_s,
            boundary=boundary,
            scratch=scratch,
            source=source,
            env=_child_environment(scratch),
        )


#: The only environment variables untrusted code inherits.
#:
#: A live escape probe found the original version handing the child this
#: process's entire environment. A kernel boundary that blocks the
#: filesystem and then passes ANTHROPIC_API_KEY in os.environ has not
#: stopped exfiltration, it has just made it require one more line. The
#: allowlist is short on purpose: anything not here is something untrusted
#: code has no business knowing.
_ENV_ALLOWLIST: frozenset[str] = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "TZ"})


def _child_environment(scratch: Path) -> dict[str, str]:
    """A minimal, secret-free environment for the sandboxed child."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _ENV_ALLOWLIST
    }
    # PATH is deliberately empty rather than absent: an empty PATH means
    # "find nothing", while an absent one makes some libraries fall back to
    # a compiled-in default that includes /usr/bin.
    env["PATH"] = ""
    env["HOME"] = str(scratch)
    env["TMPDIR"] = str(scratch)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _wrap_with_boundary(
    boundary: str,
    *,
    scratch: Path,
    read_paths: Sequence[str],
    command: Sequence[str],
) -> list[str]:
    if boundary == "seatbelt":
        profile_path = scratch / "profile.sb"
        atomic_write_text(profile_path, _seatbelt_profile(scratch=scratch, read_paths=read_paths))
        # Absolute path, because the child environment carries an empty
        # PATH by design. Naming the bare binary makes the boundary itself
        # unresolvable, and the whole run fails to spawn.
        return [_which("sandbox-exec"), "-f", str(profile_path), *command]
    if boundary == "bubblewrap":
        return [*_bubblewrap_argv(scratch=scratch, read_paths=read_paths), *command]
    return list(command)


def _which(binary: str) -> str:
    """Absolute path to a boundary binary, resolved against the real PATH."""
    resolved = shutil.which(binary)
    if not resolved:
        raise UntrustedExecutionError(f"{binary} disappeared between discovery and use")
    return resolved


def _spawn(
    argv: Sequence[str],
    request: str,
    *,
    timeout_s: float,
    boundary: str,
    scratch: Path,
    source: str,
    env: dict[str, str] | None = None,
) -> SandboxOutcome:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    try:
        # Running confined code on the runtime's own behalf is internal
        # maintenance, and the gateway refuses maintenance that does not say
        # so.
        #
        # LIVE, 2026-08-22: every turn logged "GOVERNANCE VIOLATION:
        # subprocess_gateway.run:sandbox.untrusted_python.symbolic_cognition
        # called outside governed context", recorded a degradation and raised
        # a MARGINAL fault — so the symbolic boundary this sandbox exists to
        # provide was unavailable on every single turn.
        with local_internal_governed_scope(f"sandbox.untrusted_python.{source}"):
            completed = get_subprocess_gateway().run(
                list(argv),
                input=request,
                timeout=timeout_s + 5.0,
                capture_output=True,
                cwd=str(scratch),
                env=env,
                # Executing generated Python is an effect even when the kernel
                # confines every write to this ephemeral scratch directory.
                read_only=False,
                source=f"sandbox.untrusted_python.{source}",
                accelerator_capability="auto",
            )
    except subprocess.TimeoutExpired:
        return SandboxOutcome(
            status="timeout",
            error=f"untrusted code exceeded {timeout_s}s",
            sandboxed=bool(boundary),
            boundary=boundary or "none",
        )
    except _SANDBOX_ERRORS as exc:
        record_degradation("untrusted_python", exc, action="untrusted execution failed to spawn")
        return SandboxOutcome(
            status="spawn_failed",
            error=repr(exc),
            sandboxed=bool(boundary),
            boundary=boundary or "none",
        )

    stdout = _text(getattr(completed, "stdout", ""))
    stderr = _truncate(_text(getattr(completed, "stderr", "")))
    returncode = getattr(completed, "returncode", None)

    marker = stdout.rfind(_SENTINEL)
    if marker < 0:
        # The harness never reported. Either the boundary refused to start
        # the interpreter or the child was killed; both are failures, and
        # neither may be reported as an ordinary empty result. Name which
        # one, because "no result" sends a reader looking for a bug in
        # their code when the answer is "it burned its CPU budget".
        status, detail = _classify_death(returncode)
        return SandboxOutcome(
            status=status,
            stdout=_truncate(stdout),
            stderr=stderr,
            returncode=returncode,
            sandboxed=bool(boundary),
            boundary=boundary or "none",
            error=detail,
        )

    try:
        payload = json.loads(stdout[marker + len(_SENTINEL):])
    except (json.JSONDecodeError, ValueError) as exc:
        return SandboxOutcome(
            status="no_result",
            stdout=_truncate(stdout),
            stderr=stderr,
            returncode=returncode,
            sandboxed=bool(boundary),
            boundary=boundary or "none",
            error=f"unparseable result payload: {exc!r}",
        )

    return SandboxOutcome(
        status=str(payload.get("status", "error")),
        stdout=_truncate(str(payload.get("stdout") or "")),
        stderr=_truncate(
            "\n".join(p for p in (str(payload.get("stderr") or ""), stderr) if p)
        ),
        returncode=returncode,
        sandboxed=bool(boundary),
        boundary=boundary or "none",
        results=list(payload.get("results") or []),
        error=str(payload.get("error") or ""),
    )


def _classify_death(returncode: int | None) -> tuple[str, str]:
    """Name the way a child died, from its exit status."""
    import signal

    if returncode is None:
        return "no_result", "sandboxed child produced no result payload"
    if returncode >= 0:
        return (
            "no_result",
            f"sandboxed child exited {returncode} without a result payload",
        )
    signum = -returncode
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    if signum == getattr(signal, "SIGXCPU", -1):
        return "timeout", "untrusted code exhausted its CPU budget (SIGXCPU)"
    if signum == getattr(signal, "SIGXFSZ", -1):
        return "resource_limit", "untrusted code exceeded its file-size limit (SIGXFSZ)"
    if signum in {getattr(signal, "SIGKILL", -1), getattr(signal, "SIGSEGV", -1)}:
        return "killed", f"sandboxed child terminated by {name}"
    return "killed", f"sandboxed child terminated by {name}"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _truncate(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


__all__ = [
    "DEFAULT_MEM_BYTES",
    "DEFAULT_TIMEOUT_S",
    "UNCONFINED_ENV",
    "SandboxOutcome",
    "UntrustedExecutionError",
    "available_boundary",
    "call_untrusted_function",
    "run_untrusted_script",
]
