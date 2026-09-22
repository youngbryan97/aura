"""Exact process identity helpers for lifecycle and cleanup tooling."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_IDENTITY_ERRORS = (AttributeError, OSError, TypeError, ValueError, RuntimeError)

_PYTHON_EXECUTABLE = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?$", re.IGNORECASE)
_PYTHON_OPTIONS_WITH_VALUE = frozenset({"-W", "-X", "--check-hash-based-pycs"})


def python_script_argument(cmdline: Sequence[Any]) -> str | None:
    """Return the script Python executes, excluding text and module invocations."""

    arguments = [str(item) for item in cmdline if str(item)]
    if not arguments:
        return None
    executable = Path(arguments[0]).name
    if _PYTHON_EXECUTABLE.fullmatch(executable) is None:
        return None

    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            return arguments[index] if index < len(arguments) else None
        if argument in {"-c", "-m"}:
            return None
        if argument in _PYTHON_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if argument.startswith("-W") or argument.startswith("-X"):
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument
    return None


def command_invokes_python_script(
    cmdline: Sequence[Any],
    *,
    expected_script: str | Path,
    cwd: str | Path = "",
) -> bool:
    """Match one exact script path, never a substring elsewhere in argv."""

    script_argument = python_script_argument(cmdline)
    if not script_argument:
        return False
    expected = Path(expected_script).expanduser().resolve(strict=False)
    observed = Path(script_argument).expanduser()
    if not observed.is_absolute():
        if not str(cwd or "").strip():
            return False
        observed = Path(cwd).expanduser() / observed
    return observed.resolve(strict=False) == expected


def process_invokes_python_script(
    process: Any,
    *,
    expected_script: str | Path,
) -> bool:
    return command_invokes_python_script(
        getattr(process, "cmdline", ()) or (),
        expected_script=expected_script,
        cwd=getattr(process, "cwd", "") or "",
    )


def select_script_process_tree(
    processes: Iterable[Any],
    *,
    expected_scripts: Iterable[str | Path],
    protected_pids: Iterable[int] = (),
) -> tuple[Any, ...]:
    """Select exact script roots and their observed descendants.

    Returned observations are ordered root-first for graceful termination.
    Callers must still revalidate PID creation time immediately before sending
    a signal so PID reuse cannot target an unrelated process.
    """

    observations = tuple(processes)
    protected = {int(pid) for pid in protected_pids if int(pid) > 0}
    scripts = tuple(Path(path).expanduser().resolve(strict=False) for path in expected_scripts)
    roots = {
        int(getattr(process, "pid", 0) or 0)
        for process in observations
        if int(getattr(process, "pid", 0) or 0) not in protected
        and any(
            process_invokes_python_script(process, expected_script=script)
            for script in scripts
        )
    }
    if not roots:
        return ()

    selected = []
    for process in observations:
        pid = int(getattr(process, "pid", 0) or 0)
        if pid <= 0 or pid in protected:
            continue
        ancestors = {
            int(parent)
            for parent in (getattr(process, "ancestor_pids", ()) or ())
            if int(parent) > 0
        }
        if pid in roots or ancestors & roots:
            selected.append(process)
    selected.sort(
        key=lambda process: (
            int(getattr(process, "pid", 0) or 0) not in roots,
            len(getattr(process, "ancestor_pids", ()) or ()),
            int(getattr(process, "pid", 0) or 0),
        )
    )
    return tuple(selected)


# ── Creation-time binding ─────────────────────────────────────────────────
#
# ``select_script_process_tree`` above already states the requirement:
# "Callers must still revalidate PID creation time immediately before
# sending a signal so PID reuse cannot target an unrelated process." It gave
# them no way to do it, so nobody did.
#
# CP126 against core/brain/inference_gate.py: "Recovery kills a private
# process handle without generation ownership proof. There is no request
# generation, PID start-time, model identity, foreground-owner, or
# warmup-task binding at the kill point."
#
# A PID is not an identity — it is a slot the kernel reuses, and the
# recovery path has the shape that makes reuse reachable:
#
#     proc = getattr(client, "_process", None)   # read the handle
#     ... several async checks, each an await point ...
#     await asyncio.to_thread(client._kill_and_join_blocking, proc)
#
# Between the read and the kill the worker can exit, the client can spawn a
# replacement, and on a busy host the replacement can land on the same PID.
# The kill then lands on a healthy new worker — and it looks like recovery
# working exactly as designed, because the kill succeeded and a worker did
# die.
#
# Creation time comes from the kernel and is fixed for the life of a
# process. Capture it when the decision is made; re-check it at the moment
# of the kill. A mismatch is not an error — it is the recovery having
# already happened by itself.


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID plus the one attribute that makes it unique over time."""

    pid: int
    create_time: float | None
    #: What the caller believed it was binding to — a model name, a lane, a
    #: warmup generation. Not used for matching; recorded so a refused kill
    #: can say what it thought it was killing.
    label: str = ""

    @property
    def bound(self) -> bool:
        """True when creation time was readable, i.e. the binding is real.

        A binding without ``create_time`` is a PID comparison wearing a
        dataclass. Callers that must not act on a weak binding check this.
        """
        return self.create_time is not None

    def describe(self) -> str:
        stamp = "unknown" if self.create_time is None else f"{self.create_time:.3f}"
        described = f"pid={self.pid} created={stamp}"
        return f"{described} ({self.label})" if self.label else described


def capture_identity(process: Any, *, label: str = "") -> ProcessIdentity | None:
    """Bind to ``process`` as it exists right now, or None if it has no PID."""

    pid = _pid_of(process)
    if pid is None:
        return None
    return ProcessIdentity(pid=pid, create_time=_create_time(pid), label=label)


def identity_still_current(identity: ProcessIdentity | None, process: Any) -> bool:
    """Is ``process`` still the exact process ``identity`` was taken from?

    False when the handle now names a different PID, when the PID is gone,
    or when the PID exists with a different creation time — the reuse case,
    and the only one a plain PID check misses.
    """

    if identity is None:
        return False
    pid = _pid_of(process)
    if pid is None or pid != identity.pid:
        return False
    if identity.create_time is None:
        # Nothing to compare against. Fall back to liveness, and let the
        # caller see via ``identity.bound`` that this was the weak check.
        return _pid_alive(pid)
    current = _create_time(pid)
    if current is None:
        return False
    # Creation times come back with platform-dependent resolution, so allow
    # a millisecond. PID reuse within a millisecond of the original's start
    # is not a thing.
    return abs(current - identity.create_time) < 0.001


def assert_owned(
    identity: ProcessIdentity | None,
    process: Any,
    *,
    action: str,
    subsystem: str = "process_identity",
) -> bool:
    """Check ownership at the point of action, and record a refusal.

    Returns True when the caller may proceed. A False is usually good news —
    the process it wanted to kill is already gone — so it logs at info
    rather than as a fault.
    """

    if identity_still_current(identity, process):
        return True
    described = identity.describe() if identity else "<unbound>"
    _logger.info(
        "🛡️ %s: refusing %s — %s is no longer the process this decision was "
        "bound to (exited, replaced, or PID reused).",
        subsystem,
        action,
        described,
    )
    return False


def _pid_of(process: Any) -> int | None:
    try:
        pid_int = int(getattr(process, "pid", None))
    # not a failure: an object with no usable pid has none to report.
    except (TypeError, ValueError):
        return None
    return pid_int if pid_int > 0 else None


def _create_time(pid: int) -> float | None:
    """Kernel-reported creation time, or None when it cannot be read."""

    # ResourceObserver is the single owner of process observation in this
    # codebase, and it already exposes create_time. The direct psutil
    # fallback that used to sit here was redundant AND a second, unowned
    # observation path — the exact thing the resource-observation ownership
    # gate exists to prevent. One reader, one owner.
    try:
        from core.runtime.resource_observation import get_resource_observer

        observed = get_resource_observer().process(pid)
        create_time = getattr(observed, "create_time", None)
        if create_time is not None:
            return float(create_time)
    except _IDENTITY_ERRORS as exc:
        _logger.debug("resource observer could not time pid %s: %s", pid, exc)
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    # not a failure: no such process is the answer this asks for.
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else — and a process we do not own is one
        # we must not be killing anyway.
        return True
    except OSError as exc:
        # Every other OSError is the question going unanswered, and False
        # here reads as "that process is gone".
        _logger.debug("liveness check for pid %s failed: %s: %s", pid, type(exc).__name__, exc)
        return False
    return True


__all__ = [
    "ProcessIdentity",
    "assert_owned",
    "capture_identity",
    "command_invokes_python_script",
    "identity_still_current",
    "process_invokes_python_script",
    "python_script_argument",
    "select_script_process_tree",
]
