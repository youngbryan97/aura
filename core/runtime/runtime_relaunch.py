"""Make a reboot request actually come back up.

The desktop UI's Reboot control raises SIGTERM, and the endpoint that does it
documented "supervisor restarts". That is only true under launchd, which sets
``AURA_SUPERVISED=1``. On a direct/source launch — the ordinary way this
runtime is started, PPID 1, no supervisor anywhere — the button was a kill
switch: Aura went down and nothing brought her back, with no warning that
"Reboot" meant "shut down".

So when nothing else will replace this process, arrange it here: hand the
original argv to a detached waiter that starts the replacement only AFTER this
process is gone and its port is free. Refusing to start beside a live runtime
is the whole safety property — a second 32B on this host is worse than a
runtime that stayed down — so every wait is bounded and every failure to
observe the old process exiting means the waiter declines rather than guesses.
"""

from __future__ import annotations

import errno
import os
import socket
import subprocess
import sys
import time
from typing import Any

from core.runtime.errors import record_degradation

_SUBSYSTEM = "runtime.relaunch"

# The old runtime has to unload ~20GB of weights and release its port. Two
# minutes is generous for that and still bounded; past it, the waiter declines.
EXIT_WAIT_TIMEOUT_S = 120.0
PORT_WAIT_TIMEOUT_S = 45.0
_POLL_INTERVAL_S = 0.5


def supervisor_will_restart() -> bool:
    """True only when something outside this process is going to bring it back."""

    return os.environ.get("AURA_SUPERVISED") == "1"


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            # Alive, owned by someone else. Not ours to replace.
            return True
        return True
    return True


def _port_is_free(port: int) -> bool:
    if port <= 0:
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _port_from_argv(argv: list[str]) -> int:
    for index, token in enumerate(argv):
        if token == "--port" and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return 0
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def wait_for_predecessor(pid: int, port: int) -> str:
    """Block until `pid` is gone and `port` is free. Returns "" when clear."""

    deadline = time.monotonic() + EXIT_WAIT_TIMEOUT_S
    while _process_alive(pid):
        if time.monotonic() >= deadline:
            return f"predecessor_still_running:pid={pid}"
        time.sleep(_POLL_INTERVAL_S)

    port_deadline = time.monotonic() + PORT_WAIT_TIMEOUT_S
    while not _port_is_free(port):
        if time.monotonic() >= port_deadline:
            return f"port_still_bound:port={port}"
        time.sleep(_POLL_INTERVAL_S)
    return ""


def schedule_relaunch(
    *,
    pid: int | None = None,
    argv: list[str] | None = None,
    cwd: str | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Spawn the detached waiter that will replace this runtime once it exits.

    Returns a receipt describing what was actually arranged. Callers must send
    the shutdown signal themselves — nothing here stops the runtime, so a
    failure to arrange the relaunch can be reported before anything is killed.
    """

    resolved_pid = int(pid if pid is not None else os.getpid())
    resolved_argv = list(argv if argv is not None else sys.argv)
    resolved_cwd = str(cwd or os.getcwd())
    resolved_executable = str(executable or sys.executable)

    if not resolved_argv:
        return {"scheduled": False, "reason": "no_argv_to_replay"}

    port = _port_from_argv(resolved_argv)
    command = [
        resolved_executable,
        "-m",
        "core.runtime.runtime_relaunch",
        "--pid",
        str(resolved_pid),
        "--port",
        str(port),
        "--cwd",
        resolved_cwd,
        "--",
        *resolved_argv,
    ]

    try:
        # start_new_session detaches the waiter from this process group, so the
        # SIGTERM that stops the runtime does not also stop its replacement.
        # stdout/stderr are inherited on purpose: the replacement keeps writing
        # to whatever log the original launch was redirected into.
        # Through the gateway, like every other spawn in the tree — a raw
        # Popen here is exactly the ownership hole the effect lint exists to
        # catch. allow_during_shutdown is the point of this call: the waiter
        # must be arranged BEFORE the runtime stops, or the reboot leaves Aura
        # down with nothing to bring it back.
        # Arranging your own replacement is internal maintenance, and the
        # gateway refuses maintenance that does not say so.
        #
        # LIVE DEFECT, 2026-08-22: pressing Reboot in the header answered 500
        # with "GovernanceViolationError: subprocess_gateway.spawn:
        # runtime_relaunch:schedule_relaunch called outside governed context".
        # The scope belongs here rather than at each caller: this function
        # owns the spawn, so it is the thing that has to declare it.
        from core.governance_context import local_internal_governed_scope
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        with local_internal_governed_scope("runtime_relaunch.schedule_relaunch"):
            child = get_subprocess_gateway().spawn(
                command,
                cwd=resolved_cwd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                allow_during_shutdown=True,
                source="runtime_relaunch:schedule_relaunch",
                accelerator_capability="auto",
            )
    except (OSError, ValueError) as exc:
        record_degradation(
            _SUBSYSTEM,
            exc,
            action="could not schedule the runtime relaunch; reboot would leave Aura down",
            severity="critical",
        )
        return {"scheduled": False, "reason": f"spawn_failed:{type(exc).__name__}"}

    return {
        "scheduled": True,
        "waiter_pid": child.pid,
        "replacing_pid": resolved_pid,
        "port": port,
        "argv": resolved_argv,
    }


def main(raw_args: list[str] | None = None) -> int:
    """Waiter entry point: hold until the predecessor is gone, then exec it."""

    args = list(raw_args if raw_args is not None else sys.argv[1:])
    if "--" not in args:
        print("runtime_relaunch: refusing to run without a '--' argv separator")
        return 2
    separator = args.index("--")
    options, replay = args[:separator], args[separator + 1 :]
    if not replay:
        print("runtime_relaunch: refusing to run with an empty replay argv")
        return 2

    pid = 0
    port = 0
    cwd = os.getcwd()
    for index, token in enumerate(options):
        value = options[index + 1] if index + 1 < len(options) else ""
        if token == "--pid":
            pid = int(value or 0)
        elif token == "--port":
            port = int(value or 0)
        elif token == "--cwd":
            cwd = value or cwd

    blocker = wait_for_predecessor(pid, port)
    if blocker:
        # Declining is the correct outcome: a second runtime beside a live one
        # doubles the resident model and is worse than staying down.
        print(f"runtime_relaunch: declined — {blocker}")
        return 1

    try:
        os.chdir(cwd)
    except OSError as exc:
        print(f"runtime_relaunch: cannot enter {cwd}: {exc}")
        return 1

    executable = replay[0]
    print(f"runtime_relaunch: replacing pid {pid} — exec {' '.join(replay)}")
    sys.stdout.flush()
    try:
        os.execv(executable, replay)
    except OSError as exc:
        print(f"runtime_relaunch: exec failed: {exc}")
        return 1
    return 0  # pragma: no cover - execv does not return


if __name__ == "__main__":
    raise SystemExit(main())
