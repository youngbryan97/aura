"""A child Aura limits may start the processes its limit allows, however many its user already runs (U07, F16).

LIVE 21 November 2026, typed into her window: "Find the two largest Python files under core/brain by line count, count
their lines by actually running something, ..." Her code started ``wc``, and code_repl failed it twice with
"BlockingIOError: [Errno 35] Resource temporarily unavailable". RLIMIT_NPROC counts every process of the user, on macOS
as on Linux, and her children were given 32, 64 or 256 while her user ran 275: no process could start in any of them,
make and pytest under her process supervisor included. The limit is now the user's count, taken in the parent, plus
what the child may start. Under the kernel boundary her code still may not start another program, and is now told so
as that, not as a resource that may come free.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

import pytest

from core.runtime.constrained_exec import (
    RLIMIT_PROCESSES,
    child_preexec,
    process_ceiling,
    processes_this_user_runs,
    scrubbed_env,
)

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX limits")]

#: A child that starts one more process and says what came of it.
_STARTS_ONE = (
    "import subprocess\n"
    "try:\n"
    "    print(subprocess.run(['/bin/echo', 'started'], capture_output=True, text=True).stdout.strip())\n"
    "except OSError as exc:\n"
    "    print(type(exc).__name__)\n"
)


def _child(preexec) -> str:
    done = subprocess.run([sys.executable, "-I", "-S", "-c", _STARTS_ONE], capture_output=True, text=True, timeout=60,
                          preexec_fn=preexec, start_new_session=True, env=scrubbed_env())
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_the_ceiling_is_what_the_user_runs_and_the_allowance():
    running = processes_this_user_runs()
    assert running is not None and running > 0
    assert process_ceiling(RLIMIT_PROCESSES) >= running + RLIMIT_PROCESSES - 8  # others start and end meanwhile


def test_a_child_under_its_ceiling_can_start_a_process():
    assert _child(functools.partial(child_preexec, process_ceiling(RLIMIT_PROCESSES))) == "started"


def test_the_allowance_alone_refused_every_process_start():
    if (processes_this_user_runs() or 0) <= RLIMIT_PROCESSES:
        pytest.skip("this user runs fewer processes than the allowance, so the old limit did not bind")
    assert _child(functools.partial(child_preexec, RLIMIT_PROCESSES)) == "BlockingIOError"


def test_a_job_her_process_supervisor_starts_can_start_its_own():
    from core.actuators import process_supervisor

    preexec = functools.partial(process_supervisor._rlimit_preexec, process_ceiling(process_supervisor._RLIMIT_NPROC))
    assert _child(preexec) == "started"


def test_under_the_kernel_boundary_a_process_start_is_refused_as_not_permitted():
    from core.sandbox.runner import run_untrusted
    from core.sandbox.untrusted_python import available_boundary

    if not available_boundary():
        pytest.skip("no kernel boundary on this host")
    result = run_untrusted("import subprocess\nsubprocess.run(['/bin/echo', 'started'])")
    assert result["status"] == "error"
    assert "PermissionError" in result["repr"] and "BlockingIOError" not in result["repr"], result
