#!/usr/bin/env python3
"""Run one proof step with a hard timeout and an evidence artifact.

The final-proof chain must never hang: a proof step that cannot finish
is a FAILED step with a written artifact, not an overnight mystery.
Wraps any command:

    python tools/run_proof_step.py --name dnu_battery --timeout 5400 \
        --artifact artifacts/current/proof_steps/dnu_battery.json -- \
        python tools/agi/run_dnu_agi_proof_battery.py --full ...

Writes {name, command, returncode, timed_out, duration_s, started_at,
finished_at, passed} and mirrors the child's exit code (124 on
timeout, after SIGKILLing the child's process group).
Prepared and running checkpoints precede execution. An interrupted
checkpoint keeps passed=false until the child has actually finished.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_MONO_START = 0.0


def _save(path: Path, record: dict) -> None:
    """Persist each phase atomically before further child work."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('proof_step_checkpoint'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source='proof_step_checkpoint')
        gateway.write_text(path, json.dumps(record, indent=2) + '\n', source='proof_step_checkpoint')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')

    global _MONO_START
    _MONO_START = time.monotonic()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("run_proof_step: no command given", file=sys.stderr)
        return 2

    started = time.time()
    timed_out = False
    returncode: int | None = None
    # Sleep-proofing (the 9-hour 'hang' was a lid-closed MacBook):
    # caffeinate -i holds off idle sleep for exactly this step's
    # lifetime, and the wait loop below checks WALL clock too —
    # monotonic deadlines freeze during sleep, wall time does not, so
    # a sleep-spanning step is killed on wake instead of resuming as
    # a zombie run.
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-i", *command]
    artifact = {"schema": "aura.proof_step.v1", "name": args.name, "command": command,
                "status": "prepared", "returncode": None, "timed_out": False,
                "timeout_s": args.timeout, "started_at": started, "passed": False,
                "supervisor_pid": os.getpid(), "task_outcome": "requires_independent_review"}
    _save(args.artifact, artifact)
    try:
        proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    except OSError as why:
        artifact.update(status='launch_failed', error=f'{type(why).__name__}: {why}', finished_at=time.time())
        _save(args.artifact, artifact)
        return 127
    artifact.update(status='running', child_pid=proc.pid, process_group=proc.pid)

    # The child owns its session so timeout can SIGKILL its whole group
    # without touching this wrapper. The flip side: kill THIS wrapper and
    # the detached child survives as an orphan — every zombie battery on
    # June 11 was born that way. Forward fatal signals to the child's
    # group so wrapper death is child death.
    interrupted = None
    def _reap_child(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = signum
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    previous_handlers = {}
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous_handlers[_sig] = signal.signal(_sig, _reap_child)
    try:
        _save(args.artifact, artifact)
    except BaseException:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        raise

    wall_deadline = started + args.timeout * 1.25 + 60.0
    try:
        # Bounded poll: the slice count is derived from the wall deadline,
        # so this loop terminates even if monotonic time froze in sleep.
        max_slices = math.ceil((args.timeout * 1.25 + 120.0) / .25) + 2
        returncode = None
        for _ in range(max_slices):
            remaining = args.timeout - (time.monotonic() - _MONO_START)
            try:
                returncode = proc.wait(timeout=min(.25, max(.001, remaining)))
                break
            except subprocess.TimeoutExpired:
                if remaining <= 0 or time.time() > wall_deadline:
                    raise
        if returncode is None:
            raise subprocess.TimeoutExpired(command, args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        returncode = 124
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    finished = time.time()
    if interrupted is not None:
        returncode = 128 + interrupted
    artifact.update({
        "status": 'interrupted' if interrupted is not None else 'timed_out' if timed_out else 'finished',
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(finished - started, 2),
        "finished_at": finished,
        "child_reaped": proc.returncode is not None,
        "interrupted_by": interrupted,
        "passed": (returncode == 0) and not timed_out and interrupted is None,
    })
    _save(args.artifact, artifact)

    marker = "✅" if artifact["passed"] else ("⏱️ TIMEOUT" if timed_out else "❌")
    print(
        f"{marker} proof step '{args.name}': rc={returncode} "
        f"in {artifact['duration_s']}s → {args.artifact}",
        flush=True,
    )
    return int(returncode if returncode is not None else 1)


if __name__ == "__main__":
    sys.exit(main())
