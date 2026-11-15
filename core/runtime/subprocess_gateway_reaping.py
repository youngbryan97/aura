"""How a child is bounded by its work and reaped when it ends or overruns.

Lifted whole out of `subprocess_gateway`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from typing import IO, Any


def _terminate_and_reap_python_process(
    process: Any,
    *,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> bool:
    """Bounded handle-based termination; never signal an observed PID directly."""
    from .subprocess_gateway import (
        logger,
    )


    try:
        alive = bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = True
    if alive:
        try:
            process.terminate()
        except (AttributeError, OSError, RuntimeError, ValueError):
            # not a failure: every rung may fail, the next is harder, and the
            # is_alive() below is the verdict.
            pass
    try:
        process.join(timeout=max(0.0, float(terminate_timeout_s)))
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        pass  # not a failure: see the rung above.
    try:
        alive = bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = True
    if alive:
        try:
            process.kill()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass  # not a failure: see the rung above.
        try:
            process.join(timeout=max(0.0, float(kill_timeout_s)))
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            pass  # not a failure: see the rung above.
    try:
        return not bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        # A failure: the ladder ran and the verdict cannot be read, so the caller
        # is told the process may still be alive.
        logger.warning("Could not confirm the process ended: %s", exc)
        return False


async def _terminate_async_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_s: float = 5.0,
) -> tuple[bytes | None, bytes | None]:
    """Terminate and reap an isolated async process and all descendants."""
    from .subprocess_gateway import (
        logger,
    )

    process_group_id = int(getattr(process, "_aura_process_group_id", 0) or 0)
    try:
        if process_group_id <= 0:
            process_group_id = int(os.getpgid(process.pid))
    except (OSError, ProcessLookupError, ValueError):
        if bool(getattr(process, "_aura_start_new_session", False)):
            process_group_id = int(process.pid)

    try:
        if process_group_id > 0 and process_group_id != os.getpgrp():
            os.killpg(process_group_id, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass  # not a failure: the SIGKILL rung below covers a live group.
    try:
        return await asyncio.wait_for(process.communicate(), timeout=max(0.1, grace_s))
    except TimeoutError:
        try:
            if process_group_id > 0 and process_group_id != os.getpgrp():
                os.killpg(process_group_id, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass  # not a failure: see the SIGTERM rung above.
        try:
            return await asyncio.wait_for(process.communicate(), timeout=max(0.1, grace_s))
        except TimeoutError:
            logger.error("Async subprocess group could not be reaped pid=%s pgid=%s", process.pid, process_group_id)
            return None, None


def _run_bounded_by_its_work(
    command: list[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    budget_s: float,
    capture_output: bool,
    input: str | bytes | None,
    stdin: int | None,
    stdout: int | IO[Any] | None,
    stderr: int | IO[Any] | None,
    text: bool,
    check: bool,
) -> subprocess.CompletedProcess[Any]:
    """``subprocess.run`` whose budget is the child's work, not the wall clock.

    Every ``timeout`` handed to the gateway was measured on an idle host, and
    on a loaded one measured the host instead: the MLX probe (56.6s of wall
    on 16.6s of CPU), a boot's ``git symbolic-ref`` (past 3.0s), the snippet
    verdict's walk (past 10s), the integrity guardian's ``git status`` — all
    on 2026-09-16, load 34 on 18 cores. Seventy-nine call sites carry such a
    budget. The budget now bounds the child's CPU time, and how long its CPU
    may go without advancing; a starved child is neither over budget nor
    wedged. Everything else is ``subprocess.run``: the same arguments, the
    same ``CompletedProcess``, ``TimeoutExpired`` when the bound is hit (its
    message says which bound), ``CalledProcessError`` under ``check``. A
    child whose CPU cannot be read is bounded by the wall clock, as before.
    """
    from .subprocess_gateway import (
        WorkBoundExpired,
        _child_cpu_seconds,
        _drain_stopped_child,
    )

    budget = max(0.0, float(budget_s))
    period = max(0.05, min(1.0, budget / 10.0)) if budget else 1.0
    if capture_output:
        stdout, stderr = subprocess.PIPE, subprocess.PIPE
    if input is not None:
        stdin = subprocess.PIPE
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
        # Text is read as the async path reads it: a byte that is not UTF-8 is replaced, not fatal. LIVE 15 November
        # 2026, her find_in_source skill crashed twice in two minutes: "'utf-8' codec can't decode byte 0x92 in position
        # 81584585", a cp1252 apostrophe in a tracked text file her git grep matched.
        errors="replace" if text else None,
        shell=False,
    )
    started = time.monotonic()
    cpu_seen = 0.0
    advanced_at = started
    stopped_for = ""
    out = err = None
    pending_input = input
    # A child can exit between a watch period timing out and the poll after
    # it, with its output still in the pipe. That is a child that finished,
    # and one more period drains it; only pipes still open a period after the
    # exit belong to something the child left behind.
    exit_seen = False
    try:
        while True:
            try:
                out, err = proc.communicate(pending_input, timeout=period)
                break
            except subprocess.TimeoutExpired:
                pending_input = None  # not a failure: the timeout IS the watch period
            now = time.monotonic()
            cpu = _child_cpu_seconds(proc.pid)
            if proc.poll() is not None:
                if exit_seen:
                    stopped_for = "exited child left inherited output pipes open"
                exit_seen = True
            elif cpu is None:
                if proc.poll() is None and now - started >= budget:
                    stopped_for = (
                        f"unobservable child ran {now - started:.1f}s of wall against a "
                        f"{budget:.1f}s budget"
                    )
            else:
                if cpu > cpu_seen:
                    cpu_seen, advanced_at = cpu, now
                if cpu_seen >= budget:
                    stopped_for = (
                        f"cpu budget exhausted: {cpu_seen:.1f}s of CPU against "
                        f"{budget:.1f}s after {now - started:.1f}s of wall"
                    )
                elif now - advanced_at >= budget:
                    stopped_for = (
                        f"wedged: no CPU progress for {now - advanced_at:.1f}s at "
                        f"{cpu_seen:.1f}s of CPU, {now - started:.1f}s of wall"
                    )
            if stopped_for:
                if proc.poll() is None:
                    proc.kill()
                out, err = _drain_stopped_child(proc, timeout_s=period, text=text)
                raise WorkBoundExpired(
                    command, budget, output=out, stderr=err, reason=stopped_for
                ) from None
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    completed = subprocess.CompletedProcess(command, proc.returncode, out, err)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command, output=out, stderr=err)
    completed.cpu_seconds = cpu_seen or None  # type: ignore[attr-defined]
    return completed


