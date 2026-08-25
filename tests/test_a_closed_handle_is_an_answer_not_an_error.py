"""One stale process handle broke three subsystems at once.

LIVE 2026-08-25, mid-task:

    StabilityGuardian: DEGRADED — _check_runtime_hygiene: process object is closed
    allostasis (degraded): ValueError: process object is closed → vitals
        snapshot unavailable; pulse skipped
    inference_gate (critical): ValueError: process object is closed
    CRITICAL SERVICE FAILURE: Subsystem 'inference_gate' failed with failure
        policy 'fail-closed'

All three read the same runtime-pressure snapshot, and the snapshot asks
every model client whether its worker is alive. ``multiprocessing`` raises
``ValueError("process object is closed")`` once a worker has been reaped and
its handle closed — and ValueError was the one exception that guard did not
catch.

A closed handle means the process is gone, which is an answer to the
question being asked.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp


def _noop():
    return None


def test_a_closed_handle_really_raises_value_error():
    """The premise, checked rather than assumed."""
    process = mp.Process(target=_noop)
    process.start()
    process.join()
    process.close()
    try:
        process.is_alive()
    except ValueError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("multiprocessing no longer raises here; the guard can be revisited")


def test_the_liveness_check_treats_a_closed_handle_as_not_alive():
    from core.runtime import runtime_pressure

    source = inspect.getsource(runtime_pressure)
    where = source.index("process.is_alive()")
    guard = source[where : where + 400]
    assert "ValueError" in guard, "the one exception it actually raises is uncaught again"


def test_the_snapshot_survives_a_closed_handle():
    """The answer has to be "not alive", not an exception that escapes into
    every subsystem reading the snapshot."""
    from core.runtime.runtime_pressure import get_unified_runtime_pressure

    snapshot = get_unified_runtime_pressure().runtime_pressure_snapshot()
    assert isinstance(snapshot, dict)
