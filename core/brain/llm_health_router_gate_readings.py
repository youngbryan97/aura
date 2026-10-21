"""What the generation gate reports about its leases, read without changing them.

Lifted whole out of `llm_health_router`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import time
from typing import Any


def _gate_clock() -> float:
    """The clock leases are stamped and aged on: the machine's, never a run's.

    A lease is real concurrency, a generation holding a slot while the caller
    waits on a real semaphore with a real timeout. In a subject run the
    experiment clock replaces `time.time` and advances a thirtieth of a second
    a frame, so a background generation stuck for seventy-five seconds read as
    0.6 seconds old: the foreground ladder judged it young and in budget and
    refused her reply as saturated. On the report run of 21 October that lost 9
    of 32 replies. `time.monotonic` is the clock asyncio's own timeouts use,
    and nothing replaces it.
    """
    return time.monotonic()


def _oldest_generation_gate_lease_age_s() -> float:
    from .llm_health_router import (
        _oldest_generation_gate_lease,
    )

    oldest_lease = _oldest_generation_gate_lease()
    if oldest_lease is None:
        return 0.0
    _lease_id, acquired_at, _owner = oldest_lease
    return max(0.0, _gate_clock() - float(acquired_at))


def generation_gate_snapshot() -> dict[str, Any]:
    """Return a read-only snapshot for schedulers and health probes."""
    from .llm_health_router import (
        _GENERATION_GATE_ACTIVE_LEASES,
        _GENERATION_GATE_LAST_ACQUIRED_AT,
        _GENERATION_GATE_LAST_OWNER,
        _GENERATION_GATE_LEASE_DEADLINES,
        _GENERATION_GATE_STATE_LOCK,
        _GENERATION_GATE_WAIT_S,
    )


    with _GENERATION_GATE_STATE_LOCK:
        now = _gate_clock()
        active = {
            int(lease_id): {
                "age_s": max(0.0, now - float(acquired_at)),
                "owner": str(owner or "unknown"),
                "deadline_at": _GENERATION_GATE_LEASE_DEADLINES.get(int(lease_id)),
                "deadline_remaining_s": (
                    max(
                        0.0,
                        float(_GENERATION_GATE_LEASE_DEADLINES[int(lease_id)]) - now,
                    )
                    if int(lease_id) in _GENERATION_GATE_LEASE_DEADLINES
                    else None
                ),
            }
            for lease_id, (acquired_at, owner) in _GENERATION_GATE_ACTIVE_LEASES.items()
        }
        oldest = None
        if active:
            oldest_id = max(active, key=lambda lease_id: active[lease_id]["age_s"])
            oldest = {"lease_id": oldest_id, **active[oldest_id]}
        return {
            "active_count": len(active),
            "active": active,
            "oldest": oldest,
            "last_acquired_at": float(_GENERATION_GATE_LAST_ACQUIRED_AT or 0.0),
            "last_owner": str(_GENERATION_GATE_LAST_OWNER or ""),
            "wait_budget_s": float(_GENERATION_GATE_WAIT_S),
        }


