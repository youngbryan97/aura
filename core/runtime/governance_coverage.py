"""Turns that reached cognition without a governance decision.

``core/orchestrator/mixins/message_handling.py`` carried this comment above
its Unified Will gate:

    ALL processing — user-facing or internal — must pass through the
    Unified Will. This is THE architectural invariant that makes Aura a
    unified intelligence rather than a federation.

It was not true, in three ways. The Somatic Reflex Bypass returns before the
gate for embodied-control contracts, which is deliberate and documented. The
gate is wrapped in ``if will._started:``, so every message before governance
comes up is ungated. And the gate's own ``except`` records a degradation and
continues.

The first is a designed exception. The other two were silent, and that is
the part that matters: a turn nobody governed was indistinguishable
afterwards from a turn the Will approved. This codebase has a name for that
shape — the absence of a check reported as a passed check — and here it sat
on the governance boundary itself.

Counting, not refusing. Refusing would take the conversation offline every
time governance hiccups, and a control that breaks working capability is one
somebody routes around. What must not happen is that it goes unrecorded, so
the count reaches ``runtime_health_report()["integrity"]["ungoverned_turns"]``
and a green verdict can carry the caveat.

This lives in ``core/runtime`` rather than beside the gate because the health
surface reads it, and the foundation may not import upward — the same reason
the admission-throughput estimator registers itself instead of being imported.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.GovernanceCoverage")

#: checked_lock, not threading.Lock — lockdep only sees the locks it wraps,
#: and `make lock-coverage` caught this module adding the 716th raw one on
#: the same day the ratchet was written to stop exactly that.
#:
#: REGISTRY rank: a process-wide counter, taken first and held for a dict
#: update. Nothing is acquired underneath it.
_LOCK = checked_lock("governance_coverage.ungoverned", rank=LockRank.REGISTRY)
_UNGOVERNED: dict[str, int] = {}

#: Log the first occurrence, then every Nth. A persistent condition should
#: stay visible without emitting a line per message.
_LOG_EVERY = 50


def note_ungoverned_turn(origin: str, reason: str) -> None:
    """Record that a message was processed without a Will decision."""
    key = f"{reason}:{origin}"
    with _LOCK:
        _UNGOVERNED[key] = _UNGOVERNED.get(key, 0) + 1
        count = _UNGOVERNED[key]
    if count == 1 or count % _LOG_EVERY == 0:
        logger.warning(
            "Ungoverned turn (%s) from %s — the Unified Will did not decide "
            "this message. Occurrences: %d",
            reason,
            origin,
            count,
        )


def ungoverned_turn_report() -> dict[str, Any]:
    """Turns served without a Will decision, split by why.

    "Governance had not started yet" and "the gate is erroring" present the
    same symptom and call for different responses, so they are counted apart.
    """
    with _LOCK:
        by_reason = dict(_UNGOVERNED)
    return {
        "total": sum(by_reason.values()),
        "by_reason": by_reason,
        "clean": not by_reason,
    }


def reset_governance_coverage_for_test() -> None:
    with _LOCK:
        _UNGOVERNED.clear()
