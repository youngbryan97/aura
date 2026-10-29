"""The answer a turn holds, and serving it in place of silence.

Lifted whole out of `response_generation_unitary`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.UnitaryResponse")


def _a_gate_took_something(outcome: Any) -> bool:
    """Whether this turn holds a candidate a gate suppressed and left servable."""
    if outcome is None:
        return False
    return any(
        candidate.suppressed is not None and candidate.is_recoverable
        for candidate in outcome.candidates()
    )


def _served_what_survived(
    phase: Any, new_state: Any, objective: Any, said_before: str, kwargs: dict, *, gave_up: bool = False
) -> bool:
    """Put the turn's best surviving draft in place of silence; True when it did."""
    from .response_generation_unitary import (
        _RESPONSE_RECOVERABLE_ERRORS,
        _record_response_degradation,
    )

    try:
        if not objective:
            return False
        cognition = new_state.cognition
        said_now = str(getattr(cognition, "last_response", "") or "").strip()
        if said_now and said_now != said_before.strip():
            return False
        origin = getattr(cognition, "current_origin", "")
        if not (kwargs.get("priority", False) or phase._is_user_facing_origin(origin)):
            return False
        # A benchmark turn fails closed on purpose: an empty reply is its
        # result, and handing it a draft would score text nobody served.
        if new_state.response_modifiers.get("benchmark_generation_failed_closed"):
            return False
        from core.runtime.turn_outcome import current_turn, recoverable_answer

        # A reply equal to her last one is ambiguous: a phase that committed
        # nothing leaves it in place, and one that answered the same way twice
        # commits it again. Only the first is a lost answer, so a stale reply is
        # replaced only when a gate threw something away. An empty reply needs
        # no such evidence.
        if said_now and not _a_gate_took_something(current_turn()):
            return False
        salvaged = recoverable_answer()
        if not salvaged:
            return False
        logger.warning(
            "🛡️ UnitaryResponse: the turn held an answer no gate would serve (len=%d%s). "
            "Serving it rather than nothing.",
            len(salvaged),
            ", after the phase gave up" if gave_up else "",
        )
        cognition.last_response = salvaged
        new_state.response_modifiers["served_a_suppressed_draft"] = True
        return True
    except _RESPONSE_RECOVERABLE_ERRORS as exc:
        _record_response_degradation(
            exc,
            "UnitaryResponse: salvage of a suppressed draft skipped: %s",
            action="left the turn empty because the turn ledger could not be read",
        )
        return False
