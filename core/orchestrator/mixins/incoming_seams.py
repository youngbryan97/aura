"""The seams of the incoming pipeline: helpers that take the turn, not self.

Three functions that were module level in ``incoming_logic`` and take
everything they touch as an argument — the degradation recorder, the check
that tells a parent cancellation from an awaited child's, and the social-turn
observation ``tools/extract_seam.py`` lifted out of the handler.

They are here because the module they were in reached 2,017 lines, seventeen
past the ceiling above which a new module is never grandfathered. Moving what
already had no reason to be in there is the cheapest hundred lines in the
tree, and the ceiling exists so that the cheap ones get taken before a file
is past saving.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service

logger = logging.getLogger(__name__)

__all__ = [
    "_current_task_cancellation_pending",
    "_observe_the_social_turn",
    "_record_incoming_degradation",
]


def _record_incoming_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "incoming_logic",
        error,
        severity=severity,
        action=action,
    )


def _current_task_cancellation_pending() -> bool:
    """Distinguish parent cancellation from an awaited child's cancellation."""
    task = asyncio.current_task()
    return bool(task is not None and task.cancelling())


def _observe_the_social_turn(
    *,
    _internal_update_allowed: Any,
    _live_state: Any,
    message: Any,
    payload_context: Any,
    self: Any,
) -> None:
    """Record the social turn if internal updates are allowed here.

    Moved out of ``IncomingLogicMixin._original_handle_incoming_logic`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 5 name(s) from the turn and hands back
    0.
    """
    if _internal_update_allowed:
        try:
            user_id = self._observe_social_turn(
                payload_context,
                message,
                _live_state,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            user_id = self._resolve_social_user_id(payload_context)
            payload_context["user_id"] = user_id
            _record_incoming_degradation(
                exc,
                action="continued user turn without calibrated other-agent state",
            )
            logger.error("OtherAgentState update failed: %s", exc, exc_info=True)

        try:
            self._apply_relational_memory_control(
                payload_context,
                message,
                user_id,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_incoming_degradation(
                exc,
                action="continued user turn after relational memory control failed",
            )

        try:
            discourse_tracker = optional_service("discourse_tracker")
            if discourse_tracker and _live_state is not None:
                self._fire_and_forget(
                    discourse_tracker.update(_live_state, message),
                    name="discourse_tracker_update",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError) as _dt_err:
            _record_incoming_degradation(
                _dt_err,
                action="continued user turn after discourse-tracker update failed",
            )
            logger.error("DiscourseTracker update failed: %s", _dt_err, exc_info=True)

        # Update Theory of Mind user model (rapport, trust, emotional state)
        try:
            tom = optional_service("theory_of_mind")
            if tom and user_id:
                self._fire_and_forget(
                    tom.understand_user(
                        user_id,
                        message,
                        {
                            "user_id": user_id,
                            "social_situation": payload_context.get(
                                "social_situation"
                            ),
                        },
                    ),
                    name="theory_of_mind_update",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError) as _tom_err:
            _record_incoming_degradation(
                _tom_err,
                action="continued user turn after theory-of-mind update failed",
            )
            logger.error("TheoryOfMind update failed: %s", _tom_err, exc_info=True)
