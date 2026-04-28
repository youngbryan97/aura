from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
from typing import Any, Dict, Optional

from core.constitution import get_constitutional_core
from core.health.degraded_events import record_degraded_event

logger = logging.getLogger("Aura.ImpulseGovernance")


async def run_governed_impulse(
    orchestrator: Any,
    *,
    source: str,
    summary: str,
    message: Any,
    urgency: float = 0.3,
    state_cause: Optional[str] = None,
    state_update: Optional[Dict[str, Any]] = None,
    enqueue_priority: int = 20,
) -> bool:
    """Apply one constitutional path for autonomous impulse release and affect shifts."""
    if orchestrator is None:
        return False

    constitution = get_constitutional_core(orchestrator)
    current_state = getattr(getattr(orchestrator, "state_repo", None), "_current", None)

    approved, reason, _authority_decision = await constitution.approve_initiative(
        summary,
        source=source,
        urgency=urgency,
        state=current_state,
    )
    if not approved:
        record_degraded_event(
            "impulse_governance",
            "initiative_blocked",
            detail=str(summary)[:160],
            severity="info",  # Reduced from warning — blocked initiatives are normal during idle
            classification="background_degraded",
            context={"source": source, "reason": reason},
        )
        return False

    if state_update:
        state_ok, state_reason = await constitution.approve_state_mutation(
            source,
            state_cause or summary,
            state=current_state,
        )
        if not state_ok:
            record_degraded_event(
                "impulse_governance",
                "state_shift_blocked",
                detail=str(summary)[:160],
                severity="warning",
                classification="background_degraded",
                context={"source": source, "reason": state_reason},
            )
            return False

        liquid_state = getattr(orchestrator, "liquid_state", None)
        if liquid_state is not None:
            try:
                await liquid_state.update(_caller=source, **dict(state_update))
            except Exception as exc:
                record_degradation('impulse_governance', exc)
                logger.debug("Governed impulse state update failed: %s", exc)
                record_degraded_event(
                    "impulse_governance",
                    "state_shift_failed",
                    detail=str(summary)[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"source": source, "error": type(exc).__name__},
                    exc=exc,
                )
                return False

    queue = getattr(orchestrator, "message_queue", None)
    if queue is not None and hasattr(queue, "full") and queue.full():
        record_degraded_event(
            "impulse_governance",
            "message_queue_full",
            detail=str(summary)[:160],
            severity="warning",
            classification="background_degraded",
            context={"source": source},
        )
        return False

    try:
        orchestrator.enqueue_message(
            message,
            priority=enqueue_priority,
            origin=source,
            _authority_checked=True,
        )
    except TypeError:
        try:
            orchestrator.enqueue_message(message, priority=enqueue_priority, origin=source)
        except TypeError:
            orchestrator.enqueue_message(message, priority=enqueue_priority)
    return True
