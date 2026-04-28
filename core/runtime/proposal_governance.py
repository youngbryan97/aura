from __future__ import annotations
from core.runtime.errors import record_degradation


import inspect
import logging
from typing import Any, Dict, Optional, Tuple

from core.constitution import get_constitutional_core
from core.consciousness.executive_authority import get_executive_authority
from core.health.degraded_events import record_degraded_event
from core.runtime.service_access import resolve_state_repository

logger = logging.getLogger("Aura.ProposalGovernance")


def _normalize_goal(goal: Any) -> str:
    return " ".join(str(goal or "").strip().split())


async def propose_governed_initiative_to_state(
    state: Any,
    goal: Any,
    *,
    orchestrator: Any = None,
    source: str,
    kind: str = "autonomous_thought",
    urgency: float = 0.5,
    triggered_by: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    normalized_goal = _normalize_goal(goal)
    if state is None:
        return state, {"action": "rejected", "reason": "state_missing", "goal": normalized_goal}
    if len(normalized_goal) < 4:
        return state, {"action": "rejected", "reason": "empty_goal", "goal": normalized_goal}

    constitution = get_constitutional_core(orchestrator)
    approved, reason, authority_decision = await constitution.approve_initiative(
        normalized_goal,
        source=source,
        urgency=urgency,
        state=state,
    )
    if not approved:
        record_degraded_event(
            "proposal_governance",
            "initiative_blocked",
            detail=normalized_goal[:160],
            severity="info",  # Reduced from warning — blocked proposals are normal governance
            classification="background_degraded",
            context={"source": source, "reason": reason, "kind": kind},
        )
        return state, {
            "action": "blocked",
            "reason": reason,
            "goal": normalized_goal,
            "authority_decision": authority_decision,
        }

    enriched_metadata = dict(metadata or {})
    if authority_decision is not None:
        receipt_id = getattr(authority_decision, "substrate_receipt_id", None)
        if receipt_id:
            enriched_metadata.setdefault("substrate_receipt_id", receipt_id)
    enriched_metadata.setdefault("governed", True)

    authority = get_executive_authority(orchestrator)
    return await authority.propose_initiative_to_state(
        state,
        normalized_goal,
        source=source,
        kind=kind,
        urgency=urgency,
        triggered_by=triggered_by,
        metadata=enriched_metadata,
    )


async def queue_governed_initiative(
    goal: Any,
    *,
    orchestrator: Any = None,
    source: str,
    kind: str = "autonomous_thought",
    urgency: float = 0.5,
    triggered_by: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_goal = _normalize_goal(goal)
    repo = resolve_state_repository(orchestrator, default=None)
    if repo is None:
        record_degraded_event(
            "proposal_governance",
            "state_repository_missing",
            detail=normalized_goal[:160],
            severity="warning",
            classification="background_degraded",
            context={"source": source, "kind": kind},
        )
        return {"action": "rejected", "reason": "state_repository_missing", "goal": normalized_goal}

    state = None
    getter = getattr(repo, "get_current", None)
    if callable(getter):
        try:
            state = getter()
            if inspect.isawaitable(state):
                state = await state
        except Exception as exc:
            record_degradation('proposal_governance', exc)
            logger.debug("ProposalGovernance get_current failed: %s", exc)
            state = None
    if state is None:
        state = getattr(repo, "_current", None)

    new_state, decision = await propose_governed_initiative_to_state(
        state,
        normalized_goal,
        orchestrator=orchestrator,
        source=source,
        kind=kind,
        urgency=urgency,
        triggered_by=triggered_by,
        metadata=metadata,
    )
    if new_state is not None and new_state is not state and decision.get("action") == "queued":
        commit = getattr(repo, "commit", None)
        if callable(commit):
            try:
                result = commit(new_state, f"proposal_governance:{source}")
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                record_degradation('proposal_governance', exc)
                record_degraded_event(
                    "proposal_governance",
                    "state_commit_failed",
                    detail=normalized_goal[:160],
                    severity="warning",
                    classification="background_degraded",
                    context={"source": source, "error": type(exc).__name__},
                    exc=exc,
                )
                return {
                    "action": "rejected",
                    "reason": f"state_commit_failed:{type(exc).__name__}",
                    "goal": normalized_goal,
                }
        else:
            setattr(repo, "_current", new_state)
    return decision
