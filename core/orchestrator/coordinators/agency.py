"""
Agency Coordinator for the RobustOrchestrator.
Handles skill dispatch, task execution, and agentic loop management.
"""
from core.runtime.errors import record_degradation
import logging
import asyncio
import sqlite3
from typing import Any, Dict, Optional

from core.health.degraded_events import record_degraded_event
from core.verification.decision_verifier import DecisionVerifier

logger = logging.getLogger(__name__)

# Capabilities that cannot work without a network path. When one of these
# fails and the connectivity probe is down, "offline" is the true cause and
# saying anything else sends the user debugging the wrong thing.
_NETWORK_DEPENDENT = frozenset(
    {
        "sovereign_browser",
        "sovereign_network",
        "web_search",
        "web_interlocutor",
        "research_pipeline",
        "content_fetcher",
        "social_lurker",
    }
)

_CAPABILITY_NOTE_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError, ValueError)


def _classify_capability_failure(skill_name: str, detail: str) -> str:
    """Name the cause in terms of what the user can do next, not exception type.

    She is explaining a situation, not reading a stack trace, and the thing
    that changes what happens next is whether this is a missing tool, a
    refused permission, or a network that is not there.
    """
    low = str(detail or "").lower()
    if skill_name in _NETWORK_DEPENDENT:
        try:
            from core.runtime.connectivity import get_connectivity_status

            if not get_connectivity_status().online:
                return "offline"
        except (OSError, *_CAPABILITY_NOTE_ERRORS):
            pass
    if any(word in low for word in ("denied", "not permitted", "unauthorized", "forbidden")):
        return "unauthorized"
    if any(word in low for word in ("not found", "no such", "missing", "not installed")):
        return "not_installed"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if any(word in low for word in ("refused by", "governance", "policy")):
        return "refused"
    return "failed"


def _note_capability_outcome(skill_name: str, result: Any) -> None:
    """Record a failed capability as facts for the reply to draw on."""
    if not isinstance(result, dict):
        return
    ok = result.get("ok")
    success = result.get("success")
    failed = ok is False or success is False or bool(result.get("error"))
    if not failed:
        return
    detail = str(result.get("error") or result.get("message") or "").strip()
    try:
        from core.conversation.failure_context import record_capability_failure

        record_capability_failure(
            skill_name,
            intent=f"use {skill_name.replace('_', ' ')}",
            cause=_classify_capability_failure(skill_name, detail),
            detail=detail[:400],
        )
    except _CAPABILITY_NOTE_ERRORS as exc:
        record_degradation(
            "agency.failure_context",
            exc,
            action="the reply will not be able to explain this capability failure",
            severity="debug",
        )


def _note_capability_exception(skill_name: str, exc: BaseException) -> None:
    try:
        from core.conversation.failure_context import record_capability_failure

        record_capability_failure(
            skill_name,
            intent=f"use {skill_name.replace('_', ' ')}",
            cause=_classify_capability_failure(skill_name, f"{type(exc).__name__}: {exc}"),
            detail=f"{type(exc).__name__}: {exc}"[:400],
        )
    except _CAPABILITY_NOTE_ERRORS as note_exc:
        record_degradation(
            "agency.failure_context",
            note_exc,
            action="the reply will not be able to explain this capability failure",
            severity="debug",
        )


class AgencyCoordinator:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._skill_manager = None
        self._capability_engine = None
        self._decision_verifier: Optional[DecisionVerifier] = None

    @property
    def skill_manager(self):
        if self._skill_manager is None:
            self._skill_manager = self.orchestrator._get_service("capability_engine")
        return self._skill_manager

    @property
    def capability_engine(self):
        if self._capability_engine is None:
            self._capability_engine = self.orchestrator._get_service("capability_engine")
        return self._capability_engine

    async def setup(self):
        """Initialize agency components."""
        logger.info("Initializing AgencyCoordinator...")
        self._decision_verifier = DecisionVerifier()
        if self.capability_engine is None:
            logger.warning("AgencyCoordinator setup without capability_engine")

    async def execute_skill(self, skill_name: str, params: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Executes a skill via the capability engine.

        Every outcome, good or bad, passes through here, which makes it the
        one place a failed capability can be turned into something she can
        talk about. Recording it as facts (see
        ``core/conversation/failure_context.py``) is what lets the reply say
        what actually broke instead of falling back to a line written into
        whichever skill it was — the difference between "I'm unable to browse
        the web right now" and "I can't get out to the network, that probe has
        been failing for a few minutes".
        """
        engine = self.capability_engine
        if not engine:
            logger.error("Capability engine not found for skill: %s", skill_name)
            return {"ok": False, "error": "No capability engine available"}
        
        try:
            # v22 Logic: Map context correctly
            ctx = dict(context or {})
            ctx.setdefault("proposal_source", "agency_coordinator")
            ctx.setdefault("requested_by", getattr(self.orchestrator, "_current_origin", "") or "unknown")
            ctx.setdefault("requested_via", "capability_engine")

            verifier = self._decision_verifier or DecisionVerifier()
            is_safe, confidence, reason = verifier.verify_plan(
                {"steps": [{"action": skill_name, "args": params or {}, "confidence": 1.0}]}
            )
            if not is_safe:
                record_degraded_event(
                    "agency_coordinator",
                    "skill_execution_denied",
                    detail=f"{skill_name}:{reason}",
                    severity="warning",
                    classification="foreground_blocking" if ctx.get("requested_by") in {"user", "voice", "admin", "api"} else "background_degraded",
                    context={"skill_name": skill_name, "confidence": confidence},
                )
                return {"ok": False, "error": reason, "confidence": confidence}

            if hasattr(engine, "execute_skill"):
                result = await engine.execute_skill(skill_name, params, ctx)
            else:
                result = await engine.execute(skill_name, params, ctx)
            _note_capability_outcome(skill_name, result)
            return result
        except (sqlite3.Error, OSError) as e:
            record_degradation('agency', e)
            _note_capability_exception(skill_name, e)
            logger.error("Skill execution failed for %s: %s", skill_name, e)
            record_degraded_event(
                "agency_coordinator",
                "skill_execution_failed",
                detail=f"{skill_name}:{type(e).__name__}: {e}",
                severity="error",
                classification="foreground_blocking" if (context or {}).get("origin") in {"user", "voice", "admin", "api"} else "background_degraded",
                context={"skill_name": skill_name},
                exc=e,
            )
            return {"ok": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the agency system."""
        return {
            "active_tasks": len(getattr(self.orchestrator, '_active_metabolic_tasks', set())),
            "engine_ready": self.capability_engine is not None
        }
