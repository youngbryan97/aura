"""
Agency Coordinator for the RobustOrchestrator.
Handles skill dispatch, task execution, and agentic loop management.
"""
import logging
import asyncio
from typing import Any, Dict, Optional

from core.health.degraded_events import record_degraded_event
from core.verifiers.decision_verifier import DecisionVerifier

logger = logging.getLogger(__name__)

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
        """Executes a skill via the capability engine."""
        engine = self.capability_engine
        if not engine:
            logger.error(f"Capability engine not found for skill: {skill_name}")
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
                return await engine.execute_skill(skill_name, params, ctx)
            return await engine.execute(skill_name, params, ctx)
        except Exception as e:
            logger.error(f"Skill execution failed for {skill_name}: {e}")
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
