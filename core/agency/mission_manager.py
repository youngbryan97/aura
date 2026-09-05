"""core/agency/mission_manager.py
Unified Mission Manager coordinating scheduler, project checklists, and initiative selection.
"""
import logging
from typing import Any

from core.agency.autonomous_followup import AutonomousFollowupCoordinator
from core.agency.blocked_task_manager import BlockedTaskManager
from core.agency.commitment_tracker import CommitmentTracker
from core.agency.initiative_selector import InitiativeSelector
from core.agency.project_manager import ProjectManager
from core.agency.scheduler import Scheduler
from core.agency.task_resumption import TaskResumptionManager

logger = logging.getLogger("Agency.MissionManager")


class MissionManager:
    """Canonical long-horizon goal orchestrator for Aura."""

    def __init__(self):
        self.project_manager = ProjectManager()
        self.scheduler = Scheduler()
        self.commitment_tracker = CommitmentTracker()
        self.blocked_manager = BlockedTaskManager()
        self.initiative_selector = InitiativeSelector()
        self.followup_coordinator = AutonomousFollowupCoordinator()
        self.resumption_manager = TaskResumptionManager()
        
        self._boot_resumed = False

    async def update_goals_and_drives(self, state: Any) -> None:
        """Core goal-update lifecycle step executed on each loop tick."""
        # 1. On first tick, resume previous tasks if applicable
        if not self._boot_resumed:
            resumed_goals = self.resumption_manager.resume_tasks(state.autobiographical_memory)
            state.cognition.current_goals.extend(resumed_goals)
            self._boot_resumed = True

        # 2. Check scheduled tasks
        triggered_tasks = self.scheduler.check_and_trigger()
        if triggered_tasks:
            state.cognition.pending_actions.extend(triggered_tasks)

        # 3. Check blocked gates
        unblocked = self.blocked_manager.check_gates(state)
        for uid in unblocked:
            state.cognition.current_goals.append({"id": uid, "status": "unblocked"})

        # 4. Check initiatives based on welfare boredom
        boredom = state.welfare.boredom
        initiatives = self.initiative_selector.select_initiative(boredom, state.cognition.current_goals)
        if initiatives:
            state.cognition.pending_actions.extend(initiatives)

        # 5. Check followups
        followups = self.followup_coordinator.evaluate_followups(state.cognition.current_goals)
        if followups:
            state.cognition.pending_actions.extend(followups)

        # 6. Reconcile user commitments
        completed = [t.get("id") for t in state.cognition.current_goals if t.get("status") == "completed"]
        satisfied = self.commitment_tracker.reconcile_commitments(state.commitments, completed)
        for cid in satisfied:
            for c in state.commitments:
                if c["id"] == cid:
                    c["fulfilled"] = True

        logger.info("Goals and drives updated. Active goals: %d, Pending actions: %d",
                    len(state.cognition.current_goals), len(state.cognition.pending_actions))


# Singleton Access
_mission_manager: MissionManager | None = None


def get_mission_manager() -> MissionManager:
    global _mission_manager
    if _mission_manager is None:
        _mission_manager = MissionManager()
    return _mission_manager
