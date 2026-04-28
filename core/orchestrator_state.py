from core.runtime.errors import record_degradation
import logging
from typing import Any

logger = logging.getLogger(__name__)

class OrchestratorStateMixin:
    """Mixin handling state persistence and restoration."""

    conversation_history: list[Any]
    state_manager: Any
    status: Any
    cognitive_engine: Any
    boredom: float
    stats: dict[str, Any]
    planner: Any

    async def _save_state_async(self, reason: str = "periodic") -> None:
        """Asynchronously save current system state."""
        try:
            state = self._compile_state_data()
            await self.state_manager.save_snapshot_async(state, reason)
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.error("Error in async state save: %s", e)

    def _compile_state_data(self) -> dict[str, Any]:
        """Gather state data for snapshotting."""
        return {
            "conversation_history": self.conversation_history[-20:], # Keep last 20 messages
            "boredom": getattr(self, "boredom", 0),
            "cycle_count": self.status.cycle_count,
            "thoughts_snapshot": [t.to_dict() for t in self.cognitive_engine.thoughts] if self.cognitive_engine else []
        }

    def _save_state(self, reason: str = "periodic") -> None:
        """Save current system state via StateManager (Synchronous)."""
        try:
            state = self._compile_state_data()
            self.state_manager.save_snapshot(state, reason)
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.error("Error saving state: %s", e)

    def _load_state(self) -> None:
        """Restore system state from StateManager."""
        try:
            # Try LATEST snapshot first
            data = self.state_manager.load_last_snapshot()
            
            if not data:
                logger.info("No system state snapshots found. Initializing fresh context.")
                return
                
            # Modular Restoration
            self._restore_core_metrics(data)
            # self._restore_history(data) 
            self._restore_cognition(data)
            self._restore_active_plans(data)
            
            logger.info("System state restored successfully (History skipped for fresh context)")
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.error("Error loading state: %s", e)

    def _restore_core_metrics(self, data: dict[str, Any]) -> None:
        """Restores core system metrics from snapshot."""
        try:
            metrics = data.get("metrics", {})
            self.status.cycle_count = data.get("cycle_count", self.status.cycle_count)
            self.boredom = data.get("boredom", self.boredom)
            if metrics:
                self.stats.update(metrics)
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.warning("Failed to restore core metrics: %s", e)

    def _restore_history(self, data: dict[str, Any]) -> None:
        """Restores conversation history from snapshot."""
        try:
            history = data.get("conversation_history", [])
            if isinstance(history, list) and history:
                self.conversation_history = history
                logger.info("Restored %d messages from history", len(history))
            elif not isinstance(history, list):
                logger.warning("Attempted to restore history but data was not a list. Ignoring.")
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.warning("Failed to restore history: %s", e)

    def _restore_cognition(self, data: dict[str, Any]) -> None:
        """Restores cognitive thought state from a previous snapshot."""
        try:
            thoughts = data.get("thoughts_snapshot", [])
            if not thoughts:
                return
            logger.info("Restoring %d thought snapshots from previous session", len(thoughts))
            if self.cognitive_engine and hasattr(self.cognitive_engine, "seed_thoughts"):
                self.cognitive_engine.seed_thoughts(thoughts)
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.warning("Failed to restore cognition: %s", e)

    def _restore_active_plans(self, data: dict[str, Any]) -> None:
        """Restores active goals and plans from snapshot (v13: implemented)."""
        try:
            # Restore goal hierarchy if present
            goals = data.get("active_goals", [])
            if goals and hasattr(self, 'planner') and self.planner:
                for goal_data in goals:
                    try:
                        self.planner.add_goal(
                            description=goal_data.get("description", ""),
                            priority=goal_data.get("priority", 0.5),
                            status=goal_data.get("status", "pending")
                        )
                        logger.info("Restored goal: %s", goal_data.get('description', '?')[:50])
                    except Exception as ge:
                        record_degradation('orchestrator_state', ge)
                        logger.warning("Skipped restoring goal: %s", ge)
            
            # Restore in-progress objectives
            objectives = data.get("objectives", [])
            if objectives and hasattr(self, 'status'):
                self.status.active_objectives = objectives
                logger.info("Restored %d active objectives", len(objectives))
            
            if goals:
                logger.info("Plan restoration complete: %d goals recovered", len(goals))
        except Exception as e:
            record_degradation('orchestrator_state', e)
            logger.warning("Failed to restore active plans: %s", e)
