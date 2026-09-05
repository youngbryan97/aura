"""core/executive/executive_kernel.py
Unified Executive Kernel directing action selection and decision logs.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.executive.action_arbitrator import ActionArbitrator
from core.executive.attention_controller import AttentionController
from core.executive.conflict_resolver import ExecutiveConflictResolver
from core.executive.inhibition_system import ActionInhibitor
from core.executive.permission_router import PermissionRouter
from core.executive.decision_receipt import DecisionReceiptCompiler

logger = logging.getLogger("Executive.ExecutiveKernel")


def _default_file_goal_path() -> str:
    return str(Path(tempfile.gettempdir()) / "aura_life_tick_goal.txt")


class DeliberationEngine:
    """Executes reasoning loops and updates pending action list."""

    async def deliberate(self, state: Any) -> None:
        """Assembles logical plans and appends planned actions to state."""
        if "active_plans" not in state.world_model:
            state.world_model["active_plans"] = []

        # Plan repair logic: if last action failed, trigger retry or fallback
        last_verification = state.world_model.get("last_verification", {})
        last_failed = last_verification and not last_verification.get("success")

        for g in state.cognition.current_goals:
            status = g.get("status")
            if status in ["pending", "resumed", "unblocked"] or (
                last_failed and status == "in_progress"
            ):
                g["status"] = "in_progress"
                goal_id = g.get("id")
                goal_type = g.get("type", "normal")

                if last_failed and status == "in_progress":
                    logger.warning(
                        "Plan repair active: goal %s failed. Re-planning with fallback parameters.",
                        goal_id,
                    )
                    g["retry_count"] = g.get("retry_count", 0) + 1
                    # Switch command to fallback in case of terminal failure
                    if goal_type == "terminal":
                        g["command"] = g.get("fallback_command", "echo 'fallback_success'")

                if goal_type in ["terminal", "file"]:
                    # Create deliberation plan
                    plan = {
                        "plan_id": f"plan-{goal_id}-{state.tick_count}",
                        "goal_id": goal_id,
                        "deliberation_plan": f"Execute high-risk {goal_type} operation for {goal_id}",
                        "expected_observations": ["success_exit_code"],
                        "abort_criteria": ["error_output", "timeout"],
                        "verification_plan": "Check file checksum or process exit status",
                    }
                    state.world_model["active_plans"].append(plan)

                    params = {
                        "command": g.get("command", "echo 'success'"),
                        "action": g.get("action", "write"),
                        "goal_id": goal_id,
                        "plan_id": plan["plan_id"],
                    }
                    if goal_type == "file":
                        params["path"] = str(
                            g.get("path")
                            or state.world_model.get("default_file_goal_path")
                            or _default_file_goal_path()
                        )
                    if g.get("capability_token"):
                        params["capability_token"] = g["capability_token"]

                    # Append high-risk action. Missing capability tokens remain
                    # missing so the inhibition system can fail closed.
                    state.cognition.pending_actions.append(
                        {
                            "channel": goal_type,
                            "params": params,
                        }
                    )
                else:
                    state.cognition.pending_actions.append(
                        {
                            "channel": "gesture",
                            "params": {
                                "gesture": "goal_progress_signal",
                                "goal_id": goal_id,
                                "event_type": "executive_goal_started",
                            },
                        }
                    )


class ExecutiveKernel:
    """Canonical single-will executive controller for Aura."""

    def __init__(self):
        self.arbitrator = ActionArbitrator()
        self.attention = AttentionController()
        self.resolver = ExecutiveConflictResolver()
        self.inhibitor = ActionInhibitor()
        self.router = PermissionRouter()
        self.receipt_compiler = DecisionReceiptCompiler()

    async def evaluate_tick_decisions(self, state: Any) -> None:
        """Called during the life loop tick to resolve competing goals."""
        # Clean duplicate goal nodes
        state.cognition.current_goals = self.resolver.resolve_goal_clashes(
            state.cognition.current_goals
        )

        # Sort pending actions
        state.cognition.pending_actions = self.arbitrator.arbitrate(state.cognition.pending_actions)

        # Focus attention
        state.cognition.active_attention = await self.attention.focus_attention(state)

        logger.info("Executive Kernel tick completed. Focus: %s", state.cognition.active_attention)
