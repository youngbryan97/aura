"""
Plan Mode Skill — Ported from gemini-cli/enter-plan-mode.ts

Explicit separation of reconnaissance and execution. Enforces
read-only operations while drafting complex plans.
"""

import logging
from typing import Any, Dict

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.PlanMode")


class PlanModeSkill(BaseSkill):
    name = "plan_mode"
    description = "Enter or exit Planning Mode for complex tasks."

    def __init__(self):
        # We store mode on the class to be accessible globally per process
        self.__class__.is_active = False

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        action = goal.get("params", {}).get("action", goal.get("objective", "enter"))
        
        if "exit" in action.lower():
            self.__class__.is_active = False
            return {
                "ok": True,
                "message": "Exited Planning Mode. Execution capabilities restored.",
                "note": "Aura can now make modifications."
            }
        
        self.__class__.is_active = True
        return {
            "ok": True,
            "message": "Entered Planning Mode.",
            "note": "Aura is currently restricted to READ-ONLY operations. Draft your plan and present it to the user before exiting Plan Mode to execute."
        }
