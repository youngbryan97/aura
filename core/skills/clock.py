from datetime import datetime
from typing import Any, Dict

from core.skills.base_skill import BaseSkill


class ClockSkill(BaseSkill):
    name = "clock"
    description = "Get the current date and time."
    inputs = {}
    output = "Current date and time string"

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        time_keywords = ["time", "date", "clock", "what day", "today", "hour", "minute"]
        return any(kw in obj for kw in time_keywords)

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "ok": True,
            "time": now.isoformat(),
            "readable": now.strftime("%A, %B %d, %Y %I:%M %p"),
            "summary": f"It is currently {now.strftime('%A, %B %d, %Y %I:%M %p')}."
        }