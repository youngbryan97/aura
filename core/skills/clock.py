from datetime import datetime
from typing import Any, Dict

from core.skills.base_skill import BaseSkill


class ClockSkill(BaseSkill):
    name = "clock"
    description = "Get the current date and time."
    effect_scope = "status"
    inputs = {}
    output = "Current date and time string"
    #: What a caller gets back, machine-readable. `output` above is prose: it
    #: tells a reader what to expect and tells a caller nothing it can check.
    result_schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "time": {"type": "string", "description": "ISO 8601 with an offset"},
            "readable": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["ok", "time", "readable", "summary"],
        "additionalProperties": False,
    }

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        time_keywords = ["what time", "current time", "the time", "what date", "current date", "what day", "clock", "hour", "minute"]
        return any(kw in obj for kw in time_keywords)

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().astimezone()
        readable = now.strftime("%A, %B %d, %Y %I:%M %p %Z").strip()
        return {
            "ok": True,
            "time": now.isoformat(),
            "readable": readable,
            "summary": f"It is currently {readable}."
        }
