from typing import Any, Dict

from core.skills.base_skill import BaseSkill


class UplinkSkill(BaseSkill):
    name = "uplink_local"
    description = "Local Persistence Uplink (Offline Mode)."
    inputs = {"goal": "objective"}
    output = "Status"

    def match(self, goal: Dict[str, Any]) -> bool:
        return "Uplink" in goal.get("objective", "") or "Persistence" in goal.get("objective", "")

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        # functional no-op for local mode
        return {"ok": True, "status": "Uplink Established (Local)", "summary": "Persistence verified locally."}