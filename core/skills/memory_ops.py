# skills/memory_ops.py
import logging
from typing import Any, Dict, Optional

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")

class MemoryOpsSkill(BaseSkill):
    name = "memory_ops"
    description = "Manage and update Aura's Knowledge Base (Verified Facts)."
    inputs = {
        "action": "The operation to perform (learn_fact, retrieve).",
        "key": "The fact key (e.g., 'user_name').",
        "value": "The fact value."
    }

    async def execute(self, goal: Dict, context: Dict) -> Dict:
        mem_sys = context.get("memory")
        if not mem_sys:
            return {"ok": False, "error": "Memory system not available in context."}

        params = goal.get("params", {})
        action = params.get("action")
        key = params.get("key")
        value = params.get("value")

        if action == "learn_fact":
            if not key or value is None:
                return {"ok": False, "error": "Missing key or value for learning fact."}
            
            # Call the learn_fact method on MemoryNexus
            if hasattr(mem_sys, 'learn_fact'):
                await mem_sys.learn_fact(key, value)
                return {
                    "ok": True, 
                    "summary": f"Learned fact: {key} = {value}",
                    "result": f"Knowledge base updated: {key} is now known."
                }
            else:
                return {"ok": False, "error": "Memory system does not support explicit fact learning."}
        
        elif action == "retrieve":
            if not key:
                return {"ok": False, "error": "Missing key for fact retrieval."}
            
            knowledge = getattr(mem_sys, 'knowledge', {})
            val = knowledge.get(key)
            if val is not None:
                return {"ok": True, "result": val}
            else:
                return {"ok": False, "error": f"No fact found for key: {key}"}

        return {"ok": False, "error": f"Unknown action: {action}"}